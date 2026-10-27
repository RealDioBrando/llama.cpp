#include "server-scheduler-paged.h"

#include "common.h"
#include "log.h"
#include "mtmd.h"
#include "sampling.h"

#include <algorithm>

static bool paged_task_supported(const server_task & task) {
    // chat templates are applied in the HTTP layer and arrive here as
    // COMPLETION tasks; grammar lives in the per-slot sampler chain and works
    // unchanged on the paged path; n_cmpl > 1 is handled by the parent/child
    // launch flow
    if (task.type != SERVER_TASK_TYPE_COMPLETION) {
        return false;
    }
    switch (task.params.res_type) {
        case TASK_RESPONSE_TYPE_NONE:        // /completion
        case TASK_RESPONSE_TYPE_OAI_CMPL:    // /v1/completions
        case TASK_RESPONSE_TYPE_OAI_CHAT:    // /v1/chat/completions
            break;
        default:
            return false;
    }
    return true;
}

server_scheduler_paged::~server_scheduler_paged() {
    if (sched) {
        llama_paged_scheduler_free(sched);
    }
}

bool server_scheduler_paged::init() {
    GGML_ASSERT(sched == nullptr);

    llama_context * ctx = host.ctx();

    if (llama_n_batch(ctx) != llama_n_ubatch(ctx)) {
        SRV_ERR("%s", "--kv-paged requires -b == -ub (batch and ubatch sizes must match)\n");
        return false;
    }

    sched = llama_paged_scheduler_init(ctx);
    if (sched == nullptr) {
        SRV_ERR("%s", "failed to initialize the paged scheduler (is the model on a single device with full offload?)\n");
        return false;
    }

    const int32_t max_request_id = llama_paged_scheduler_max_request_id(sched);
    if (max_request_id < (int32_t) host.slot_pool().size()) {
        SRV_ERR("paged context allows %d request id(s) but %zu slot(s) are configured, reduce -np\n",
                max_request_id, host.slot_pool().size());
        llama_paged_scheduler_free(sched);
        sched = nullptr;
        return false;
    }

    // the scheduler enforces a per-sequence context limit of llama_n_ctx,
    // there is no per-slot split like the unified KV
    for (auto & slot : host.slot_pool()) {
        slot.n_ctx = llama_n_ctx(ctx);
    }

    cancelled.assign(host.slot_pool().size(), 0);

    llama_paged_scheduler_set_on_finish(sched, on_finish_thunk, this);

    if (!llama_paged_scheduler_warmup(sched, ctx)) {
        SRV_WRN("%s", "paged scheduler warmup failed, first requests may be slow\n");
    }

    SRV_INF("%s", "paged scheduler initialized\n");
    return true;
}

server_slot * server_scheduler_paged::pick_idle_slot() {
    server_slot * best = nullptr;
    for (auto & slot : host.slot_pool()) {
        if (slot.is_processing()) {
            continue;
        }
        // slot affinity is worthless here: the scheduler's block-level
        // prefix cache reuses KV across any slot
        if (best == nullptr || slot.t_last_used < best->t_last_used) {
            best = &slot;
        }
    }
    return best;
}

bool server_scheduler_paged::launch(server_task && task) {
    if (sched == nullptr) {
        host.send_error(task, "paged scheduler is not available", ERROR_TYPE_SERVER);
        return true;
    }

    if (task.id_slot != -1) {
        host.send_error(task, "id_slot is not supported with --kv-paged", ERROR_TYPE_INVALID_REQUEST);
        return true;
    }

    if (task.tokens.has_mtmd) {
        // multimodal (12.4): launch as ordered text/media segments; media
        // segments are encoded on demand when the scheduler pauses on them
        return launch_multimodal(std::move(task));
    }

    if (!paged_task_supported(task)) {
        host.send_error(task, "this feature is not supported with --kv-paged", ERROR_TYPE_NOT_SUPPORTED);
        return true;
    }

    // launch one request and its prompt into the scheduler; returns false
    // when the caller must defer the task
    auto launch_single = [&](server_task && t, server_slot * forced_slot) -> bool {
        server_slot * slot = forced_slot ? forced_slot : pick_idle_slot();
        if (slot == nullptr) {
            return false;
        }

        if (!host.launch_slot_with_task(*slot, std::move(t))) {
            // rejected, error already sent
            return true;
        }

        server_slot & s = *slot;
        s.stats.update_prompt_start();
        // server_tokens is move-only; paged inputs are text-only (mtmd gated off)
        s.prompt.tokens = server_tokens(s.task->tokens.get_text_tokens(), /* has_mtmd = */ false);

        llama_tokens prompt_tokens = s.task->tokens.get_text_tokens();

        if (prompt_tokens.empty()) {
            // mirror the default path: an empty prompt completes immediately
            SLT_WRN(s, "%s", "empty prompt - releasing slot\n");
            s.print_timings();
            host.send_final_response(s);
            s.release();
            return true;
        }

        if ((int32_t) prompt_tokens.size() >= s.n_ctx) {
            host.send_error(s, "prompt is too long for the paged context", ERROR_TYPE_EXCEED_CONTEXT_SIZE);
            s.release();
            return true;
        }

        if (!llama_paged_scheduler_add_request(sched, prompt_tokens.data(), (int32_t) prompt_tokens.size(), s.id)) {
            host.send_error(s, "failed to queue the request in the paged scheduler", ERROR_TYPE_SERVER);
            s.release();
            return true;
        }
        return true;
    };

    if (!task.is_parent()) {
        return launch_single(std::move(task), nullptr);
    }

    // n_cmpl > 1: the parent plus every child needs a slot atomically,
    // mirroring launch_slots_with_parent_task on the default path
    const int64_t id_parent = task.id;
    std::vector<server_slot *> free_slots;
    for (auto & s : host.slot_pool()) {
        if (!s.is_processing()) {
            free_slots.push_back(&s);
        }
    }
    std::sort(free_slots.begin(), free_slots.end(),
              [](const server_slot * a, const server_slot * b) { return a->t_last_used < b->t_last_used; });

    const size_t n_children = task.child_tasks.size();
    if (free_slots.size() < 1 + n_children) {
        return false; // no capacity: the caller defers the whole task
    }

    auto release_task_slots = [&](int64_t id) {
        for (auto & s : host.slot_pool()) {
            if (s.task && (s.task->id == id || s.task->id_parent == id)) {
                s.release();
            }
        }
    };

    // children first: they wait in SLOT_STATE_WAIT_OTHER without a scheduler
    // request until the parent's prompt is done
    for (size_t i = 0; i < n_children; ++i) {
        if (!host.launch_slot_with_task(*free_slots[i + 1], std::move(task.child_tasks[i]))) {
            release_task_slots(id_parent);
            return true; // rejected, error already sent
        }
    }

    if (!launch_single(std::move(task), free_slots[0])) {
        release_task_slots(id_parent);
        return false;
    }
    return true;
}

void server_scheduler_paged::cancel(int64_t id_task) {
    for (auto & slot : host.slot_pool()) {
        if (!slot.task || slot.task->id != id_task) {
            continue;
        }
        if (sched != nullptr) {
            cancelled[slot.id] = 1;
            if (!llama_paged_scheduler_cancel(sched, slot.id)) {
                // unknown to the scheduler: already reaped or never queued
                cancelled[slot.id] = 0;
                slot.release();
            }
        } else {
            slot.release();
        }
        break;
    }
}

void server_scheduler_paged::fail_all(const std::string & why) {
    for (auto & slot : host.slot_pool()) {
        if (!slot.is_processing()) {
            continue;
        }
        host.send_error(slot, why, ERROR_TYPE_SERVER);
        cancelled[slot.id] = 1;
        if (!llama_paged_scheduler_cancel(sched, slot.id)) {
            cancelled[slot.id] = 0;
            slot.release();
        }
    }
}

void server_scheduler_paged::on_finish_thunk(int32_t request_id, const llama_token * /*tokens*/,
                                             int32_t /*n_tokens*/, void * user_data) {
    static_cast<server_scheduler_paged *>(user_data)->on_finish(request_id);
}

void server_scheduler_paged::on_finish(int32_t request_id) {
    auto & slots = host.slot_pool();
    if (request_id < 0 || request_id >= (int32_t) slots.size()) {
        return;
    }

    server_slot & slot = slots[request_id];

    if (!slot.is_processing()) {
        return; // already released by a failure path
    }

    if (cancelled[request_id]) {
        cancelled[request_id] = 0;
    } else if (slot.stop == STOP_TYPE_NONE) {
        // the scheduler ended the sequence on its own (per-seq context full)
        slot.truncated      = true;
        slot.stop           = STOP_TYPE_LIMIT;
        slot.has_next_token = false;
        slot.print_timings();
        host.send_final_response(slot);
    }

    // release -> pop_deferred_task wakes any queued request for this slot;
    // the scheduler id is free again for the next add_request
    slot.release();
}

void server_scheduler_paged::update() {
    auto & slots = host.slot_pool();

    bool any_processing = false;
    for (const auto & slot : slots) {
        if (slot.is_processing()) {
            any_processing = true;
            break;
        }
    }

    if (any_processing) {
        host.post_next_response();
    } else {
        n_empty_steps = 0;
        host.metrics_flush_idle();
        return;
    }

    // multimodal (12.4): paused requests are waiting for a media segment's
    // encoder output; encode it on demand and append to resume the request
    for (auto & slot : slots) {
        if (!slot.is_processing() || slot.mctx == nullptr) {
            continue;
        }
        llama_paged_seq_state st;
        if (!llama_paged_scheduler_get_seq_state(sched, slot.id, &st) || !st.paused) {
            SLT_DBG(slot, "mm: slot not paused (paused=%d)\n", st.paused);
            continue;
        }
        SLT_WRN(slot, "mm: encoding pending media chunk at row %d\n", st.n_past);

        // the pending media chunk: the (n_media_appended+1)-th media chunk of
        // the task, in order. NOTE: the boundary must NOT be derived from
        // prompt.tokens.size() - server_tokens rows and scheduler rows use
        // different scales (one sentinel row per media chunk vs
        // n_tokens rows per chunk) and diverge after the first media segment
        size_t search_from = 0;
        const mtmd_input_chunk * target = nullptr;
        size_t target_idx = 0;
        for (int k = 0; k <= slot.n_media_appended; ++k) {
            auto [c, i] = slot.task->tokens.find_next_media_chunk(search_from);
            if (c == nullptr) {
                break;
            }
            target        = c->get();
            target_idx    = i;
            search_from   = i + mtmd_input_chunk_get_n_tokens(c->get());
        }
        if (target == nullptr) {
            SLT_ERR(slot, "%s", "paused but no pending media chunk found\n");
            host.send_error(slot, "failed to encode media chunk", ERROR_TYPE_SERVER);
            slot.release();
            continue;
        }
        const auto & chunk = target;
        const int32_t n_tok = (int32_t) mtmd_input_chunk_get_n_tokens(target);

        // encode the chunk (encode half of process_mtmd_chunk); done inline on
        // the update thread - no yield, the media decode must stay on the
        // same thread as the scheduler step
        float * embd = nullptr;
        try {
            slot.mbatch.reset(mtmd_batch_init(slot.mctx));
            if (mtmd_batch_add_chunk(slot.mbatch.get(), chunk) != 0) {
                SLT_ERR(slot, "failed to add media chunk to the encoder batch, idx = %zu\n", target_idx);
                host.send_error(slot, "failed to encode media chunk", ERROR_TYPE_SERVER);
                slot.release();
                continue;
            }
            if (mtmd_batch_encode(slot.mbatch.get()) != 0) {
                SLT_ERR(slot, "failed to encode media chunk, idx = %zu\n", target_idx);
                host.send_error(slot, "failed to encode media chunk", ERROR_TYPE_SERVER);
                slot.release();
                continue;
            }
            embd = mtmd_batch_get_output_embd(slot.mbatch.get(), chunk);
        } catch (const std::exception & e) {
            SLT_ERR(slot, "mm encode exception: %s\n", e.what());
            host.send_error(slot, "failed to encode media chunk", ERROR_TYPE_SERVER);
            slot.release();
            continue;
        }
        if (embd == nullptr) {
            SLT_ERR(slot, "failed to encode media chunk, idx = %zu\n", target_idx);
            host.send_error(slot, "failed to encode media chunk", ERROR_TYPE_SERVER);
            slot.release();
            continue;
        }

        // mrope positions from the image tokens at the paused boundary
        std::vector<llama_pos> pos;
        const int32_t n_ppe = llama_model_n_pos_per_embd(llama_get_model(host.ctx()));
        if (n_ppe > 1) {
            const auto * image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
            pos.reserve((size_t) n_tok * n_ppe);
            for (int32_t r = 0; r < n_tok; ++r) {
                const mtmd_decoder_pos p = mtmd_image_tokens_get_decoder_pos(image_tokens, (llama_pos) st.n_past, r);
                pos.push_back(p.t);
                pos.push_back(p.y);
                pos.push_back(p.x);
                pos.push_back(p.z);
            }
        }

        llama_paged_segment seg = {};
        seg.tokens      = nullptr;
        seg.embd        = embd;
        seg.n_tokens    = n_tok;
        seg.embd_stride = llama_model_n_embd_inp(llama_get_model(host.ctx()));
        seg.pos         = pos.empty() ? nullptr : pos.data();
        bool app_ok = llama_paged_scheduler_append_segment(sched, slot.id, &seg);
        SLT_WRN(slot, "mm: append done ok=%d n_tok=%d stride=%d pos_n=%zu\n", (int) app_ok, n_tok, seg.embd_stride, pos.size());
        if (!app_ok) {
            host.send_error(slot, "failed to append the media segment", ERROR_TYPE_SERVER);
            slot.release();
            continue;
        }

        // accounting: the chunk's rows are prompt rows (placeholders)
        slot.prompt.tokens.push_back_placeholder(target);
        slot.n_media_appended++;
        slot.stats.n_prompt_processed += (uint64_t) n_tok;
        slot.stats.update_prompt_last();
    }

    // speculative decoding: draft for the generating slots and attach the
    // drafts to the scheduler before the batch is built
    if (host.spec_ctx() != nullptr) {
        prepare_drafts();
    }

    // reap finished requests, schedule waiting/swapped ones and build the batch
    if (!llama_paged_scheduler_prepare_batch(sched, &paged_batch)) {
        fail_all("paged scheduler deadlock: not enough KV blocks to make progress");
        return;
    }

    if (paged_batch.n_tokens == 0) {
        // in-flight requests but nothing schedulable this step
        if (++n_empty_steps >= 64) {
            fail_all("paged scheduler made no progress for 64 steps");
        }
        return;
    }
    n_empty_steps = 0;

    host.metrics_pre_decode();

    if (llama_decode(host.ctx(), paged_batch) != 0) {
        llama_synchronize(host.ctx());
        fail_all("llama_decode failed on the paged path");
        return;
    }
    llama_synchronize(host.ctx());

    host.metrics_flush_prompt();

    const llama_paged_batch_info * info = llama_paged_scheduler_get_batch_info(sched);
    GGML_ASSERT(info != nullptr);

    server_metrics & m = host.srv_metrics();
    m.n_decode++;

    const int64_t t_now = ggml_time_us();

    // feedback storage; pointers into fb_tok must stay stable until the
    // llama_paged_scheduler_update call below
    std::vector<llama_token>            fb_tok;
    std::vector<llama_paged_feedback>   fb;
    fb_tok.reserve(2 * (size_t) info->n_seq + 32);
    fb.reserve(2 * (size_t) info->n_seq + 32);

    uint64_t n_prompt_tokens = 0;

    for (int i = 0; i < info->n_seq; ++i) {
        const int32_t off    = info->batch_offsets[i];
        const int32_t len    = info->batch_lens[i];
        const int32_t req_id = paged_batch.seq_id[off][0];
        server_slot & slot   = slots[req_id];

        llama_paged_seq_state st;
        GGML_ASSERT(llama_paged_scheduler_get_seq_state(sched, req_id, &st));

        const bool final_prefill = len > 1 && (int32_t) st.n_past + len == (int32_t) st.n_prompt_total;
        const bool sample_now    = len == 1 || final_prefill;

        m.n_tokens_max = std::max<uint64_t>(m.n_tokens_max, st.n_past + len);

        if (st.n_decoded == 0) {
            // prefill step for this sequence
            slot.state = final_prefill ? SLOT_STATE_DONE_PROMPT : SLOT_STATE_PROCESSING_PROMPT;
            slot.stats.n_prompt_processed += len;
            slot.stats.set_prompt_last(t_now);
            n_prompt_tokens += len;

            if (final_prefill) {
                // tokens skipped thanks to the scheduler's prefix cache
                slot.stats.n_prompt_cached = st.n_prompt_total - st.n_prompt;
                if (slot.task->params.stream && slot.task->params.return_progress) {
                    host.send_partial_response(slot, {}, true);
                }
            }
        } else if (slot.state != SLOT_STATE_GENERATING) {
            slot.state = SLOT_STATE_GENERATING;
        }

        // speculative verify: the batch holds 1 + n_draft rows for this seq,
        // all with logits; sample_and_accept_n walks them. checked before the
        // sample_now skip, because a verify batch has len = 1 + n_draft > 1
        // and would be mistaken for a mid-prefill chunk otherwise
        if (slot.can_speculate() && !slot.spec_draft.empty()) {
            verify_slot(slot, off, len, fb_tok, fb);
            continue;
        }

        if (!sample_now) {
            // mid-prefill chunk: no logits, no feedback
            continue;
        }

        // the last row of the sequence in this batch carries the logits
        slot.i_batch = off + len - 1;
        llama_token tok = common_sampler_sample(slot.smpl.get(), host.ctx(), slot.i_batch);
        common_sampler_accept(slot.smpl.get(), tok, true);
        slot.i_batch = -1;

        slot.stats.n_gen += 1;
        if (slot.stats.n_gen == 1) {
            slot.stats.update_prompt_last();
            slot.t_print_last = t_now;
            slot.n_gen_last = 0;
        }
        slot.stats.update_gen_last();

        // the sampled token becomes part of the prompt state before the
        // stop checks, mirroring handle_last_sampled_token()
        slot.prompt.tokens.push_back(tok);

        completion_token_output result;
        result.tok          = tok;
        result.text_to_send = common_token_to_piece(host.ctx(), tok,
                host.params().special ||
                slot.task->params.sampling.preserved_tokens.count(tok) > 0);
        result.prob         = 1.0f;

        if (slot.task->params.sampling.n_probs > 0) {
            host.populate_token_probs(slot, result, slot.task->params.post_sampling_probs,
                                      host.params().special, off + len - 1);
        }

        bool keep_generating = host.process_token(result, slot);
        if (!keep_generating) {
            slot.print_timings();
            host.send_final_response(slot);
        } else {
            slot.print_timings_tg();
        }

        if (slot.state == SLOT_STATE_DONE_PROMPT) {
            slot.state = SLOT_STATE_GENERATING;
            if (slot.can_speculate() && host.spec_ctx() != nullptr) {
                common_speculative_begin(host.spec_ctx(), slot.id, slot.prompt.tokens.get_text_tokens());
            }
        }

        fb_tok.push_back(tok);
        fb.push_back({ slot.id, &fb_tok.back(), 1, keep_generating ? 0 : 1 });

        // n_cmpl > 1: fork the prefilled parent into its waiting children and
        // sample each child's first token from this row's logits
        if (final_prefill && keep_generating && slot.task->is_parent()) {
            fork_children(slot, off + len - 1, fb_tok, fb);
        }
    }

    for (const auto & slot : slots) {
        if (slot.is_processing()) {
            m.n_busy_slots++;
        }
    }

    host.metrics_queue_prompt(n_prompt_tokens);

    llama_paged_scheduler_update(sched, &paged_batch, fb.data(), (int32_t) fb.size());

    // draft rollbacks run after update(), which is what advanced the
    // scheduler n_past over the verify batch rows
    for (const auto & t : spec_pending_truncates) {
        llama_paged_scheduler_truncate(sched, t.first, t.second);
    }
    spec_pending_truncates.clear();
}

void server_scheduler_paged::fork_children(server_slot & parent, int32_t tok_idx,
                                            std::vector<llama_token> & fb_tok,
                                            std::vector<llama_paged_feedback> & fb) {
    for (auto & child : host.slot_pool()) {
        if (child.state != SLOT_STATE_WAIT_OTHER || !child.task || !parent.task) {
            continue;
        }
        if (child.task->id_parent != parent.task->id) {
            continue;
        }

        if (!llama_paged_scheduler_fork_request(sched, parent.id, child.id)) {
            host.send_error(child, "failed to fork the request in the paged scheduler", ERROR_TYPE_SERVER);
            child.release();
            continue;
        }

        // clone the prompt state from the parent (KV is shared by reference)
        child.prompt.tokens = server_tokens(parent.prompt.tokens.get_text_tokens(), /* has_mtmd = */ false);
        child.stats         = parent.stats;
        child.state         = SLOT_STATE_GENERATING;

        // the child samples its own first token from the parent's logits row
        child.i_batch = tok_idx;
        llama_token tok = common_sampler_sample(child.smpl.get(), host.ctx(), tok_idx);
        common_sampler_accept(child.smpl.get(), tok, true);
        child.i_batch = -1;

        child.stats.n_gen += 1;
        child.stats.update_gen_last();

        completion_token_output result;
        result.tok          = tok;
        result.text_to_send = common_token_to_piece(host.ctx(), tok,
                host.params().special ||
                child.task->params.sampling.preserved_tokens.count(tok) > 0);
        result.prob         = 1.0f;

        if (child.task->params.sampling.n_probs > 0) {
            host.populate_token_probs(child, result, child.task->params.post_sampling_probs,
                                      host.params().special, tok_idx);
        }

        fb_tok.push_back(tok);
        bool keep_generating = host.process_token(result, child);
        if (!keep_generating) {
            child.print_timings();
            host.send_final_response(child);
        }
        fb.push_back({ child.id, &fb_tok.back(), 1, keep_generating ? 0 : 1 });
    }
}

bool server_scheduler_paged::launch_multimodal(server_task && task) {
    server_slot * slot = pick_idle_slot();
    if (slot == nullptr) {
        return false;  // defer until a slot frees up
    }

    if (!host.launch_slot_with_task(*slot, std::move(task))) {
        // rejected, error already sent
        return true;
    }

    server_slot & s = *slot;
    s.stats.update_prompt_start();
    s.mctx = host.mmctx();
    if (s.mctx == nullptr) {
        host.send_error(s, "multimodal request but the server has no mmproj loaded", ERROR_TYPE_SERVER);
        s.release();
        return true;
    }

    // row-aligned token view: media rows are LLAMA_TOKEN_NULL placeholders
    llama_tokens rows;
    rows.reserve(s.task->tokens.size());
    for (size_t r = 0; r < s.task->tokens.size(); ++r) {
        rows.push_back(s.task->tokens[r]);
    }
    s.prompt.tokens = server_tokens(rows, /* has_mtmd = */ true);
    s.n_media_appended = 0;

    // ordered text/media segments; media segments stay pending (embd NULL)
    // until the scheduler pauses on them and the server appends the encoder
    // output
    std::vector<llama_paged_segment> segs;
    size_t i = 0;
    while (i < rows.size()) {
        if (rows[i] == LLAMA_TOKEN_NULL) {
            const auto & chunk = s.task->tokens.find_chunk(i);
            llama_paged_segment seg = {};
            seg.embd     = nullptr;
            seg.n_tokens = (int32_t) mtmd_input_chunk_get_n_tokens(chunk.get());
            segs.push_back(seg);
            i += seg.n_tokens;
            continue;
        }
        llama_paged_segment seg = {};
        seg.tokens   = rows.data() + i;
        seg.n_tokens = 0;
        while (i < rows.size() && rows[i] != LLAMA_TOKEN_NULL) {
            seg.n_tokens++;
            i++;
        }
        segs.push_back(seg);
    }

    if (!llama_paged_scheduler_add_request_segs(sched, segs.data(), (int32_t) segs.size(), s.id)) {
        host.send_error(s, "failed to queue the multimodal request in the paged scheduler", ERROR_TYPE_SERVER);
        s.release();
        return true;
    }
    return true;
}

void server_scheduler_paged::prepare_drafts() {
    bool any_drafting = false;

    for (auto & slot : host.slot_pool()) {
        // media rows cannot be drafted: the draft context mirrors text only
        if (!slot.can_speculate() || !slot.is_processing() || slot.state != SLOT_STATE_GENERATING ||
            slot.task->tokens.has_mtmd) {
            continue;
        }
        if (!slot.spec_draft.empty()) {
            continue; // carried over from a partial round, reused as-is
        }

        const int n_draft_max = slot.get_n_draft_max();
        if (n_draft_max <= 0) {
            continue;
        }

        // roll the draft context back to the verified end; its KV beyond the
        // prompt belongs to rejected drafts of the previous round
        if (slot.ctx_dft) {
            llama_memory_seq_rm(llama_get_memory(slot.ctx_dft), slot.id,
                                (llama_pos) slot.prompt.tokens.size(), -1);
        }

        slot.spec_prompt = slot.prompt.tokens.get_text_tokens();
        common_speculative_get_draft_params(host.spec_ctx(), slot.id) = {
            /* .drafting = */ true,
            /* .n_max    = */ n_draft_max,
            /* .pos0     = */ slot.prompt.tokens.pos_next(),
            /* .id_last  = */ slot.sampled,
            /* .prompt   = */ &slot.spec_prompt,
            /* .result   = */ &slot.spec_draft,
        };
        any_drafting = true;
    }

    if (any_drafting) {
        // draft on the main thread while a worker serves the task queue
        host.yield([&]() { common_speculative_draft(host.spec_ctx()); });
    }

    for (auto & slot : host.slot_pool()) {
        if (!slot.can_speculate() || slot.state != SLOT_STATE_GENERATING || slot.spec_draft.empty()) {
            continue;
        }
        if (!llama_paged_scheduler_attach_draft(sched, slot.id, slot.spec_draft.data(),
                                                (int32_t) slot.spec_draft.size())) {
            // budget exhausted: decode without a draft this round
            slot.spec_draft.clear();
            continue;
        }
        // tentatively extend the prompt state with the draft tokens; the
        // pending sampled token is already the last entry of prompt.tokens
        // (pushed by the plain path or by verify_slot)
        slot.prompt.tokens.insert(slot.spec_draft);
    }
}

bool server_scheduler_paged::verify_slot(server_slot & slot, int32_t off, int32_t len,
                                          std::vector<llama_token> & fb_tok,
                                          std::vector<llama_paged_feedback> & fb) {
    const int32_t n_draft = (int32_t) slot.spec_draft.size();
    std::vector<int32_t> idxs(n_draft + 1);
    for (int32_t j = 0; j <= n_draft; ++j) {
        idxs[j] = off + len - n_draft - 1 + j;
    }

    // verify: sample every row, keep the drafts the target agrees with plus
    // the target's bonus token (identical to the default path)
    llama_tokens accepted = common_sampler_sample_and_accept_n(slot.smpl.get(), host.ctx(), idxs, slot.spec_draft);
    slot.spec_draft.clear();

    const int32_t k = (int32_t) accepted.size() - 1;   // accepted drafts
    common_speculative_accept(host.spec_ctx(), slot.id, (uint16_t) k);

    slot.stats.n_draft_accepted    += k;
    slot.stats.n_draft_verif_steps += 1;
    if (slot.n_accepted_per_pos.size() < (size_t) host.params().speculative.draft.n_max) {
        slot.n_accepted_per_pos.resize(host.params().speculative.draft.n_max, 0);
    }
    for (int32_t j = 0; j < k && j < (int32_t) slot.n_accepted_per_pos.size(); ++j) {
        slot.n_accepted_per_pos[j]++;
    }

    // prompt fixup: drop the tentative extension, keep the verified prefix;
    // the bonus token becomes the new pending input
    slot.prompt.tokens.keep_first(slot.prompt.tokens.size() - n_draft);
    slot.prompt.tokens.insert({accepted.begin(), accepted.end() - 1});
    slot.prompt.tokens.push_back(accepted.back());
    slot.sampled = accepted.back();

    if (k < n_draft) {
        // rejected draft rows are still in the KV: roll back to the verified
        // end; the trailing bonus is pending, not yet written to the KV.
        // deferred: the scheduler n_past advances over this verify batch only
        // in the update call below, so the rollback cannot run yet
        spec_pending_truncates.emplace_back(slot.id, (int32_t) slot.prompt.tokens.size() - 1);
    }

    // emit the accepted tokens through the regular stop/streaming pipeline
    int32_t n_emitted = 0;
    bool keep_generating = true;
    for (int32_t j = 0; j <= k; ++j) {
        const llama_token tok = accepted[j];
        slot.stats.n_gen += 1;
        if (slot.stats.n_gen == 1) {
            slot.stats.update_prompt_last();
            slot.t_print_last = ggml_time_us();
            slot.n_gen_last = 0;
        }
        slot.stats.update_gen_last();

        completion_token_output result;
        result.tok          = tok;
        result.text_to_send = common_token_to_piece(host.ctx(), tok,
                host.params().special ||
                slot.task->params.sampling.preserved_tokens.count(tok) > 0);
        result.prob         = 1.0f;

        if (slot.task->params.sampling.n_probs > 0) {
            host.populate_token_probs(slot, result, slot.task->params.post_sampling_probs,
                                      host.params().special, idxs[std::min(j + 1, n_draft)]);
        }

        keep_generating = host.process_token(result, slot);
        n_emitted++;
        if (!keep_generating) {
            break;
        }
    }
    slot.print_timings_tg();

    fb_tok.insert(fb_tok.end(), accepted.begin(), accepted.begin() + n_emitted);
    fb.push_back({ slot.id, fb_tok.data() + (fb_tok.size() - n_emitted), n_emitted, keep_generating ? 0 : 1 });
    if (!keep_generating) {
        slot.print_timings();
        host.send_final_response(slot);
    }
    return keep_generating;
}

std::unique_ptr<server_scheduler_interface> server_scheduler_create(server_scheduler_host & host) {
    if (host.params().kv_paged) {
        return std::make_unique<server_scheduler_paged>(host);
    }
    return nullptr;
}
