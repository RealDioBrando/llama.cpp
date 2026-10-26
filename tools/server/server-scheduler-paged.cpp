#include "server-scheduler-paged.h"

#include "common.h"
#include "log.h"
#include "sampling.h"

#include <algorithm>

static bool paged_task_supported(const server_task & task) {
    // phase 1: text completion only (chat templates are applied in the HTTP
    // layer and arrive here as COMPLETION tasks, gated by res_type below)
    if (task.type != SERVER_TASK_TYPE_COMPLETION) {
        return false;
    }
    switch (task.params.res_type) {
        case TASK_RESPONSE_TYPE_NONE:        // /completion
        case TASK_RESPONSE_TYPE_OAI_CMPL:    // /v1/completions
            break;
        default:
            return false;
    }
    // grammar / structured output is phase 2
    if (task.params.sampling.grammar.type != COMMON_GRAMMAR_TYPE_NONE) {
        return false;
    }
    if (task.params.n_cmpl > 1) {
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

    if (!paged_task_supported(task)) {
        host.send_error(task, "this feature is not supported with --kv-paged", ERROR_TYPE_NOT_SUPPORTED);
        return true;
    }

    server_slot * slot = pick_idle_slot();
    if (slot == nullptr) {
        return false; // no capacity: the caller defers the task
    }

    if (!host.launch_slot_with_task(*slot, std::move(task))) {
        return true; // rejected, error already sent
    }

    server_slot & s = *slot;
    s.stats.update_prompt_start();
    // server_tokens is move-only; paged phase 1 is text-only (mtmd gated off)
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

    std::vector<llama_token> sampled_tokens;
    std::vector<int8_t>      stop_flags;
    sampled_tokens.reserve(info->n_seq);
    stop_flags.reserve(info->n_seq);

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

        llama_token tok = LLAMA_TOKEN_NULL;
        bool keep_generating = true;

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
        } else {
            slot.state = SLOT_STATE_GENERATING;
        }

        if (sample_now) {
            // the last token of the sequence in this batch carries the logits
            slot.i_batch = off + len - 1;
            tok = common_sampler_sample(slot.smpl.get(), host.ctx(), slot.i_batch);
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

            keep_generating = host.process_token(result, slot);
            if (!keep_generating) {
                slot.print_timings();
                host.send_final_response(slot);
            } else {
                slot.print_timings_tg();
            }

            slot.state = SLOT_STATE_GENERATING;
        }

        sampled_tokens.push_back(tok);
        stop_flags.push_back(keep_generating ? 0 : 1);
    }

    for (const auto & slot : slots) {
        if (slot.is_processing()) {
            m.n_busy_slots++;
        }
    }

    host.metrics_queue_prompt(n_prompt_tokens);

    llama_paged_scheduler_update(sched, &paged_batch, sampled_tokens.data(), stop_flags.data());
}

std::unique_ptr<server_scheduler_interface> server_scheduler_create(server_scheduler_host & host) {
    if (host.params().kv_paged) {
        return std::make_unique<server_scheduler_paged>(host);
    }
    return nullptr;
}
