#include "ggml.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-paged-scheduler-impl.h"

#include <algorithm>
#include <vector>

struct llama_paged_scheduler {
    llama_paged_scheduler_impl impl;

    llama_paged_scheduler(uint32_t n_ctx, uint32_t block_sz, uint32_t n_batch, int32_t n_pos_per_embd,
                          llama_kv_cache_paged * kv_manager,
                          llama_memory_hybrid_paged * hybrid_memory,
                          bool priority_enabled) :
        impl(n_ctx, block_sz, n_batch, n_pos_per_embd, kv_manager, hybrid_memory, priority_enabled) {}
};

LLAMA_API struct llama_paged_scheduler * llama_paged_scheduler_init(struct llama_context * ctx) {
    if (!ctx) {
        return nullptr;
    }

    // Get the paged kv cache, either directly or through the hybrid wrapper.
    auto * memory = ctx->get_memory();
    auto * hybrid_memory = dynamic_cast<llama_memory_hybrid_paged *>(memory);
    auto * paged_kv = hybrid_memory ? hybrid_memory->get_paged_mem()
                                    : dynamic_cast<llama_kv_cache_paged *>(memory);
    if (!paged_kv) {
        LLAMA_LOG_ERROR(
            "%s: context does not have a paged KV cache. "
            "Make sure to pass --kv-paged (-kvp) and use a "
            "supported architecture. SWA architectures (gemma3, llama4, etc.) "
            "are not yet supported.\n",
            __func__);
        return nullptr;
    }

    // Extract params
    const uint32_t n_ctx    = ctx->n_ctx();
    const uint32_t block_sz = ctx->block_size();
    const uint32_t n_batch  = ctx->n_batch();
    if (!hybrid_memory) {
        GGML_ASSERT(n_batch == ctx->n_ubatch() && "kv_paged requires n_batch == n_ubatch.");
    }
    const int32_t n_pos_per_embd = llama_model_n_pos_per_embd(llama_get_model(ctx));

    try {
        return new llama_paged_scheduler(n_ctx, block_sz, n_batch, n_pos_per_embd, paged_kv, hybrid_memory,
                                         ctx->paged_priority());
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: Error when creating llama_paged_scheduler: %s\n", __func__, e.what());
        return nullptr;
    }
}

LLAMA_API void llama_paged_scheduler_free(struct llama_paged_scheduler * sched) {
    if (sched) {
        delete sched;
    }
}

LLAMA_API bool llama_paged_scheduler_warmup(struct llama_paged_scheduler * sched,
                                            struct llama_context *         ctx) {
    if (!sched || !ctx) {
        return false;
    }

    const uint32_t n_ctx = llama_n_ctx(ctx);
    const uint32_t n_tokens = std::min<uint32_t>(llama_n_batch(ctx), n_ctx > 0 ? n_ctx - 1 : 0);
    if (n_tokens == 0) {
        return false;
    }

    const bool prefix_cache_was_enabled = sched->impl.set_prefix_cache_enabled(false);
    auto run_pass = [&](uint32_t pass_tokens) {
        std::vector<llama_token> tokens(pass_tokens, 0);
        const llama_token bos = llama_vocab_bos(llama_model_get_vocab(llama_get_model(ctx)));
        if (bos != LLAMA_TOKEN_NULL) {
            tokens[0] = bos;
        }

        if (!llama_paged_scheduler_add_request(sched, tokens.data(), (int32_t) tokens.size(), 0, /*priority=*/0)) {
            return false;
        }

        llama_batch batch = {};
        bool decoded = false;
        while (true) {
            if (!llama_paged_scheduler_prepare_batch(sched, &batch)) {
                return false;
            }
            if (batch.n_tokens == 0) {
                break;
            }
            if (llama_decode(ctx, batch) != 0) {
                return false;
            }
            llama_synchronize(ctx);

            const auto * info = llama_paged_scheduler_get_batch_info(sched);
            static const llama_token warm_token = 0;
            std::vector<llama_paged_feedback> fb(info->n_seq);
            for (int32_t i = 0; i < info->n_seq; ++i) {
                fb[i] = { batch.seq_id[info->batch_offsets[i]][0], &warm_token, 1, decoded ? 1 : 0 };
            }
            llama_paged_scheduler_update(sched, &batch, fb.data(), (int32_t) fb.size());
            decoded = true;
        }
        return true;
    };

    // The tiny pass loads common CANN kernels before the paged graph. Keep
    // the large pass to one ubatch so warmup does not exercise the known
    // multi-ubatch async race on 310P.
    const uint32_t tiny_tokens = std::min<uint32_t>(n_tokens, 2);
    const uint32_t graph_tokens = std::min<uint32_t>(n_tokens, llama_n_ubatch(ctx));
    bool ok = run_pass(tiny_tokens);
    if (ok && graph_tokens > tiny_tokens) {
        ok = run_pass(graph_tokens);
    }

    sched->impl.set_prefix_cache_enabled(prefix_cache_was_enabled);
    if (ok) {
        LLAMA_LOG_INFO("%s: warmed paged prefill and decode with %d token(s)\n", __func__, (int) n_tokens);
    }
    return ok;
}

LLAMA_API bool llama_paged_scheduler_prepare_batch(struct llama_paged_scheduler * sched, struct llama_batch * batch) {
    if (!sched || !batch) {
        return false;
    }

    llama_scheduler_status status;
    try {
        status = sched->impl.step(*batch);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, e.what());
        return false;
    }

    if (status == llama_scheduler_status::DEADLOCK) {
        LLAMA_LOG_ERROR("%s: Deadlock detected.\n", __func__);
        return false;
    }

    return true;
}

LLAMA_API bool llama_paged_scheduler_add_request(struct llama_paged_scheduler * sched,
                                                 const llama_token *            tokens,
                                                 int32_t                        n_tokens,
                                                 int32_t                        request_id,
                                                 int32_t                        priority) {
    if (!sched || !tokens) {
        return false;
    }

    llama_sequence_group group;
    group.request_id = request_id;
    group.n_prompt   = n_tokens;
    group.n_decoded  = 0;
    group.n_past     = 0;
    group.priority   = priority;  // clamping and warnings happen in queue_request()
    for (int i = 0; i < n_tokens; ++i) {
        group.logical_seq.push_back(tokens[i]);
    }
    group.t_arrival_time = ggml_time_us();  // microseconds

    return sched->impl.queue_request(group);
}

LLAMA_API bool llama_paged_scheduler_add_request_segs(struct llama_paged_scheduler *      sched,
                                                      const struct llama_paged_segment * segs,
                                                      int32_t                             n_segs,
                                                      int32_t                             request_id) {
    if (!sched || !segs || n_segs <= 0) {
        return false;
    }

    llama_sequence_group group;
    group.request_id     = request_id;
    group.n_prompt       = 0;
    group.n_decoded      = 0;
    group.n_past         = 0;
    group.t_arrival_time = ggml_time_us();

    bool has_media = false;
    for (int32_t i = 0; i < n_segs; ++i) {
        const llama_paged_segment & s = segs[i];
        if (s.n_tokens <= 0) {
            LLAMA_LOG_ERROR("%s: request %d has an empty segment (index %d).\n", __func__, request_id, i);
            return false;
        }

        llama_seq_segment seg;
        seg.start_row = (uint32_t) group.logical_seq.size();
        seg.n_tokens  = (uint32_t) s.n_tokens;
        seg.is_media  = (s.tokens == nullptr);
        if (seg.is_media) {
            has_media = true;
            seg.embd        = s.embd;
            seg.embd_stride = s.embd_stride;
            if (s.pos) {
                seg.pos.assign(s.pos, s.pos + (size_t) s.n_tokens * sched->impl.get_n_pos_per_embd());
            }
            group.logical_seq.insert(group.logical_seq.end(), s.n_tokens, LLAMA_TOKEN_NULL);
        } else {
            group.logical_seq.insert(group.logical_seq.end(), s.tokens, s.tokens + s.n_tokens);
        }
        group.segments.push_back(std::move(seg));
    }
    group.has_media      = has_media;
    group.n_prompt       = (uint32_t) group.logical_seq.size();
    group.n_prompt_total = (uint32_t) group.logical_seq.size();

    return sched->impl.queue_request(std::move(group));
}

LLAMA_API bool llama_paged_scheduler_append_segment(struct llama_paged_scheduler *      sched,
                                                    int32_t                             request_id,
                                                    const struct llama_paged_segment * seg) {
    if (!sched || !seg) {
        return false;
    }

    llama_sequence_group * group = sched->impl.get_group_from_id(request_id);
    if (group == nullptr) {
        LLAMA_LOG_ERROR("%s: request_id=%d does not exist.\n", __func__, request_id);
        return false;
    }
    LLAMA_LOG_ERROR("%s: request %d append at n_past=%d, n_segments=%zu\n", __func__, request_id, group->n_past,
                   group->segments.size());

    for (auto & s : group->segments) {
        if (s.is_media && s.embd == nullptr && s.start_row == group->n_past) {
            if (seg->embd == nullptr || seg->embd_stride <= 0) {
                LLAMA_LOG_ERROR("%s: request %d appended segment has no embeddings.\n", __func__, request_id);
                return false;
            }
            s.embd        = seg->embd;
            s.embd_stride = seg->embd_stride;
            if (seg->pos && seg->n_tokens) {
                s.pos.assign(seg->pos, seg->pos + (size_t) seg->n_tokens * sched->impl.get_n_pos_per_embd());
            }
            group->paused_external = false;
            if (group->status == llama_sequence_group_status::PAUSED_EXTERNAL) {
                group->status = llama_sequence_group_status::RUNNING;
            }
            LLAMA_LOG_INFO("%s: request %d segment at row %d appended (%d rows)\n", __func__, request_id,
                           (int) s.start_row, s.n_tokens);
            return true;
        }
    }
    LLAMA_LOG_ERROR("%s: request %d has no pending media segment at row %d.\n", __func__, request_id, group->n_past);
    return false;
}

LLAMA_API void llama_paged_scheduler_update(struct llama_paged_scheduler *      sched,
                                            struct llama_batch *               batch,
                                            const struct llama_paged_feedback * feedback,
                                            int32_t                             n_feedback) {
    if (!sched) {
        return;
    }
    // a batch of mid-prefill rows carries no feedback but must still advance
    // the KV positions
    if ((!feedback || n_feedback <= 0) && (!batch || batch->n_tokens == 0)) {
        return;
    }
    if (batch != nullptr && batch->n_tokens > 0) {
        const auto * info = sched->impl.get_curr_batch_info();
        GGML_ASSERT(info != nullptr && "no batch info was set.");
    }
    sched->impl.update(batch, feedback, n_feedback);
}

LLAMA_API bool llama_paged_scheduler_fork_request(struct llama_paged_scheduler * sched,
                                                   int32_t                        src_id,
                                                   int32_t                        dst_id) {
    if (!sched) {
        return false;
    }
    return sched->impl.fork_request(src_id, dst_id);
}

LLAMA_API bool llama_paged_scheduler_attach_draft(struct llama_paged_scheduler * sched,
                                                   int32_t                        request_id,
                                                   const llama_token *            draft,
                                                   int32_t                        n_draft) {
    if (!sched) {
        return false;
    }
    return sched->impl.attach_draft(request_id, draft, n_draft);
}

LLAMA_API bool llama_paged_scheduler_truncate(struct llama_paged_scheduler * sched,
                                               int32_t                        request_id,
                                               int32_t                        n_past_new) {
    if (!sched) {
        return false;
    }
    return sched->impl.truncate(request_id, n_past_new);
}

LLAMA_API void llama_paged_scheduler_set_on_finish(struct llama_paged_scheduler * sched,
                                                   llama_paged_on_finish_cb       cb,
                                                   void *                         user_data) {
    sched->impl.set_on_finish(cb, user_data);
}

LLAMA_API bool llama_paged_scheduler_get_seq_state(struct llama_paged_scheduler * sched,
                                                   int32_t                        request_id,
                                                   struct llama_paged_seq_state * out_state) {
    if (!sched || !out_state) {
        return false;
    }

    llama_sequence_group * group = sched->impl.get_group_from_id(request_id);
    if (group == nullptr) {
        LLAMA_LOG_ERROR("%s: request_id=%d does not exist.", __func__, request_id);
        return false;
    }

    out_state->request_id       = group->request_id;
    out_state->n_prompt         = group->n_prompt;
    out_state->n_prompt_total   = group->n_prompt_total;
    out_state->n_decoded        = group->n_decoded;
    out_state->n_past           = group->n_past;
    out_state->paused           = group->paused_external ? 1 : 0;
    out_state->t_arrival_us     = group->t_arrival_time;
    out_state->t_first_token_us = group->t_first_token_us;
    out_state->priority         = group->priority;

    switch (group->status) {
        case llama_sequence_group_status::PENDING:
            out_state->status = LLAMA_PAGED_SEQ_STATUS_UNKNOWN;
            break;
        case llama_sequence_group_status::WAITING:
            out_state->status = LLAMA_PAGED_SEQ_STATUS_WAITING;
            break;
        case llama_sequence_group_status::RUNNING:
            out_state->status = LLAMA_PAGED_SEQ_STATUS_RUNNING;
            break;
        case llama_sequence_group_status::SWAPPED:
            out_state->status = LLAMA_PAGED_SEQ_STATUS_SWAPPED;
            break;
        case llama_sequence_group_status::FINISHED:
            out_state->status = LLAMA_PAGED_SEQ_STATUS_FINISHED;
            break;
    }
    return true;
}

LLAMA_API const struct llama_paged_batch_info * llama_paged_scheduler_get_batch_info(
    const struct llama_paged_scheduler * sched) {
    if (!sched) {
        return nullptr;
    }
    return sched->impl.get_curr_batch_info();
}

LLAMA_API bool llama_paged_scheduler_cancel(struct llama_paged_scheduler * sched, int32_t request_id) {
    if (!sched) {
        return false;
    }
    return sched->impl.cancel(request_id);
}

LLAMA_API int32_t llama_paged_scheduler_max_request_id(const struct llama_paged_scheduler * sched) {
    if (!sched) {
        return 0;
    }
    return sched->impl.max_request_id();
}
