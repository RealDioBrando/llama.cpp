#include "ggml.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-paged-scheduler-impl.h"

#include <algorithm>
#include <vector>

struct llama_paged_scheduler {
    llama_paged_scheduler_impl impl;

    llama_paged_scheduler(uint32_t n_ctx, uint32_t block_sz, uint32_t n_batch,
                          llama_kv_cache_paged * kv_manager,
                          llama_memory_hybrid_paged * hybrid_memory) :
        impl(n_ctx, block_sz, n_batch, kv_manager, hybrid_memory) {}
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

    try {
        return new llama_paged_scheduler(n_ctx, block_sz, n_batch, paged_kv, hybrid_memory);
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

        if (!llama_paged_scheduler_add_request(sched, tokens.data(), (int32_t) tokens.size(), 0)) {
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
            std::vector<llama_token> next_tokens(info->n_seq, 0);
            std::vector<int8_t> stop(info->n_seq, decoded ? 1 : 0);
            llama_paged_scheduler_update(sched, &batch, next_tokens.data(), stop.data());
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
                                                 int32_t                        request_id) {
    if (!sched || !tokens) {
        return false;
    }

    llama_sequence_group group;
    group.request_id = request_id;
    group.n_prompt   = n_tokens;
    group.n_decoded  = 0;
    group.n_past     = 0;
    for (int i = 0; i < n_tokens; ++i) {
        group.logical_seq.push_back(tokens[i]);
    }
    group.t_arrival_time = ggml_time_us();  // int64_t milliseconds

    return sched->impl.queue_request(group);
}

LLAMA_API void llama_paged_scheduler_update(struct llama_paged_scheduler * sched,
                                            struct llama_batch *           batch,
                                            const llama_token *            tokens,
                                            const int8_t *                 stop_flags) {
    if (!sched || !batch || !tokens || !stop_flags) {
        return;
    }

    const auto * info = sched->impl.get_curr_batch_info();
    GGML_ASSERT(info != nullptr && "no batch info was set.");
    std::vector<llama_token> tokens_vec(tokens, tokens + info->n_seq);
    sched->impl.update(*batch, tokens_vec, stop_flags);
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
    out_state->t_arrival_us     = group->t_arrival_time;
    out_state->t_first_token_us = group->t_first_token_us;
    return true;
}

LLAMA_API const struct llama_paged_batch_info * llama_paged_scheduler_get_batch_info(
    const struct llama_paged_scheduler * sched) {
    if (!sched) {
        return nullptr;
    }
    return sched->impl.get_curr_batch_info();
}
