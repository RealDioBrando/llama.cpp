// tests/test-paged-kv-e2e.cpp
//
// End-to-end equivalence test for paged KV cache.
// We compare top-K agreement rather than raw logit values because the paged
// attention path uses a custom CUDA kernel with online softmax, while the
// unified path uses standard ggml attention with two-pass softmax. The two
// produce mathematically equivalent results but with different F16
// accumulation order, drifts on the order of 0.05-0.5 in raw
// logit values is expected and not a correctness issue. Top-K set agreement
// is robust to this drift while still catching the real issues (e.g. cross-device
// reads, layout corruption, MQA broadcast bugs) which produce wildly
// different distributions.
//
// Also samples N_COMPARE tokens greedy as a secondary 'cheap' check.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#define EXPECT_TRUE(x)                                                   \
    do {                                                                 \
        if (!(x)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            throw std::runtime_error("FAILED assertion.");               \
        }                                                                \
    } while (0)

static constexpr const char * TEST_PROMPT       = "Once upon a time there was a lovely";

// Long enough to reach at least one checkpoint lattice point (N=4 blocks)
// before the prompt end. With the lattice rule the deepest reusable boundary
// is the largest b <= (prompt_blocks - 1) with b % N == 0.
static constexpr const char * TEST_LONG_PROMPT  =
    "Once upon a time there was a lovely little girl who lived with her mother "
    "in a small cottage near the edge of the forest. Every morning she walked "
    "to the river to fetch water, and every evening she sat by the fire and "
    "listened to the stories her mother told about the world beyond the hills. "
    "One winter the river froze, so she followed the deer tracks through the "
    "snow until she found a clearing where the moonlight fell like silver on "
    "the ice, and there she waited for the morning to come.";
static constexpr int          N_PREDICT         = 16;
static constexpr int          N_COMPARE         = 4;  // token-equivalence window
static constexpr int          TOP_K             = 5;
static constexpr int          MIN_TOP_K_OVERLAP = 4;  // at least 4 of top-5 must match

// Result of running one path: prefill-final logits + sampled token sequence.
struct path_result {
    std::vector<float>       prefill_logits;  // [n_vocab]
    std::vector<llama_token> tokens;          // [N_PREDICT]
    int                      n_vocab = 0;
};

static path_result run_non_paged(const std::string & model_path) {
    common_params params;
    params.model.path    = model_path;
    params.n_ctx         = 256;
    params.n_batch       = 64;
    params.n_ubatch      = 64;
    params.n_predict     = N_PREDICT;
    params.sampling.temp = 0.0f;  // greedy
    params.warmup        = false;
    params.kv_paged      = false;
    params.cpuparams.n_threads       = 1;
    params.cpuparams_batch.n_threads = 1;

    auto            init  = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    EXPECT_TRUE(model != nullptr);
    EXPECT_TRUE(ctx != nullptr);

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, TEST_PROMPT, true);
    EXPECT_TRUE(!prompt_tokens.empty());

    // Prefill
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    EXPECT_TRUE(llama_decode(ctx, batch) == 0);

    // Capture prefill-final logits BEFORE any further decode steps overwrite them.
    path_result result;
    result.n_vocab = n_vocab;
    {
        const float * raw = llama_get_logits_ith(ctx, -1);  // last logit
        EXPECT_TRUE(raw != nullptr);
        result.prefill_logits.assign(raw, raw + n_vocab);
    }

    // Sample N_PREDICT tokens for the secondary token-equivalence check.
    common_sampler * smpl = common_sampler_init(model, params.sampling);
    EXPECT_TRUE(smpl != nullptr);

    llama_token cur = -1;
    for (int i = 0; i < N_PREDICT; ++i) {
        cur = common_sampler_sample(smpl, ctx, -1);
        common_sampler_accept(smpl, cur, true);
        result.tokens.push_back(cur);
        if (llama_vocab_is_eog(vocab, cur)) {
            break;
        }

        llama_batch step = llama_batch_get_one(&cur, 1);
        EXPECT_TRUE(llama_decode(ctx, step) == 0);
    }

    common_sampler_free(smpl);
    return result;
}

static path_result run_paged(const std::string & model_path) {
    common_params params;
    params.model.path    = model_path;
    params.n_ctx         = 256;
    params.n_batch       = 64;
    params.n_ubatch      = 64;
    params.n_predict     = N_PREDICT;
    params.sampling.temp = 0.0f;  // greedy
    params.warmup        = false;
    params.kv_paged      = true;
    // The paged path requires every layer on one device; this host exposes
    // several CANN devices and the default layer split would spread the model.
    params.split_mode    = LLAMA_SPLIT_MODE_NONE;
    params.main_gpu      = 0;
    params.n_gpu_blocks  = 64;
    params.n_cpu_blocks  = 16;
    params.n_sequences   = 1;
    params.n_parallel    = 1;
    params.fit_params    = false;
    params.n_gpu_blocks  = 4096;
    params.n_cpu_blocks  = 512;
    params.cpuparams.n_threads       = 1;
    params.cpuparams_batch.n_threads = 1;

    auto            init  = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    EXPECT_TRUE(model != nullptr);
    EXPECT_TRUE(ctx != nullptr);

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    llama_paged_scheduler * sched = llama_paged_scheduler_init(ctx);
    EXPECT_TRUE(sched != nullptr);

    // load the CANN kernels and prime the paged graph before the first real
    // batch, mirroring the server's paged warmup
    llama_paged_scheduler_warmup(sched, ctx);

    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, TEST_PROMPT, true);
    EXPECT_TRUE(!prompt_tokens.empty());

    bool ok = llama_paged_scheduler_add_request(sched, prompt_tokens.data(), prompt_tokens.size(), 0, /*priority=*/0);
    EXPECT_TRUE(ok);

    common_sampler * smpl = common_sampler_init(model, params.sampling);
    EXPECT_TRUE(smpl != nullptr);

    path_result result;
    result.n_vocab                      = n_vocab;
    bool        captured_prefill_logits = false;
    llama_batch batch                   = {};

    while ((int) result.tokens.size() < N_PREDICT) {
        bool prepared = llama_paged_scheduler_prepare_batch(sched, &batch);
        EXPECT_TRUE(prepared);
        if (batch.n_tokens == 0) {
            break;
        }

        EXPECT_TRUE(llama_decode(ctx, batch) == 0);
        llama_synchronize(ctx);

        const llama_paged_batch_info * info = llama_paged_scheduler_get_batch_info(sched);
        EXPECT_TRUE(info != nullptr && info->n_seq == 1);

        const int32_t last_idx = info->batch_offsets[0] + info->batch_lens[0] - 1;

        // First decode is the prefill — capture its final logits before
        // sampling anything else.
        if (!captured_prefill_logits) {
            const float * raw = llama_get_logits_ith(ctx, last_idx);
            EXPECT_TRUE(raw != nullptr);
            result.prefill_logits.assign(raw, raw + n_vocab);
            captured_prefill_logits = true;
        }

        llama_token next = common_sampler_sample(smpl, ctx, last_idx);
        common_sampler_accept(smpl, next, true);
        result.tokens.push_back(next);

        bool   stop      = llama_vocab_is_eog(vocab, next) || (int) result.tokens.size() >= N_PREDICT;
        int8_t stop_flag = stop ? 1 : 0;
        llama_paged_feedback fb = { batch.seq_id[info->batch_offsets[0]][0], &next, 1, stop_flag };
        llama_paged_scheduler_update(sched, &batch, &fb, 1);
        if (stop) {
            break;
        }
    }

    common_sampler_free(smpl);
    llama_paged_scheduler_free(sched);
    return result;
}

static void compare_results(const path_result & ref, const path_result & paged) {
    auto top_k = [](const std::vector<float> & l, int k) {
        std::vector<int> idx(l.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&l](int a, int b) { return l[a] > l[b]; });
        idx.resize(k);
        return idx;
    };

    const auto top_ref   = top_k(ref.prefill_logits, TOP_K);
    const auto top_paged = top_k(paged.prefill_logits, TOP_K);

    // Argmax must match: the most-confident next token should be identical.
    EXPECT_TRUE(top_ref[0] == top_paged[0]);

    // Top-K set overlap: at least MIN_TOP_K_OVERLAP of the K most likely
    // tokens must appear in both distributions.
    std::set<int> ref_set(top_ref.begin(), top_ref.end());
    int           overlap = 0;
    for (int t : top_paged) {
        if (ref_set.count(t)) {
            overlap++;
        }
    }

    fprintf(stderr, "test-paged-kv-e2e: top-%d argmax match: ref=%d paged=%d\n", TOP_K, top_ref[0], top_paged[0]);
    fprintf(stderr, "test-paged-kv-e2e: top-%d set overlap: %d/%d (require >= %d)\n", TOP_K, overlap, TOP_K,
            MIN_TOP_K_OVERLAP);

    if (overlap < MIN_TOP_K_OVERLAP) {
        fprintf(stderr,
                "FAIL: top-%d distributions diverge too much. Only %d of %d most-likely "
                "tokens match between ref and paged. Real correctness issue likely.\n",
                TOP_K, overlap, TOP_K);
        fprintf(stderr, "  ref:   ");
        for (int t : top_ref) {
            fprintf(stderr, "%d(%.3f) ", t, ref.prefill_logits[t]);
        }
        fprintf(stderr, "\n  paged: ");
        for (int t : top_paged) {
            fprintf(stderr, "%d(%.3f) ", t, paged.prefill_logits[t]);
        }
        fprintf(stderr, "\n");
        throw std::runtime_error("FAILED test.");
    }

    // Token-level secondary check: first N_COMPARE tokens must match.
    // We don't compare beyond N_COMPARE because greedy sampling on small
    // models is sensitive to argmax tiebreakers, and minor floating-point
    // accumulation differences between the paged and non-paged paths can
    // flip individual tokens after a handful of decode steps.
    EXPECT_TRUE((int) ref.tokens.size() >= N_COMPARE);
    EXPECT_TRUE((int) paged.tokens.size() >= N_COMPARE);
    for (int i = 0; i < N_COMPARE; ++i) {
        if (ref.tokens[i] != paged.tokens[i]) {
            fprintf(stderr, "FAIL: token %d differs in the equivalence window: ref=%d paged=%d\n", i, ref.tokens[i],
                    paged.tokens[i]);
            throw std::runtime_error("FAILED test.");
        }
    }
    fprintf(stderr, "test-paged-kv-e2e: PASSED\n");
}

// Exercises cancel on waiting and running requests, unknown-id rejection and
// request-id reuse after cancel.
static void run_cancel(const std::string & model_path) {
    common_params params;
    params.model.path    = model_path;
    params.n_ctx         = 256;
    params.n_batch       = 64;
    params.n_ubatch      = 64;
    params.n_predict     = N_PREDICT;
    params.sampling.temp = 0.0f;  // greedy
    params.warmup        = false;
    params.kv_paged      = true;
    params.split_mode    = LLAMA_SPLIT_MODE_NONE;
    params.main_gpu      = 0;
    params.n_sequences   = 2;
    params.n_parallel    = 2;
    params.fit_params    = false;
    params.n_gpu_blocks  = 4096;
    params.n_cpu_blocks  = 512;
    params.cpuparams.n_threads       = 1;
    params.cpuparams_batch.n_threads = 1;

    auto            init  = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    EXPECT_TRUE(model != nullptr);
    EXPECT_TRUE(ctx != nullptr);

    llama_paged_scheduler * sched = llama_paged_scheduler_init(ctx);
    EXPECT_TRUE(sched != nullptr);

    llama_paged_scheduler_warmup(sched, ctx);

    EXPECT_TRUE(llama_paged_scheduler_max_request_id(sched) >= 2);

    std::vector<int32_t> finished;
    llama_paged_scheduler_set_on_finish(
        sched,
        [](int32_t request_id, const llama_token *, int32_t, void * ud) {
            static_cast<std::vector<int32_t> *>(ud)->push_back(request_id);
        },
        &finished);

    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, TEST_PROMPT, true);
    EXPECT_TRUE(!prompt_tokens.empty());

    EXPECT_TRUE(llama_paged_scheduler_add_request(sched, prompt_tokens.data(), (int32_t) prompt_tokens.size(), 0));
    EXPECT_TRUE(llama_paged_scheduler_add_request(sched, prompt_tokens.data(), (int32_t) prompt_tokens.size(), 1));
    // duplicate id must be rejected while the request is queued
    EXPECT_TRUE(!llama_paged_scheduler_add_request(sched, prompt_tokens.data(), (int32_t) prompt_tokens.size(), 0));

    common_sampler * smpl = common_sampler_init(model, params.sampling);
    EXPECT_TRUE(smpl != nullptr);

    llama_batch batch = {};
    int         steps = 0;
    while (steps < 8) {
        EXPECT_TRUE(llama_paged_scheduler_prepare_batch(sched, &batch));
        if (batch.n_tokens == 0) {
            break;
        }

        EXPECT_TRUE(llama_decode(ctx, batch) == 0);
        llama_synchronize(ctx);

        const llama_paged_batch_info * info = llama_paged_scheduler_get_batch_info(sched);
        EXPECT_TRUE(info != nullptr);

        std::vector<llama_token> tokens;
        std::vector<int8_t>      stops;
        for (int i = 0; i < info->n_seq; ++i) {
            const int32_t off = info->batch_offsets[i];
            const int32_t len = info->batch_lens[i];
            llama_paged_seq_state st = {};
            EXPECT_TRUE(llama_paged_scheduler_get_seq_state(sched, batch.seq_id[off][0], &st));
            llama_token t = LLAMA_TOKEN_NULL;
            if (len == 1 || (len > 1 && (int32_t) st.n_past + len == (int32_t) st.n_prompt_total)) {
                t = common_sampler_sample(smpl, ctx, off + len - 1);
                common_sampler_accept(smpl, t, true);
            }
            tokens.push_back(t);
            stops.push_back(0);
        }
        std::vector<llama_paged_feedback> fb;
        for (int i = 0; i < info->n_seq; ++i) {
            if (tokens[i] != LLAMA_TOKEN_NULL) {
                const int32_t off = info->batch_offsets[i];
                fb.push_back({ batch.seq_id[off][0], &tokens[i], 1, stops[i] });
            }
        }
        llama_paged_scheduler_update(sched, &batch, fb.data(), (int32_t) fb.size());
        steps++;

        if (steps == 1) {
            // one waiting request is admitted per step, so request 1 is
            // still waiting: cancel it and check the callback fired
            EXPECT_TRUE(llama_paged_scheduler_cancel(sched, 1));
            EXPECT_TRUE(!llama_paged_scheduler_cancel(sched, 42));
            EXPECT_TRUE(finished.size() == 1 && finished[0] == 1);
        } else if (steps == 2) {
            // request 0 is running: cancel it mid-generation
            EXPECT_TRUE(llama_paged_scheduler_cancel(sched, 0));
            EXPECT_TRUE(finished.size() == 2 && finished[1] == 0);
        }
    }

    EXPECT_TRUE(finished.size() == 2);

    // the id is free again after cancel
    EXPECT_TRUE(llama_paged_scheduler_add_request(sched, prompt_tokens.data(), (int32_t) prompt_tokens.size(), 0));
    EXPECT_TRUE(llama_paged_scheduler_cancel(sched, 0));

    common_sampler_free(smpl);
    llama_paged_scheduler_free(sched);
    fprintf(stderr, "test-paged-kv-e2e: cancel PASSED\n");
}

// Two identical requests, run one after the other. The second one must reuse
// a linear-state checkpoint + full-attention blocks, and must produce the same
// next-token distribution as the first (full prefill).
static void test_prefix_reuse(const std::string & model_path) {
    common_params params;
    params.model.path    = model_path;
    params.n_ctx         = 512;
    params.n_batch       = 64;
    params.n_ubatch      = 64;
    params.n_predict     = 8;
    params.sampling.temp = 0.0f;
    params.warmup        = false;
    params.kv_paged      = true;
    params.split_mode    = LLAMA_SPLIT_MODE_NONE;
    params.main_gpu      = 0;
    params.n_gpu_blocks  = 4096;
    params.n_cpu_blocks  = 512;
    params.n_sequences   = 2;
    params.n_parallel    = 2;
    params.fit_params    = false;
    params.cpuparams.n_threads       = 1;
    params.cpuparams_batch.n_threads = 1;

    auto            init  = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    EXPECT_TRUE(model != nullptr);
    EXPECT_TRUE(ctx != nullptr);

    const llama_vocab * vocab      = llama_model_get_vocab(model);
    const int           n_vocab    = llama_vocab_n_tokens(vocab);
    const int           block_size = (int) params.block_size;

    llama_paged_scheduler * sched = llama_paged_scheduler_init(ctx);
    EXPECT_TRUE(sched != nullptr);

    std::vector<llama_token> prompt = common_tokenize(ctx, TEST_LONG_PROMPT, true);
    EXPECT_TRUE((int) prompt.size() >= 3 * block_size);

    // Runs one request until it stops. `first_batch` optionally supplies an
    // already-prepared batch (used to inspect the admission state).
    auto run_request = [&](int32_t request_id, llama_batch * first_batch,
                           std::vector<float> * first_logits) {
        if (!first_batch) {
            EXPECT_TRUE(llama_paged_scheduler_add_request(sched, prompt.data(), (int32_t) prompt.size(), request_id, 0));
        }

        common_sampler * smpl = common_sampler_init(model, params.sampling);
        EXPECT_TRUE(smpl != nullptr);

        llama_batch batch  = {};
        int32_t     consumed = 0;
        bool        have_batch = first_batch != nullptr;
        int32_t     base_past  = 0;
        if (have_batch) {
            batch = *first_batch;
            llama_paged_seq_state st0 = {};
            EXPECT_TRUE(llama_paged_scheduler_get_seq_state(sched, request_id, &st0));
            base_past = (int32_t) st0.n_past;
        }

        while (true) {
            if (!have_batch) {
                EXPECT_TRUE(llama_paged_scheduler_prepare_batch(sched, &batch));
            }
            have_batch = false;
            if (batch.n_tokens == 0) {
                break;
            }
            EXPECT_TRUE(llama_decode(ctx, batch) == 0);
            llama_synchronize(ctx);

            const auto * info = llama_paged_scheduler_get_batch_info(sched);
            EXPECT_TRUE(info != nullptr && info->n_seq == 1);

            consumed += info->batch_lens[0];
            const bool final_prefill = base_past + consumed == (int32_t) prompt.size();

            if (!final_prefill) {
                // intermediate prefill chunk (even a one-token one), no output token yet
                llama_paged_scheduler_update(sched, &batch, nullptr, 0);
                continue;
            }

            const int32_t last_idx = info->batch_offsets[0] + info->batch_lens[0] - 1;
            if (first_logits && final_prefill) {
                const float * raw = llama_get_logits_ith(ctx, last_idx);
                EXPECT_TRUE(raw != nullptr);
                first_logits->assign(raw, raw + n_vocab);
            }

            llama_token next = common_sampler_sample(smpl, ctx, last_idx);
            common_sampler_accept(smpl, next, true);

            // one generated token is enough to validate the prefix
            llama_paged_feedback fb = { request_id, &next, 1, /*stop=*/1 };
            llama_paged_scheduler_update(sched, &batch, &fb, 1);
            break;
        }

        common_sampler_free(smpl);
    };

    std::vector<float> logits0;
    run_request(0, nullptr, &logits0);
    EXPECT_TRUE((int) logits0.size() == n_vocab);

    // Second identical request: the first prepared batch must start from the
    // reused prefix and leave at least one suffix token.
    EXPECT_TRUE(llama_paged_scheduler_add_request(sched, prompt.data(), (int32_t) prompt.size(), 1, 0));
    llama_batch batch = {};
    EXPECT_TRUE(llama_paged_scheduler_prepare_batch(sched, &batch));
    EXPECT_TRUE(batch.n_tokens > 0);

    llama_paged_seq_state st = {};
    EXPECT_TRUE(llama_paged_scheduler_get_seq_state(sched, 1, &st));
    // Hybrid models only checkpoint on the lattice b % N == base % N (default
    // N = 4, base = 0), so the deepest reusable boundary is the largest
    // lattice point below the prompt end. A non-hybrid paged cache has no
    // linear state to clamp it and can reuse every matched block.
    const int  ckpt_interval  = 4;
    const int  max_blocks     = ((int) prompt.size() - 1) / block_size;
    const bool hybrid         = llama_model_is_hybrid(model);
    const int  expected_blocks = hybrid ? (max_blocks / ckpt_interval) * ckpt_interval : max_blocks;
    const int  expected_reuse  = expected_blocks * block_size;
    EXPECT_TRUE(expected_reuse > 0);
    fprintf(stderr, "test-paged-kv-e2e: request 1 reused %u tokens (expected %d)\n", st.n_past, expected_reuse);
    EXPECT_TRUE((int) st.n_past == expected_reuse);

    std::vector<float> logits1;
    run_request(1, &batch, &logits1);
    EXPECT_TRUE((int) logits1.size() == n_vocab);

    const int k = 5;
    auto top_k = [](const std::vector<float> & l, int kk) {
        std::vector<int> idx(l.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&l](int a, int b) { return l[a] > l[b]; });
        idx.resize(kk);
        return idx;
    };
    const auto t0 = top_k(logits0, k);
    const auto t1 = top_k(logits1, k);
    EXPECT_TRUE(t0[0] == t1[0]);
    std::set<int> s0(t0.begin(), t0.end());
    int overlap = 0;
    for (int t : t1) {
        overlap += s0.count(t) ? 1 : 0;
    }
    fprintf(stderr, "test-paged-kv-e2e: prefix-reuse top-%d overlap: %d/%d\n", k, overlap, k);
    EXPECT_TRUE(overlap >= 4);

    llama_paged_scheduler_free(sched);
    fprintf(stderr, "test-paged-kv-e2e: PREFIX REUSE PASSED\n");
}

// Two independent requests are prefilled separately, then decoded in the same
// batch. This exercises the multi-sequence paged row map and state scatter.
static void test_multi_seq_decode(const std::string & model_path, const path_result & ref) {
    common_params params;
    params.model.path    = model_path;
    params.n_ctx         = 512;
    params.n_batch       = 64;
    params.n_ubatch      = 64;
    params.n_predict     = 4;
    params.sampling.temp = 0.0f;
    params.warmup        = false;
    params.kv_paged      = true;
    params.split_mode    = LLAMA_SPLIT_MODE_NONE;
    params.main_gpu      = 0;
    params.n_gpu_blocks  = 4096;
    params.n_cpu_blocks  = 512;
    params.n_sequences   = 2;
    params.n_parallel    = 2;
    params.fit_params    = false;
    params.cpuparams.n_threads       = 1;
    params.cpuparams_batch.n_threads = 1;

    auto            init  = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    EXPECT_TRUE(model != nullptr);
    EXPECT_TRUE(ctx != nullptr);

    llama_paged_scheduler * sched = llama_paged_scheduler_init(ctx);
    EXPECT_TRUE(sched != nullptr);

    std::vector<llama_token> prompt0 = common_tokenize(ctx, TEST_PROMPT, true);
    std::vector<llama_token> prompt1 = common_tokenize(ctx, TEST_LONG_PROMPT, true);

    EXPECT_TRUE(llama_paged_scheduler_add_request(sched, prompt0.data(), (int32_t) prompt0.size(), 0, 0));
    EXPECT_TRUE(llama_paged_scheduler_add_request(sched, prompt1.data(), (int32_t) prompt1.size(), 1, 0));

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    common_sampler * smpl = common_sampler_init(model, params.sampling);
    EXPECT_TRUE(smpl != nullptr);

    bool     saw_multi = false;
    int      done[2]   = { 0, 0 };
    int      consumed[2] = { 0, 0 };
    std::vector<float> logits0;

    llama_batch batch = {};
    for (int step = 0; step < 16 && (done[0] < 2 || done[1] < 2); ++step) {
        EXPECT_TRUE(llama_paged_scheduler_prepare_batch(sched, &batch));
        if (batch.n_tokens == 0) {
            break;
        }
        EXPECT_TRUE(llama_decode(ctx, batch) == 0);
        llama_synchronize(ctx);

        const auto * info = llama_paged_scheduler_get_batch_info(sched);
        EXPECT_TRUE(info != nullptr);
        saw_multi |= info->n_seq > 1;

        std::vector<llama_token>          next(info->n_seq, 0);
        std::vector<int8_t>               stop(info->n_seq, 0);
        std::vector<llama_paged_feedback> fbs;
        for (int s = 0; s < info->n_seq; ++s) {
            const int32_t offset = info->batch_offsets[s];
            const int32_t last   = offset + info->batch_lens[s] - 1;
            const int32_t rid    = batch.seq_id[offset][0];
            EXPECT_TRUE(rid == 0 || rid == 1);

            consumed[rid] += info->batch_lens[s];
            const int prompt_len = rid == 0 ? (int) prompt0.size() : (int) prompt1.size();
            if (info->batch_lens[s] > 1 && consumed[rid] < prompt_len) {
                continue;  // intermediate prefill chunk, no output yet
            }

            if (rid == 0 && logits0.empty()) {
                const float * raw = llama_get_logits_ith(ctx, last);
                EXPECT_TRUE(raw != nullptr);
                logits0.assign(raw, raw + n_vocab);
            }

            next[s] = common_sampler_sample(smpl, ctx, last);
            common_sampler_accept(smpl, next[s], true);
            done[rid]++;
            stop[s] = done[rid] >= 2 ? 1 : 0;
            fbs.push_back({ rid, &next[s], 1, stop[s] });
        }
        llama_paged_scheduler_update(sched, &batch, fbs.data(), (int32_t) fbs.size());
    }

    fprintf(stderr, "test-paged-kv-e2e: multi-seq batch observed: %s\n", saw_multi ? "yes" : "no");
    EXPECT_TRUE(saw_multi);
    EXPECT_TRUE((int) logits0.size() == ref.n_vocab);

    auto top_k = [](const std::vector<float> & l, int k) {
        std::vector<int> idx(l.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&l](int a, int b) { return l[a] > l[b]; });
        idx.resize(k);
        return idx;
    };
    const auto t0 = top_k(ref.prefill_logits, 5);
    const auto t1 = top_k(logits0, 5);
    EXPECT_TRUE(t0[0] == t1[0]);
    std::set<int> s0(t0.begin(), t0.end());
    int overlap = 0;
    for (int t : t1) {
        overlap += s0.count(t) ? 1 : 0;
    }
    fprintf(stderr, "test-paged-kv-e2e: multi-seq top-5 overlap: %d/5\n", overlap);
    EXPECT_TRUE(overlap >= 4);

    common_sampler_free(smpl);
    llama_paged_scheduler_free(sched);
    fprintf(stderr, "test-paged-kv-e2e: MULTI-SEQ PASSED\n");
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PAGED)) {
        fprintf(stderr, "usage: %s -m <model>\n", argv[0]);
        return 1;
    }
    if (params.model.path.empty()) {
        fprintf(stderr, "skip: no --model provided\n");
        return 0;
    }

    common_init();
    llama_backend_init();

    fprintf(stderr, "test-paged-kv-e2e: running non-paged reference\n");
    path_result ref = run_non_paged(params.model.path);
    fprintf(stderr, "  got %zu tokens, %d-vocab logits\n", ref.tokens.size(), ref.n_vocab);

    fprintf(stderr, "test-paged-kv-e2e: running paged path\n");
    path_result paged = run_paged(params.model.path);
    fprintf(stderr, "  got %zu tokens, %d-vocab logits\n", paged.tokens.size(), paged.n_vocab);

    compare_results(ref, paged);

    fprintf(stderr, "test-paged-kv-e2e: running cancel case\n");
    run_cancel(params.model.path);

    fprintf(stderr, "test-paged-kv-e2e: testing prefix reuse\n");
    test_prefix_reuse(params.model.path);

    fprintf(stderr, "test-paged-kv-e2e: testing multi-seq decode\n");
    test_multi_seq_decode(params.model.path, ref);

    llama_backend_free();
    return 0;
}
