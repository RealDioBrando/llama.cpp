// tests/req_priority/test-priority-e2e.cpp
//
// End-to-end inference test for request-priority scheduling.
//
// The same four requests are queued twice on the paged scheduler: once in
// default FCFS mode and once with --enable-priority semantics. It checks:
//   1. FCFS serves requests in enqueue order, ignoring the priority argument.
//   2. Priority mode serves them in priority order (smaller value first).
//   3. The first generated token of every request is identical in both modes
//      (prefill runs as a single-sequence batch, so this must be bit-exact).
//
// The test defaults to the CPU backend and a single device so it runs on any CI
// machine (the paged path requires all layers on one device). Pass `-ngl N`
// (with `-sm none -mg <id>` if needed) to run the same checks on an accelerator.
// It is meant to be run with the small model downloaded by the
// test-download-model fixture.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#define EXPECT_TRUE(x)                                                   \
    do {                                                                 \
        if (!(x)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            throw std::runtime_error("FAILED assertion.");               \
        }                                                                \
    } while (0)

static constexpr int32_t N_REQUESTS = 4;
static constexpr int32_t N_PREDICT  = 4;

// Distinct first tokens avoid any accidental prefix sharing.
static const char * k_prompts[N_REQUESTS] = {
    "France's capital city is",
    "Mount Everest is located in",
    "At zero degrees Celsius water",
    "Two plus two equals four",
};

struct run_result {
    std::vector<int32_t>                        admission_order;  // first appearance in a batch
    std::map<int32_t, std::vector<llama_token>> tokens;
};

struct run_cfg {
    int32_t         n_ctx        = 256;
    int32_t         n_batch      = 64;
    int32_t         n_ubatch     = 64;
    int32_t         block_size   = 16;
    uint32_t        n_gpu_blocks = 128;
    uint32_t        n_cpu_blocks = 32;
    int32_t         n_gpu_layers = 0;
    llama_split_mode split_mode  = LLAMA_SPLIT_MODE_NONE;
    int32_t         main_gpu     = 0;
};

static run_result run_mode(const std::string & model_path, bool priority, const run_cfg & cfg) {
    common_params params;
    params.model.path    = model_path;
    params.n_ctx         = cfg.n_ctx;
    params.n_batch       = cfg.n_batch;
    params.n_ubatch      = cfg.n_ubatch;
    params.block_size    = cfg.block_size;
    params.sampling.temp = 0.0f;  // greedy
    params.warmup        = false;
    params.kv_paged      = true;
    params.enable_priority = priority;
    params.n_sequences   = N_REQUESTS;
    params.n_parallel    = N_REQUESTS;
    params.n_gpu_blocks  = cfg.n_gpu_blocks;
    params.n_cpu_blocks  = cfg.n_cpu_blocks;
    params.fit_params    = false;
    // Defaults keep the test environment-independent; -ngl/-sm/-mg can move it
    // to an accelerator. The paged path still requires a single device.
    params.n_gpu_layers              = cfg.n_gpu_layers;
    params.split_mode                = cfg.split_mode;
    params.main_gpu                  = cfg.main_gpu;
    params.cpuparams.n_threads       = 2;
    params.cpuparams_batch.n_threads = 2;

    auto            init  = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    EXPECT_TRUE(model != nullptr);
    EXPECT_TRUE(ctx != nullptr);

    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_paged_scheduler * sched = llama_paged_scheduler_init(ctx);
    EXPECT_TRUE(sched != nullptr);

    std::vector<std::vector<llama_token>> prompts;
    prompts.reserve(N_REQUESTS);
    for (int32_t i = 0; i < N_REQUESTS; ++i) {
        std::vector<llama_token> tokens = common_tokenize(ctx, k_prompts[i], true);
        EXPECT_TRUE(!tokens.empty());
        prompts.push_back(std::move(tokens));
    }

    // Priority is always supplied (FCFS records and ignores it, warning once).
    // With -i the most urgent request is the last one enqueued.
    for (int32_t i = 0; i < N_REQUESTS; ++i) {
        const bool ok = llama_paged_scheduler_add_request(sched, prompts[i].data(), (int32_t) prompts[i].size(), i,
                                                          /*priority=*/ -i);
        EXPECT_TRUE(ok);
    }

    std::map<int32_t, common_sampler *> samplers;
    for (int32_t i = 0; i < N_REQUESTS; ++i) {
        samplers[i] = common_sampler_init(model, params.sampling);
        EXPECT_TRUE(samplers[i] != nullptr);
    }

    run_result              result;
    std::map<int32_t, bool> seen;
    int32_t                 n_finished = 0;
    llama_batch             batch      = {};

    for (int step = 0; step < 1000 && n_finished < N_REQUESTS; ++step) {
        EXPECT_TRUE(llama_paged_scheduler_prepare_batch(sched, &batch));
        if (batch.n_tokens == 0) {
            break;
        }

        EXPECT_TRUE(llama_decode(ctx, batch) == 0);
        llama_synchronize(ctx);

        const llama_paged_batch_info * info = llama_paged_scheduler_get_batch_info(sched);
        EXPECT_TRUE(info != nullptr && info->n_seq > 0);

        std::vector<llama_token> sampled(info->n_seq, LLAMA_TOKEN_NULL);
        std::vector<int8_t>      stop(info->n_seq, 0);

        for (int i = 0; i < info->n_seq; ++i) {
            const int32_t rid = batch.seq_id[info->batch_offsets[i]][0];

            if (!seen[rid]) {
                seen[rid] = true;
                result.admission_order.push_back(rid);
            }

            const llama_paged_seq_state state = [&] {
                llama_paged_seq_state s = {};
                EXPECT_TRUE(llama_paged_scheduler_get_seq_state(sched, rid, &s));
                return s;
            }();

            const bool final_prefill = info->batch_lens[i] > 1 &&
                                       state.n_past + info->batch_lens[i] == state.n_prompt_total;
            if (info->batch_lens[i] == 1 || final_prefill) {
                const int32_t last = info->batch_offsets[i] + info->batch_lens[i] - 1;
                common_sampler * smpl = samplers.at(rid);

                const llama_token tok = common_sampler_sample(smpl, ctx, last);
                common_sampler_accept(smpl, tok, /*accept_grammar=*/true);
                sampled[i] = tok;

                std::vector<llama_token> & toks = result.tokens[rid];
                toks.push_back(tok);
                stop[i] = (llama_vocab_is_eog(vocab, tok) || (int32_t) toks.size() >= N_PREDICT) ? 1 : 0;
                if (stop[i]) {
                    ++n_finished;
                }
            }
        }

        llama_paged_scheduler_update(sched, &batch, sampled.data(), stop.data());
    }

    EXPECT_TRUE(n_finished == N_REQUESTS);

    for (auto & [rid, smpl] : samplers) {
        common_sampler_free(smpl);
    }
    llama_paged_scheduler_free(sched);

    return result;
}

int main(int argc, char ** argv) {
    common_params params;
    // Default to a single device and CPU offload; -sm/-mg can be overridden.
    // (n_gpu_layers cannot be pre-set before parsing: the option table asserts
    // its default is negative, so -1/auto is mapped to 0 = CPU below.)
    params.split_mode   = LLAMA_SPLIT_MODE_NONE;
    params.main_gpu     = 0;
    // Paged scheduling requires n_batch == n_ubatch. Keep the defaults equal so
    // passing neither (or both) is fine; a mismatched pair is rejected below.
    params.n_batch      = 64;
    params.n_ubatch     = 64;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PAGED)) {
        fprintf(stderr,
                "usage: %s -m <model> [-ngl N] [-mg ID] [-c N -b N -ub N] [-kvbls N] [-ngpub N -ncpub N]\n"
                "  default backend is CPU; pass -ngl N to offload, with -sm none -mg <id> for paged KV\n",
                argv[0]);
        return 1;
    }
    if (params.model.path.empty()) {
        fprintf(stderr, "skip: no --model provided\n");
        return 0;
    }
    if (params.n_batch != params.n_ubatch) {
        fprintf(stderr, "error: paged scheduling requires -b == -ub (got %d != %d)\n", (int) params.n_batch,
                (int) params.n_ubatch);
        return 1;
    }

    run_cfg cfg;
    cfg.n_ctx        = params.n_ctx > 0 ? (int32_t) params.n_ctx : 256;
    cfg.n_batch      = params.n_batch;
    cfg.n_ubatch     = params.n_ubatch;
    cfg.block_size   = params.block_size;
    cfg.n_gpu_blocks = params.n_gpu_blocks_set ? params.n_gpu_blocks : 128;
    cfg.n_cpu_blocks = params.n_cpu_blocks_set ? params.n_cpu_blocks : 32;
    cfg.n_gpu_layers = params.n_gpu_layers < 0 ? 0 : params.n_gpu_layers;  // -1/auto -> CPU
    cfg.split_mode   = params.split_mode;
    cfg.main_gpu     = params.main_gpu;

    // The scheduler reads these at construction time. Disabling the prefix
    // cache keeps block reuse (and therefore logits) independent of ordering.
    setenv("LLAMA_PAGED_PREFIX_CACHE", "0", 1);
    setenv("LLAMA_PAGED_PREFIX_CKPT", "0", 1);

    common_init();
    llama_backend_init();

    try {
        fprintf(stderr, "test-priority-e2e: running FCFS\n");
        const run_result fcfs = run_mode(params.model.path, /*priority=*/ false, cfg);

        fprintf(stderr, "test-priority-e2e: running priority mode\n");
        const run_result prio = run_mode(params.model.path, /*priority=*/ true, cfg);

        // 1. FCFS: queue order, priority must be ignored.
        const std::vector<int32_t> want_fcfs = { 0, 1, 2, 3 };
        fprintf(stderr, "  FCFS admission order:     ");
        for (const int32_t id : fcfs.admission_order) {
            fprintf(stderr, "%d ", id);
        }
        fprintf(stderr, "\n");
        EXPECT_TRUE(fcfs.admission_order == want_fcfs);

        // 2. Priority: smallest value first (-3, -2, -1, 0 -> 3, 2, 1, 0).
        const std::vector<int32_t> want_prio = { 3, 2, 1, 0 };
        fprintf(stderr, "  priority admission order: ");
        for (const int32_t id : prio.admission_order) {
            fprintf(stderr, "%d ", id);
        }
        fprintf(stderr, "\n");
        EXPECT_TRUE(prio.admission_order == want_prio);

        // 3. Every request is served in both modes and its first token (the
        //    prefill result, computed as a single-sequence batch) is identical.
        EXPECT_TRUE(fcfs.tokens.size() == (size_t) N_REQUESTS);
        EXPECT_TRUE(prio.tokens.size() == (size_t) N_REQUESTS);
        for (int32_t i = 0; i < N_REQUESTS; ++i) {
            const auto & f = fcfs.tokens.at(i);
            const auto & p = prio.tokens.at(i);
            EXPECT_TRUE(!f.empty());
            EXPECT_TRUE(!p.empty());
            EXPECT_TRUE(f[0] == p[0]);
        }

        fprintf(stderr, "test-priority-e2e: PASSED\n");
    } catch (const std::exception & e) {
        fprintf(stderr, "test-priority-e2e: %s\n", e.what());
        llama_backend_free();
        return 1;
    }

    llama_backend_free();
    return 0;
}
