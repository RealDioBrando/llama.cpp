// tests/req_priority/test-priority-e2e.cpp
//
// End-to-end inference test for request-priority scheduling.
//
// Phase 1 - admission order. The same six requests are queued twice on the
// paged scheduler: once in default FCFS mode and once with --enable-priority
// semantics. Priorities are increasing (request i gets priority -i), so the
// last request is the most urgent. It checks:
//   1. FCFS serves requests in enqueue order, ignoring the priority argument.
//   2. Priority mode serves them in priority order (smaller value first).
//   3. The first generated token of every request is identical in both modes
//      (prefill runs as a single-sequence batch, so this must be bit-exact).
//
// Phase 2 - swap-out / swap-in order. Four requests run under GPU-block
// pressure: every resident request holds exactly one block and there is no
// free block, so once they all cross the block boundary some of them must be
// swapped out to CPU. Prompts are padded so every request reaches the boundary
// on the same step, and the two survivors then finish on the same step, which
// swaps the evicted requests back in together. Two priority assignments are
// compared against the same FCFS run (FCFS ignores priorities):
//   4. Swap-out victims: with priorities {0:-1, 1:0, 2:-3, 3:-5} the two least
//      urgent requests (ids 0 and 1) are evicted and the two most urgent (ids
//      2 and 3) keep running. FCFS instead evicts the two newest arrivals
//      (ids 2 and 3) and keeps the two oldest.
//   5. Swap-in order: with priorities {0:-5, 1:-3, 2:0, 3:-1} the swapped pair
//      (ids 2 and 3) must come back most-urgent-first (3 before 2), whereas
//      FCFS brings it back in arrival order (2 before 3).
// Both phases drive real prefill + decode runs; phase 2 also checks that every
// request still generates all of its requested tokens after the swap round
// trips. It sizes its own block budget (N_SWAP_REQUESTS GPU blocks) and block
// size, so -ngpub/-ncpub do not apply to it.
//
// The test defaults to the CPU backend and a single device so it runs on any CI
// machine (the paged path requires all layers on one device). Pass `-ngl N`
// (with `-sm none -mg <id>` if needed) to run the same checks on an accelerator.
// It is meant to be run with the small model downloaded by the
// test-download-model fixture.
//
// Build (tests must be enabled; LLAMA_BUILD_TESTS defaults to ON for a
// standalone build; the binary is written to build/bin/):
//
//   # Ascend NPU build (this fork)
//   export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.1.0
//   cmake -B build -DGGML_CANN=ON -DGGML_CUDA=OFF
//   cmake --build build --target test-priority-e2e -j
//
// Run:
//
//   # On an accelerator (the paged path requires every layer on one device)
//   source /usr/local/Ascend/cann-9.1.0/set_env.sh
//   ./build/bin/test-priority-e2e -m /path/to/model.gguf -ngl 99 -sm none -mg 0
//
// Other arguments after -m are parsed by common_params_parse: -c N, -b N,
// -ub N (n_batch must equal n_ubatch), -kvbls N (KV block size), -ngpub N,
// -ncpub N. Use an F16 model on the CANN backend; quantized (e.g. Q8_0)
// matmul is not supported there and aborts.
//
// ctest:
//
//   ctest --test-dir build -R test-priority-e2e
//
// ctest pulls tinyllamas/stories15M through the test-download-model fixture,
// which needs network access to huggingface.co. On offline machines run the
// binary directly with any local .gguf as shown above.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
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

static constexpr int32_t N_REQUESTS      = 6;
static constexpr int32_t N_PREDICT       = 4;
static constexpr int32_t N_SWAP_REQUESTS = 4;

// Distinct first tokens avoid any accidental prefix sharing.
static const char * k_prompts[N_REQUESTS] = {
    "France's capital city is",
    "Mount Everest is located in",
    "At zero degrees Celsius water",
    "Two plus two equals four",
    "The largest ocean on Earth is",
    "A group of lions is called",
};

struct run_result {
    std::vector<int32_t>                        admission_order;  // first appearance in a batch
    std::map<int32_t, std::vector<llama_token>> tokens;
    // Per step, recorded after prepare_batch() and before the decode:
    std::vector<std::map<int32_t, llama_paged_seq_status>> step_states;
    std::vector<std::vector<int32_t>>                      step_batch_order;
    // Tokenized prompts actually used (used to build the swap scenario).
    std::vector<std::vector<llama_token>> prompts;
};

struct run_cfg {
    int32_t          n_ctx        = 256;
    int32_t          n_batch      = 64;
    int32_t          n_ubatch     = 64;
    int32_t          block_size   = 16;
    uint32_t         n_gpu_blocks = 128;
    uint32_t         n_cpu_blocks = 32;
    int32_t          n_gpu_layers = 0;
    llama_split_mode split_mode   = LLAMA_SPLIT_MODE_NONE;
    int32_t          main_gpu     = 0;
    int32_t          n_predict    = N_PREDICT;
    // Per request. Empty means request i gets priority -i.
    std::vector<int32_t> priorities;
};

static run_result run_mode(const std::string & model_path,
                           bool                priority,
                           const run_cfg &     cfg,
                           std::vector<std::vector<llama_token>> prompts = {}) {
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
    params.n_sequences   = (int32_t) (prompts.empty() ? N_REQUESTS : prompts.size());
    params.n_parallel    = params.n_sequences;
    params.n_gpu_blocks  = cfg.n_gpu_blocks;
    params.n_cpu_blocks  = cfg.n_cpu_blocks;
    // Block accounting in this test assumes every block is usable.
    params.kv_paged_watermark = 0.0f;
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

    if (prompts.empty()) {
        prompts.reserve(N_REQUESTS);
        for (int32_t i = 0; i < N_REQUESTS; ++i) {
            std::vector<llama_token> tokens = common_tokenize(ctx, k_prompts[i], true);
            EXPECT_TRUE(!tokens.empty());
            prompts.push_back(std::move(tokens));
        }
    }
    const int32_t n_requests = (int32_t) prompts.size();

    // Priority is always supplied (FCFS records and ignores it, warning once).
    // With the default mapping the most urgent request is the last enqueued.
    for (int32_t i = 0; i < n_requests; ++i) {
        const int32_t prio = cfg.priorities.empty() ? -i : cfg.priorities.at(i);
        const bool    ok   = llama_paged_scheduler_add_request(sched, prompts[i].data(),
                                                             (int32_t) prompts[i].size(), i, prio);
        EXPECT_TRUE(ok);
    }

    std::map<int32_t, common_sampler *> samplers;
    for (int32_t i = 0; i < n_requests; ++i) {
        samplers[i] = common_sampler_init(model, params.sampling);
        EXPECT_TRUE(samplers[i] != nullptr);
    }

    run_result              result;
    std::map<int32_t, bool> seen;
    std::vector<bool>       done(n_requests, false);
    int32_t                 n_finished = 0;
    llama_batch             batch      = {};

    for (int step = 0; step < 1000 && n_finished < n_requests; ++step) {
        EXPECT_TRUE(llama_paged_scheduler_prepare_batch(sched, &batch));
        if (batch.n_tokens == 0) {
            break;
        }

        const llama_paged_batch_info * info = llama_paged_scheduler_get_batch_info(sched);
        EXPECT_TRUE(info != nullptr && info->n_seq > 0);

        // Snapshot the lifecycle state of every request for this step. Taken
        // before the decode so that swap-out / swap-in decisions made while
        // preparing the batch are visible.
        std::map<int32_t, llama_paged_seq_status> states;
        for (int32_t rid = 0; rid < n_requests; ++rid) {
            if (done[rid]) {
                states[rid] = LLAMA_PAGED_SEQ_STATUS_FINISHED;
                continue;
            }
            llama_paged_seq_state state = {};
            EXPECT_TRUE(llama_paged_scheduler_get_seq_state(sched, rid, &state));
            states[rid] = state.status;
        }
        result.step_states.push_back(std::move(states));

        std::vector<int32_t> batch_order;
        batch_order.reserve(info->n_seq);
        for (int i = 0; i < info->n_seq; ++i) {
            batch_order.push_back(batch.seq_id[info->batch_offsets[i]][0]);
        }
        result.step_batch_order.push_back(std::move(batch_order));

        EXPECT_TRUE(llama_decode(ctx, batch) == 0);
        llama_synchronize(ctx);

        std::vector<llama_token> sampled(info->n_seq, LLAMA_TOKEN_NULL);
        std::vector<int8_t>      stop(info->n_seq, 0);
        std::vector<llama_paged_feedback> fbs;

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
                stop[i] = (llama_vocab_is_eog(vocab, tok) || (int32_t) toks.size() >= cfg.n_predict) ? 1 : 0;
                if (stop[i]) {
                    ++n_finished;
                    done[rid] = true;
                }
                fbs.push_back({ rid, &sampled[i], 1, stop[i] });
            }
        }

        llama_paged_scheduler_update(sched, &batch, fbs.empty() ? nullptr : fbs.data(), (int32_t) fbs.size());
    }

    EXPECT_TRUE(n_finished == n_requests);

    for (auto & [rid, smpl] : samplers) {
        common_sampler_free(smpl);
    }
    llama_paged_scheduler_free(sched);

    result.prompts = std::move(prompts);
    return result;
}

// Scheduler ordering rules used by priority mode, mirrored here because the
// test only sees the public API. Request ids double as arrival indices: the
// phases queue requests 0..n-1 in order, so a larger id arrived later.
static bool evicted_before(int32_t a, int32_t b, const std::vector<int32_t> & prio, bool by_priority) {
    if (by_priority && prio[a] != prio[b]) {
        return prio[a] > prio[b];  // larger value = lower priority = evicted first
    }
    return a > b;  // later arrival first
}

static bool served_before(int32_t a, int32_t b, const std::vector<int32_t> & prio, bool by_priority) {
    if (by_priority && prio[a] != prio[b]) {
        return prio[a] < prio[b];  // smaller value = higher priority = swapped in first
    }
    return a < b;  // earlier arrival first
}

// Build a scenario where every resident request holds exactly one GPU block
// and crosses the block boundary at the same step, forcing a swap-out.
static void build_swap_scenario(const run_cfg &                              base,
                                const std::vector<std::vector<llama_token>> & tokenized,
                                run_cfg &                                     out_cfg,
                                std::vector<std::vector<llama_token>> &       out_prompts) {
    EXPECT_TRUE((int32_t) tokenized.size() >= N_SWAP_REQUESTS);
    std::vector<std::vector<llama_token>> prompts(tokenized.begin(), tokenized.begin() + N_SWAP_REQUESTS);

    size_t p_max = 0;
    for (const auto & p : prompts) {
        p_max = std::max(p_max, p.size());
    }
    // Pad every prompt to the same length so all requests reach block
    // boundaries on the same step.
    for (auto & p : prompts) {
        p.resize(p_max, p.back());
    }

    // One block must hold the whole prompt plus the decode slot reserved at
    // admission, so block_size > p_max.
    int32_t block_size = base.block_size;
    while ((size_t) block_size <= p_max) {
        block_size *= 2;
    }

    out_cfg              = base;
    out_cfg.block_size   = block_size;
    out_cfg.n_gpu_blocks = N_SWAP_REQUESTS;     // one block per resident request, no spare
    out_cfg.n_cpu_blocks = 2 * N_SWAP_REQUESTS;
    // Decode until just past the first block boundary (n_past == block_size).
    out_cfg.n_predict = block_size + 3 - (int32_t) p_max;
    out_cfg.n_ctx     = std::max<int32_t>(out_cfg.n_ctx, (int32_t) p_max + out_cfg.n_predict + 8);
    // Priorities are scenario-specific; FCFS ignores them.
    out_cfg.priorities.clear();
    out_prompts        = std::move(prompts);
}

static void check_swap_scenario(const char *                 label,
                                const run_result &           r,
                                const run_cfg &              cfg,
                                bool                         by_priority,
                                const std::vector<int32_t> & want_outs,
                                const std::vector<int32_t> & want_ins) {
    const int32_t n = (int32_t) r.prompts.size();
    EXPECT_TRUE(n == N_SWAP_REQUESTS);
    // FCFS ignores priorities; mirror the scheduler default so the ordering
    // helpers below have a well-defined key.
    std::vector<int32_t> prio = cfg.priorities;
    if (prio.empty()) {
        prio.resize(n);
        for (int32_t i = 0; i < n; ++i) {
            prio[i] = -i;
        }
    }
    EXPECT_TRUE((int32_t) prio.size() == n);

    std::vector<int32_t> swap_outs;  // ids swapped out, in step order
    std::vector<int32_t> swap_ins;   // ids swapped in, in service order

    for (size_t k = 1; k < r.step_states.size(); ++k) {
        const auto & prev = r.step_states[k - 1];
        const auto & curr = r.step_states[k];

        std::vector<int32_t> running_before;
        std::vector<int32_t> swapped_before;
        std::vector<int32_t> out;
        std::vector<int32_t> in;
        for (int32_t rid = 0; rid < n; ++rid) {
            const auto before = prev.at(rid);
            const auto after  = curr.at(rid);
            if (before == LLAMA_PAGED_SEQ_STATUS_RUNNING) {
                running_before.push_back(rid);
            }
            if (before == LLAMA_PAGED_SEQ_STATUS_SWAPPED) {
                swapped_before.push_back(rid);
            }
            if (before == LLAMA_PAGED_SEQ_STATUS_RUNNING && after == LLAMA_PAGED_SEQ_STATUS_SWAPPED) {
                out.push_back(rid);
            }
            if (before == LLAMA_PAGED_SEQ_STATUS_SWAPPED && after == LLAMA_PAGED_SEQ_STATUS_RUNNING) {
                in.push_back(rid);
            }
        }

        if (!out.empty()) {
            // The victims must be exactly the highest-ranked entries of the
            // eviction order among the requests that were running.
            std::sort(running_before.begin(), running_before.end(),
                      [&](int32_t a, int32_t b) { return evicted_before(a, b, prio, by_priority); });
            std::vector<int32_t> want(running_before.begin(), running_before.begin() + out.size());
            std::vector<int32_t> got = out;
            std::sort(got.begin(), got.end());
            std::sort(want.begin(), want.end());
            EXPECT_TRUE(got == want);
            swap_outs.insert(swap_outs.end(), got.begin(), got.end());
        }

        if (!in.empty()) {
            // Swap-ins must be a prefix of the swapped queue under the service
            // order, and the batch reflects that order.
            std::sort(swapped_before.begin(), swapped_before.end(),
                      [&](int32_t a, int32_t b) { return served_before(a, b, prio, by_priority); });
            std::vector<int32_t> want(swapped_before.begin(), swapped_before.begin() + in.size());

            std::vector<int32_t> in_order;
            for (const int32_t rid : r.step_batch_order[k]) {
                if (std::find(in.begin(), in.end(), rid) != in.end()) {
                    in_order.push_back(rid);
                }
            }
            EXPECT_TRUE(in_order == want);
            swap_ins.insert(swap_ins.end(), in_order.begin(), in_order.end());
        }
    }

    fprintf(stderr, "  %-34s swap-out order: ", label);
    for (const int32_t id : swap_outs) {
        fprintf(stderr, "%d ", id);
    }
    fprintf(stderr, "\n  %-34s swap-in order:  ", label);
    for (const int32_t id : swap_ins) {
        fprintf(stderr, "%d ", id);
    }
    fprintf(stderr, "\n");

    // The pressure scenario must actually have swapped requests out and back.
    EXPECT_TRUE(!swap_outs.empty());
    EXPECT_TRUE(!swap_ins.empty());
    EXPECT_TRUE(swap_outs == want_outs);
    EXPECT_TRUE(swap_ins == want_ins);

    // Every request still generates its full output after the swap round trip.
    for (int32_t rid = 0; rid < n; ++rid) {
        EXPECT_TRUE((int32_t) r.tokens.at(rid).size() == cfg.n_predict);
    }
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
    if (params.n_batch < N_REQUESTS) {
        // After admission every request decodes in the same step, so the batch
        // must hold one token per request.
        fprintf(stderr, "error: -b must be >= %d (got %d)\n", N_REQUESTS, (int) params.n_batch);
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
        fprintf(stderr, "test-priority-e2e: admission order: running FCFS\n");
        const run_result fcfs = run_mode(params.model.path, /*priority=*/ false, cfg);

        fprintf(stderr, "test-priority-e2e: admission order: running priority mode\n");
        const run_result prio = run_mode(params.model.path, /*priority=*/ true, cfg);

        // 1. FCFS: queue order, priority must be ignored.
        const std::vector<int32_t> want_fcfs = { 0, 1, 2, 3, 4, 5 };
        fprintf(stderr, "  FCFS admission order:     ");
        for (const int32_t id : fcfs.admission_order) {
            fprintf(stderr, "%d ", id);
        }
        fprintf(stderr, "\n");
        EXPECT_TRUE(fcfs.admission_order == want_fcfs);

        // 2. Priority: smallest value first (-5, -4, ..., 0 -> 5, 4, ..., 0).
        const std::vector<int32_t> want_prio = { 5, 4, 3, 2, 1, 0 };
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

        // 4./5. Swap-out and swap-in must follow the same ordering rules.
        run_cfg                               swap_cfg;
        std::vector<std::vector<llama_token>> swap_prompts;
        build_swap_scenario(cfg, fcfs.prompts, swap_cfg, swap_prompts);

        // FCFS ignores priorities, so one run covers both priority assignments.
        fprintf(stderr, "test-priority-e2e: swap pressure (%d GPU blocks, %d requests): running FCFS\n",
                (int) swap_cfg.n_gpu_blocks, N_SWAP_REQUESTS);
        const run_result swap_fcfs = run_mode(params.model.path, /*priority=*/ false, swap_cfg, swap_prompts);
        check_swap_scenario("FCFS", swap_fcfs, swap_cfg, /*by_priority=*/ false, { 2, 3 }, { 2, 3 });

        // 4. Priority mode evicts the two least urgent residents, not the two
        //    newest arrivals.
        run_cfg swap_prio_outs            = swap_cfg;
        swap_prio_outs.priorities         = { -1, 0, -3, -5 };
        fprintf(stderr, "test-priority-e2e: swap pressure: running priority mode (eviction)\n");
        const run_result swap_prio_out = run_mode(params.model.path, /*priority=*/ true, swap_prio_outs, swap_prompts);
        check_swap_scenario("priority (eviction)", swap_prio_out, swap_prio_outs, /*by_priority=*/ true, { 0, 1 },
                            { 0, 1 });

        // 5. Priority mode swaps the evicted pair back in most-urgent-first,
        //    the reverse of their arrival order.
        run_cfg swap_prio_ins             = swap_cfg;
        swap_prio_ins.priorities          = { -5, -3, 0, -1 };
        fprintf(stderr, "test-priority-e2e: swap pressure: running priority mode (swap-in)\n");
        const run_result swap_prio_in = run_mode(params.model.path, /*priority=*/ true, swap_prio_ins, swap_prompts);
        check_swap_scenario("priority (swap-in)", swap_prio_in, swap_prio_ins, /*by_priority=*/ true, { 2, 3 },
                            { 3, 2 });

        fprintf(stderr, "test-priority-e2e: PASSED\n");
    } catch (const std::exception & e) {
        fprintf(stderr, "test-priority-e2e: %s\n", e.what());
        llama_backend_free();
        return 1;
    }

    llama_backend_free();
    return 0;
}
