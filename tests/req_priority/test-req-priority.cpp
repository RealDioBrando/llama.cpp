// Unit tests for request-priority scheduling in the paged scheduler.
//
// Covers:
//   - the waiting/swapped queue abstraction (FCFS deque + priority heap)
//   - ordering predicates (priority -> arrival -> enqueue sequence)
//   - the dual-mode scheduler: default FCFS is unchanged, priority mode orders
//     admission / swap-in / prefill selection / eviction
//
// These tests sit next to test-paged-kv but focus only on request priority.

#include "ggml-backend.h"
#include "llama-kv-cache-paged.h"
#include "llama-paged-queue.h"
#include "llama-paged-scheduler-impl.h"

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <vector>

#define TEST(name) static void name()
#define RUN(name)                                   \
    do {                                            \
        fprintf(stderr, "  running %-48s ", #name); \
        fflush(stderr);                             \
        name();                                     \
        fprintf(stderr, "OK\n");                    \
    } while (0)

#define EXPECT_EQ(a, b)                                                                                          \
    do {                                                                                                         \
        auto _a = (a);                                                                                           \
        auto _b = (b);                                                                                           \
        if (_a != _b) {                                                                                          \
            fprintf(stderr, "\n  FAIL %s:%d: expected %s == %s, got %lld vs %lld\n", __FILE__, __LINE__, #a, #b, \
                    (long long) _a, (long long) _b);                                                             \
            std::abort();                                                                                        \
        }                                                                                                        \
    } while (0)

#define EXPECT_TRUE(x)                                                       \
    do {                                                                     \
        if (!(x)) {                                                          \
            fprintf(stderr, "\n  FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
            std::abort();                                                    \
        }                                                                    \
    } while (0)
#define EXPECT_FALSE(x) EXPECT_TRUE(!(x))

//
// Queue-level helpers and tests
//

static llama_sequence_group_ptr make_qgroup(int32_t id, int32_t priority, int64_t t_arrival, uint64_t arrival_seq) {
    auto g            = std::make_shared<llama_sequence_group>();
    g->request_id     = id;
    g->priority       = priority;
    g->t_arrival_time = t_arrival;
    g->arrival_seq    = arrival_seq;
    return g;
}

static std::vector<int32_t> drain(llama_paged_queue & q) {
    std::vector<int32_t> ids;
    while (!q.empty()) {
        llama_sequence_group_ptr g = q.pop();
        EXPECT_TRUE(g != nullptr);
        ids.push_back(g->request_id);
    }
    return ids;
}

static void expect_ids(const std::vector<int32_t> & got, const std::vector<int32_t> & want) {
    EXPECT_EQ(got.size(), want.size());
    for (size_t i = 0; i < got.size() && i < want.size(); ++i) {
        EXPECT_EQ(got[i], want[i]);
    }
}

TEST(test_paged_queue_fcfs_order) {
    llama_paged_fcfs_queue q;
    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.size(), 0u);

    q.push(make_qgroup(0, 0, 0, 0));
    q.push(make_qgroup(1, 0, 1, 1));
    q.push(make_qgroup(2, 0, 2, 2));
    EXPECT_EQ(q.size(), 3u);
    EXPECT_EQ(q.peek()->request_id, 0);

    expect_ids(drain(q), { 0, 1, 2 });
    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.size(), 0u);
}

TEST(test_paged_queue_fcfs_keyed_insert) {
    llama_paged_fcfs_queue q;
    // Out-of-order arrival keys must be inserted at their keyed position.
    q.push(make_qgroup(2, 0, 2, 2));
    q.push(make_qgroup(0, 0, 0, 0));
    q.push(make_qgroup(1, 0, 1, 1));
    expect_ids(drain(q), { 0, 1, 2 });
}

TEST(test_paged_queue_fcfs_same_arrival_fifo) {
    llama_paged_fcfs_queue q;
    // Same timestamp: arrival_seq decides, FIFO (not LIFO).
    q.push(make_qgroup(1, 0, 100, 1));
    q.push(make_qgroup(2, 0, 100, 2));
    q.push(make_qgroup(3, 0, 100, 3));
    expect_ids(drain(q), { 1, 2, 3 });
}

TEST(test_paged_queue_priority_order) {
    llama_paged_priority_queue q;
    q.push(make_qgroup(10, /*prio*/ 2, /*t*/ 100, /*seq*/ 1));
    q.push(make_qgroup(11, /*prio*/ -7, /*t*/ 200, /*seq*/ 2));
    q.push(make_qgroup(12, /*prio*/ -7, /*t*/ 50, /*seq*/ 3));
    q.push(make_qgroup(13, /*prio*/ -7, /*t*/ 50, /*seq*/ 4));

    // -7 first (t=50 tie broken by seq), then -7 t=200, then prio 2.
    expect_ids(drain(q), { 12, 13, 11, 10 });
}

TEST(test_paged_queue_priority_reinsert) {
    llama_paged_priority_queue q;
    q.push(make_qgroup(10, /*prio*/ -1, /*t*/ 100, /*seq*/ 1));
    q.push(make_qgroup(11, /*prio*/ 0, /*t*/ 200, /*seq*/ 2));

    llama_sequence_group_ptr first = q.pop();
    EXPECT_EQ(first->request_id, 10);

    // Recomputed requests re-enter with unchanged keys: they must land back at
    // their keyed rank, not at the front.
    q.push(std::move(first));
    expect_ids(drain(q), { 10, 11 });
}

TEST(test_paged_queue_empty_size) {
    llama_paged_priority_queue q;
    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.size(), 0u);

    q.push(make_qgroup(1, 0, 0, 0));
    EXPECT_FALSE(q.empty());
    EXPECT_EQ(q.size(), 1u);

    llama_sequence_group_ptr g = q.pop();
    EXPECT_EQ(g->request_id, 1);
    EXPECT_TRUE(q.empty());
    EXPECT_EQ(q.size(), 0u);
}

//
// Scheduler helpers and tests
//

struct paged_test_fixture {
    ggml_backend_t                              backend = nullptr;
    std::unique_ptr<llama_kv_cache_paged>       kv;
    std::unique_ptr<llama_paged_scheduler_impl> sched;

    ~paged_test_fixture() {
        if (backend) {
            ggml_backend_free(backend);
        }
    }

    paged_test_fixture()                                       = default;
    paged_test_fixture(paged_test_fixture &&)                  = default;
    paged_test_fixture & operator=(paged_test_fixture &&)      = default;
    paged_test_fixture(const paged_test_fixture &)             = delete;
    paged_test_fixture & operator=(const paged_test_fixture &) = delete;
};

static paged_test_fixture make_fixture(uint32_t n_ctx          = 128,
                                       uint32_t block_size     = 16,
                                       uint32_t n_batch        = 64,
                                       uint32_t n_gpu_blocks   = 8,
                                       uint32_t n_cpu_blocks   = 2,
                                       bool     priority_enabled = false) {
    paged_test_fixture fixture;
    fixture.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(fixture.backend != nullptr);

    fixture.kv = std::unique_ptr<llama_kv_cache_paged>(new llama_kv_cache_paged(
        /*head_dim=*/64u,
        /*n_heads_kv=*/4u,
        block_size,
        /*n_layers=*/2u,
        n_batch,
        /*n_seq_max=*/8u));
    fixture.kv->init(fixture.backend, fixture.backend, GGML_TYPE_F16, n_gpu_blocks, n_cpu_blocks, /*watermark=*/0.0f);

    fixture.sched = std::unique_ptr<llama_paged_scheduler_impl>(
        new llama_paged_scheduler_impl(n_ctx, block_size, n_batch, /*n_pos_per_embd=*/1, fixture.kv.get(), nullptr, priority_enabled));

    // Identical dummy prompts would otherwise hit the prefix cache and rewrite
    // n_prompt/n_past, which makes block budgets unstable.
    fixture.sched->set_prefix_cache_enabled(false);
    return fixture;
}

static llama_sequence_group make_group(int32_t request_id, uint32_t n_prompt, int32_t priority = 0) {
    llama_sequence_group group;
    group.request_id     = request_id;
    group.n_prompt       = n_prompt;
    group.n_decoded      = 0;
    group.n_past         = 0;
    group.priority       = priority;
    group.t_arrival_time = request_id;  // control ordering based on request_id
    group.logical_seq.assign(n_prompt, /*dummy token=*/1);
    return group;
}

// Step once and feed back a token for every scheduled sequence. Returns the
// request id that was served (the tests always have n_seq == 1). Mid-prefill
// chunks advance the KV only: feedback tokens would bump n_decoded and make
// a chunked prefill look like a finished decode.
static int32_t step_and_update(paged_test_fixture & f, bool stop) {
    llama_batch batch = {};
    EXPECT_TRUE(f.sched->step(batch) == llama_scheduler_status::OK);
    const auto * info = f.sched->get_curr_batch_info();
    EXPECT_TRUE(info->n_seq == 1);
    const int32_t rid = (int32_t) batch.seq_id[0][0];

    std::vector<llama_token> tok(info->n_seq, 7);
    std::vector<int8_t>      flags(info->n_seq, stop ? 1 : 0);

    auto * group = f.sched->get_group_from_id(rid);
    const bool final_prefill = info->batch_lens[0] > 1 && group != nullptr &&
                               (int32_t) group->n_past + info->batch_lens[0] == (int32_t) group->n_prompt_total;

    llama_paged_feedback fb = { rid, tok.data(), (int32_t) tok.size(), flags[0] };
    if (final_prefill || info->batch_lens[0] == 1) {
        f.sched->update(&batch, &fb, 1);
    } else {
        f.sched->update(&batch, nullptr, 0); // mid-prefill chunk: KV advance only
    }
    llama_batch_free(batch);
    return rid;
}

TEST(test_scheduler_priority_default_fcfs) {
    auto fixture = make_fixture(128, 16, 64, 8, 2, /*priority_enabled=*/true);

    for (int32_t id = 0; id < 3; ++id) {
        EXPECT_TRUE(fixture.sched->queue_request(make_group(id, 8, /*priority=*/0)));
    }

    // Equal priorities: pure arrival order.
    EXPECT_EQ(step_and_update(fixture, true), 0);
    EXPECT_EQ(step_and_update(fixture, true), 1);
    EXPECT_EQ(step_and_update(fixture, true), 2);
}

TEST(test_scheduler_fcfs_mode_ignores_priority) {
    // Priority scheduling disabled: a later, higher-priority request must NOT
    // jump ahead.
    auto fixture = make_fixture(128, 16, 64, 8, 2, /*priority_enabled=*/false);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(0, 8, /*priority=*/0)));
    EXPECT_TRUE(fixture.sched->queue_request(make_group(1, 8, /*priority=*/-10)));

    EXPECT_EQ(step_and_update(fixture, true), 0);
    EXPECT_EQ(step_and_update(fixture, true), 1);
}

TEST(test_scheduler_arrival_seq_monotonic) {
    auto fixture = make_fixture(128, 16, 64, 8, 2, /*priority_enabled=*/true);

    auto g10 = make_group(10, 8);
    auto g11 = make_group(11, 8);
    auto g12 = make_group(12, 8);
    // Deliberately decreasing timestamps: seq must still be monotonic.
    g10.t_arrival_time = 300;
    g11.t_arrival_time = 200;
    g12.t_arrival_time = 100;

    EXPECT_TRUE(fixture.sched->queue_request(g10));
    EXPECT_TRUE(fixture.sched->queue_request(g11));
    EXPECT_TRUE(fixture.sched->queue_request(g12));

    const auto * a = fixture.sched->get_group_from_id(10);
    const auto * b = fixture.sched->get_group_from_id(11);
    const auto * c = fixture.sched->get_group_from_id(12);
    EXPECT_TRUE(a && b && c);
    EXPECT_TRUE(a->arrival_seq < b->arrival_seq);
    EXPECT_TRUE(b->arrival_seq < c->arrival_seq);
}

TEST(test_scheduler_priority_higher_admitted_first) {
    auto fixture = make_fixture(128, 16, 64, 8, 2, /*priority_enabled=*/true);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(0, 8, /*priority=*/0)));
    EXPECT_TRUE(fixture.sched->queue_request(make_group(1, 8, /*priority=*/-10)));

    llama_batch batch = {};
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    const auto * info = fixture.sched->get_curr_batch_info();
    EXPECT_EQ(info->n_seq, 1);
    EXPECT_EQ((int32_t) batch.seq_id[0][0], 1);

    // The skipped request is untouched.
    const auto * low = fixture.sched->get_group_from_id(0);
    EXPECT_TRUE(low != nullptr);
    EXPECT_TRUE(low->status == llama_sequence_group_status::WAITING);
    EXPECT_EQ(low->n_past, 0u);
}

TEST(test_scheduler_priority_old_low_yields_to_new_high) {
    auto fixture = make_fixture(128, 16, 64, 8, 2, /*priority_enabled=*/true);

    auto low  = make_group(0, 8, /*priority=*/5);
    auto high = make_group(1, 8, /*priority=*/0);
    low.t_arrival_time  = 1;
    high.t_arrival_time = 1000;

    EXPECT_TRUE(fixture.sched->queue_request(low));
    EXPECT_TRUE(fixture.sched->queue_request(high));

    EXPECT_EQ(step_and_update(fixture, true), 1);
    EXPECT_EQ(step_and_update(fixture, true), 0);
}

TEST(test_scheduler_priority_equal_tie_break_fifo) {
    // Part A: ids 3 then 1, same priority and timestamp -> 3 then 1.
    {
        auto fixture = make_fixture(128, 16, 64, 8, 2, /*priority_enabled=*/true);
        auto g3 = make_group(3, 8, 0);
        auto g1 = make_group(1, 8, 0);
        g3.t_arrival_time = 42;
        g1.t_arrival_time = 42;
        EXPECT_TRUE(fixture.sched->queue_request(g3));
        EXPECT_TRUE(fixture.sched->queue_request(g1));

        EXPECT_EQ(step_and_update(fixture, true), 3);
        EXPECT_EQ(step_and_update(fixture, true), 1);
    }
    // Part B: reversed insertion order -> 1 then 3.
    {
        auto fixture = make_fixture(128, 16, 64, 8, 2, /*priority_enabled=*/true);
        auto g3 = make_group(3, 8, 0);
        auto g1 = make_group(1, 8, 0);
        g3.t_arrival_time = 42;
        g1.t_arrival_time = 42;
        EXPECT_TRUE(fixture.sched->queue_request(g1));
        EXPECT_TRUE(fixture.sched->queue_request(g3));

        EXPECT_EQ(step_and_update(fixture, true), 1);
        EXPECT_EQ(step_and_update(fixture, true), 3);
    }
}

TEST(test_scheduler_waiting_order_priority_then_arrival) {
    auto fixture = make_fixture(128, 16, 64, 8, 2, /*priority_enabled=*/true);

    const int32_t priorities[] = { -5, 0, -5, 0 };
    for (int32_t id = 0; id < 4; ++id) {
        EXPECT_TRUE(fixture.sched->queue_request(make_group(id, 8, priorities[id])));
    }

    // (prio, t): 0=(-5,0), 2=(-5,2), 1=(0,1), 3=(0,3)
    EXPECT_EQ(step_and_update(fixture, true), 0);
    EXPECT_EQ(step_and_update(fixture, true), 2);
    EXPECT_EQ(step_and_update(fixture, true), 1);
    EXPECT_EQ(step_and_update(fixture, true), 3);
}

TEST(test_scheduler_running_stays_arrival_ordered) {
    auto fixture = make_fixture(128, 16, 64, 8, 2, /*priority_enabled=*/true);

    // B is higher priority and gets admitted first, but A arrived earlier.
    EXPECT_TRUE(fixture.sched->queue_request(make_group(0, 8, /*priority=*/0)));
    EXPECT_TRUE(fixture.sched->queue_request(make_group(1, 8, /*priority=*/-10)));

    // Step 1: B prefills alone.
    EXPECT_EQ(step_and_update(fixture, false), 1);
    // Step 2: A prefills alone (prefill phase arbitration).
    EXPECT_EQ(step_and_update(fixture, false), 0);

    // Step 3: both decode; `running` must be in arrival order (A before B).
    llama_batch batch = {};
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    const auto * info = fixture.sched->get_curr_batch_info();
    EXPECT_EQ(info->n_seq, 2);
    EXPECT_EQ((int32_t) batch.seq_id[info->batch_offsets[0]][0], 0);
    EXPECT_EQ((int32_t) batch.seq_id[info->batch_offsets[1]][0], 1);

    std::vector<llama_token> tok(info->n_seq, 7);
    std::vector<int8_t>      stop(info->n_seq, 0);
    llama_paged_feedback     fbs[2] = {
        { batch.seq_id[info->batch_offsets[0]][0], &tok[0], 1, stop[0] },
        { batch.seq_id[info->batch_offsets[1]][0], &tok[1], 1, stop[1] },
    };
    fixture.sched->update(&batch, fbs, 2);
}

TEST(test_scheduler_priority_ordering_strict_total_order) {
    llama_sequence_group X;
    X.priority = 0;   X.t_arrival_time = 5;  X.arrival_seq = 1;
    llama_sequence_group Y;
    Y.priority = -5;  Y.t_arrival_time = 9;  Y.arrival_seq = 2;
    llama_sequence_group Z;
    Z.priority = -5;  Z.t_arrival_time = 5;  Z.arrival_seq = 3;
    llama_sequence_group W;
    W.priority = -5;  W.t_arrival_time = 5;  W.arrival_seq = 4;

    const llama_sequence_group * groups[] = { &X, &Y, &Z, &W };

    // Irreflexive.
    for (const auto * g : groups) {
        EXPECT_FALSE(group_precedes(*g, *g));
    }
    // Antisymmetric + total: exactly one direction for each distinct pair.
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = 0; j < 4; ++j) {
            if (i == j) {
                continue;
            }
            EXPECT_FALSE(group_precedes(*groups[i], *groups[j]) && group_precedes(*groups[j], *groups[i]));
        }
    }
    for (size_t i = 0; i < 4; ++i) {
        for (size_t j = i + 1; j < 4; ++j) {
            EXPECT_TRUE(group_precedes(*groups[i], *groups[j]) || group_precedes(*groups[j], *groups[i]));
        }
    }
    // Transitive and expected ranking: Z < W < Y < X.
    EXPECT_TRUE(group_precedes(Z, W));
    EXPECT_TRUE(group_precedes(W, Y));
    EXPECT_TRUE(group_precedes(Z, Y));
    EXPECT_TRUE(group_precedes(Y, X));

    // Sorting by arrival only ignores priority (used by running/FCFS).
    EXPECT_TRUE(arrival_precedes(X, Z));
    EXPECT_TRUE(arrival_precedes(Z, W));
    EXPECT_TRUE(arrival_precedes(W, Y));
    EXPECT_TRUE(arrival_precedes(X, Y));
}

// Replicates the eviction scan so the selection rule is testable without
// exposing scheduler internals. The real scheduler now scans the whole running
// list; groups already in candidates are removed by the caller before eviction.
static int eviction_victim_index(const std::vector<llama_sequence_group> & run, bool by_priority) {
    int victim = -1;
    for (size_t i = 0; i < run.size(); ++i) {
        if (run[i].status == llama_sequence_group_status::FINISHED) {
            continue;
        }
        if (victim < 0 || eviction_precedes(run[i], run[victim], by_priority)) {
            victim = (int) i;
        }
    }
    return victim;
}

TEST(test_scheduler_priority_eviction_victim) {
    // Arrival order: A(prio +10), B(prio 0), H(prio -10). The global scan must
    // pick A even though it was processed first and may be in candidates.
    std::vector<llama_sequence_group> run(3);
    run[0].request_id = 0; run[0].priority = 10;  run[0].arrival_seq = 1;
    run[1].request_id = 1; run[1].priority = 0;   run[1].arrival_seq = 2;
    run[2].request_id = 2; run[2].priority = -10; run[2].arrival_seq = 3;

    EXPECT_EQ(eviction_victim_index(run, /*by_priority=*/true), 0);
    EXPECT_EQ(run[0].request_id, 0);

    // Same priority: latest arrival (largest arrival_seq) wins.
    std::vector<llama_sequence_group> same(3);
    same[0].priority = 0; same[0].arrival_seq = 1;
    same[1].priority = 0; same[1].arrival_seq = 3;
    same[2].priority = 0; same[2].arrival_seq = 2;
    EXPECT_EQ(eviction_victim_index(same, /*by_priority=*/true), 1);

    // FCFS ignores priority and still sacrifices the latest arrival.
    std::vector<llama_sequence_group> fcfs(2);
    fcfs[0].priority = -100; fcfs[0].arrival_seq = 1;
    fcfs[1].priority = 100;  fcfs[1].arrival_seq = 2;
    EXPECT_EQ(eviction_victim_index(fcfs, /*by_priority=*/false), 1);

    // FINISHED groups are never victims.
    same[1].status = llama_sequence_group_status::FINISHED;
    EXPECT_EQ(eviction_victim_index(same, /*by_priority=*/true), 2);
}

TEST(test_scheduler_priority_evicts_scheduled_low_priority) {
    // A (low priority) is scanned first and takes the last free block. B (high
    // priority) then fails to allocate. The victim must be A even though A is
    // already part of this step's candidates; A is removed from candidates and
    // swapped out so that B can proceed.
    auto fixture = make_fixture(128, 16, 64, /*n_gpu_blocks=*/3, /*n_cpu_blocks=*/2,
                                /*priority_enabled=*/true);

    // A arrives first but is lower priority.
    EXPECT_TRUE(fixture.sched->queue_request(make_group(0, 15, /*priority=*/10)));
    EXPECT_EQ(step_and_update(fixture, false), 0);  // A prefill -> decoding

    // B arrives later with higher priority.
    EXPECT_TRUE(fixture.sched->queue_request(make_group(1, 15, /*priority=*/-10)));
    EXPECT_EQ(step_and_update(fixture, false), 1);  // B prefill; A's decode is dropped

    // One step with both decoding: both reach n_past == 16 (block boundary),
    // leaving exactly one free GPU block.
    {
        llama_batch batch = {};
        EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
        const auto * info = fixture.sched->get_curr_batch_info();
        EXPECT_EQ(info->n_seq, 2);
        std::vector<llama_token> tok(info->n_seq, 7);
        std::vector<int8_t>      stop(info->n_seq, 0);
        llama_paged_feedback     fbs[2] = {
            { batch.seq_id[info->batch_offsets[0]][0], &tok[0], 1, stop[0] },
            { batch.seq_id[info->batch_offsets[1]][0], &tok[1], 1, stop[1] },
        };
        fixture.sched->update(&batch, fbs, 2);
        llama_batch_free(batch);
    }

    // A takes the last block, B fails. A is the global lowest-priority victim,
    // so it must be dropped from candidates and swapped out; B then runs.
    llama_batch batch = {};
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    const auto * info = fixture.sched->get_curr_batch_info();
    EXPECT_EQ(info->n_seq, 1);
    EXPECT_EQ((int32_t) batch.seq_id[0][0], 1);  // B, not the already-scheduled A

    const auto * a = fixture.sched->get_group_from_id(0);
    const auto * b = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(a != nullptr && b != nullptr);
    EXPECT_TRUE(a->status == llama_sequence_group_status::SWAPPED);
    EXPECT_TRUE(!a->block_table.empty());
    EXPECT_FALSE(fixture.kv->is_block_gpu(a->block_table[0]));  // moved to CPU
    EXPECT_TRUE(b->status == llama_sequence_group_status::RUNNING);
    EXPECT_EQ(b->block_table.size(), 2u);
    EXPECT_TRUE(fixture.kv->is_block_gpu(b->block_table[1]));
}

TEST(test_scheduler_recompute_preserves_priority_and_arrival_seq) {
    // 2 GPU blocks and only 1 CPU block: B holds 2 blocks, so eviction cannot
    // swap it out and must fall back to recomputation.
    auto fixture = make_fixture(128, 16, 64, /*n_gpu_blocks=*/2, /*n_cpu_blocks=*/1,
                                /*priority_enabled=*/true);

    // A arrives first but has lower priority; B (higher priority) is admitted.
    EXPECT_TRUE(fixture.sched->queue_request(make_group(0, 20, /*priority=*/0)));
    EXPECT_TRUE(fixture.sched->queue_request(make_group(1, 20, /*priority=*/-5)));

    const auto * b_before = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(b_before != nullptr);
    const int64_t  t_before   = b_before->t_arrival_time;
    const uint64_t seq_before = b_before->arrival_seq;
    EXPECT_EQ(b_before->priority, -5);

    bool recomputed = false;
    for (int step = 0; step < 30 && !recomputed; ++step) {
        llama_batch batch = {};
        EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
        const auto * info = fixture.sched->get_curr_batch_info();
        EXPECT_TRUE(info->n_seq == 1);
        EXPECT_EQ((int32_t) batch.seq_id[0][0], 1);  // B stays the queue head

        auto * b = fixture.sched->get_group_from_id(1);
        EXPECT_TRUE(b != nullptr);
        if (b->n_past == 0 && b->n_decoded == 0 && step > 0) {
            // Recompute happened and B was immediately re-admitted as a fresh
            // prefill in this same step.
            EXPECT_EQ(b->priority, -5);
            EXPECT_EQ(b->t_arrival_time, t_before);
            EXPECT_EQ(b->arrival_seq, seq_before);
            EXPECT_EQ(b->n_prompt, 20u);
            EXPECT_EQ(b->logical_seq.size(), 20u);
            recomputed = true;
            break;
        }

        std::vector<llama_token> tok(info->n_seq, 7);
        std::vector<int8_t>      stop(info->n_seq, 0);
        std::vector<llama_paged_feedback> fbs;
        for (int i = 0; i < info->n_seq; ++i) {
            fbs.push_back({ batch.seq_id[info->batch_offsets[i]][0], &tok[i], 1, stop[i] });
        }
        fixture.sched->update(&batch, fbs.data(), (int32_t) fbs.size());
    }
    EXPECT_TRUE(recomputed);

    // A was starved by the strict head-of-queue rule and never ran.
    const auto * a = fixture.sched->get_group_from_id(0);
    EXPECT_TRUE(a != nullptr);
    EXPECT_TRUE(a->status == llama_sequence_group_status::WAITING);
    EXPECT_EQ(a->n_past, 0u);
}

TEST(test_scheduler_fcfs_prefill_uses_first_candidate) {
    // FCFS keeps the pre-existing rule: serve the first prefill candidate, not
    // the earliest-arrived one. B is already running a chunked prefill when A
    // (earlier arrival) is admitted; B is still first in candidates.
    auto fixture = make_fixture(128, 16, /*n_batch=*/16, /*n_gpu_blocks=*/8, /*n_cpu_blocks=*/2,
                                /*priority_enabled=*/false);

    // B: prompt longer than n_batch so it spans multiple steps.
    EXPECT_TRUE(fixture.sched->queue_request(make_group(1, 40)));
    EXPECT_EQ(step_and_update(fixture, false), 1);

    auto * b = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(b != nullptr);
    EXPECT_EQ(b->n_decoded, 0u);  // still prefilling

    // A arrives later in queue order but has an earlier timestamp.
    EXPECT_TRUE(fixture.sched->queue_request(make_group(0, 8)));

    llama_batch batch = {};
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    const auto * info = fixture.sched->get_curr_batch_info();
    EXPECT_EQ(info->n_seq, 1);
    EXPECT_EQ((int32_t) batch.seq_id[0][0], 1);  // B, the first candidate
    EXPECT_EQ(info->batch_lens[0], 16);          // B's next prefill chunk

    // A was admitted into running but is not the selected prefill.
    const auto * a = fixture.sched->get_group_from_id(0);
    EXPECT_TRUE(a != nullptr);
    EXPECT_TRUE(a->status == llama_sequence_group_status::RUNNING);
    EXPECT_EQ(a->n_past, 0u);
}

TEST(test_scheduler_priority_head_blocks_lower) {
    // Head needs 3 blocks, only 2 exist: strict head-of-queue means no progress
    // for anyone, including the low-priority request behind it.
    auto fixture = make_fixture(128, 16, 64, /*n_gpu_blocks=*/2, /*n_cpu_blocks=*/1,
                                /*priority_enabled=*/true);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(0, 32, /*priority=*/-10)));
    EXPECT_TRUE(fixture.sched->queue_request(make_group(1, 15, /*priority=*/0)));

    for (int i = 0; i < 3; ++i) {
        llama_batch            batch  = {};
        llama_scheduler_status status = fixture.sched->step(batch);
        EXPECT_TRUE(status == llama_scheduler_status::DEADLOCK);
    }

    const auto * head = fixture.sched->get_group_from_id(0);
    const auto * low  = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(head != nullptr && low != nullptr);
    EXPECT_TRUE(head->status == llama_sequence_group_status::WAITING);
    EXPECT_EQ(head->n_past, 0u);
    EXPECT_TRUE(low->status == llama_sequence_group_status::WAITING);
    EXPECT_EQ(low->n_past, 0u);
}

int main(int /*argc*/, char ** /*argv*/) {
    fprintf(stderr, "test-req-priority: queue abstraction\n");
    RUN(test_paged_queue_fcfs_order);
    RUN(test_paged_queue_fcfs_keyed_insert);
    RUN(test_paged_queue_fcfs_same_arrival_fifo);
    RUN(test_paged_queue_priority_order);
    RUN(test_paged_queue_priority_reinsert);
    RUN(test_paged_queue_empty_size);

    fprintf(stderr, "test-req-priority: scheduler ordering\n");
    RUN(test_scheduler_priority_default_fcfs);
    RUN(test_scheduler_fcfs_mode_ignores_priority);
    RUN(test_scheduler_arrival_seq_monotonic);
    RUN(test_scheduler_priority_higher_admitted_first);
    RUN(test_scheduler_priority_old_low_yields_to_new_high);
    RUN(test_scheduler_priority_equal_tie_break_fifo);
    RUN(test_scheduler_waiting_order_priority_then_arrival);
    RUN(test_scheduler_running_stays_arrival_ordered);
    RUN(test_scheduler_priority_ordering_strict_total_order);
    RUN(test_scheduler_priority_eviction_victim);
    RUN(test_scheduler_priority_evicts_scheduled_low_priority);

    fprintf(stderr, "test-req-priority: scheduler edge cases\n");
    RUN(test_scheduler_recompute_preserves_priority_and_arrival_seq);
    RUN(test_scheduler_fcfs_prefill_uses_first_candidate);
    RUN(test_scheduler_priority_head_blocks_lower);

    fprintf(stderr, "test-req-priority: ALL PASSED\n");
    return 0;
}
