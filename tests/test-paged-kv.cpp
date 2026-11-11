#include "ggml-backend.h"
#include "llama-block-manager.h"
#include "llama-kv-cache-paged.h"
#include "llama-full-prefix-index.h"
#include "llama-linear-prefix-index.h"
#include "llama-paged-scheduler-impl.h"
#include "llama-vocab.h"

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>

#define TEST(name) static void name()
#define RUN(name)                                   \
    do {                                            \
        fprintf(stderr, "  running %-40s ", #name); \
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

// Testing block_manager main functionality

TEST(test_block_manager_leak_simple) {
    const uint32_t n_gpu_blocks = 16;
    const uint32_t n_cpu_blocks = 8;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), n_cpu_blocks);

    auto gpu_ids = block_manager.checkout_gpu_blocks(10);
    EXPECT_EQ(gpu_ids.size(), 10u);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), (n_gpu_blocks - 10u));

    auto cpu_ids = block_manager.checkout_cpu_blocks(5);
    EXPECT_EQ(cpu_ids.size(), 5u);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), (n_cpu_blocks - 5u));

    block_manager.release_gpu_blocks(gpu_ids);
    block_manager.release_cpu_blocks(cpu_ids);

    EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), n_cpu_blocks);
}

// Testing that we always return to full (stress test)
TEST(test_block_manager_leak_repeated) {
    const uint32_t block_size   = 16;
    const uint32_t n_gpu_blocks = 64;
    const uint32_t n_cpu_blocks = 32;
    const float    watermark    = 0.0f;
    const int      n_iter       = 1000;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    for (int iter = 0; iter < n_iter; ++iter) {
        const uint32_t n   = (iter % block_size) + 1;
        auto           ids = block_manager.checkout_gpu_blocks(n);
        EXPECT_EQ(ids.size(), (size_t) n);
        block_manager.release_gpu_blocks(ids);
        EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);
    }
}

// Attempting to check-out more blocks than available. It should return empty.
TEST(test_block_manager_checkout_too_many) {
    const uint32_t n_gpu_blocks = 8;
    const uint32_t n_cpu_blocks = 4;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    auto ids = block_manager.checkout_gpu_blocks(9);
    EXPECT_EQ(ids.size(), 0u);
    EXPECT_EQ(block_manager.n_free_gpu_blocks(), n_gpu_blocks);

    auto ids2 = block_manager.checkout_cpu_blocks(5);
    EXPECT_EQ(ids2.size(), 0u);
    EXPECT_EQ(block_manager.n_free_cpu_blocks(), n_cpu_blocks);
}

TEST(test_block_manager_watermark) {
    // watermark=0.2 means 2 blocks reserved as safety (always consider max gpu blocks)
    const uint32_t n_gpu_blocks = 10;
    const uint32_t n_cpu_blocks = 10;
    const float    watermark    = 0.2f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    // 10 free, safety=2
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(8));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(9));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(10));

    auto ids = block_manager.checkout_gpu_blocks(5);
    EXPECT_EQ(ids.size(), 5u);
    // 5 free, safety=2
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(3));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(4));

    block_manager.release_gpu_blocks(ids);
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(8));
}

TEST(test_block_manager_watermark_zero) {
    // watermark=0 means the entire pool is requestable.
    const uint32_t n_gpu_blocks = 10;
    const uint32_t n_cpu_blocks = 10;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);
    EXPECT_TRUE(block_manager.has_free_gpu_blocks(10));
    EXPECT_FALSE(block_manager.has_free_gpu_blocks(11));
}

TEST(test_block_manager_gpu_cpu_disjoint) {
    // Just making sure CPU and GPU blocks are disjoint
    const uint32_t n_gpu_blocks = 8;
    const uint32_t n_cpu_blocks = 4;
    const float    watermark    = 0.0f;

    llama_block_manager block_manager;
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);

    auto gpu_ids = block_manager.checkout_gpu_blocks(8);
    auto cpu_ids = block_manager.checkout_cpu_blocks(4);

    for (auto id : gpu_ids) {
        EXPECT_TRUE(block_manager.is_gpu(id));
    }
    for (auto id : cpu_ids) {
        EXPECT_FALSE(block_manager.is_gpu(id));
    }

    block_manager.release_gpu_blocks(gpu_ids);
    block_manager.release_cpu_blocks(cpu_ids);
}

// Testing llama_kv_cache_paged book-keeping

// kv_cache_paged with arbitrary shape (we only care about the bookkeeping)
static ggml_backend_t make_kv_backend() {
    static ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    return backend;
}

static llama_kv_cache_paged make_kv() {
    return llama_kv_cache_paged(
        /*head_dim=*/64,
        /*n_heads_kv=*/4,
        /*block_size=*/16,
        /*n_layers=*/2,
        /*n_ubatch=*/32,
        /*n_seq_max=*/8);
}

TEST(test_seq_pos_default_unknown) {
    auto kv = make_kv();
    EXPECT_EQ(kv.seq_pos_min(0), -1);
    EXPECT_EQ(kv.seq_pos_max(0), -1);
    EXPECT_EQ(kv.seq_pos_min(42), -1);
}

TEST(test_seq_pos_set_and_get) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_max_pos(0, 17);
    EXPECT_EQ(kv.seq_pos_min(0), 5);
    EXPECT_EQ(kv.seq_pos_max(0), 17);

    // Independent per seq_id.
    kv.set_seq_min_pos(1, 100);
    EXPECT_EQ(kv.seq_pos_min(1), 100);
    EXPECT_EQ(kv.seq_pos_min(0), 5);  // unchanged
}

TEST(test_seq_pos_overwrite) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_min_pos(0, 9);
    EXPECT_EQ(kv.seq_pos_min(0), 9);
}

TEST(test_seq_pos_seq_rm_removes) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_max_pos(0, 17);
    EXPECT_EQ(kv.seq_pos_min(0), 5);

    kv.seq_rm(0, 0, 0);
    EXPECT_EQ(kv.seq_pos_min(0), -1);
    EXPECT_EQ(kv.seq_pos_max(0), -1);
}

TEST(test_seq_pos_clear_removes_all) {
    auto kv = make_kv();
    kv.set_seq_min_pos(0, 5);
    kv.set_seq_min_pos(1, 7);
    kv.set_seq_min_pos(2, 9);

    kv.clear(/*data=*/false);
    EXPECT_EQ(kv.seq_pos_min(0), -1);
    EXPECT_EQ(kv.seq_pos_min(1), -1);
    EXPECT_EQ(kv.seq_pos_min(2), -1);
}

TEST(test_free_blocks_releases_to_pool) {
    // Initialize a KV cache with a real CPU backend (used as both "GPU" and CPU).
    // This is fine for testing
    ggml_backend_t backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(backend != nullptr);

    const uint32_t n_gpu_blocks = 16;
    const uint32_t n_cpu_blocks = 8;
    const float    watermark    = 0.0f;

    auto kv = make_kv();
    kv.init(/*backend_gpu=*/backend,
            /*backend_cpu=*/backend, GGML_TYPE_F16, n_gpu_blocks, n_cpu_blocks, watermark);

    // allocate() pulls from the GPU block pool. After releasing, the count
    // must return to the initial value.
    const uint32_t n_gpu_initial = kv.get_num_gpu_blocks();
    EXPECT_EQ(n_gpu_initial, n_gpu_blocks);

    llama_sequence_group group;
    group.request_id = 0;
    group.n_prompt   = 32;  // 2 blocks
    group.n_decoded  = 0;

    bool success = kv.allocate(/*num_tokens=*/0, group);
    EXPECT_TRUE(success);
    EXPECT_EQ(group.block_table.size(), 2u);

    // Allocating a second sequence further reduces the pool.
    llama_sequence_group group2;
    group2.request_id = 1;
    group2.n_prompt   = 48;  // 3 blocks
    group2.n_decoded  = 0;

    success = kv.allocate(0, group2);
    EXPECT_TRUE(success);
    EXPECT_EQ(group2.block_table.size(), 3u);

    // Free the first sequence: we release 2 blocks
    kv.free_blocks(group);
    EXPECT_EQ(group.block_table.size(), 0u);

    llama_sequence_group group3;
    group3.request_id = 2;
    group3.n_prompt   = 32;  // 2 blocks
    group3.n_decoded  = 0;
    success           = kv.allocate(0, group3);
    EXPECT_TRUE(success);
    EXPECT_EQ(group3.block_table.size(), 2u);

    // All blocks released
    kv.free_blocks(group2);
    kv.free_blocks(group3);

    llama_sequence_group group_full;
    group_full.request_id = 99;
    group_full.n_prompt   = n_gpu_blocks * 16;  // exactly n_gpu_blocks worth
    group_full.n_decoded  = 0;
    success               = kv.allocate(0, group_full);
    EXPECT_TRUE(success);
    EXPECT_EQ(group_full.block_table.size(), 16u);
    kv.free_blocks(group_full);

    ggml_backend_free(backend);
}

// Testing scheduler

// For easy testing and clean-up.
// KV cache paged + scheduler (must free CPU backend)
struct paged_test_fixture {
    ggml_backend_t                              backend = nullptr;
    std::unique_ptr<llama_kv_cache_paged>       kv;
    std::unique_ptr<llama_paged_scheduler_impl> sched;

    ~paged_test_fixture() {
        if (backend) {
            ggml_backend_free(backend);
        }
    }

    // rule of 5
    paged_test_fixture()                                       = default;
    paged_test_fixture(paged_test_fixture &&)                  = default;
    paged_test_fixture & operator=(paged_test_fixture &&)      = default;
    paged_test_fixture(const paged_test_fixture &)             = delete;
    paged_test_fixture & operator=(const paged_test_fixture &) = delete;
};

static paged_test_fixture make_fixture(uint32_t n_ctx        = 128,
                                       uint32_t block_size   = 16,
                                       uint32_t n_batch      = 64,
                                       uint32_t n_gpu_blocks = 4,
                                       uint32_t n_cpu_blocks = 2) {
    paged_test_fixture fixture;
    fixture.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    EXPECT_TRUE(fixture.backend != nullptr);

    fixture.kv = std::unique_ptr<llama_kv_cache_paged>(new llama_kv_cache_paged(
        /*head_dim=*/64u,
        /*n_heads_kv=*/4u,
        /*block_size=*/block_size,
        /*n_layers=*/2u,
        /*n_ubatch=*/n_batch,
        /*n_seq_max=*/8u));
    fixture.kv->init(fixture.backend, fixture.backend, GGML_TYPE_F16, n_gpu_blocks, n_cpu_blocks, /*watermark=*/0.0f);

    fixture.sched = std::unique_ptr<llama_paged_scheduler_impl>(
        new llama_paged_scheduler_impl(n_ctx, block_size, n_batch, /*n_pos_per_embd=*/1, fixture.kv.get()));
    return fixture;
}

static llama_sequence_group make_group(int32_t request_id, uint32_t n_prompt) {
    llama_sequence_group group;
    group.request_id     = request_id;
    group.n_prompt       = n_prompt;
    group.n_decoded      = 0;
    group.n_past         = 0;
    group.t_arrival_time = request_id;  // control ordering based on request_id
    group.logical_seq.assign(n_prompt, /*dummy token=*/1);
    return group;
}

TEST(test_scheduler_no_deadlock_on_empty) {
    // No requests queued means no deadlock
    auto                   fixture = make_fixture();
    llama_batch            batch   = {};
    llama_scheduler_status status  = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::OK);
    EXPECT_EQ(batch.n_tokens, 0);
}

TEST(test_scheduler_deadlock_oversize_waiting_request) {
    // There are 2 blocks, the waiting request needs 3.
    // Attempt to process prefill for this requets will fail and the request will remain in waiting (deadlock).
    auto fixture = make_fixture(128, 16, 64, /*n_gpu_blocks=*/2, /*n_cpu_blocks=*/1);
    auto group   = make_group(0, /*n_prompts=*/32);

    bool queued = fixture.sched->queue_request(group);
    EXPECT_TRUE(queued);

    llama_batch            batch  = {};
    llama_scheduler_status status = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::DEADLOCK);

    // Next steps will continue remained deadlocked
    status = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::DEADLOCK);
}

TEST(test_scheduler_rejects_oversized_prompt) {
    auto fixture = make_fixture(/*n_ctx=*/64, /*block_size=*/16, /*n_batch=*/128,
                                /*n_gpu_blocks=*/32, /*n_cpu_blocks=*/8);

    bool queued = fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/64));
    EXPECT_FALSE(queued);

    queued = fixture.sched->queue_request(make_group(/*id=*/1, /*n_prompt=*/128));
    EXPECT_FALSE(queued);

    // A request just under the limit is accepted.
    queued = fixture.sched->queue_request(make_group(/*id=*/2, /*n_prompt=*/63));
    EXPECT_TRUE(queued);
}

TEST(test_scheduler_single_request_decode) {
    auto fixture = make_fixture(/*n_ctx=*/128, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/8, /*n_cpu_blocks=*/2);
    fixture.sched->set_prefix_cache_enabled(false);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/32)));

    llama_batch            batch  = {};
    llama_scheduler_status status = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::OK);
    EXPECT_EQ(batch.n_tokens, 32);

    const auto * info = fixture.sched->get_curr_batch_info();
    EXPECT_TRUE(info != nullptr);
    EXPECT_EQ(info->n_seq, 1);
    EXPECT_EQ(info->batch_lens[0], 32);
    EXPECT_EQ(info->context_lens[0], 32);

    std::vector<llama_token> sampled(1, 42);
    std::vector<int8_t>      stop(1, 0);
    llama_paged_feedback     fb = { /*request_id=*/0, sampled.data(), (int32_t) sampled.size(), stop[0] };
    fixture.sched->update(&batch, &fb, 1);

    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_EQ(batch.n_tokens, 1);

    info = fixture.sched->get_curr_batch_info();
    EXPECT_EQ(info->n_seq, 1);
    EXPECT_EQ(info->batch_lens[0], 1);
    EXPECT_EQ(info->context_lens[0], 33);
}

TEST(test_scheduler_multi_request_decode) {
    auto fixture = make_fixture(/*n_ctx=*/128, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/8, /*n_cpu_blocks=*/2);
    fixture.sched->set_prefix_cache_enabled(false);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/0, /*n_prompt=*/32)));
    EXPECT_TRUE(fixture.sched->queue_request(make_group(/*id=*/1, /*n_prompt=*/32)));

    llama_batch batch = {};
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_EQ(batch.n_tokens, 32);

    std::vector<llama_token> sampled0(1, 42);
    std::vector<int8_t>      stop0(1, 0);
    llama_paged_feedback     fb0 = { /*request_id=*/0, sampled0.data(), (int32_t) sampled0.size(), stop0[0] };
    fixture.sched->update(&batch, &fb0, 1);

    // The scheduler drains prefills one request at a time before decoding.
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_EQ(batch.n_tokens, 32);

    std::vector<llama_token> sampled1(1, 43);
    std::vector<int8_t>      stop1(1, 0);
    llama_paged_feedback     fb1 = { /*request_id=*/1, sampled1.data(), (int32_t) sampled1.size(), stop1[0] };
    fixture.sched->update(&batch, &fb1, 1);

    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_EQ(batch.n_tokens, 2);

    const auto * info = fixture.sched->get_curr_batch_info();
    EXPECT_TRUE(info != nullptr);
    EXPECT_EQ(info->n_seq, 2);
    EXPECT_EQ(info->batch_lens[0], 1);
    EXPECT_EQ(info->batch_lens[1], 1);
    EXPECT_EQ(info->context_lens[0], 33);
    EXPECT_EQ(info->context_lens[1], 33);
}

// Testing prefix cache

TEST(test_block_manager_generation) {
    llama_block_manager bm;
    bm.init(/*n_gpu=*/4, /*n_cpu=*/2, 0.0f);

    auto ids = bm.checkout_gpu_blocks(1);
    EXPECT_EQ(ids.size(), 1u);
    const uint32_t id = ids[0];
    const uint32_t g0 = bm.block_generation(id);

    // Allocated blocks keep their generation; releasing bumps it so stale
    // prefix entries can detect the reuse.
    EXPECT_EQ(bm.block_generation(id), g0);
    bm.release_gpu_blocks({ id });
    EXPECT_EQ(bm.block_generation(id), g0 + 1);

    auto ids2 = bm.checkout_gpu_blocks(1);
    EXPECT_EQ(ids2.size(), 1u);
    EXPECT_EQ(ids2[0], id);
    EXPECT_EQ(bm.block_generation(id), g0 + 1);
    bm.release_gpu_blocks(ids2);
}

TEST(test_prefix_block_stale_generation) {
    auto kv = make_kv();
    kv.init(make_kv_backend(), make_kv_backend(), GGML_TYPE_F16, /*n_gpu=*/4, /*n_cpu=*/2, 0.0f);

    llama_sequence_group group;
    group.request_id = 0;
    group.n_prompt   = 32;
    EXPECT_TRUE(kv.allocate(0, group));
    EXPECT_EQ(group.block_table.size(), 2u);

    const uint32_t id = group.block_table[0];
    kv.register_prefix_block(/*hash=*/1234, id, /*owner=*/0, /*idx=*/0, /*first=*/1, /*last=*/2);
    EXPECT_TRUE(kv.find_prefix_block(1234, 1, 2) != nullptr);

    // Once the block goes back to the pool the entry must not validate.
    kv.release_block(id);
    EXPECT_TRUE(kv.find_prefix_block(1234, 1, 2) == nullptr);

    kv.free_blocks(group);
}

TEST(test_block_manager_add_ref) {
    llama_block_manager bm;
    bm.init(/*n_gpu=*/4, /*n_cpu=*/2, 0.0f);

    auto ids = bm.checkout_gpu_blocks(2);
    EXPECT_EQ(ids.size(), 2u);

    EXPECT_TRUE(bm.add_ref_block(ids[0]));
    EXPECT_EQ(bm.block_ref_count(ids[0]), 2u);
    EXPECT_EQ(bm.n_free_gpu_blocks(), 2u);

    EXPECT_FALSE(bm.add_ref_block(999));
    bm.release_gpu_blocks({ids[1]});
    EXPECT_EQ(bm.block_ref_count(ids[1]), 0u);
    EXPECT_FALSE(bm.add_ref_block(ids[1]));

    bm.release_gpu_blocks({ids[0]});
    EXPECT_EQ(bm.block_ref_count(ids[0]), 1u);
    EXPECT_EQ(bm.n_free_gpu_blocks(), 3u);
    bm.release_gpu_blocks({ids[0]});
    EXPECT_EQ(bm.block_ref_count(ids[0]), 0u);
    EXPECT_EQ(bm.n_free_gpu_blocks(), 4u);
}

TEST(test_prefix_cache_register_find) {
    llama_full_prefix_index pc;
    pc.register_block(100, /*block_id=*/7, /*generation=*/3, /*owner=*/0, /*idx=*/0, /*first=*/1, /*last=*/2);

    const auto * e = pc.find(100, 1, 2);
    EXPECT_TRUE(e != nullptr);
    EXPECT_EQ(e->block_id, 7u);
    EXPECT_EQ(e->generation, 3u);
    EXPECT_TRUE(pc.find(100, 9, 9) == nullptr);

    EXPECT_TRUE(pc.update_block_id(100, 8, /*new_generation=*/4));
    EXPECT_EQ(pc.find(100, 1, 2)->block_id, 8u);
    EXPECT_EQ(pc.find(100, 1, 2)->generation, 4u);
}

TEST(test_prefix_cache_cached_lifecycle) {
    llama_full_prefix_index pc;
    pc.register_block(1, 10, 0, 0, 0, 1, 1);
    pc.register_block(2, 11, 0, 0, 1, 1, 1);
    pc.set_cached_state(1, true);
    pc.set_cached_state(2, true);
    pc.set_cached_state(2, false);

    uint64_t h    = 0;
    uint32_t id   = 0;
    uint32_t gen  = 0;
    EXPECT_TRUE(pc.pop_lru_cached(h, id, gen));
    EXPECT_EQ(h, 1ull);
    EXPECT_EQ(id, 10u);
    EXPECT_FALSE(pc.pop_lru_cached(h, id, gen));
}

TEST(test_prefix_cache_ckpt_slots) {
    llama_linear_prefix_index pc;
    pc.insert(/*hash=*/42, /*block=*/6, /*gen=*/0, /*pos=*/64);
    pc.insert(/*hash=*/43, /*block=*/7, /*gen=*/0, /*pos=*/128);

    EXPECT_TRUE(pc.find(42) != nullptr);
    EXPECT_EQ(pc.find(43)->pos, 128);

    const std::vector<uint64_t> chain = { 42, 43 };
    const auto * deepest = pc.find_deepest(chain, 2);
    EXPECT_TRUE(deepest != nullptr);
    EXPECT_EQ(deepest->chain_hash, 43ull);

    EXPECT_TRUE(pc.find_deepest(chain, 1) != nullptr);
    EXPECT_EQ(pc.find_deepest(chain, 1)->chain_hash, 42ull);

    uint64_t h = 0;
    uint32_t id = 0;
    uint32_t gen = 0;
    EXPECT_TRUE(pc.pop_lru(h, id, gen));
    EXPECT_EQ(h, 42ull);
    EXPECT_TRUE(pc.pop_lru(h, id, gen));
    EXPECT_EQ(h, 43ull);
    EXPECT_FALSE(pc.pop_lru(h, id, gen));
}

TEST(test_scheduler_prefix_cache_hit) {
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/8, /*n_cpu_blocks=*/2);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(0, /*n_prompt=*/32)));
    llama_batch            batch  = {};
    llama_scheduler_status status = fixture.sched->step(batch);
    EXPECT_TRUE(status == llama_scheduler_status::OK);
    EXPECT_EQ(batch.n_tokens, 32);

    const auto * info0 = fixture.sched->get_curr_batch_info();
    const uint32_t blk0 = (uint32_t) info0->block_table[0];
    const uint32_t blk1 = (uint32_t) info0->block_table[1];
    std::vector<llama_token> sampled(info0->n_seq, 7);
    std::vector<int8_t>      stop(info0->n_seq, 1);
    llama_paged_feedback fb = { /*request_id=*/0, sampled.data(), (int32_t) sampled.size(), stop[0] };
    fixture.sched->update(&batch, &fb, 1);

    // Finalizing the request keeps its registered blocks cache-resident.
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_EQ(fixture.kv->block_ref_count(blk0), 1u);
    EXPECT_EQ(fixture.kv->block_ref_count(blk1), 1u);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(1, /*n_prompt=*/48)));
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    auto * g1 = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(g1 != nullptr);
    EXPECT_EQ(g1->n_past, 32u);
    EXPECT_EQ(g1->n_prompt, 16u);
    EXPECT_EQ(g1->block_table[0], blk0);
    EXPECT_EQ(g1->block_table[1], blk1);
    // The cache's extra reference is transferred to the reusing request, so
    // the block is owned exactly once while the request is running.
    EXPECT_EQ(fixture.kv->block_ref_count(blk0), 1u);

    const llama_block_ids table1 = g1->block_table;
    std::vector<llama_token> sampled1(1, 8);
    llama_paged_feedback fb1 = { /*request_id=*/1, sampled1.data(), (int32_t) sampled1.size(), stop[0] };
    fixture.sched->update(&batch, &fb1, 1);
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_EQ(fixture.kv->block_ref_count(blk0), 1u);
    EXPECT_EQ(fixture.kv->block_ref_count(blk1), 1u);
    // Req1 registered its first fresh block before finishing; it stays
    // cache-resident while the remaining fresh blocks go back to the pool.
    EXPECT_EQ(fixture.kv->block_ref_count(table1[2]), 1u);
    EXPECT_EQ(fixture.kv->block_ref_count(table1[3]), 0u);
}

TEST(test_scheduler_prefix_cache_zero_suffix) {
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/8, /*n_cpu_blocks=*/2);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(0, /*n_prompt=*/32)));
    llama_batch batch = {};
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    std::vector<llama_token> sampled(1, 7);
    std::vector<int8_t>      stop(1, 1);
    llama_paged_feedback fb = { /*request_id=*/0, sampled.data(), (int32_t) sampled.size(), stop[0] };
    fixture.sched->update(&batch, &fb, 1);
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);

    // An exact prompt match cannot replay the last token, so one block
    // stays unshared to keep a non-empty suffix.
    EXPECT_TRUE(fixture.sched->queue_request(make_group(1, /*n_prompt=*/32)));
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    auto * g1 = fixture.sched->get_group_from_id(1);
    EXPECT_TRUE(g1 != nullptr);
    EXPECT_EQ(g1->n_past, 16u);
    EXPECT_EQ(g1->n_prompt, 16u);
}

TEST(test_scheduler_prefix_cache_eviction) {
    auto fixture = make_fixture(/*n_ctx=*/256, /*block_size=*/16, /*n_batch=*/64,
                                /*n_gpu_blocks=*/5, /*n_cpu_blocks=*/2);

    EXPECT_TRUE(fixture.sched->queue_request(make_group(0, /*n_prompt=*/32)));
    llama_batch batch = {};
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    std::vector<llama_token> sampled(1, 7);
    std::vector<int8_t>      stop(1, 1);
    llama_paged_feedback fb = { /*request_id=*/0, sampled.data(), (int32_t) sampled.size(), stop[0] };
    fixture.sched->update(&batch, &fb, 1);
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);

    // Only 3 blocks are free; the fresh request needs 5, so the cache
    // must release its resident blocks for admission to succeed.
    auto group1 = make_group(1, /*n_prompt=*/33);
    for (auto & t : group1.logical_seq) {
        t = 9;  // unrelated tokens: no prefix match
    }
    EXPECT_TRUE(fixture.sched->queue_request(group1));
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_EQ(batch.n_tokens, 33);
}

TEST(test_scheduler_draft_uses_step_budget) {
    auto fixture = make_fixture(/*n_ctx=*/128, /*block_size=*/16, /*n_batch=*/16,
                                /*n_gpu_blocks=*/16, /*n_cpu_blocks=*/2);
    fixture.sched->set_prefix_cache_enabled(false);
    EXPECT_TRUE(fixture.sched->queue_request(make_group(0, 32)));

    llama_batch batch = {};
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    fixture.sched->update(&batch, nullptr, 0);
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    const llama_token          sampled = 42;
    const llama_paged_feedback fb      = { 0, &sampled, 1, 0 };
    fixture.sched->update(&batch, &fb, 1);

    const llama_token draft[] = { 43, 44, 45 };
    EXPECT_FALSE(fixture.sched->attach_draft(1, draft, 3));
    EXPECT_FALSE(fixture.sched->attach_draft(0, nullptr, 3));
    EXPECT_FALSE(fixture.sched->attach_draft(0, draft, 0));
    EXPECT_TRUE(fixture.sched->attach_draft(0, draft, 3));
    EXPECT_FALSE(fixture.sched->attach_draft(0, draft, 3));
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_TRUE(batch.n_tokens == 4);
    EXPECT_TRUE(batch.pos[0] == 32);
    EXPECT_TRUE(batch.token[0] == sampled);
    EXPECT_TRUE(batch.token[3] == draft[2]);
}

TEST(test_scheduler_media_spans_batches) {
    auto fixture = make_fixture(/*n_ctx=*/128, /*block_size=*/16, /*n_batch=*/16,
                                /*n_gpu_blocks=*/16, /*n_cpu_blocks=*/2);
    fixture.sched->set_prefix_cache_enabled(false);
    std::vector<float> embeddings(48 * 4, 1.0f);
    auto               group = make_group(0, 53);
    group.has_media          = true;
    std::fill_n(group.logical_seq.begin() + 2, 48, LLAMA_TOKEN_NULL);
    llama_seq_segment prefix = {};
    prefix.n_tokens          = 2;
    llama_seq_segment media  = {};
    media.start_row          = 2;
    media.n_tokens           = 48;
    media.is_media           = true;
    media.embd               = embeddings.data();
    media.embd_stride        = 4;
    llama_seq_segment suffix = {};
    suffix.start_row         = 50;
    suffix.n_tokens          = 3;
    group.segments           = { prefix, media, suffix };
    EXPECT_TRUE(fixture.sched->queue_request(std::move(group)));

    llama_batch batch = {};
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_TRUE(batch.n_tokens == 2);
    fixture.sched->update(&batch, nullptr, 0);
    for (int chunk = 0; chunk < 3; ++chunk) {
        EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
        EXPECT_TRUE(batch.n_tokens == 16);
        EXPECT_TRUE(batch.token == nullptr);
        EXPECT_TRUE(batch.embd == embeddings.data() + chunk * 16 * 4);
        fixture.sched->update(&batch, nullptr, 0);
    }
}

TEST(test_scheduler_swapped_media_waits_for_batch_budget) {
    auto fixture = make_fixture(/*n_ctx=*/128, /*block_size=*/16, /*n_batch=*/16,
                                /*n_gpu_blocks=*/6, /*n_cpu_blocks=*/4);
    fixture.sched->set_prefix_cache_enabled(false);
    const llama_token sampled = 42;
    llama_batch       batch   = {};
    for (int32_t id = 0; id < 2; ++id) {
        EXPECT_TRUE(fixture.sched->queue_request(make_group(id, 1)));
        EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
        EXPECT_TRUE(batch.n_tokens == 1);
        EXPECT_TRUE(batch.seq_id[0][0] == id);
        const llama_paged_feedback fb = { id, &sampled, 1, 0 };
        fixture.sched->update(&batch, &fb, 1);
    }

    std::vector<float> embeddings(48 * 4, 1.0f);
    auto               group = make_group(2, 53);
    group.has_media          = true;
    std::fill_n(group.logical_seq.begin() + 2, 48, LLAMA_TOKEN_NULL);
    llama_seq_segment prefix = {};
    prefix.n_tokens          = 2;
    llama_seq_segment media  = {};
    media.start_row          = 2;
    media.n_tokens           = 48;
    media.is_media           = true;
    media.embd               = embeddings.data();
    media.embd_stride        = 4;
    llama_seq_segment suffix = {};
    suffix.start_row         = 50;
    suffix.n_tokens          = 3;
    group.segments           = { prefix, media, suffix };
    EXPECT_TRUE(fixture.sched->queue_request(std::move(group)));
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_TRUE(batch.n_tokens == 2);
    EXPECT_TRUE(batch.seq_id[0][0] == 2);
    fixture.sched->update(&batch, nullptr, 0);

    // Expanding request 0 evicts the newer media request to the CPU pool.
    std::vector<llama_token> draft(15, 43);
    EXPECT_TRUE(fixture.sched->attach_draft(0, draft.data(), (int32_t) draft.size()));
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_TRUE(batch.n_tokens == 16);
    EXPECT_TRUE(batch.seq_id[0][0] == 0);
    auto * restored = fixture.sched->get_group_from_id(2);
    EXPECT_TRUE(restored != nullptr);
    EXPECT_TRUE(restored->status == llama_sequence_group_status::SWAPPED);
    EXPECT_TRUE(restored->n_past == 2);
    EXPECT_TRUE(restored->block_table.size() == 4);
    for (const auto block : restored->block_table) {
        EXPECT_TRUE(block >= 6);
    }
    EXPECT_TRUE(fixture.kv->n_free_cpu_blocks() == 0);
    std::vector<llama_token> accepted(16, 43);
    accepted.back() = 44;
    const llama_paged_feedback verified = { 0, accepted.data(), (int32_t) accepted.size(), 0 };
    fixture.sched->update(&batch, &verified, 1);

    // Free enough GPU blocks so only the remaining 15-row batch budget blocks swap-in.
    EXPECT_TRUE(fixture.sched->cancel(1));
    EXPECT_TRUE(fixture.kv->n_free_gpu_blocks() == 4);
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_TRUE(batch.n_tokens == 1);
    EXPECT_TRUE(batch.token != nullptr);
    EXPECT_TRUE(batch.seq_id[0][0] == 0);
    EXPECT_TRUE(restored->status == llama_sequence_group_status::SWAPPED);
    EXPECT_TRUE(restored->n_past == 2);
    EXPECT_TRUE(fixture.kv->n_free_cpu_blocks() == 0);
    const llama_paged_feedback stopped = { 0, &sampled, 1, 1 };
    fixture.sched->update(&batch, &stopped, 1);

    for (int chunk = 0; chunk < 3; ++chunk) {
        EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
        EXPECT_TRUE(restored->status == llama_sequence_group_status::RUNNING);
        EXPECT_TRUE(batch.n_tokens == 16);
        EXPECT_TRUE(batch.token == nullptr);
        EXPECT_TRUE(batch.seq_id[0][0] == 2);
        EXPECT_TRUE(batch.embd == embeddings.data() + chunk * 16 * 4);
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            EXPECT_TRUE(batch.pos[i] == 2 + chunk * 16 + i);
        }
        for (const auto block : restored->block_table) {
            EXPECT_TRUE(block < 6);
        }
        EXPECT_TRUE(fixture.kv->n_free_cpu_blocks() == 4);
        fixture.sched->update(&batch, nullptr, 0);
        EXPECT_TRUE(restored->n_past == (uint32_t) (2 + (chunk + 1) * 16));
    }
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_TRUE(batch.n_tokens == 3);
    EXPECT_TRUE(batch.token != nullptr);
    EXPECT_TRUE(batch.seq_id[0][0] == 2);
    EXPECT_TRUE(batch.pos[0] == 50);
    const llama_paged_feedback finished = { 2, &sampled, 1, 1 };
    fixture.sched->update(&batch, &finished, 1);
    EXPECT_TRUE(fixture.sched->step(batch) == llama_scheduler_status::OK);
    EXPECT_TRUE(batch.n_tokens == 0);
    EXPECT_TRUE(fixture.kv->n_free_gpu_blocks() == 6);
    EXPECT_TRUE(fixture.kv->n_free_cpu_blocks() == 4);
}

static void check_paged_metadata_split(const std::vector<int32_t> & lengths,
                                       uint32_t                     n_ubatch,
                                       uint32_t                     n_pos      = 1,
                                       bool                         init_batch = false) {
    int32_t              n_tokens = 0;
    std::vector<int32_t> offsets;
    std::vector<int32_t> context_lens;
    std::vector<int32_t> block_table;
    for (size_t s = 0; s < lengths.size(); ++s) {
        offsets.push_back(n_tokens);
        n_tokens += lengths[s];
        context_lens.push_back(100 * (int32_t) (s + 1) + lengths[s]);
        for (int b = 0; b < 3; ++b) {
            block_table.push_back(10 * (int32_t) (s + 1) + b);
        }
    }
    std::vector<float>          embeddings(n_tokens, 1.0f);
    std::vector<llama_pos>      positions(n_tokens * n_pos, 0);
    std::vector<int32_t>        n_seq_id(n_tokens, 1);
    std::vector<llama_seq_id>   seq_ids(n_tokens);
    std::vector<llama_seq_id *> seq_ptrs(n_tokens);
    std::vector<int8_t>         logits(n_tokens, 1);
    std::vector<int32_t>        write_slots(n_tokens);
    for (size_t s = 0; s < lengths.size(); ++s) {
        for (int32_t t = 0; t < lengths[s]; ++t) {
            const int32_t row = offsets[s] + t;
            seq_ids[row]      = 7 - 2 * (int32_t) s;
            seq_ptrs[row]     = &seq_ids[row];
            positions[row]    = n_pos == 1 ? 100 * (int32_t) (s + 1) + t : 4;
            for (uint32_t p = 1; p < n_pos; ++p) {
                positions[p * n_tokens + row] = t;
            }
            write_slots[row] = 1000 + row;
        }
    }
    llama_batch batch               = {};
    batch.n_tokens                  = n_tokens;
    batch.embd                      = embeddings.data();
    batch.pos                       = positions.data();
    batch.n_seq_id                  = n_seq_id.data();
    batch.seq_id                    = seq_ptrs.data();
    batch.logits                    = logits.data();
    llama_paged_batch_info info     = {};
    info.n_tokens                   = n_tokens;
    info.n_seq                      = (int32_t) lengths.size();
    info.n_blocks_per_seq           = 3;
    info.write_slots                = write_slots.data();
    info.block_table                = block_table.data();
    info.context_lens               = context_lens.data();
    info.batch_offsets              = offsets.data();
    std::vector<int32_t> batch_lens = lengths;
    info.batch_lens                 = batch_lens.data();

    llama_vocab        vocab;
    llama_batch_allocr allocr(n_pos);
    EXPECT_TRUE(allocr.init(batch, vocab, nullptr, 1, 8, false));
    auto fixture = make_fixture();
    fixture.kv->set_paged_batch_info(&info);
    llama_memory_context_ptr context;
    if (init_batch) {
        context = fixture.kv->init_batch(allocr, n_ubatch, false);
    } else {
        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = allocr.split_equal(n_ubatch, false, 0);
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch));
        }
        context = fixture.kv->create_batch_context(ubatches);
    }
    EXPECT_TRUE(context->get_status() == LLAMA_MEMORY_STATUS_SUCCESS);
    auto *                paged = static_cast<llama_kv_cache_paged_context *>(context.get());
    std::vector<uint32_t> consumed(lengths.size(), 0);
    do {
        const auto & ubatch = context->get_ubatch();
        EXPECT_TRUE(paged->get_n_tokens() == (int32_t) ubatch.n_tokens);
        EXPECT_TRUE(paged->get_batch_size() == (int32_t) ubatch.n_seqs);
        for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
            const uint32_t local = s * ubatch.n_seq_tokens;
            const uint32_t src   = (7 - ubatch.seq_id[local][0]) / 2;
            EXPECT_TRUE(src < lengths.size());
            EXPECT_TRUE(paged->get_batch_offsets()[s] == (int32_t) local);
            EXPECT_TRUE(paged->get_batch_lens()[s] == (int32_t) ubatch.n_seq_tokens);
            EXPECT_TRUE(paged->get_context_lens()[s] ==
                        100 * (int32_t) (src + 1) + (int32_t) (consumed[src] + ubatch.n_seq_tokens));
            for (int b = 0; b < 3; ++b) {
                EXPECT_TRUE(paged->get_block_table()[s * 3 + b] == block_table[src * 3 + b]);
            }
            for (uint32_t t = 0; t < ubatch.n_seq_tokens; ++t) {
                const uint32_t original = offsets[src] + consumed[src] + t;
                EXPECT_TRUE(ubatch.data->batch_idxs[local + t] == (int32_t) original);
                EXPECT_TRUE(paged->get_write_slots()[local + t] == write_slots[original]);
            }
            consumed[src] += ubatch.n_seq_tokens;
        }
    } while (context->next());
    for (size_t s = 0; s < lengths.size(); ++s) {
        EXPECT_TRUE(consumed[s] == (uint32_t) lengths[s]);
    }
}

TEST(test_paged_metadata_first_ubatch) {
    check_paged_metadata_split({ 10 }, 4, 1, true);
}

int main(int /*argc*/, char ** /*argv*/) {
    RUN(test_paged_metadata_first_ubatch);
    fprintf(stderr, "test-paged-kv: block_manager\n");
    RUN(test_block_manager_leak_simple);
    RUN(test_block_manager_leak_repeated);
    RUN(test_block_manager_checkout_too_many);
    RUN(test_block_manager_watermark);
    RUN(test_block_manager_watermark_zero);
    RUN(test_block_manager_gpu_cpu_disjoint);

    fprintf(stderr, "test-paged-kv: llama_kv_cache_paged seq_pos\n");
    RUN(test_seq_pos_default_unknown);
    RUN(test_seq_pos_set_and_get);
    RUN(test_seq_pos_overwrite);
    RUN(test_seq_pos_seq_rm_removes);
    RUN(test_seq_pos_clear_removes_all);

    fprintf(stderr, "test-paged-kv: llama_kv_cache_paged free_blocks\n");
    RUN(test_free_blocks_releases_to_pool);

    fprintf(stderr, "test-paged-kv: llama_kv_cache_paged scheduler\n");
    RUN(test_scheduler_no_deadlock_on_empty);
    RUN(test_scheduler_deadlock_oversize_waiting_request);
    RUN(test_scheduler_rejects_oversized_prompt);
    RUN(test_scheduler_single_request_decode);
    RUN(test_scheduler_multi_request_decode);

    fprintf(stderr, "test-paged-kv: prefix cache\n");
    RUN(test_block_manager_generation);
    RUN(test_prefix_block_stale_generation);
    RUN(test_block_manager_add_ref);
    RUN(test_prefix_cache_register_find);
    RUN(test_prefix_cache_cached_lifecycle);
    RUN(test_prefix_cache_ckpt_slots);
    RUN(test_scheduler_prefix_cache_hit);
    RUN(test_scheduler_prefix_cache_zero_suffix);
    RUN(test_scheduler_prefix_cache_eviction);
    RUN(test_scheduler_draft_uses_step_budget);
    RUN(test_scheduler_media_spans_batches);
    RUN(test_scheduler_swapped_media_waits_for_batch_budget);

    fprintf(stderr, "test-paged-kv: ALL PASSED\n");
    return 0;
}
