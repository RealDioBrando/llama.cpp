// Unit and integration tests for the paged prefix-cache eviction policy
// callback.
//
// The block-manager tests need no model / context / scheduler. The setter
// lifecycle tests (install / swap / NULL reset / clear() reinstall) use a real
// llama_kv_cache_paged + llama_paged_scheduler_impl on the CPU backend, the
// same fixture pattern as tests/test-paged-kv.cpp.
//
// See docs/kvcache-manager/prefix-cache-evict-policy.md:
//   - section 5.2: batch selection / validation / atomic failure contract of
//     llama_block_manager::checkout_gpu_blocks();
//   - section 7.1: setter contract, including reinstall after clear();
//   - section 10: test plan (legacy equivalence, return 0, short batch,
//     selective victims, over-sacrifice, pool kind, plain-only fill, setter /
//     reset semantics);
//   - section 7.2: example policy callback style.
//
// The free-list node accessors (llama_paged_free_block_*) are only valid while
// a callback is running, so the policies below use them from inside callbacks.

#include "ggml-backend.h"
#include "llama-block-manager.h"
#include "llama-kv-cache-paged.h"
#include "llama-paged-scheduler-impl.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

static int g_failures = 0;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);  \
            g_failures++;                                                    \
        }                                                                    \
    } while (0)

#define CHECK_EQ(a, b)                                                                          \
    do {                                                                                        \
        const long long _a = (long long) (a);                                                   \
        const long long _b = (long long) (b);                                                   \
        if (_a != _b) {                                                                         \
            fprintf(stderr, "FAIL %s:%d: expected %s == %s, got %lld vs %lld\n", __FILE__,       \
                    __LINE__, #a, #b, _a, _b);                                                  \
            g_failures++;                                                                       \
        }                                                                                       \
    } while (0)

static bool ids_contain(const std::vector<uint32_t> & ids, uint32_t id) {
    for (const uint32_t v : ids) {
        if (v == id) {
            return true;
        }
    }
    return false;
}

static void check_ids(const char * what, const std::vector<uint32_t> & got,
                      const std::vector<uint32_t> & want) {
    bool ok = got.size() == want.size();
    if (ok) {
        for (size_t i = 0; i < got.size(); ++i) {
            if (got[i] != want[i]) {
                ok = false;
                break;
            }
        }
    }
    if (ok) {
        return;
    }
    fprintf(stderr, "FAIL %s: got [", what);
    for (const uint32_t v : got) {
        fprintf(stderr, "%u ", v);
    }
    fprintf(stderr, "] want [");
    for (const uint32_t v : want) {
        fprintf(stderr, "%u ", v);
    }
    fprintf(stderr, "]\n");
    g_failures++;
}

// ---------------------------------------------------------------------------
// Policy callbacks
// ---------------------------------------------------------------------------

// Shared recording state for the "fixed answer" picker.
struct cb_state {
    int                   calls         = 0;
    llama_paged_pool_kind last_pool     = LLAMA_PAGED_POOL_FULL;
    uint32_t              last_n_needed = 0;
    uint32_t              last_capacity = 0;
    std::vector<uint32_t> ids;
};
static cb_state g_cb;

static void reset_cb(const std::vector<uint32_t> & ids = {}) {
    g_cb = cb_state();
    g_cb.ids = ids;
}

// Returns exactly the ids configured through reset_cb() (up to out_capacity).
// An empty list therefore means "no block may be sacrificed".
static uint32_t pick_fixed_cb(void * ud, llama_paged_pool_kind pool, uint32_t n_needed,
                              const llama_paged_free_block * head, uint32_t * out_ids,
                              uint32_t out_capacity) {
    (void) ud;
    (void) head;
    g_cb.calls++;
    g_cb.last_pool     = pool;
    g_cb.last_n_needed = n_needed;
    g_cb.last_capacity = out_capacity;

    uint32_t n = 0;
    for (const uint32_t id : g_cb.ids) {
        if (n >= out_capacity) {
            break;
        }
        out_ids[n++] = id;
    }
    return n;
}

// Skips cache-resident nodes tagged `ud` (e.g. "keep") and returns the first
// n_needed of the rest, walking the real free list through the C accessors.
// Selected hashes are recorded to exercise llama_paged_free_block_hash().
static std::vector<uint64_t> g_picked_hashes;

static uint32_t pick_skip_tag_cb(void * ud, llama_paged_pool_kind pool, uint32_t n_needed,
                                 const llama_paged_free_block * head, uint32_t * out_ids,
                                 uint32_t out_capacity) {
    (void) pool;
    const char * skip = (const char *) ud;

    uint32_t n = 0;
    for (const llama_paged_free_block * it = head;
         it != nullptr && n < n_needed && n < out_capacity; it = llama_paged_free_block_next(it)) {
        if (!llama_paged_free_block_has_hash(it)) {
            continue;
        }
        // Membership lookup (the legacy tag() accessor returns only one
        // representative tag, so it must not be used for "skip tag X").
        if (skip != nullptr && llama_paged_free_block_has_tag(it, skip)) {
            continue;
        }
        out_ids[n++] = llama_paged_free_block_id(it);
        g_picked_hashes.push_back(llama_paged_free_block_hash(it));
    }
    return n;
}

// Calls the default picker with a deliberately small out_capacity, records the
// result and refuses the sacrifice so the block manager state stays untouched.
static uint32_t g_probe_ret      = 0;
static uint32_t g_probe_ids[2]   = { 0, 0 };

static uint32_t pick_probe_capacity_cb(void * ud, llama_paged_pool_kind pool, uint32_t n_needed,
                                       const llama_paged_free_block * head, uint32_t * out_ids,
                                       uint32_t out_capacity) {
    (void) ud;
    (void) n_needed;
    (void) out_ids;
    (void) out_capacity;

    uint32_t probe[2] = { 0, 0 };
    g_probe_ret       = llama_paged_evict_pick_default(nullptr, pool, 5, head, probe, 2);
    g_probe_ids[0]    = probe[0];
    g_probe_ids[1]    = probe[1];
    return 0;
}

// ---------------------------------------------------------------------------
// Multi-tag ownership: one block carries every tag that used it while cached
// ---------------------------------------------------------------------------

static void test_multi_tag_set() {
    llama_block_manager bm;
    bm.init(4, 0, 0.0f);
    bm.set_block_hash(0, 0xB000);

    // Write the lexicographically LATER tag first, so the representative tag
    // differs from the tag written first to the block.
    bm.set_tag(0, "multi-tag-written-first");
    bm.add_tag(0, "multi-tag-interning-early");

    CHECK_EQ(bm.tag_count(0), 2u);
    CHECK(bm.has_tag(0, "multi-tag-interning-early"));
    CHECK(bm.has_tag(0, "multi-tag-written-first"));
    CHECK(!bm.has_tag(0, "multi-tag-absent"));
    // Representative = lexicographically first, NOT the tag written first.
    CHECK(bm.tag(0) == "multi-tag-interning-early");

    // tag_at() enumerates in lexicographic order.
    CHECK(std::strcmp(bm.tag_at(0, 0), "multi-tag-interning-early") == 0);
    CHECK(std::strcmp(bm.tag_at(0, 1), "multi-tag-written-first") == 0);
    CHECK(bm.tag_at(0, 2) == nullptr);
}

static void test_tags_survive_cache_resident_release() {
    llama_block_manager bm;
    bm.init(4, 0, 0.0f);
    bm.set_block_hash(0, 0xB100);
    bm.set_tag(0, "owner");

    CHECK(bm.touch_block(0));
    bm.release_gpu_blocks({ 0 });          // ref -> 0, hash kept

    CHECK(bm.block_has_hash(0));
    CHECK_EQ(bm.block_ref_count(0), 0u);
    CHECK_EQ(bm.tag_count(0), 1u);
    CHECK(bm.has_tag(0, "owner"));
}

static void test_tags_cleared_on_eviction() {
    llama_block_manager bm;
    bm.init(4, 0, 0.0f);
    bm.set_block_hash(0, 0xB200);
    bm.set_tag(0, "gone");

    const auto ids = bm.checkout_gpu_blocks(1);   // legacy path evicts block 0
    CHECK_EQ(ids.size(), 1u);
    CHECK_EQ(ids[0], 0u);
    CHECK(!bm.block_has_hash(0));
    CHECK_EQ(bm.tag_count(0), 0u);
    CHECK(bm.tag(0).empty());
}

static void test_tags_cleared_on_clear_hash() {
    llama_block_manager bm;
    bm.init(4, 0, 0.0f);
    bm.set_block_hash(0, 0xB300);
    bm.set_tag(0, "gone");

    bm.clear_block_hash(0);
    CHECK_EQ(bm.tag_count(0), 0u);
}

// A block tagged before it ever gets a hash must lose its tag when it is
// released to the plain (non-cache-resident) free pool, so the next owner does
// not inherit it.
static void test_tags_cleared_on_release_to_plain() {
    llama_block_manager bm;
    bm.init(2, 0, 0.0f);

    const auto ids = bm.checkout_gpu_blocks(1);   // ref = 1, no hash yet
    CHECK_EQ(ids.size(), 1u);
    if (ids.empty()) {
        return;
    }
    const uint32_t id = ids[0];
    bm.set_tag(id, "plain-release-tag");
    CHECK_EQ(bm.tag_count(id), 1u);

    bm.release_gpu_blocks({ id });                // ref -> 0, !has_hash
    CHECK_EQ(bm.block_ref_count(id), 0u);
    CHECK_EQ(bm.tag_count(id), 0u);
    CHECK(bm.tag(id).empty());

    // A later checkout starts with an empty tag set (the freed block went back
    // to the queue tail, so drain both blocks and check all of them).
    const auto reused = bm.checkout_gpu_blocks(2);
    CHECK_EQ(reused.size(), 2u);
    for (const uint32_t rid : reused) {
        CHECK_EQ(bm.tag_count(rid), 0u);
    }
}

// Exercise every public C accessor from inside a picker callback, where node
// pointers are valid. Also checks the NULL sentinel for untagged nodes and
// that has_tag() is a pure read-only query.
struct accessor_probe {
    int         nodes     = 0;
    uint32_t    tagged_id = 0;
    uint32_t    untagged_id = 0;
    uint32_t    tagged_count = 0;
    std::string at0;
    std::string at1;
    std::string rep;
    int         has_early_str = -1;
    int         has_unknown   = -1;
    int         untagged_tag_is_null = -1;

    // Victim: the untagged node, i.e. the documented fallback victim.
    uint32_t    victim_id = 0;
};

static uint32_t pick_inspect_accessors_cb(void * ud, llama_paged_pool_kind pool, uint32_t n_needed,
                                          const llama_paged_free_block * head, uint32_t * out_ids,
                                          uint32_t out_capacity) {
    (void) pool;
    auto * p = (accessor_probe *) ud;
    for (const llama_paged_free_block * it = head; it != nullptr; it = llama_paged_free_block_next(it)) {
        p->nodes++;
        const uint32_t id      = llama_paged_free_block_id(it);
        const uint32_t n_tags  = llama_paged_free_block_tag_count(it);
        if (n_tags != 0) {
            p->tagged_id    = id;
            p->tagged_count = n_tags;
            const char * a  = llama_paged_free_block_tag_at(it, 0);
            const char * b  = llama_paged_free_block_tag_at(it, 1);
            const char * r  = llama_paged_free_block_tag(it);
            p->at0 = a != nullptr ? a : "";
            p->at1 = b != nullptr ? b : "";
            p->rep = r != nullptr ? r : "<null>";
            p->has_early_str = llama_paged_free_block_has_tag(it, "c-accessor-early");
            p->has_unknown   = llama_paged_free_block_has_tag(it, "c-accessor-never-written");
            if (p->victim_id == 0) {
                p->victim_id = id;   // fall back to a tagged block if needed
            }
        } else {
            p->untagged_id           = id;
            p->untagged_tag_is_null  = llama_paged_free_block_tag(it) == nullptr ? 1 : 0;
            p->victim_id             = id;   // prefer the untagged fallback victim
        }
    }
    if (n_needed == 0 || out_capacity == 0 || p->victim_id == 0) {
        return 0;
    }
    out_ids[0] = p->victim_id;
    return 1;
}

static void test_c_free_block_accessors() {
    llama_block_manager bm;
    bm.init(2, 0, 0.0f);
    bm.set_block_hash(0, 0xF000);
    bm.set_block_hash(1, 0xF001);

    // Both tags are written to block 0; tag() reports the lexicographically
    // first one, not necessarily the tag written first.
    accessor_probe probe;
    bm.set_tag(0, "c-accessor-late");
    bm.add_tag(0, "c-accessor-early");

    bm.set_evict_pick_cb(pick_inspect_accessors_cb, &probe);
    const auto ids = bm.checkout_gpu_blocks(1);   // sacrifices the untagged node 1

    CHECK_EQ(probe.nodes, 2);
    CHECK_EQ(probe.tagged_count, 2u);
    CHECK(probe.at0 == "c-accessor-early");
    CHECK(probe.at1 == "c-accessor-late");
    CHECK(probe.rep == "c-accessor-early");       // representative
    CHECK_EQ(probe.has_early_str, 1);
    CHECK_EQ(probe.has_unknown, 0);
    CHECK_EQ(probe.untagged_tag_is_null, 1);      // NULL sentinel, not ""
    CHECK_EQ(ids.size(), 1u);
    if (!ids.empty()) {
        CHECK_EQ(ids[0], probe.untagged_id);
    }
}

// ---------------------------------------------------------------------------
// Setter-lifecycle callbacks (scheduler / cache integration tests)
// ---------------------------------------------------------------------------

struct sched_cb_state {
    int                   calls     = 0;
    llama_paged_pool_kind last_pool = LLAMA_PAGED_POOL_FULL;
};
static sched_cb_state g_sched_cb;

// Refuses every sacrifice (expected back-pressure, design 8.4).
static uint32_t pick_zero_cb(void * ud, llama_paged_pool_kind pool, uint32_t n_needed,
                             const llama_paged_free_block * head, uint32_t * out_ids, uint32_t out_capacity) {
    (void) ud;
    (void) n_needed;
    (void) head;
    (void) out_ids;
    (void) out_capacity;
    g_sched_cb.calls++;
    g_sched_cb.last_pool = pool;
    return 0;
}

// Returns the first n_needed cache-resident nodes in queue order.
static uint32_t pick_head_cached_cb(void * ud, llama_paged_pool_kind pool, uint32_t n_needed,
                                    const llama_paged_free_block * head, uint32_t * out_ids,
                                    uint32_t out_capacity) {
    (void) ud;
    g_sched_cb.calls++;
    g_sched_cb.last_pool = pool;
    uint32_t n = 0;
    for (const llama_paged_free_block * it = head;
         it != nullptr && n < n_needed && n < out_capacity; it = llama_paged_free_block_next(it)) {
        if (!llama_paged_free_block_has_hash(it)) {
            continue;
        }
        out_ids[n++] = llama_paged_free_block_id(it);
    }
    return n;
}

// Violates the callback contract by returning more ids than the buffer holds.
static uint32_t pick_overflow_cb(void * ud, llama_paged_pool_kind pool, uint32_t n_needed,
                                 const llama_paged_free_block * head, uint32_t * out_ids,
                                 uint32_t out_capacity) {
    (void) ud;
    (void) pool;
    (void) n_needed;
    (void) head;
    (void) out_ids;
    g_cb.calls++;
    return out_capacity + 1;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Turns every GPU free block into a cache-resident one (free queue order is
// preserved, so node ids appear at the queue head in ascending order).
static void hash_all_free(llama_block_manager & bm, uint32_t n_gpu, uint64_t base,
                          const char * tag) {
    for (uint32_t id = 0; id < n_gpu; ++id) {
        bm.set_block_hash(id, base + id);
        if (tag != nullptr) {
            bm.set_tag(id, tag);
        }
    }
}

// ---------------------------------------------------------------------------
// 1. Legacy path: no callback installed -> plain FIFO queue-head order
// ---------------------------------------------------------------------------

static void test_legacy_path() {
    llama_block_manager bm;
    bm.init(8, 0, 0.0f);

    CHECK_EQ(bm.n_plain_free_gpu_blocks(), 8u);

    const auto first = bm.checkout_gpu_blocks(3);
    check_ids("legacy pops queue head", first, { 0, 1, 2 });

    const auto second = bm.checkout_gpu_blocks(2);
    check_ids("legacy continues in queue order", second, { 3, 4 });

    CHECK_EQ(bm.n_free_gpu_blocks(), 3u);
    CHECK_EQ(bm.n_cached_gpu_blocks(), 0u);
    CHECK_EQ(bm.n_allocatable_gpu_blocks(), 3u);
}

// ---------------------------------------------------------------------------
// 2. Picker returns 0 with n_needed > 0 -> atomic failure, no state change
// ---------------------------------------------------------------------------

static void test_callback_returns_zero() {
    llama_block_manager bm;
    bm.init(8, 0, 0.0f);
    hash_all_free(bm, 8, 0x1000, "seed");

    CHECK_EQ(bm.n_plain_free_gpu_blocks(), 0u);
    CHECK_EQ(bm.n_free_gpu_blocks(), 0u);
    CHECK_EQ(bm.n_cached_gpu_blocks(), 8u);

    reset_cb({});  // picker refuses to sacrifice anything
    bm.set_evict_pick_cb(pick_fixed_cb, nullptr);

    const size_t free_before   = bm.n_free_gpu_blocks();
    const size_t cached_before = bm.n_cached_gpu_blocks();
    const size_t alloc_before  = bm.n_allocatable_gpu_blocks();

    const auto ids = bm.checkout_gpu_blocks(2);

    CHECK(ids.empty());
    CHECK_EQ(g_cb.calls, 1);
    CHECK_EQ(g_cb.last_n_needed, 2u);  // 2 requested, 0 plain -> 2 sacrifices needed

    // Atomic failure: nothing was popped, cleared or re-tagged.
    CHECK_EQ(bm.n_free_gpu_blocks(), free_before);
    CHECK_EQ(bm.n_cached_gpu_blocks(), cached_before);
    CHECK_EQ(bm.n_allocatable_gpu_blocks(), alloc_before);
    CHECK(bm.block_has_hash(0));
    CHECK_EQ(bm.block_ref_count(0), 0u);
}

// ---------------------------------------------------------------------------
// 3. Selective victims: only the ids named by the picker are sacrificed
// ---------------------------------------------------------------------------

static void test_selective_victims() {
    llama_block_manager bm;
    bm.init(8, 0, 0.0f);
    hash_all_free(bm, 8, 0x2000, "seed");

    reset_cb({ 2, 5 });
    bm.set_evict_pick_cb(pick_fixed_cb, nullptr);

    std::vector<llama_block_manager::evicted_block> evicted;
    const auto ids = bm.checkout_gpu_blocks(2, &evicted);

    check_ids("picker's victims in returned order", ids, { 2, 5 });
    CHECK_EQ(evicted.size(), 2u);
    if (evicted.size() == 2) {
        CHECK_EQ(evicted[0].block_id, 2u);
        CHECK_EQ(evicted[0].hash, 0x2002u);
        CHECK_EQ(evicted[1].block_id, 5u);
        CHECK_EQ(evicted[1].hash, 0x2005u);
    }

    // Selected victims lost their cache identity and are owned by the caller.
    CHECK(!bm.block_has_hash(2));
    CHECK(!bm.block_has_hash(5));
    CHECK_EQ(bm.block_ref_count(2), 1u);
    CHECK_EQ(bm.block_ref_count(5), 1u);
    CHECK(bm.tag(2).empty());

    // Unselected cache-resident blocks keep hash + tag and stay free.
    for (const uint32_t id : { 0u, 1u, 3u, 4u, 6u, 7u }) {
        CHECK(bm.block_has_hash(id));
        CHECK_EQ(bm.block_ref_count(id), 0u);
    }
    CHECK(bm.tag(0) == "seed");

    CHECK_EQ(bm.n_cached_gpu_blocks(), 6u);
    CHECK_EQ(bm.n_free_gpu_blocks(), 0u);
    CHECK_EQ(bm.n_allocatable_gpu_blocks(), 6u);
}

// Same contract, but the policy is expressed over tags via the C accessors.
static void test_tag_based_picker() {
    llama_block_manager bm;
    bm.init(8, 0, 0.0f);
    hash_all_free(bm, 8, 0xA000, "seed");
    bm.set_tag(0, "keep");
    bm.set_tag(1, "keep");

    g_picked_hashes.clear();
    bm.set_evict_pick_cb(pick_skip_tag_cb, (void *) "keep");

    const auto ids = bm.checkout_gpu_blocks(2);

    check_ids("tag policy skips 'keep' blocks", ids, { 2, 3 });
    CHECK_EQ(g_picked_hashes.size(), 2u);
    if (g_picked_hashes.size() == 2) {
        CHECK_EQ(g_picked_hashes[0], 0xA002u);
        CHECK_EQ(g_picked_hashes[1], 0xA003u);
    }
    CHECK(bm.block_has_hash(0));
    CHECK(bm.block_has_hash(1));
    CHECK(bm.has_tag(0, "keep"));
    CHECK(bm.has_tag(1, "keep"));
    // Blocks 0/1 also carry the "seed" tag: the set holds both tags.
    CHECK(bm.tag_count(0) >= 2u);
    CHECK(bm.tag_count(1) >= 2u);
}

// ---------------------------------------------------------------------------
// 4. Over-sacrifice: extras are really sacrificed and become plain free blocks
// ---------------------------------------------------------------------------

static void test_over_sacrifice() {
    llama_block_manager bm;
    bm.init(8, 0, 0.0f);
    hash_all_free(bm, 8, 0x3000, "seed");

    // Requesting 1 block needs 1 sacrifice, but the picker hands back 3.
    reset_cb({ 7, 6, 5 });
    bm.set_evict_pick_cb(pick_fixed_cb, nullptr);

    std::vector<llama_block_manager::evicted_block> evicted;
    const auto ids = bm.checkout_gpu_blocks(1, &evicted);

    check_ids("only the needed victim is returned", ids, { 7 });
    CHECK_EQ(evicted.size(), 3u);  // every sacrificed block is reported

    CHECK_EQ(bm.block_ref_count(7), 1u);
    CHECK_EQ(bm.block_ref_count(6), 0u);
    CHECK_EQ(bm.block_ref_count(5), 0u);
    CHECK(!bm.block_has_hash(6));
    CHECK(!bm.block_has_hash(5));

    CHECK_EQ(bm.n_cached_gpu_blocks(), 5u);      // 8 - 3 sacrificed
    CHECK_EQ(bm.n_allocatable_gpu_blocks(), 7u); // extras are free again
    CHECK_EQ(bm.n_free_gpu_blocks(), 2u);        // the two extras are plain

    // Extras were pushed back at the tail as plain blocks: a plain-only
    // checkout hands them out and never consults the picker again.
    const int calls_before = g_cb.calls;
    const auto extras      = bm.checkout_gpu_blocks(2);
    CHECK_EQ(extras.size(), 2u);
    CHECK(ids_contain(extras, 5));
    CHECK(ids_contain(extras, 6));
    CHECK_EQ(g_cb.calls, calls_before);
}

// ---------------------------------------------------------------------------
// 5. Duplicate / invalid ids -> atomic failure, no state change
// ---------------------------------------------------------------------------

static void test_duplicate_ids() {
    llama_block_manager bm;
    bm.init(8, 0, 0.0f);
    hash_all_free(bm, 8, 0x4000, "seed");

    reset_cb({ 3, 3 });
    bm.set_evict_pick_cb(pick_fixed_cb, nullptr);

    const size_t cached_before = bm.n_cached_gpu_blocks();
    const size_t alloc_before  = bm.n_allocatable_gpu_blocks();

    const auto ids = bm.checkout_gpu_blocks(2);
    CHECK(ids.empty());
    CHECK_EQ(bm.n_cached_gpu_blocks(), cached_before);
    CHECK_EQ(bm.n_allocatable_gpu_blocks(), alloc_before);
    CHECK(bm.block_has_hash(3));
    CHECK_EQ(bm.block_ref_count(3), 0u);

    // Out-of-range ids are ignored as well; they cannot make up a short batch.
    reset_cb({ 3, 1000 });
    const auto ids2 = bm.checkout_gpu_blocks(2);
    CHECK(ids2.empty());
    CHECK_EQ(bm.n_cached_gpu_blocks(), cached_before);
    CHECK_EQ(bm.n_allocatable_gpu_blocks(), alloc_before);
    CHECK(bm.block_has_hash(3));
    CHECK_EQ(bm.block_ref_count(3), 0u);
}

// ---------------------------------------------------------------------------
// 6. The pool kind configured on the block manager reaches the callback
// ---------------------------------------------------------------------------

static void test_pool_kind() {
    llama_block_manager bm;
    bm.init(8, 0, 0.0f);
    hash_all_free(bm, 8, 0x5000, "seed");

    reset_cb({ 0, 1 });
    bm.set_pool_kind(LLAMA_PAGED_POOL_LINEAR);
    bm.set_evict_pick_cb(pick_fixed_cb, nullptr);

    const auto ids = bm.checkout_gpu_blocks(2);
    check_ids("linear pool victims", ids, { 0, 1 });
    CHECK_EQ(g_cb.calls, 1);
    CHECK_EQ(g_cb.last_pool, LLAMA_PAGED_POOL_LINEAR);

    // Switching the pool kind must be visible on the next eviction.
    reset_cb({ 6 });
    bm.set_pool_kind(LLAMA_PAGED_POOL_FULL);
    const auto ids2 = bm.checkout_gpu_blocks(1);
    check_ids("full pool victims", ids2, { 6 });
    CHECK_EQ(g_cb.last_pool, LLAMA_PAGED_POOL_FULL);
}

// ---------------------------------------------------------------------------
// 7. Plain-only fill: cached blocks are never touched when plain blocks suffice
// ---------------------------------------------------------------------------

static void test_plain_only_fill() {
    llama_block_manager bm;
    bm.init(8, 0, 0.0f);

    // Cache-resident blocks sit at the queue head. A legacy FIFO pop would
    // evict them; the picker-based path must skip them instead.
    bm.set_block_hash(0, 0x7000);
    bm.set_tag(0, "head");
    bm.set_block_hash(1, 0x7001);
    bm.set_tag(1, "head");

    reset_cb({ 0, 1 });  // would evict the head blocks if ever consulted
    bm.set_evict_pick_cb(pick_fixed_cb, nullptr);

    // 4 <= 6 plain blocks -> no sacrifice needed, picker must not run.
    const auto ids = bm.checkout_gpu_blocks(4);
    check_ids("plain free blocks only", ids, { 2, 3, 4, 5 });
    CHECK_EQ(g_cb.calls, 0);

    CHECK(bm.block_has_hash(0));
    CHECK(bm.block_has_hash(1));
    CHECK_EQ(bm.block_ref_count(0), 0u);
    CHECK_EQ(bm.block_ref_count(1), 0u);
    CHECK(bm.tag(0) == "head");

    CHECK_EQ(bm.n_cached_gpu_blocks(), 2u);
    CHECK_EQ(bm.n_free_gpu_blocks(), 2u);        // blocks 6 and 7
    CHECK_EQ(bm.n_allocatable_gpu_blocks(), 4u); // 0, 1, 6, 7
}

// ---------------------------------------------------------------------------
// 8. Default picker: earliest cache-resident nodes in queue order
// ---------------------------------------------------------------------------

static void test_default_picker() {
    llama_block_manager bm;
    bm.init(8, 0, 0.0f);
    // Interleave plain (0,1,3,5,7) and cache-resident (2,4,6) nodes.
    bm.set_block_hash(2, 0x8002);
    bm.set_block_hash(4, 0x8004);
    bm.set_block_hash(6, 0x8006);

    bm.set_evict_pick_cb(llama_paged_evict_pick_default, nullptr);

    // 7 requested, 5 plain -> 2 sacrifices. Default picks the earliest cached
    // nodes (2 then 4); the rest is filled with plain blocks only.
    std::vector<llama_block_manager::evicted_block> evicted;
    const auto ids = bm.checkout_gpu_blocks(7, &evicted);
    check_ids("default: earliest cached then plain fill", ids, { 2, 4, 0, 1, 3, 5, 7 });
    CHECK_EQ(evicted.size(), 2u);
    if (evicted.size() == 2) {
        CHECK_EQ(evicted[0].block_id, 2u);
        CHECK_EQ(evicted[0].hash, 0x8002u);
        CHECK_EQ(evicted[1].block_id, 4u);
        CHECK_EQ(evicted[1].hash, 0x8004u);
    }
    CHECK(bm.block_has_hash(6));  // unselected cached block survives
    CHECK_EQ(bm.n_cached_gpu_blocks(), 1u);
    CHECK_EQ(bm.n_allocatable_gpu_blocks(), 1u);

    // Degenerate inputs return 0 without writing anything.
    uint32_t out[4] = { 0, 0, 0, 0 };
    CHECK_EQ(llama_paged_evict_pick_default(nullptr, LLAMA_PAGED_POOL_FULL, 2, nullptr, out, 4), 0u);
    CHECK_EQ(llama_paged_evict_pick_default(nullptr, LLAMA_PAGED_POOL_FULL, 2, nullptr, out, 0), 0u);
    CHECK_EQ(llama_paged_evict_pick_default(nullptr, LLAMA_PAGED_POOL_FULL, 0, nullptr, out, 4), 0u);

    // out_capacity caps the batch: ask for 5 victims with room for 2 only.
    llama_block_manager bm2;
    bm2.init(8, 0, 0.0f);
    bm2.set_block_hash(2, 0x9002);
    bm2.set_block_hash(4, 0x9004);
    bm2.set_block_hash(6, 0x9006);

    g_probe_ret    = 0;
    g_probe_ids[0] = 0;
    g_probe_ids[1] = 0;
    bm2.set_evict_pick_cb(pick_probe_capacity_cb, nullptr);

    const auto none = bm2.checkout_gpu_blocks(6);  // needs 1 sacrifice
    CHECK(none.empty());                           // the probe returned 0
    CHECK_EQ(g_probe_ret, 2u);
    CHECK_EQ(g_probe_ids[0], 2u);
    CHECK_EQ(g_probe_ids[1], 4u);
}

// ---------------------------------------------------------------------------
// 9. cb == NULL after an install is byte-identical to never installing one
// ---------------------------------------------------------------------------

static void test_legacy_null_reset_equivalence() {
    llama_block_manager never;
    llama_block_manager reset;
    never.init(8, 0, 0.0f);
    reset.init(8, 0, 0.0f);
    hash_all_free(never, 8, 0xB000, "seed");
    hash_all_free(reset, 8, 0xB000, "seed");

    // Install, then uninstall through the documented NULL setter.
    reset_cb({});
    reset.set_evict_pick_cb(pick_fixed_cb, nullptr);
    reset.set_evict_pick_cb(nullptr, nullptr);

    std::vector<llama_block_manager::evicted_block> ev_never;
    std::vector<llama_block_manager::evicted_block> ev_reset;
    const auto ids_never = never.checkout_gpu_blocks(3, &ev_never);
    const auto ids_reset = reset.checkout_gpu_blocks(3, &ev_reset);

    check_ids("NULL setter restores legacy FIFO", ids_reset, ids_never);
    CHECK_EQ(g_cb.calls, 0);
    CHECK_EQ(ev_reset.size(), ev_never.size());
    for (size_t i = 0; i < ev_reset.size() && i < ev_never.size(); ++i) {
        CHECK_EQ(ev_reset[i].block_id, ev_never[i].block_id);
        CHECK_EQ(ev_reset[i].hash, ev_never[i].hash);
    }
}

// ---------------------------------------------------------------------------
// 10. checkout_cpu_blocks never consults the GPU eviction policy
// ---------------------------------------------------------------------------

static void test_cpu_checkout_ignores_policy() {
    llama_block_manager bm;
    bm.init(4, 4, 0.0f);

    // Cache-resident blocks on both pools. CPU ids start after the GPU pool.
    for (uint32_t id = 0; id < 4; ++id) {
        bm.set_block_hash(id, 0xC000 + id);
        bm.set_tag(id, "seed");
    }
    for (uint32_t id = 4; id < 8; ++id) {
        bm.set_block_hash(id, 0xC100 + id);
        bm.set_tag(id, "seed");
    }

    reset_cb({ 0, 1 });
    bm.set_evict_pick_cb(pick_fixed_cb, nullptr);
    const int calls_before = g_cb.calls;

    const auto ids = bm.checkout_cpu_blocks(2);
    check_ids("CPU checkout keeps FIFO order", ids, { 4, 5 });
    CHECK_EQ(g_cb.calls, calls_before);  // GPU picker not invoked
    CHECK_EQ(bm.n_cached_gpu_blocks(), 4u);
}

// ---------------------------------------------------------------------------
// 11. A picker naming a CPU-pool id is rejected (block_at id-space overlap)
// ---------------------------------------------------------------------------

static void test_cpu_id_rejected() {
    llama_block_manager bm;
    bm.init(4, 4, 0.0f);
    hash_all_free(bm, 4, 0xD000, "seed");
    // CPU id 4 is a free, cache-resident CPU block: the dangerous case for
    // block_at() because the id exists but belongs to the other pool.
    bm.set_block_hash(4, 0xD004);

    reset_cb({ 4 });
    bm.set_evict_pick_cb(pick_fixed_cb, nullptr);

    const auto ids = bm.checkout_gpu_blocks(1);
    CHECK(ids.empty());
    CHECK_EQ(g_cb.calls, 1);
    CHECK(bm.gpu_checkout_refused_by_policy());
    CHECK_EQ(bm.n_cached_gpu_blocks(), 4u);
    CHECK_EQ(bm.n_allocatable_gpu_blocks(), 4u);
    CHECK(bm.block_has_hash(4));
    CHECK_EQ(bm.block_ref_count(4), 0u);

    bm.reset_gpu_checkout_refusal();
    CHECK(!bm.gpu_checkout_refused_by_policy());
}

// ---------------------------------------------------------------------------
// 12. A callback returning more ids than out_capacity fails atomically
//    (no GGML_ABORT on user-supplied output)
// ---------------------------------------------------------------------------

static void test_callback_over_capacity_fails_atomically() {
    llama_block_manager bm;
    bm.init(4, 0, 0.0f);
    hash_all_free(bm, 4, 0xE000, "seed");

    reset_cb({});
    bm.set_evict_pick_cb(pick_overflow_cb, nullptr);

    const auto ids = bm.checkout_gpu_blocks(1);
    CHECK(ids.empty());
    CHECK_EQ(g_cb.calls, 1);
    CHECK_EQ(bm.n_cached_gpu_blocks(), 4u);
    CHECK_EQ(bm.n_allocatable_gpu_blocks(), 4u);
    CHECK_EQ(bm.block_ref_count(0), 0u);
}

// ---------------------------------------------------------------------------
// Scheduler / cache integration (real paged cache + scheduler, CPU backend)
// ---------------------------------------------------------------------------

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

static paged_test_fixture make_fixture(uint32_t n_gpu_blocks = 4, uint32_t n_cpu_blocks = 1) {
    paged_test_fixture fixture;
    fixture.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    CHECK(fixture.backend != nullptr);

    fixture.kv = std::unique_ptr<llama_kv_cache_paged>(new llama_kv_cache_paged(
        /*head_dim=*/64u,
        /*n_heads_kv=*/4u,
        /*block_size=*/16u,
        /*n_layers=*/2u,
        /*n_ubatch=*/64u,
        /*n_seq_max=*/8u));
    fixture.kv->init(fixture.backend, fixture.backend, GGML_TYPE_F16, n_gpu_blocks, n_cpu_blocks, /*watermark=*/0.0f);

    fixture.sched = std::unique_ptr<llama_paged_scheduler_impl>(
        new llama_paged_scheduler_impl(/*n_ctx=*/128, /*block_sz=*/16, /*n_batch=*/64,
                                       /*n_pos_per_embd=*/1, fixture.kv.get()));
    return fixture;
}

static llama_sequence_group make_sched_group(int32_t request_id, uint32_t n_prompt) {
    llama_sequence_group group;
    group.request_id     = request_id;
    group.n_prompt       = n_prompt;
    group.n_decoded      = 0;
    group.n_past         = 0;
    group.t_arrival_time = request_id;
    group.logical_seq.assign(n_prompt, /*dummy token=*/1);
    return group;
}

// Turns every GPU block into a cache-resident free block (ref_count == 0,
// has_hash == true), consuming the whole GPU pool.
static void make_gpu_cache_resident(llama_kv_cache_paged & kv, uint32_t n_gpu, uint64_t hash_base,
                                    int32_t req_base) {
    for (uint32_t i = 0; i < n_gpu; ++i) {
        llama_sequence_group group = make_sched_group(req_base + (int32_t) i, /*n_prompt=*/16);
        if (!kv.allocate(/*num_tokens=*/0, group)) {
            CHECK(false && "cache-resident fill: allocate failed");
            return;
        }
        CHECK_EQ(group.block_table.size(), 1u);
        if (group.block_table.empty()) {
            return;
        }
        kv.register_prefix_block(hash_base + i, group.block_table.front(), group.request_id, /*block_idx=*/0,
                                 /*first_tok=*/1, /*last_tok=*/2);
        kv.free_blocks(group);
    }
}

// ---------------------------------------------------------------------------
// 13. sched == NULL returns false (public C API)
// ---------------------------------------------------------------------------

static void test_public_setter_null_sched() {
    CHECK(!llama_paged_scheduler_set_evict_decision(nullptr, pick_zero_cb, nullptr));
}

// ---------------------------------------------------------------------------
// 14. Policy refusal is back-pressure, not a scheduler deadlock; the policy can
//     be switched between two steps and legacy FIFO restored with cb == NULL.
// ---------------------------------------------------------------------------

static void test_scheduler_refusal_not_deadlock_and_policy_switch() {
    auto fixture = make_fixture();
    make_gpu_cache_resident(*fixture.kv, /*n_gpu=*/4, 0xF000, /*req_base=*/100);
    CHECK_EQ(fixture.kv->n_cached_gpu_blocks(), 4u);
    CHECK_EQ(fixture.kv->n_allocatable_gpu_blocks(), 4u);

    g_sched_cb = sched_cb_state();
    fixture.sched->set_evict_decision(pick_zero_cb, nullptr);

    CHECK(fixture.sched->queue_request(make_sched_group(/*request_id=*/0, /*n_prompt=*/16)));

    llama_batch batch = {};
    CHECK(fixture.sched->step(batch) == llama_scheduler_status::OK);
    CHECK_EQ(batch.n_tokens, 0);          // nothing schedulable this step
    CHECK_EQ(g_sched_cb.calls, 1);
    CHECK_EQ(g_sched_cb.last_pool, LLAMA_PAGED_POOL_FULL);
    CHECK(fixture.kv->gpu_checkout_refused_by_policy());

    // Switch policy between two steps: this picker sacrifices the first cached
    // block, so the waiting request can now be admitted.
    g_sched_cb = sched_cb_state();
    fixture.sched->set_evict_decision(pick_head_cached_cb, nullptr);

    CHECK(fixture.sched->step(batch) == llama_scheduler_status::OK);
    CHECK_EQ(batch.n_tokens, 16);
    CHECK_EQ(g_sched_cb.calls, 1);
    CHECK_EQ(g_sched_cb.last_pool, LLAMA_PAGED_POOL_FULL);
    CHECK_EQ(fixture.kv->n_cached_gpu_blocks(), 2u);  // full prompt + one decode slot

    // cb == NULL restores legacy FIFO: no callback, queue-head eviction.
    g_sched_cb = sched_cb_state();
    fixture.sched->set_evict_decision(nullptr, nullptr);
    CHECK(fixture.sched->queue_request(make_sched_group(/*request_id=*/1, /*n_prompt=*/16)));
    CHECK(fixture.sched->step(batch) == llama_scheduler_status::OK);
    CHECK_EQ(batch.n_tokens, 16);
    CHECK_EQ(g_sched_cb.calls, 0);
}

// ---------------------------------------------------------------------------
// 15. clear() rebuilds the block manager and the policy is re-installed
// ---------------------------------------------------------------------------

static void test_policy_reinstalled_after_clear() {
    auto fixture = make_fixture();
    make_gpu_cache_resident(*fixture.kv, /*n_gpu=*/4, 0xA000, /*req_base=*/200);

    g_sched_cb = sched_cb_state();
    fixture.sched->set_evict_decision(pick_zero_cb, nullptr);

    llama_sequence_group group = make_sched_group(/*request_id=*/1, /*n_prompt=*/16);
    CHECK(!fixture.kv->allocate(/*num_tokens=*/0, group));  // policy refuses
    CHECK_EQ(g_sched_cb.calls, 1);
    CHECK_EQ(fixture.kv->n_cached_gpu_blocks(), 4u);        // atomic failure

    // Rebuild the block manager exactly like llama_memory_clear() does.
    fixture.kv->clear(/*data=*/false);
    CHECK_EQ(fixture.kv->n_cached_gpu_blocks(), 0u);
    make_gpu_cache_resident(*fixture.kv, /*n_gpu=*/4, 0xA100, /*req_base=*/300);

    llama_sequence_group group2 = make_sched_group(/*request_id=*/2, /*n_prompt=*/16);
    CHECK(!fixture.kv->allocate(/*num_tokens=*/0, group2));
    CHECK_EQ(g_sched_cb.calls, 2);                         // policy still installed
    CHECK_EQ(fixture.kv->n_cached_gpu_blocks(), 4u);
}


// ---------------------------------------------------------------------------
// 16. Reusing/ sharing a cache-resident prefix block unions the requesting tag
//     into the block's set (design 10.5b), and collect_tag_blocks() attributes
//     the block to every tag in the set.
// ---------------------------------------------------------------------------

static void test_prefix_reuse_unions_tags() {
    auto fixture = make_fixture(/*n_gpu_blocks=*/4, /*n_cpu_blocks=*/1);

    llama_sequence_group group = make_sched_group(/*request_id=*/0, /*n_prompt=*/16);
    group.tag = "reuse-A";
    CHECK(fixture.kv->allocate(/*num_tokens=*/0, group));
    CHECK_EQ(group.block_table.size(), 1u);
    if (group.block_table.empty()) {
        return;
    }
    const uint32_t id   = group.block_table.front();
    const uint64_t hash = 0x5A5A0001u;
    fixture.kv->register_prefix_block(hash, id, /*owner_seq=*/0, /*block_idx=*/0,
                                      /*first_tok=*/1, /*last_tok=*/2);
    fixture.kv->free_blocks(group);              // ref -> 0, still cache-resident
    CHECK(fixture.kv->block_tag(id) == "reuse-A");

    // Prefix hit: acquire_blocks() must union the hitting tag in its second
    // pass. With the old single-owner model the tag set would stay {A}.
    CHECK(fixture.kv->acquire_blocks({ id }, "reuse-B"));

    std::unordered_map<std::string, std::unordered_set<uint32_t>> owned;
    fixture.kv->collect_tag_blocks(owned, nullptr);
    CHECK(owned["reuse-A"].count(id) == 1u);
    CHECK(owned["reuse-B"].count(id) == 1u);

    // Shared reference path (add_ref_blocks via share_blocks) unions too.
    CHECK(fixture.kv->add_ref_blocks({ id }, "reuse-C"));
    owned.clear();
    fixture.kv->collect_tag_blocks(owned, nullptr);
    CHECK(owned["reuse-C"].count(id) == 1u);

    // The representative tag is the lexicographically first one ("reuse-A").
    CHECK(fixture.kv->block_tag(id) == "reuse-A");

    // swap_in also re-tags the restored private GPU block (swap_out clears the
    // tag of the unhashed GPU copy, swap_in writes it back).
    llama_sequence_group swap_group = make_sched_group(/*request_id=*/1, /*n_prompt=*/16);
    swap_group.tag = "swap-A";
    CHECK(fixture.kv->allocate(/*num_tokens=*/0, swap_group));
    CHECK_EQ(swap_group.block_table.size(), 1u);
    if (!swap_group.block_table.empty()) {
        CHECK(fixture.kv->swap_out(swap_group));
        CHECK(fixture.kv->swap_in(swap_group));
        CHECK_EQ(swap_group.block_table.size(), 1u);
        if (!swap_group.block_table.empty()) {
            CHECK(fixture.kv->block_tag(swap_group.block_table.front()) == "swap-A");
        }
    }
}

// ---------------------------------------------------------------------------

static void run(const char * name, void (*fn)()) {
    const int before = g_failures;
    fn();
    fprintf(stderr, "%s %s\n", g_failures == before ? "PASS" : "FAIL", name);
}

int main() {
    run("legacy_path", test_legacy_path);
    run("callback_returns_zero", test_callback_returns_zero);
    run("selective_victims", test_selective_victims);
    run("tag_based_picker", test_tag_based_picker);
    run("over_sacrifice", test_over_sacrifice);
    run("duplicate_ids", test_duplicate_ids);
    run("pool_kind", test_pool_kind);
    run("plain_only_fill", test_plain_only_fill);
    run("default_picker", test_default_picker);
    run("legacy_null_reset_equivalence", test_legacy_null_reset_equivalence);
    run("cpu_checkout_ignores_policy", test_cpu_checkout_ignores_policy);
    run("cpu_id_rejected", test_cpu_id_rejected);
    run("callback_over_capacity_fails_atomically", test_callback_over_capacity_fails_atomically);
    run("public_setter_null_sched", test_public_setter_null_sched);
    run("scheduler_refusal_not_deadlock_and_policy_switch", test_scheduler_refusal_not_deadlock_and_policy_switch);
    run("policy_reinstalled_after_clear", test_policy_reinstalled_after_clear);
    run("multi_tag_set", test_multi_tag_set);
    run("tags_survive_cache_resident_release", test_tags_survive_cache_resident_release);
    run("tags_cleared_on_release_to_plain", test_tags_cleared_on_release_to_plain);
    run("tags_cleared_on_eviction", test_tags_cleared_on_eviction);
    run("tags_cleared_on_clear_hash", test_tags_cleared_on_clear_hash);
    run("c_free_block_accessors", test_c_free_block_accessors);
    run("prefix_reuse_unions_tags", test_prefix_reuse_unions_tags);

    if (g_failures != 0) {
        fprintf(stderr, "FAIL: %d check(s) failed\n", g_failures);
        return 1;
    }
    fprintf(stderr, "PASS\n");
    return 0;
}
