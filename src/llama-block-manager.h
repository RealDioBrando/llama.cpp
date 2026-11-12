#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

// Physical block accountant for one paged pool.
//
// A block is either held by one or more requests (ref_count > 0) or it sits in
// the free queue (ref_count == 0). Free blocks may additionally carry a prefix
// hash: such a block is cache-resident - its K/V (or state) data is still valid
// and the prefix index still points at it. Allocation pops from the queue head;
// if the popped block carries a hash, the caller is handed an `evicted_block`
// record so it can drop the corresponding prefix-index entry (lazy eviction).
//
// The free queue is an intrusive FIFO (doubly linked) because the cache wants
// prefix-preserving eviction (release in reverse order) and O(1) touch/removal
// when a free cached block is reused.

// Free-queue node. This is also the concrete type behind the opaque C API type
// `llama_paged_free_block` (declared in include/llama.h): an eviction picker
// callback receives the head of the real free queue and walks it through
// llama_paged_free_block_next() - zero copy, no snapshot.
//
// Node addresses are stable for the lifetime of the owning registry: the
// registry vectors are only resized inside llama_block_manager::init(), never
// while a node pointer (internal or handed to a callback) is alive.
struct llama_paged_free_block {
    uint32_t id        = 0;
    uint32_t ref_count = 0;
    bool     is_gpu    = false;

    // Prefix-cache identity. `has_hash` is the authoritative "cache resident"
    // bit; `hash` is the chain hash registered for this block.
    bool     has_hash  = false;
    uint64_t hash      = 0;

    // Every tag that has used this block while it stayed cache-resident
    // (historical union, not just the current owner). Only GPU blocks are
    // tagged: the CPU pool does not participate in eviction and is being
    // refactored. Tags are stored as plain strings in a std::set, so
    // membership is O(log k) and enumeration is lexicographic. No "first
    // owner" is tracked. The set survives free() while the block keeps its
    // hash and is cleared when the hash is dropped or a non-cached block is
    // freed.
    std::set<std::string> tags;


    // Intrusive free-queue links (raw node pointers, nullptr = none).
    bool                     in_free = false;
    llama_paged_free_block * prev    = nullptr;
    llama_paged_free_block * next    = nullptr;
};

class llama_block_manager {
  public:
    // The free lists are stored as raw node pointers into the two registry
    // vectors below. Copying the object would deep-copy the vectors but leave
    // the copied links pointing into the source buffers, so copying is not
    // allowed. Moves are fine: the vector buffers (and therefore the nodes)
    // are stolen as a whole, so the link pointers stay valid in the target.
    llama_block_manager()                                        = default;
    llama_block_manager(const llama_block_manager &)             = delete;
    llama_block_manager & operator=(const llama_block_manager &) = delete;
    llama_block_manager(llama_block_manager &&)                  = default;
    llama_block_manager & operator=(llama_block_manager &&)      = default;

    // Emitted when a checkout pops a cache-resident block. The hash entry must
    // be removed from the owning prefix index before the block is reused.
    struct evicted_block {
        uint64_t hash      = 0;
        uint32_t block_id  = 0;
    };

    // Eviction policy for GPU checkouts (see
    // docs/kvcache-manager/prefix-cache-evict-policy.md). The callback is only
    // invoked when the current allocation MUST sacrifice cache-resident blocks;
    // `evict_pick_cb == nullptr` keeps the legacy FIFO behaviour unchanged.
    //
    // The cache owning this block manager installs both the callback and the
    // pool kind (full-attention vs linear-state block-id spaces overlap, so the
    // kind must come from the owner).
    llama_paged_evict_pick_cb evict_pick_cb = nullptr;
    void *                    evict_pick_ud = nullptr;
    llama_paged_pool_kind     pool_kind     = LLAMA_PAGED_POOL_FULL;

    void set_evict_pick_cb(llama_paged_evict_pick_cb cb, void * ud);
    void set_pool_kind(llama_paged_pool_kind kind);

    // True when a GPU checkout failed only because the installed eviction
    // picker refused to name enough victims (design §8.4). That is expected
    // back-pressure, not capacity exhaustion, so the scheduler must not treat
    // it as a deadlock. The flag accumulates until reset by its owner (once per
    // scheduling step) and is cleared by init().
    bool gpu_checkout_refused_by_policy() const;
    void reset_gpu_checkout_refusal();

  private:
    using physical_block = llama_paged_free_block;

    using physical_block_bool = std::vector<physical_block>;
    using physical_block_ids  = std::vector<uint32_t>;

    physical_block_bool gpu_registry;
    physical_block_bool cpu_registry;

    // Free queues (FIFO): pop_front for allocation, push_back for release.
    physical_block * free_head_gpu = nullptr;
    physical_block * free_tail_gpu = nullptr;
    physical_block * free_head_cpu = nullptr;
    physical_block * free_tail_cpu = nullptr;

    size_t n_free_gpu = 0;   // queue length (includes cache-resident blocks)
    size_t n_free_cpu = 0;
    size_t n_cached_gpu = 0; // subset of n_free_gpu with has_hash == true
    size_t n_cached_cpu = 0;

    bool gpu_checkout_refused_by_policy_ = false;

    uint32_t watermark_gpu_safety_num_blocks;
    uint32_t watermark_cpu_safety_num_blocks;

    uint32_t total_num_gpu_blocks;
    uint32_t total_num_cpu_blocks;

    physical_block *       block_at(uint32_t id);
    const physical_block * block_at(uint32_t id) const;

    // Drop the whole tag set. Must be called exactly
    // where cache identity is lost, never while the block keeps has_hash.
    void clear_tags(physical_block & b);

    // Intrusive FIFO primitives, node-pointer based.
    void            queue_push_back(physical_block * b);
    physical_block * queue_pop_front(bool gpu);
    void            queue_remove(physical_block * b);

    // GPU checkout helpers. "plain" means !has_hash, i.e. NOT cache-resident.
    physical_block_ids pop_front_n_gpu(uint32_t num_blocks,
                                       std::vector<evicted_block> * evicted);
    physical_block *   pop_front_plain_gpu();
    physical_block_ids pop_front_plain_n_gpu(uint32_t num_blocks);

  public:
    void init(uint32_t n_gpu, uint32_t n_cpu, float watermark);

    // "free" keeps the historical statistics meaning: blocks that are free
    // AND not cache-resident. `allocatable` is the whole queue, including
    // cache-resident blocks that allocation is allowed to evict lazily.
    size_t n_free_gpu_blocks() const;
    size_t n_free_cpu_blocks() const;
    size_t n_allocatable_gpu_blocks() const;
    size_t n_allocatable_cpu_blocks() const;
    size_t n_cached_gpu_blocks() const;

    // Number of GPU free-queue nodes with has_hash == false (O(1) from the
    // n_free_gpu / n_cached_gpu counters).
    size_t n_plain_free_gpu_blocks() const;

    bool has_free_gpu_blocks(uint32_t num_requested_blocks) const;
    bool has_free_cpu_blocks(uint32_t num_requested_blocks) const;

    // Allocation. Cache-resident victims are reported through `evicted` (GPU
    // only; CPU blocks are not part of the prefix index in this phase).
    //
    // When an eviction callback is installed, GPU checkout either sacrifices
    // exactly the blocks named by the callback or fails atomically (empty
    // result, no block touched). Plain free blocks are never sacrificed.
    physical_block_ids checkout_gpu_blocks(uint32_t num_blocks,
                                           std::vector<evicted_block> * evicted = nullptr);
    physical_block_ids checkout_cpu_blocks(uint32_t num_blocks,
                                           std::vector<evicted_block> * evicted = nullptr);

    // Share an already allocated block: ref_count++, no data copy.
    // Returns false when the block is not currently allocated.
    bool add_ref_block(uint32_t id);

    // Prefix-cache reuse: a ref_count == 0 block is taken out of the free
    // queue (it stays cache-resident, the hash is kept) and becomes owned by
    // the caller. A ref_count > 0 block is simply shared.
    bool touch_block(uint32_t id);

    // 0 means the block is free (or unknown).
    uint32_t block_ref_count(uint32_t id) const;

    bool     block_has_hash(uint32_t id) const;
    uint64_t block_hash(uint32_t id) const;
    void     set_block_hash(uint32_t id, uint64_t hash);
    void     clear_block_hash(uint32_t id);

    // Add a tag to the block's tag set (deduplicated). There is no
    // primary/first-owner tag: tag() below returns the lexicographically first
    // one only as a representative for legacy single-tag callers.
    // Writing a tag on a free non-cached block is a caller bug and only
    // produces a warning; an empty tag is ignored.
    void set_tag(uint32_t id, const std::string & tag);
    void add_tag(uint32_t id, const std::string & tag);

    // Set membership over the block's tag strings.
    bool has_tag(uint32_t id, const std::string & tag) const;

    // Number of tags currently recorded on the block (0 for free/unknown ids).
    uint32_t tag_count(uint32_t id) const;

    // Name of the index-th tag in lexicographic order, nullptr when out of
    // range.
    const char * tag_at(uint32_t id, uint32_t index) const;

    // Representative tag name (lexicographically first of the set, not
    // necessarily the tag written first to this block), for legacy single-tag
    // callers. Empty string for free/unknown blocks and for blocks with no
    // tags.
    const std::string & tag(uint32_t id) const;

    void release_gpu_blocks(const physical_block_ids & freed_blocks);
    void release_cpu_blocks(const physical_block_ids & freed_blocks);

    bool is_gpu(uint32_t block) const;
};
