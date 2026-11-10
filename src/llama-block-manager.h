#pragma once

#include <cstddef>
#include <cstdint>
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
class llama_block_manager {
  public:
    // Emitted when a checkout pops a cache-resident block. The hash entry must
    // be removed from the owning prefix index before the block is reused.
    struct evicted_block {
        uint64_t hash      = 0;
        uint32_t block_id  = 0;
    };

  private:
    struct physical_block {
        uint32_t id         = 0;
        uint32_t ref_count  = 0;
        bool     is_gpu     = false;

        // Prefix-cache identity. `has_hash` is the authoritative "cache
        // resident" bit; `hash` is the chain hash registered for this block.
        bool     has_hash   = false;
        uint64_t hash       = 0;

        // Owner tag of the request that first checked this block out. It
        // survives free() while the block stays cache-resident, and is
        // cleared when the hash is dropped or a non-cached block is freed.
        std::string tag;

        // Intrusive free-queue links (global block ids; -1 = none).
        bool    in_free = false;
        int32_t prev    = -1;
        int32_t next    = -1;
    };

    using physical_block_bool = std::vector<physical_block>;
    using physical_block_ids  = std::vector<uint32_t>;

    physical_block_bool gpu_registry;
    physical_block_bool cpu_registry;

    // Free queues (FIFO): pop_front for allocation, push_back for release.
    int32_t free_head_gpu = -1;
    int32_t free_tail_gpu = -1;
    int32_t free_head_cpu = -1;
    int32_t free_tail_cpu = -1;

    size_t n_free_gpu = 0;   // queue length (includes cache-resident blocks)
    size_t n_free_cpu = 0;
    size_t n_cached_gpu = 0; // subset of n_free_gpu with has_hash == true
    size_t n_cached_cpu = 0;

    uint32_t watermark_gpu_safety_num_blocks;
    uint32_t watermark_cpu_safety_num_blocks;

    uint32_t total_num_gpu_blocks;
    uint32_t total_num_cpu_blocks;

    physical_block *       block_at(uint32_t id);
    const physical_block * block_at(uint32_t id) const;

    void     queue_push_back(uint32_t id);
    uint32_t queue_pop_front(bool gpu);
    void     queue_remove(uint32_t id);

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

    bool has_free_gpu_blocks(uint32_t num_requested_blocks) const;
    bool has_free_cpu_blocks(uint32_t num_requested_blocks) const;

    // Allocation. Cache-resident victims are reported through `evicted` (GPU
    // only; CPU blocks are not part of the prefix index in this phase).
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

    // Owner tag of the block that first checked it out. Writing a tag on a
    // free non-cached block is a caller bug and only produces a warning.
    void set_tag(uint32_t id, const std::string & tag);

    // Returns an empty string for free/unknown blocks and out-of-range ids.
    const std::string & tag(uint32_t id) const;

    void release_gpu_blocks(const physical_block_ids & freed_blocks);
    void release_cpu_blocks(const physical_block_ids & freed_blocks);

    bool is_gpu(uint32_t block) const;
};
