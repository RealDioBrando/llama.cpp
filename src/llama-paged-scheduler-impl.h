#pragma once

#include "llama-kv-cache-paged.h"
#include "llama-memory-hybrid-paged.h"
#include "llama-paged-cache-manager.h"
#include "llama-paged-queue.h"
#include "llama-paged-tag-manager.h"

#include <clocale>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

enum class llama_scheduler_status {
    OK,
    DEADLOCK,  // cannot make progress
};

class llama_paged_scheduler_impl {
  public:
    llama_paged_scheduler_impl(
            uint32_t n_ctx,
            uint32_t block_sz,
            int32_t  n_batch,
            int32_t  n_pos_per_embd,
            llama_kv_cache_paged * kv_manager,
            llama_memory_hybrid_paged * hybrid_memory = nullptr,
            bool priority_enabled = false,
            uint32_t ckpt_interval = 4,
            uint32_t ckpt_base     = 0);

    llama_scheduler_status step(llama_batch & batch);
    bool                   queue_request(llama_sequence_group group);
    bool                   add_request_segs(const llama_paged_segment * segs, int32_t n_segs, int32_t request_id);
    bool                   append_segment(int32_t request_id, const llama_paged_segment & seg);
    void update(const llama_batch * batch, const llama_paged_feedback * feedback, int32_t n_feedback);
    void set_on_finish(llama_paged_on_finish_cb cb, void * user_data);
    // Install (or, with cb == nullptr, uninstall) the GPU eviction picker on
    // both paged caches. Must run on the scheduler thread, not during step().
    void set_evict_decision(llama_paged_evict_pick_cb cb, void * user_data);
    bool cancel(int32_t request_id);
    bool fork_request(int32_t src_id, int32_t dst_id);
    bool attach_draft(int32_t request_id, const llama_token * draft, int32_t n_draft);
    bool truncate(int32_t request_id, int32_t n_past_new);
    int32_t max_request_id() const;
    llama_sequence_group *         get_group_from_id(int32_t request_id) const;
    int32_t                        get_n_pos_per_embd() const { return n_pos_per_embd; }
    const llama_paged_batch_info * get_curr_batch_info() const;
    bool set_prefix_cache_enabled(bool enabled);

    //
    // Tag accounting (doc tag-kv-block-usage). All queries must run on the
    // scheduler thread; nothing here takes a lock.
    //

    // Attach an owner tag (uint64, 0 = none) to a not-yet-admitted request.
    // Returns false when the request is unknown or already holds blocks / a
    // cache reservation (tagging after the first schedule step is rejected).
    bool set_request_tag(int32_t request_id, uint64_t tag);

    // Per-tag usage derived from owned blocks only. Tags that merely share
    // someone else's blocks appear with zero counts.
    void all_tag_block_usage(std::map<uint64_t, llama_paged_tag_usage> & out) const;
    bool tag_block_usage(uint64_t tag, llama_paged_tag_usage & out) const;

    // Block id detail for one tag. Ids are sorted ascending and the two id
    // spaces (full / linear) are independent.
    bool tag_blocks(uint64_t tag,
                    std::vector<uint32_t> & full_owned,
                    std::vector<uint32_t> & full_shared,
                    std::vector<uint32_t> & linear_owned) const;

    // Visit (tag, kind, block_id, owned) for one tag or all tags when `tag`
    // is nullptr. kind: 0 = full, 1 = linear.
    void for_each_tag_block(const uint64_t * tag, const llama_paged_tag_manager::block_cb & cb) const;

    void kv_cache_block_stats(llama_paged_pool_stats & out) const;

  private:
    // Snapshot of the tag -> block mapping for all live requests. The tag
    // aggregation itself lives in llama_paged_tag_manager; the scheduler only
    // contributes the active block tables.
    llama_paged_tag_manager::snapshot materialize_tag_blocks() const;

    void insert_sorted_by_arrival_time(llama_sequence_group_ptr new_group, llama_sequence_group_list & list);

    // Remove and release-ownership the group with this request_id, or nullptr.
    static llama_sequence_group_ptr take_from_list(llama_sequence_group_list & list, int32_t request_id);

    bool check_deadlock(uint32_t n_candidates, uint32_t n_swapped, uint32_t n_waiting) const;
    bool check_livelock(uint32_t n_swapped, uint32_t prev_n_swapped);

    void set_running(llama_sequence_group_ptr group);
    void set_swapped(llama_sequence_group_ptr group);
    void set_waiting(llama_sequence_group_ptr group);

    void finish(llama_sequence_group & group);

    int32_t get_curr_decode_tokens() const;

    // Number of prefill tokens to schedule for this group this step. Hybrid
    // requests stop at the next checkpoint lattice boundary so that the state
    // at that boundary can be checkpointed and reused by another request
    // (doc 5.3.4).
    int32_t prefill_chunk_tokens(const llama_sequence_group & group) const;

    // Evict one group from `running` and return it (may be nullptr if every
    // group is FINISHED). Priority mode picks the lowest priority, FCFS the
    // most recent arrival. Groups already added to this step's candidates are
    // eligible; the caller removes them from `candidates` before eviction.
    llama_sequence_group_ptr evict();
    void process_running_list(llama_sequence_group_raw_list & candidates, int32_t & row_budget);
    void process_swapped_list(llama_sequence_group_raw_list & candidates, int32_t & row_budget);
    void process_waiting_list(llama_sequence_group_raw_list & candidates, int32_t row_budget);

    void swap_out_or_recompute(llama_sequence_group_ptr group_ptr);

    int32_t calculate_global_slot_index(int32_t token_pos, std::vector<uint32_t> & block_table);

    void clear_batch(llama_batch & batch);
    void populate_batch_from(const llama_sequence_group_raw_list & candidates, llama_batch & batch);

    // Scheduling mode is fixed at construction time (see enable_priority).
    const bool priority_enabled_;
    // Monotonic enqueue sequence, assigned in queue_request().
    uint64_t next_arrival_seq = 0;
    // Only warn once when a priority is supplied in FCFS mode.
    bool warned_priority_ignored = false;

    // running stays arrival-ordered: allocator order and eviction depend on it.
    llama_sequence_group_list running;
    // waiting / swapped use the strategy selected by priority_enabled_.
    std::unique_ptr<llama_paged_queue> swapped;
    std::unique_ptr<llama_paged_queue> waiting;

    // Used for fast lookups
    std::unordered_map<int32_t, llama_sequence_group *> id_to_group;

    const uint32_t         n_seq_max_ctx;
    const uint32_t         block_size;
    const int32_t          n_batch;
    const int32_t          n_pos_per_embd;  // 1, or 4 for mrope models (9.4)
    llama_paged_batch_info curr_info = {};

    // Owns no cache; binds the full-attention cache and the optional linear
    // state cache and centralizes matching, reservation and eviction.
    std::unique_ptr<llama_paged_cache_manager> cache_manager;
    // Query-time tag accounting over the same two caches.
    std::unique_ptr<llama_paged_tag_manager> tag_manager;
    bool prefix_cache_enabled = true;

    // mrope models take a sectioned position array ([t][y][x][z]); owned by
    // populate_batch_from for the lifetime of the built batch
    std::vector<llama_pos> populate_pos;

    // embd batches: the init-allocated token array of the built batch
    // (null-detached so llama_batch_free skips it); freed in clear_batch
    llama_token * populate_token_buf = nullptr;

    uint32_t n_livelock_steps   = 0;
    uint32_t prev_n_swapped     = 0;
    uint32_t max_livelock_steps = 20;

    // Callback for output tracking
    llama_paged_on_finish_cb on_finish_cb        = nullptr;
    void *                   on_finish_user_data = nullptr;
};
