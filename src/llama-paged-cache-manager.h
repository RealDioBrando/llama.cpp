#pragma once

#include "llama-kv-cache-paged.h"
#include "llama-linear-state-cache.h"
#include "llama-sequence-group.h"

#include <cstdint>
#include <vector>

// Non-owning facade between the paged scheduler and the typed caches.
//
// The scheduler decides *what* should happen (admit a waiting request, swap a
// running one out, release a finished one). The facade decides *how* it is
// done: prefix matching, block reservation, checkpoint creation, capacity
// preflight and eviction. It owns no memory and must not outlive either cache.
class llama_paged_cache_manager {
  public:
    llama_paged_cache_manager(llama_kv_cache_paged *      full,
                              llama_linear_state_cache *  linear,
                              uint32_t                    block_size,
                              bool                        prefix_enabled,
                              uint32_t                    ckpt_interval = 4,
                              uint32_t                    ckpt_base     = 0);

    llama_kv_cache_paged *     full()   const { return full_cache; }
    llama_linear_state_cache * linear() const { return linear_cache; }
    uint32_t                   get_block_size() const { return block_size; }

    // Checkpoint lattice: a block boundary b is eligible when
    // b % ckpt_interval == ckpt_base % ckpt_interval (doc 5.3).
    uint32_t ckpt_interval_blocks() const { return ckpt_interval; }
    uint32_t ckpt_base_blocks() const { return ckpt_base; }

    void set_prefix_enabled(bool enabled);
    bool prefix_enabled() const;

    //
    // Prefix cache
    //

    // Find the longest reusable prefix. Never reserves anything. The returned
    // boundary is always a multiple of block_size and always leaves at least
    // one suffix token.
    llama_prefix_candidate match_prefix_candidate(const std::vector<llama_token> & tokens) const;

    // Revalidate the stored candidate (rematching when stale) and reserve the
    // full-attention blocks and the linear state block atomically. On success
    // the group's n_past / n_prompt / block_table / block_hashes are updated.
    // Returns false only when admission cannot proceed at all (e.g. no GPU
    // state block); "no reusable prefix" is success with n_past == 0.
    bool reserve_prefix_candidate(llama_sequence_group & group);

    // Reserve KV blocks for `num_tokens` additional tokens.
    bool allocate_blocks(llama_sequence_group & group, int32_t num_tokens);

    // Free one cached prefix block or demote/evict one cached checkpoint.
    bool evict_cached_blocks();

    // Called after a decode step, once n_past has been updated.
    void register_tokens(llama_sequence_group & group);

    // Create a sparse checkpoint when the current boundary is eligible.
    void maybe_create_checkpoint(llama_sequence_group & group);

    // Roll a sequence back to `n_past_new`: release full blocks past the
    // boundary and select the matching linear rollback plane. Used after a
    // speculative verify step that accepted fewer rows than it decoded.
    void truncate_sequence(llama_sequence_group & group, llama_pos n_past_new);

    //
    // Swapping
    //

    bool can_swap_out(const llama_sequence_group & group) const;
    bool can_swap_in(const llama_sequence_group & group) const;

    // Move full K/V and the active linear state together. Both directions are
    // preflighted first; on failure no half state is published.
    bool swap_out(llama_sequence_group & group);
    bool swap_in(llama_sequence_group & group);

    //
    // Release
    //

    void release_request(llama_sequence_group & group);

    //
    // Introspection
    //

    uint32_t n_free_gpu_blocks() const;
    uint32_t n_free_linear_gpu_blocks() const;
    uint32_t n_gpu_blocks() const;

    // Batch metadata consumed by the paged attention graph input.
    void set_paged_batch_info(const llama_paged_batch_info * info);

    // Per-sequence position bookkeeping (full-attention side).
    llama_pos seq_pos_min(llama_seq_id seq_id) const;
    llama_pos seq_pos_max(llama_seq_id seq_id) const;
    void      set_seq_min_pos(llama_seq_id seq_id, llama_pos pos);
    void      set_seq_max_pos(llama_seq_id seq_id, llama_pos pos);

    static uint64_t hash_block(uint64_t                      prev,
                               const std::vector<llama_token> & tokens,
                               uint32_t                        block_idx,
                               uint32_t                        block_size);

  private:
    bool candidate_is_current(const llama_prefix_candidate & c) const;
    void clear_prefix_for_full_prefill(llama_sequence_group & group);

    llama_kv_cache_paged *     full_cache   = nullptr;
    llama_linear_state_cache * linear_cache = nullptr;

    const uint32_t block_size;
    const uint32_t ckpt_interval;
    const uint32_t ckpt_base;
};
