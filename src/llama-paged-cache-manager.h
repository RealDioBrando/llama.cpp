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

    // GPU eviction policy installed on both pools (full-attention and, when
    // present, linear state). cb == nullptr restores the legacy FIFO path.
    // Servers register once and the facade forwards to the two caches.
    void set_evict_pick_cb(llama_paged_evict_pick_cb cb, void * ud);

    // True when a GPU allocation in the current step failed only because the
    // installed eviction picker refused (full or linear pool). The scheduler
    // uses this to distinguish policy back-pressure from capacity exhaustion.
    bool gpu_checkout_refused_by_policy() const;
    // Clears the per-step refusal flag on both caches.
    void reset_gpu_checkout_refusal();

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

    // Admission / growth entry point.
    //
    // For a group that is not admitted yet (cache_reserved == false) this is
    // the whole admission transaction: it reserves the linear ACTIVE state
    // block and the prefix-cache block refs, then grows the full-attention
    // block table for `num_tokens` additional tokens. Either everything is
    // reserved or nothing is: on failure the group keeps holding no blocks
    // and stays in the waiting queue (vLLM allocate_slots() == None).
    //
    // For an already admitted group (decode growth, fork tail block) the
    // reservation is reused and only full-attention KV is grown; failure
    // leaves the group untouched for the caller to preempt/swap.
    bool allocate_blocks(llama_sequence_group & group, int32_t num_tokens);

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
