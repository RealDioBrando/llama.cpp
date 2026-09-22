#pragma once

#include "llama.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

// Block-granular prefix cache index for the paged scheduler.
// Blocks are shared by reference; the caller owns block lifetime via the
// block manager ref counts. Checkpoint slots are reserved seq ids that
// hold a forkable GDN state (one recurrent cell each).
struct llama_paged_prefix_entry {
    uint64_t    hash       = 0;
    uint32_t    block_id   = 0;
    int32_t     owner_seq  = -1;
    uint32_t    block_idx  = 0;
    llama_token first_tok  = 0;
    llama_token last_tok   = 0;
    int64_t     last_used  = 0;
    bool        cached     = false;  // block kept alive by the cache after its owner finished
};

struct llama_paged_ckpt_entry {
    uint64_t  chain_hash = 0;
    int32_t   ckpt_seq   = -1;
    llama_pos pos        = -1;
    int64_t   last_used  = 0;
    bool      used       = false;
};

class llama_paged_prefix_cache {
  public:
    void set_enabled(bool v) { enabled_ = v; }
    bool enabled() const { return enabled_; }

    // Reserve seq ids [base, base + n) for GDN state checkpoints.
    void set_ckpt_slots(int32_t base, uint32_t n);
    bool seq_reserved(int32_t seq_id) const;
    uint32_t n_ckpt_slots() const { return ckpt_slots.size(); }

    // Returns nullptr when the hash is unknown or the cheap token
    // sanity check fails. Caller must verify the block is still
    // allocated and on the GPU pool before sharing it.
    const llama_paged_prefix_entry * find_block(uint64_t hash, llama_token first_tok, llama_token last_tok);

    void register_block(uint64_t hash, uint32_t block_id, int32_t owner_seq, uint32_t block_idx,
                        llama_token first_tok, llama_token last_tok);

    // Keep the registry pointing at the current physical block after a
    // swap moved it between the GPU and CPU pools.
    void update_block_id(uint64_t hash, uint32_t new_block_id);

    // Cache-resident lifecycle: entries whose owning request finished keep
    // their block alive (ref stays 1) until evicted under pressure.
    void set_cached_state(uint64_t hash, bool cached);
    std::vector<uint64_t> cached_lru() const;
    bool pop_lru_cached(uint64_t & hash, uint32_t & block_id);

    // Pick a checkpoint slot for a chain that now ends at chain_hash.
    // Returns the reserved seq id to (re)bind; the caller must seq_rm it
    // before seq_cp-ing fresh state into it.
    int32_t acquire_ckpt_slot(uint64_t chain_hash, llama_pos pos);

    const llama_paged_ckpt_entry * find_ckpt(uint64_t chain_hash) const;

  private:
    bool enabled_ = true;
    int32_t ckpt_seq_base = -1;
    std::vector<llama_paged_ckpt_entry> ckpt_slots;
    std::unordered_map<uint64_t, llama_paged_prefix_entry> blocks;
};
