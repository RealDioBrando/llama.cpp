#pragma once

#include "llama.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

// Block-granular prefix index for the paged full-attention KV cache.
//
// The index only maps a token-block hash chain to the physical block that
// currently holds the K/V data, plus the generation of that block. It does
// not own the K/V data or the block lifetime - both belong to
// llama_kv_cache_paged. Callers must validate an entry before using it:
//
//   entry present
//   && entry->generation == block_manager.block_generation(entry->block_id)
//   && block is still allocated and on the GPU pool
//
// Checking the generation closes the stale-hit window opened when a block is
// released and later reused for unrelated tokens.
struct llama_full_prefix_entry {
    uint64_t    hash       = 0;
    uint32_t    block_id   = 0;
    uint32_t    generation = 0;
    int32_t     owner_seq  = -1;
    uint32_t    block_idx  = 0;
    llama_token first_tok  = 0;
    llama_token last_tok   = 0;
    int64_t     last_used  = 0;
    bool        cached     = false;  // block kept alive by the cache after its owner finished
};

class llama_full_prefix_index {
  public:
    void set_enabled(bool v) { enabled_ = v; }
    bool enabled() const { return enabled_; }

    // Returns nullptr when the hash is unknown. The cheap first/last token
    // sanity check guards against hash collisions; callers still have to
    // verify the generation and the pool state.
    const llama_full_prefix_entry * find(uint64_t hash, llama_token first_tok, llama_token last_tok) const;

    // Hash-only lookup (no collision sanity check).
    const llama_full_prefix_entry * find_hash(uint64_t hash) const;

    // Remove the entry only when it still points at `block_id`, so a request
    // finishing does not drop an entry re-registered for another request.
    bool remove_if_block(uint64_t hash, uint32_t block_id);

    void register_block(uint64_t    hash,
                        uint32_t    block_id,
                        uint32_t    generation,
                        int32_t     owner_seq,
                        uint32_t    block_idx,
                        llama_token first_tok,
                        llama_token last_tok);

    // Keep the registry pointing at the current physical block after a swap
    // moved the block between the GPU and CPU pools.
    bool update_block_id(uint64_t hash, uint32_t new_block_id, uint32_t new_generation);

    // Re-point only when the entry still refers to `expected_block_id`. Used
    // by swaps so a request does not hijack an entry owned by another copy of
    // the same token chain.
    bool update_block_id_if(uint64_t hash, uint32_t expected_block_id, uint32_t new_block_id,
                            uint32_t new_generation);

    void remove(uint64_t hash);

    // Cache-resident lifecycle: entries whose owning request finished keep
    // their block alive (ref stays 1) until evicted under pressure.
    void set_cached_state(uint64_t hash, bool cached);

    bool pop_lru_cached(uint64_t & hash, uint32_t & block_id, uint32_t & generation);

    // Drop every entry that points at the given physical block.
    void invalidate_block(uint32_t block_id);

    void clear() { blocks_.clear(); }

    size_t size() const { return blocks_.size(); }

  private:
    bool     enabled_ = true;
    uint64_t use_seq_ = 0;  // monotonic LRU stamp

    std::unordered_map<uint64_t, llama_full_prefix_entry> blocks_;
};
