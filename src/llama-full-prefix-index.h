#pragma once

#include "llama.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

// Block-granular prefix index for the paged full-attention KV cache.
//
// The index maps a token-block hash chain to the physical block that currently
// holds the K/V data. It does not own the K/V data or the block lifetime - both
// belong to llama_kv_cache_paged. An entry is live while the pointed block is
// still tagged with the same hash (block.has_hash && block.hash == entry.hash)
// and remains on the GPU pool:
//
//   entry present
//   && block_manager.block_has_hash(entry.block_id)
//   && block_manager.block_hash(entry.block_id) == entry.hash
//   && block_manager.is_gpu(entry.block_id)
//
// A live entry may point at a free (ref_count == 0) block: that is the normal
// "cache resident" state. Checkout is the only point that invalidates an entry.
struct llama_full_prefix_entry {
    uint64_t    hash      = 0;
    uint32_t    block_id  = 0;
    int32_t     owner_seq = -1;
    uint32_t    block_idx = 0;
    llama_token first_tok = 0;  // collision sanity check
    llama_token last_tok  = 0;
};

class llama_full_prefix_index {
  public:
    void set_enabled(bool v) { enabled_ = v; }
    bool enabled() const { return enabled_; }

    // Returns nullptr when the hash is unknown. The cheap first/last token
    // sanity check guards against hash collisions; callers still have to
    // verify the block still owns the hash.
    const llama_full_prefix_entry * find(uint64_t hash, llama_token first_tok, llama_token last_tok) const;

    // Hash-only lookup (no collision sanity check).
    const llama_full_prefix_entry * find_hash(uint64_t hash) const;

    // Remove the entry only when it still points at `block_id`, so an evicted
    // or recycled copy cannot drop an entry re-registered for another block.
    bool remove_if_block(uint64_t hash, uint32_t block_id);

    void register_block(uint64_t    hash,
                        uint32_t    block_id,
                        int32_t     owner_seq,
                        uint32_t    block_idx,
                        llama_token first_tok,
                        llama_token last_tok);

    // Keep the registry pointing at the current physical block after the data
    // was copied into another block (fork/swap).
    bool update_block_id(uint64_t hash, uint32_t new_block_id);

    void remove(uint64_t hash);

    // Drop every entry that points at the given physical block.
    void invalidate_block(uint32_t block_id);

    void clear() { blocks_.clear(); }

    size_t size() const { return blocks_.size(); }

    const std::unordered_map<uint64_t, llama_full_prefix_entry> & entries() const { return blocks_; }

  private:
    bool     enabled_ = true;
    std::unordered_map<uint64_t, llama_full_prefix_entry> blocks_;
};
