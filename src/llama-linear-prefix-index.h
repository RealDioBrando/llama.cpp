#pragma once

#include "llama.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

// Checkpoint index for the linear-attention state cache.
//
// A checkpoint is a read-only copy of the complete recurrent state at a
// token-block boundary. It is keyed by the full token-block hash chain
// (including the block that ends at the boundary). The index stores the
// physical state block; liveness is validated against the linear cache's
// block manager (the block must still own the chain hash).
struct llama_linear_prefix_entry {
    uint64_t  chain_hash = 0;
    uint32_t  block_id   = 0;   // physical state block holding the checkpoint
    llama_pos pos        = -1;  // token boundary covered by the checkpoint
    int64_t   last_used  = 0;
};

class llama_linear_prefix_index {
  public:
    void set_enabled(bool v) { enabled_ = v; }
    bool enabled() const { return enabled_; }

    // Insert or replace the checkpoint for this hash chain. The previous
    // entry, if any, is returned through old_block_id so the caller can
    // release it.
    bool insert(uint64_t chain_hash, uint32_t block_id, llama_pos pos,
                uint32_t * old_block_id = nullptr);

    const llama_linear_prefix_entry * find(uint64_t chain_hash) const;

    // Deepest checkpoint whose chain hash is in `chain_hashes` and whose
    // boundary does not exceed `max_blocks` blocks. `chain_hashes[b]` must
    // be the hash chain covering blocks [0, b]. Returns nullptr when none.
    const llama_linear_prefix_entry * find_deepest(const std::vector<uint64_t> & chain_hashes,
                                                   uint32_t                     max_blocks) const;

    bool remove(uint64_t chain_hash);

    // Remove the entry only when it still points at `block_id`.
    bool remove_if_block(uint64_t chain_hash, uint32_t block_id);

    // Drop every entry pointing at the given physical block.
    void invalidate_block(uint32_t block_id);

    // LRU eviction over checkpoints, oldest first.
    bool pop_lru(uint64_t & chain_hash, uint32_t & block_id);

    void touch(uint64_t chain_hash);

    void clear() { entries_.clear(); }
    size_t size() const { return entries_.size(); }

    const std::unordered_map<uint64_t, llama_linear_prefix_entry> & entries() const { return entries_; }

  private:
    bool     enabled_ = true;
    uint64_t use_seq_ = 0;

    std::unordered_map<uint64_t, llama_linear_prefix_entry> entries_;
};
