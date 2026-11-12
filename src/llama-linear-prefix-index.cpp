#include "llama-linear-prefix-index.h"

#include <algorithm>

bool llama_linear_prefix_index::insert(uint64_t  chain_hash,
                                       uint32_t  block_id,
                                       llama_pos pos,
                                       uint32_t * old_block_id) {
    if (old_block_id) {
        *old_block_id = 0;
    }
    const auto it = entries_.find(chain_hash);
    if (it != entries_.end()) {
        if (old_block_id) {
            *old_block_id = it->second.block_id;
        }
        entries_.erase(it);
    }

    llama_linear_prefix_entry e;
    e.chain_hash = chain_hash;
    e.block_id   = block_id;
    e.pos        = pos;
    e.last_used  = ++use_seq_;
    entries_[chain_hash] = e;
    return true;
}

const llama_linear_prefix_entry * llama_linear_prefix_index::find(uint64_t chain_hash) const {
    const auto it = entries_.find(chain_hash);
    return it == entries_.end() ? nullptr : &it->second;
}

const llama_linear_prefix_entry * llama_linear_prefix_index::find_deepest(
        const std::vector<uint64_t> & chain_hashes, uint32_t max_blocks) const {
    const uint32_t n = std::min<uint32_t>(max_blocks, (uint32_t) chain_hashes.size());
    for (uint32_t b = n; b > 0; --b) {
        const auto it = entries_.find(chain_hashes[b - 1]);
        if (it != entries_.end()) {
            return &it->second;
        }
    }
    return nullptr;
}

bool llama_linear_prefix_index::remove(uint64_t chain_hash) {
    return entries_.erase(chain_hash) > 0;
}

bool llama_linear_prefix_index::remove_if_block(uint64_t chain_hash, uint32_t block_id) {
    const auto it = entries_.find(chain_hash);
    if (it == entries_.end() || it->second.block_id != block_id) {
        return false;
    }
    entries_.erase(it);
    return true;
}

void llama_linear_prefix_index::invalidate_block(uint32_t block_id) {
    for (auto it = entries_.begin(); it != entries_.end();) {
        if (it->second.block_id == block_id) {
            it = entries_.erase(it);
        } else {
            ++it;
        }
    }
}

bool llama_linear_prefix_index::pop_lru(uint64_t & chain_hash, uint32_t & block_id) {
    const llama_linear_prefix_entry * victim = nullptr;
    for (const auto & [h, e] : entries_) {
        if (!victim || e.last_used < victim->last_used) {
            victim = &e;
        }
    }
    if (!victim) {
        return false;
    }
    chain_hash = victim->chain_hash;
    block_id   = victim->block_id;
    entries_.erase(chain_hash);
    return true;
}

void llama_linear_prefix_index::touch(uint64_t chain_hash) {
    const auto it = entries_.find(chain_hash);
    if (it != entries_.end()) {
        it->second.last_used = ++use_seq_;
    }
}
