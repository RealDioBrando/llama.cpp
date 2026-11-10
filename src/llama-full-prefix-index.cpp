#include "llama-full-prefix-index.h"

const llama_full_prefix_entry * llama_full_prefix_index::find(uint64_t         hash,
                                                              llama_token      first_tok,
                                                              llama_token      last_tok) const {
    const auto it = blocks_.find(hash);
    if (it == blocks_.end()) {
        return nullptr;
    }
    const llama_full_prefix_entry & e = it->second;
    if (e.first_tok != first_tok || e.last_tok != last_tok) {
        return nullptr;
    }
    return &e;
}

const llama_full_prefix_entry * llama_full_prefix_index::find_hash(uint64_t hash) const {
    const auto it = blocks_.find(hash);
    return it == blocks_.end() ? nullptr : &it->second;
}

bool llama_full_prefix_index::remove_if_block(uint64_t hash, uint32_t block_id) {
    const auto it = blocks_.find(hash);
    if (it == blocks_.end() || it->second.block_id != block_id) {
        return false;
    }
    blocks_.erase(it);
    return true;
}

void llama_full_prefix_index::register_block(uint64_t    hash,
                                             uint32_t    block_id,
                                             int32_t     owner_seq,
                                             uint32_t    block_idx,
                                             llama_token first_tok,
                                             llama_token last_tok) {
    llama_full_prefix_entry e;
    e.hash      = hash;
    e.block_id  = block_id;
    e.owner_seq = owner_seq;
    e.block_idx = block_idx;
    e.first_tok = first_tok;
    e.last_tok  = last_tok;
    blocks_[hash] = e;
}

bool llama_full_prefix_index::update_block_id(uint64_t hash, uint32_t new_block_id) {
    const auto it = blocks_.find(hash);
    if (it == blocks_.end()) {
        return false;
    }
    it->second.block_id = new_block_id;
    return true;
}

void llama_full_prefix_index::remove(uint64_t hash) {
    blocks_.erase(hash);
}

void llama_full_prefix_index::invalidate_block(uint32_t block_id) {
    for (auto it = blocks_.begin(); it != blocks_.end();) {
        if (it->second.block_id == block_id) {
            it = blocks_.erase(it);
        } else {
            ++it;
        }
    }
}
