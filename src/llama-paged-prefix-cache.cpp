#include "llama-paged-prefix-cache.h"

#include "llama-impl.h"

#include <algorithm>

void llama_paged_prefix_cache::set_ckpt_slots(int32_t base, uint32_t n) {
    ckpt_seq_base = base;
    ckpt_slots.assign(n, {});
    for (uint32_t i = 0; i < n; ++i) {
        ckpt_slots[i].ckpt_seq = base + (int32_t) i;
    }
    LLAMA_LOG("%s: reserved %d checkpoint slot(s) at seq ids [%d, %d)\n", __func__, (int) n, base, base + (int32_t) n);
}

bool llama_paged_prefix_cache::seq_reserved(int32_t seq_id) const {
    if (ckpt_seq_base < 0 || ckpt_slots.empty()) {
        return false;
    }
    return seq_id >= ckpt_seq_base && seq_id < ckpt_seq_base + (int32_t) ckpt_slots.size();
}

const llama_paged_prefix_entry * llama_paged_prefix_cache::find_block(uint64_t hash, llama_token first_tok,
                                                                      llama_token last_tok) {
    auto it = blocks.find(hash);
    if (it == blocks.end()) {
        return nullptr;
    }
    if (it->second.first_tok != first_tok || it->second.last_tok != last_tok) {
        return nullptr;
    }
    it->second.last_used = ggml_time_us();
    return &it->second;
}

void llama_paged_prefix_cache::register_block(uint64_t hash, uint32_t block_id, int32_t owner_seq, uint32_t block_idx,
                                              llama_token first_tok, llama_token last_tok) {
    auto & e = blocks[hash];
    e.hash      = hash;
    e.block_id  = block_id;
    e.owner_seq = owner_seq;
    e.block_idx = block_idx;
    e.first_tok = first_tok;
    e.last_tok  = last_tok;
    e.last_used = ggml_time_us();
}

void llama_paged_prefix_cache::update_block_id(uint64_t hash, uint32_t new_block_id) {
    auto it = blocks.find(hash);
    if (it != blocks.end()) {
        it->second.block_id = new_block_id;
    }
}

void llama_paged_prefix_cache::set_cached_state(uint64_t hash, bool cached) {
    auto it = blocks.find(hash);
    if (it == blocks.end()) {
        return;
    }
    it->second.cached = cached;
}

std::vector<uint64_t> llama_paged_prefix_cache::cached_lru() const {
    std::vector<const llama_paged_prefix_entry *> entries;
    for (const auto & kv : blocks) {
        if (kv.second.cached) {
            entries.push_back(&kv.second);
        }
    }
    std::sort(entries.begin(), entries.end(),
              [](const llama_paged_prefix_entry * a, const llama_paged_prefix_entry * b) {
                  return a->last_used < b->last_used;
              });
    std::vector<uint64_t> hashes;
    hashes.reserve(entries.size());
    for (const auto * e : entries) {
        hashes.push_back(e->hash);
    }
    return hashes;
}

bool llama_paged_prefix_cache::pop_lru_cached(uint64_t & hash, uint32_t & block_id) {
    const auto hashes = cached_lru();
    if (hashes.empty()) {
        return false;
    }
    auto it = blocks.find(hashes.front());
    hash     = it->second.hash;
    block_id = it->second.block_id;
    it->second.cached = false;
    return true;
}

int32_t llama_paged_prefix_cache::acquire_ckpt_slot(uint64_t chain_hash, llama_pos pos) {
    if (ckpt_slots.empty()) {
        return -1;
    }

    int32_t pick = -1;
    int64_t oldest = 0;
    for (uint32_t i = 0; i < ckpt_slots.size(); ++i) {
        auto & s = ckpt_slots[i];
        if (s.used && s.chain_hash == chain_hash) {
            // Refresh the existing checkpoint for this chain.
            s.last_used = ggml_time_us();
            s.pos       = pos;
            return s.ckpt_seq;
        }
        if (!s.used) {
            pick = (int32_t) i;
            break;
        }
        if (pick < 0 || s.last_used < oldest) {
            pick   = (int32_t) i;
            oldest = s.last_used;
        }
    }

    auto & s = ckpt_slots[pick];
    s.used       = true;
    s.chain_hash = chain_hash;
    s.pos        = pos;
    s.last_used  = ggml_time_us();
    return s.ckpt_seq;
}

const llama_paged_ckpt_entry * llama_paged_prefix_cache::find_ckpt(uint64_t chain_hash) const {
    for (const auto & s : ckpt_slots) {
        if (s.used && s.chain_hash == chain_hash) {
            return &s;
        }
    }
    return nullptr;
}
