#include "llama-paged-tag-manager.h"

#include <algorithm>

llama_paged_tag_manager::llama_paged_tag_manager(llama_kv_cache_paged & full,
                                                 llama_linear_state_cache * linear) :
    full_cache(&full), linear_cache(linear) {
    GGML_ASSERT(full_cache && "tag manager requires a full-attention cache");
}

llama_paged_tag_manager::snapshot llama_paged_tag_manager::materialize(
        const std::vector<request_ref> & requests) const {
    snapshot s;

    // owned: reusable blocks grouped by the owner tag stored on the physical
    // block. collect_tag_blocks() applies the liveness filter (GPU-resident,
    // still owning the hash; ref_count may be 0 for cache-resident blocks).
    std::unordered_set<uint32_t> reusable_full;
    full_cache->collect_tag_blocks(s.owned_full, &reusable_full);
    if (linear_cache != nullptr) {
        linear_cache->collect_tag_blocks(s.owned_linear);
    }

    // shared: references held by tagged requests to reusable blocks that they
    // do not own. Derived from the live block tables, never counted.
    for (const request_ref & req : requests) {
        if (req.tag == nullptr || req.tag->empty() || req.block_table == nullptr) {
            continue;
        }
        const std::string & tag = *req.tag;
        const auto owned_it = s.owned_full.find(tag);
        for (const uint32_t block_id : *req.block_table) {
            if (reusable_full.count(block_id) == 0) {
                continue;  // partial tail, swapped or already released
            }
            if (owned_it != s.owned_full.end() && owned_it->second.count(block_id) > 0) {
                continue;  // owned, not shared
            }
            s.shared_full[tag].insert(block_id);
        }
    }
    return s;
}

llama_paged_tag_usage llama_paged_tag_manager::usage(const snapshot & s, const std::string & tag) {
    llama_paged_tag_usage out;
    if (const auto it = s.owned_full.find(tag); it != s.owned_full.end()) {
        out.full_blocks = it->second.size();
    }
    if (const auto it = s.owned_linear.find(tag); it != s.owned_linear.end()) {
        out.linear_blocks = it->second.size();
    }
    return out;
}

void llama_paged_tag_manager::all_usage(const snapshot & s,
                                        std::map<std::string, llama_paged_tag_usage> & out) {
    out.clear();
    auto touch = [&out](const std::string & tag) -> llama_paged_tag_usage & {
        return out[tag];
    };
    for (const auto & [tag, ids] : s.owned_full) {
        touch(tag).full_blocks = ids.size();
    }
    for (const auto & [tag, ids] : s.owned_linear) {
        touch(tag).linear_blocks = ids.size();
    }
    for (const auto & [tag, ids] : s.shared_full) {
        (void) ids;
        touch(tag);
    }
}

void llama_paged_tag_manager::block_ids(const snapshot & s, const std::string & tag,
                                        std::vector<uint32_t> & full_owned,
                                        std::vector<uint32_t> & full_shared,
                                        std::vector<uint32_t> & linear_owned) {
    full_owned.clear();
    full_shared.clear();
    linear_owned.clear();

    if (const auto it = s.owned_full.find(tag); it != s.owned_full.end()) {
        full_owned.assign(it->second.begin(), it->second.end());
    }
    if (const auto it = s.shared_full.find(tag); it != s.shared_full.end()) {
        full_shared.assign(it->second.begin(), it->second.end());
    }
    if (const auto it = s.owned_linear.find(tag); it != s.owned_linear.end()) {
        linear_owned.assign(it->second.begin(), it->second.end());
    }

    std::sort(full_owned.begin(), full_owned.end());
    std::sort(full_shared.begin(), full_shared.end());
    std::sort(linear_owned.begin(), linear_owned.end());
}

void llama_paged_tag_manager::for_each_block(const snapshot & s, const std::string * tag, const block_cb & cb) {
    auto emit = [&](const std::string & t,
                    const std::unordered_map<std::string, std::unordered_set<uint32_t>> & owned,
                    const std::unordered_map<std::string, std::unordered_set<uint32_t>> * shared,
                    int32_t kind) {
        if (const auto it = owned.find(t); it != owned.end()) {
            std::vector<uint32_t> ids(it->second.begin(), it->second.end());
            std::sort(ids.begin(), ids.end());
            for (const uint32_t id : ids) {
                cb(t, kind, id, true);
            }
        }
        if (shared != nullptr) {
            if (const auto it = shared->find(t); it != shared->end()) {
                std::vector<uint32_t> ids(it->second.begin(), it->second.end());
                std::sort(ids.begin(), ids.end());
                for (const uint32_t id : ids) {
                    cb(t, kind, id, false);
                }
            }
        }
    };

    // ordered union of every materialized tag (or just the requested one)
    std::map<std::string, int> tags;
    if (tag != nullptr) {
        tags[*tag] = 0;
    } else {
        for (const auto & [t, ids] : s.owned_full) {
            (void) ids;
            tags[t] = 0;
        }
        for (const auto & [t, ids] : s.owned_linear) {
            (void) ids;
            tags[t] = 0;
        }
        for (const auto & [t, ids] : s.shared_full) {
            (void) ids;
            tags[t] = 0;
        }
    }

    for (const auto & [t, _] : tags) {
        (void) _;
        emit(t, s.owned_full, &s.shared_full, 0);
        emit(t, s.owned_linear, nullptr, 1);
    }
}

llama_paged_pool_stats llama_paged_tag_manager::block_stats() const {
    llama_paged_pool_stats out;

    out.full.total_blocks = full_cache->get_num_gpu_blocks();
    out.full.free_blocks  = full_cache->n_free_gpu_blocks();
    out.full.cached_blocks = full_cache->n_cached_gpu_blocks();
    out.full.allocatable_blocks = full_cache->n_allocatable_gpu_blocks();
    out.full.used_blocks  = out.full.total_blocks - out.full.free_blocks;

    if (linear_cache != nullptr) {
        out.linear.total_blocks = linear_cache->get_num_gpu_blocks();
        out.linear.free_blocks  = linear_cache->n_free_gpu_blocks();
        out.linear.cached_blocks = linear_cache->n_cached_gpu_blocks();
        out.linear.allocatable_blocks = linear_cache->n_allocatable_gpu_blocks();
        out.linear.used_blocks  = out.linear.total_blocks - out.linear.free_blocks;
    }
    return out;
}
