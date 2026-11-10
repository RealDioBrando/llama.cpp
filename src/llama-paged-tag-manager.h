#pragma once

#include "llama-kv-cache-paged.h"
#include "llama-linear-state-cache.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// Aggregated KV footprint of one owner tag (doc tag-kv-block-usage):
//   full_blocks   - live, reusable full-attention blocks owned by the tag;
//   linear_blocks - live GPU checkpoints owned by the tag.
// Shared references are exposed as block ids only and never counted.
struct llama_paged_tag_usage {
    uint64_t full_blocks   = 0;
    uint64_t linear_blocks = 0;
};

struct llama_paged_block_stats {
    uint64_t total_blocks = 0;
    uint64_t used_blocks  = 0;
    uint64_t free_blocks  = 0;
};

struct llama_paged_pool_stats {
    llama_paged_block_stats full;
    llama_paged_block_stats linear;
};

// Query-time tag accounting over the paged caches (doc
// docs/kvcache-manager/tag-kv-block-usage.md).
//
// Only *reusable* blocks count: live, GPU-resident full prefix-index blocks
// and live GPU checkpoints. Partial tail blocks, linear ACTIVE state blocks,
// CPU-pool blocks and swapped blocks are invisible. The owner tag itself is
// stored on the physical block by llama_block_manager; this class keeps no
// state of its own and materializes a snapshot per query:
//
//   1. owned_*    : reusable blocks grouped by their physical owner tag
//                   (llama_kv_cache_paged / llama_linear_state_cache);
//   2. shared_full: block-table entries of tagged requests that point at a
//                   reusable block owned by someone else.
//
// Because shared references are derived on demand, a request finishing only
// drops its own shared entries; the owner mapping is unaffected as long as
// the block stays cache-resident. Must be used on the scheduler thread.
class llama_paged_tag_manager {
  public:
    // Live request view needed to derive shared references. Both pointers
    // must outlive the materialize() call.
    struct request_ref {
        const std::string *     tag         = nullptr;
        const llama_block_ids * block_table = nullptr;
    };

    // One materialization of the tag -> block mapping.
    struct snapshot {
        std::unordered_map<std::string, std::unordered_set<uint32_t>> owned_full;
        std::unordered_map<std::string, std::unordered_set<uint32_t>> owned_linear;
        std::unordered_map<std::string, std::unordered_set<uint32_t>> shared_full;
    };

    // kind: 0 = full, 1 = linear; owned: true = counted, false = shared only.
    using block_cb = std::function<void(const std::string & tag, int32_t kind, uint32_t block_id, bool owned)>;

    llama_paged_tag_manager(llama_kv_cache_paged & full, llama_linear_state_cache * linear = nullptr);

    // One-shot materialization over the given live requests.
    snapshot materialize(const std::vector<request_ref> & requests) const;

    //
    // Snapshot queries. Unknown tags report zero usage / empty id lists.
    //

    static llama_paged_tag_usage usage(const snapshot & s, const std::string & tag);

    // Every tag seen in any of the three maps. Tags that only share someone
    // else's blocks appear with zero counts.
    static void all_usage(const snapshot & s, std::map<std::string, llama_paged_tag_usage> & out);

    // Ids are sorted ascending; the full and linear id spaces are independent.
    static void block_ids(const snapshot & s, const std::string & tag,
                          std::vector<uint32_t> & full_owned,
                          std::vector<uint32_t> & full_shared,
                          std::vector<uint32_t> & linear_owned);

    // Visits one tag, or every tag when `tag` is nullptr, owned before shared
    // and full before linear.
    static void for_each_block(const snapshot & s, const std::string * tag, const block_cb & cb);

    // npu pool statistics; used + free == total. The CPU pool is reserved
    // and reported as all-zero by the C API.
    llama_paged_pool_stats block_stats() const;

  private:
    llama_kv_cache_paged *     full_cache;
    llama_linear_state_cache * linear_cache;
};
