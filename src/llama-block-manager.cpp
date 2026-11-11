#include "llama-block-manager.h"

#include "llama-impl.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <mutex>
#include <unordered_map>

namespace {

// Process-wide tag interner. Tag strings are mapped to small uint32_t ids so
// that per-block tag sets are plain sorted integer vectors: an eviction policy
// resolves its tag(s) once, then queries blocks with a binary search instead of
// repeated string comparisons. Ids start at 1 (0 = no tag) and never change;
// names are stored in a deque so c_str() pointers stay stable for the process
// lifetime and can be handed out through the C API.
class tag_interner {
  public:
    uint32_t intern(const std::string & tag) {
        if (tag.empty()) {
            return 0;
        }
        std::lock_guard<std::mutex> lock(mu);
        const auto it = ids.find(tag);
        if (it != ids.end()) {
            return it->second;
        }
        names.push_back(tag);
        const uint32_t tag_id = (uint32_t) names.size();
        ids.emplace(names.back(), tag_id);
        return tag_id;
    }

    uint32_t find(const std::string & tag) const {
        if (tag.empty()) {
            return 0;
        }
        std::lock_guard<std::mutex> lock(mu);
        const auto it = ids.find(tag);
        return it == ids.end() ? 0 : it->second;
    }

    const std::string & name(uint32_t tag_id) const {
        static const std::string empty;
        if (tag_id == 0) {
            return empty;
        }
        std::lock_guard<std::mutex> lock(mu);
        if (tag_id > names.size()) {
            return empty;
        }
        return names[tag_id - 1];
    }

  private:
    mutable std::mutex                        mu;
    std::unordered_map<std::string, uint32_t> ids;
    std::deque<std::string>                   names;   // id - 1 indexes this
};

// One instance shared by all pools: tag ids are process-wide and stable.
tag_interner g_tag_interner;

} // namespace

void llama_block_manager::init(uint32_t n_gpu, uint32_t n_cpu, float watermark) {
    LLAMA_LOG_INFO("%s: Block manager initialized: n_free_gpu_blocks=%d, n_free_cpu_blocks=%d\n", __func__, n_gpu,
                   n_cpu);

    // Make init idempotent: callers may reuse an instance to reset the pools.
    // NOTE: registry nodes are exposed as raw pointers (internal free-list
    // links and the llama_paged_free_block C API), therefore the vectors must
    // only ever be resized here - never while a node pointer is alive.
    gpu_registry.clear();
    cpu_registry.clear();
    free_head_gpu = free_tail_gpu = nullptr;
    free_head_cpu = free_tail_cpu = nullptr;
    n_free_gpu = n_free_cpu = 0;
    n_cached_gpu = n_cached_cpu = 0;
    gpu_checkout_refused_by_policy_ = false;

    total_num_gpu_blocks = n_gpu;
    total_num_cpu_blocks = n_cpu;

    watermark_gpu_safety_num_blocks = std::ceil(total_num_gpu_blocks * watermark);
    watermark_cpu_safety_num_blocks = std::ceil(total_num_cpu_blocks * watermark);

    // The only resize: addresses of &gpu_registry[i] / &cpu_registry[i] stay
    // valid (and thus free-list pointers and callback-visible nodes stay valid)
    // until the next init().
    gpu_registry.resize(n_gpu);
    cpu_registry.resize(n_cpu);

    // Ascending ids -> the queue head is block 0. The cache releases a
    // request's blocks in reverse order, so the chain tail ends up at the head
    // and is evicted first.
    for (uint32_t i = 0; i < n_gpu; ++i) {
        gpu_registry[i].id     = i;
        gpu_registry[i].is_gpu = true;
        queue_push_back(&gpu_registry[i]);
    }
    for (uint32_t i = 0; i < n_cpu; ++i) {
        const uint32_t id      = i + total_num_gpu_blocks;
        cpu_registry[i].id     = id;
        cpu_registry[i].is_gpu = false;
        queue_push_back(&cpu_registry[i]);
    }
}

void llama_block_manager::set_evict_pick_cb(llama_paged_evict_pick_cb cb, void * ud) {
    // cb == nullptr restores the legacy FIFO path (no callback installed).
    evict_pick_cb = cb;
    evict_pick_ud = ud;
}

void llama_block_manager::set_pool_kind(llama_paged_pool_kind kind) {
    pool_kind = kind;
}

bool llama_block_manager::gpu_checkout_refused_by_policy() const {
    return gpu_checkout_refused_by_policy_;
}

void llama_block_manager::reset_gpu_checkout_refusal() {
    gpu_checkout_refused_by_policy_ = false;
}

llama_block_manager::physical_block * llama_block_manager::block_at(uint32_t id) {
    if (id < total_num_gpu_blocks) {
        return &gpu_registry[id];
    }
    const uint32_t cpu_id = id - total_num_gpu_blocks;
    if (cpu_id >= total_num_cpu_blocks) {
        return nullptr;
    }
    return &cpu_registry[cpu_id];
}

const llama_block_manager::physical_block * llama_block_manager::block_at(uint32_t id) const {
    return const_cast<llama_block_manager *>(this)->block_at(id);
}

void llama_block_manager::queue_push_back(physical_block * b) {
    GGML_ASSERT(b && !b->in_free);
    b->in_free = true;
    b->next    = nullptr;
    b->prev    = b->is_gpu ? free_tail_gpu : free_tail_cpu;
    if (b->prev) {
        b->prev->next = b;
    } else if (b->is_gpu) {
        free_head_gpu = b;
    } else {
        free_head_cpu = b;
    }
    if (b->is_gpu) {
        free_tail_gpu = b;
        n_free_gpu++;
    } else {
        free_tail_cpu = b;
        n_free_cpu++;
    }
}

llama_block_manager::physical_block * llama_block_manager::queue_pop_front(bool gpu) {
    physical_block * & head = gpu ? free_head_gpu : free_head_cpu;
    physical_block * & tail = gpu ? free_tail_gpu : free_tail_cpu;
    GGML_ASSERT(head);
    physical_block * b = head;
    head = b->next;
    if (head) {
        head->prev = nullptr;
    } else {
        tail = nullptr;
    }
    b->in_free = false;
    b->prev    = nullptr;
    b->next    = nullptr;
    if (gpu) {
        n_free_gpu--;
    } else {
        n_free_cpu--;
    }
    return b;
}

void llama_block_manager::queue_remove(physical_block * b) {
    GGML_ASSERT(b && b->in_free);
    if (b->prev) {
        b->prev->next = b->next;
    } else if (b->is_gpu) {
        free_head_gpu = b->next;
    } else {
        free_head_cpu = b->next;
    }
    if (b->next) {
        b->next->prev = b->prev;
    } else if (b->is_gpu) {
        free_tail_gpu = b->prev;
    } else {
        free_tail_cpu = b->prev;
    }
    b->in_free = false;
    b->prev    = nullptr;
    b->next    = nullptr;
    if (b->is_gpu) {
        n_free_gpu--;
    } else {
        n_free_cpu--;
    }
}

size_t llama_block_manager::n_free_gpu_blocks() const {
    GGML_ASSERT(n_cached_gpu <= n_free_gpu);
    return n_free_gpu - n_cached_gpu;
}

size_t llama_block_manager::n_free_cpu_blocks() const {
    GGML_ASSERT(n_cached_cpu <= n_free_cpu);
    return n_free_cpu - n_cached_cpu;
}

size_t llama_block_manager::n_allocatable_gpu_blocks() const {
    return n_free_gpu;
}

size_t llama_block_manager::n_allocatable_cpu_blocks() const {
    return n_free_cpu;
}

size_t llama_block_manager::n_cached_gpu_blocks() const {
    return n_cached_gpu;
}

size_t llama_block_manager::n_plain_free_gpu_blocks() const {
    // O(1): every queued GPU node is counted in n_free_gpu, and exactly the
    // cache-resident ones (has_hash == true, all of them queued) are counted in
    // n_cached_gpu. The plain block count is therefore the difference; no queue
    // walk here, so an installed policy costs nothing while plain blocks suffice.
    GGML_ASSERT(n_cached_gpu <= n_free_gpu);
    return n_free_gpu - n_cached_gpu;
}

bool llama_block_manager::has_free_gpu_blocks(uint32_t num_requested_blocks) const {
    const size_t curr_free_gpus = n_free_gpu;
    if (curr_free_gpus < watermark_gpu_safety_num_blocks) {
        return false;
    }
    return num_requested_blocks <= (curr_free_gpus - watermark_gpu_safety_num_blocks);
}

bool llama_block_manager::has_free_cpu_blocks(uint32_t num_requested_blocks) const {
    const size_t curr_free_cpus = n_free_cpu;
    if (curr_free_cpus < watermark_cpu_safety_num_blocks) {
        return false;
    }
    return num_requested_blocks <= (curr_free_cpus - watermark_cpu_safety_num_blocks);
}

llama_block_manager::physical_block_ids
llama_block_manager::pop_front_n_gpu(uint32_t num_blocks, std::vector<evicted_block> * evicted) {
    physical_block_ids new_ids = {};
    new_ids.reserve(num_blocks);
    for (uint32_t i = 0; i < num_blocks; ++i) {
        physical_block * b = queue_pop_front(/*gpu=*/true);

        // Lazy eviction: the block's data is still valid but the pool needs
        // it back. Report the hash so the owner can drop the index entry.
        if (b->has_hash) {
            if (evicted) {
                evicted->push_back({ b->hash, b->id });
            }
            b->has_hash = false;
            b->hash     = 0;
            clear_tags(*b);
            n_cached_gpu--;
        }
        b->ref_count += 1;
        new_ids.push_back(b->id);
    }
    return new_ids;
}

llama_block_manager::physical_block * llama_block_manager::pop_front_plain_gpu() {
    // Never touch cache-resident blocks here: callers only reach this helper
    // when a plain (non-cached) block is known to be available.
    physical_block * b = free_head_gpu;
    while (b && b->has_hash) {
        b = b->next;
    }
    GGML_ASSERT(b && "pop_front_plain_gpu: no plain free GPU block");
    queue_remove(b);
    // Defensive: a plain (non-cached) block must not carry any tags.
    clear_tags(*b);
    b->ref_count += 1;
    return b;
}

llama_block_manager::physical_block_ids llama_block_manager::pop_front_plain_n_gpu(uint32_t num_blocks) {
    physical_block_ids new_ids = {};
    new_ids.reserve(num_blocks);
    for (uint32_t i = 0; i < num_blocks; ++i) {
        new_ids.push_back(pop_front_plain_gpu()->id);
    }
    return new_ids;
}

llama_block_manager::physical_block_ids
llama_block_manager::checkout_gpu_blocks(uint32_t num_blocks, std::vector<evicted_block> * evicted) {
    if (!has_free_gpu_blocks(num_blocks)) {
        return {};
    }

    // No policy installed -> legacy FIFO, byte-identical to the old behaviour.
    if (evict_pick_cb == nullptr) {
        return pop_front_n_gpu(num_blocks, evicted);
    }

    // 1. Plain free blocks are never sacrificed: only the shortfall beyond them
    //    must come from cache-resident blocks named by the callback.
    const uint32_t n_plain     = (uint32_t) n_plain_free_gpu_blocks();
    const uint32_t n_sacrifice = num_blocks > n_plain ? num_blocks - n_plain : 0;
    if (n_sacrifice == 0) {
        // Enough plain blocks: take only those. Do NOT fall back to the legacy
        // pop path - the queue head may be a cache-resident block and popping
        // it would bypass the policy.
        return pop_front_plain_n_gpu(num_blocks);
    }
    if (n_cached_gpu < n_sacrifice) {
        return {};   // the pool itself cannot satisfy the request
    }

    // 2. Zero copy: hand the real free-list head to the user for traversal.
    //    The capacity may exceed n_sacrifice so a policy can deliberately
    //    sacrifice a larger batch.
    const uint32_t out_capacity = std::max(n_sacrifice, (uint32_t) n_cached_gpu);
    std::vector<uint32_t> out(out_capacity);
    const uint32_t k = evict_pick_cb(evict_pick_ud, pool_kind, n_sacrifice, free_head_gpu, out.data(), out_capacity);
    if (k > out_capacity) {
        // Contract violation on user-supplied output: the callback must return
        // the number of ids it wrote, never more than out_capacity. Fail the
        // allocation atomically instead of aborting the process (a third-party
        // policy bug must not take the server down).
        LLAMA_LOG_WARN("%s: eviction picker returned %u ids with capacity %u; failing the allocation\n", __func__, k,
                       out_capacity);
        gpu_checkout_refused_by_policy_ = true;
        return {};
    }
    if (k == 0) {
        gpu_checkout_refused_by_policy_ = true;
        return {};   // the policy allows no sacrifice -> atomic failure
    }

    // 3. Validate + dedup BEFORE any mutation: accept only ids that are
    //    currently cache-resident GPU blocks sitting in the free queue.
    //    Duplicates are rejected because they would make the count check pass
    //    while fewer blocks than needed actually get removed.
    std::vector<uint32_t> victims;
    victims.reserve(k);
    for (uint32_t i = 0; i < k; ++i) {
        const physical_block * b = block_at(out[i]);
        if (!b || !b->is_gpu || !b->in_free || !b->has_hash) {
            continue;   // out of range / CPU block / not queued / plain block
        }
        if (std::find(victims.begin(), victims.end(), b->id) != victims.end()) {
            continue;   // duplicate id (nothing removed yet)
        }
        victims.push_back(b->id);
    }
    if (victims.size() < n_sacrifice) {
        // Not enough valid victims: the policy refused this allocation.
        gpu_checkout_refused_by_policy_ = true;
        return {};   // atomic failure, nothing changed
    }

    // 4. Sacrifice in one shot: unlink + drop hash/tag + report + take a ref.
    physical_block_ids out_ids;
    out_ids.reserve(victims.size());
    for (const uint32_t id : victims) {
        physical_block * b = block_at(id);
        GGML_ASSERT(b && b->in_free && b->has_hash);
        queue_remove(b);
        if (evicted) {
            evicted->push_back({ b->hash, b->id });
        }
        b->has_hash = false;
        b->hash     = 0;
        clear_tags(*b);
        GGML_ASSERT(n_cached_gpu > 0);
        n_cached_gpu--;
        b->ref_count += 1;
        out_ids.push_back(b->id);
    }
    GGML_ASSERT(out_ids.size() >= n_sacrifice);

    // Blocks sacrificed beyond the current need (k > n_sacrifice) are really
    // evicted: release them so they become plain free blocks at the queue tail
    // and cannot be reused as prefix-cache hits anymore.
    for (size_t i = n_sacrifice; i < out_ids.size(); ++i) {
        release_gpu_blocks({ out_ids[i] });
    }
    out_ids.resize(n_sacrifice);

    // 5. Fill the remaining slots with plain free blocks only: cache-resident
    //    blocks the policy did not name must stay untouched.
    while (out_ids.size() < num_blocks) {
        out_ids.push_back(pop_front_plain_gpu()->id);
    }
    return out_ids;
}

llama_block_manager::physical_block_ids
llama_block_manager::checkout_cpu_blocks(uint32_t num_blocks, std::vector<evicted_block> * evicted) {
    physical_block_ids new_ids = {};
    if (!has_free_cpu_blocks(num_blocks)) {
        return new_ids;
    }

    new_ids.reserve(num_blocks);
    for (uint32_t i = 0; i < num_blocks; ++i) {
        physical_block * b = queue_pop_front(/*gpu=*/false);

        if (b->has_hash) {
            if (evicted) {
                evicted->push_back({ b->hash, b->id });
            }
            b->has_hash = false;
            b->hash     = 0;
            clear_tags(*b);
            n_cached_cpu--;
        }
        b->ref_count += 1;
        new_ids.push_back(b->id);
    }
    return new_ids;
}

void llama_block_manager::release_gpu_blocks(const physical_block_ids & freed_blocks_ids) {
    // Reverse iteration is load-bearing: the cache releases a request's block
    // table in order, and pushing it back-to-front leaves the chain tail at the
    // free-queue head so eviction preserves the longest prefix.
    for (auto it = freed_blocks_ids.rbegin(); it != freed_blocks_ids.rend(); ++it) {
        const uint32_t id = *it;
        if (id >= total_num_gpu_blocks) {
            LLAMA_LOG_ERROR("%s: release of invalid GPU block %u\n", __func__, id);
            continue;
        }
        physical_block & b = gpu_registry[id];
        if (b.ref_count == 0) {
            LLAMA_LOG_ERROR("%s: release of free/invalid GPU block %u\n", __func__, id);
            continue;
        }
        b.ref_count -= 1;
        if (b.ref_count == 0) {
            b.ref_count = 0;
            if (!b.has_hash) {
                clear_tags(b);
            } else {
                n_cached_gpu++;
            }
            queue_push_back(&b);
        }
    }
}

void llama_block_manager::release_cpu_blocks(const physical_block_ids & freed_blocks_ids) {
    for (auto it = freed_blocks_ids.rbegin(); it != freed_blocks_ids.rend(); ++it) {
        const uint32_t id = *it;
        if (id < total_num_gpu_blocks || id - total_num_gpu_blocks >= total_num_cpu_blocks) {
            LLAMA_LOG_ERROR("%s: release of invalid CPU block %u\n", __func__, id);
            continue;
        }
        physical_block & b = cpu_registry[id - total_num_gpu_blocks];
        if (b.ref_count == 0) {
            LLAMA_LOG_ERROR("%s: release of free/invalid CPU block %u\n", __func__, id);
            continue;
        }
        b.ref_count -= 1;
        if (b.ref_count == 0) {
            b.ref_count = 0;
            if (!b.has_hash) {
                clear_tags(b);
            } else {
                n_cached_cpu++;
            }
            queue_push_back(&b);
        }
    }
}

bool llama_block_manager::is_gpu(uint32_t block_id) const {
    return block_id < total_num_gpu_blocks;
}

bool llama_block_manager::add_ref_block(uint32_t id) {
    physical_block * b = block_at(id);
    if (!b || b->ref_count == 0) {
        return false;
    }
    b->ref_count += 1;
    return true;
}

bool llama_block_manager::touch_block(uint32_t id) {
    physical_block * b = block_at(id);
    if (!b) {
        return false;
    }
    if (b->ref_count == 0) {
        if (!b->in_free) {
            return false;
        }
        queue_remove(b);
        if (b->has_hash) {
            if (b->is_gpu) {
                n_cached_gpu--;
            } else {
                n_cached_cpu--;
            }
        }
    }
    b->ref_count += 1;
    return true;
}

uint32_t llama_block_manager::block_ref_count(uint32_t id) const {
    const physical_block * b = block_at(id);
    return b ? b->ref_count : 0;
}

bool llama_block_manager::block_has_hash(uint32_t id) const {
    const physical_block * b = block_at(id);
    return b && b->has_hash;
}

uint64_t llama_block_manager::block_hash(uint32_t id) const {
    const physical_block * b = block_at(id);
    return b ? b->hash : 0;
}

void llama_block_manager::set_block_hash(uint32_t id, uint64_t hash) {
    physical_block * b = block_at(id);
    GGML_ASSERT(b && "set_block_hash on invalid block");
    GGML_ASSERT(b->ref_count > 0 || b->in_free);
    GGML_ASSERT(!b->has_hash || b->hash == hash);
    if (b->in_free && !b->has_hash) {
        // Hashing a block that is already in the free queue makes it a
        // cache-resident block; keep the cached counters in sync with the
        // checkout/touch/clear decrements.
        if (b->is_gpu) {
            n_cached_gpu++;
        } else {
            n_cached_cpu++;
        }
    }
    b->has_hash = true;
    b->hash     = hash;
}

void llama_block_manager::clear_block_hash(uint32_t id) {
    physical_block * b = block_at(id);
    if (!b || !b->has_hash) {
        return;
    }
    if (b->in_free) {
        if (b->is_gpu) {
            n_cached_gpu--;
        } else {
            n_cached_cpu--;
        }
    }
    b->has_hash = false;
    b->hash     = 0;
    clear_tags(*b);
}

void llama_block_manager::clear_tags(physical_block & b) {
    b.tag_ids.clear();
}

void llama_block_manager::add_tag(uint32_t id, const std::string & tag) {
    physical_block * b = block_at(id);
    if (!b) {
        LLAMA_LOG_WARN("%s: tag write on invalid block %u\n", __func__, id);
        return;
    }
    if (tag.empty()) {
        return;   // empty tags are never recorded
    }
    if (!b->is_gpu) {
        // The CPU pool is out of scope: it is being refactored and never takes
        // part in eviction-policy decisions. Keep CPU blocks untagged.
        return;
    }
    if (b->ref_count == 0 && !b->has_hash) {
        // Tagging a plain free block would leak the owner into a future
        // checkout. Cache-resident free blocks keep their tags instead.
        LLAMA_LOG_WARN("%s: tag '%s' rejected on free block %u\n", __func__, tag.c_str(), id);
        return;
    }
    const uint32_t tag_id = g_tag_interner.intern(tag);
    if (tag_id == 0) {
        return;
    }
    const auto it = std::lower_bound(b->tag_ids.begin(), b->tag_ids.end(), tag_id);
    if (it == b->tag_ids.end() || *it != tag_id) {
        b->tag_ids.insert(it, tag_id);
    }
}

void llama_block_manager::set_tag(uint32_t id, const std::string & tag) {
    // Multi-tag semantics: set_tag never replaces the accumulated set; it only
    // adds to it.
    add_tag(id, tag);
}

bool llama_block_manager::has_tag_id(uint32_t id, uint32_t tag_id) const {
    if (tag_id == 0) {
        return false;
    }
    const physical_block * b = block_at(id);
    if (!b) {
        return false;
    }
    return std::binary_search(b->tag_ids.begin(), b->tag_ids.end(), tag_id);
}

uint32_t llama_block_manager::tag_count(uint32_t id) const {
    const physical_block * b = block_at(id);
    return b ? (uint32_t) b->tag_ids.size() : 0;
}

const char * llama_block_manager::tag_at(uint32_t id, uint32_t index) const {
    const physical_block * b = block_at(id);
    if (!b || index >= b->tag_ids.size()) {
        return nullptr;
    }
    return g_tag_interner.name(b->tag_ids[index]).c_str();
}

const std::string & llama_block_manager::tag(uint32_t id) const {
    static const std::string empty;
    const physical_block * b = block_at(id);
    if (!b || b->tag_ids.empty()) {
        return empty;
    }
    return g_tag_interner.name(b->tag_ids.front());
}

// ---------------------------------------------------------------------------
// C API accessors for the opaque llama_paged_free_block node.
//
// The declarations live in include/llama.h (extern "C"). Nodes are only valid
// during an eviction-picker callback call; the accessors just read the real
// free-list node fields, no snapshot, no allocation.
// ---------------------------------------------------------------------------

const llama_paged_free_block * llama_paged_free_block_next(const llama_paged_free_block * block) {
    return block != nullptr ? block->next : nullptr;
}

uint32_t llama_paged_free_block_id(const llama_paged_free_block * block) {
    return block != nullptr ? block->id : 0;
}

int32_t llama_paged_free_block_has_hash(const llama_paged_free_block * block) {
    return block != nullptr && block->has_hash ? 1 : 0;
}

uint64_t llama_paged_free_block_hash(const llama_paged_free_block * block) {
    return block != nullptr ? block->hash : 0;
}

const char * llama_paged_free_block_tag(const llama_paged_free_block * block) {
    if (block == nullptr) {
        return nullptr;
    }
    if (block->tag_ids.empty()) {
        return nullptr;   // matches the public contract: NULL when untagged
    }
    return g_tag_interner.name(block->tag_ids.front()).c_str();
}

uint32_t llama_paged_free_block_tag_count(const llama_paged_free_block * block) {
    return block != nullptr ? (uint32_t) block->tag_ids.size() : 0;
}

const char * llama_paged_free_block_tag_at(const llama_paged_free_block * block, uint32_t index) {
    if (block == nullptr || index >= block->tag_ids.size()) {
        return nullptr;
    }
    return g_tag_interner.name(block->tag_ids[index]).c_str();
}

int32_t llama_paged_free_block_has_tag(const llama_paged_free_block * block, const char * tag) {
    if (block == nullptr || tag == nullptr) {
        return 0;
    }
    const uint32_t tag_id = g_tag_interner.find(tag);
    if (tag_id == 0) {
        return 0;   // unknown tag: do not insert from a read-only query
    }
    return std::binary_search(block->tag_ids.begin(), block->tag_ids.end(), tag_id) ? 1 : 0;
}

int32_t llama_paged_free_block_has_tag_id(const llama_paged_free_block * block, uint32_t tag_id) {
    if (block == nullptr || tag_id == 0) {
        return 0;
    }
    return std::binary_search(block->tag_ids.begin(), block->tag_ids.end(), tag_id) ? 1 : 0;
}

uint32_t llama_paged_tag_intern(const char * tag) {
    return tag != nullptr ? g_tag_interner.intern(tag) : 0;
}

const char * llama_paged_tag_name(uint32_t tag_id) {
    const std::string & name = g_tag_interner.name(tag_id);
    return name.empty() ? nullptr : name.c_str();
}
