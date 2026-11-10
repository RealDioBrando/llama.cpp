#include "llama-block-manager.h"

#include "llama-impl.h"

#include <cmath>

void llama_block_manager::init(uint32_t n_gpu, uint32_t n_cpu, float watermark) {
    LLAMA_LOG_INFO("%s: Block manager initialized: n_free_gpu_blocks=%d, n_free_cpu_blocks=%d\n", __func__, n_gpu,
                   n_cpu);

    // Make init idempotent: callers may reuse an instance to reset the pools.
    gpu_registry.clear();
    cpu_registry.clear();
    free_head_gpu = free_tail_gpu = -1;
    free_head_cpu = free_tail_cpu = -1;
    n_free_gpu = n_free_cpu = 0;
    n_cached_gpu = n_cached_cpu = 0;

    total_num_gpu_blocks = n_gpu;
    total_num_cpu_blocks = n_cpu;

    watermark_gpu_safety_num_blocks = std::ceil(total_num_gpu_blocks * watermark);
    watermark_cpu_safety_num_blocks = std::ceil(total_num_cpu_blocks * watermark);

    gpu_registry.resize(n_gpu);
    cpu_registry.resize(n_cpu);

    // Ascending ids -> the queue head is block 0. The cache releases a
    // request's blocks in reverse order, so the chain tail ends up at the head
    // and is evicted first.
    for (uint32_t i = 0; i < n_gpu; ++i) {
        gpu_registry[i].id     = i;
        gpu_registry[i].is_gpu = true;
        queue_push_back(i);
    }
    for (uint32_t i = 0; i < n_cpu; ++i) {
        const uint32_t id      = i + total_num_gpu_blocks;
        cpu_registry[i].id     = id;
        cpu_registry[i].is_gpu = false;
        queue_push_back(id);
    }
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

void llama_block_manager::queue_push_back(uint32_t id) {
    physical_block * b = block_at(id);
    GGML_ASSERT(b && !b->in_free);
    b->in_free = true;
    b->next    = -1;
    b->prev    = (b->is_gpu ? free_tail_gpu : free_tail_cpu);
    if (b->prev >= 0) {
        block_at((uint32_t) b->prev)->next = (int32_t) id;
    } else if (b->is_gpu) {
        free_head_gpu = (int32_t) id;
    } else {
        free_head_cpu = (int32_t) id;
    }
    if (b->is_gpu) {
        free_tail_gpu = (int32_t) id;
        n_free_gpu++;
    } else {
        free_tail_cpu = (int32_t) id;
        n_free_cpu++;
    }
}

uint32_t llama_block_manager::queue_pop_front(bool gpu) {
    int32_t & head = gpu ? free_head_gpu : free_head_cpu;
    int32_t & tail = gpu ? free_tail_gpu : free_tail_cpu;
    GGML_ASSERT(head >= 0);
    const uint32_t id = (uint32_t) head;
    physical_block * b = block_at(id);
    head = b->next;
    if (head >= 0) {
        block_at((uint32_t) head)->prev = -1;
    } else {
        tail = -1;
    }
    b->in_free = false;
    b->prev    = -1;
    b->next    = -1;
    if (gpu) {
        n_free_gpu--;
    } else {
        n_free_cpu--;
    }
    return id;
}

void llama_block_manager::queue_remove(uint32_t id) {
    physical_block * b = block_at(id);
    GGML_ASSERT(b && b->in_free);
    physical_block * prev = b->prev >= 0 ? block_at((uint32_t) b->prev) : nullptr;
    physical_block * next = b->next >= 0 ? block_at((uint32_t) b->next) : nullptr;
    if (prev) {
        prev->next = b->next;
    } else if (b->is_gpu) {
        free_head_gpu = b->next;
    } else {
        free_head_cpu = b->next;
    }
    if (next) {
        next->prev = b->prev;
    } else if (b->is_gpu) {
        free_tail_gpu = b->prev;
    } else {
        free_tail_cpu = b->prev;
    }
    b->in_free = false;
    b->prev    = -1;
    b->next    = -1;
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
llama_block_manager::checkout_gpu_blocks(uint32_t num_blocks, std::vector<evicted_block> * evicted) {
    physical_block_ids new_ids = {};
    if (!has_free_gpu_blocks(num_blocks)) {
        return new_ids;
    }

    new_ids.reserve(num_blocks);
    for (uint32_t i = 0; i < num_blocks; ++i) {
        const uint32_t id = queue_pop_front(/*gpu=*/true);
        physical_block & b = gpu_registry[id];

        // Lazy eviction: the block's data is still valid but the pool needs
        // it back. Report the hash so the owner can drop the index entry.
        if (b.has_hash) {
            if (evicted) {
                evicted->push_back({ b.hash, id });
            }
            b.has_hash = false;
            b.hash     = 0;
            b.tag.clear();
            n_cached_gpu--;
        }
        b.ref_count += 1;
        new_ids.push_back(id);
    }
    return new_ids;
}

llama_block_manager::physical_block_ids
llama_block_manager::checkout_cpu_blocks(uint32_t num_blocks, std::vector<evicted_block> * evicted) {
    physical_block_ids new_ids = {};
    if (!has_free_cpu_blocks(num_blocks)) {
        return new_ids;
    }

    new_ids.reserve(num_blocks);
    for (uint32_t i = 0; i < num_blocks; ++i) {
        const uint32_t id = queue_pop_front(/*gpu=*/false);
        physical_block & b = cpu_registry[id - total_num_gpu_blocks];

        if (b.has_hash) {
            if (evicted) {
                evicted->push_back({ b.hash, id });
            }
            b.has_hash = false;
            b.hash     = 0;
            b.tag.clear();
            n_cached_cpu--;
        }
        b.ref_count += 1;
        new_ids.push_back(id);
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
                b.tag.clear();
            } else {
                n_cached_gpu++;
            }
            queue_push_back(id);
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
                b.tag.clear();
            } else {
                n_cached_cpu++;
            }
            queue_push_back(id);
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
        queue_remove(id);
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
    b->tag.clear();
}

void llama_block_manager::set_tag(uint32_t id, const std::string & tag) {
    physical_block * b = block_at(id);
    if (!b) {
        LLAMA_LOG_WARN("%s: tag write on invalid block %u\n", __func__, id);
        return;
    }
    if (b->ref_count == 0 && !b->has_hash) {
        // Tagging a plain free block would leak the owner into a future
        // checkout. Cache-resident free blocks keep their owner instead.
        LLAMA_LOG_WARN("%s: tag '%s' rejected on free block %u\n", __func__, tag.c_str(), id);
        return;
    }
    b->tag = tag;
}

const std::string & llama_block_manager::tag(uint32_t id) const {
    static const std::string empty;
    const physical_block * b = block_at(id);
    return b ? b->tag : empty;
}
