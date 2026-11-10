#include "llama-block-manager.h"

#include "llama-impl.h"

#include <cmath>

void llama_block_manager::init(uint32_t n_gpu, uint32_t n_cpu, float watermark) {
    LLAMA_LOG_INFO("%s: Block manager initialized: n_free_gpu_blocks=%d, n_free_cpu_blocks=%d\n", __func__, n_gpu,
                   n_cpu);

    // Make init idempotent: callers may reuse an instance to reset the pools.
    gpu_registry.clear();
    cpu_registry.clear();
    free_gpu_ids.clear();
    free_cpu_ids.clear();

    total_num_gpu_blocks = n_gpu;
    total_num_cpu_blocks = n_cpu;

    watermark_gpu_safety_num_blocks = std::ceil(total_num_gpu_blocks * watermark);
    watermark_cpu_safety_num_blocks = std::ceil(total_num_cpu_blocks * watermark);

    gpu_registry.resize(n_gpu);
    for (uint32_t i = 0; i < n_gpu; ++i) {
        gpu_registry[i].id     = i;
        gpu_registry[i].is_gpu = true;
        free_gpu_ids.push_back(i);
    }

    cpu_registry.resize(n_cpu);
    for (uint32_t i = 0; i < n_cpu; ++i) {
        cpu_registry[i].id     = i + total_num_gpu_blocks;
        cpu_registry[i].is_gpu = false;
        free_cpu_ids.push_back(i + total_num_gpu_blocks);
    }
}

size_t llama_block_manager::n_free_gpu_blocks() const {
    return free_gpu_ids.size();
}

size_t llama_block_manager::n_free_cpu_blocks() const {
    return free_cpu_ids.size();
}

bool llama_block_manager::has_free_gpu_blocks(uint32_t num_requested_blocks) const {
    size_t curr_free_gpus = free_gpu_ids.size();
    if (curr_free_gpus < watermark_gpu_safety_num_blocks) {
        return false;
    }
    return num_requested_blocks <= (curr_free_gpus - watermark_gpu_safety_num_blocks);
}

bool llama_block_manager::has_free_cpu_blocks(uint32_t num_requested_blocks) const {
    size_t curr_free_cpus = free_cpu_ids.size();
    if (curr_free_cpus < watermark_cpu_safety_num_blocks) {
        return false;
    }
    return num_requested_blocks <= (curr_free_cpus - watermark_cpu_safety_num_blocks);
}

llama_block_manager::physical_block_ids llama_block_manager::checkout_gpu_blocks(uint32_t num_blocks) {
    physical_block_ids new_ids = {};
    if (num_blocks > free_gpu_ids.size()) {
        return new_ids;
    }

    new_ids.insert(new_ids.end(), std::make_move_iterator(free_gpu_ids.end() - num_blocks),
                   std::make_move_iterator(free_gpu_ids.end()));
    free_gpu_ids.erase(free_gpu_ids.end() - num_blocks, free_gpu_ids.end());

    for (const uint32_t & id : new_ids) {
        gpu_registry[id].ref_count += 1;
    }
    return new_ids;
}

llama_block_manager::physical_block_ids llama_block_manager::checkout_cpu_blocks(uint32_t num_blocks) {
    physical_block_ids new_ids = {};
    if (num_blocks > free_cpu_ids.size()) {
        return new_ids;
    }

    new_ids.insert(new_ids.end(), std::make_move_iterator(free_cpu_ids.end() - num_blocks),
                   std::make_move_iterator(free_cpu_ids.end()));
    free_cpu_ids.erase(free_cpu_ids.end() - num_blocks, free_cpu_ids.end());

    for (const uint32_t & id : new_ids) {
        cpu_registry[id - total_num_gpu_blocks].ref_count += 1;
    }
    return new_ids;
}

void llama_block_manager::release_gpu_blocks(const physical_block_ids & freed_blocks_ids) {
    for (const uint32_t & id : freed_blocks_ids) {
        if (id >= total_num_gpu_blocks || gpu_registry[id].ref_count == 0) {
            LLAMA_LOG_ERROR("%s: release of free/invalid GPU block %u\n", __func__, id);
            continue;
        }
        gpu_registry[id].ref_count -= 1;
        if (gpu_registry[id].ref_count <= 0) {
            gpu_registry[id].ref_count = 0;
            gpu_registry[id].generation += 1;
            gpu_registry[id].tag.clear();
            free_gpu_ids.push_back(id);
        }
    }
}

void llama_block_manager::release_cpu_blocks(const physical_block_ids & freed_blocks_ids) {
    for (const uint32_t & id : freed_blocks_ids) {
        if (id < total_num_gpu_blocks || id - total_num_gpu_blocks >= total_num_cpu_blocks ||
            cpu_registry[id - total_num_gpu_blocks].ref_count == 0) {
            LLAMA_LOG_ERROR("%s: release of free/invalid CPU block %u\n", __func__, id);
            continue;
        }
        cpu_registry[id - total_num_gpu_blocks].ref_count -= 1;
        if (cpu_registry[id - total_num_gpu_blocks].ref_count <= 0) {
            cpu_registry[id - total_num_gpu_blocks].ref_count = 0;
            cpu_registry[id - total_num_gpu_blocks].generation += 1;
            cpu_registry[id - total_num_gpu_blocks].tag.clear();
            free_cpu_ids.push_back(id);
        }
    }
}

bool llama_block_manager::is_gpu(uint32_t block_id) const {
    return block_id < total_num_gpu_blocks;
}

bool llama_block_manager::add_ref_block(uint32_t id) {
    if (id < total_num_gpu_blocks) {
        if (gpu_registry[id].ref_count == 0) {
            return false;
        }
        gpu_registry[id].ref_count += 1;
        return true;
    }
    const uint32_t cpu_id = id - total_num_gpu_blocks;
    if (cpu_id >= total_num_cpu_blocks || cpu_registry[cpu_id].ref_count == 0) {
        return false;
    }
    cpu_registry[cpu_id].ref_count += 1;
    return true;
}

uint32_t llama_block_manager::block_ref_count(uint32_t id) const {
    if (id < total_num_gpu_blocks) {
        return gpu_registry[id].ref_count;
    }
    const uint32_t cpu_id = id - total_num_gpu_blocks;
    if (cpu_id >= total_num_cpu_blocks) {
        return 0;
    }
    return cpu_registry[cpu_id].ref_count;
}

uint32_t llama_block_manager::block_generation(uint32_t id) const {
    if (id < total_num_gpu_blocks) {
        return gpu_registry[id].generation;
    }
    const uint32_t cpu_id = id - total_num_gpu_blocks;
    if (cpu_id >= total_num_cpu_blocks) {
        return 0;
    }
    return cpu_registry[cpu_id].generation;
}

void llama_block_manager::set_tag(uint32_t id, const std::string & tag) {
    physical_block * blk = nullptr;
    if (id < total_num_gpu_blocks) {
        blk = &gpu_registry[id];
    } else {
        const uint32_t cpu_id = id - total_num_gpu_blocks;
        if (cpu_id >= total_num_cpu_blocks) {
            LLAMA_LOG_WARN("%s: tag write on invalid block %u\n", __func__, id);
            return;
        }
        blk = &cpu_registry[cpu_id];
    }
    if (blk->ref_count == 0) {
        // Tagging a free block would leave an owner on a block that has no
        // request behind it; a later checkout outside allocate() (swap_in)
        // would silently inherit that stale owner. Refuse and warn.
        LLAMA_LOG_WARN("%s: tag '%s' rejected on free block %u\n", __func__, tag.c_str(), id);
        return;
    }
    blk->tag = tag;
}

const std::string & llama_block_manager::tag(uint32_t id) const {
    static const std::string empty;
    if (id < total_num_gpu_blocks) {
        return gpu_registry[id].tag;
    }
    const uint32_t cpu_id = id - total_num_gpu_blocks;
    if (cpu_id >= total_num_cpu_blocks) {
        return empty;
    }
    return cpu_registry[cpu_id].tag;
}
