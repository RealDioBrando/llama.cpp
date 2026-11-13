#include "llama-kv-cache-paged.h"

#include <algorithm>
#include <utility>

#include "llama-impl.h"

//
// llama_kv_cache_paged
//

llama_kv_cache_paged::llama_kv_cache_paged(uint32_t head_dim,
                                           uint32_t n_heads_kv,
                                           uint32_t block_size,
                                           uint32_t n_layers,
                                           uint32_t n_ubatch,
                                           uint32_t n_seq_max) :
    kv_type(GGML_TYPE_F16),
    head_dim(head_dim),
    n_heads_kv(n_heads_kv),
    block_size(block_size),
    n_layers(n_layers),
    n_ubatch(n_ubatch),
    n_seq_max(n_seq_max),
    num_gpu_blocks(0),
    num_cpu_blocks(0),
    gpu_backend(nullptr),
    cpu_backend(nullptr) {}

void llama_kv_cache_paged::init(ggml_backend_t backend_gpu,
                                ggml_backend_t backend_cpu,
                                enum ggml_type type,
                                uint32_t       n_gpu_blocks,
                                uint32_t       n_cpu_blocks,
                                float          watermark,
                                std::function<bool(uint32_t)> filter) {
    GGML_ASSERT(backend_cpu && "backend_cpu is nullptr");
    GGML_ASSERT(backend_gpu && "backend_gpu is nullptr");
    const ggml_backend_dev_t dev = ggml_backend_get_device(backend_gpu);
    if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        LLAMA_LOG_WARN(
            "%s: no GPU device found, allocating KV block pool on CPU. "
            "This is valid for testing but it will be slow.\n",
            __func__);
    }

    GGML_ASSERT(n_gpu_blocks && "n_gpu_blocks need to be greater than 0.");
    GGML_ASSERT(n_cpu_blocks && "n_cpu_blocks need to be greater than 0.");

    LLAMA_LOG_INFO(
        "%s: initializing paged KV cache. n_gpu_blocks=%d, n_cpu_blocks=%d, block_size=%d, watermark=%0.2f\n", __func__,
        n_gpu_blocks, n_cpu_blocks, block_size, watermark);
    num_gpu_blocks = n_gpu_blocks;
    num_cpu_blocks = n_cpu_blocks;
    kv_type        = type;
    gpu_backend    = backend_gpu;
    cpu_backend    = backend_cpu;
    // Per-block bytes must match the 4D tensor layout [head_dim, block_size,
    // n_heads_kv, n_blocks]; ggml_row_size also handles quantized row padding.
    block_bytes = (uint32_t) (ggml_row_size(kv_type, head_dim) * block_size * n_heads_kv);
    this->watermark = watermark;

    // Set up GPU context and tensors. One K pool and one V pool per layer.
    struct ggml_init_params gpu_params;
    gpu_params.mem_size   = ggml_tensor_overhead() * 5 * n_layers;
    gpu_params.mem_buffer = NULL;
    gpu_params.no_alloc   = true;

    struct ggml_context * ctx_gpu = ggml_init(gpu_params);

    for (uint32_t il = 0; il < n_layers; ++il) {
        if (filter && !filter(il)) {
            k_gpu_layers.push_back(nullptr);
            v_gpu_layers.push_back(nullptr);
            continue;
        }
        ggml_tensor * k_layer_gpu =
            ggml_new_tensor_4d(ctx_gpu, type, head_dim, block_size, n_heads_kv, n_gpu_blocks);
        ggml_tensor * v_layer_gpu =
            ggml_new_tensor_4d(ctx_gpu, type, head_dim, block_size, n_heads_kv, n_gpu_blocks);
        // the meta backend (tensor parallelism) identifies the pools by name
        // and slices them on the KV head axis
        ggml_format_name(k_layer_gpu, "cache_paged_k_l%u", il);
        ggml_format_name(v_layer_gpu, "cache_paged_v_l%u", il);
        k_gpu_layers.push_back(k_layer_gpu);
        v_gpu_layers.push_back(v_layer_gpu);
    }

    // Allocate on GPU backend
    ggml_backend_buffer_t buf_gpu = ggml_backend_alloc_ctx_tensors(ctx_gpu, backend_gpu);
    GGML_ASSERT(buf_gpu && "Failed to allocate GPU KV cache buffer");
    ggml_backend_buffer_clear(buf_gpu, 0);  // zero out the cache
    for (uint32_t il = 0; il < n_layers; ++il) {
        if (!k_gpu_layers[il]) {
            continue;
        }
        GGML_ASSERT(k_gpu_layers[il]->buffer && v_gpu_layers[il]->buffer && "GPU layer tensor has null buffer");
    }
    ctx_buf_gpu = {ggml_context_ptr(ctx_gpu), ggml_backend_buffer_ptr(buf_gpu)};

    // Set up CPU context and tensor (for swapping)
    struct ggml_init_params cpu_params;
    cpu_params.mem_size           = ggml_tensor_overhead() * 5 * n_layers;
    cpu_params.mem_buffer         = NULL;
    cpu_params.no_alloc           = true;
    struct ggml_context * ctx_cpu = ggml_init(cpu_params);
    for (uint32_t il = 0; il < n_layers; ++il) {
        if (filter && !filter(il)) {
            k_cpu_layers.push_back(nullptr);
            v_cpu_layers.push_back(nullptr);
            continue;
        }
        ggml_tensor * k_layer_cpu =
            ggml_new_tensor_4d(ctx_cpu, type, head_dim, block_size, n_heads_kv, n_cpu_blocks);
        ggml_tensor * v_layer_cpu =
            ggml_new_tensor_4d(ctx_cpu, type, head_dim, block_size, n_heads_kv, n_cpu_blocks);
        k_cpu_layers.push_back(k_layer_cpu);
        v_cpu_layers.push_back(v_layer_cpu);
    }

    // Allocate on the CPU backend (using pinned memory for faster PCIe transfer)
    ggml_backend_buffer_t buf_cpu = ggml_backend_alloc_ctx_tensors(ctx_cpu, backend_cpu);
    GGML_ASSERT(buf_cpu && "Failed to allocate CPU KV cache buffer");
    ggml_backend_buffer_clear(buf_cpu, 0);  // zero out the cache
    for (uint32_t il = 0; il < n_layers; ++il) {
        if (!k_cpu_layers[il]) {
            continue;
        }
        GGML_ASSERT(k_cpu_layers[il]->buffer && v_cpu_layers[il]->buffer && "CPU layer tensor has null buffer");
    }
    ctx_buf_cpu = {ggml_context_ptr(ctx_cpu), ggml_backend_buffer_ptr(buf_cpu)};

    // Setting up our block accountant
    block_manager.init(n_gpu_blocks, n_cpu_blocks, watermark);
    // The pool kind is fixed by the owner; the callback is whatever was staged
    // on the cache (a no-op when init() runs before the setter, as usual).
    block_manager.set_pool_kind(LLAMA_PAGED_POOL_FULL);
    block_manager.set_evict_pick_cb(evict_pick_cb_, evict_pick_ud_);
}

void llama_kv_cache_paged::set_evict_pick_cb(llama_paged_evict_pick_cb cb, void * ud) {
    // Stored so clear() can re-install them after the block manager is rebuilt.
    evict_pick_cb_ = cb;
    evict_pick_ud_ = ud;
    block_manager.set_pool_kind(LLAMA_PAGED_POOL_FULL);
    block_manager.set_evict_pick_cb(cb, ud);
}

bool llama_kv_cache_paged::gpu_checkout_refused_by_policy() const {
    return block_manager.gpu_checkout_refused_by_policy();
}

void llama_kv_cache_paged::reset_gpu_checkout_refusal() {
    block_manager.reset_gpu_checkout_refusal();
}

bool llama_kv_cache_paged::allocate(int32_t num_tokens, llama_sequence_group & group) {
    uint32_t curr_block_count     = group.block_table.size();
    // n_past counts tokens already covered by block_table (including a
    // prefix shared through the prefix cache).
    uint32_t total_num_tokens     = group.n_past + (group.n_decoded > 0 ? 0u : group.n_prompt) + num_tokens;
    uint32_t num_requested_blocks = std::ceil((float) total_num_tokens / block_size) - curr_block_count;
    LLAMA_LOG_DEBUG("%s: curr_block_count=%d, total_num_tokens=%d, num_requested_blocks=%d\n", __func__,
                    curr_block_count, total_num_tokens, num_requested_blocks);

    if (num_requested_blocks == 0) {
        return true;
    }

    if (!block_manager.has_free_gpu_blocks(num_requested_blocks)) {
        LLAMA_LOG_DEBUG("%s: insufficient GPU blocks. Requested: %d.\n", __func__, num_requested_blocks);
        return false;
    }

    // Checkout may evict cache-resident prefix blocks; the wrapper drops their
    // index entries before the blocks are handed out.
    llama_block_ids new_ids = checkout_gpu_blocks(num_requested_blocks);
    if (new_ids.size() != num_requested_blocks) {
        if (new_ids.empty()) {
            // An installed eviction policy may legitimately refuse to sacrifice
            // cache-resident blocks (design §8.4): the request simply stays
            // waiting, so this is expected back-pressure, not an error.
            LLAMA_LOG_DEBUG("%s: GPU checkout refused %d block(s); request stays waiting.\n", __func__,
                            num_requested_blocks);
        } else {
            LLAMA_LOG_ERROR("%s: checkout returned %zu of %d blocks.\n", __func__, new_ids.size(), num_requested_blocks);
            block_manager.release_gpu_blocks(new_ids);
        }
        return false;
    }
    // Fresh allocations carry the owner tag of the requesting group. The tag
    // is only an ownership record; the block enters the tag -> block mapping
    // once it is filled and registered in the prefix index.
    for (const uint32_t id : new_ids) {
        block_manager.set_tag(id, group.tag);
    }
    concat_block_ids(group.block_table, new_ids);
    LLAMA_LOG_DEBUG("%s: successfully allocated %d.\n", __func__, num_requested_blocks);
    return true;
}

llama_block_ids llama_kv_cache_paged::checkout_gpu_blocks(uint32_t num_blocks) {
    std::vector<llama_block_manager::evicted_block> evicted;
    llama_block_ids ids = block_manager.checkout_gpu_blocks(num_blocks, &evicted);
    for (const auto & ev : evicted) {
        // Identity-checked removal: a same-chain entry owned by another block
        // must survive.
        prefix_index.remove_if_block(ev.hash, ev.block_id);
    }
    return ids;
}

void llama_kv_cache_paged::free_blocks(llama_sequence_group & group) {
    if (group.block_table.empty()) {
        return;
    }

    llama_block_ids blocks_to_free_gpu;
    llama_block_ids blocks_to_free_cpu;

    for (uint32_t block_id : group.block_table) {
        if (block_manager.is_gpu(block_id)) {
            blocks_to_free_gpu.push_back(block_id);
        } else {
            blocks_to_free_cpu.push_back(block_id);
        }
    }

    // Hash entries are intentionally kept for blocks that stay cache-resident.
    if (!blocks_to_free_gpu.empty()) {
        block_manager.release_gpu_blocks(blocks_to_free_gpu);
    }
    if (!blocks_to_free_cpu.empty()) {
        block_manager.release_cpu_blocks(blocks_to_free_cpu);
    }

    group.block_table.clear();
    seq_rm(group.request_id, llama_pos{}, llama_pos{});
}

bool llama_kv_cache_paged::release_block(uint32_t block_id) {
    if (block_manager.is_gpu(block_id)) {
        block_manager.release_gpu_blocks({block_id});
    } else {
        block_manager.release_cpu_blocks({block_id});
    }
    return true;
}

void llama_kv_cache_paged::copy_block_content(uint32_t src_block_id, uint32_t dst_block_id) {
    const bool same_pool  = block_manager.is_gpu(src_block_id) == block_manager.is_gpu(dst_block_id);
    const bool dst_is_gpu = block_manager.is_gpu(dst_block_id);

    // do_block_copy models a cross-pool transfer; for a same-pool copy feed it
    // the matching src/dst layer sets directly
    if (same_pool) {
        const auto & k = dst_is_gpu ? k_gpu_layers : k_cpu_layers;
        const auto & v = dst_is_gpu ? v_gpu_layers : v_cpu_layers;
        const uint32_t src_local = dst_is_gpu ? src_block_id : src_block_id - num_gpu_blocks;
        const uint32_t dst_local = dst_is_gpu ? dst_block_id : dst_block_id - num_gpu_blocks;

        std::vector<uint8_t> staging(block_bytes);
        for (uint32_t il = 0; il < n_layers; ++il) {
            if (!k[il]) {
                continue;
            }
            const size_t src_offset = (size_t) src_local * block_bytes;
            const size_t dst_offset = (size_t) dst_local * block_bytes;
            ggml_backend_tensor_get(k[il], staging.data(), src_offset, block_bytes);
            ggml_backend_tensor_set(k[il], staging.data(), dst_offset, block_bytes);
            ggml_backend_tensor_get(v[il], staging.data(), src_offset, block_bytes);
            ggml_backend_tensor_set(v[il], staging.data(), dst_offset, block_bytes);
        }
        return;
    }

    do_block_copy({src_block_id}, {dst_block_id}, /*to_gpu=*/dst_is_gpu);
}

uint32_t llama_kv_cache_paged::n_free_gpu_blocks() const {
    return (uint32_t) block_manager.n_free_gpu_blocks();
}

uint32_t llama_kv_cache_paged::n_free_cpu_blocks() const {
    return (uint32_t) block_manager.n_free_cpu_blocks();
}

uint32_t llama_kv_cache_paged::n_allocatable_gpu_blocks() const {
    return (uint32_t) block_manager.n_allocatable_gpu_blocks();
}

uint32_t llama_kv_cache_paged::n_cached_gpu_blocks() const {
    return (uint32_t) block_manager.n_cached_gpu_blocks();
}

void llama_kv_cache_paged::do_block_copy(const llama_block_ids & src_ids,
                                         const llama_block_ids & new_ids,
                                         bool                    to_gpu) {
    const uint32_t num_blocks = src_ids.size();
    LLAMA_LOG_DEBUG("%s: num_blocks_size=%d, new_ids_size=%ld\n", __func__, num_blocks, new_ids.size());
    GGML_ASSERT(num_blocks == new_ids.size() && "src_ids and new_ids do not have the same size.");

    const auto & src_k = to_gpu ? k_cpu_layers : k_gpu_layers;
    const auto & dst_k = to_gpu ? k_gpu_layers : k_cpu_layers;
    const auto & src_v = to_gpu ? v_cpu_layers : v_gpu_layers;
    const auto & dst_v = to_gpu ? v_gpu_layers : v_cpu_layers;

    GGML_ASSERT(src_k.size() == n_layers && src_v.size() == n_layers && "src layer count mismatch.");
    GGML_ASSERT(dst_k.size() == n_layers && dst_v.size() == n_layers && "dst layer count mismatch.");

    // Buffer on HOST to faciliate block data transfer
    // Note: an optimization would be to use views and async copies. Beware of
    // memory overhead heurisitcs.
    std::vector<uint8_t> staging(block_bytes);

    for (uint32_t il = 0; il < n_layers; ++il) {
        if (!src_k[il]) {
            continue;
        }
        for (uint32_t i = 0; i < num_blocks; ++i) {
            const uint32_t src_global = src_ids[i];
            const uint32_t dst_global = new_ids[i];

            // GPU and CPu blocks may differ (usually CPU < GPU)
            // We substract the diffence to calculate where the local starts before we calculate offsets
            const uint32_t src_local = to_gpu ? src_global - num_gpu_blocks : src_global;
            const uint32_t dst_local = to_gpu ? dst_global : dst_global - num_gpu_blocks;

            const size_t src_offset = (size_t) src_local * block_bytes;
            const size_t dst_offset = (size_t) dst_local * block_bytes;

            // Put src tensor into HOST staging buffer
            // Put tensor from HOST staging into dst tensor
            ggml_backend_tensor_get(src_k[il], staging.data(), src_offset, block_bytes);
            ggml_backend_tensor_set(dst_k[il], staging.data(), dst_offset, block_bytes);
            ggml_backend_tensor_get(src_v[il], staging.data(), src_offset, block_bytes);
            ggml_backend_tensor_set(dst_v[il], staging.data(), dst_offset, block_bytes);
        }
    }
}

bool llama_kv_cache_paged::can_swap_in(llama_sequence_group & group) const {
    const uint32_t num_blocks = group.block_table.size();
    if (num_blocks == 0) {
        return true;
    }
    // n_prompt is only refreshed at admission, so a mid-prefill group must
    // derive the remaining prompt from n_past instead of the stale field.
    const int32_t  n_remaining   = std::max<int32_t>(0, (int32_t) group.n_prompt_total - (int32_t) group.n_past);
    const uint32_t next_tokens   = group.n_decoded > 0
                                       ? 1u + (uint32_t) group.pending_draft.size()
                                       : (uint32_t) n_remaining + 1u;
    const uint32_t needed_blocks = (group.n_past + next_tokens + block_size - 1) / block_size;
    const uint32_t n_new_ids     = std::max(num_blocks, needed_blocks);
    return block_manager.has_free_gpu_blocks(n_new_ids);
}

bool llama_kv_cache_paged::can_swap_out(llama_sequence_group & group) const {
    const uint32_t num_blocks = group.block_table.size();
    return num_blocks == 0 || block_manager.has_free_cpu_blocks(num_blocks);
}

bool llama_kv_cache_paged::swap_in(llama_sequence_group & group) {
    const uint32_t num_blocks = group.block_table.size();
    if (num_blocks == 0) {
        return true;
    }

    // A swapped group is pushed straight into the next batch after swap-in, so
    // the restored table must already cover the token(s) about to be written.
    // In particular a decode group can be swapped out with its table exactly
    // covering the prefix (n_past on a block boundary): without one extra
    // block, calculate_global_slot_index() reads block_table[n_past/block_size]
    // out of bounds. Prefill groups that were swapped out before their prompt
    // was fully reserved get the remainder reserved here as well.
    // n_prompt is only refreshed at admission, so a mid-prefill group must
    // derive the remaining prompt from n_past instead of the stale field.
    const int32_t  n_remaining = std::max<int32_t>(0, (int32_t) group.n_prompt_total - (int32_t) group.n_past);
    const uint32_t next_tokens = group.n_decoded > 0
                                     ? 1u + (uint32_t) group.pending_draft.size()
                                     : (uint32_t) n_remaining + 1u;
    const uint32_t needed_blocks = (group.n_past + next_tokens + block_size - 1) / block_size;
    const uint32_t n_new_ids     = std::max(num_blocks, needed_blocks);

    // A potential optimization to reduce thrashing is to have a heuristic to check if
    // if we can continue decoding after swap_in.
    if (!block_manager.has_free_gpu_blocks(n_new_ids)) {
        return false;
    }

    llama_block_ids new_ids = checkout_gpu_blocks(n_new_ids);
    if (new_ids.size() != n_new_ids) {
        return false;
    }
    llama_block_ids copy_dst(new_ids.begin(), new_ids.begin() + num_blocks);
    do_block_copy(group.block_table, copy_dst, /*to_gpu=*/true);

    // The restored GPU copies are private and unhashed. If the original
    // cache-resident copy was evicted while the request was swapped out, try to
    // re-register the restored full blocks so the chain stays reusable. If a
    // live entry already exists for the hash (first-wins), this is a no-op.
    for (size_t i = 0; i < group.block_hashes.size() && i < num_blocks; ++i) {
        if ((i + 1) * block_size > group.logical_seq.size()) {
            break;
        }
        const uint64_t hash = group.block_hashes[i];
        const auto *   e    = prefix_index.find_hash(hash);
        if (e && prefix_block_is_live(e->block_id, hash)) {
            continue;  // a live copy already owns this chain
        }
        register_prefix_block(hash, new_ids[i], group.request_id, (uint32_t) i,
                              group.logical_seq[i * block_size],
                              group.logical_seq[(i + 1) * block_size - 1]);
    }

    free_blocks(group);
    group.block_table = std::move(new_ids);

    // The restored GPU copies are private and unhashed. Record the requesting
    // tag so the block keeps the full set of tags that used it while it stays
    // cache-resident. set_tag is allowed here: every restored block is owned
    // by the group (ref_count > 0).
    for (const uint32_t id : group.block_table) {
        block_manager.set_tag(id, group.tag);
    }
    return true;
}

bool llama_kv_cache_paged::swap_out(llama_sequence_group & group) {
    const uint32_t num_blocks = group.block_table.size();
    if (num_blocks == 0) {
        return true;
    }

    if (!block_manager.has_free_cpu_blocks(num_blocks)) {
        return false;
    }

    llama_block_ids new_ids = block_manager.checkout_cpu_blocks(num_blocks);
    if (new_ids.size() != num_blocks) {
        return false;
    }
    do_block_copy(group.block_table, new_ids, /*to_gpu=*/false);

    // The CPU copies are private and unhashed. Keeping the hash on the old GPU
    // blocks is deliberate: if this group was their last holder they become
    // cache-resident (ref_count == 0) and remain hittable; if they are shared,
    // the hash stays with the live GPU copy. The index never points at a CPU
    // block in this phase.
    free_blocks(group);
    group.block_table = new_ids;
    return true;
}

void llama_kv_cache_paged::set_paged_batch_info(const llama_paged_batch_info * info) {
    last_paged_info = info;
}

llama_memory_context_ptr llama_kv_cache_paged::create_batch_context(
        const std::vector<llama_ubatch> & ubatches) {
    if (!last_paged_info) {
        return std::make_unique<llama_kv_cache_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    auto ctx = std::make_unique<llama_kv_cache_paged_context>(this, ubatches);
    ctx->set_batch_data(*last_paged_info);
    ctx->derive_current_batch_data();

    return ctx;
}

uint32_t llama_kv_cache_paged::get_num_gpu_blocks() const {
    return num_gpu_blocks;
}

uint32_t llama_kv_cache_paged::get_num_cpu_blocks() const {
    return num_cpu_blocks;
}

uint64_t llama_kv_cache_paged::block_tag(uint32_t block_id) const {
    return block_manager.tag(block_id);
}

void llama_kv_cache_paged::collect_tag_blocks(
        std::unordered_map<uint64_t, std::unordered_set<uint32_t>> & owned,
        std::unordered_set<uint32_t> *                               reusable_ids) const {
    for (const auto & [hash, e] : prefix_index.entries()) {
        (void) hash;
        const uint32_t id = e.block_id;
        // Same liveness filter the prefix matcher applies: GPU-resident and
        // still owning the hash. ref_count == 0 is valid: that is a cache-
        // resident block kept in the free queue.
        if (!block_manager.is_gpu(id) ||
            !block_manager.block_has_hash(id) ||
            block_manager.block_hash(id) != e.hash) {
            continue;
        }
        if (reusable_ids) {
            reusable_ids->insert(id);
        }
        // Attribute the block to EVERY tag that used it while cache-resident.
        // With the multi-tag model there is no owner/primary tag, and tag()
        // would expose only one representative, silently dropping the others
        // from the per-tag statistics.
        const uint32_t n_tags = block_manager.tag_count(id);
        for (uint32_t t = 0; t < n_tags; ++t) {
            const uint64_t tag = block_manager.tag_at(id, t);
            if (tag != 0) {
                owned[tag].insert(id);
            }
        }
    }
}

void llama_kv_cache_paged::concat_block_ids(llama_block_ids &       to_block_table,
                                            const llama_block_ids & from_block_table) {
    to_block_table.insert(to_block_table.end(), from_block_table.begin(), from_block_table.end());
}

bool llama_kv_cache_paged::share_blocks(llama_sequence_group & group, const llama_block_ids & src_ids) {
    if (!add_ref_blocks(src_ids, group.tag)) {
        return false;
    }
    concat_block_ids(group.block_table, src_ids);
    return true;
}

bool llama_kv_cache_paged::add_ref_blocks(const llama_block_ids & ids, uint64_t tag) {
    llama_block_ids added;
    added.reserve(ids.size());
    for (const uint32_t id : ids) {
        if (!block_manager.add_ref_block(id)) {
            if (!added.empty()) {
                for (const uint32_t done : added) {
                    if (block_manager.is_gpu(done)) {
                        block_manager.release_gpu_blocks({ done });
                    } else {
                        block_manager.release_cpu_blocks({ done });
                    }
                }
            }
            return false;
        }
        added.push_back(id);
    }
    // Tag only after the whole operation succeeded: on failure the caller gets
    // its refs back and the blocks must not have picked up a phantom tag.
    if (tag != 0) {
        for (const uint32_t id : ids) {
            block_manager.set_tag(id, tag);
        }
    }
    return true;
}

bool llama_kv_cache_paged::acquire_blocks(const llama_block_ids & ids, uint64_t tag) {
    llama_block_ids acquired;
    acquired.reserve(ids.size());
    for (const uint32_t id : ids) {
        const bool ok = block_manager.block_ref_count(id) == 0 ? block_manager.touch_block(id)
                                                               : block_manager.add_ref_block(id);
        if (!ok) {
            // All-or-nothing: undo the refs acquired so far. release_* keeps
            // any hash entry, so cache residency is preserved.
            for (auto it = acquired.rbegin(); it != acquired.rend(); ++it) {
                if (block_manager.is_gpu(*it)) {
                    block_manager.release_gpu_blocks({ *it });
                } else {
                    block_manager.release_cpu_blocks({ *it });
                }
            }
            return false;
        }
        acquired.push_back(id);
    }
    // Second pass: record the requesting tag only once the whole acquisition
    // succeeded, so an atomic failure leaves every tag set untouched.
    if (tag != 0) {
        for (const uint32_t id : ids) {
            block_manager.set_tag(id, tag);
        }
    }
    return true;
}

void llama_kv_cache_paged::set_prefix_enabled(bool enabled) {
    prefix_index.set_enabled(enabled);
}

bool llama_kv_cache_paged::prefix_enabled() const {
    return prefix_index.enabled();
}

const llama_full_prefix_entry * llama_kv_cache_paged::find_prefix_block(uint64_t    hash,
                                                                        llama_token first_tok,
                                                                        llama_token last_tok) const {
    const auto * e = prefix_index.find(hash, first_tok, last_tok);
    if (!e || !prefix_block_is_live(e->block_id, e->hash)) {
        return nullptr;
    }
    return e;
}

bool llama_kv_cache_paged::prefix_block_is_live(uint32_t block_id, uint64_t hash) const {
    return block_manager.is_gpu(block_id) &&
           block_manager.block_has_hash(block_id) &&
           block_manager.block_hash(block_id) == hash;
}

void llama_kv_cache_paged::register_prefix_block(uint64_t    hash,
                                                 uint32_t    block_id,
                                                 int32_t     owner_seq,
                                                 uint32_t    block_idx,
                                                 llama_token first_tok,
                                                 llama_token last_tok) {
    // First-wins: if another physical block already owns this chain, keep it.
    // The duplicate copy stays unhashed and is reclaimed as a plain free block.
    const auto * old = prefix_index.find_hash(hash);
    if (old) {
        if (prefix_block_is_live(old->block_id, hash)) {
            return;
        }
        prefix_index.remove(hash);  // stale entry, replace it
    }

    if (block_manager.block_has_hash(block_id)) {
        GGML_ASSERT(block_manager.block_hash(block_id) == hash && "block already owns another hash");
    } else {
        block_manager.set_block_hash(block_id, hash);
    }
    prefix_index.register_block(hash, block_id, owner_seq, block_idx, first_tok, last_tok);
}

void llama_kv_cache_paged::invalidate_prefix_block(uint32_t block_id) {
    if (!block_manager.block_has_hash(block_id)) {
        return;
    }
    prefix_index.remove_if_block(block_manager.block_hash(block_id), block_id);
    block_manager.clear_block_hash(block_id);
}

uint32_t llama_kv_cache_paged::block_ref_count(uint32_t block_id) const {
    return block_manager.block_ref_count(block_id);
}

bool llama_kv_cache_paged::is_block_gpu(uint32_t block_id) const {
    return block_manager.is_gpu(block_id);
}

bool llama_kv_cache_paged::block_has_hash(uint32_t block_id) const {
    return block_manager.block_has_hash(block_id);
}

uint64_t llama_kv_cache_paged::block_hash(uint32_t block_id) const {
    return block_manager.block_hash(block_id);
}

// llama_memory_i

llama_memory_context_ptr llama_kv_cache_paged::init_batch(llama_batch_allocr & balloc,
                                                          uint32_t             n_ubatch,
                                                          bool /*embd_all*/) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            // Paged metadata requires one token count per sequence set.
            // split_simple emits one pseudo-sequence per token and hides
            // prefill from phase-sensitive backends.
            auto ubatch = balloc.split_equal(n_ubatch, false, 0);
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch));
        }

        // Failed to find a suitable split
        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            LLAMA_LOG_ERROR("%s: split did not consume the batch: used=%d of %d tokens\n", __func__,
                            (int) balloc.get_n_used(), (int) balloc.get_n_tokens());
            break;
        }

        auto ctx = std::make_unique<llama_kv_cache_paged_context>(this, std::move(ubatches));

        // Do not use balloc's internal batch. It does not carry any paged metadata.
        GGML_ASSERT(last_paged_info && "no paged batch info set before init_batch was called.");
        ctx->set_batch_data(*last_paged_info);
        ctx->derive_current_batch_data();
        return ctx;
    } while (false);

    return std::make_unique<llama_kv_cache_paged_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

// Used by llama_context scheduler to dry-run
llama_memory_context_ptr llama_kv_cache_paged::init_full() {
    LLAMA_LOG_DEBUG("%s: reserving graph for n_ubatch=%d, n_seq_max=%d, num_gpu_blocks=%d\n", __func__, n_ubatch,
                    n_seq_max, num_gpu_blocks);

    // Create a "dummy" ubatch that represents the maximum capacity
    // of the system to let the scheduler reserve enough space for metadata.
    llama_ubatch ubatch = {};
    ubatch.n_tokens     = n_ubatch;   // maximum tokens
    ubatch.n_seqs       = n_seq_max;  // maximum sequences
    ubatch.n_pos        = 1;

    std::vector<llama_ubatch> ubatches = { ubatch };

    auto ctx = std::make_unique<llama_kv_cache_paged_context>(this, ubatches);

    ctx->set_batch_size(n_seq_max);       // maximum possible sequences
    ctx->set_n_tokens(n_ubatch);          // representative token count
    ctx->set_max_blocks(num_gpu_blocks);  // every block could theoretically belong to one seq

    return ctx;
}

llama_memory_context_ptr llama_kv_cache_paged::init_update(llama_context * /*lctx*/, bool /*optimize*/) {
    // Paged KV has no host-side pending updates (apply is a no-op). Returning
    // SUCCESS made the generic memory_update path reset the previous graph and
    // rebuild the worst-case reserve graph on every decode step, which broke
    // graph reuse and added ~4 ms/step host overhead.
    return std::make_unique<llama_kv_cache_paged_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

struct ggml_tensor * llama_kv_cache_paged::get_k_tensor(int layer_idx) const {
    GGML_ASSERT(layer_idx >= 0 && (size_t) layer_idx < k_gpu_layers.size());
    return k_gpu_layers[layer_idx];
}

struct ggml_tensor * llama_kv_cache_paged::get_v_tensor(int layer_idx) const {
    GGML_ASSERT(layer_idx >= 0 && (size_t) layer_idx < v_gpu_layers.size());
    return v_gpu_layers[layer_idx];
}

void llama_kv_cache_paged::clear(bool data) {
    sequence_positions.clear();
    prefix_index.clear();

    block_manager = llama_block_manager();
    block_manager.init(num_gpu_blocks, num_cpu_blocks, watermark);
    // The fresh block manager lost the eviction policy: re-install it so that
    // llama_memory_clear() does not silently disable it.
    block_manager.set_pool_kind(LLAMA_PAGED_POOL_FULL);
    block_manager.set_evict_pick_cb(evict_pick_cb_, evict_pick_ud_);

    if (data) {
        for (const auto * pool : { &k_gpu_layers, &v_gpu_layers, &k_cpu_layers, &v_cpu_layers }) {
            for (auto * t : *pool) {
                if (t) {
                    ggml_backend_tensor_memset(t, 0, 0, ggml_nbytes(t));
                }
            }
        }
    }
}

bool llama_kv_cache_paged::seq_rm(llama_seq_id seq_id, llama_pos /*p0*/, llama_pos /*p1*/) {
    sequence_positions.erase(seq_id);
    return true;
}

llama_pos llama_kv_cache_paged::seq_pos_min(llama_seq_id seq_id) const {
    auto it = sequence_positions.find(seq_id);
    return (it != sequence_positions.end()) ? it->second.min : -1;
}

llama_pos llama_kv_cache_paged::seq_pos_max(llama_seq_id seq_id) const {
    auto it = sequence_positions.find(seq_id);
    return (it != sequence_positions.end()) ? it->second.max : -1;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_paged::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> breakdown;
    const size_t                                 n_gpu_kvs = k_gpu_layers.size();
    const size_t                                 n_cpu_kvs = k_cpu_layers.size();

    for (size_t il = 0; il < n_layers; ++il) {
        auto * k_gpu = (il < n_gpu_kvs) ? k_gpu_layers[il] : nullptr;
        auto * v_gpu = (il < n_gpu_kvs) ? v_gpu_layers[il] : nullptr;
        if (k_gpu) {
            breakdown[ggml_backend_buffer_get_type(k_gpu->buffer)] += ggml_nbytes(k_gpu);
        }
        if (v_gpu) {
            breakdown[ggml_backend_buffer_get_type(v_gpu->buffer)] += ggml_nbytes(v_gpu);
        }
        auto * k_cpu = (il < n_cpu_kvs) ? k_cpu_layers[il] : nullptr;
        auto * v_cpu = (il < n_cpu_kvs) ? v_cpu_layers[il] : nullptr;
        if (k_cpu) {
            breakdown[ggml_backend_buffer_get_type(k_cpu->buffer)] += ggml_nbytes(k_cpu);
        }
        if (v_cpu) {
            breakdown[ggml_backend_buffer_get_type(v_cpu->buffer)] += ggml_nbytes(v_cpu);
        }
    }
    return breakdown;
}

void llama_kv_cache_paged::set_seq_min_pos(llama_seq_id seq_id, llama_pos new_min) {
    sequence_positions[seq_id].min = new_min;
}

void llama_kv_cache_paged::set_seq_max_pos(llama_seq_id seq_id, llama_pos new_max) {
    sequence_positions[seq_id].max = new_max;
}

// llama_kv_cache_paged_context

void llama_kv_cache_paged_context::set_batch_data(const llama_paged_batch_info & info) {
    batch_info          = &info;
    paged_write_slots   = info.write_slots;
    paged_block_table   = info.block_table;
    paged_context_lens  = info.context_lens;
    paged_batch_offsets = info.batch_offsets;
    paged_batch_lens    = info.batch_lens;
    n_tokens            = info.n_tokens;
    max_blocks          = info.n_blocks_per_seq;
    batch_size          = info.n_seq;
}

void llama_kv_cache_paged_context::derive_current_batch_data() {
    if (!batch_info || ubatches.empty()) {
        return;
    }

    const llama_ubatch & cur = ubatches[i_cur];
    const int32_t n_seq_tokens = (int32_t) cur.n_seq_tokens;

    // Original batch index of every ubatch token. The ubatch is a split of the
    // scheduler's flat batch and may contain only a subset of its sequences
    // (an earlier sequence can be exhausted while a later one still has rows),
    // so the ubatch position is not the batch slot.
    const int32_t * batch_idxs = nullptr;
    if (cur.data && (int32_t) cur.data->batch_idxs.size() == (int32_t) cur.n_tokens) {
        batch_idxs = cur.data->batch_idxs.data();
    }

    local_write_slots.resize(cur.n_tokens);
    local_context_lens.resize(cur.n_seqs);
    local_batch_offsets.resize(cur.n_seqs);
    local_batch_lens.resize(cur.n_seqs);
    // The graph sees one block-table row per ubatch sequence, so gather those
    // rows by batch slot as well.
    local_block_table.resize((size_t) cur.n_seqs * (size_t) max_blocks);

    int32_t local_offset = 0;
    for (int32_t s = 0; s < (int32_t) cur.n_seqs; ++s) {
        const llama_seq_id seq_id = cur.seq_id[s * n_seq_tokens][0];

        // flat batch index of this sequence's first token in this ubatch
        const int32_t seq_base = batch_idxs
                                     ? batch_idxs[s * n_seq_tokens]
                                     : batch_info->batch_offsets[s] + seq_token_offsets[seq_id];

        // batch slot whose token range contains seq_base
        int32_t slot = -1;
        for (int32_t c = 0; c < batch_info->n_seq; ++c) {
            if (seq_base >= batch_info->batch_offsets[c] &&
                seq_base < batch_info->batch_offsets[c] + batch_info->batch_lens[c]) {
                slot = c;
                break;
            }
        }
        GGML_ASSERT(slot >= 0 && "paged metadata does not cover the ubatch sequence");

        local_batch_offsets[s] = local_offset;
        local_batch_lens[s]    = n_seq_tokens;
        const int32_t seq_last =
            batch_idxs ? batch_idxs[s * n_seq_tokens + n_seq_tokens - 1] : seq_base + n_seq_tokens - 1;
        GGML_ASSERT(seq_last >= batch_info->batch_offsets[slot] &&
                    seq_last < batch_info->batch_offsets[slot] + batch_info->batch_lens[slot]);
        // Context length counts KV rows, not the temporal coordinate of an image.
        local_context_lens[s] = batch_info->context_lens[slot] - batch_info->batch_lens[slot] + seq_last -
                                batch_info->batch_offsets[slot] + 1;

        const int32_t * src_blocks = batch_info->block_table + (size_t) slot * (size_t) max_blocks;
        std::copy(src_blocks, src_blocks + max_blocks,
                  local_block_table.begin() + (size_t) s * (size_t) max_blocks);

        for (int32_t t = 0; t < n_seq_tokens; ++t) {
            const int32_t bi = batch_idxs ? batch_idxs[s * n_seq_tokens + t] : seq_base + t;
            local_write_slots[local_offset + t] = batch_info->write_slots[bi];
        }

        if (!batch_idxs) {
            seq_token_offsets[seq_id] += n_seq_tokens;
        }

        local_offset += n_seq_tokens;
    }

    n_tokens   = cur.n_tokens;
    batch_size = cur.n_seqs;

    paged_write_slots   = local_write_slots.data();
    paged_block_table   = local_block_table.data();
    paged_context_lens  = local_context_lens.data();
    paged_batch_offsets = local_batch_offsets.data();
    paged_batch_lens    = local_batch_lens.data();
}

bool llama_kv_cache_paged_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    if (++i_cur >= ubatches.size()) {
        return false;
    }

    derive_current_batch_data();
    return true;
}

bool llama_kv_cache_paged_context::apply() {
    // Nothing to do for paged KV cache, return true to allow for execution
    return true;
}

const llama_ubatch & llama_kv_cache_paged_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_cur];
}

struct ggml_tensor * llama_kv_cache_paged_context::get_k(int layer_idx) const {
    GGML_ASSERT(manager && "manager has not been initialized.");
    return manager->get_k_tensor(layer_idx);
}

struct ggml_tensor * llama_kv_cache_paged_context::get_v(int layer_idx) const {
    GGML_ASSERT(manager && "manager has not been initialized.");
    return manager->get_v_tensor(layer_idx);
}

int32_t llama_kv_cache_paged_context::get_n_tokens() const {
    return n_tokens;
}

int32_t llama_kv_cache_paged_context::get_batch_size() const {
    return batch_size;
}

int32_t llama_kv_cache_paged_context::get_max_blocks() const {
    return max_blocks;
}

int32_t * llama_kv_cache_paged_context::get_write_slots() const {
    return paged_write_slots;
}

int32_t * llama_kv_cache_paged_context::get_block_table() const {
    return paged_block_table;
}

int32_t * llama_kv_cache_paged_context::get_context_lens() const {
    return paged_context_lens;
}

int32_t * llama_kv_cache_paged_context::get_batch_offsets() const {
    return paged_batch_offsets;
}

int32_t * llama_kv_cache_paged_context::get_batch_lens() const {
    return paged_batch_lens;
}

void llama_kv_cache_paged_context::set_n_tokens(int32_t new_n_tokens) {
    n_tokens = new_n_tokens;
}

void llama_kv_cache_paged_context::set_batch_size(int32_t new_batch_size) {
    batch_size = new_batch_size;
}

void llama_kv_cache_paged_context::set_max_blocks(int32_t new_max_blocks) {
    max_blocks = new_max_blocks;
}
