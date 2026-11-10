#include "llama-linear-state-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

llama_linear_state_cache::llama_linear_state_cache(const llama_model & model,
                                                   ggml_type         type_r,
                                                   ggml_type         type_s,
                                                   bool              offload,
                                                   uint32_t          n_gpu_blocks,
                                                   uint32_t          n_cpu_blocks,
                                                   uint32_t          n_seq_max,
                                                   uint32_t          n_rs_seq,
                                                   ggml_backend_t    backend_gpu,
                                                   ggml_backend_t    backend_cpu) :
    n_rs_seq(n_rs_seq),
    n_gpu_blocks(n_gpu_blocks),
    n_cpu_blocks(n_cpu_blocks),
    n_active_reserve(n_seq_max) {
    GGML_ASSERT(n_gpu_blocks > 0 && n_cpu_blocks > 0);

    llama_recurrent_state_pool pool;
    pool.backend_gpu   = backend_gpu;
    pool.backend_cpu   = backend_cpu;
    pool.n_gpu_blocks  = n_gpu_blocks;
    pool.n_cpu_blocks  = n_cpu_blocks;

    mem = std::make_unique<llama_memory_recurrent>(
            model, type_r, type_s, offload,
            /*mem_size=*/ n_gpu_blocks,
            /*n_seq_max=*/ std::max(1u, n_seq_max),
            n_rs_seq,
            [&](int32_t il) { return il >= 0 && model.hparams.is_recr(il); },
            &pool);

    block_manager.init(n_gpu_blocks, n_cpu_blocks, /*watermark=*/ 0.0f);

    LLAMA_LOG_INFO("%s: linear state cache: %u GPU blocks, %u CPU blocks, %u planes\n", __func__,
                   n_gpu_blocks, n_cpu_blocks, 1 + n_rs_seq);
}

//
// helpers
//

bool llama_linear_state_cache::has_active(llama_seq_id seq_id) const {
    return seqs.find(seq_id) != seqs.end();
}

bool llama_linear_state_cache::is_active_on_gpu(llama_seq_id seq_id) const {
    const auto it = seqs.find(seq_id);
    return it != seqs.end() && block_manager.is_gpu(it->second.block_id);
}

void llama_linear_state_cache::release_block(uint32_t block_id) {
    if (block_manager.is_gpu(block_id)) {
        block_manager.release_gpu_blocks({ block_id });
    } else {
        block_manager.release_cpu_blocks({ block_id });
    }
}

bool llama_linear_state_cache::allocate_gpu_block(uint32_t & block_id) {
    while (!block_manager.has_free_gpu_blocks(1)) {
        if (!demote_one_checkpoint() && !evict_one_checkpoint(/*placement=*/1)) {
            return false;
        }
    }
    const auto ids = block_manager.checkout_gpu_blocks(1);
    if (ids.empty()) {
        return false;
    }
    block_id = ids.front();
    return true;
}

bool llama_linear_state_cache::allocate_cpu_block(uint32_t & block_id) {
    while (!block_manager.has_free_cpu_blocks(1)) {
        if (!evict_one_checkpoint(/*placement=*/0)) {
            return false;
        }
    }
    const auto ids = block_manager.checkout_cpu_blocks(1);
    if (ids.empty()) {
        return false;
    }
    block_id = ids.front();
    return true;
}

bool llama_linear_state_cache::alloc_checkpoint_block(uint32_t & block_id) {
    if (block_manager.has_free_gpu_blocks(1)) {
        block_id = block_manager.checkout_gpu_blocks(1).front();
        return true;
    }
    if (block_manager.has_free_cpu_blocks(1)) {
        block_id = block_manager.checkout_cpu_blocks(1).front();
        return true;
    }
    // A checkpoint may land in either pool. Prefer evicting a GPU checkpoint
    // (its data is still hot), then fall back to the CPU pool.
    if (evict_one_checkpoint(/*placement=*/1)) {
        return alloc_checkpoint_block(block_id);
    }
    if (evict_one_checkpoint(/*placement=*/0)) {
        return alloc_checkpoint_block(block_id);
    }
    return false;
}

const llama_linear_prefix_entry * llama_linear_state_cache::oldest_checkpoint(int placement) const {
    const llama_linear_prefix_entry * victim = nullptr;
    for (const auto & [hash, e] : checkpoint_index.entries()) {
        GGML_UNUSED(hash);
        const bool gpu = block_manager.is_gpu(e.block_id);
        if (placement == 1 && !gpu) {
            continue;
        }
        if (placement == 0 && gpu) {
            continue;
        }
        if (!victim || e.last_used < victim->last_used) {
            victim = &e;
        }
    }
    return victim;
}

bool llama_linear_state_cache::evict_checkpoint(uint64_t chain_hash, llama_linear_prefix_entry e) {
    if (!checkpoint_index.remove(chain_hash)) {
        return false;
    }
    // invalidate_block is redundant after remove, but keeps the invariant if
    // two hash chains ever alias the same block
    checkpoint_index.invalidate_block(e.block_id);
    release_block(e.block_id);
    LLAMA_LOG_DEBUG("%s: evicted checkpoint block %u (pos %d)\n", __func__, e.block_id, (int) e.pos);
    return true;
}

bool llama_linear_state_cache::evict_one_checkpoint(int placement) {
    const llama_linear_prefix_entry * victim = oldest_checkpoint(placement);
    if (!victim) {
        return false;
    }
    return evict_checkpoint(victim->chain_hash, *victim);
}

bool llama_linear_state_cache::demote_checkpoint(uint64_t chain_hash, llama_linear_prefix_entry e) {
    GGML_ASSERT(block_manager.is_gpu(e.block_id));

    uint32_t cpu_id = 0;
    if (!block_manager.has_free_cpu_blocks(1)) {
        // No CPU room to demote into: evicting still frees a GPU block.
        return evict_checkpoint(chain_hash, e);
    }
    const auto ids = block_manager.checkout_cpu_blocks(1);
    if (ids.empty()) {
        return false;
    }
    cpu_id = ids.front();

    mem->copy_state_block(e.block_id, /*src_gpu=*/true, cpu_id - n_gpu_blocks, /*dst_gpu=*/false);
    release_block(e.block_id);

    if (!checkpoint_index.insert(chain_hash, cpu_id, block_manager.block_generation(cpu_id), e.pos)) {
        release_block(cpu_id);
        return false;
    }
    LLAMA_LOG_DEBUG("%s: demoted checkpoint block %u to CPU block %u\n", __func__, e.block_id, cpu_id);
    return true;
}

bool llama_linear_state_cache::demote_one_checkpoint() {
    const llama_linear_prefix_entry * victim = oldest_checkpoint(1);
    if (!victim) {
        return false;
    }
    return demote_checkpoint(victim->chain_hash, *victim);
}

//
// active state
//

bool llama_linear_state_cache::ensure_active(llama_seq_id seq_id) {
    if (seqs.count(seq_id)) {
        return true;
    }

    uint32_t block_id = 0;
    if (!allocate_gpu_block(block_id)) {
        LLAMA_LOG_WARN("%s: no GPU state block available for seq %d\n", __func__, seq_id);
        return false;
    }
    mem->zero_state_block(block_id, /*gpu=*/true);

    seq_state st;
    st.block_id = block_id;
    st.pos      = -1;
    st.rollback = 0;
    seqs[seq_id] = st;
    return true;
}

bool llama_linear_state_cache::release(llama_seq_id seq_id) {
    const auto it = seqs.find(seq_id);
    if (it == seqs.end()) {
        return false;
    }
    release_block(it->second.block_id);
    seqs.erase(it);
    return true;
}

bool llama_linear_state_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst) {
    if (seq_id_src == seq_id_dst) {
        return true;
    }
    const auto src = seqs.find(seq_id_src);
    if (src == seqs.end()) {
        // nothing to fork: make sure the destination does not keep stale state
        release(seq_id_dst);
        return false;
    }
    if (seqs.count(seq_id_dst)) {
        release(seq_id_dst);
    }

    uint32_t dst_id = 0;
    if (!allocate_gpu_block(dst_id)) {
        return false;
    }

    const bool src_gpu = block_manager.is_gpu(src->second.block_id);
    const uint32_t src_local = src_gpu ? src->second.block_id : src->second.block_id - n_gpu_blocks;
    mem->copy_state_block(src_local, src_gpu, dst_id, /*dst_gpu=*/true);

    seq_state st;
    st.block_id = dst_id;
    st.pos      = src->second.pos;
    st.rollback = 0;
    seqs[seq_id_dst] = st;
    return true;
}

bool llama_linear_state_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    const auto it = seqs.find(seq_id);
    if (it == seqs.end()) {
        return true;  // nothing to remove
    }
    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    seq_state & st = it->second;
    const bool rm_all = p0 == 0 && p1 == std::numeric_limits<llama_pos>::max();
    if (rm_all) {
        release(seq_id);
        return true;
    }

    // Tail rollback via one of the saved planes.
    if (0 < p0 && p0 <= st.pos && p1 > st.pos) {
        const llama_pos rollback = st.pos - (p0 - 1);
        if (st.rollback == 0 && rollback >= 1 && rollback <= (llama_pos) n_rs_seq) {
            st.rollback = (uint32_t) rollback;
            st.pos      = p0 - 1;
            return true;
        }
        return false;
    }

    if (p0 <= st.pos && st.pos < p1) {
        // the whole state falls into the removed range
        release(seq_id);
        return true;
    }

    return true;
}

void llama_linear_state_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (shift == 0) {
        return;
    }
    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }
    const auto it = seqs.find(seq_id);
    if (it != seqs.end() && p0 <= it->second.pos && it->second.pos < p1) {
        it->second.pos += shift;
    }
}

void llama_linear_state_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
    }
    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }
    const auto it = seqs.find(seq_id);
    if (it != seqs.end() && p0 <= it->second.pos && it->second.pos < p1) {
        it->second.pos /= d;
    }
}

void llama_linear_state_cache::seq_keep(llama_seq_id seq_id) {
    for (auto it = seqs.begin(); it != seqs.end();) {
        if (it->first == seq_id) {
            ++it;
        } else {
            release_block(it->second.block_id);
            it = seqs.erase(it);
        }
    }
}

void llama_linear_state_cache::set_seq_pos(llama_seq_id seq_id, llama_pos pos) {
    const auto it = seqs.find(seq_id);
    if (it != seqs.end()) {
        it->second.pos = pos;
    }
}

llama_pos llama_linear_state_cache::seq_pos_min(llama_seq_id seq_id) const {
    const auto it = seqs.find(seq_id);
    return it == seqs.end() ? -1 : it->second.pos;
}

llama_pos llama_linear_state_cache::seq_pos_max(llama_seq_id seq_id) const {
    const auto it = seqs.find(seq_id);
    // neutral element for the min() in llama_memory_hybrid_paged::seq_pos_max
    return it == seqs.end() ? std::numeric_limits<llama_pos>::max() : it->second.pos;
}

//
// checkpoints
//

bool llama_linear_state_cache::checkpoint_is_live(const llama_linear_prefix_entry & e) const {
    return block_manager.block_ref_count(e.block_id) > 0 &&
           block_manager.block_generation(e.block_id) == e.generation;
}

bool llama_linear_state_cache::has_live_checkpoint(uint64_t chain_hash) const {
    const auto * e = checkpoint_index.find(chain_hash);
    return e && checkpoint_is_live(*e);
}

bool llama_linear_state_cache::has_checkpoint_capacity() const {
    // Slack beyond the active reserve, on either pool (doc 10.5). The pools
    // are not statically partitioned: idle active slots may hold checkpoints.
    const uint32_t gpu_slack = n_gpu_blocks > n_active_reserve ? n_gpu_blocks - n_active_reserve : 0;
    const uint32_t cpu_slack = n_cpu_blocks > n_active_reserve ? n_cpu_blocks - n_active_reserve : 0;
    return gpu_slack + cpu_slack > 0;
}

bool llama_linear_state_cache::create_checkpoint(llama_seq_id seq_id, uint64_t chain_hash, llama_pos pos) {
    const auto it = seqs.find(seq_id);
    if (it == seqs.end() || !block_manager.is_gpu(it->second.block_id)) {
        return false;
    }

    // Idempotent: never allocate a second block for a chain that already has
    // a live checkpoint.
    if (has_live_checkpoint(chain_hash)) {
        return true;
    }

    uint32_t ckpt_id = 0;
    if (!alloc_checkpoint_block(ckpt_id)) {
        LLAMA_LOG_DEBUG("%s: no state block available for checkpoint (seq %d, pos %d)\n", __func__, seq_id,
                        (int) pos);
        return false;
    }

    const bool ckpt_gpu = block_manager.is_gpu(ckpt_id);
    const uint32_t dst_local = ckpt_gpu ? ckpt_id : ckpt_id - n_gpu_blocks;
    mem->copy_state_block(it->second.block_id, /*src_gpu=*/true, dst_local, ckpt_gpu);

    const auto * old = checkpoint_index.find(chain_hash);
    const bool   has_old = old != nullptr;
    const uint32_t old_block = has_old ? old->block_id : 0;

    checkpoint_index.insert(chain_hash, ckpt_id, block_manager.block_generation(ckpt_id), pos);
    if (has_old) {
        checkpoint_index.invalidate_block(old_block);
        release_block(old_block);
    }

    LLAMA_LOG_DEBUG("%s: checkpoint for seq %d at pos %d -> block %u (%s)\n", __func__, seq_id, (int) pos,
                    ckpt_id, ckpt_gpu ? "GPU" : "CPU");
    return true;
}

const llama_linear_prefix_entry * llama_linear_state_cache::find_checkpoint(
        const std::vector<uint64_t> & chain_hashes, uint32_t max_blocks) const {
    return checkpoint_index.find_deepest(chain_hashes, max_blocks);
}

bool llama_linear_state_cache::fork_checkpoint(uint64_t chain_hash, llama_seq_id seq_id) {
    const auto * e = checkpoint_index.find(chain_hash);
    if (!e || !checkpoint_is_live(*e)) {
        return false;
    }

    // The allocation below may demote or evict a checkpoint, possibly this
    // one, so remember the physical source and revalidate it afterwards.
    const uint32_t src_block = e->block_id;
    const uint32_t src_gen   = e->generation;
    const llama_pos pos      = e->pos;

    // Replace any stale active block for this sequence.
    if (seqs.count(seq_id)) {
        release(seq_id);
    }

    uint32_t active_id = 0;
    if (!allocate_gpu_block(active_id)) {
        return false;
    }

    e = checkpoint_index.find(chain_hash);
    if (!e || e->block_id != src_block || e->generation != src_gen || !checkpoint_is_live(*e)) {
        release_block(active_id);
        return false;
    }

    const bool src_gpu = block_manager.is_gpu(src_block);
    const uint32_t src_local = src_gpu ? src_block : src_block - n_gpu_blocks;
    mem->copy_state_block(src_local, src_gpu, active_id, /*dst_gpu=*/true);

    seq_state st;
    st.block_id = active_id;
    // The checkpoint boundary counts processed tokens; seq_state::pos is the
    // last processed token position (boundary - 1).
    st.pos      = pos > 0 ? pos - 1 : -1;
    st.rollback = 0;
    seqs[seq_id] = st;
    checkpoint_index.touch(chain_hash);
    return true;
}

//
// swapping
//

bool llama_linear_state_cache::can_swap_out(llama_seq_id seq_id) const {
    const auto it = seqs.find(seq_id);
    if (it == seqs.end() || !block_manager.is_gpu(it->second.block_id)) {
        return true;
    }
    // A free CPU block, or an evictable CPU checkpoint that frees one.
    return block_manager.n_free_cpu_blocks() > 0 || oldest_checkpoint(/*placement=*/0) != nullptr;
}

bool llama_linear_state_cache::can_swap_in(llama_seq_id seq_id) const {
    const auto it = seqs.find(seq_id);
    if (it == seqs.end() || block_manager.is_gpu(it->second.block_id)) {
        return true;
    }
    return block_manager.n_free_gpu_blocks() > 0 || oldest_checkpoint(/*placement=*/1) != nullptr;
}

bool llama_linear_state_cache::swap_out(llama_seq_id seq_id) {
    const auto it = seqs.find(seq_id);
    if (it == seqs.end() || !block_manager.is_gpu(it->second.block_id)) {
        return true;
    }
    uint32_t cpu_id = 0;
    if (!allocate_cpu_block(cpu_id)) {
        return false;
    }
    mem->copy_state_block(it->second.block_id, /*src_gpu=*/true, cpu_id - n_gpu_blocks, /*dst_gpu=*/false);
    release_block(it->second.block_id);
    it->second.block_id = cpu_id;
    return true;
}

bool llama_linear_state_cache::swap_in(llama_seq_id seq_id) {
    const auto it = seqs.find(seq_id);
    if (it == seqs.end() || block_manager.is_gpu(it->second.block_id)) {
        return true;
    }
    uint32_t gpu_id = 0;
    if (!allocate_gpu_block(gpu_id)) {
        return false;
    }
    mem->copy_state_block(it->second.block_id - n_gpu_blocks, /*src_gpu=*/false, gpu_id, /*dst_gpu=*/true);
    release_block(it->second.block_id);
    it->second.block_id = gpu_id;
    return true;
}

//
// capacity
//

uint32_t llama_linear_state_cache::n_free_gpu_blocks() const {
    return (uint32_t) block_manager.n_free_gpu_blocks();
}

uint32_t llama_linear_state_cache::n_free_cpu_blocks() const {
    return (uint32_t) block_manager.n_free_cpu_blocks();
}

bool llama_linear_state_cache::has_free_gpu_blocks(uint32_t n) const {
    return block_manager.has_free_gpu_blocks(n);
}

bool llama_linear_state_cache::has_free_cpu_blocks(uint32_t n) const {
    return block_manager.has_free_cpu_blocks(n);
}

//
// batch mapping
//

bool llama_linear_state_cache::prepare_batch(const std::vector<llama_ubatch> & ubatches,
                                             std::vector<llama_recurrent_batch_rows> & out) {
    out.clear();
    out.reserve(ubatches.size());
    rollback_to_reset.clear();

    std::unordered_map<llama_seq_id, bool> rollback_consumed;

    for (const llama_ubatch & ubatch : ubatches) {
        GGML_ASSERT(ubatch.equal_seqs());
        const uint32_t n_seqs       = ubatch.n_seqs;
        const uint32_t n_seq_tokens = ubatch.n_seq_tokens;

        llama_recurrent_batch_rows rows;
        rows.n_seqs   = n_seqs;
        rows.n_planes = 1 + n_rs_seq;
        rows.read_rows.assign(n_seqs, -1);
        rows.write_rows.assign((size_t) n_seqs * rows.n_planes, -1);

        for (uint32_t s = 0; s < n_seqs; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[s * n_seq_tokens][0];
            const auto it = seqs.find(seq_id);
            if (it == seqs.end()) {
                LLAMA_LOG_ERROR("%s: seq %d has no active linear state block\n", __func__, seq_id);
                return false;
            }
            const seq_state & st = it->second;
            if (!block_manager.is_gpu(st.block_id)) {
                LLAMA_LOG_ERROR("%s: seq %d active linear state is not on the GPU pool\n", __func__, seq_id);
                return false;
            }

            uint32_t plane = 0;
            if (!rollback_consumed[seq_id]) {
                plane = st.rollback;
                rollback_consumed[seq_id] = true;
            }

            const uint32_t local = st.block_id;  // GPU ids are local ids
            rows.read_rows[s] = (int32_t)(plane * n_gpu_blocks + local);
            for (uint32_t p = 0; p < rows.n_planes; ++p) {
                rows.write_rows[(size_t) s * rows.n_planes + p] = (int32_t)(p * n_gpu_blocks + local);
            }
        }

        out.push_back(std::move(rows));
    }

    // Defer clearing the consumed rollback planes until the caller has
    // successfully created the batch contexts.
    for (const auto & [seq_id, consumed] : rollback_consumed) {
        if (consumed) {
            rollback_to_reset.push_back(seq_id);
        }
    }

    return true;
}

void llama_linear_state_cache::commit_prepared_batch() {
    for (const llama_seq_id seq_id : rollback_to_reset) {
        const auto it = seqs.find(seq_id);
        if (it != seqs.end()) {
            it->second.rollback = 0;
        }
    }
    rollback_to_reset.clear();
}

void llama_linear_state_cache::clear() {
    for (const auto & [seq_id, st] : seqs) {
        GGML_UNUSED(seq_id);
        release_block(st.block_id);
    }
    seqs.clear();

    for (const auto & [hash, e] : checkpoint_index.entries()) {
        GGML_UNUSED(hash);
        release_block(e.block_id);
    }
    checkpoint_index.clear();

    block_manager = llama_block_manager();
    block_manager.init(n_gpu_blocks, n_cpu_blocks, /*watermark=*/ 0.0f);

    rollback_to_reset.clear();

    mem->clear(/*data=*/true);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_linear_state_cache::memory_breakdown() const {
    return mem->memory_breakdown();
}
