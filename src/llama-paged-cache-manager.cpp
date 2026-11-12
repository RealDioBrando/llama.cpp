#include "llama-paged-cache-manager.h"

#include "llama-impl.h"

#include <algorithm>

llama_paged_cache_manager::llama_paged_cache_manager(llama_kv_cache_paged *     full,
                                                     llama_linear_state_cache * linear,
                                                     uint32_t                   block_size,
                                                     bool                       prefix_enabled,
                                                     uint32_t                   ckpt_interval,
                                                     uint32_t                   ckpt_base) :
    full_cache(full),
    linear_cache(linear),
    block_size(block_size),
    ckpt_interval(ckpt_interval),
    ckpt_base(ckpt_base) {
    GGML_ASSERT(full_cache && "paged cache manager requires a full-attention cache");
    set_prefix_enabled(prefix_enabled);
}

void llama_paged_cache_manager::set_prefix_enabled(bool enabled) {
    full_cache->set_prefix_enabled(enabled);
    if (linear_cache) {
        linear_cache->set_prefix_enabled(enabled);
    }
}

bool llama_paged_cache_manager::prefix_enabled() const {
    return full_cache->prefix_enabled();
}

void llama_paged_cache_manager::set_evict_pick_cb(llama_paged_evict_pick_cb cb, void * ud) {
    full_cache->set_evict_pick_cb(cb, ud);
    if (linear_cache) {
        linear_cache->set_evict_pick_cb(cb, ud);
    }
}

bool llama_paged_cache_manager::gpu_checkout_refused_by_policy() const {
    return full_cache->gpu_checkout_refused_by_policy() ||
           (linear_cache && linear_cache->gpu_checkout_refused_by_policy());
}

void llama_paged_cache_manager::reset_gpu_checkout_refusal() {
    full_cache->reset_gpu_checkout_refusal();
    if (linear_cache) {
        linear_cache->reset_gpu_checkout_refusal();
    }
}

uint64_t llama_paged_cache_manager::hash_block(uint64_t                         prev,
                                               const std::vector<llama_token> & tokens,
                                               uint32_t                         block_idx,
                                               uint32_t                         block_size) {
    uint64_t h = 1469598103934665603ULL ^ prev;
    for (uint32_t i = block_idx * block_size; i < (block_idx + 1) * block_size; ++i) {
        const llama_token t = tokens[i];
        for (size_t b = 0; b < sizeof(llama_token); ++b) {
            h ^= (uint64_t)((t >> (8 * b)) & 0xff);
            h *= 1099511628211ULL;
        }
    }
    return h;
}

//
// matching and reservation
//

llama_prefix_candidate llama_paged_cache_manager::match_prefix_candidate(
        const std::vector<llama_token> & tokens) const {
    llama_prefix_candidate c;

    if (!prefix_enabled() || tokens.size() < block_size) {
        return c;
    }

    // Always leave at least one suffix token, otherwise the first output
    // logit cannot be produced and the last prompt token would be applied to
    // the recurrent state twice.
    const uint32_t max_blocks = (uint32_t)((tokens.size() - 1) / block_size);
    if (max_blocks == 0) {
        return c;
    }

    uint64_t h = 0;
    for (uint32_t b = 0; b < max_blocks; ++b) {
        h = hash_block(h, tokens, b, block_size);
        const auto * e = full_cache->find_prefix_block(h, tokens[b * block_size],
                                                       tokens[(b + 1) * block_size - 1]);
        if (!e) {
            break;
        }
        c.block_ids.push_back(e->block_id);
        c.hashes.push_back(h);
    }

    if (c.block_ids.empty()) {
        return c;
    }

    uint32_t n_blocks = (uint32_t) c.block_ids.size();

    if (linear_cache) {
        // A hybrid request can only reuse the boundary covered by a linear
        // checkpoint. Checkpoints are sparse, so walk down to the deepest
        // live checkpoint at or below the full-attention match.
        uint32_t limit = n_blocks;
        const llama_linear_prefix_entry * ckpt = nullptr;
        while (limit > 0) {
            const llama_linear_prefix_entry * cand = linear_cache->find_checkpoint(c.hashes, limit);
            if (!cand) {
                break;
            }
            const uint32_t cand_blocks = (uint32_t)(cand->pos / (llama_pos) block_size);
            if (linear_cache->checkpoint_is_live(*cand)) {
                ckpt = cand;
                break;
            }
            if (cand_blocks == 0) {
                break;
            }
            limit = cand_blocks - 1;
        }
        if (!ckpt) {
            return llama_prefix_candidate{};  // no checkpoint: full prefill
        }
        const uint32_t ckpt_blocks = (uint32_t)(ckpt->pos / (llama_pos) block_size);
        GGML_ASSERT(ckpt_blocks > 0);
        n_blocks = std::min(n_blocks, ckpt_blocks);
        c.has_checkpoint  = true;
        c.checkpoint_hash = ckpt->chain_hash;
    }

    if (n_blocks == 0) {
        return llama_prefix_candidate{};
    }

    c.block_ids.resize(n_blocks);
    c.hashes.resize(n_blocks);
    c.n_blocks  = n_blocks;
    c.reuse_pos = (llama_pos) n_blocks * block_size;
    c.valid     = true;
    return c;
}

bool llama_paged_cache_manager::candidate_is_current(const llama_prefix_candidate & c) const {
    if (!c.valid || c.n_blocks == 0) {
        return false;
    }
    for (uint32_t i = 0; i < c.n_blocks; ++i) {
        // ref_count == 0 is valid: the block is cache-resident in the free
        // queue. Liveness is defined by the block still owning the hash.
        if (!full_cache->prefix_block_is_live(c.block_ids[i], c.hashes[i])) {
            return false;
        }
    }
    if (c.has_checkpoint) {
        const auto * e = linear_cache->find_checkpoint(c.hashes, c.n_blocks);
        if (!e || e->chain_hash != c.checkpoint_hash || !linear_cache->checkpoint_is_live(*e)) {
            return false;
        }
    }
    return true;
}

void llama_paged_cache_manager::clear_prefix_for_full_prefill(llama_sequence_group & group) {
    group.n_past    = 0;
    group.n_prompt  = group.n_prompt_total;
    group.block_table.clear();
    group.block_hashes.clear();
    group.prefix_candidate = {};
}

bool llama_paged_cache_manager::reserve_prefix_candidate(llama_sequence_group & group) {
    const llama_seq_id seq_id = group.request_id;

    // Idempotent: an admission attempt that failed on KV capacity may leave
    // the reservation in place until the caller explicitly releases it.
    if (group.cache_reserved) {
        return true;
    }

    if (linear_cache && !linear_cache->ensure_active(seq_id, group.tag)) {
        LLAMA_LOG_WARN("%s: request %d cannot obtain an active state block\n", __func__, seq_id);
        return false;
    }

    if (!prefix_enabled()) {
        clear_prefix_for_full_prefill(group);
        group.cache_reserved = true;
        return true;
    }

    // Revalidate; the cache may have changed while the request was waiting.
    if (!candidate_is_current(group.prefix_candidate)) {
        group.prefix_candidate = match_prefix_candidate(group.logical_seq);
    }
    const llama_prefix_candidate c = group.prefix_candidate;

    if (!c.valid || c.n_blocks == 0) {
        clear_prefix_for_full_prefill(group);
        group.cache_reserved = true;
        return true;
    }

    // Copy the checkpoint into the active state block first: it can still
    // fail on eviction and must not leave block refs behind.
    if (linear_cache && c.has_checkpoint) {
        if (!linear_cache->fork_checkpoint(c.checkpoint_hash, seq_id, group.tag)) {
            LLAMA_LOG_DEBUG("%s: request %d checkpoint became stale, rematching\n", __func__, seq_id);
            group.prefix_candidate = match_prefix_candidate(group.logical_seq);
            const llama_prefix_candidate c2 = group.prefix_candidate;
            if (!c2.valid || c2.n_blocks == 0 || !candidate_is_current(c2) ||
                !linear_cache->fork_checkpoint(c2.checkpoint_hash, seq_id, group.tag)) {
                if (!linear_cache->ensure_active(seq_id, group.tag)) {
                    return false;
                }
                clear_prefix_for_full_prefill(group);
                return true;
            }
            // fall through with the refreshed candidate
            if (!full_cache->acquire_blocks(c2.block_ids, group.tag)) {
                linear_cache->release(seq_id);
                return false;
            }
            group.block_table  = c2.block_ids;
            group.block_hashes = c2.hashes;
            group.n_past       = c2.reuse_pos;
            group.n_prompt     = (uint32_t) group.logical_seq.size() - group.n_past;
            group.prefix_candidate = c2;
            group.cache_reserved   = true;
            return true;
        }
    }

    if (!full_cache->acquire_blocks(c.block_ids, group.tag)) {
        if (linear_cache) {
            linear_cache->release(seq_id);
            linear_cache->ensure_active(seq_id, group.tag);
        }
        clear_prefix_for_full_prefill(group);
        group.cache_reserved = true;
        return true;
    }

    group.block_table  = c.block_ids;
    group.block_hashes = c.hashes;
    LLAMA_LOG_DEBUG("%s: request %d reuses %u cached block(s) (%d tokens)\n", __func__, seq_id, c.n_blocks,
                    (int) c.reuse_pos);
    group.n_past       = c.reuse_pos;
    group.n_prompt     = (uint32_t) group.logical_seq.size() - group.n_past;
    group.cache_reserved = true;
    return true;
}

bool llama_paged_cache_manager::allocate_blocks(llama_sequence_group & group, int32_t num_tokens) {
    // One admission transaction. A group that has not been admitted yet gets
    // its linear ACTIVE state block, prefix-cache refs and full-attention KV
    // slots together: it either becomes fully runnable or keeps holding
    // nothing. Callers only have to leave it in the waiting queue on failure
    // (vLLM allocate_slots() == None semantics).
    const bool newly_admitted = !group.cache_reserved;
    if (newly_admitted && !reserve_prefix_candidate(group)) {
        // The linear pool is exhausted. reserve_prefix_candidate() leaves no
        // reservation behind on this path (ensure_active / fork_checkpoint are
        // atomic), so there is nothing to roll back.
        return false;
    }

    if (!full_cache->allocate(num_tokens, group)) {
        if (newly_admitted) {
            // Undo exactly what this call reserved. A running group keeps its
            // KV/linear state: its caller preempts or swaps it out instead.
            release_request(group);
            group.n_past   = 0;
            group.n_prompt = group.n_prompt_total;
        }
        return false;
    }
    return true;
}

//
// post-decode hooks
//

void llama_paged_cache_manager::register_tokens(llama_sequence_group & group) {
    // media rows carry no token ids: their blocks can neither be hashed nor
    // safely reused (9.4)
    if (prefix_enabled() && !group.has_media) {
        const uint32_t full_blocks = group.n_past / block_size;
        for (uint32_t b = (uint32_t) group.block_hashes.size(); b < full_blocks; ++b) {
            if ((uint64_t)(b + 1) * block_size > group.logical_seq.size() || b >= group.block_table.size()) {
                break;
            }
            const uint64_t prev = (b > 0) ? group.block_hashes[b - 1] : 0;
            const uint64_t h    = hash_block(prev, group.logical_seq, b, block_size);
            full_cache->register_prefix_block(h, group.block_table[b], group.request_id, b,
                                              group.logical_seq[b * block_size],
                                              group.logical_seq[(b + 1) * block_size - 1]);
            group.block_hashes.push_back(h);
        }
    }

    // Position tracking is needed for bounded rollback even when prefix
    // caching is disabled.
    if (linear_cache && group.n_past > 0) {
        linear_cache->set_seq_pos(group.request_id, (llama_pos) group.n_past - 1);
    }
}

void llama_paged_cache_manager::maybe_create_checkpoint(llama_sequence_group & group) {
    if (!linear_cache || !prefix_enabled() || group.has_media || group.n_past == 0 || ckpt_interval == 0 ||
        block_size == 0 || !linear_cache->has_checkpoint_capacity()) {
        return;
    }

    // A checkpoint must describe an actually processed, complete token-block
    // boundary (doc invariant 9.3 #11).
    if (group.n_past % block_size != 0) {
        return;
    }

    const uint32_t b = group.n_past / block_size;
    // Global congruent lattice (doc 5.3): every request creates checkpoints at
    // the same block indices so shared prefixes can reuse them across requests.
    if (b == 0 || b % ckpt_interval != ckpt_base % ckpt_interval) {
        return;
    }
    if (group.block_hashes.size() < b) {
        return;  // hashes are registered by register_tokens()
    }

    const uint64_t hash = group.block_hashes[b - 1];
    if (linear_cache->has_live_checkpoint(hash)) {
        return;  // never allocate a second block for the same chain
    }
    linear_cache->create_checkpoint(group.request_id, hash, (llama_pos) group.n_past);
}

void llama_paged_cache_manager::truncate_sequence(llama_sequence_group & group, llama_pos n_past_new) {
    const uint32_t n_keep_blocks = (uint32_t) ((n_past_new + (llama_pos) block_size - 1) / block_size);
    if (n_keep_blocks < group.block_table.size()) {
        const llama_block_ids tail(group.block_table.begin() + n_keep_blocks, group.block_table.end());
        for (const uint32_t block_id : tail) {
            // Speculative/draft blocks are normally unregistered, but never
            // let a discarded block keep a valid prefix entry.
            if (full_cache->block_ref_count(block_id) == 1) {
                full_cache->invalidate_prefix_block(block_id);
            }
            full_cache->release_block(block_id);
        }
        group.block_table.resize(n_keep_blocks);
    }
    if (group.block_hashes.size() > n_keep_blocks) {
        group.block_hashes.resize(n_keep_blocks);
    }
    group.n_past = (uint32_t) n_past_new;

    // Roll the recurrent state back through one of the saved snapshot planes.
    // The distance is bounded by n_rs_seq; a larger rollback means the caller
    // decoded more draft rows than the state cache can undo.
    if (linear_cache && !group.has_media) {
        if (!linear_cache->seq_rm(group.request_id, n_past_new, -1)) {
            LLAMA_LOG_WARN("%s: request %d cannot roll back linear state to %d\n", __func__, group.request_id,
                           (int) n_past_new);
        }
    }
}

//
// swapping
//

bool llama_paged_cache_manager::can_swap_out(const llama_sequence_group & group) const {
    llama_sequence_group & g = const_cast<llama_sequence_group &>(group);
    if (!full_cache->can_swap_out(g)) {
        return false;
    }
    return !linear_cache || linear_cache->can_swap_out(group.request_id);
}

bool llama_paged_cache_manager::can_swap_in(const llama_sequence_group & group) const {
    llama_sequence_group & g = const_cast<llama_sequence_group &>(group);
    if (!full_cache->can_swap_in(g)) {
        return false;
    }
    return !linear_cache || linear_cache->can_swap_in(group.request_id);
}

bool llama_paged_cache_manager::swap_out(llama_sequence_group & group) {
    if (!can_swap_out(group)) {
        return false;
    }
    if (linear_cache && !linear_cache->swap_out(group.request_id)) {
        return false;
    }
    if (!full_cache->swap_out(group)) {
        if (linear_cache) {
            linear_cache->swap_in(group.request_id);  // rollback: the GPU block was just freed
        }
        return false;
    }
    return true;
}

bool llama_paged_cache_manager::swap_in(llama_sequence_group & group) {
    if (!can_swap_in(group)) {
        return false;
    }
    if (linear_cache && !linear_cache->swap_in(group.request_id)) {
        return false;
    }
    if (!full_cache->swap_in(group)) {
        if (linear_cache) {
            linear_cache->swap_out(group.request_id);
        }
        return false;
    }
    return true;
}

//
// release
//

void llama_paged_cache_manager::release_request(llama_sequence_group & group) {
    // Releasing a block table never drops its prefix-index entries: blocks
    // whose hash is still registered become cache-resident in the free queue.
    full_cache->free_blocks(group);
    if (linear_cache) {
        linear_cache->release(group.request_id);
    }
    group.prefix_candidate = {};
    group.cache_reserved   = false;
}

uint32_t llama_paged_cache_manager::n_free_gpu_blocks() const {
    return full_cache->n_free_gpu_blocks();
}

uint32_t llama_paged_cache_manager::n_free_linear_gpu_blocks() const {
    return linear_cache ? linear_cache->n_free_gpu_blocks() : 0;
}

uint32_t llama_paged_cache_manager::n_gpu_blocks() const {
    return full_cache->get_num_gpu_blocks();
}

void llama_paged_cache_manager::set_paged_batch_info(const llama_paged_batch_info * info) {
    full_cache->set_paged_batch_info(info);
}

llama_pos llama_paged_cache_manager::seq_pos_min(llama_seq_id seq_id) const {
    return full_cache->seq_pos_min(seq_id);
}

llama_pos llama_paged_cache_manager::seq_pos_max(llama_seq_id seq_id) const {
    return full_cache->seq_pos_max(seq_id);
}

void llama_paged_cache_manager::set_seq_min_pos(llama_seq_id seq_id, llama_pos pos) {
    full_cache->set_seq_min_pos(seq_id, pos);
}

void llama_paged_cache_manager::set_seq_max_pos(llama_seq_id seq_id, llama_pos pos) {
    full_cache->set_seq_max_pos(seq_id, pos);
}
