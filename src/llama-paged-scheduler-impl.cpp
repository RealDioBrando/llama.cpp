#include "llama-paged-scheduler-impl.h"

#include "llama-impl.h"

#include "llama-paged-prefix-cache.h"

#include <algorithm>

llama_paged_scheduler_impl::llama_paged_scheduler_impl(uint32_t               n_ctx,
                                                       uint32_t               block_sz,
                                                       int32_t                n_batch,
                                                       llama_kv_cache_paged * kv_manager,
                                                       llama_memory_hybrid_paged * hybrid_memory) :
    n_seq_max_ctx(n_ctx),
    block_size(block_sz),
    n_batch(n_batch),
    kv_cache_manager(kv_manager),
    hybrid_memory_manager(hybrid_memory) {
    if (const char * env = getenv("LLAMA_PAGED_PREFIX_CACHE")) {
        prefix_cache_enabled = atoi(env) != 0;
    }
    prefix_cache.set_enabled(prefix_cache_enabled);

    if (const char * env = getenv("LLAMA_PAGED_PREFIX_CKPT")) {
        const int v = atoi(env);
        n_ckpt_requested = (uint32_t)(v > 0 ? v : 0);
    }

    if (prefix_cache_enabled && hybrid_memory_manager && n_ckpt_requested > 0) {
        const uint32_t n_seq_max = hybrid_memory_manager->get_n_seq_max();
        if (n_seq_max > n_ckpt_requested) {
            prefix_cache.set_ckpt_slots((int32_t)(n_seq_max - n_ckpt_requested), n_ckpt_requested);
        } else {
            LLAMA_LOG_INFO(
                    "%s: prefix cache checkpoints disabled: n_seq_max (%d) must exceed the requested slot count (%d)\n",
                    __func__, (int) n_seq_max, (int) n_ckpt_requested);
        }
    }

    kv_cache_manager->set_block_move_cb([this](uint64_t hash, uint32_t new_block_id) {
        prefix_cache.update_block_id(hash, new_block_id);
    });
}

bool llama_paged_scheduler_impl::check_deadlock(uint32_t n_candidates, uint32_t n_swapped, uint32_t n_waiting) const {
    if (n_candidates == 0 && (n_swapped > 0 || n_waiting > 0)) {
        LLAMA_LOG_ERROR(
            "%s: Scheduler deadlock detected. "
            "%d sequence(s) are swapped out and %d are waiting, "
            "but there are not enough free GPU blocks to make progress. "
            "Hint: increase n_gpu_blocks (currently %d) or reduce n_sequences.\n",
            __func__, n_swapped, n_waiting, kv_cache_manager->get_num_gpu_blocks());
        return true;
    }
    return false;
}

bool llama_paged_scheduler_impl::check_livelock(uint32_t n_swapped, uint32_t prev_n_swapped) {
    // swapped count is non-zero and not decreasing
    if (n_swapped > 0 && n_swapped >= prev_n_swapped) {
        n_livelock_steps++;
        if (n_livelock_steps >= max_livelock_steps) {
            LLAMA_LOG_ERROR(
                "%s: Livelock detected. Swapped count has been "
                "non-decreasing for %d steps (currently %d swapped). "
                "Increase n_gpu_blocks (currently %d) or reduce "
                "n_sequences.\n",
                __func__, n_livelock_steps, n_swapped, kv_cache_manager->get_num_gpu_blocks());
            return true;
        }
    } else {
        // Swapped count decreased — swap-ins are happening, reset counter
        n_livelock_steps = 0;
    }
    return false;
}

int32_t llama_paged_scheduler_impl::get_curr_decode_tokens() const {
    return running.size();
}

llama_scheduler_status llama_paged_scheduler_impl::step(llama_batch & batch) {
    // Free previous inference batches
    clear_batch(batch);

    llama_sequence_group_raw_list candidates;
    process_running_list(candidates);
    process_swapped_list(candidates);

    // A paged ubatch assumes every sequence contributes the same number of
    // tokens. Keep prefill and decode in separate scheduling steps.
    // Drain queued prefills before decoding. Otherwise the first ready
    // sequence starts decoding while later sequences wait and a queued batch
    // runs with a tiny active batch.
    if (!waiting.empty()) {
        process_waiting_list(candidates, n_batch);
    }

    auto prefill_it = std::find_if(candidates.begin(), candidates.end(),
                                   [](const llama_sequence_group * group) {
                                       return group && group->n_decoded == 0;
                                   });
    if (prefill_it != candidates.end()) {
        llama_sequence_group * prefill = *prefill_it;
        candidates.clear();
        candidates.push_back(prefill);
    }

    const uint32_t n_running    = running.size();
    const uint32_t n_swapped    = swapped.size();
    const uint32_t n_waiting    = waiting.size();
    const uint32_t n_candidates = candidates.size();

    LLAMA_LOG_INFO("%s: Scheduler status: running=%d, swapped=%d, waiting=%d, candidates=%d\n", __func__, n_running,
                   n_swapped, n_waiting, n_candidates);

    const bool deadlock = check_deadlock(n_candidates, n_swapped, n_waiting);
    const bool livelock = check_livelock(n_swapped, prev_n_swapped);  // updates n_livelock_steps
    if (deadlock || livelock) {
        return llama_scheduler_status::DEADLOCK;
    }

    prev_n_swapped = n_swapped;
    populate_batch_from(candidates, batch);
    kv_cache_manager->set_paged_batch_info(&curr_info);
    return llama_scheduler_status::OK;
}

bool llama_paged_scheduler_impl::queue_request(llama_sequence_group group) {
    // Rejecting any requests that exceeds max context for a seq
    if (group.n_prompt >= n_seq_max_ctx) {
        LLAMA_LOG_ERROR("%s: request %d exceeds max context (%d > %d).\n", __func__, group.request_id, group.n_prompt,
                        n_seq_max_ctx);
        return false;
    }

    if (prefix_cache.seq_reserved(group.request_id)) {
        LLAMA_LOG_ERROR("%s: request %d uses a seq id reserved for prefix-cache checkpoints.\n", __func__,
                        group.request_id);
        return false;
    }

    group.n_prompt_total = (uint32_t) group.logical_seq.size();

    auto group_ptr = std::make_unique<llama_sequence_group>(std::move(group));

    id_to_group[group_ptr->request_id] = group_ptr.get();

    try_apply_prefix_cache(*group_ptr);

    set_waiting(std::move(group_ptr));
    return true;
}

uint64_t llama_paged_scheduler_impl::hash_block(uint64_t prev, const std::vector<llama_token> & tokens,
                                                uint32_t block_idx, uint32_t block_size) {
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

void llama_paged_scheduler_impl::try_apply_prefix_cache(llama_sequence_group & group) {
    if (!prefix_cache_enabled || group.n_past > 0 || !group.block_table.empty()) {
        return;
    }

    const uint32_t max_blocks = (uint32_t)(group.logical_seq.size() / block_size);
    if (max_blocks == 0) {
        return;
    }

    llama_block_ids      shared_ids;
    std::vector<uint64_t> hashes;
    uint64_t             h = 0;
    for (uint32_t b = 0; b < max_blocks; ++b) {
        h               = hash_block(h, group.logical_seq, b, block_size);
        const auto * e = prefix_cache.find_block(h, group.logical_seq[b * block_size],
                                                 group.logical_seq[(b + 1) * block_size - 1]);
        if (!e || kv_cache_manager->block_ref_count(e->block_id) == 0 || !kv_cache_manager->is_block_gpu(e->block_id)) {
            break;
        }
        shared_ids.push_back(e->block_id);
        hashes.push_back(h);
    }

    if (shared_ids.empty()) {
        return;
    }

    // A zero-token suffix cannot produce the first output logits, and
    // replaying the last prompt token would double-apply it to the GDN
    // state. Keep at least one suffix token by dropping the last block.
    while (!shared_ids.empty() && group.logical_seq.size() == shared_ids.size() * block_size) {
        shared_ids.pop_back();
        hashes.pop_back();
    }
    if (shared_ids.empty()) {
        return;
    }

    if (hybrid_memory_manager && prefix_cache.n_ckpt_slots() > 0) {
        const auto * ck = prefix_cache.find_ckpt(hashes.back());
        if (!ck) {
            LLAMA_LOG_DEBUG("%s: no GDN checkpoint for the matched prefix, falling back to full prefill\n", __func__);
            return;
        }
        // Fork the GDN state at the matched boundary; the first suffix
        // ubatch triggers the recurrent cell COW split.
        hybrid_memory_manager->seq_cp(ck->ckpt_seq, group.request_id, -1, -1);
    }

    if (!kv_cache_manager->share_blocks(group, shared_ids)) {
        return;
    }

    group.block_hashes = std::move(hashes);
    for (const uint64_t h : group.block_hashes) {
        prefix_cache.set_cached_state(h, false);
    }
    group.n_past   = (uint32_t) shared_ids.size() * block_size;
    group.n_prompt = (uint32_t) group.logical_seq.size() - group.n_past;

    LLAMA_LOG("%s: request %d reuses %d cached block(s) (%u tokens), prefilling %u suffix tokens\n", __func__,
              group.request_id, (int) shared_ids.size(), group.n_past, group.n_prompt);
}

void llama_paged_scheduler_impl::register_completed_blocks(llama_sequence_group & group) {
    const uint32_t full_blocks = group.n_past / block_size;
    for (uint32_t b = group.block_hashes.size(); b < full_blocks; ++b) {
        if ((uint64_t)(b + 1) * block_size > group.logical_seq.size() || b >= group.block_table.size()) {
            break;
        }
        const uint64_t prev = (b > 0) ? group.block_hashes[b - 1] : 0;
        const uint64_t h    = hash_block(prev, group.logical_seq, b, block_size);
        prefix_cache.register_block(h, group.block_table[b], group.request_id, b, group.logical_seq[b * block_size],
                                    group.logical_seq[(b + 1) * block_size - 1]);
        group.block_hashes.push_back(h);
    }
}

void llama_paged_scheduler_impl::maybe_create_checkpoint(llama_sequence_group & group) {
    if (group.block_hashes.size() < group.n_past / block_size) {
        return;
    }
    const uint64_t chain_hash = group.block_hashes.back();
    const int32_t  slot_seq   = prefix_cache.acquire_ckpt_slot(chain_hash, (llama_pos) group.n_past);
    if (slot_seq < 0) {
        return;
    }
    hybrid_memory_manager->seq_rm(slot_seq, -1, -1);
    hybrid_memory_manager->seq_cp(group.request_id, slot_seq, -1, -1);
    LLAMA_LOG_DEBUG("%s: checkpoint for request %d at pos %d bound to seq %d\n", __func__, group.request_id,
                    (int) group.n_past, slot_seq);
}

bool llama_paged_scheduler_impl::evict_cached_blocks() {
    uint64_t hash = 0;
    uint32_t block_id = 0;
    if (!prefix_cache.pop_lru_cached(hash, block_id)) {
        return false;
    }
    kv_cache_manager->release_block(block_id);
    LLAMA_LOG("%s: evicted cache-resident prefix block %u\n", __func__, block_id);
    return true;
}

void llama_paged_scheduler_impl::insert_sorted_by_arrival_time(llama_sequence_group_ptr    new_group_ptr,
                                                               llama_sequence_group_list & list) {
    GGML_ASSERT(new_group_ptr && "New group cannot be sorted because it's nullptr.");
    auto it =
        std::lower_bound(list.begin(), list.end(), new_group_ptr->t_arrival_time,
                         [](const llama_sequence_group_ptr & group, int64_t time) {
                             GGML_ASSERT(group && "group cannot be checked for arrival time because it's nullptr.");
                             return group->t_arrival_time < time;
                         });
    list.insert(it, std::move(new_group_ptr));
}

void llama_paged_scheduler_impl::set_running(llama_sequence_group_ptr group_ptr) {
    GGML_ASSERT(group_ptr && group_ptr->status != llama_sequence_group_status::RUNNING &&
                "Request is already running.");
    group_ptr->status = llama_sequence_group_status::RUNNING;
    insert_sorted_by_arrival_time(std::move(group_ptr), running);
}

void llama_paged_scheduler_impl::set_swapped(llama_sequence_group_ptr group_ptr) {
    GGML_ASSERT(group_ptr && group_ptr->status != llama_sequence_group_status::SWAPPED &&
                "Request is already swapped.");
    group_ptr->status = llama_sequence_group_status::SWAPPED;
    insert_sorted_by_arrival_time(std::move(group_ptr), swapped);
}

void llama_paged_scheduler_impl::set_waiting(llama_sequence_group_ptr group_ptr) {
    GGML_ASSERT(group_ptr && group_ptr->status != llama_sequence_group_status::WAITING &&
                "Request is already waiting.");
    group_ptr->status = llama_sequence_group_status::WAITING;
    insert_sorted_by_arrival_time(std::move(group_ptr), waiting);
}

void llama_paged_scheduler_impl::finish(llama_sequence_group & group) {
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr.");
    GGML_ASSERT(group.status == llama_sequence_group_status::FINISHED && "Request was not marked as finished.");
    // We prioritize user CB, otherwise we log by default
    if (on_finish_cb) {
        // TODO perhaps just have the callback take sequence_group and user_data
        on_finish_cb(group.request_id, group.logical_seq.data(), (int32_t) group.logical_seq.size(),
                     on_finish_user_data);
    } else {
        LLAMA_LOG_DEBUG("%s: Request: %d generated %d tokens.\n", __func__, group.request_id, group.n_decoded);
    }
    // Registered prefix blocks stay cache-resident (ref stays 1) until
    // evicted under allocation pressure.
    llama_block_ids      keep;
    std::vector<uint64_t> keep_hashes;
    for (size_t i = 0; i < group.block_hashes.size() && i < group.block_table.size(); ++i) {
        const uint32_t id = group.block_table[i];
        if (kv_cache_manager->block_ref_count(id) == 1 && kv_cache_manager->is_block_gpu(id)) {
            keep.push_back(id);
            keep_hashes.push_back(group.block_hashes[i]);
        }
    }
    kv_cache_manager->free_blocks(group, keep);
    for (const uint64_t h : group.block_hashes) {
        const bool kept = std::find(keep_hashes.begin(), keep_hashes.end(), h) != keep_hashes.end();
        prefix_cache.set_cached_state(h, kept);
    }
    if (hybrid_memory_manager) {
        hybrid_memory_manager->seq_rm(group.request_id, -1, -1);
    }
    group.status = llama_sequence_group_status::FINISHED;
    id_to_group.erase(group.request_id);
}

// Try to swap a running sequence out to CPU.
// if the CPU pool is full, fall back to recomputation by resetting the sequence's decode state
// and sending it back to the waiting queue.
//
// Takes ownership of group_ptr.
// On return, the sequence is either in the swapped list (CPU pool had room)
// or the waiting list (recomputed).
void llama_paged_scheduler_impl::swap_out_or_recompute(llama_sequence_group_ptr group_ptr) {
    GGML_ASSERT(group_ptr && "group_ptr is nullptr");
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr");

    const int32_t rid = group_ptr->request_id;

    const bool swap_ok = kv_cache_manager->swap_out(*group_ptr);
    if (swap_ok) {
        LLAMA_LOG_DEBUG("%s: (swapped_out) request_id=%d was swapped out to make room.\n", __func__, rid);
        set_swapped(std::move(group_ptr));
        return;
    }

    // There was not enough CPU memory to swap the request (recomputation)
    llama_block_ids      keep;
    std::vector<uint64_t> keep_hashes;
    for (size_t i = 0; i < group_ptr->block_hashes.size() && i < group_ptr->block_table.size(); ++i) {
        const uint32_t id = group_ptr->block_table[i];
        if (kv_cache_manager->block_ref_count(id) == 1 && kv_cache_manager->is_block_gpu(id)) {
            keep.push_back(id);
            keep_hashes.push_back(group_ptr->block_hashes[i]);
        }
    }
    kv_cache_manager->free_blocks(*group_ptr, keep);
    for (const uint64_t h : group_ptr->block_hashes) {
        const bool kept = std::find(keep_hashes.begin(), keep_hashes.end(), h) != keep_hashes.end();
        prefix_cache.set_cached_state(h, kept);
    }
    kv_cache_manager->seq_rm(rid, -1, -1);
    if (hybrid_memory_manager) {
        hybrid_memory_manager->seq_rm(rid, -1, -1);
    }
    group_ptr->n_past    = 0;
    group_ptr->n_decoded = 0;
    group_ptr->logical_seq.resize(group_ptr->n_prompt_total);
    group_ptr->n_prompt  = group_ptr->n_prompt_total;
    group_ptr->block_hashes.clear();
    try_apply_prefix_cache(*group_ptr);

    LLAMA_LOG_DEBUG("%s: (recomputation) request_id=%d was sent for recomputation.\n", __func__, rid);
    set_waiting(std::move(group_ptr));
}

void llama_paged_scheduler_impl::evict() {
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr.");
    LLAMA_LOG_DEBUG("%s: Eviction requested...\n", __func__);
    if (running.empty()) {
        return;
    }

    llama_sequence_group_ptr most_recent_request = std::move(running.back());
    running.pop_back();
    GGML_ASSERT(most_recent_request && "request selected for eviction is nullptr.");

    swap_out_or_recompute(std::move(most_recent_request));
}

void llama_paged_scheduler_impl::process_running_list(llama_sequence_group_raw_list & candidates) {
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr.");

    llama_sequence_group_list::iterator it = running.begin();
    while (it != running.end()) {
        llama_sequence_group * group = it->get();
        GGML_ASSERT(group && "group is nullptr.");

        if (group->status == llama_sequence_group_status::FINISHED) {
            finish(*group);
            it = running.erase(it);
            continue;
        }

        // Dynamically allocate more blocks to decode the request
        uint32_t current_capacity  = group->block_table.size() * block_size;
        uint32_t required_capacity = group->n_past + 1;
        LLAMA_LOG_DEBUG(
            "%s: (running) request_id=%d: current_capacity (tokens)=%d toks, required capacity (tokens) = %d toks\n",
            __func__, group->request_id, current_capacity, required_capacity);
        if (required_capacity >= current_capacity) {
            LLAMA_LOG_DEBUG("%s: (running_pending) request_id=%d: requires a new block to decode.\n", __func__,
                            group->request_id);
            bool success = kv_cache_manager->allocate(1, *group);  // decode phase
            while (!success && evict_cached_blocks()) {
                success = kv_cache_manager->allocate(1, *group);
            }
            if (!success) {
                if (running.size() > 1) {
                    const bool curr_is_back = (std::next(it) == running.end());
                    // Evict pops the back of the list (most recent request)
                    evict();

                    // Evict might have removed the current group from running
                    if (curr_is_back) {
                        // Current group was evicted
                        it = running.end();
                        continue;
                    }

                    // Try allocating again after eviction
                    success = kv_cache_manager->allocate(1, *group);
                }

                if (!success) {
                    // If allocate failed, it means we must evict the current request
                    llama_sequence_group_ptr self = std::move(*it);
                    it                            = running.erase(it);

                    swap_out_or_recompute(std::move(self));
                    continue;
                }
            }
            LLAMA_LOG_DEBUG("%s: (running_restored) request_id=%d: found a new block to continue decoding.\n", __func__,
                            group->request_id);
        }

        // A request is a candidate if we there is still room for generation without adding blocks
        // or if there was enough GPU memory to allocate another physical block.
        candidates.push_back(group);
        ++it;
    }
}

void llama_paged_scheduler_impl::process_swapped_list(llama_sequence_group_raw_list & candidates) {
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr.");
    llama_sequence_group_list::iterator it = swapped.begin();
    while (it != swapped.end()) {
        llama_sequence_group * group = it->get();
        GGML_ASSERT(group && "the group to swap is nullptr.");
        const bool success = kv_cache_manager->swap_in(*group);
        if (!success) {
            // We respect FCFS, so we stop here to prevent a younger swapped request from jumping ahead.
            break;
        }
        candidates.push_back(group);
        llama_sequence_group_ptr group_ptr = std::move(*it);
        LLAMA_LOG_DEBUG("%s: (swapped_in) request_id=%d back in for processing.\n", __func__, group_ptr->request_id);
        set_running(std::move(group_ptr));
        it = swapped.erase(it);
    }
}

void llama_paged_scheduler_impl::process_waiting_list(llama_sequence_group_raw_list & candidates,
                                                      int32_t                         remaining_token_budget) {
    GGML_ASSERT(kv_cache_manager && "kv_cache_manager is nullptr.");
    llama_sequence_group_list::iterator it    = waiting.begin();
    size_t                              count = 0;
    while (it != waiting.end()) {
        llama_sequence_group * group = it->get();
        GGML_ASSERT(group && "the waiting group is nullptr.");

        // The cache may have grown since this request was queued.
        try_apply_prefix_cache(*group);

        const int32_t chunk_tokens = std::min<int32_t>((int32_t) group->n_prompt, remaining_token_budget);
        if (chunk_tokens <= 0) {
            break;
        }
        ++count;
        // Reserve the full prompt plus one decode slot; only chunk_tokens run now.
        bool success = kv_cache_manager->allocate(1, *group);
        while (!success && evict_cached_blocks()) {
            success = kv_cache_manager->allocate(1, *group);
        }
        if (!success) {
            // We respect FCFS, so we stop here to prevent a younger waiting request from jumping ahead.
            break;
        }
        candidates.push_back(group);
        llama_sequence_group_ptr group_ptr = std::move(*it);
        LLAMA_LOG_DEBUG("%s: (start) request_id=%d sent for processing.\n", __func__, group_ptr->request_id);
        set_running(std::move(group_ptr));
        it = waiting.erase(it);
        break;
    }
    if (count > 0) {
        LLAMA_LOG_DEBUG("%s: Started %ld waiting requests\n", __func__, count);
    }
}

int32_t llama_paged_scheduler_impl::calculate_global_slot_index(int32_t                 token_pos,
                                                                std::vector<uint32_t> & block_table) {
    GGML_ASSERT(block_size && "block_size needs to be greater than 0");
    const int32_t block_table_id = token_pos / block_size;
    const int32_t offset         = token_pos % block_size;

    const size_t block_table_size = block_table.size();
    if ((size_t) block_table_id >= block_table_size) {
        LLAMA_LOG_ERROR("%s: block_table_id=%d is OOB for pos=%d. Block table size=%ld.\n", __func__, block_table_id,
                        token_pos, block_table_size);
        LLAMA_LOG_ERROR("%s: block_table_contents: [ ", __func__);
        for (size_t id = 0; id < block_table_size; ++id) {
            LLAMA_LOG_ERROR("%d ", block_table[id]);
            if (id == block_table_size - 1) {
                LLAMA_LOG_ERROR("]\n");
            }
        }
        GGML_ASSERT(false && "block_table_id OOB");
    }
    const int32_t block_id = block_table.at(block_table_id);

    return (block_id * block_size) + offset;
}

void llama_paged_scheduler_impl::clear_batch(llama_batch & batch) {
    LLAMA_LOG_DEBUG("%s: clearing batch.", __func__);
    // Invalidate last scheduled batch info before freeing the arrays
    // (MUST be called before the delete[]).
    kv_cache_manager->set_paged_batch_info(nullptr);

    delete[] curr_info.write_slots;
    delete[] curr_info.block_table;
    delete[] curr_info.context_lens;
    delete[] curr_info.batch_offsets;
    delete[] curr_info.batch_lens;
    curr_info = {};  // reset to defaults

    if (batch.n_tokens == 0) {
        return;
    }

    llama_batch_free(batch);
    batch.n_tokens = 0;
}

void llama_paged_scheduler_impl::populate_batch_from(const llama_sequence_group_raw_list & candidates,
                                                     llama_batch &                         batch) {
    if (candidates.empty()) {
        LLAMA_LOG_DEBUG("%s: No candidates for this step.\n", __func__);
        batch.n_tokens = 0;
        return;
    }
    int32_t total_tokens = 0;
    int32_t batch_size   = candidates.size();
    int32_t max_blocks   = 0;

    LLAMA_LOG_DEBUG("%s: Creating batch from candidates (%d requests). n_batch=%d\n", __func__, batch_size, n_batch);

    // Calculating required sizes
    for (const auto & group : candidates) {
        GGML_ASSERT(group && "candidate request is nullptr.");
        total_tokens += (group->n_decoded > 0) ? 1 : std::min<int32_t>((int32_t) group->n_prompt, n_batch);
        max_blocks = std::max(max_blocks, (int32_t) group->block_table.size());
    }

    GGML_ASSERT(total_tokens <= (int32_t) n_batch && "total_tokens exceeds n_batch — token budget logic is broken");

    // Initialize the batch (assumed it was cleared before)
    batch = llama_batch_init(total_tokens, 0, 1);
    GGML_ASSERT(batch.token != nullptr && "llama_batch_init failed to allocate tokens.");

    batch.n_tokens = total_tokens;

    curr_info.n_seq            = batch_size;
    curr_info.n_tokens         = total_tokens;
    curr_info.n_blocks_per_seq = max_blocks;

    curr_info.write_slots   = new int32_t[total_tokens];
    curr_info.block_table   = new int32_t[batch_size * max_blocks];
    curr_info.context_lens  = new int32_t[batch_size];
    curr_info.batch_offsets = new int32_t[batch_size];
    curr_info.batch_lens    = new int32_t[batch_size];
    LLAMA_LOG_DEBUG("%s: created llama_batch: n_seq=%d, n_tokens=%d, n_blocks_per_seq=%d\n", __func__, curr_info.n_seq,
                    batch.n_tokens, curr_info.n_blocks_per_seq);

    int32_t token_offset = 0;
    for (int seq_id = 0; seq_id < batch_size; ++seq_id) {
        llama_sequence_group * group = candidates[seq_id];
        GGML_ASSERT(group && "Make sure the candidates are not nullptr.");

        const bool    is_prefill = group->n_decoded == 0;
        const int32_t new_tokens = is_prefill ? std::min<int32_t>((int32_t) group->n_prompt, n_batch) : 1;

        if (is_prefill) {
            GGML_ASSERT(group->logical_seq.size() >= (size_t)(group->n_past + new_tokens) &&
                        "logical_seq too small for prefill");
        } else {
            GGML_ASSERT(!group->logical_seq.empty() && "logical_seq empty during decode");
        }

        for (int token_idx = 0; token_idx < new_tokens; ++token_idx) {
            int32_t batch_start_id = token_offset + token_idx;

            batch.token[batch_start_id] =
                    is_prefill ? group->logical_seq[group->n_past + token_idx] : group->logical_seq.back();
            batch.pos[batch_start_id]   = group->n_past + token_idx;  // n_past starts at 0

            batch.n_seq_id[batch_start_id]  = 1;
            batch.seq_id[batch_start_id][0] = group->request_id;

            batch.logits[batch_start_id] = (token_idx == (new_tokens - 1));  // only the last token

            int32_t token_pos                     = group->n_past + token_idx;
            curr_info.write_slots[batch_start_id] = calculate_global_slot_index(token_pos, group->block_table);
            LLAMA_LOG_DEBUG("%s: llama_batch seq_id: %d (req_id %d) token %d: pos: %d, global_slot_idx=%d\n", __func__,
                            seq_id, group->request_id, token_idx, token_pos, curr_info.write_slots[batch_start_id]);
        }

        // Populate block table (1D): [batch_size * max_blocks]
        const int32_t curr_block_table_size = group->block_table.size();
        for (int block = 0; block < max_blocks; ++block) {
            int  flattened_id                   = (seq_id * max_blocks) + block;  // row-major
            bool need_padding                   = block >= curr_block_table_size;
            curr_info.block_table[flattened_id] = need_padding ? 0 : group->block_table[block];
        }

        curr_info.context_lens[seq_id]  = group->n_past + new_tokens;
        curr_info.batch_offsets[seq_id] = token_offset;
        curr_info.batch_lens[seq_id]    = new_tokens;
        token_offset += new_tokens;
    }
}

// new_tokens contain 1 token per sequence in the batch
void llama_paged_scheduler_impl::update(const llama_batch &              batch,
                                        const std::vector<llama_token> & new_tokens,
                                        const int8_t *                   stop_flags) {
    GGML_ASSERT((int32_t) new_tokens.size() >= curr_info.n_seq && "new_tokens size does not match with batch size.");
    GGML_ASSERT(stop_flags != nullptr && "stop_flags can't be null");

    for (int i = 0; i < curr_info.n_seq; ++i) {
        int32_t token_offset = curr_info.batch_offsets[i];
        int32_t request_id   = batch.seq_id[token_offset][0];

        auto it = id_to_group.find(request_id);
        if (it == id_to_group.end()) {
            LLAMA_LOG_WARN("%s: request_id %d not found in scheduler, skipping\n", __func__, request_id);
            continue;
        }

        llama_sequence_group * group = it->second;
        GGML_ASSERT(group && "group is nullptr.");

        const int32_t batch_len      = curr_info.batch_lens[i];
        const bool    final_prefill = batch_len > 1 &&
                                      ((int32_t) group->n_past + batch_len ==
                                       (int32_t) group->logical_seq.size());

        // TTFT
        if ((batch_len == 1 || final_prefill) && group->n_decoded == 0) {
            group->t_first_token_us = ggml_time_us();
        }

        // Setting token ranges
        llama_pos range_min = kv_cache_manager->seq_pos_min(group->request_id);
        if (range_min == -1) {
            kv_cache_manager->set_seq_min_pos(group->request_id, batch.pos[token_offset]);
        }
        int32_t last_token_in_batch_idx = token_offset + batch_len - 1;
        kv_cache_manager->set_seq_max_pos(group->request_id, batch.pos[last_token_in_batch_idx]);

        group->n_past += batch_len;
        if (batch_len == 1 || final_prefill) {
            group->n_decoded++;
            group->logical_seq.push_back(new_tokens[i]);
        }

        if (prefix_cache_enabled) {
            register_completed_blocks(*group);
            if (hybrid_memory_manager && prefix_cache.n_ckpt_slots() > 0 && group->n_past > 0 &&
                group->n_past == group->n_prompt_total) {
                maybe_create_checkpoint(*group);
            }
        }

        // Default stop flags are n_seq_max
        if (stop_flags[i] || group->n_past >= n_seq_max_ctx) {
            group->status = llama_sequence_group_status::FINISHED;
        }
    }
}

void llama_paged_scheduler_impl::set_on_finish(llama_paged_on_finish_cb cb, void * user_data) {
    on_finish_cb        = cb;
    on_finish_user_data = user_data;
}

bool llama_paged_scheduler_impl::set_prefix_cache_enabled(bool enabled) {
    const bool old = prefix_cache_enabled;
    prefix_cache_enabled = enabled;
    prefix_cache.set_enabled(enabled);
    return old;
}

llama_sequence_group * llama_paged_scheduler_impl::get_group_from_id(int32_t request_id) const {
    return id_to_group.count(request_id) ? id_to_group.at(request_id) : nullptr;
}

const llama_paged_batch_info * llama_paged_scheduler_impl::get_curr_batch_info() const {
    return &curr_info;
}
