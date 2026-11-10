#include "llama-paged-scheduler-impl.h"

#include "llama-impl.h"

#include <algorithm>

llama_paged_scheduler_impl::llama_paged_scheduler_impl(uint32_t               n_ctx,
                                                       uint32_t               block_sz,
                                                       int32_t                n_batch,
                                                       int32_t                n_pos_per_embd,
                                                       llama_kv_cache_paged * kv_manager,
                                                       llama_memory_hybrid_paged * hybrid_memory,
                                                       bool                   priority_enabled,
                                                       uint32_t               ckpt_interval,
                                                       uint32_t               ckpt_base) :
    priority_enabled_(priority_enabled),
    swapped(llama_paged_queue_create(priority_enabled)),
    waiting(llama_paged_queue_create(priority_enabled)),
    n_seq_max_ctx(n_ctx),
    block_size(block_sz),
    n_batch(n_batch),
    n_pos_per_embd(n_pos_per_embd) {
    if (const char * env = getenv("LLAMA_PAGED_PREFIX_CACHE")) {
        prefix_cache_enabled = atoi(env) != 0;
    }

    llama_linear_state_cache * linear = hybrid_memory ? hybrid_memory->get_linear_mem() : nullptr;
    cache_manager = std::make_unique<llama_paged_cache_manager>(kv_manager, linear, block_sz, prefix_cache_enabled,
                                                                ckpt_interval, ckpt_base);
}

bool llama_paged_scheduler_impl::check_deadlock(uint32_t n_candidates, uint32_t n_swapped, uint32_t n_waiting) const {
    if (n_candidates == 0 && (n_swapped > 0 || n_waiting > 0)) {
        LLAMA_LOG_ERROR(
            "%s: Scheduler deadlock detected. "
            "%d sequence(s) are swapped out and %d are waiting, "
            "but there are not enough free GPU blocks to make progress. "
            "Hint: increase n_gpu_blocks (currently %d) or reduce n_sequences. "
            "Free linear state blocks: %d.\n",
            __func__, n_swapped, n_waiting, cache_manager->n_gpu_blocks(),
            (int) cache_manager->n_free_linear_gpu_blocks());
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
                __func__, n_livelock_steps, n_swapped, cache_manager->n_gpu_blocks());
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

int32_t llama_paged_scheduler_impl::prefill_chunk_tokens(const llama_sequence_group & group) const {
    const int32_t available = (int32_t) group.logical_seq.size() - (int32_t) group.n_past;
    int32_t n = std::min<int32_t>(available, n_batch);

    // A hybrid request can only reuse a prefix covered by a linear checkpoint.
    // Stop the prefill at the next lattice boundary so an ubatch never steps
    // over a checkpoint slot that would otherwise be created (doc 5.3.4).
    if (cache_manager->linear() && prefix_cache_enabled && !group.has_media &&
        block_size > 0 && group.n_decoded == 0 && cache_manager->linear()->has_checkpoint_capacity() &&
        cache_manager->ckpt_interval_blocks() > 0) {
        const uint32_t b_now = (uint32_t) (group.n_past / block_size);
        const uint32_t N     = cache_manager->ckpt_interval_blocks();
        const uint32_t base  = cache_manager->ckpt_base_blocks();

        // Smallest delta > 0 with (b_now + delta) % N == base % N.
        const uint32_t rem   = (base % N + N - (b_now % N)) % N;
        const uint32_t delta = rem == 0 ? N : rem;

        const uint64_t next_pos = (uint64_t) (b_now + delta) * block_size;
        if (next_pos > group.n_past) {
            n = std::min(n, (int32_t) (next_pos - group.n_past));
        }
    }

    return n;
}

llama_scheduler_status llama_paged_scheduler_impl::step(llama_batch & batch) {
    // Free previous inference batches
    clear_batch(batch);

    llama_sequence_group_raw_list candidates;
    int32_t row_budget = n_batch;
    process_running_list(candidates, row_budget);
    process_swapped_list(candidates, row_budget);

    // A paged ubatch assumes every sequence contributes the same number of
    // tokens. Keep prefill and decode in separate scheduling steps.
    // Drain queued prefills before decoding. Otherwise the first ready
    // sequence starts decoding while later sequences wait and a queued batch
    // runs with a tiny active batch.
    if (!waiting->empty()) {
        process_waiting_list(candidates, n_batch);
    }

    // Keep exactly one prefill candidate. FCFS keeps the first one in
    // candidates order (running -> swapped -> waiting); priority mode picks
    // the best-ranked prefill across all candidates.
    llama_sequence_group * best_prefill = nullptr;
    for (llama_sequence_group * group : candidates) {
        if (!group || group->n_decoded != 0) {
            continue;
        }
        if (!best_prefill || (priority_enabled_ && group_precedes(*group, *best_prefill))) {
            best_prefill = group;
        }
    }
    if (best_prefill) {
        candidates.clear();
        candidates.push_back(best_prefill);
    }

    // multimodal (9.4): an embd step (a media segment's rows fed as
    // batch.embd) runs exclusively - text and embd rows cannot share a batch
    auto media_it = std::find_if(candidates.begin(), candidates.end(), [](const llama_sequence_group * group) {
        if (!group) {
            return false;
        }
        const llama_seq_segment * seg = group->segment_at(group->n_past);
        return seg && seg->is_media;
    });
    if (media_it != candidates.end()) {
        llama_sequence_group * media = *media_it;
        candidates.clear();
        candidates.push_back(media);
    }

    const uint32_t n_running    = running.size();
    const uint32_t n_swapped    = swapped->size();
    const uint32_t n_waiting    = waiting->size();
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
    cache_manager->set_paged_batch_info(&curr_info);
    return llama_scheduler_status::OK;
}

bool llama_paged_scheduler_impl::queue_request(llama_sequence_group group) {
    if (group.logical_seq.empty()) {
        LLAMA_LOG_ERROR("%s: request %d has an empty prompt.\n", __func__, group.request_id);
        return false;
    }

    if (id_to_group.count(group.request_id)) {
        // Overwriting a live id would orphan the existing group: update()
        // resolves request ids through this map and would silently drop the
        // new request. A finished group keeps its entry until the next step
        // finalizes it, so ids must not be reused while still outstanding.
        LLAMA_LOG_ERROR("%s: request %d is already outstanding.\n", __func__, group.request_id);
        return false;
    }

    // Rejecting any requests that exceeds max context for a seq
    if (group.n_prompt >= n_seq_max_ctx) {
        LLAMA_LOG_ERROR("%s: request %d exceeds max context (%d > %d).\n", __func__, group.request_id, group.n_prompt,
                        n_seq_max_ctx);
        return false;
    }

    if (prefix_cache_enabled && !group.has_media) {
        // Queueing only records a candidate; block refs and the active linear
        // state block are reserved at admission time. Media rows carry no
        // token ids, so blocks containing them can neither be hashed nor
        // safely reused (9.4).
        group.prefix_candidate = cache_manager->match_prefix_candidate(group.logical_seq);
    }

    group.n_prompt_total = (uint32_t) group.logical_seq.size();

    // Bounded so the ordering keys stay well-defined even for bogus callers.
    if (group.priority > LLAMA_PAGED_PRIORITY_LIMIT || group.priority < -LLAMA_PAGED_PRIORITY_LIMIT) {
        LLAMA_LOG_WARN("%s: request %d priority %d is out of range, clamping to [-%d, %d].\n", __func__,
                       group.request_id, group.priority, LLAMA_PAGED_PRIORITY_LIMIT, LLAMA_PAGED_PRIORITY_LIMIT);
        group.priority = std::clamp(group.priority, -LLAMA_PAGED_PRIORITY_LIMIT, LLAMA_PAGED_PRIORITY_LIMIT);
    }
    if (!priority_enabled_ && group.priority != 0 && !warned_priority_ignored) {
        LLAMA_LOG_WARN("%s: request %d has priority %d but priority scheduling is disabled; "
                       "priority is recorded and ignored.\n",
                       __func__, group.request_id, group.priority);
        warned_priority_ignored = true;
    }
    // Assigned here (not in the C API wrapper) so tests and direct callers
    // also get a well-defined tie-break key.
    group.arrival_seq = next_arrival_seq++;

    auto group_ptr = std::make_shared<llama_sequence_group>(std::move(group));

    id_to_group[group_ptr->request_id] = group_ptr.get();

    set_waiting(std::move(group_ptr));
    return true;
}

bool llama_paged_scheduler_impl::add_request_segs(const llama_paged_segment * segs, int32_t n_segs,
                                                  int32_t request_id) {
    if (n_segs <= 0 || segs == nullptr) {
        LLAMA_LOG_ERROR("%s: request %d has no segments.\n", __func__, request_id);
        return false;
    }

    llama_sequence_group group;
    group.request_id     = request_id;
    group.t_arrival_time = ggml_time_us();

    bool has_media = false;
    for (int32_t i = 0; i < n_segs; ++i) {
        const llama_paged_segment & s = segs[i];
        if (s.n_tokens <= 0) {
            LLAMA_LOG_ERROR("%s: request %d has an empty segment (index %d).\n", __func__, request_id, i);
            return false;
        }

        llama_seq_segment seg;
        seg.start_row = (uint32_t) group.logical_seq.size();
        seg.n_tokens  = (uint32_t) s.n_tokens;
        seg.is_media  = (s.tokens == nullptr);
        if (seg.is_media) {
            has_media = true;
            if (s.embd != nullptr && s.embd_stride <= 0) {
                LLAMA_LOG_ERROR("%s: request %d media segment %d has no embd stride.\n", __func__, request_id, i);
                return false;
            }
            seg.embd        = s.embd;
            seg.embd_stride = s.embd_stride;
            if (s.pos) {
                seg.pos.assign(s.pos, s.pos + (size_t) s.n_tokens * n_pos_per_embd);
            }
            group.logical_seq.insert(group.logical_seq.end(), s.n_tokens, LLAMA_TOKEN_NULL);
        } else {
            group.logical_seq.insert(group.logical_seq.end(), s.tokens, s.tokens + s.n_tokens);
        }
        group.segments.push_back(std::move(seg));
    }
    group.has_media = has_media;

    return queue_request(std::move(group));
}

bool llama_paged_scheduler_impl::append_segment(int32_t request_id, const llama_paged_segment & seg) {
    auto it = id_to_group.find(request_id);
    if (it == id_to_group.end()) {
        LLAMA_LOG_ERROR("%s: request %d not found.\n", __func__, request_id);
        return false;
    }
    llama_sequence_group * group = it->second;

    for (auto & s : group->segments) {
        if (s.is_media && s.embd == nullptr && s.start_row == group->n_past) {
            if (seg.embd == nullptr || seg.embd_stride <= 0) {
                LLAMA_LOG_ERROR("%s: request %d appended segment has no embeddings.\n", __func__, request_id);
                return false;
            }
            s.embd        = seg.embd;
            s.embd_stride = seg.embd_stride;
            if (seg.pos && seg.n_tokens) {
                s.pos.assign(seg.pos, seg.pos + (size_t) seg.n_tokens * n_pos_per_embd);
            }
            group->paused_external = false;
            if (group->status == llama_sequence_group_status::PAUSED_EXTERNAL) {
                group->status = llama_sequence_group_status::RUNNING;
            }
            LLAMA_LOG_DEBUG("%s: request %d media segment appended at row %d (%d rows)\n", __func__, request_id,
                            (int) s.start_row, s.n_tokens);
            return true;
        }
    }
    LLAMA_LOG_ERROR("%s: request %d has no pending media segment at row %d.\n", __func__, request_id, group->n_past);
    return false;
}

void llama_paged_scheduler_impl::insert_sorted_by_arrival_time(llama_sequence_group_ptr    new_group_ptr,
                                                               llama_sequence_group_list & list) {
    GGML_ASSERT(new_group_ptr && "New group cannot be sorted because it's nullptr.");
    auto it = std::lower_bound(
        list.begin(), list.end(), *new_group_ptr,
        [](const llama_sequence_group_ptr & group, const llama_sequence_group & key) {
            GGML_ASSERT(group && "group cannot be checked for arrival time because it's nullptr.");
            return arrival_precedes(*group, key);
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
    swapped->push(std::move(group_ptr));
}

void llama_paged_scheduler_impl::set_waiting(llama_sequence_group_ptr group_ptr) {
    GGML_ASSERT(group_ptr && group_ptr->status != llama_sequence_group_status::WAITING &&
                "Request is already waiting.");
    group_ptr->status = llama_sequence_group_status::WAITING;
    // FCFS inserts back at its keyed position; PRIORITY re-inserts into the
    // heap. Recomputed requests keep their (priority, t, seq) keys, so a plain
    // push restores their original rank -- no push_front is needed.
    waiting->push(std::move(group_ptr));
}

void llama_paged_scheduler_impl::finish(llama_sequence_group & group) {
    GGML_ASSERT(cache_manager && "cache_manager is nullptr.");
    GGML_ASSERT(group.status == llama_sequence_group_status::FINISHED && "Request was not marked as finished.");
    // Release the scheduler-visible id before running the callback: the
    // callback may submit the next turn of the same session with the same
    // request_id, and that new group must not be erased afterwards.
    id_to_group.erase(group.request_id);
    group.status = llama_sequence_group_status::FINISHED;

    // We prioritize user CB, otherwise we log by default
    if (on_finish_cb) {
        // TODO perhaps just have the callback take sequence_group and user_data
        on_finish_cb(group.request_id, group.logical_seq.data(), (int32_t) group.logical_seq.size(),
                     on_finish_user_data);
    } else {
        LLAMA_LOG_DEBUG("%s: Request: %d generated %d tokens.\n", __func__, group.request_id, group.n_decoded);
    }
    // Registered prefix blocks stay cache-resident (ref stays 1) until
    // evicted under allocation pressure; the linear active state is released.
    cache_manager->release_request(group);
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
    GGML_ASSERT(cache_manager && "cache_manager is nullptr");

    const int32_t rid = group_ptr->request_id;

    const bool swap_ok = cache_manager->swap_out(*group_ptr);
    if (swap_ok) {
        LLAMA_LOG_DEBUG("%s: (swapped_out) request_id=%d was swapped out to make room.\n", __func__, rid);
        set_swapped(std::move(group_ptr));
        return;
    }

    // There was not enough CPU memory to swap the request (recomputation)
    // media rows cannot be recomputed: their KV comes from the vision
    // encoder, re-feeding the logical sequence would feed placeholders
    if (group_ptr->has_media) {
        LLAMA_LOG_ERROR("%s: request %d holds media rows and cannot be recomputed; failing it\n", __func__, rid);
        cache_manager->release_request(*group_ptr);
        finish(*group_ptr);
        return;
    }

    cache_manager->release_request(*group_ptr);
    group_ptr->n_past    = 0;
    group_ptr->n_decoded = 0;
    group_ptr->logical_seq.resize(group_ptr->n_prompt_total);
    group_ptr->n_prompt  = group_ptr->n_prompt_total;
    group_ptr->prefix_candidate = cache_manager->match_prefix_candidate(group_ptr->logical_seq);

    LLAMA_LOG_DEBUG("%s: (recomputation) request_id=%d was sent for recomputation.\n", __func__, rid);
    set_waiting(std::move(group_ptr));
}

llama_sequence_group_ptr llama_paged_scheduler_impl::evict() {
    GGML_ASSERT(cache_manager && "cache_manager is nullptr.");
    LLAMA_LOG_DEBUG("%s: Eviction requested...\n", __func__);

    // Consider the whole running list. A victim that was already added to this
    // step's `candidates` is fine as long as the caller removes it from that
    // vector before swap-out/recompute; the batch is built afterwards, so a CPU
    // block id can never reach populate_batch_from(). Restricting the scan to
    // [it, end) used to protect candidates implicitly, but it also shielded
    // low-priority groups that happened to be scheduled first, causing a
    // priority inversion within the step.
    auto victim = running.end();
    for (auto it = running.begin(); it != running.end(); ++it) {
        const llama_sequence_group & g = **it;
        if (g.status == llama_sequence_group_status::FINISHED) {
            continue;
        }
        if (victim == running.end() || eviction_precedes(g, **victim, priority_enabled_)) {
            victim = it;
        }
    }
    if (victim == running.end()) {
        return nullptr;
    }

    llama_sequence_group_ptr evicted = std::move(*victim);
    running.erase(victim);
    return evicted;
}

void llama_paged_scheduler_impl::process_running_list(llama_sequence_group_raw_list & candidates,
                                                      int32_t &                       row_budget) {
    GGML_ASSERT(cache_manager && "cache_manager is nullptr.");

    llama_sequence_group_list::iterator it = running.begin();
    while (it != running.end()) {
        llama_sequence_group * group = it->get();
        GGML_ASSERT(group && "group is nullptr.");

        if (group->status == llama_sequence_group_status::FINISHED) {
            finish(*group);
            it = running.erase(it);
            continue;
        }

        // multimodal (9.4): a group waiting for the server to append a media
        // segment's encoder output pauses here; excluded from candidates and
        // from the capacity accounting until append_segment resumes it
        if (group->paused_external || group->pending_media_at(group->n_past)) {
            group->paused_external  = true;
            group->status           = llama_sequence_group_status::PAUSED_EXTERNAL;
            LLAMA_LOG_DEBUG("%s: (paused_external) request_id=%d waits for a media segment\n", __func__,
                            group->request_id);
            ++it;
            continue;
        }

        // rows this request contributes to the next step (verify steps carry
        // one row per draft token; media steps carry their segment's rows)
        uint32_t rows = group->pending_draft.empty() ? 1u : 1u + (uint32_t) group->pending_draft.size();
        const llama_seq_segment * seg = group->segment_at(group->n_past);
        if (seg && seg->is_media) {
            rows = seg->start_row + seg->n_tokens - group->n_past;
        }
        if (rows > (uint32_t) row_budget) {
            // token budget exhausted; this request decodes on a later step
            break;
        }

        // Dynamically allocate more blocks to decode the request
        uint32_t current_capacity  = group->block_table.size() * block_size;
        uint32_t required_capacity = group->n_past + rows;
        LLAMA_LOG_DEBUG(
            "%s: (running) request_id=%d: current_capacity (tokens)=%d toks, required capacity (tokens) = %d toks\n",
            __func__, group->request_id, current_capacity, required_capacity);
        if (required_capacity >= current_capacity) {
            LLAMA_LOG_DEBUG("%s: (running_pending) request_id=%d: requires a new block to decode.\n", __func__,
                            group->request_id);
            bool success = cache_manager->allocate_blocks(*group, rows);  // decode phase
            while (!success && cache_manager->evict_cached_blocks()) {
                success = cache_manager->allocate_blocks(*group, rows);
            }
            if (!success) {
                if (running.size() > 1) {
                    llama_sequence_group_ptr victim = evict();
                    if (victim) {
                        // Compare object identity: request ids are not enforced
                        // to be unique by queue_request().
                        const bool victim_is_curr = (victim.get() == group);
                        if (!victim_is_curr) {
                            // The victim may already be part of this step's
                            // candidates. Drop it before swap-out/recompute so
                            // populate_batch_from() never sees CPU block ids.
                            auto cit = std::find(candidates.begin(), candidates.end(), victim.get());
                            if (cit != candidates.end()) {
                                candidates.erase(cit);
                            }
                        }
                        swap_out_or_recompute(std::move(victim));

                        // Evict might have removed the current group from running.
                        if (victim_is_curr) {
                            it = running.end();
                            continue;
                        }

                        // Try allocating again after eviction. Reserve the
                        // full verify-row count, not just one slot
                        success = cache_manager->allocate_blocks(*group, rows);
                    }
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
        row_budget -= (int32_t) rows;
        ++it;
    }
}

void llama_paged_scheduler_impl::process_swapped_list(llama_sequence_group_raw_list & candidates,
                                                      int32_t &                       row_budget) {
    GGML_ASSERT(cache_manager && "cache_manager is nullptr.");
    while (!swapped->empty()) {
        llama_sequence_group * group = swapped->peek();
        GGML_ASSERT(group && "the group to swap is nullptr.");
        if (row_budget <= 0) {
            break;
        }
        // A swapped decode group rejoins with its draft rows, not just one.
        const uint32_t rows = 1u + (uint32_t) group->pending_draft.size();
        if (rows > (uint32_t) row_budget) {
            break;
        }
        if (!cache_manager->swap_in(*group)) {
            // Swap in queue order (FCFS = earliest arrival, PRIORITY = highest
            // priority); if the head cannot be swapped in we stop, so a younger
            // request cannot jump ahead.
            LLAMA_LOG_DEBUG("%s: head request_id=%d (prio=%d) cannot be swapped in, stop swapping this step\n",
                            __func__, group->request_id, group->priority);
            break;
        }
        candidates.push_back(group);
        row_budget -= (int32_t) rows;
        llama_sequence_group_ptr group_ptr = swapped->pop();
        LLAMA_LOG_DEBUG("%s: (swapped_in) request_id=%d back in for processing.\n", __func__,
                        group_ptr->request_id);
        set_running(std::move(group_ptr));
    }
}

void llama_paged_scheduler_impl::process_waiting_list(llama_sequence_group_raw_list & candidates,
                                                      int32_t                         row_budget) {
    GGML_ASSERT(cache_manager && "cache_manager is nullptr.");
    if (waiting->empty()) {
        return;
    }

    // Head of the queue: FCFS = earliest arrival, PRIORITY = highest priority.
    llama_sequence_group * group = waiting->peek();
    GGML_ASSERT(group && "the waiting group is nullptr.");

    // The cache may have grown since this request was queued.
    if (!cache_manager->reserve_prefix_candidate(*group)) {
        return;
    }

    const int32_t chunk_tokens = std::min<int32_t>((int32_t) group->n_prompt, row_budget);
    if (chunk_tokens <= 0) {
        // Do not leave a WAITING request holding refs / an active state block.
        cache_manager->release_request(*group);
        group->n_past   = 0;
        group->n_prompt = group->n_prompt_total;
        return;  // token budget exhausted for this step
    }

    // Reserve the full prompt plus one decode slot; only chunk_tokens run now.
    bool success = cache_manager->allocate_blocks(*group, 1);
    while (!success && cache_manager->evict_cached_blocks()) {
        success = cache_manager->allocate_blocks(*group, 1);
    }
    if (!success) {
        // Strict head-of-queue semantics (matches vLLM): do not skip an
        // inadmissible head to serve later requests. Drop the reservation so
        // a waiting request never holds block refs or an active state block.
        cache_manager->release_request(*group);
        group->n_past   = 0;
        group->n_prompt = group->n_prompt_total;
        LLAMA_LOG_DEBUG("%s: head request_id=%d (prio=%d) is inadmissible, stop scheduling waiting this step\n",
                        __func__, group->request_id, group->priority);
        return;
    }

    candidates.push_back(group);
    llama_sequence_group_ptr group_ptr = waiting->pop();
    LLAMA_LOG_DEBUG("%s: (start) request_id=%d sent for processing.\n", __func__, group_ptr->request_id);
    set_running(std::move(group_ptr));
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
    cache_manager->set_paged_batch_info(nullptr);

    delete[] curr_info.write_slots;
    delete[] curr_info.block_table;
    delete[] curr_info.context_lens;
    delete[] curr_info.batch_offsets;
    delete[] curr_info.batch_lens;
    curr_info = {};  // reset to defaults

    // detach pointers llama_batch_free must not free: media batches point
    // embd into the server-owned encoder output, and mrope batches point pos
    // into populate_pos; the init-allocated token buffer of an embd batch is
    // stashed in populate_token_buf and freed here manually
    if (populate_token_buf) {
        free(populate_token_buf);
        populate_token_buf = nullptr;
    }
    batch.embd = nullptr;
    if (n_pos_per_embd > 1) {
        batch.pos = nullptr;
    }

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

    // Calculating required sizes. a group whose head segment is a media
    // segment contributes its remaining media rows as a pure-embd slice
    // (9.4: text and embd rows cannot share a batch)
    for (const auto & group : candidates) {
        GGML_ASSERT(group && "candidate request is nullptr.");
        const bool     is_prefill   = group->n_decoded == 0 && group->n_prompt > 0;
        const uint32_t n_draft_rows = group->pending_draft.size();
        const llama_seq_segment * seg = group->segment_at(group->n_past);
        int32_t rows;
        if (seg && seg->is_media) {
            // pure-embd slice: capped by the token budget
            rows = std::min<int32_t>((int32_t)(seg->start_row + seg->n_tokens - group->n_past), n_batch);
        } else if (is_prefill) {
            // hybrid models stop at a checkpoint boundary (see prefill_chunk_tokens)
            rows = prefill_chunk_tokens(*group);
            // text rows stop at the next media segment boundary (9.4)
            if (seg && !seg->is_media) {
                rows = std::min<int32_t>(rows, (int32_t)(seg->start_row + seg->n_tokens - group->n_past));
            }
        } else {
            rows = 1 + (int32_t) n_draft_rows;
        }
        total_tokens += rows;
        max_blocks = std::max(max_blocks, (int32_t) group->block_table.size());
    }

    GGML_ASSERT(total_tokens <= (int32_t) n_batch && "total_tokens exceeds n_batch — token budget logic is broken");

    // Initialize the batch (assumed it was cleared before)
    batch = llama_batch_init(total_tokens, 0, 1);
    GGML_ASSERT(batch.token != nullptr && "llama_batch_init failed to allocate tokens.");

    // mrope models: text rows broadcast one position per token, media (embd)
    // rows use a sectioned layout [section][total_tokens]; the latter needs
    // more room than the init-allocated buffer, so replace it for the
    // lifetime of this batch
    if (n_pos_per_embd > 1) {
        populate_pos.assign((size_t) total_tokens * n_pos_per_embd, 0);
        batch.pos = populate_pos.data();
    }

    batch.n_tokens = total_tokens;

    curr_info.n_seq            = batch_size;
    curr_info.n_tokens         = total_tokens;
    // Keep the block table shape fixed so ACL graph replay matches across
    // decode steps. Invalid padding is ignored because context_lens bounds
    // each sequence.
    const int32_t block_table_capacity =
            (int32_t) ((n_seq_max_ctx + block_size - 1) / block_size);
    max_blocks                 = std::max(max_blocks, block_table_capacity);
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

        const bool     is_prefill   = group->n_decoded == 0 && group->n_prompt > 0;
        const uint32_t n_draft_rows = group->pending_draft.size();
        const llama_seq_segment * seg = group->segment_at(group->n_past);
        const bool     media_step    = seg && seg->is_media;
        int32_t  new_tokens;
        if (media_step) {
            new_tokens = std::min<int32_t>((int32_t)(seg->start_row + seg->n_tokens - group->n_past), n_batch);
        } else if (is_prefill) {
            // hybrid models stop at a checkpoint boundary (see prefill_chunk_tokens)
            new_tokens = prefill_chunk_tokens(*group);
            // text rows stop at the next media segment boundary (9.4)
            if (seg && !seg->is_media) {
                new_tokens = std::min<int32_t>(new_tokens,
                        (int32_t)(seg->start_row + seg->n_tokens - group->n_past));
            }
        } else {
            new_tokens = 1 + (int32_t) n_draft_rows;
        }

        if (media_step) {
            // pure-embd slice: rows come from the server-owned encoder
            // output; no token ids (9.4). null out the token array so the
            // batch allocr validation skips the placeholder ids (-1); the
            // init-allocated buffer is freed in clear_batch
            GGML_ASSERT(seg->embd != nullptr && "media step without encoder output");
            batch.embd           = const_cast<float *>(seg->embd) +
                        (size_t)(group->n_past - seg->start_row) * (size_t) seg->embd_stride;
            populate_token_buf   = batch.token;
            batch.token          = nullptr;
        } else if (is_prefill) {
            GGML_ASSERT(group->logical_seq.size() >= (size_t)(group->n_past + new_tokens) &&
                        "logical_seq too small for prefill");
        } else {
            GGML_ASSERT(!group->logical_seq.empty() && "logical_seq empty during decode");
        }

        for (int token_idx = 0; token_idx < new_tokens; ++token_idx) {
            int32_t batch_start_id = token_offset + token_idx;

            if (!media_step) {
                if (is_prefill) {
                    batch.token[batch_start_id] = group->logical_seq[group->n_past + token_idx];
                } else if (token_idx == 0) {
                    batch.token[batch_start_id] = group->logical_seq.back();
                } else {
                    batch.token[batch_start_id] = group->pending_draft[token_idx - 1];
                }
            }

            batch.pos[batch_start_id]   = group->n_past + token_idx;  // n_past starts at 0

            batch.n_seq_id[batch_start_id]  = 1;
            batch.seq_id[batch_start_id][0] = group->request_id;

            // speculative verify rows need logits at every position so the
            // caller can verify each draft token
            batch.logits[batch_start_id] = is_prefill ? (token_idx == (new_tokens - 1)) : 1;

            int32_t token_pos                     = group->n_past + token_idx;
            curr_info.write_slots[batch_start_id] = calculate_global_slot_index(token_pos, group->block_table);
            LLAMA_LOG_DEBUG("%s: llama_batch seq_id: %d (req_id %d) token %d: pos: %d, global_slot_idx=%d\n", __func__,
                            seq_id, group->request_id, token_idx, token_pos, curr_info.write_slots[batch_start_id]);
        }

        // mrope models take the sectioned position array; media rows carry
        // their per-row positions from the segment
        // mrope models: text rows broadcast one position per token (written
        // in the loop above, the batch layer replicates it to every section);
        // only media (embd) rows use the sectioned layout [section][token]
        // over the whole batch, carrying the per-row positions of the segment
        if (n_pos_per_embd > 1 && media_step) {
            for (int s = 0; s < n_pos_per_embd; ++s) {
                for (int token_idx = 0; token_idx < new_tokens; ++token_idx) {
                    const int32_t rel = group->n_past - (int32_t) seg->start_row + token_idx;
                    populate_pos[(size_t) s * total_tokens + token_offset + token_idx] =
                        seg->pos[(size_t) rel * n_pos_per_embd + s];
                }
            }
        }

        group->draft_rows_in_batch = (uint32_t) group->pending_draft.size();
        group->pending_draft.clear();  // consumed by this step

        // Populate block table (1D): [batch_size * max_blocks]
        const int32_t curr_block_table_size = group->block_table.size();
        for (int block = 0; block < max_blocks; ++block) {
            int  flattened_id                   = (seq_id * max_blocks) + block;  // row-major
            bool need_padding                   = block >= curr_block_table_size;
            curr_info.block_table[flattened_id] = need_padding ? -1 : group->block_table[block];
        }

        curr_info.context_lens[seq_id]  = group->n_past + new_tokens;
        curr_info.batch_offsets[seq_id] = token_offset;
        curr_info.batch_lens[seq_id]    = new_tokens;
        token_offset += new_tokens;
    }
}

// feedback-driven post-decode bookkeeping. in-batch requests advance their KV
// positions by the batch rows, then append their feedback tokens; requests not
// present in the batch (forked children sampled out-of-batch) get token
// bookkeeping only.
void llama_paged_scheduler_impl::update(const llama_batch *                 batch,
                                        const llama_paged_feedback *        feedback,
                                        int32_t                             n_feedback) {
    GGML_ASSERT(n_feedback == 0 || feedback != nullptr && "feedback can't be null");

    auto apply_tokens = [&](llama_sequence_group * group, const llama_paged_feedback & fb) {
        for (int32_t t = 0; t < fb.n_tokens; ++t) {
            group->logical_seq.push_back(fb.tokens[t]);
            group->n_decoded++;
        }
        if (fb.stop || group->n_past >= n_seq_max_ctx) {
            group->status = llama_sequence_group_status::FINISHED;
        }
    };

    std::vector<bool> matched(n_feedback, false);

    if (batch && batch->n_tokens > 0 && curr_info.n_seq > 0) {
        for (int i = 0; i < curr_info.n_seq; ++i) {
            const int32_t token_offset = curr_info.batch_offsets[i];
            const int32_t request_id   = batch->seq_id[token_offset][0];

            auto it = id_to_group.find(request_id);
            if (it == id_to_group.end()) {
                LLAMA_LOG_WARN("%s: request_id %d not found in scheduler, skipping\n", __func__, request_id);
                continue;
            }

            llama_sequence_group * group = it->second;
            GGML_ASSERT(group && "group is nullptr.");

            const int32_t batch_len     = curr_info.batch_lens[i];
            const bool    was_prefill   = group->n_decoded == 0;
            const bool    reaches_end   = (int32_t) group->n_past + batch_len ==
                                          (int32_t) group->logical_seq.size();
            const bool    final_prefill = was_prefill && reaches_end;

            // TTFT: the first token can be sampled once the prefill is complete.
            if (final_prefill) {
                group->t_first_token_us = ggml_time_us();
            }

            // Setting token ranges
            llama_pos range_min = cache_manager->seq_pos_min(group->request_id);
            if (range_min == -1) {
                cache_manager->set_seq_min_pos(group->request_id, batch->pos[token_offset]);
            }
            int32_t last_token_in_batch_idx = token_offset + batch_len - 1;
            cache_manager->set_seq_max_pos(group->request_id, batch->pos[last_token_in_batch_idx]);

            group->n_past += batch_len;

            const llama_paged_feedback * fb = nullptr;
            for (int32_t f = 0; f < n_feedback; ++f) {
                if (!matched[f] && feedback[f].request_id == request_id) {
                    matched[f] = true;
                    fb         = &feedback[f];
                    break;
                }
            }

            // draft rows are tentatively decoded; the caller accepts a prefix
            // via the feedback and rolls the rest back via truncate, so blocks
            // from this step are not registered in the prefix cache
            const bool had_draft = group->draft_rows_in_batch > 0;
            group->draft_rows_in_batch = 0;

            if (fb) {
                // appends the feedback tokens and increments n_decoded; an
                // intermediate prefill chunk legitimately has no feedback
                apply_tokens(group, *fb);
            } else if (!had_draft && (batch_len == 1 || final_prefill)) {
                LLAMA_LOG_WARN("%s: no feedback for in-batch request %d\n", __func__, request_id);
            }

            if (!had_draft) {
                // Always track positions (rollback); prefix registration and
                // checkpoints are gated inside the cache manager. Media rows
                // carry no token ids and must never be hashed.
                cache_manager->register_tokens(*group);
                if (!group->has_media) {
                    cache_manager->maybe_create_checkpoint(*group);
                }
            } else if (auto * linear = cache_manager->linear()) {
                // Verify rows were executed even though they are not registered
                // in the prefix cache. Keep the rollback position in sync so a
                // later truncate() can select the right snapshot plane.
                if (group->n_past > 0) {
                    linear->set_seq_pos(group->request_id, (llama_pos) group->n_past - 1);
                }
            }
        }
    }

    // out-of-batch feedback (forked children)
    for (int32_t f = 0; f < n_feedback; ++f) {
        if (matched[f]) {
            continue;
        }
        auto it = id_to_group.find(feedback[f].request_id);
        if (it == id_to_group.end()) {
            LLAMA_LOG_WARN("%s: feedback request_id %d not found in scheduler, skipping\n", __func__,
                           feedback[f].request_id);
            continue;
        }
        apply_tokens(it->second, feedback[f]);
    }
}

void llama_paged_scheduler_impl::set_on_finish(llama_paged_on_finish_cb cb, void * user_data) {
    on_finish_cb        = cb;
    on_finish_user_data = user_data;
}

llama_sequence_group_ptr llama_paged_scheduler_impl::take_from_list(llama_sequence_group_list & list,
                                                                    int32_t                     request_id) {
    for (auto it = list.begin(); it != list.end(); ++it) {
        if (*it && (*it)->request_id == request_id) {
            llama_sequence_group_ptr group_ptr = std::move(*it);
            list.erase(it);
            return group_ptr;
        }
    }
    return nullptr;
}

bool llama_paged_scheduler_impl::cancel(int32_t request_id) {
    auto it = id_to_group.find(request_id);
    if (it == id_to_group.end()) {
        return false;
    }

    llama_sequence_group_ptr group_ptr = take_from_list(running, request_id);
    if (!group_ptr) {
        group_ptr = waiting->take(request_id);
    }
    if (!group_ptr) {
        group_ptr = swapped->take(request_id);
    }
    if (!group_ptr) {
        return false;
    }

    LLAMA_LOG("%s: request %d cancelled (was waiting=%d swapped=%d running=%d)\n", __func__, request_id,
              group_ptr->status == llama_sequence_group_status::WAITING,
              group_ptr->status == llama_sequence_group_status::SWAPPED,
              group_ptr->status == llama_sequence_group_status::RUNNING);

    // Reap synchronously: free_blocks releases GPU and CPU-pool blocks alike,
    // so a swapped-out request never needs a swap-in to be reaped.
    group_ptr->status = llama_sequence_group_status::FINISHED;
    finish(*group_ptr);
    return true;
}

int32_t llama_paged_scheduler_impl::max_request_id() const {
    // prefix-cache checkpoints no longer occupy request ids (linear state
    // blocks are indexed independently), so the whole seq space is usable
    return (int32_t) cache_manager->full()->get_n_seq_max();
}

bool llama_paged_scheduler_impl::fork_request(int32_t src_id, int32_t dst_id) {
    if (src_id == dst_id) {
        return false;
    }

    auto it = id_to_group.find(src_id);
    if (it == id_to_group.end()) {
        LLAMA_LOG_ERROR("%s: src request %d not found.\n", __func__, src_id);
        return false;
    }
    llama_sequence_group * src = it->second;

    // the fork point is a fully-written prompt: logical_seq must be exactly
    // the prompt (the caller forks before applying the parent's feedback)
    if (src->logical_seq.size() != src->n_prompt_total || src->block_table.empty()) {
        LLAMA_LOG_ERROR("%s: src request %d has not completed its prompt.\n", __func__, src_id);
        return false;
    }
    if (id_to_group.count(dst_id)) {
        LLAMA_LOG_ERROR("%s: dst request %d is still queued or running.\n", __func__, dst_id);
        return false;
    }
    const uint32_t n_past_new  = src->n_prompt_total;
    const uint32_t full_blocks = n_past_new / block_size;
    const uint32_t tail        = n_past_new % block_size;

    llama_sequence_group child = *src;
    child.request_id       = dst_id;
    child.n_past           = n_past_new;   // the decode already wrote the full prompt
    child.n_prompt         = 0;
    child.n_decoded        = 0;
    child.t_arrival_time   = ggml_time_us();
    child.t_first_token_us = 0;
    child.pending_draft.clear();
    child.draft_rows_in_batch = 0;
    child.block_table.assign(src->block_table.begin(), src->block_table.begin() + full_blocks);
    child.block_hashes.assign(src->block_hashes.begin(),
                              src->block_hashes.begin() + std::min<size_t>(src->block_hashes.size(), full_blocks));
    child.prefix_candidate = {};
    child.cache_reserved   = true;
    child.status = llama_sequence_group_status::PENDING;

    if (!cache_manager->full()->add_ref_blocks(child.block_table)) {
        cache_manager->full()->free_blocks(child);
        LLAMA_LOG_ERROR("%s: failed to share blocks for dst request %d.\n", __func__, dst_id);
        return false;
    }

    // eagerly copy the partially-filled tail block: parent and child must not
    // write the same physical block. a tail-aligned prompt shares everything.
    if (tail != 0) {
        // The child already holds `full_blocks` shared full blocks, so exactly
        // one new block is needed: index `full_blocks`, which holds the
        // partial tail and the next decode position. Reserving a full
        // block_size worth of tokens would ask for a second block and make
        // block_table.back() point past the tail block, so the copy below is
        // addressed by index instead.
        if (!cache_manager->allocate_blocks(child, 1)) {
            cache_manager->full()->free_blocks(child);
            LLAMA_LOG_ERROR("%s: failed to allocate the tail block for dst request %d.\n", __func__, dst_id);
            return false;
        }
        GGML_ASSERT(full_blocks < src->block_table.size() && full_blocks < child.block_table.size());
        cache_manager->full()->copy_block_content(src->block_table[full_blocks], child.block_table[full_blocks]);
    }

    // the child needs its own mutable linear state; copy-on-fork from the parent
    if (cache_manager->linear() && !cache_manager->linear()->seq_cp(src->request_id, dst_id)) {
        cache_manager->full()->free_blocks(child);
        LLAMA_LOG_ERROR("%s: failed to fork linear state for dst request %d.\n", __func__, dst_id);
        return false;
    }

    child.status = llama_sequence_group_status::RUNNING;
    auto child_ptr = std::make_unique<llama_sequence_group>(std::move(child));
    id_to_group[dst_id] = child_ptr.get();
    insert_sorted_by_arrival_time(std::move(child_ptr), running);
    return true;
}

bool llama_paged_scheduler_impl::attach_draft(int32_t request_id, const llama_token * draft, int32_t n_draft) {
    if (!draft || n_draft <= 0) {
        return false;
    }

    auto it = id_to_group.find(request_id);
    if (it == id_to_group.end()) {
        return false;
    }
    llama_sequence_group * group = it->second;
    if (!group->pending_draft.empty()) {
        return false;
    }
    // the verify step must fit the token budget; when it does not the caller
    // simply decodes without a draft this round
    if ((int32_t) group->n_past + 1 + n_draft > n_batch) {
        LLAMA_LOG_WARN("%s: request %d draft of %d tokens exceeds n_batch, skipping\n", __func__, request_id, n_draft);
        return false;
    }

    group->pending_draft.assign(draft, draft + n_draft);
    return true;
}

bool llama_paged_scheduler_impl::truncate(int32_t request_id, int32_t n_past_new) {
    auto it = id_to_group.find(request_id);
    if (it == id_to_group.end()) {
        return false;
    }
    llama_sequence_group * group = it->second;
    if (n_past_new < 0 || (int32_t) group->n_past < n_past_new) {
        LLAMA_LOG_ERROR("%s: request %d cannot truncate from %d to %d.\n", __func__, request_id, group->n_past,
                        n_past_new);
        return false;
    }
    if ((int32_t) group->n_past == n_past_new) {
        return true;
    }

    // keep the partial block holding n_past_new as well: releasing it would
    // drop the KV of all positions below n_past_new on the next block-table
    // rotation (the block cache block is not zeroed, but the table may be
    // repopulated with different block ids). The linear state rolls back to
    // the same boundary through its saved snapshot plane.
    cache_manager->truncate_sequence(*group, (llama_pos) n_past_new);

    // keep the pending token (last element) consistent with the new boundary
    group->logical_seq.resize((size_t) n_past_new + 1);
    cache_manager->set_seq_max_pos(request_id, (llama_pos)(n_past_new - 1));
    return true;
}

bool llama_paged_scheduler_impl::set_prefix_cache_enabled(bool enabled) {
    const bool old = prefix_cache_enabled;
    prefix_cache_enabled = enabled;
    cache_manager->set_prefix_enabled(enabled);
    return old;
}

llama_sequence_group * llama_paged_scheduler_impl::get_group_from_id(int32_t request_id) const {
    return id_to_group.count(request_id) ? id_to_group.at(request_id) : nullptr;
}

const llama_paged_batch_info * llama_paged_scheduler_impl::get_curr_batch_info() const {
    return &curr_info;
}
