#pragma once

// Paged continuous-batching scheduler for llama-server, driving the
// llama_paged_scheduler_* C API (see examples/paged/paged.cpp for the raw
// loop this mirrors). Enabled with --kv-paged.

#include "server-scheduler.h"

#include <mutex>

struct llama_paged_scheduler;
struct mtmd_input_chunk;

class server_scheduler_paged final : public server_scheduler_interface {
  public:
    explicit server_scheduler_paged(server_context_port & port) : port(port) {}
    ~server_scheduler_paged() override;

    bool init() override;
    bool launch(server_task && task) override;
    void cancel(int64_t id_task) override;
    void update() override;
    bool get_kv_stats(llama_paged_kv_cache_stats & out) override;
    bool get_tag_usage(const std::string & tag, bool include_blocks, json & out) override;
    bool set_evict_decision(llama_paged_evict_pick_cb cb, void * user_data) override;

  private:
    server_slot * pick_idle_slot();
    bool launch_multimodal(server_task && task);

    // multimodal (9.4): the pending media chunk of a slot, in order; returns
    // nullptr when all chunks are already appended
    const mtmd_input_chunk * find_media_chunk(server_slot & slot, size_t * idx);

    // multimodal (9.4): encode the next pending media chunk on a worker
    // thread so the clip run overlaps the scheduler steps
    void start_mm_prefetch(server_slot & slot);

    // multimodal (9.4): warm the clip encoder with one synthetic image so
    // the first real image request does not pay the clip first-compile cost
    void mm_warmup();

    // serializes clip encodes: the mmproj context is shared by all slots and
    // the prefetch workers
    std::mutex mm_encode_mutex;

    void fail_all(const std::string & why);
    void on_finish(int32_t request_id);
    static void on_finish_thunk(int32_t request_id, const llama_token * tokens, int32_t n_tokens, void * user_data);

    // built-in env-gated policy (LLAMA_SERVER_PAGED_EVICT_PROTECT_TAGS):
    // skip cache-resident nodes carrying any of the protected tag ids
    static uint32_t evict_pick_protect_tags(void * ud, llama_paged_pool_kind pool,
                                            uint32_t n_needed, const llama_paged_free_block * head,
                                            uint32_t * out_ids, uint32_t out_capacity);

    // n_cmpl > 1: fork the prefilled parent into the waiting child slots and
    // sample each child's first token out-of-batch from the parent's logits
    void fork_children(server_slot & parent, int32_t tok_idx,
                       std::vector<llama_token> & fb_tok, std::vector<llama_paged_feedback> & fb);

    // speculative decoding: generate drafts for the generating slots, attach
    // them to the scheduler and extend the prompt state
    void prepare_drafts();

    // speculative decoding: verify the attached drafts for one slot, roll back
    // on partial acceptance and emit the accepted tokens. returns false when
    // a stop condition fired during emission
    bool verify_slot(server_slot & slot, int32_t off, int32_t len,
                     std::vector<llama_token> & fb_tok, std::vector<llama_paged_feedback> & fb);

    server_context_port & port;

    llama_paged_scheduler * sched = nullptr;

    // interned ids of the built-in eviction policy's protected tags. Owned
    // here (not by the scheduler) so they outlive it; the callback user data
    // is `this`.
    std::vector<uint32_t> evict_protect_tags;

    // owned and recycled by llama_paged_scheduler_prepare_batch
    llama_batch paged_batch = {};

    // per-slot flag: the request was cancelled client-side, do not send a
    // final response when the scheduler reaps it
    std::vector<int8_t> cancelled;

    // busy-poll safeguard: in-flight requests but the scheduler keeps
    // producing empty batches
    int n_empty_steps = 0;

    // draft rollbacks deferred to after llama_paged_scheduler_update, which
    // is what advances the scheduler n_past over the verify batch; truncating
    // earlier would be rejected as n_past_new > n_past
    std::vector<std::pair<int32_t, int32_t>> spec_pending_truncates;
};
