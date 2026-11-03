#pragma once

// Pluggable scheduling strategy for the server main loop.
//
// The default slot scheduling stays inline in server-context.cpp. When
// --kv-paged is set, server_scheduler_create() returns a paged continuous
// batching scheduler that owns the decode loop instead (see
// server-scheduler-paged.h).

#include "server-slot.h"

#include "llama.h"

struct mtmd_context;

#include <functional>
#include <memory>

struct common_params;
struct common_speculative;
struct server_queue;

// port on server_context_impl exposing the services a scheduler may use.
// implemented by server_context_impl.
// all methods are called from the main loop thread only.
struct server_context_port {
    virtual ~server_context_port() = default;

    virtual common_params & params() = 0;
    virtual llama_context * ctx() = 0;
    virtual std::vector<server_slot> & slot_pool() = 0;
    virtual server_queue & queue() = 0;

    // shared launch path: builds the sampler chain, assigns the task to the
    // slot and sets its state to STARTED. returns false if the task was
    // rejected (error already sent to the client)
    virtual bool launch_slot_with_task(server_slot & slot, server_task && task) = 0;

    // shared sampling-side helpers
    virtual bool process_token(completion_token_output & result, server_slot & slot) = 0;
    virtual void populate_token_probs(const server_slot & slot, completion_token_output & result,
                                      bool post_sampling, bool special, int idx) const = 0;
    virtual void send_partial_response(server_slot & slot, const completion_token_output & tkn,
                                       bool is_progress, bool is_begin = false) = 0;
    virtual void send_final_response(server_slot & slot) = 0;
    virtual void send_error(const server_slot & slot, const std::string & error,
                            error_type type = ERROR_TYPE_SERVER) = 0;
    virtual void send_error(const server_task & task, const std::string & error,
                            error_type type = ERROR_TYPE_SERVER) = 0;

    // metrics hooks (same semantics as the default path)
    virtual void metrics_pre_decode() = 0;
    virtual void metrics_queue_prompt(uint64_t n_tokens) = 0;
    virtual void metrics_flush_prompt() = 0;
    virtual void metrics_flush_idle() = 0;
    virtual server_metrics & srv_metrics() = 0;

    // speculative decoding context, nullptr when disabled
    virtual common_speculative * spec_ctx() = 0;

    // multimodal context for encoding media segments, nullptr when disabled
    virtual mtmd_context * mmctx() = 0;

    // run work on the main thread while a worker serves the task queue,
    // keeping /metrics responsive (see server_queue::yield_to_queue)
    virtual void yield(std::function<void()> && work) = 0;

    // keep the main loop spinning while requests are in flight
    virtual void post_next_response() = 0;
};

// scheduling strategy over the shared llama_context.
// all methods are called from the main loop thread only.
struct server_scheduler_interface {
    virtual ~server_scheduler_interface() = default;

    // called once after load_model() (also after waking up from sleep)
    virtual bool init() = 0;

    // accept one inference-family task. return false when there is no
    // capacity: the caller defers the task via queue_tasks.defer(), like the
    // default path
    virtual bool launch(server_task && task) = 0;

    // abort a queued or running task (CANCEL task, by task id)
    virtual void cancel(int64_t id_task) = 0;

    // one iteration: build batch, llama_decode, sample, emit results
    virtual void update() = 0;
};

// returns the paged scheduler when params.kv_paged is set, nullptr otherwise
// (the default path stays inline in server-context.cpp)
std::unique_ptr<server_scheduler_interface> server_scheduler_create(server_context_port & port);
