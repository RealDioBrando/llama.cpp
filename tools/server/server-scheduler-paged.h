#pragma once

// Paged continuous-batching scheduler for llama-server, driving the
// llama_paged_scheduler_* C API (see examples/paged/paged.cpp for the raw
// loop this mirrors). Enabled with --kv-paged.

#include "server-scheduler.h"

struct llama_paged_scheduler;

class server_scheduler_paged final : public server_scheduler_interface {
  public:
    explicit server_scheduler_paged(server_scheduler_host & host) : host(host) {}
    ~server_scheduler_paged() override;

    bool init() override;
    bool launch(server_task && task) override;
    void cancel(int64_t id_task) override;
    void update() override;

  private:
    server_slot * pick_idle_slot();
    void fail_all(const std::string & why);
    void on_finish(int32_t request_id);
    static void on_finish_thunk(int32_t request_id, const llama_token * tokens, int32_t n_tokens, void * user_data);

    server_scheduler_host & host;

    llama_paged_scheduler * sched = nullptr;

    // owned and recycled by llama_paged_scheduler_prepare_batch
    llama_batch paged_batch = {};

    // per-slot flag: the request was cancelled client-side, do not send a
    // final response when the scheduler reaps it
    std::vector<int8_t> cancelled;

    // busy-poll safeguard: in-flight requests but the scheduler keeps
    // producing empty batches
    int n_empty_steps = 0;
};
