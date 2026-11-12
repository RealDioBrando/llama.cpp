/*
 * Per-device worker thread for CANN tensor parallelism.
 *
 * The CANN runtime keeps thread-local execution caches (e.g. OpExecCache), so a
 * single host thread that ping-pongs between devices can hit wrong cache
 * entries. With multiple NPUs each CANN backend therefore gets one dedicated
 * worker thread that owns all graph submissions and collectives of its device.
 *
 * Jobs are queued FIFO and never waited per job: the caller posts a whole step
 * (graphs and collectives interleaved) and observes host-side submission
 * completion once via cann_worker_wait_idle(). Device completion is governed
 * by stream ordering, not by host waits.
 *
 * The worker itself is device-agnostic: the graph executor and the collective
 * callbacks bind the device themselves (ggml_cann_set_device is thread-local),
 * which keeps this header free of CANN internals.
 */

#ifndef CANN_WORKER_H
#define CANN_WORKER_H

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <thread>

#include "ggml-backend.h"
#include "ggml.h"

struct cann_worker_job {
    enum class kind_t { graph, callback } kind = kind_t::graph;

    ggml_cgraph * graph = nullptr;                        // kind == graph
    void (*callback)(ggml_backend_t, void *) = nullptr;   // kind == callback
    void * callback_arg = nullptr;
    void (*callback_arg_free)(void *) = nullptr;          // frees callback_arg after the callback ran

    ggml_status status = GGML_STATUS_SUCCESS;
};

struct cann_worker {
    using graph_fn_t = ggml_status (*)(ggml_backend_t, ggml_cgraph *);

    std::thread             thread;
    std::mutex              mutex;
    std::condition_variable cv;

    bool started  = false;
    bool stop     = false;
    bool running  = false; // a popped job is executing (idle == jobs.empty() && !running)

    // Jobs run strictly in FIFO order, so the device stream ordering follows
    // the post order: graphs and collectives of one step interleave safely.
    std::deque<cann_worker_job> jobs;

    // first failed job status, surfaced (and reset) by cann_worker_wait_idle
    ggml_status fail_status = GGML_STATUS_SUCCESS;

    ggml_backend_t backend  = nullptr;
    graph_fn_t     graph_fn = nullptr;
};

inline void cann_worker_loop(cann_worker * w) {
    std::unique_lock<std::mutex> lock(w->mutex);
    while (true) {
        w->cv.wait(lock, [&] { return w->stop || !w->jobs.empty(); });

        while (!w->jobs.empty()) {
            const cann_worker_job job = w->jobs.front();
            w->jobs.pop_front();
            w->running = true;
            lock.unlock();

            ggml_status status = GGML_STATUS_SUCCESS;
            if (job.kind == cann_worker_job::kind_t::graph) {
                status = w->graph_fn(w->backend, job.graph);
            } else {
                job.callback(w->backend, job.callback_arg);
                if (job.callback_arg_free) {
                    job.callback_arg_free(job.callback_arg);
                }
            }

            lock.lock();
            w->running = false;
            if (status != GGML_STATUS_SUCCESS && w->fail_status == GGML_STATUS_SUCCESS) {
                w->fail_status = status;
            }
            w->cv.notify_all();
        }

        if (w->stop) {
            break;
        }
    }
}

inline void cann_worker_start(cann_worker * w, ggml_backend_t backend, cann_worker::graph_fn_t graph_fn) {
    std::lock_guard<std::mutex> lock(w->mutex);
    GGML_ASSERT(!w->started);
    w->started = true;
    w->stop    = false;
    w->backend = backend;
    w->graph_fn = graph_fn;
    w->thread   = std::thread(cann_worker_loop, w);
}

// Drain the pending jobs (if any) and join the worker thread.
inline void cann_worker_stop(cann_worker * w) {
    {
        std::unique_lock<std::mutex> lock(w->mutex);
        if (!w->started) {
            return;
        }
        w->cv.wait(lock, [&] { return w->jobs.empty() && !w->running; });
        w->stop = true;
    }
    w->cv.notify_all();
    if (w->thread.joinable()) {
        w->thread.join();
    }
    w->started = false;
}

// Queue a graph without waiting. The caller observes host-side submission
// completion via cann_worker_wait_idle() (device execution may outlive it).
inline void cann_worker_post_graph(cann_worker * w, ggml_cgraph * graph) {
    std::unique_lock<std::mutex> lock(w->mutex);
    GGML_ASSERT(w->started);
    cann_worker_job job;
    job.kind  = cann_worker_job::kind_t::graph;
    job.graph = graph;
    w->jobs.push_back(job);
    lock.unlock();
    w->cv.notify_all();
}

// Queue a callback without waiting. If arg_free is set, the worker takes
// ownership of arg and frees it after the callback ran.
inline void cann_worker_post_callback(cann_worker * w, void (*callback)(ggml_backend_t, void *),
                                      void * arg, void (*arg_free)(void *) = nullptr) {
    std::unique_lock<std::mutex> lock(w->mutex);
    GGML_ASSERT(w->started);
    cann_worker_job job;
    job.kind              = cann_worker_job::kind_t::callback;
    job.callback          = callback;
    job.callback_arg      = arg;
    job.callback_arg_free = arg_free;
    w->jobs.push_back(job);
    lock.unlock();
    w->cv.notify_all();
}

// Wait until every queued job finished host-side submission, return the first
// failed job status (reset on read). Device completion is not implied.
inline ggml_status cann_worker_wait_idle(cann_worker * w) {
    std::unique_lock<std::mutex> lock(w->mutex);
    if (!w->started) {
        return GGML_STATUS_SUCCESS;
    }
    w->cv.wait(lock, [&] { return w->jobs.empty() && !w->running; });
    const ggml_status status = w->fail_status;
    w->fail_status = GGML_STATUS_SUCCESS;
    return status;
}

#endif // CANN_WORKER_H
