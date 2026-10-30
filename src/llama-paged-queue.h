#pragma once

#include "llama-sequence-group.h"

#include <algorithm>
#include <deque>
#include <memory>
#include <queue>
#include <vector>

//
// Request ordering keys shared by the scheduler and the waiting/swapped queues.
// The direction of the priority ordering is defined exactly once here so the
// queue implementations and the scheduling predicates cannot drift apart.
//

// true if `a` should be served before `b` under priority scheduling.
// Semantics: smaller `priority` values are served first; equal priorities
// fall back to arrival order (t_arrival_time, then arrival_seq).
inline bool group_precedes(const llama_sequence_group & a, const llama_sequence_group & b) {
    if (a.priority != b.priority) {
        return a.priority < b.priority;  // do not subtract: int32 can overflow
    }
    if (a.t_arrival_time != b.t_arrival_time) {
        return a.t_arrival_time < b.t_arrival_time;
    }
    return a.arrival_seq < b.arrival_seq;  // final key makes the order total
}

// true if `a` reached the scheduler before `b`. Used by the FCFS queue and by
// the arrival-ordered `running` list.
inline bool arrival_precedes(const llama_sequence_group & a, const llama_sequence_group & b) {
    if (a.t_arrival_time != b.t_arrival_time) {
        return a.t_arrival_time < b.t_arrival_time;
    }
    return a.arrival_seq < b.arrival_seq;
}

// true if `candidate` should be evicted instead of the current `victim`.
// Priority mode sacrifices the lowest priority (largest value); ties and FCFS
// sacrifice the most recently arrived request.
inline bool eviction_precedes(const llama_sequence_group & candidate,
                              const llama_sequence_group & victim,
                              bool                         by_priority) {
    if (by_priority && candidate.priority != victim.priority) {
        return candidate.priority > victim.priority;
    }
    return candidate.arrival_seq > victim.arrival_seq;
}

// Minimal queue interface used for both `waiting` and `swapped`.
// Ownership lives in the queue; peek() returns a borrowed pointer.
class llama_paged_queue {
  public:
    virtual ~llama_paged_queue() = default;

    virtual void                  push(llama_sequence_group_ptr g) = 0;
    virtual llama_sequence_group * peek() const                     = 0;
    virtual llama_sequence_group_ptr pop()                          = 0;
    virtual bool                  empty() const                     = 0;
    virtual size_t                size() const                      = 0;
};

// FCFS order: std::deque with keyed insertion.
// New requests land at the back naturally, but swapped/recompute requests carry
// older keys, so insertion must respect (t_arrival_time, arrival_seq) to keep
// the pre-existing admission/swap-in order.
class llama_paged_fcfs_queue : public llama_paged_queue {
  public:
    void push(llama_sequence_group_ptr g) override {
        GGML_ASSERT(g && "cannot enqueue a nullptr group");
        auto it = std::lower_bound(queue_.begin(), queue_.end(), *g,
                                   [](const llama_sequence_group_ptr & queued, const llama_sequence_group & key) {
                                       return arrival_precedes(*queued, key);
                                   });
        queue_.insert(it, std::move(g));
    }

    llama_sequence_group * peek() const override {
        GGML_ASSERT(!queue_.empty() && "peek() on an empty FCFS queue");
        return queue_.front().get();
    }

    llama_sequence_group_ptr pop() override {
        GGML_ASSERT(!queue_.empty() && "pop() on an empty FCFS queue");
        llama_sequence_group_ptr g = std::move(queue_.front());
        queue_.pop_front();
        return g;
    }

    bool   empty() const override { return queue_.empty(); }
    size_t size() const override { return queue_.size(); }

  private:
    std::deque<llama_sequence_group_ptr> queue_;
};

// std::priority_queue builds a max-heap according to its comparator, while the
// heap top must be the request that group_precedes() ranks first. Passing the
// reversed predicate yields the required min-heap semantics (same as vLLM's
// heapq-based PriorityRequestQueue).
struct group_follows {
    bool operator()(const llama_sequence_group_ptr & a, const llama_sequence_group_ptr & b) const {
        return group_precedes(*b, *a);
    }
};

// Priority order: std::priority_queue over shared_ptr elements.
// shared_ptr is copyable, so top() (which returns a const reference) can be
// copied out before pop(); a unique_ptr cannot be extracted from the public
// interface because it is neither copyable nor returned by top().
class llama_paged_priority_queue : public llama_paged_queue {
  public:
    void push(llama_sequence_group_ptr g) override {
        GGML_ASSERT(g && "cannot enqueue a nullptr group");
        queue_.push(std::move(g));
    }

    llama_sequence_group * peek() const override {
        GGML_ASSERT(!queue_.empty() && "peek() on an empty priority queue");
        return queue_.top().get();
    }

    llama_sequence_group_ptr pop() override {
        GGML_ASSERT(!queue_.empty() && "pop() on an empty priority queue");
        // top() only exposes a const reference, so copy the shared_ptr and
        // let pop() release the queue's own reference.
        llama_sequence_group_ptr g = queue_.top();
        queue_.pop();
        return g;
    }

    bool   empty() const override { return queue_.empty(); }
    size_t size() const override { return queue_.size(); }

  private:
    std::priority_queue<llama_sequence_group_ptr, std::vector<llama_sequence_group_ptr>, group_follows> queue_;
};

inline std::unique_ptr<llama_paged_queue> llama_paged_queue_create(bool by_priority) {
    if (by_priority) {
        return std::make_unique<llama_paged_priority_queue>();
    }
    return std::make_unique<llama_paged_fcfs_queue>();
}
