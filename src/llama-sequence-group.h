#pragma once

#include "llama.h"

#include <cstdint>
#include <list>
#include <memory>
#include <vector>

enum class llama_sequence_group_status {
    PENDING,
    WAITING,
    RUNNING,
    SWAPPED,
    FINISHED,
    PAUSED_EXTERNAL  // 12.4: waiting for the server to append a media segment's embeddings
};

using llama_block_ids = std::vector<uint32_t>;

// multimodal (12.4): one ordered piece of a request's prompt. text rows carry
// token ids in logical_seq; media rows occupy one KV position each but their
// input embeddings come from the vision encoder (server-owned buffer), and
// each media row carries n_pos_per_embd mrope positions instead of one.
struct llama_seq_segment {
    uint32_t start_row = 0;              // first row index (== logical position)
    uint32_t n_tokens  = 0;              // rows in this segment
    bool     is_media  = false;
    const float * embd        = nullptr; // media: encoder output [n_tokens x embd_stride]; nullptr = pending encode
    int32_t       embd_stride = 0;        // media: floats per row (n_embd_inp)
    std::vector<llama_pos> pos;          // media mrope: n_tokens * 4, row-major [row][4]
};

struct llama_sequence_group {
    int32_t                     request_id = -1;
    llama_sequence_group_status status     = llama_sequence_group_status::PENDING;

    int64_t t_arrival_time   = 0;
    int64_t t_first_token_us = 0;

    // Request priority: smaller values are served first, default 0.
    // Read-only after queue_request().
    int32_t priority = 0;
    // Monotonic enqueue sequence assigned by the scheduler in queue_request().
    // Only used to break ties between equal (priority, t_arrival_time).
    uint64_t arrival_seq = 0;

    uint32_t n_prompt  = 0;
    uint32_t n_prompt_total = 0;  // prompt length before any prefix-cache split
    uint32_t n_decoded = 0;
    uint32_t n_past    = 0;

    std::vector<llama_token> logical_seq;
    llama_block_ids          block_table;
    std::vector<uint64_t>    block_hashes;  // per completed block, prefix cache chain

    // speculative decoding: draft tokens tentatively added to the next step.
    // consumed (cleared) by populate_batch_from
    std::vector<llama_token> pending_draft;

    // multimodal (12.4): ordered segmentation covering all rows; empty for
    // legacy text-only requests. media segments are appended with their
    // encoder output by the server when the scheduler pauses on them.
    std::vector<llama_seq_segment> segments;
    bool has_media       = false;  // any media segment (no prefix cache, no recompute)
    bool paused_external = false;  // waiting for the server to append the pending media segment

    // the segment containing this row, or nullptr for legacy text-only groups
    const llama_seq_segment * segment_at(uint32_t row) const {
        for (const auto & s : segments) {
            if (row >= s.start_row && row < s.start_row + s.n_tokens) {
                return &s;
            }
        }
        return nullptr;
    }

    // the segment at `row` is a media segment whose encoder output has not
    // been appended yet
    bool pending_media_at(uint32_t row) const {
        const llama_seq_segment * s = segment_at(row);
        return s && s->is_media && s->embd == nullptr;
    }
};

using llama_sequence_group_raw_list = std::vector<llama_sequence_group *>;
using llama_sequence_group_ptr      = std::shared_ptr<llama_sequence_group>;
using llama_sequence_group_list     = std::list<llama_sequence_group_ptr>;
