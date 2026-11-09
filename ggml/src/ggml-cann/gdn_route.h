// Host routing before any GDN submission. No fallback after an ACLNN error.
#pragma once

#include <cstdint>

namespace gdn_route {

// Conservative bounds for the recurrent kernel's signed token/state indices.
// K is GGML's snapshot count, not a feature dimension or an accepted-token count.
inline bool snapshot_eligible(int64_t b, int64_t t, int64_t k, int64_t h, int64_t d) {
    if (b < 1 || t < 1 || k < 2 || b > INT32_MAX || t > INT32_MAX || k > INT32_MAX ||
        h < 1 || h > 256 || (d != 64 && d != 128)) {
        return false;
    }
    const int64_t w = t < k ? t : k;
    const int64_t state_size = h * d * d;
    return b <= INT32_MAX / t / (h * d) &&
           b <= INT32_MAX / (w + 1) / state_size &&
           b <= INT32_MAX / k / state_size;
}

inline int64_t snapshot_index(int64_t b, int64_t t, int64_t batch, int64_t tokens, int64_t k) {
    const int64_t w = tokens < k ? tokens : k;
    const int64_t slot = tokens - 1 - t;
    return (slot < w ? slot : w) * batch + b;
}

enum class PrefillRoute : int { kRecurrent = 0, kChunkStandard = 3 };

// Aligned shapes accepted by this backend route; the operator supports a broader domain.
inline bool aligned_shape_eligible(int64_t nk, int64_t nv, int64_t dk, int64_t dv)
{
    return nk > 0 && nv > 0 && nk <= 64 && nv <= 64 &&
           dk > 0 && dv > 0 && dk <= 128 && dv <= 128 &&
           dk % 16 == 0 && dv % 16 == 0 && nv % nk == 0;
}

// RefChain uses padded 128-dim storage and signed 32-bit scheduler offsets.
// t is per sequence here; the operator receives total tokens t*b.
inline bool standard_eligible(int64_t b, int64_t t, int64_t nk, int64_t nv, int64_t dk, int64_t dv,
                              bool odd_heads_supported = false)
{
    if (!aligned_shape_eligible(nk, nv, dk, dv) ||
        b < 1 || t < 1 || b > INT32_MAX || t > INT32_MAX) {
        return false;
    }
    // Odd heads require package support and the backend's equal-head D64/128 domain.
    if (nv % 2 != 0 && (!odd_heads_supported || nk != nv || dk != dv || (dk != 64 && dk != 128))) {
        return false;
    }
    const int64_t internal_nv = nv % 2 ? nv + 1 : nv;
    const int64_t total = t * b;
    const int64_t chunks = total / 64 + (total % 64 != 0) + b - 1;
    const int64_t limit = (INT32_MAX - 1) / (internal_nv * 128 * 128);
    // The H bound also bounds token buffers and metadata; B bounds states.
    return chunks <= limit && b <= limit;
}

// Select before submission. A failed GetWorkspaceSize/Run never changes the route.
inline PrefillRoute prefill_route(int64_t t, int64_t b, int64_t nk, int64_t nv, int64_t dk, int64_t dv,
                                  bool use_chunk, bool chunk_syms, bool odd_heads_supported = false)
{
    return t != 1 && use_chunk && chunk_syms && standard_eligible(b, t, nk, nv, dk, dv, odd_heads_supported)
        ? PrefillRoute::kChunkStandard : PrefillRoute::kRecurrent;
}

// Restrict this opt-in optimization to the measured workload class.
// Other supported inputs keep the recurrent snapshot path.
inline int64_t snapshot_prefix(int64_t b, int64_t t, int64_t k, int64_t h, int64_t d,
                               bool enabled, bool use_chunk, bool chunk_syms, bool odd_heads_supported) {
    if (!enabled || !snapshot_eligible(b, t, k, h, d)) {
        return 0;
    }
    const int64_t prefix = t - (t < k ? t : k);
    const bool measured_class = d == 128 && h == 4 && (b == 2 || b == 4) && (k == 4 || k == 9);
    return measured_class && prefix >= 508 && prefill_route(prefix, b, h, h, d, d, use_chunk, chunk_syms,
                                        odd_heads_supported) == PrefillRoute::kChunkStandard ? prefix : 0;
}

inline const char * route_name(PrefillRoute r)
{
    return r == PrefillRoute::kChunkStandard ? "chunk-standard" : "recurrent";
}

}  // namespace gdn_route
