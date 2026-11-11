/*
 * HCCL communication hooks for CANN tensor parallelism (see
 * cann-tp-combined-design.md section 6). The meta backend probes these via
 * ggml_backend_reg_get_proc_address and uses them to allreduce PARTIAL
 * tensors; if they are absent or refuse a tensor, the meta backend falls back
 * to its generic copy+add butterfly reduction.
 *
 * The proc address names must match the ones probed by the meta backend:
 *   ggml_backend_comm_init
 *   ggml_backend_comm_free
 *   ggml_backend_comm_allreduce_tensor
 */

#ifndef CANN_COMM_H
#define CANN_COMM_H

#include <stddef.h>

#include "ggml-backend.h"

// Internal helpers implemented in ggml-cann.cpp, used by cann-comm.cpp.
int32_t       ggml_backend_cann_device_id(ggml_backend_t backend);
struct cann_worker;

// Returns the per-device worker of the backend, or nullptr when the worker is
// disabled (single device or GGML_CANN_DISABLE_WORKER=1).
cann_worker * ggml_backend_cann_get_worker(ggml_backend_t backend);

// Communication hooks probed by the meta backend, implemented in cann-comm.cpp.
void * ggml_backend_cann_comm_init(ggml_backend_t * backends, size_t n_backends);
void   ggml_backend_cann_comm_free(void * comm_ctx);
bool   ggml_backend_cann_comm_allreduce_tensor(void * comm_ctx, struct ggml_tensor ** tensors);

#endif // CANN_COMM_H
