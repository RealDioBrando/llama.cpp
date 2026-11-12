#include "cann-comm.h"

#include "cann-worker.h"
#include "common.h"
#include "ggml-backend-impl.h"
#include "ggml-cann.h"

#include <hccl/hccl.h>
#include <hccl/hccl_comm.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <vector>

#define HCCL_CHECK(stmt)                                                                            \
    do {                                                                                            \
        HcclResult _hccl_ret = (stmt);                                                              \
        if (_hccl_ret != HCCL_SUCCESS) {                                                            \
            GGML_LOG_ERROR("[CANN comm] HCCL error %d at %s:%d\n", (int) _hccl_ret, __FILE__, __LINE__); \
            GGML_ABORT("HCCL error: " #stmt);                                                       \
        }                                                                                           \
    } while (0)

namespace {

struct cann_tp_comm {
    std::vector<ggml_backend_t> backends;
    std::vector<int32_t>        device_ids;
    std::vector<HcclComm>       comms;
};

bool hccl_data_type_for(enum ggml_type type, HcclDataType * out) {
    switch (type) {
        case GGML_TYPE_F32:
            *out = HCCL_DATA_TYPE_FP32;
            return true;
        case GGML_TYPE_F16:
            *out = HCCL_DATA_TYPE_FP16;
            return true;
        case GGML_TYPE_BF16:
            *out = HCCL_DATA_TYPE_BFP16;
            return true;
        default:
            return false;
    }
}

struct allreduce_args {
    int32_t      device = 0;
    HcclComm     comm   = nullptr;
    void *       buf    = nullptr;
    uint64_t     count  = 0;
    size_t       nbytes = 0;
    HcclDataType dtype  = HCCL_DATA_TYPE_FP32;
    bool         zero   = false; // inactive shard: clear before the reduce
    aclrtStream  stream = nullptr;
};

void allreduce_callback(ggml_backend_t backend, void * varg) {
    GGML_UNUSED(backend);
    allreduce_args * args = (allreduce_args *) varg;

    ggml_cann_set_device(args->device);

    // An inactive shard holds stale data from the previous step, it must be
    // zeroed before participating in the sum. Same stream => ordered before
    // the AllReduce.
    if (args->zero) {
        ACL_CHECK(aclrtMemsetAsync(args->buf, args->nbytes, 0, args->nbytes, args->stream));
    }

    HCCL_CHECK(HcclAllReduce(args->buf, args->buf, args->count, args->dtype, HCCL_REDUCE_SUM,
                             args->comm, args->stream));
}

static void allreduce_args_free(void * p) {
    delete (allreduce_args *) p;
}

} // namespace

void * ggml_backend_cann_comm_init(ggml_backend_t * backends, size_t n_backends) {
    if (n_backends == 0 || backends == nullptr) {
        return nullptr;
    }
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cann(backends[i])) {
            return nullptr;
        }
    }

    auto * comm = new cann_tp_comm();
    comm->backends.assign(backends, backends + n_backends);
    comm->device_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        comm->device_ids.push_back(ggml_backend_cann_device_id(backends[i]));
    }
    for (size_t i = 0; i < n_backends; i++) {
        // one communicator per device, ranks follow the backend order
        if (std::find(comm->device_ids.begin(), comm->device_ids.end(), comm->device_ids[i]) !=
                comm->device_ids.begin() + i) {
            GGML_LOG_ERROR("%s: duplicate device in TP group\n", __func__);
            delete comm;
            return nullptr;
        }
    }

    comm->comms.assign(n_backends, nullptr);
    HcclResult res = HcclCommInitAll((uint32_t) n_backends, comm->device_ids.data(), comm->comms.data());
    if (res != HCCL_SUCCESS) {
        GGML_LOG_ERROR("%s: HcclCommInitAll failed with %d\n", __func__, (int) res);
        delete comm;
        return nullptr;
    }

    return comm;
}

void ggml_backend_cann_comm_free(void * comm_ctx) {
    auto * comm = (cann_tp_comm *) comm_ctx;
    if (comm == nullptr) {
        return;
    }

    // All collectives must be complete before the communicators are destroyed.
    for (auto * backend : comm->backends) {
        ggml_backend_synchronize(backend);
    }
    for (size_t i = 0; i < comm->comms.size(); i++) {
        if (comm->comms[i] != nullptr) {
            ggml_cann_set_device(comm->device_ids[i]);
            HcclResult res = HcclCommDestroy(comm->comms[i]);
            if (res != HCCL_SUCCESS) {
                GGML_LOG_ERROR("%s: HcclCommDestroy(%zu) failed with %d\n", __func__, i, (int) res);
            }
        }
    }

    delete comm;
}

bool ggml_backend_cann_comm_allreduce_tensor(void * comm_ctx, struct ggml_tensor ** tensors) {
    auto * comm = (cann_tp_comm *) comm_ctx;
    if (comm == nullptr || tensors == nullptr) {
        return false;
    }
    const size_t n = comm->backends.size();

    // Validation happens before anything is enqueued: on failure the meta
    // backend may still fall back to its generic allreduce.
    HcclDataType dtype;
    if (!hccl_data_type_for(tensors[0]->type, &dtype)) {
        return false;
    }
    const int64_t count = ggml_nelements(tensors[0]);
    for (size_t i = 0; i < n; i++) {
        if (tensors[i] == nullptr || tensors[i]->data == nullptr) {
            return false;
        }
        if (tensors[i]->type != tensors[0]->type ||
                ggml_nelements(tensors[i]) != count ||
                !ggml_is_contiguous(tensors[i])) {
            return false;
        }
    }
    if (count == 0) {
        return true;
    }

    std::vector<std::unique_ptr<allreduce_args>> args(n);
    std::vector<cann_worker *>  workers(n, nullptr);
    size_t n_workers = 0;
    for (size_t i = 0; i < n; i++) {
        ggml_backend_cann_context * cann_ctx =
            (ggml_backend_cann_context *) comm->backends[i]->context;
        args[i] = std::make_unique<allreduce_args>();
        args[i]->device = comm->device_ids[i];
        args[i]->comm   = comm->comms[i];
        args[i]->buf    = tensors[i]->data;
        args[i]->count  = (uint64_t) count;
        args[i]->nbytes = ggml_nbytes(tensors[i]);
        args[i]->dtype  = dtype;
        // zero-sized input slices leave the shard inactive, its contents are stale
        args[i]->zero   = (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0;
        args[i]->stream = cann_ctx->stream();

        workers[i] = ggml_backend_cann_get_worker(comm->backends[i]);
        n_workers += workers[i] != nullptr;
    }

    // AllReduce execution: each rank's HCCL call is posted to its own worker
    // (FIFO naturally orders it after the graphs already posted for that
    // rank), and the host returns immediately, per design principle 5 (no
    // per-step host synchronize after a collective). Ordering relies on the
    // workers' FIFO submission; callback failures surface at the next
    // synchronize (wait_idle in the backend sync path).
    if (n_workers == n) {
        for (size_t i = 0; i < n; i++) {
            cann_worker_post_callback(workers[i], allreduce_callback, args[i].release(), allreduce_args_free);
        }
    } else {
        // Diagnostic mode (GGML_CANN_DISABLE_WORKER=1): HCCL collectives
        // cannot be submitted sequentially from one thread (rank 0 blocks in
        // HcclAllReduce before rank 1 is submitted, deadlock). Let the meta
        // backend use its fallback reduction instead.
        return false;
    }

    return true;
}
