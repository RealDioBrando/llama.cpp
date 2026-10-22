#include "weight_layout.h"

#include "acl_tensor.h"
#include "common.h"

#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_permute.h>
#include <aclnnop/aclnn_trans_matmul_weight.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_set>

namespace {

constexpr int GGML_CANN_LAYOUT_MAX_DEVICES = 16;

struct nz_workspace {
    std::mutex mtx;
    void *  ptr       = nullptr;
    size_t  allocated = 0;

    void realloc(size_t size) {
        if (size <= allocated) {
            return;
        }
        clear();
        ACL_CHECK(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST));
        allocated = size;
    }

    void clear() {
        if (ptr != nullptr) {
            ACL_CHECK(aclrtFree(ptr));
            ptr       = nullptr;
            allocated = 0;
        }
    }
};

nz_workspace workspaces[GGML_CANN_LAYOUT_MAX_DEVICES];
nz_workspace expert_workspaces[GGML_CANN_LAYOUT_MAX_DEVICES];

bool has_suffix(const std::string & name, const std::unordered_set<std::string> & suffixes) {
    for (const auto & suffix : suffixes) {
        if (name.find(suffix) != std::string::npos) {
            return true;
        }
    }
    return false;
}

bool is_dense_weight_name(const ggml_tensor * tensor) {
    static const std::unordered_set<std::string> suffixes = {
        "output.weight",      "attn_q.weight",      "attn_k.weight",      "attn_v.weight",
        "attn_output.weight", "attn_qkv.weight",    "attn_gate.weight",   "ffn_gate.weight",
        "ffn_up.weight",      "ffn_down.weight",    "ffn_gate_inp.weight", "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight", "ffn_down_shexp.weight", "ffn_gate_inp_shexp.weight",
        "ssm_alpha.weight",   "ssm_beta.weight",    "ssm_out.weight",
    };
    return has_suffix(ggml_get_name(tensor), suffixes);
}

bool is_expert_weight_name(const ggml_tensor * tensor) {
    static const std::unordered_set<std::string> suffixes = {
        "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
        "ffn_gate_up_exps.weight",
    };
    return has_suffix(ggml_get_name(tensor), suffixes);
}

bool can_transform_slice(const ggml_tensor * tensor) {
    return (tensor->ne[0] % 32 == 0) && (tensor->ne[1] % 32 == 0);
}

void transform_2d_weight(void * data, aclDataType type, size_t type_size, int64_t ne[2], size_t nb[2],
                         int device) {
    acl_tensor_ptr weight = ggml_cann_create_tensor(data, type, type_size, ne, nb, 2, ACL_FORMAT_ND);
    uint64_t       workspace_size = 0;
    aclOpExecutor * executor = nullptr;

    ACL_CHECK(aclnnTransMatmulWeightGetWorkspaceSize(weight.get(), &workspace_size, &executor));

    std::lock_guard<std::mutex> lock(workspaces[device % GGML_CANN_LAYOUT_MAX_DEVICES].mtx);
    workspaces[device % GGML_CANN_LAYOUT_MAX_DEVICES].realloc(workspace_size);
    ACL_CHECK(aclnnTransMatmulWeight(workspaces[device % GGML_CANN_LAYOUT_MAX_DEVICES].ptr,
                                     workspace_size, executor, nullptr));
}

static aclTensor * make_2d_nd_tensor(void * data, aclDataType type, int64_t rows, int64_t cols,
                                     int64_t row_stride_elements) {
    int64_t ne[2]      = { rows, cols };
    int64_t stride[2]  = { row_stride_elements, 1 };
    int64_t storage[2] = { rows, cols };
    return aclCreateTensor(ne, 2, type, stride, 0, ACL_FORMAT_ND, storage, 2, data);
}

static void transform_expert_weight(void * data, aclDataType type, size_t type_size, int64_t ne[2],
                                    int64_t n_experts, int device) {
    const int64_t rows     = ne[0];
    const int64_t cols     = ne[1];
    const size_t  nd_bytes = rows * cols * type_size;

    ggml_tensor shape_only = {};
    shape_only.ne[0] = rows;
    shape_only.ne[1] = cols;
    shape_only.type = GGML_TYPE_F16;
    const size_t nz_bytes = calculate_2d_nz_size(&shape_only);
    const size_t transform_bytes = std::max(nd_bytes, nz_bytes);

    nz_workspace & ws = expert_workspaces[device % GGML_CANN_LAYOUT_MAX_DEVICES];
    std::lock_guard<std::mutex> lock(ws.mtx);
    ws.realloc(transform_bytes);

    for (int64_t e = 0; e < n_experts; ++e) {
        void * expert_ptr = (char *) data + e * nd_bytes;
        acl_tensor_ptr src(make_2d_nd_tensor(expert_ptr, type, rows, cols, cols));
        acl_tensor_ptr dst(make_2d_nd_tensor(ws.ptr, type, cols, rows, rows));
        const int64_t perm_data[2] = { 1, 0 };
        acl_int_array_ptr perm(aclCreateIntArray(perm_data, 2));

        uint64_t perm_ws_size = 0;
        aclOpExecutor * perm_executor = nullptr;
        ACL_CHECK(aclnnPermuteGetWorkspaceSize(src.get(), perm.get(), dst.get(),
                                               &perm_ws_size, &perm_executor));
        void * perm_ws = nullptr;
        if (perm_ws_size > 0) {
            ACL_CHECK(aclrtMalloc(&perm_ws, perm_ws_size, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        ACL_CHECK(aclnnPermute(perm_ws, perm_ws_size, perm_executor, nullptr));
        if (perm_ws) {
            aclrtFree(perm_ws);
        }
        ACL_CHECK(aclrtSynchronizeStream(nullptr));

        uint64_t nz_ws_size = 0;
        aclOpExecutor * nz_executor = nullptr;
        ACL_CHECK(aclnnTransMatmulWeightGetWorkspaceSize(dst.get(), &nz_ws_size, &nz_executor));
        void * nz_ws = nullptr;
        if (nz_ws_size > 0) {
            ACL_CHECK(aclrtMalloc(&nz_ws, nz_ws_size, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        ACL_CHECK(aclnnTransMatmulWeight(nz_ws, nz_ws_size, nz_executor, nullptr));
        if (nz_ws) {
            aclrtFree(nz_ws);
        }
        ACL_CHECK(aclrtSynchronizeStream(nullptr));
        ACL_CHECK(aclrtMemcpy(expert_ptr, nd_bytes, ws.ptr, nz_bytes, ACL_MEMCPY_DEVICE_TO_DEVICE));
    }
}

size_t calculate_2d_nz_size(const ggml_tensor * tensor) {
    int64_t shape[2] = { tensor->ne[1], tensor->ne[0] };
    const aclIntArray * acl_shape = aclCreateIntArray(shape, 2);
    size_t              size      = 0;
    ACL_CHECK(aclnnCalculateMatmulWeightSizeV2(acl_shape, ggml_cann_type_mapping(tensor->type), &size));
    ACL_CHECK(aclDestroyIntArray(acl_shape));
    return size;
}

}  // namespace

bool ggml_cann_moe_grouped_enabled() {
    static const bool enabled =
        parse_bool(get_env_as_lowercase("GGML_CANN_MOE_GROUPED").value_or("on"));
    return enabled;
}

bool ggml_cann_weight_layout_enabled() {
    static const bool enabled = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));
    return enabled;
}

bool ggml_cann_is_weight_layout_candidate(const ggml_tensor * tensor) {
    if (!ggml_cann_weight_layout_enabled() || tensor->type == GGML_TYPE_BF16 ||
        !ggml_is_contiguous(tensor)) {
        return false;
    }

    // Expert NZ is consumed by the grouped-matmul paths in moe_ops and
    // moe_fused_ops.
    if (tensor->ne[2] > 1 || tensor->ne[3] > 1) {
        return tensor->ne[3] == 1 && tensor->type == GGML_TYPE_F16 && is_expert_weight_name(tensor) &&
               ggml_cann_moe_grouped_enabled() &&
               can_transform_slice(tensor);
    }

    return is_dense_weight_name(tensor) && can_transform_slice(tensor);
}

size_t ggml_cann_weight_layout_alloc_size(const ggml_tensor * tensor) {
    if (!ggml_cann_is_weight_layout_candidate(tensor)) {
        return ggml_nbytes(tensor);
    }

    if (tensor->ne[2] > 1) {
        // In-place expert transforms cannot expand one expert slice into the next.
        ggml_tensor slice = *tensor;
        slice.ne[2] = 1;
        slice.ne[3] = 1;
        slice.nb[1] = tensor->nb[1];
        slice.nb[2] = tensor->nb[2];
        const size_t nd_size  = tensor->ne[0] * tensor->ne[1] * ggml_type_size(tensor->type);
        const size_t nz_size  = calculate_2d_nz_size(&slice);
        return tensor->ne[2] * std::max(nd_size, nz_size);
    }

    return std::max(ggml_nbytes(tensor), calculate_2d_nz_size(tensor));
}

void ggml_cann_weight_format_to_nz(ggml_tensor * tensor, int device) {
    if (!ggml_cann_is_weight_layout_candidate(tensor)) {
        return;
    }

    if (tensor->ne[2] > 1) {
        int64_t ne[2] = { tensor->ne[0], tensor->ne[1] };
        transform_expert_weight(tensor->data, ggml_cann_type_mapping(tensor->type),
                                ggml_type_size(tensor->type), ne, tensor->ne[2], device);
        return;
    }

    transform_2d_weight(tensor->data, ggml_cann_type_mapping(tensor->type),
                        ggml_type_size(tensor->type), tensor->ne, tensor->nb, device);
}

#ifdef ASCEND_310P
bool ggml_cann_tied_output_weight_candidate(const ggml_tensor * tensor) {
    static const bool tied_output_enabled = parse_bool(
        get_env_as_lowercase("GGML_CANN_TIED_OUTPUT_NZ").value_or("on"));
    return tied_output_enabled && ggml_cann_weight_layout_enabled() && tensor->type == GGML_TYPE_F16 &&
           ggml_is_contiguous(tensor) && tensor->ne[2] == 1 && tensor->ne[3] == 1 &&
           strcmp(ggml_get_name(tensor), "token_embd.weight") == 0 &&
           can_transform_slice(tensor);
}

size_t ggml_cann_tied_output_weight_alloc_size(const ggml_tensor * tensor) {
    if (!ggml_cann_tied_output_weight_candidate(tensor)) {
        return 0;
    }
    return std::max(ggml_nbytes(tensor), calculate_2d_nz_size(tensor));
}

void ggml_cann_tied_output_weight_to_nz(const ggml_tensor * tensor, void * dst, int device) {
    if (!ggml_cann_tied_output_weight_candidate(tensor)) {
        return;
    }

    int64_t ne[2] = { tensor->ne[0], tensor->ne[1] };
    size_t  nb[2] = { tensor->nb[0], tensor->nb[1] };
    transform_2d_weight(dst, ggml_cann_type_mapping(tensor->type),
                        ggml_type_size(tensor->type), ne, nb, device);
}

bool ggml_cann_prepare_tied_output_weight(const ggml_tensor * tensor,
                                          ggml_cann_tied_output_weight_cache * cache,
                                          int device) {
    if (!ggml_cann_tied_output_weight_candidate(tensor) || cache == nullptr) {
        return false;
    }

    auto it = cache->entries.find(tensor);
    if (it != cache->entries.end()) {
        return true;
    }

    const size_t tied_bytes = ggml_cann_tied_output_weight_alloc_size(tensor);
    if (tied_bytes == 0) {
        return false;
    }

    auto entry = std::make_unique<ggml_cann_tied_output_weight_cache::entry>();
    ACL_CHECK(aclrtMalloc(&entry->buffer, tied_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    entry->size = tied_bytes;
    ACL_CHECK(aclrtMemcpy(entry->buffer, ggml_nbytes(tensor), tensor->data,
                          ggml_nbytes(tensor), ACL_MEMCPY_DEVICE_TO_DEVICE));
    ggml_cann_tied_output_weight_to_nz(tensor, entry->buffer, device);
    cache->entries.emplace(tensor, std::move(entry));
    return true;
}
#endif
