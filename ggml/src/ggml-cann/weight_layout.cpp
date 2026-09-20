#include "weight_layout.h"

#include "acl_tensor.h"
#include "common.h"

#include <aclnnop/aclnn_matmul.h>
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

size_t calculate_2d_nz_size(const ggml_tensor * tensor) {
    int64_t shape[2] = { tensor->ne[1], tensor->ne[0] };
    const aclIntArray * acl_shape = aclCreateIntArray(shape, 2);
    size_t              size      = 0;
    ACL_CHECK(aclnnCalculateMatmulWeightSizeV2(acl_shape, ggml_cann_type_mapping(tensor->type), &size));
    ACL_CHECK(aclDestroyIntArray(acl_shape));
    return size;
}

}  // namespace

bool ggml_cann_weight_layout_enabled() {
    static const bool enabled = parse_bool(get_env_as_lowercase("GGML_CANN_WEIGHT_NZ").value_or("on"));
    return enabled;
}

bool ggml_cann_is_weight_layout_candidate(const ggml_tensor * tensor) {
    if (!ggml_cann_weight_layout_enabled() || tensor->type == GGML_TYPE_BF16 ||
        !ggml_is_contiguous(tensor)) {
        return false;
    }

    // Expert NZ needs the grouped-matmul paths to build FRACTAL_NZ tensors
    // (moe_ops / moe_fused_ops in the upstream module). This branch does not
    // carry them yet, and buffer set_tensor asserts ne[2] == 1, so keep
    // expert weights in ND format here.
    if (tensor->ne[2] > 1 || tensor->ne[3] > 1) {
        return false;
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
        return nz_size <= nd_size ? ggml_nbytes(tensor) : ggml_nbytes(tensor) + (nz_size - nd_size);
    }

    return std::max(ggml_nbytes(tensor), calculate_2d_nz_size(tensor));
}

void ggml_cann_weight_format_to_nz(ggml_tensor * tensor, int device) {
    if (!ggml_cann_is_weight_layout_candidate(tensor)) {
        return;
    }

    if (tensor->ne[2] > 1) {
        for (int64_t e = 0; e < tensor->ne[2]; ++e) {
            int64_t ne[2]     = { tensor->ne[0], tensor->ne[1] };
            size_t  nb[2]     = { tensor->nb[0], tensor->nb[1] };
            void *  expert_ptr = (char *) tensor->data + e * tensor->nb[2];
            transform_2d_weight(expert_ptr, ggml_cann_type_mapping(tensor->type),
                                ggml_type_size(tensor->type), ne, nb, device);
        }
        return;
    }

    transform_2d_weight(tensor->data, ggml_cann_type_mapping(tensor->type),
                        ggml_type_size(tensor->type), tensor->ne, tensor->nb, device);
}
