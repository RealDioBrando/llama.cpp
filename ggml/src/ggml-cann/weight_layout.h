#ifndef CANN_WEIGHT_LAYOUT_H
#define CANN_WEIGHT_LAYOUT_H

#include "ggml.h"

#include <cstddef>

bool ggml_cann_weight_layout_enabled();
bool ggml_cann_is_weight_layout_candidate(const ggml_tensor * tensor);
size_t ggml_cann_weight_layout_alloc_size(const ggml_tensor * tensor);
void ggml_cann_weight_format_to_nz(ggml_tensor * tensor, int device);

#ifdef ASCEND_310P
bool ggml_cann_tied_output_weight_candidate(const ggml_tensor * tensor);
size_t ggml_cann_tied_output_weight_alloc_size(const ggml_tensor * tensor);
void ggml_cann_tied_output_weight_to_nz(const ggml_tensor * tensor, void * dst, int device);
#endif

#endif
