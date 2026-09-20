#ifndef CANN_WEIGHT_LAYOUT_H
#define CANN_WEIGHT_LAYOUT_H

#include "ggml.h"

#include <cstddef>

bool ggml_cann_weight_layout_enabled();
bool ggml_cann_is_weight_layout_candidate(const ggml_tensor * tensor);
size_t ggml_cann_weight_layout_alloc_size(const ggml_tensor * tensor);
void ggml_cann_weight_format_to_nz(ggml_tensor * tensor, int device);

#endif
