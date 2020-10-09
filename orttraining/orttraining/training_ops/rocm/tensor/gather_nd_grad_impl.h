// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/rocm/shared_inc/rocm_utils.h"

namespace onnxruntime {
namespace rocm {

template <typename T>
void GatherNDGradImpl(
    const size_t num_slices,
    const void* update_data,
    void* output_data,
    const size_t slice_size,
    const int64_t* input_slice_offsets_data);

}  // namespace rocm
}  // namespace onnxruntime
