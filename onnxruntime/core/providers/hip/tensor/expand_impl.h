// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>

#include "core/common/common.h"
#include "core/framework/data_types.h"
#include "core/providers/hip/hip_common.h"
#include "core/providers/hip/hip_utils.h"

namespace onnxruntime {
namespace hip {

Status ExpandImpl(
    const size_t element_size,
    const int N_output,
    const int N_input,
    const void* input_data,
    void* output_data,
    HipKernel::HipAsyncBuffer<fast_divmod>& fdm_output_strides, 
    HipKernel::HipAsyncBuffer<int64_t>& input_view_strides);


}  // namespace hip
}  // namespace onnxruntime
