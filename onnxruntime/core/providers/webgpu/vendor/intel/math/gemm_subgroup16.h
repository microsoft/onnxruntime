// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/vendor/intel/math/gemm_subgroup_utils.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

// The subgroup-16 kernel runs kSubgroup16WorkGroupSizeY rows of kSubgroup16WorkGroupSizeX invocations and relies on
// each row being exactly one subgroup, so it requires subgroup size kMatMulSubgroup16Size.
const uint32_t kSubgroup16WorkGroupSizeX = kMatMulSubgroup16Size;
const uint32_t kSubgroup16WorkGroupSizeY = 8;

// Each invocation computes elements_per_thread[1] rows x 4 columns of the output. Gemm callers must not transpose
// A or B, and `alpha` scales the product before mm_write (which applies any bias).
Status MakeMatMulSubgroup16Source(ShaderHelper& shader,
                                  const InlinedVector<int64_t>& elements_per_thread,
                                  const ShaderIndicesHelper* batch_dims,
                                  bool is_vec4,
                                  bool a_vec4,
                                  bool use_f32_accumulation,
                                  float alpha = 1.0f,
                                  bool need_handle_matmul = true);

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
