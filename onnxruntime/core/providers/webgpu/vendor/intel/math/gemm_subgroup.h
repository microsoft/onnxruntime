// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {
namespace intel {

const uint32_t kSubgroupLogicalWorkGroupSizeX = 32;
const uint32_t kSubgroupLogicalWorkGroupSizeY = 8;
const uint32_t kSubgroupLogicalWorkGroupSizeZ = 1;

bool CanApplySubgroup(const ComputeContext& context, int64_t M, int64_t N, int64_t K, bool transA = false, bool transB = false);

int64_t ElementsPerThreadY(bool is_vec4, uint32_t M);

Status MakeMatMulSubgroupSource(ShaderHelper& shader,
                                const InlinedVector<int64_t>& elements_per_thread,
                                const ShaderIndicesHelper* batch_dims,
                                bool is_vec4,
                                bool transpose_a = false,
                                bool transpose_b = false,
                                float alpha = 1.0f,
                                bool need_handle_matmul = true);

}  // namespace intel
}  // namespace webgpu
}  // namespace onnxruntime
