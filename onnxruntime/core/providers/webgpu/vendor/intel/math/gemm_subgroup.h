// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/shader_helper.h"

namespace onnxruntime {
namespace webgpu {

const uint32_t kSubgroupLogicalWorkGroupSizeX = 32;
const uint32_t kSubgroupLogicalWorkGroupSizeY = 8;
const uint32_t kSubgroupLogicalWorkGroupSizeZ = 1;

Status MakeMatMulSubgroupVec4Source(ShaderHelper& shader,
                                    const InlinedVector<int64_t>& elements_per_thread,
                                    const std::string& data_type,
                                    const ShaderIndicesHelper* batch_dims,
                                    bool transpose_a = false,
                                    bool transpose_b = false,
                                    float alpha = 1.0f,
                                    bool need_handle_matmul = true,
                                    uint32_t tile_inner = 32);

Status MakeMatMulSubgroupSource(ShaderHelper& shader,
                                const InlinedVector<int64_t>& elements_per_thread,
                                const std::string& data_type,
                                const ShaderIndicesHelper* batch_dims,
                                bool transpose_a = false,
                                bool transpose_b = false,
                                float alpha = 1.0f,
                                bool need_handle_matmul = true,
                                uint32_t tile_inner = 32);

}  // namespace webgpu
}  // namespace onnxruntime
