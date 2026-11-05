// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"

namespace onnxruntime::webgpu {
class ShaderVariableHelper;
}

namespace onnxruntime::contrib::webgpu {

// These C++ generation helpers append reusable WGSL fragments to the caller's
// shader. They do not create standalone GPU programs or dispatch GPU work.
//
// Append the rank-32 down-projection body to the generic single-vector decode
// shader. The base projection and T = X * A_down run in the same dispatch using
// separate workgroups. X uses vec4 loads; uniforms.num_N_tile marks the four
// appended adapter workgroups.
Status GenerateQLoRADown(onnxruntime::webgpu::ShaderHelper& shader, const onnxruntime::webgpu::ShaderVariableHelper& x,
                         const onnxruntime::webgpu::ShaderVariableHelper& lora_at,
                         const onnxruntime::webgpu::ShaderVariableHelper& lora_a_scales,
                         const onnxruntime::webgpu::ShaderVariableHelper& lora_down,
                         bool acc_f32, bool use_subgroups = false);

// Append WGSL helper functions for up-weight dequantization and delta = T * B_up.
// Generic and DP4A fused prefill shaders and the separate decode output shader
// reuse these functions. Subgroup-matrix shaders reuse only the dequantization
// helper and implement the multiplication with their own matrix instructions.
Status GenerateQLoRAUp(onnxruntime::webgpu::ShaderHelper& shader,
                       const onnxruntime::webgpu::ShaderVariableHelper& lora_t,
                       const onnxruntime::webgpu::ShaderVariableHelper& lora_b,
                       const onnxruntime::webgpu::ShaderVariableHelper& lora_b_scales,
                       bool acc_f32, bool vector_output = false);

// Dispatch the decode output shader to compute Y = base + T * B_up.
Status ApplyQLoRAOutput(onnxruntime::webgpu::ComputeContext& context, const Tensor* base, const Tensor* t,
                        const Tensor* b, const Tensor* scales, Tensor* y);

}  // namespace onnxruntime::contrib::webgpu
