// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/program.h"

namespace onnxruntime::contrib::webgpu {

void GenerateQLoRADown(onnxruntime::webgpu::ShaderHelper& shader, const std::string& base_groups,
                       const std::string& x_value, bool acc_f32, bool use_subgroups = false);
void GenerateQLoRAUp(onnxruntime::webgpu::ShaderHelper& shader, bool acc_f32, bool vector_output = false);
Status ApplyQLoRAOutput(onnxruntime::webgpu::ComputeContext& context, const Tensor* base, const Tensor* t,
                        const Tensor* b, const Tensor* scales, Tensor* y);

}  // namespace onnxruntime::contrib::webgpu
