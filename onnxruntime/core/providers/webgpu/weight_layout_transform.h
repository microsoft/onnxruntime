// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/tensor.h"
#include <string>

namespace onnxruntime {
namespace webgpu {

class ComputeContext;
class WeightLayoutTransformCache;

// Transform weight tensor to specified format
// Returns the transformed tensor (either from cache or newly created)
Status TransformWeightLayout(
    ComputeContext& context,
    const Tensor* weight,
    const std::string& weight_name,
    const std::string& format_descriptor,
    WeightLayoutTransformCache& cache,
    /*out*/ const Tensor*& transformed_weight);

}  // namespace webgpu
}  // namespace onnxruntime
