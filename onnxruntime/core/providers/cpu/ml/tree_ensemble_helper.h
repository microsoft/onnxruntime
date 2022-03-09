// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace ml {

std::vector<double> GetVectorAttrsOrDefault(const OpKernelInfo& info, const std::string& name, const std::vector<double>& default_value);
std::vector<float> GetVectorAttrsOrDefault(const OpKernelInfo& info, const std::string& name, const std::vector<float>& default_value);

}  // namespace ml
}  // namespace onnxruntime
