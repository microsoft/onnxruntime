// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace ml {

Status GetVectorAttrsOrDefault(const OpKernelInfo& info, const std::string& name, std::vector<double>& data);
Status GetVectorAttrsOrDefault(const OpKernelInfo& info, const std::string& name, std::vector<float>& data);

}  // namespace ml
}  // namespace onnxruntime
