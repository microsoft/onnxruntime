// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <tvm/tvm.h>

#include "core/codegen/passes/utils/codegen_context.h"

namespace onnxruntime {
namespace nuphar {
namespace internal {

tvm::Tensor SoftmaxInternal(const tvm::Tensor& input, int64_t axis, tvm_codegen::CodeGenContext& ctx_codegen, const std::string& name, bool logarithmic);

}  // namespace internal
}  // namespace nuphar
}  // namespace onnxruntime
