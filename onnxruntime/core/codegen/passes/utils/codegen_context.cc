// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/utils/codegen_context.h"

#include "core/codegen/common/common.h"

namespace onnxruntime {
namespace tvm_codegen {

CodeGenContext::CodeGenContext(
    const codegen::CodeGenHandle* handle)
    : handle_(handle), unname_symbol_counter_(0) {}

tvm::Var CodeGenContext::GetOrCreateDynamicDim(const std::string& name) {
  if (dynamic_dims_.count(name) == 0)
    dynamic_dims_.emplace(name, tvm::Var(name));

  return dynamic_dims_.at(name);
}

std::string CodeGenContext::CreateUnnamedSymbol() {
  return "unnamed_" + std::to_string(unname_symbol_counter_++);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
