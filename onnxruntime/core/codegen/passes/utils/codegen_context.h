// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/common/handle.h"
#include "core/codegen/common/common.h"
#include "core/common/common.h"
#include "core/framework/data_types.h"
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

// CodeGenContext is a data structure involving across passes
// Compiler developers can use it to store meta data
// to support fine-grained control of code generation
class CodeGenContext {
 public:
  CodeGenContext(const codegen::CodeGenHandle* handle);

  virtual ~CodeGenContext() = default;

  // returns tvm::Var for the dynamic dim
  tvm::Var GetOrCreateDynamicDim(const std::string& name);

  const codegen::CodeGenHandle* GetCodeGenHandle() const {
    return handle_;
  }

  std::string CreateUnnamedSymbol();

 protected:
  std::unordered_map<std::string, tvm::Var> dynamic_dims_;

  const codegen::CodeGenHandle* handle_;

  int unname_symbol_counter_;
};

// Add Promote for CodeGenContext
DYNAMIC_PROMOTE(CodeGenContext)

}  // namespace tvm_codegen
}  // namespace onnxruntime
