// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/math/binary_ops.h"
#include "core/codegen/mti/tensor/reshape_ops.h"
#include "core/framework/op_kernel_info.h"

namespace onnxruntime {
namespace tvm_codegen {

tvm::Tensor Sum(const tvm::Tensor& lhs, const tvm::Tensor& rhs, const std::string& name) {
  return Add(lhs, rhs, name);
}

// helper local macro defines Evaluate of BINARY_OP OpIRCreators
#define VARIADIC_OP(name)                                      \
  Status GENERIC_OP_IR_CREATOR_CLASS(name)::Evaluate(          \
      const tvm::Array<tvm::Tensor>& inputs,                   \
      const Node& node,                                        \
      CodeGenContext&,                                         \
      tvm::Array<tvm::Tensor>& outputs) {                      \
    tvm::Tensor Y = Identity(inputs[0], node.Name() + "0");    \
    for (size_t i = 1; i < inputs.size(); ++i)                 \
      Y = name(Y, inputs[i], node.Name() + std::to_string(i)); \
    outputs.push_back(Y);                                      \
    return Status::OK();                                       \
  }

LIST_VARIADIC_OPS()

#undef VARIADIC_OP

}  // namespace tvm_codegen
}  // namespace onnxruntime
