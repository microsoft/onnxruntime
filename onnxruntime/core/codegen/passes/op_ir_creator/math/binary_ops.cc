// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/common/op_macro.h"
#include "core/codegen/mti/math/binary_ops.h"
#include "core/codegen/mti/tensor/cast_ops.h"

namespace onnxruntime {
namespace tvm_codegen {

// helper local macro defines Evaluate of BINARY_OP OpIRCreators
#define BINARY_OP(name)                                      \
  Status GENERIC_OP_IR_CREATOR_CLASS(name)::Evaluate(        \
      const tvm::Array<tvm::Tensor>& inputs,                 \
      const Node& node,                                      \
      CodeGenContext&,                                       \
      tvm::Array<tvm::Tensor>& outputs) {                    \
    tvm::Tensor Y = name(inputs[0], inputs[1], node.Name()); \
    outputs.push_back(Y);                                    \
    return Status::OK();                                     \
  }

LIST_BINARY_OPS()

#undef BINARY_OP

// helper local macro defines Evaluate of BINARY_CMP_OP OpIRCreators
#define BINARY_CMP_OP(name)                                                                               \
  Status GENERIC_OP_IR_CREATOR_CLASS(name)::Evaluate(                                                     \
      const tvm::Array<tvm::Tensor>& inputs,                                                              \
      const Node& node,                                                                                   \
      CodeGenContext&,                                                                                    \
      tvm::Array<tvm::Tensor>& outputs) {                                                                 \
    tvm::Tensor Y = Cast(name(inputs[0], inputs[1], node.Name()), HalideIR::UInt(8), "cast_bool_" #name); \
    outputs.push_back(Y);                                                                                 \
    return Status::OK();                                                                                  \
  }

LIST_BINARY_CMP_OPS()

#undef BINARY_CMP_OP

}  // namespace tvm_codegen
}  // namespace onnxruntime
