// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/math/matmul_ops.h"

namespace onnxruntime {
namespace tvm_codegen {

// Evaluate of MatMul OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(MatMul)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext&,
    tvm::Array<tvm::Tensor>& outputs) {
  tvm::Tensor Y = MatMul(inputs[0], inputs[1], node.Name() + "_MatMul");
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
