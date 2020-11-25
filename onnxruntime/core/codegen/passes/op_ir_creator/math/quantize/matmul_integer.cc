// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/math/binary_ops.h"
#include "core/codegen/mti/math/matmul_ops.h"
#include "core/codegen/mti/tensor/cast_ops.h"

namespace onnxruntime {
namespace tvm_codegen {

// Evaluate of MatMulInteger OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(MatMulInteger)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  const auto& A = inputs[0];
  const auto& B = inputs[1];
  auto& name = node.Name();

  // A generic path, cast to int32
  // Support skipped trailing inputs
  auto A_Int32 = (node.InputDefs().size() >= 3 && node.InputDefs()[2]->Exists())
                     ? Sub(Cast(A, HalideIR::Int(32)), Cast(inputs[2], HalideIR::Int(32)))
                     : Cast(A, HalideIR::Int(32));
  auto B_Int32 = (node.InputDefs().size() >= 4 && node.InputDefs()[3]->Exists())
                     ? Sub(Cast(B, HalideIR::Int(32)), Cast(inputs[3], HalideIR::Int(32)))
                     : Cast(B, HalideIR::Int(32));
  tvm::Tensor Y = MatMul(A_Int32, B_Int32, name + "_MatMulInteger");
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
