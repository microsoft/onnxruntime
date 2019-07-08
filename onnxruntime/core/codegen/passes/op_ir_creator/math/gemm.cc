// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/math/gemm.h"
#include "core/framework/op_kernel_info.h"

namespace onnxruntime {
namespace tvm_codegen {

// Evaluate of Gemm OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Gemm)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext& /*ctx_codegen*/,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

  tvm::Tensor A = inputs[0];
  tvm::Tensor B = inputs[1];
  tvm::Tensor C = inputs[2];

  int64_t trans_A, trans_B;
  ORT_RETURN_IF_ERROR(attrs.GetAttr<int64_t>("transA", &trans_A));
  ORT_RETURN_IF_ERROR(attrs.GetAttr<int64_t>("transB", &trans_B));

  float alpha, beta;
  ORT_ENFORCE(attrs.GetAttr<float>("alpha", &alpha).IsOK());
  ORT_ENFORCE(attrs.GetAttr<float>("beta", &beta).IsOK());

  tvm::Tensor Y = Gemm(A, B, C, trans_A != 0, trans_B != 0, alpha, beta, node.Name() + "_Gemm");
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
