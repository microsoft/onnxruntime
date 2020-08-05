// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/codegen/mti/math/binary_ops.h"
#include "core/codegen/mti/math/gemm.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/common.h"
#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"
#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"
#include "core/providers/nuphar/mti_x86/math/matmul_ops.h"

namespace onnxruntime {
namespace nuphar {

Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(Gemm)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);

  tvm::Tensor Y;
  auto& A = inputs[0];
  auto& B = inputs[1];
  tvm::Tensor C;

  int64_t trans_a, trans_b;
  float alpha, beta;
  ORT_RETURN_IF_ERROR(info.GetAttr<int64_t>("transA", &trans_a));
  ORT_RETURN_IF_ERROR(info.GetAttr<int64_t>("transB", &trans_b));
  ORT_RETURN_IF_ERROR(info.GetAttr<float>("alpha", &alpha));
  ORT_RETURN_IF_ERROR(info.GetAttr<float>("beta", &beta));

  // bias is optional
  if (inputs.size() < 3) {
    beta = 0;
    C = tvm_codegen::MakeZeroTensor({1}, A->dtype, node.Name() + "_zero");
  } else {
    C = inputs[2];
  }

  // use native sgemm for floating point
  if (A->dtype == HalideIR::Float(32) &&
      B->dtype == HalideIR::Float(32) &&
      GemmExternCpu(A, B, Y, !!trans_a, !!trans_b, node.Name() + "_gemm")) {
    if (beta != 0) {
      tvm::Tensor beta_bias = (beta == 1) ? C : tvm_codegen::Mul(tvm::make_const(tvm::Float(32), beta), C);
      Y = tvm_codegen::Add((alpha == 1) ? Y : tvm_codegen::Mul(tvm::make_const(tvm::Float(32), alpha), Y),
                           beta_bias, node.Name() + "_add_bias");
    } else {
      Y = (alpha == 1) ? Y : tvm_codegen::Mul(tvm::make_const(tvm::Float(32), alpha), Y);
    }
    outputs.push_back(Y);
    return Status::OK();
  }

  // fallback to default MTI ops
  Y = tvm_codegen::Gemm(A, B, C, trans_a, trans_b, alpha, beta, node.Name());
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace nuphar
}  // namespace onnxruntime
