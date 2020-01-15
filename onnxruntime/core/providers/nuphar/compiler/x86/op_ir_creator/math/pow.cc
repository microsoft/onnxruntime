// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel_info.h"
#include "core/providers/nuphar/common/nuphar_tvm_utils.h"
#include "core/providers/nuphar/compiler/nuphar_codegen_ctx.h"
#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"
#include "core/providers/nuphar/mti_x86/math/pow.h"

namespace onnxruntime {
namespace nuphar {

Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(Pow)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  ORT_ENFORCE(inputs.size() == 2);

  struct {
    tvm::Expr expr;
    bool is_scalar;
  } constant_scalars[2];

  for (size_t i = 0; i < 2; ++i) {
    ProtoHelperNodeContext ctx(node);
    OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);
    NupharCodeGenCtx* ctx_nuphar = Promote<NupharCodeGenCtx>(&ctx_codegen);

    ORT_ENFORCE(i < node.InputDefs().size());
    const auto* tensor = ctx_nuphar->GetOrtInitializerTensor(node.InputDefs()[i]->Name());
    constant_scalars[i].is_scalar = TryCreateConstantScalar(constant_scalars[i].expr, tensor);
  }
  tvm::Tensor Y;
  if (constant_scalars[0].is_scalar)
    Y = Pow(constant_scalars[0].expr, inputs[1], node.Name());
  else if (constant_scalars[1].is_scalar)
    Y = Pow(inputs[0], constant_scalars[1].expr, node.Name());
  else
    Y = Pow(inputs[0], inputs[1], node.Name());
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace nuphar
}  // namespace onnxruntime
