// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"

#include "core/framework/op_kernel_info.h"
#include "core/providers/common.h"
#include "core/providers/nuphar/compiler/x86/x86_target_info.h"
#include "core/providers/nuphar/mti_x86/math/logsoftmax.h"

namespace onnxruntime {
namespace nuphar {

// Evaluate of LogSoftmax OpIRCreator
Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(LogSoftmax)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);

  int64_t axis_i64;
  ORT_RETURN_IF_ERROR(info.GetAttr<int64_t>("axis", &axis_i64));
  axis_i64 = HandleNegativeAxis(axis_i64, gsl::narrow_cast<int64_t>(inputs[0]->shape.size()));

  CodeGenTargetX86* target = dynamic_cast<CodeGenTargetX86*>(ctx_codegen.GetCodeGenHandle()->codegen_target);
  ORT_ENFORCE(target != nullptr);
  int64_t natural_vector_size = target->NaturalVectorWidth(inputs[0]->dtype.bits());

  tvm::Tensor Y = nuphar::LogSoftmax(inputs[0], axis_i64, natural_vector_size);
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace nuphar
}  // namespace onnxruntime
