// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"

#include "core/providers/nuphar/mti_x86/math/logsoftmax.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace nuphar {

// Evaluate of LogSoftmax OpIRCreator
Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(LogSoftmax)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext&,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);

  int64_t axis_i64;
  ORT_RETURN_IF_ERROR(info.GetAttr<int64_t>("axis", &axis_i64));

  axis_i64 = HandleNegativeAxis(axis_i64, gsl::narrow_cast<int64_t>(inputs[0]->shape.size()));
  tvm::Tensor Y = nuphar::LogSoftmax(inputs[0], axis_i64);
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace nuphar
}  // namespace onnxruntime
