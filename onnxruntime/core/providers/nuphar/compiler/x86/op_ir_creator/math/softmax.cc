// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"

#include "core/providers/nuphar/mti_x86/math/softmax.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace tvm_codegen {

// Evaluate of Softmax OpIRCreator
Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(Softmax)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext&,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);

  int64_t axis_i64;
  ORT_RETURN_IF_ERROR(info.GetAttr<int64_t>("axis", &axis_i64));

  axis_i64 = HandleNegativeAxis(axis_i64, gsl::narrow_cast<int64_t>(inputs[0]->shape.size()));
  tvm::Tensor Y = nuphar_codegen::Softmax(inputs[0], axis_i64);
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
