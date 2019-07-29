// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/math/softmax.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace tvm_codegen {

// Evaluate of Softmax OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Softmax)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);

  int64_t axis_i64;
  ORT_RETURN_IF_ERROR(info.GetAttr<int64_t>("axis", &axis_i64));

  axis_i64 = HandleNegativeAxis(axis_i64, gsl::narrow_cast<int64_t>(inputs[0]->shape.size()));
  tvm::Tensor Y = Softmax(inputs[0], axis_i64, node.Name() + "_Softmax");
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
