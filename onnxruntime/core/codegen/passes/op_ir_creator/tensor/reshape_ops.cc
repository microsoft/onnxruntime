// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/reshape_ops.h"
#include "core/codegen/passes/utils/ort_tvm_utils.h"
#include "core/framework/op_kernel_info.h"

namespace onnxruntime {
namespace tvm_codegen {

// Evaluate of Dropout OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Dropout)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  tvm::Tensor Y = Identity(inputs[0]);
  outputs.push_back(Y);

  // optional mask
  // Support skipped trailing outputs
  if (node.OutputDefs().size() > 1 && node.OutputDefs()[1]->Exists()) {
    // A fake mask with all ones
    auto l = [&](const tvm::Array<tvm::Var>& /*indices*/) {
      return tvm::make_const(tvm::UInt(8), 1);
    };
    tvm::Tensor mask = tvm::compute(inputs[0]->shape, l, "mask");
    outputs.push_back(mask);
  }

  return Status::OK();
}

// Evaluate of Flatten OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Flatten)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext&,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

  int64_t axis;
  ORT_RETURN_IF_ERROR(attrs.GetAttr<int64_t>("axis", &axis));

  tvm::Tensor Y = Flatten(inputs[0], axis, node.Name() + "_Flatten");
  outputs.push_back(Y);
  return Status::OK();
}

// Evaluate of Identity OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Identity)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node&,
    CodeGenContext&,
    tvm::Array<tvm::Tensor>& outputs) {
  tvm::Tensor Y = Identity(inputs[0]);
  outputs.push_back(Y);
  return Status::OK();
}

// Evaluate of Reshape OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Reshape)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  tvm::Tensor Y = Reshape(inputs[0], ShapeToTvmArray(node.OutputDefs()[0], ctx_codegen), node.Name() + "_Reshape");
  outputs.push_back(Y);
  return Status::OK();
}

// Evaluate of Squeeze OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Squeeze)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  tvm::Tensor Y = Reshape(inputs[0], ShapeToTvmArray(node.OutputDefs()[0], ctx_codegen), node.Name() + "_Squeeze");
  outputs.push_back(Y);
  return Status::OK();
}

// Evaluate of Unsqueeze OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Unsqueeze)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  tvm::Tensor Y = Reshape(inputs[0], ShapeToTvmArray(node.OutputDefs()[0], ctx_codegen), node.Name() + "_Unsqueeze");
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
