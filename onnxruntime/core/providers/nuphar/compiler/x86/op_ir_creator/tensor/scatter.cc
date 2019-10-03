// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/op_ir_creator/all_ops.h"

#include "core/framework/op_kernel_info.h"
#include "core/providers/nuphar/mti_x86/tensor/scatter.h"

namespace onnxruntime {
namespace nuphar {

static Status ScatterCommon(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext&,
    tvm::Array<tvm::Tensor>& outputs,
    const std::string& name) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

  // The default value of optional attribute axis is 0
  int64_t axis = attrs.GetAttrOrDefault<int64_t>("axis", 0);

  tvm::Tensor Y = Scatter(inputs[0], axis, inputs[1], inputs[2], name);
  outputs.push_back(Y);
  return Status::OK();
}

// Evaluate of Scatter OpIRCreator
Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(Scatter)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& codegen_ctx,
    tvm::Array<tvm::Tensor>& outputs) {
  return ScatterCommon(inputs, node, codegen_ctx, outputs, node.Name() + "_ScatterElements");
}

// Evaluate of ScatterElements OpIRCreator
Status NUPHAR_TVM_X86_OP_IR_CREATOR_CLASS(ScatterElements)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    tvm_codegen::CodeGenContext& codegen_ctx,
    tvm::Array<tvm::Tensor>& outputs) {
  return ScatterCommon(inputs, node, codegen_ctx, outputs, node.Name() + "_Scatter");
}

}  // namespace nuphar
}  // namespace onnxruntime
