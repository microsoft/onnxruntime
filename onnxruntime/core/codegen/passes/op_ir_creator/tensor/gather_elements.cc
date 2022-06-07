// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/tensor/gather_elements.h"
#include "core/framework/op_kernel_info.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace tvm_codegen {

// Evaluate of GatherElements OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(GatherElements)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext&,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

  int64_t axis;
  ORT_ENFORCE(attrs.GetAttr<int64_t>("axis", &axis).IsOK());
  axis = HandleNegativeAxis(axis, gsl::narrow_cast<int64_t>(inputs[0]->shape.size()));

  tvm::Tensor Y = GatherElements(inputs[0], axis, inputs[1], node.Name() + "_GatherElements");
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
