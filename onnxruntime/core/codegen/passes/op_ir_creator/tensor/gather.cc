// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/tensor/gather.h"
#include "core/framework/op_kernel_info.h"

namespace onnxruntime {
namespace tvm_codegen {

// Evaluate of Gather OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Gather)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext&,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

  int64_t axis;
  ORT_ENFORCE(attrs.GetAttr<int64_t>("axis", &axis).IsOK());

  tvm::Tensor Y = Gather(inputs[0], axis, inputs[1], node.Name() + "_Gather");
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
