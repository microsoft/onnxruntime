// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/target/generic/op_ir_creator/all_ops.h"

#include "core/codegen/mti/math/unary_ops.h"
#include "core/framework/op_kernel_info.h"

namespace onnxruntime {
namespace tvm_codegen {

// Evaluate of Clip OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Clip)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext&,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);

  float max_value, min_value;
  ORT_RETURN_IF_ERROR(info.GetAttr<float>("max", &max_value));
  ORT_RETURN_IF_ERROR(info.GetAttr<float>("min", &min_value));

  tvm::Tensor Y = Clip(inputs[0], min_value, max_value, node.Name() + "_Clip");
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
