// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/crop.h"
#include "core/framework/op_kernel_info.h"

namespace onnxruntime {
namespace tvm_codegen {

// Evaluate of Crop OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Crop)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext&,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

  if (inputs[0]->shape.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input is expected to have four dimensions corresponding to [N,C,H,W]");
  }

  std::vector<int64_t> border;
  std::vector<int64_t> scale;

  ORT_ENFORCE(attrs.GetAttrs<int64_t>("border", border).IsOK());
  // scale is optional and status is false when omit
  bool is_ok = attrs.GetAttrs<int64_t>("scale", scale).IsOK();
  ORT_UNUSED_PARAMETER(is_ok);

  if (border.size() != 4) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Attribute border needs to be specified with four border elements");
  }

  tvm::Tensor Y = Crop(inputs[0], ToTvmArray(border), ToTvmArray(scale), node.Name() + "_Crop");
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
