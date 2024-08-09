// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/pad_ops.h"
#include "core/framework/op_kernel_info.h"

namespace onnxruntime {
namespace tvm_codegen {

// Evaluate of Pad OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Pad)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext&,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);

  std::string mode;
  std::vector<int64_t> pads;
  float value;

  ORT_THROW_IF_ERROR(attrs.GetAttr<std::string>("mode", &mode));
  ORT_THROW_IF_ERROR(attrs.GetAttrs<int64_t>("pads", pads));
  ORT_THROW_IF_ERROR(attrs.GetAttr<float>("value", &value));

  if (mode != "constant" && mode != "edge" && mode != "reflect")
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Pad: Unsupported padding mode!");

  if (pads.size() != 2 * inputs[0]->shape.size())
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Pad: pads rank does not match inputs rank!");

  std::vector<int64_t> pad_before, pad_after;
  size_t offset = pads.size() / 2;
  for (size_t i = 0; i < offset; i++) {
    pad_before.push_back(pads[i]);
    pad_after.push_back(pads[i + offset]);
  }

  tvm::Tensor Y = Pad(inputs[0], ToTvmArray(pad_before), ToTvmArray(pad_after), value, mode, node.Name() + "_Pad");
  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
