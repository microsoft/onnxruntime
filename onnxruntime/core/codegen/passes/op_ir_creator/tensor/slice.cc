// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"
#include "core/codegen/passes/utils/ort_tvm_utils.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/slice.h"
#include "core/framework/op_kernel_info.h"
#include "core/framework/tensorprotoutils.h"

#include <tvm/ir_pass.h>

namespace onnxruntime {
namespace tvm_codegen {

Status SliceCommon(const tvm::Array<tvm::Tensor>& inputs,
                   const Node& node,
                   tvm::Array<tvm::Tensor>& outputs,
                   const std::vector<int64_t>& starts,
                   const std::vector<int64_t>& ends,
                   const std::vector<int64_t>& axes1,
                   const std::vector<int64_t>& steps1) {
  ORT_RETURN_IF_NOT(nullptr != node.InputDefs()[0], "nullptr == node.InputDefs()[0]");

  std::vector<int64_t> axes;
  if (axes1.size() == 0) {
    for (size_t i = 0; i < starts.size(); ++i) {
      axes.push_back(gsl::narrow_cast<int64_t>(i));
    }
  } else {
    axes = axes1;
  }

  std::vector<int64_t> steps;
  if (steps1.size() == 0) {
    steps.resize(starts.size(), 1);
  } else {
    steps = steps1;
  }

  tvm::Tensor Y = Slice(inputs[0], starts, ends, axes, steps, node.Name() + "_Slice");
  outputs.push_back(Y);
  return Status::OK();
}

// Evaluate of Slice OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Slice)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);

  // NOTE that in opset 10, Slice has changed starts/ends/axes from attribute to input
  // which may lead to dynamic output shape.
  int version = ctx_codegen.GetCodeGenHandle()->domain_version_lookup_func(node.Domain());
  ORT_RETURN_IF_NOT(version <= 9, "Dynamic Slice is not supported yet");

  std::vector<int64_t> starts, ends, steps;
  ORT_RETURN_IF_ERROR(info.GetAttrs<int64_t>("starts", starts));
  ORT_RETURN_IF_ERROR(info.GetAttrs<int64_t>("ends", ends));
  ORT_RETURN_IF_NOT(starts.size() == ends.size(), "starts.size() != ends.size()");

  auto axes = info.GetAttrsOrDefault<int64_t>("axes");

  return SliceCommon(inputs, node, outputs, starts, ends, axes, steps);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
