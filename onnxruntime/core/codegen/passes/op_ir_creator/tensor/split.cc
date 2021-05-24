// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/split.h"
#include "core/framework/op_kernel_info.h"

namespace onnxruntime {
namespace tvm_codegen {

// Evaluate of Split OpIRCreator
Status GENERIC_OP_IR_CREATOR_CLASS(Split)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext&,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);

  int64_t axis;
  ORT_RETURN_IF_ERROR(info.GetAttr<int64_t>("axis", &axis));
  axis = HandleNegativeAxis(axis, gsl::narrow_cast<int64_t>(inputs[0]->shape.size()));
  std::vector<int64_t> split_sizes;

  int64_t split_size_sum = 0;
  if (info.GetAttrs("split", split_sizes).IsOK()) {
    // optional
    split_size_sum = std::accumulate(split_sizes.cbegin(), split_sizes.cend(), 0LL);
    ORT_RETURN_IF_NOT(std::all_of(split_sizes.cbegin(), split_sizes.cend(), [](int64_t value) { return value > 0; }),
                      "Invalid value in 'split' attribute. All values must be > 0");

    // check split sizes
    for (size_t i = 0; i < node.OutputDefs().size(); ++i) {
      ORT_RETURN_IF_NOT(split_sizes[i] == ShapeValue(node.OutputDefs()[i], gsl::narrow<int>(axis)),
                        "split_sizes[i] != ShapeValue(node.OutputDefs()[i], axis)");
    }

  } else {
    for (size_t i = 0; i < node.OutputDefs().size(); ++i) {
      split_sizes.push_back(ShapeValue(node.OutputDefs()[i], gsl::narrow<int>(axis)));
      split_size_sum += split_sizes[i];
    }
  }

  // check total size
  if (ShapeHasValue(node.InputDefs()[0], axis)) {
    int64_t input_axis_dim = ShapeValue(node.InputDefs()[0], axis);
    if (split_size_sum != input_axis_dim) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Cannot split using values in 'split' attribute. Axis=", axis,
                             " Dim being splitted=", input_axis_dim,
                             " Sum of sizes in 'split' (must equal size of selected axis) was ", split_size_sum);
    }
  }

  tvm::Array<tvm::Tensor> output_tensors = Split(inputs[0], ToTvmArray(split_sizes), axis, node.Name() + "_Split");
  for (size_t i = 0; i < node.OutputDefs().size(); ++i) {
    outputs.push_back(output_tensors[i]);
  }
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
