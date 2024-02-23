// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/passes/op_ir_creator/all_ops.h"

#include "core/codegen/mti/nn/conv_ops.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/concat_ops.h"
#include "core/codegen/mti/tensor/split.h"
#include "core/codegen/passes/utils/ort_tvm_utils.h"
#include "core/framework/op_kernel_info.h"

namespace onnxruntime {
namespace tvm_codegen {

Status GENERIC_OP_IR_CREATOR_CLASS(Conv)::Evaluate(
    const tvm::Array<tvm::Tensor>& inputs,
    const Node& node,
    CodeGenContext& ctx_codegen,
    tvm::Array<tvm::Tensor>& outputs) {
  ProtoHelperNodeContext ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> info(&ctx);

  // Attributes
  int64_t group;
  std::string auto_pad;
  std::vector<int64_t> kernel_shape, strides, dilations, pads;

  info.GetAttrOrDefault<int64_t>("group", &group, 1);
  info.GetAttrOrDefault<std::string>("auto_pad", &auto_pad, "NOTSET");

  ORT_THROW_IF_ERROR(info.GetAttrs<int64_t>("kernel_shape", kernel_shape));
  ORT_ENFORCE(kernel_shape.size() <= 2, "Only support 1D/2D convolution currently!");
  ORT_THROW_IF_ERROR(info.GetAttrs<int64_t>("strides", strides));

  dilations = info.GetAttrs<int64_t>("dilations", dilations).IsOK() ? dilations : std::vector<int64_t>(kernel_shape.size(), 1);
  ORT_ENFORCE(dilations == std::vector<int64_t>(kernel_shape.size(), 1), "Only support dilation is 1 currently");

  pads = info.GetAttrs<int64_t>("pads", pads).IsOK() ? pads : std::vector<int64_t>(kernel_shape.size() * 2, 0);

  // auto_pad
  if (auto_pad != "NOTSET") {
    auto rank = inputs[0]->shape.size() - 2;
    ORT_ENFORCE(rank > 0);
    for (uint64_t i = 0; i < rank; i++) {
      if (auto_pad == "VALID") {
        pads[i] = 0;
        pads[i + rank] = 0;
      } else if (auto_pad == "SAME_UPPER" || auto_pad == "SAME_LOWER") {
        // TODO: handle symbolic dim
        ORT_ENFORCE(ShapeHasValue(node.InputDefs()[0], 2 + i));

        int64_t input_dim_value = ShapeValue(node.InputDefs()[0], 2 + i);
        int64_t output_dim_value = (input_dim_value + strides[i] - 1) / strides[i];
        int64_t pad_needed = (output_dim_value - 1) * strides[i] + kernel_shape[i] - input_dim_value;

        pads[i] = auto_pad == "SAME_LOWER" ? (pad_needed + 1) / 2 : pad_needed / 2;
        pads[i + rank] = pad_needed - pads[i];
      } else {
        ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unknown auto_pad value");
      }
    }
  }

  // Inputs
  tvm::Tensor X = inputs[0];
  tvm::Tensor W = inputs[1];
  // Outputs
  tvm::Tensor Y;
  tvm::Array<tvm::Expr> Y_shape = ShapeToTvmArray(node.OutputDefs()[0], ctx_codegen);

  // 1-D convolution
  if (kernel_shape.size() == 1) {
    Y = Conv1D(X, W, Y_shape, ToTvmArray(strides), ToTvmArray(pads), node.Name() + "_Conv1D");
  }
  // 2-D convolution
  else if (kernel_shape.size() == 2) {
    if (group == 1) {
      Y = Conv2D(X, W, Y_shape, ToTvmArray(strides), ToTvmArray(pads), node.Name() + "_Conv2D");
    } else {
      int64_t channel_out = ShapeValue(node.InputDefs()[1], 0);
      int64_t channel_in = ShapeValue(node.InputDefs()[1], 1);
      ORT_ENFORCE(channel_out % group == 0);

      int64_t cout_group = channel_out / group;
      Y_shape.Set(1, Y_shape[1] / gsl::narrow_cast<int>(group));

      tvm::Array<tvm::Integer> split_index0;
      tvm::Array<tvm::Integer> split_index1;

      for (int i = 1; i < group; i++) {
        split_index0.push_back(i * channel_in);
        split_index1.push_back(i * cout_group);
      }

      auto input_groups = SplitWithIndices(X, split_index0, 1);
      auto weight_groups = SplitWithIndices(W, split_index1, 0);

      // FIXME: This will trigger a llvm buffer overflow when group is too large
      // TODO: fix this change it to batched gemm/conv
      tvm::Array<tvm::Tensor> output_tensors;
      for (int i = 0; i < group; i++) {
        auto output_tensor = Conv2D(input_groups[i],
                                    weight_groups[i],
                                    Y_shape,
                                    ToTvmArray(strides),
                                    ToTvmArray(pads),
                                    node.Name() + "_Conv2D");
        output_tensors.push_back(output_tensor);
      }
      Y = Concat(output_tensors, 1);
    }
  }

  // Add bias if provided
  // Support skipped trailing inputs
  if (node.InputDefs().size() > 2 && node.InputDefs()[2]->Exists()) {
    tvm::Tensor B = inputs[2];
    Y = tvm::compute(
        Y_shape,
        [&](const tvm::Array<tvm::Var>& indices) {
          return Y(indices) + B(indices[1]);
        });
  }

  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
