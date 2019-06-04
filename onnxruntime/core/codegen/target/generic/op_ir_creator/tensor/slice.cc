// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/target/generic/op_ir_creator/all_ops.h"
#include "core/codegen/target/ort_tvm_utils.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/slice.h"
#include "core/framework/op_kernel_info.h"

#include <tvm/ir_pass.h>

namespace onnxruntime {
namespace tvm_codegen {

// local constexpr for INT_MAX
constexpr int64_t max_range = INT_MAX;

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

  std::vector<int64_t> starts, ends;
  ORT_RETURN_IF_ERROR(info.GetAttrs<int64_t>("starts", starts));
  ORT_RETURN_IF_ERROR(info.GetAttrs<int64_t>("ends", ends));
  ORT_RETURN_IF_NOT(starts.size() == ends.size());

  auto axes = info.GetAttrsOrDefault<int64_t>("axes");
  if (axes.size() == 0) {
    for (size_t i = 0; i < starts.size(); ++i) {
      axes.push_back(gsl::narrow_cast<int64_t>(i));
    }
  }

  ORT_RETURN_IF_NOT(nullptr != node.InputDefs()[0]);
  const ONNX_NAMESPACE::TensorShapeProto* shape_proto = node.InputDefs()[0]->Shape();

  tvm::Array<tvm::Integer> tvm_starts, tvm_ends;
  bool empty = false;

  for (int dim = 0; dim < shape_proto->dim_size(); ++dim) {
    auto axes_iter = std::find(axes.begin(), axes.end(), dim);
    const ONNX_NAMESPACE::TensorShapeProto_Dimension& proto_dim = shape_proto->dim(dim);
    bool found_in_axes = (axes_iter != axes.end());
    if (!found_in_axes) {
      tvm_starts.push_back(0);
      if (proto_dim.has_dim_value()) {
        tvm_ends.push_back(proto_dim.dim_value());
      } else {
        tvm_ends.push_back(max_range);
      }
    } else {
      auto axes_index = axes_iter - axes.begin();
      int64_t start = starts[axes_index];
      int64_t end = ends[axes_index];
      if (proto_dim.has_dim_value()) {
        int64_t dim_max = proto_dim.dim_value();
        if (start < 0) start += dim_max;
        if (end < 0) end += dim_max;
        start = std::min(dim_max, std::max(static_cast<int64_t>(0), start));
        end = std::min(dim_max, std::max(start, end));
      }
      tvm_starts.push_back(start);
      tvm_ends.push_back(end);
      empty = empty || (start == end);
    }
  }

  tvm::Tensor Y;
  if (empty) {
    tvm::Array<tvm::Expr> shape;
    for (size_t dim = 0; dim < gsl::narrow_cast<size_t>(shape_proto->dim_size()); ++dim) {
      shape.push_back(tvm::ir::Simplify(tvm_ends[dim] - tvm_starts[dim]));
    }
    Y = MakeZeroTensor(shape, inputs[0]->dtype, node.Name() + "_zeros");
  } else {
    Y = Slice(inputs[0], tvm_starts, tvm_ends, node.Name() + "_Slice");
  }

  outputs.push_back(Y);
  return Status::OK();
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
