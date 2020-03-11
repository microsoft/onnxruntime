// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <string>

#include "gradient_builder_base.h"

namespace onnxruntime {
namespace training {

void ComputeBroadcastBackwardAxes(
    const std::vector<Dimension>& A_dims,
    const std::vector<Dimension>& B_dims,
    std::vector<int64_t>* A_axes,
    std::vector<int64_t>* B_axes) {
  if (A_axes) A_axes->clear();
  if (B_axes) B_axes->clear();

  int ndim = int(std::max(A_dims.size(), B_dims.size()));
  int i = int(A_dims.size() - 1);
  int j = int(B_dims.size() - 1);
  int k = ndim - 1;

  for (; i >= 0 && j >= 0; --k) {
    if (A_dims[i].has_dim_value() && B_dims[j].has_dim_value()) {
      auto A_dim = A_dims[i].dim_value(),
           B_dim = B_dims[j].dim_value();

      if (A_dim != B_dim) {
        if (A_axes && A_dim == 1) {
          A_axes->push_back(gsl::narrow_cast<int64_t>(k));
        }
        if (B_axes && B_dim == 1) {
          B_axes->push_back(gsl::narrow_cast<int64_t>(k));
        }
      }
    } else if (A_dims[i].has_dim_param() && B_dims[j].has_dim_param()) {
      auto A_dim = A_dims[i].dim_param(),
           B_dim = B_dims[j].dim_param();
      if (A_dim != B_dim) {
        ORT_THROW("Error");
      }
    }
    // TODO : complete othere cases

    --i;
    --j;
  }

  if (A_axes && i < 0) {
    for (; k >= 0; --k) {
      A_axes->push_back(gsl::narrow_cast<int64_t>(k));
    }

  } else if (B_axes) {
    for (; k >= 0; --k) {
      B_axes->push_back(gsl::narrow_cast<int64_t>(k));
    }
  }
}

std::vector<Dimension> GetShape(const ArgDef& arg_def) {
  std::vector<Dimension> shape;
  const auto& dims = arg_def.type_proto->tensor_type().shape().dim();
  for (auto dim = dims.begin(); dim < dims.end(); dim++) {
    shape.push_back(*dim);
  }
  return shape;
}

void GradientBuilderBase::HandleBroadcasting(const ArgDef& input_grad,
                                             const ArgDef& target,
                                             const ArgDef& output_grad,
                                             const std::vector<int64_t>& reduce_axes,
                                             std::vector<NodeDef>& output) const {
  std::unordered_set<size_t> reduce_axes_set(reduce_axes.begin(), reduce_axes.end());
  std::vector<Dimension> reduced_shape;
  auto input_grad_shape = GetShape(input_grad);
  auto target_shape = GetShape(target);

  bool keep_dims = (input_grad_shape.size() == target_shape.size());

  Dimension dim_one;
  dim_one.set_dim_value(int64_t(1));

  for (size_t i = 0; i < input_grad_shape.size(); ++i) {
    if (reduce_axes_set.count(i) > 0) {
      if (keep_dims) {
        reduced_shape.push_back(dim_one);
      }
    } else {
      reduced_shape.push_back(input_grad_shape[i]);
    }
  }

  bool skip_reshape = true;
  if (target_shape.size() != reduced_shape.size()) {
    skip_reshape = false;
  } else {
    for (size_t i = 0; i < target_shape.size(); ++i) {
      if (target_shape[i].dim_value() != reduced_shape[i].dim_value() ||
          target_shape[i].dim_param() != reduced_shape[i].dim_param()) {
        skip_reshape = false;
        break;
      }
    }
  }

  if (skip_reshape) {
    output.push_back(
        NodeDef("ReduceSum",
                {input_grad},
                {output_grad},
                {{"keepdims", ONNX_NAMESPACE::MakeAttribute("keepdims", static_cast<int64_t>(keep_dims))},
                 {"axes", ONNX_NAMESPACE::MakeAttribute("axes", reduce_axes)}}));
  } else {
    ArgDef reduce_grad_arg = IA("ReduceSum_" + input_grad.name + "_for_" + target.name);
    output.push_back(
        NodeDef("ReduceSum",
                {input_grad},
                {reduce_grad_arg},
                {{"keepdims", ONNX_NAMESPACE::MakeAttribute("keepdims", int64_t(1))},
                 {"axes", ONNX_NAMESPACE::MakeAttribute("axes", reduce_axes)}}));

    ArgDef target_shape_arg = IA(target.name + "_shape");
    output.push_back(
        NodeDef("Shape",
                {target},
                {target_shape_arg}));

    output.push_back(
        NodeDef("Reshape",
                {reduce_grad_arg, target_shape_arg},
                {output_grad}));
  }
}

}  // namespace training
}  // namespace onnxruntime
