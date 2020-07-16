// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/common/exceptions.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/providers/common.h"
#include "core/util/math.h"

namespace onnxruntime {

// A helper struct holding attributes for Conv-family ops
struct ConvAttributes {
  explicit ConvAttributes(const OpNodeProtoHelper<ProtoHelperNodeContext>& info) {
    std::string auto_pad_str;
    auto status = info.GetAttr<std::string>("auto_pad", &auto_pad_str);
    auto_pad = status.IsOK() ? StringToAutoPadType(auto_pad_str) : AutoPadType::NOTSET;

    kernel_shape_specified = info.GetAttrs<int64_t>("kernel_shape", kernel_shape_).IsOK();

    status = info.GetAttrs<int64_t>("strides", strides);
    if (!status.IsOK() || strides.empty()) {
      strides.resize(kernel_shape_.size(), 1);
    }

    status = info.GetAttrs<int64_t>("pads", pads);
    if (!status.IsOK()) {
      pads.resize(kernel_shape_.size() * 2, 0);
    }

    status = info.GetAttrs<int64_t>("dilations", dilations);
    if (!status.IsOK() || dilations.empty()) {
      dilations.resize(kernel_shape_.size(), 1);
    }

    status = info.GetAttr<int64_t>("group", &group);
    if (!status.IsOK()) {
      group = 1;
    }

#if false
    // TODO: Re-enable when attributes values are guaranteed to be filled.
    // Make sure empty strides or dilations are defaulted to 1 if necessary
    std::string auto_pad_str;
    ORT_ENFORCE(info.GetAttr<std::string>("auto_pad", &auto_pad_str).IsOK());
    auto_pad = StringToAutoPadType(auto_pad_str);
    ORT_ENFORCE(info.GetAttr<int64_t>("group", &group).IsOK());
    ORT_ENFORCE(info.GetAttrs<int64_t>("kernel_shape", kernel_shape_).IsOK());
    ORT_ENFORCE(info.GetAttrs<int64_t>("strides", strides).IsOK());
    ORT_ENFORCE(info.GetAttrs<int64_t>("pads", pads).IsOK());
    ORT_ENFORCE(info.GetAttrs<int64_t>("dilations", dilations).IsOK());
#endif
  }

  ~ConvAttributes() = default;

  Status ComputeKernelShape(const TensorShape& weight_shape, std::vector<int64_t>& kernel_shape) const {
    if (kernel_shape_specified) {
      kernel_shape = kernel_shape_;
      if (kernel_shape.size() + 2 != weight_shape.NumDimensions()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel_shape num_dims is not compatible with W num_dims.",
                               " kernel_shape: ", TensorShape(kernel_shape).ToString().c_str(),
                               " W: ", weight_shape.ToString().c_str());
      }
      for (size_t i = 0; i < kernel_shape.size(); ++i) {
        if (kernel_shape[i] != weight_shape[i + 2]) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "kernel_shape is not compatible with W shape.",
                                 " kernel_shape: ", TensorShape(kernel_shape).ToString().c_str(),
                                 " W: ", weight_shape.ToString().c_str());
        }
      }
    } else {
      auto& weight_dims = weight_shape.GetDims();
      kernel_shape = std::vector<int64_t>(weight_dims.begin() + 2, weight_dims.end());
    }

    return Status::OK();
  }

  Status ValidateInputShape(const Tensor* X, const Tensor* W) const {
    const int64_t C = X->Shape()[1];
    const int64_t M = W->Shape()[0];

    if (X->Shape().NumDimensions() != W->Shape().NumDimensions()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "X num_dims does not match W num_dims.",
                             " X: ", X->Shape().ToString().c_str(),
                             " W: ", W->Shape().ToString().c_str());
    }

    if (C != W->Shape()[1] * group) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input channels C is not equal to kernel channels * group.",
                             " C: ", C,
                             " kernel channels: ", W->Shape()[1],
                             " group: ", group);
    }

    if (M % group != 0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Output channels M is not divisible by group.",
                             " M: ", M,
                             " group: ", group);
    }
    return Status::OK();
  }

  Status InferOutputShape(const TensorShape& input_shape,
                          const std::vector<int64_t>& kernel_shape,
                          const std::vector<int64_t>& strides_p,
                          const std::vector<int64_t>& dilations_p,
                          std::vector<int64_t>& pads_p,
                          std::vector<int64_t>& output_shape,
                          bool force_symmetric_auto_padding = false) const {
    size_t rank = input_shape.NumDimensions();
    for (size_t dim = 0; dim < rank; ++dim) {
      if (dim >= strides_p.size() || dim >= kernel_shape.size() ||
          dim >= dilations_p.size() || dim >= pads_p.size() ||
          rank + dim >= pads_p.size()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Out of bound access to array");
      }
      int64_t dim_size = 0;
      ORT_RETURN_IF_ERROR(ComputePadAndOutputShape(input_shape[dim],
                                                   strides_p[dim],
                                                   kernel_shape[dim],
                                                   dilations_p[dim],
                                                   auto_pad,
                                                   pads_p.at(dim),
                                                   pads_p.at(input_shape.NumDimensions() + dim),
                                                   dim_size,
                                                   force_symmetric_auto_padding));
      if (dim_size <= 0) {
        return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid input shape: " + input_shape.ToString());
      }
      output_shape.push_back(dim_size);
    }
    return Status::OK();
  }

  bool HasStridesOneAndNoPadding() const {
    if (std::all_of(strides.begin(), strides.end(), [](int64_t v) { return v == 1; })) {
      if (std::all_of(pads.begin(), pads.end(), [](int64_t v) { return v == 0; })) {
        return true;
      }
    }
    return false;
  }

  AutoPadType auto_pad;
  int64_t group;
  bool kernel_shape_specified;
  std::vector<int64_t> strides;
  std::vector<int64_t> pads;
  std::vector<int64_t> dilations;
  std::string activation;
  float alpha;

 private:
  std::vector<int64_t> kernel_shape_;  // must use ComputeKernelShape(...), instead of kernel_shape_
};

}  // namespace onnxruntime
