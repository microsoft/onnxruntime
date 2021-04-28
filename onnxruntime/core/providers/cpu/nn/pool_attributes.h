// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include "core/common/common.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/common.h"

namespace onnxruntime {

// A helper struct holding attributes for Pool-family ops
struct PoolAttributes {
  static bool IsGlobalPooling(const std::string& op_name) {
    return op_name == "GlobalAveragePool" || op_name == "GlobalMaxPool" || op_name == "GlobalLpPool";
  }

  PoolAttributes(const OpNodeProtoHelper<ProtoHelperNodeContext>& info,
                 const std::string& op_name, int start_version)
      : global_pooling(IsGlobalPooling(op_name)) {
    if (global_pooling) {
      return;
    }

    ORT_ENFORCE(info.GetAttrs<int64_t>("kernel_shape", kernel_shape).IsOK(),
                "No kernel shape is set.");

    std::string auto_padding;
    ORT_ENFORCE(info.GetAttr<std::string>("auto_pad", &auto_padding).IsOK());
    auto_pad = StringToAutoPadType(auto_padding);

    if (!info.GetAttrs<int64_t>("pads", pads).IsOK() || pads.empty()) {
      pads.resize(kernel_shape.size() * 2, 0);
    }

    if (!info.GetAttrs<int64_t>("strides", strides).IsOK() || strides.empty()) {
      strides.resize(kernel_shape.size(), 1);
    }

    if (!info.GetAttr<int64_t>("ceil_mode", &ceil_mode).IsOK()) {
      ceil_mode = 0;
    }

    default_dilations = false;
    if (!info.GetAttrs<int64_t>("dilations", dilations).IsOK() || dilations.empty()) {
      dilations.resize(kernel_shape.size(), 1);
      default_dilations = true;
    } else {
      default_dilations = std::all_of(dilations.begin(), dilations.end(), [](int64_t i) { return i == 1; });
    }

    if (op_name == "AveragePool") {
      int64_t temp;
      ORT_ENFORCE(info.GetAttr<int64_t>("count_include_pad", &temp).IsOK());
      count_include_pad = (temp != 0);
    }

    if (op_name == "MaxPool") {
      if (start_version >= 8) {
        ORT_ENFORCE(info.GetAttr("storage_order", &storage_order).IsOK());
      }
    }

    for (size_t dim = 0; dim < kernel_shape.size(); ++dim) {
      ORT_ENFORCE(kernel_shape[dim] > 0);
      ORT_ENFORCE(pads[dim] < kernel_shape[dim] && pads[dim + kernel_shape.size()] < kernel_shape[dim],
                  "Pad should be smaller than kernel.");
    }

    ORT_ENFORCE(strides.size() == kernel_shape.size());
    ORT_ENFORCE(dilations.size() == kernel_shape.size(),
                "Dilations dimensions should match kernel shape");
  }

  const bool global_pooling;

  bool count_include_pad{};
  int64_t storage_order{0};  // MaxPool_8 only. 0 is row major, and 1 is column major. Default is 0.
  int64_t ceil_mode{0};      // Introduced in MaxPool_10
  std::vector<int64_t> kernel_shape;
  std::vector<int64_t> pads;
  std::vector<int64_t> strides;
  std::vector<int64_t> dilations;  // Introduced in MaxPool_10
  // default_dilations is true if dilations is not set or all dilations are 1
  bool default_dilations;
  AutoPadType auto_pad;

  std::vector<int64_t> SetOutputSize(const TensorShape& input_shape,
                                     int64_t output_channel,
                                     std::vector<int64_t>* actual_pads) const {
    ORT_ENFORCE(input_shape.Size() > 0 || input_shape[0] == 0,
                "Invalid input shape. Only N can be zero. Got:", input_shape);
    std::vector<int64_t> output_dims;
    int64_t N = input_shape[0];
    InferOutputSize(input_shape.GetDims(), &output_dims, actual_pads);

    output_dims.insert(output_dims.begin(), {N, output_channel});

    return output_dims;
  }

  void InferOutputSize(const std::vector<int64_t>& input_dims,
                       std::vector<int64_t>* output_dims,
                       std::vector<int64_t>* actual_pads) const {
    ORT_ENFORCE(input_dims.size() >= 2);
    if (global_pooling) {
      output_dims->assign(input_dims.size() - 2, 1);
    } else {
      for (size_t dim = 0; dim < input_dims.size() - 2; ++dim) {
        int64_t dim_size = 0;
        ComputeSizePadDilations(static_cast<int>(input_dims[dim + 2]),
                                strides[dim],
                                kernel_shape[dim],
                                &actual_pads->at(dim),
                                &actual_pads->at(input_dims.size() + dim - 2),
                                dilations[dim],
                                &dim_size);
        output_dims->push_back(dim_size);
      }
    }
  }

  void ComputeSizePadDilations(const int64_t in_size,
                               const int64_t stride,
                               const int64_t kernel,
                               int64_t* pad_head,
                               int64_t* pad_tail,
                               int64_t dilation,
                               int64_t* out_size) const {
    if (auto_pad != AutoPadType::NOTSET) {
      switch (auto_pad) {
        case AutoPadType::VALID:
          *pad_head = 0;
          *pad_tail = 0;
          *out_size = ComputeOutputSize(in_size, stride, kernel, 0, dilation);
          break;
        case AutoPadType::SAME_LOWER: {
          int64_t legacy_target_size = (in_size + stride - 1) / stride;
          int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_size;
          *pad_head = (pad_needed + 1) / 2;
          *pad_tail = pad_needed - *pad_head;
          *out_size = ComputeOutputSize(in_size, stride, kernel, pad_needed, dilation);
          break;
        }
        case AutoPadType::SAME_UPPER: {
          int64_t legacy_target_size = (in_size + stride - 1) / stride;
          int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_size;
          *pad_head = pad_needed / 2;
          *pad_tail = pad_needed - *pad_head;
          *out_size = ComputeOutputSize(in_size, stride, kernel, pad_needed, dilation);
          break;
        }
        default: {
          ORT_THROW("Unsupported AutoPad Type.");
        }
      }
    } else {
      *out_size = ComputeOutputSize(in_size, stride, kernel, *pad_head + *pad_tail, dilation);
    }
  }

  int64_t ComputeOutputSize(int64_t in_size,
                            int64_t stride,
                            int64_t kernel,
                            int64_t pad_needed,
                            int64_t dilation) const {
    if (ceil_mode == 0) {
      return static_cast<int64_t>(static_cast<float>(in_size + pad_needed - dilation * (kernel - 1) - 1) / stride + 1);
    }
    return static_cast<int64_t>(
        std::ceil(static_cast<float>(in_size + pad_needed - dilation * (kernel - 1) - 1) / stride + 1));
  }
};

}  // namespace onnxruntime
