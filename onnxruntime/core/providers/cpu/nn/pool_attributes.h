// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include "core/common/safeint.h"
#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/common.h"
#endif

namespace onnxruntime {

// A helper struct holding attributes for Pool-family ops
struct PoolAttributes {
  static bool IsGlobalPooling(const std::string& op_name) {
    return op_name == "GlobalAveragePool" || op_name == "GlobalMaxPool" || op_name == "GlobalLpPool";
  }

#ifdef SHARED_PROVIDER
  // Shared providers don't know about OpNodeProtoHelper
  PoolAttributes(const OpKernelInfo& info,
#else
  template <typename KernelInfoType>
  PoolAttributes(const KernelInfoType& info,
#endif
                 const std::string& op_name, int start_version)
      : global_pooling(IsGlobalPooling(op_name)) {
    if (global_pooling) {
      return;
    }

    ORT_ENFORCE(info.GetAttrs("kernel_shape", kernel_shape).IsOK(),
                "No kernel shape is set.");

    std::string auto_padding;
    if (op_name != "MaxUnpool") {
      ORT_ENFORCE(info.template GetAttr<std::string>("auto_pad", &auto_padding).IsOK());
    }
    auto_pad = StringToAutoPadType(auto_padding);

    if (!info.GetAttrs("pads", pads).IsOK() || pads.empty()) {
      pads.resize(kernel_shape.size() * 2, 0);
    }
    ORT_ENFORCE(pads.size() == kernel_shape.size() * 2,
                "'pads' must have twice the kernel_shape rank (2 entries per spatial dim). Got pads size: ",
                pads.size(), ", expected: ", kernel_shape.size() * 2);

    if (!info.GetAttrs("strides", strides).IsOK() || strides.empty()) {
      strides.resize(kernel_shape.size(), 1);
    }

    if (!info.template GetAttr<int64_t>("ceil_mode", &ceil_mode).IsOK()) {
      ceil_mode = 0;
    }

    default_dilations = false;
    if (!info.GetAttrs("dilations", dilations).IsOK() || dilations.empty()) {
      dilations.resize(kernel_shape.size(), 1);
      default_dilations = true;
    } else {
      default_dilations = std::all_of(dilations.begin(), dilations.end(), [](int64_t i) { return i == 1; });
    }

    if (op_name == "AveragePool") {
      int64_t temp;
      ORT_ENFORCE(info.template GetAttr<int64_t>("count_include_pad", &temp).IsOK());
      count_include_pad = (temp != 0);
    }

    if (op_name == "MaxPool") {
      if (start_version >= 8) {
        ORT_ENFORCE(info.GetAttr("storage_order", &storage_order).IsOK());
        ORT_ENFORCE(storage_order == 0 || storage_order == 1,
                    "storage_order must be 0 (row-major) or 1 (column-major). Got: ", storage_order);
      }
    }

    for (size_t dim = 0; dim < kernel_shape.size(); ++dim) {
      ORT_ENFORCE(kernel_shape[dim] > 0);
      ORT_ENFORCE(pads[dim] < kernel_shape[dim] && pads[dim + kernel_shape.size()] < kernel_shape[dim],
                  "Pad should be smaller than kernel.");
    }

    ORT_ENFORCE(strides.size() == kernel_shape.size(),
                "Strides dimensions should match kernel shape");
    for (auto stride : strides) {
      ORT_ENFORCE(stride > 0, "All stride values must be positive, got: ", stride);
    }
    ORT_ENFORCE(dilations.size() == kernel_shape.size(),
                "Dilations dimensions should match kernel shape");
    for (auto dilation : dilations) {
      ORT_ENFORCE(dilation > 0, "All dilation values must be positive, got: ", dilation);
    }
  }

  const bool global_pooling;

  bool count_include_pad{};
  int64_t storage_order{0};  // MaxPool_8 only. 0 is row major, and 1 is column major. Default is 0.
  int64_t ceil_mode{0};      // Introduced in MaxPool_10
  TensorShapeVector kernel_shape;
  TensorShapeVector pads;
  TensorShapeVector strides;
  TensorShapeVector dilations;  // Introduced in MaxPool_10
  // default_dilations is true if dilations is not set or all dilations are 1
  bool default_dilations{false};
  AutoPadType auto_pad{AutoPadType::NOTSET};

  TensorShapeVector SetOutputSize(const TensorShape& input_shape,
                                  int64_t output_channel,
                                  TensorShapeVector* actual_pads,
                                  bool is_nhwc = false) const {
    ORT_ENFORCE(input_shape.NumDimensions() >= 2,
                "Input must have at least 2 dimensions (N, C, ...spatial). Got rank: ",
                input_shape.NumDimensions());
    ORT_ENFORCE(input_shape.Size() > 0 || input_shape[0] == 0,
                "Invalid input shape. Only N can be zero. Got:", input_shape);
    TensorShapeVector output_dims;
    int64_t N = input_shape[0];
    InferOutputSize(input_shape.GetDims(), &output_dims, actual_pads, is_nhwc);
    if (is_nhwc) {
      output_dims.insert(output_dims.begin(), N);
      output_dims.push_back(output_channel);
    } else {
      output_dims.insert(output_dims.begin(), {N, output_channel});
    }
    return output_dims;
  }

  void InferOutputSize(gsl::span<const int64_t> input_dims,
                       TensorShapeVector* output_dims,
                       TensorShapeVector* actual_pads,
                       bool is_nhwc = false) const {
    ORT_ENFORCE(input_dims.size() >= 2,
                "Input must have at least 2 dimensions (batch and channel) before the spatial dims. "
                "Got rank: ",
                input_dims.size());
    if (global_pooling) {
      output_dims->assign(input_dims.size() - 2, 1);
    } else {
      ORT_ENFORCE(input_dims.size() - 2 == kernel_shape.size(),
                  "kernel_shape rank must match the input spatial rank. Input spatial rank: ",
                  input_dims.size() - 2, ", kernel_shape rank: ", kernel_shape.size());
      for (size_t dim = 0; dim < input_dims.size() - 2; ++dim) {
        int64_t dim_size = 0;
        auto spatial_dim = is_nhwc ? input_dims[dim + 1] : input_dims[dim + 2];
        ComputeSizePadDilations(spatial_dim,
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
      // TODO: Per the ONNX spec, auto_pad and explicit pads are mutually exclusive. ORT currently
      // accepts both and overwrites any explicit pads below when auto_pad is set, rather than
      // rejecting the model, to preserve backward compatibility with existing models.
      switch (auto_pad) {
        case AutoPadType::VALID:
          *pad_head = 0;
          *pad_tail = 0;
          *out_size = ComputeOutputSize(in_size, stride, kernel, 0, 0, dilation);
          break;
        case AutoPadType::SAME_LOWER: {
          int64_t legacy_target_size = (in_size + stride - 1) / stride;
          int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_size;
          *pad_head = (pad_needed + 1) / 2;
          *pad_tail = pad_needed - *pad_head;
          *out_size = ComputeOutputSize(in_size, stride, kernel, *pad_head, *pad_tail, dilation);
          break;
        }
        case AutoPadType::SAME_UPPER: {
          int64_t legacy_target_size = (in_size + stride - 1) / stride;
          int64_t pad_needed = (legacy_target_size - 1) * stride + kernel - in_size;
          *pad_head = pad_needed / 2;
          *pad_tail = pad_needed - *pad_head;
          *out_size = ComputeOutputSize(in_size, stride, kernel, *pad_head, *pad_tail, dilation);
          break;
        }
        default: {
          ORT_THROW("Unsupported AutoPad Type.");
        }
      }
    } else {
      *out_size = ComputeOutputSize(in_size, stride, kernel, *pad_head, *pad_tail, dilation);
    }
  }
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// Chance of arithmetic overflow could be reduced
#pragma warning(disable : 26451)
#endif
  int64_t ComputeOutputSize(int64_t in_size,
                            int64_t stride,
                            int64_t kernel,
                            int64_t pad_head,
                            int64_t pad_tail,
                            int64_t dilation) const {
    // SafeInt wraps the residual output-size arithmetic below (the numerator terms and the +1 /
    // ceil-mode add and subtract) to catch int64 overflow. It does not cover the auto_pad SAME_*
    // padding math in the caller or the numerator / stride division.
    int64_t numerator = SafeInt<int64_t>(in_size) + pad_head + pad_tail - SafeInt<int64_t>(dilation) * (kernel - 1) - 1;
    int64_t out_size = static_cast<int64_t>(SafeInt<int64_t>(numerator / stride) + 1);

    if (ceil_mode == 1) {
      int64_t ceil_div = static_cast<int64_t>(std::ceil(static_cast<float>(numerator) / stride));
      out_size = static_cast<int64_t>(SafeInt<int64_t>(ceil_div) + 1);
      // Ensure that the last pooling starts inside the image (at least 1 pixel)
      // Reference: https://github.com/onnx/onnx/pull/5741
      int64_t last_pool_start = static_cast<int64_t>((SafeInt<int64_t>(out_size) - 1) * stride);
      int64_t input_extent = static_cast<int64_t>(SafeInt<int64_t>(in_size) + pad_head);
      if (last_pool_start >= input_extent) {
        --out_size;
      }
    }
    ORT_ENFORCE(out_size >= 0,
                "Calculated output dimension is negative. Check kernel_shape, pads, strides and dilations.");
    return out_size;
  }
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
};

}  // namespace onnxruntime
