// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"
#include "core/providers/common.h"
#endif

#include "utils.h"

namespace onnxruntime {

class UnsqueezeBase {
 public:
  struct Prepare {
    const Tensor* input_tensor = nullptr;
    Tensor* output_tensor = nullptr;
  };

  Status PrepareCompute(OpKernelContext* context, Prepare& p) const;

 protected:
  UnsqueezeBase(const OpKernelInfo& info) {
    size_t num_inputs = info.GetInputCount();
    if (num_inputs == 1) {  // axes must be a valid attribute
      ORT_ENFORCE(info.GetAttrs("axes", axes_).IsOK(), "Missing/Invalid 'axes' attribute value");
    }
  }

  static TensorShapeVector ComputeOutputShape(
      const TensorShape& input_shape,
      const TensorShapeVector& axes) {
    TensorShapeVector output_shape;
    auto num_dimensions = input_shape.NumDimensions();

    auto total_num_dimensions = num_dimensions + axes.size();
    // Handle negtive axis, then resort and uniq.
    TensorShapeVector axes_corrected(axes.size());
    for (size_t i = 0; i < axes.size(); i++) {
      axes_corrected[i] = HandleNegativeAxis(axes[i], total_num_dimensions);
    }
    std::sort(axes_corrected.begin(), axes_corrected.end());
    axes_corrected.erase(std::unique(axes_corrected.begin(), axes_corrected.end()), axes_corrected.end());
    ORT_ENFORCE(axes_corrected.size() == axes.size(), "Axes input has duplicate values.");
    ORT_ENFORCE(axes_corrected.size() > 0, "Unsqueeze axes is empty.");
    size_t corr = 0;
    size_t j = 0;
    for (size_t i = 0; i < total_num_dimensions; ++i) {
      if (j < axes_corrected.size() && axes_corrected[j] == static_cast<int64_t>(i)) {
        output_shape.push_back(1);
        ++j;
        ++corr;
        continue;
      }
      output_shape.push_back(input_shape[i - corr]);
    }
    return output_shape;
  }

  TensorShapeVector axes_;
};

class Unsqueeze final : public OpKernel, public UnsqueezeBase {
 public:
  Unsqueeze(const OpKernelInfo& info) : OpKernel(info), UnsqueezeBase(info) {}
  Status Compute(OpKernelContext* context) const override;
};

}  // namespace onnxruntime
