// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "utils.h"
#include "core/providers/common.h"

namespace onnxruntime {

class SqueezeBase {
 protected:
  explicit SqueezeBase(const OpKernelInfo& info) {
    std::vector<int64_t> axes;
    // Parse attribute 'axes'
    Status status = info.GetAttrs<int64_t>("axes", axes); 

    // Handle out of order and repeating dims when 'axes' exists.
    if (status.IsOK()) {
      std::sort(axes.begin(), axes.end());
      axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
      axes_ = axes;
    }
  }

  static std::vector<int64_t> ComputeOutputShape(
      const TensorShape& input_shape,
      const TensorShape& axes) {
    size_t j = 0;
    std::vector<int64_t> output_shape;
    auto num_dimensions = input_shape.NumDimensions();

    // Handle negtive axis, then resort and uniq.
    std::vector<int64_t> axes_corrected(axes.NumDimensions());
    for (size_t i = 0; i < axes.NumDimensions(); i++) {
      axes_corrected[i] = HandleNegativeAxis(axes[i], num_dimensions);
    }
    std::sort(axes_corrected.begin(), axes_corrected.end());
    axes_corrected.erase(std::unique(axes_corrected.begin(), axes_corrected.end()), axes_corrected.end());

    for (size_t i = 0; i < num_dimensions; ++i) {
      if ((j < axes_corrected.size() && axes_corrected[j] == static_cast<int64_t>(i)) ||
          (axes_corrected.size() == 0 && input_shape[i] == 1)) {
        ORT_ENFORCE(input_shape[i] == 1, "Dimension of input ", i, " must be 1 instead of ", input_shape[i],
                    ". shape=", input_shape);
        ++j;
        continue;
      }
      output_shape.push_back(input_shape[i]);
    }
    return output_shape;
  }

  TensorShape axes_;
};

class Squeeze final : public OpKernel, public SqueezeBase {
 public:
  explicit Squeeze(const OpKernelInfo& info) : OpKernel(info), SqueezeBase(info) {}

  Status Compute(OpKernelContext* context) const override {
    const auto* X = context->Input<Tensor>(0);
    const TensorShape& X_shape = X->Shape();
    std::vector<int64_t> output_shape = ComputeOutputShape(X_shape.GetDims(), axes_);

    Tensor* Y = context->Output(0, TensorShape(output_shape));

    CopyCpuTensor(X, Y);

    return Status::OK();
  }
};

}  // namespace onnxruntime