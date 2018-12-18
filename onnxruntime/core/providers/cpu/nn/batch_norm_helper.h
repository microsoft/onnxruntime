// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/tensor.h"
#include <sstream>

namespace onnxruntime {
class BatchNormHelper {
 public:
  static common::Status ValidateInputs(const Tensor* X,
                                       const Tensor* scale,
                                       const Tensor* B,
                                       const Tensor* mean,
                                       const Tensor* var) {
    // defined as per spec and used for validation
    constexpr int kNumInputScaleDimensions = 1;
    constexpr int kNumInputBiasDimensions = 1;
    constexpr int kNumInputMeanDimensions = 1;
    constexpr int kNumInputVarianceDimensions = 1;
    //constexpr int kMinCudaNumDims = 4;
    //constexpr int kMaxCudaNumDims = 5;

    if (X->Shape().GetDims().empty()) {
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Invalid input X: Empty dimensions");
    }

    int64_t num_channels = X->Shape().GetDims()[1];

    if (scale->Shape().NumDimensions() != kNumInputScaleDimensions) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input scale: NumDimensions() != ", kNumInputScaleDimensions);
    }
    if (scale->Shape().GetDims()[0] != num_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input scale: 0th dimension != ", num_channels);
    }

    if (B->Shape().NumDimensions() != kNumInputBiasDimensions) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input B: NumDimensions() != ", kNumInputBiasDimensions);
    }
    if (B->Shape().GetDims()[0] != num_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input B: 0th dimension != ", num_channels);
    }

    if (mean->Shape().NumDimensions() != kNumInputMeanDimensions) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input mean: NumDimensions() != ", kNumInputMeanDimensions);
    }
    if (mean->Shape().GetDims()[0] != num_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input mean: 0th dimension != ", num_channels);
    }

    if (var->Shape().NumDimensions() != kNumInputVarianceDimensions) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input var: NumDimensions() != ", kNumInputVarianceDimensions);
    }
    if (var->Shape().GetDims()[0] != num_channels) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid input var: 0th dimension != ", num_channels);
    }

    return common::Status::OK();
  }

  static void NormalizeDims(const TensorShape& x_shape, std::vector<int64_t>& new_dims) {
    new_dims.clear();
    auto& orig_dims = x_shape.GetDims();
    if (orig_dims.size() == 4 /*supported size by CUDA*/ ||
        orig_dims.size() == 5 /*supported size by CUDA*/) {
      new_dims = orig_dims;
      return;
    }

    auto rank = x_shape.NumDimensions();
    auto num_samples = rank > 0 ? orig_dims[0] : 1;  // NCHW
    auto num_channels = rank > 1 ? orig_dims[1] : 1;
    auto width = rank > 3 ? orig_dims[3] : 1;
    auto height = rank > 2 ? orig_dims[2] : 1;
    new_dims = {num_samples, num_channels, height, width};
  }
};
}  // namespace onnxruntime
