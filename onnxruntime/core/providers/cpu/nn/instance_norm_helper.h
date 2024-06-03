// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/status.h"
#include "core/framework/tensor.h"
#endif
#include <sstream>
#include <utility>

namespace onnxruntime {

class InstanceNormHelper {
 public:
  static common::Status ValidateInputs(const Tensor* input, const Tensor* scale, const Tensor* B,
                                       bool is_nhwc = false) {
    const auto rank = input->Shape().NumDimensions();
    if (rank < 3) {
      std::ostringstream ostr;
      ostr << "Invalid input data: number of dimensions is less than 3: " << input->Shape().NumDimensions();
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
    }
    if (scale->Shape().NumDimensions() != 1) {
      std::ostringstream ostr;
      ostr << "Invalid input scale: number of dimensions is not 1: " << scale->Shape().NumDimensions();
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
    }
    auto in_dims = input->Shape().GetDims();
    auto in_channels = is_nhwc ? in_dims[rank - 1] : in_dims[1];

    if (scale->Shape().Size() != in_channels) {
      std::ostringstream ostr;
      ostr << "Mismatch between input data and scale: size of scale != input channel count " << scale->Shape().Size()
           << " vs. " << in_channels;
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
    }

    if (B->Shape().NumDimensions() != 1) {
      std::ostringstream ostr;
      ostr << "Invalid input B: number of dimensions is not 1: " << B->Shape().NumDimensions();
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
    }

    if (B->Shape().Size() != in_channels) {
      std::ostringstream ostr;
      ostr << "Mismatch between input data and B: size of B != input channel count " << B->Shape().Size() << " vs. "
           << in_channels;
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
    }

    return common::Status::OK();
  }
};

}  // namespace onnxruntime
