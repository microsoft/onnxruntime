// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/status.h"
#include "core/framework/tensor.h"
#endif
#include <sstream>

namespace onnxruntime {

class InstanceNormHelper {
 public:
  static common::Status ValidateInputs(const Tensor* input, const Tensor* scale, const Tensor* B) {
    if (input->Shape().NumDimensions() < 3) {
      std::ostringstream ostr;
      ostr << "Invalid input data: number of dimensions is less than 3: " << input->Shape().NumDimensions();
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
    }
    if (scale->Shape().NumDimensions() != 1) {
      std::ostringstream ostr;
      ostr << "Invalid input scale: number of dimensions is not 1: " << scale->Shape().NumDimensions();
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
    }
    if (scale->Shape().Size() != input->Shape().GetDims()[1]) {
      std::ostringstream ostr;
      ostr << "Mismatch between input data and scale: size of scale != input channel count "
           << scale->Shape().Size() << " vs. " << input->Shape().GetDims()[1];
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
    }

    if (B->Shape().NumDimensions() != 1) {
      std::ostringstream ostr;
      ostr << "Invalid input B: number of dimensions is not 1: " << B->Shape().NumDimensions();
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
    }

    if (B->Shape().Size() != input->Shape().GetDims()[1]) {
      std::ostringstream ostr;
      ostr << "Mismatch between input data and B: size of B != input channel count "
           << B->Shape().Size() << " vs. " << input->Shape().GetDims()[1];
      return common::Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, ostr.str());
    }

    return common::Status::OK();
  }
};

}  // namespace onnxruntime
