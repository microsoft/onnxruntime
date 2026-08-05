// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

inline Status ValidateHyperInputs(const Tensor& hidden, const Tensor& weight, const Tensor& bias,
                                  const Tensor& scale, bool is_head, int64_t& rows,
                                  int64_t& streams, int64_t& hidden_size) {
  ORT_RETURN_IF_NOT(hidden.Shape().NumDimensions() == 4 && weight.Shape().NumDimensions() == 2 &&
                        bias.Shape().NumDimensions() == 1 && scale.Shape().NumDimensions() == 1,
                    "HyperConnection inputs have invalid ranks.");
  streams = hidden.Shape()[2];
  hidden_size = hidden.Shape()[3];
  rows = hidden.Shape()[0] * hidden.Shape()[1];
  const int64_t output_size = is_head ? streams : (2 + streams) * streams;
  ORT_RETURN_IF_NOT(streams > 0 && hidden_size > 0 && weight.Shape()[0] == output_size &&
                        weight.Shape()[1] == streams * hidden_size && bias.Shape().Size() == output_size &&
                        scale.Shape().Size() == (is_head ? 1 : 3) && weight.IsDataType<float>() &&
                        bias.IsDataType<float>() && scale.IsDataType<float>(),
                    "HyperConnection parameter shapes or types are inconsistent with hidden_streams.");
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime