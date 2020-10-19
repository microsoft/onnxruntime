// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lstm_base.h"

namespace onnxruntime {
Status LSTMBase::ValidateInputs(const Tensor& X, const TensorShape& W_shape, const TensorShape& R_shape,
                                const Tensor* B, const Tensor* sequence_lens, const Tensor* initial_h,
                                const Tensor* initial_c, const Tensor* P, int batch_size) const {
  auto status =
      rnn::detail::ValidateCommonRnnInputs(X, W_shape, R_shape, B, 4, sequence_lens, initial_h, num_directions_, hidden_size_);
  ORT_RETURN_IF_ERROR(status);

  if (initial_c != nullptr) {
    auto& initial_c_shape = initial_c->Shape();

    if (initial_c_shape.NumDimensions() != 3 || initial_c_shape[0] != num_directions_ ||
        initial_c_shape[1] != batch_size || initial_c_shape[2] != hidden_size_)

      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input initial_c must have shape {", num_directions_, ",", batch_size,
                             ",", hidden_size_, "}. Actual:", initial_c_shape);
  }

  if (P != nullptr) {
    auto& p_shape = P->Shape();

    if (p_shape.NumDimensions() != 2 || p_shape[0] != num_directions_ || p_shape[1] != 3 * hidden_size_)

      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Input P must have shape {", num_directions_, ",", 3 * hidden_size_,
                             "}. Actual:", p_shape);
  }

  return Status::OK();
}
}  // namespace onnxruntime
