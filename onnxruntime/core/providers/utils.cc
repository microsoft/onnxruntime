// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/providers/utils.h"

namespace onnxruntime {
namespace utils {

#if !defined(DISABLE_OPTIONAL_TYPE)
common::Status OutputOptionalWithoutDataHelper(const ONNX_NAMESPACE::TypeProto& input_type_proto,
                                               OpKernelContext* context, int output_index) {
  if (utils::HasOptionalTensorType(input_type_proto)) {
    context->OutputOptionalWithoutData<Tensor>(output_index);
  } else if (utils::HasOptionalTensorSequenceType(input_type_proto)) {
    context->OutputOptionalWithoutData<TensorSeq>(output_index);
  } else {
    // Will never hit this as we don't support any other type than Tensor and TensorSeq
    // for optional type
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported type");
  }

  return Status::OK();
}
#endif

bool ReciprocalIsAFactorOfN(int64_t n, float scale) {
  bool is_factor = false;
  if (scale > 0.f && scale < 1.f) {
    const double factor = 1.0 / scale;
    const double factor_rounded = std::round(factor);
    constexpr double epsilon = 1.0e-4;  // arbitrarily small enough
    if (std::abs(factor - factor_rounded) < epsilon) {
      // result is integer. check if a factor of n
      const int64_t factor_i = static_cast<int64_t>(factor_rounded);
      is_factor = n % factor_i == 0;
    }
  }

  return is_factor;
}

Status ComputeBroadcastOutputShape(const std::string& node_name, const TensorShape& lhs_shape, const TensorShape& rhs_shape, TensorShape& out_shape) {
  size_t lhs_rank = lhs_shape.NumDimensions();
  size_t rhs_rank = rhs_shape.NumDimensions();
  size_t out_rank = std::max(lhs_rank, rhs_rank);

  std::vector<int64_t> output_dims(out_rank, 0);
  for (size_t i = 0; i < out_rank; ++i) {
    int64_t lhs_dim = 1;
    if (i < lhs_rank)
      lhs_dim = lhs_shape[lhs_rank - 1 - i];
    int64_t rhs_dim = 1;
    if (i < rhs_rank)
      rhs_dim = rhs_shape[rhs_rank - 1 - i];
    int64_t max = std::max(lhs_dim, rhs_dim);
    int64_t min = std::min(lhs_dim, rhs_dim);
    int64_t out_dim = (min == 0 ? min : max);  // special case a dim value of 0.
    if (lhs_dim != out_dim && lhs_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": left operand cannot broadcast on dim ", lhs_rank - 1 - i,
                             " LeftShape: ", lhs_shape.ToString(), ", RightShape: ", rhs_shape.ToString());
    if (rhs_dim != out_dim && rhs_dim != 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, node_name, ": right operand cannot broadcast on dim ", rhs_rank - 1 - i,
                             " LeftShape: ", lhs_shape.ToString(), ", RightShape: ", rhs_shape.ToString());
    output_dims[out_rank - 1 - i] = out_dim;
  }
  out_shape = TensorShape(output_dims);
  return Status::OK();
}
}  // namespace utils
}  // namespace onnxruntime
