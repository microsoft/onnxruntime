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
}  // namespace utils
}  // namespace onnxruntime
