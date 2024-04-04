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
}  // namespace utils
}  // namespace onnxruntime
