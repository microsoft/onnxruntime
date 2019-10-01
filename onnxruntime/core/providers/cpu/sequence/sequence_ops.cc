// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/sequence/sequence_ops.h"

using namespace onnxruntime::common;

namespace onnxruntime {

#define REG_TYPED_KERNEL(T1)                                              \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                         \
      SequenceLength,                                                     \
      11,                                                                 \
      T1,                                                                 \
      KernelDefBuilder()                                                  \
          .TypeConstraint("S", DataTypeImpl::GetSequenceTensorType<T1>()) \
          .TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),   \
      SequenceLength<T1>);

REG_TYPED_KERNEL(float);

template <typename T>
Status SequenceLength<T>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<VectorTensor>(0);
  ORT_ENFORCE(X != nullptr, "Got null input.");

  auto* Y = context->Output(0, {});
  auto* Y_data = Y->template MutableData<int64_t>();
  *Y_data = static_cast<int64_t>(X->size());

  return Status::OK();
}

}  // namespace onnxruntime
