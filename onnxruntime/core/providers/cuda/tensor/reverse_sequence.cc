// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "reverse_sequence.h"
#include "reverse_sequence_impl.h"

#include "core/providers/cpu/tensor/utils.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    ReverseSequence,
    kOnnxDomain,
    10,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes()),
    ReverseSequenceOp);

#define ReverseSequenceCallCudaImplTypeAs(T, TEqual)                                                 \
  if (X.IsDataType<T>()) {                                                                           \
    CUDA_RETURN_IF_ERROR(ReverseSequenceCudaImpl(                                                    \
        Stream(),                                                                                    \
        reinterpret_cast<const typename ToCudaType<TEqual>::MappedType*>(X.template Data<T>()),      \
        seq_lengths.Data<int64_t>(),                                                                 \
        reinterpret_cast<typename ToCudaType<TEqual>::MappedType*>(Y.template MutableData<T>()),     \
        gsl::narrow<int>(batch_size), gsl::narrow<int>(max_seq_len), gsl::narrow<int>(element_size), \
        time_major_));                                                                               \
    return Status::OK();                                                                             \
  }

Status ReverseSequenceOp::ComputeInternal(OpKernelContext* context) const {
  const auto& X = *context->Input<Tensor>(0);
  const auto& dims = X.Shape();

  const auto batch_size = time_major_ ? dims[1] : dims[0];
  const auto max_seq_len = time_major_ ? dims[0] : dims[1];
  const auto element_size = dims.SizeFromDimension(2);

  const auto& seq_lengths = *context->Input<Tensor>(1);
  const auto& seq_len_shape = seq_lengths.Shape();

  if (seq_len_shape.NumDimensions() != 1 || seq_len_shape[0] != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "sequence_lens shape must be {batch_size}. Got:",
                           seq_len_shape, ". batch_size=", batch_size);
  }
  auto& Y = *context->Output(0, dims);

  ReverseSequenceCallCudaImplTypeAs(float, int32_t);
  ReverseSequenceCallCudaImplTypeAs(int32_t, int32_t);
  ReverseSequenceCallCudaImplTypeAs(uint32_t, int32_t);

  ReverseSequenceCallCudaImplTypeAs(MLFloat16, int16_t);
  ReverseSequenceCallCudaImplTypeAs(int16_t, int16_t);
  ReverseSequenceCallCudaImplTypeAs(uint16_t, int16_t);

  ReverseSequenceCallCudaImplTypeAs(int8_t, int8_t);
  ReverseSequenceCallCudaImplTypeAs(uint8_t, int8_t);
  ReverseSequenceCallCudaImplTypeAs(bool, int8_t);

  ReverseSequenceCallCudaImplTypeAs(int64_t, int64_t);
  ReverseSequenceCallCudaImplTypeAs(double, int64_t);
  ReverseSequenceCallCudaImplTypeAs(uint64_t, int64_t);

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Type for ", X.DataType(), " is not supported yet in ReverseSequence.");
}

}  // namespace cuda
}  // namespace onnxruntime
