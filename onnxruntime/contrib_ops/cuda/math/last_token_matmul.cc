// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "last_token_matmul.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status LastTokenMatMul<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  const Tensor* input_1 = ctx->Input<Tensor>(0);
  const Tensor* input_2 = ctx->Input<Tensor>(1);

  const auto& input_1_shape = input_1->Shape().GetDims();
  const auto& input_2_shape = input_2->Shape().GetDims();

  if ((input_1_shape.size() != 3) && (input_2_shape.size() != 2)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "First input must be 3D and the second input must be 2D");
  }

  if (input_1_shape[2] != input_2_shape[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "MatMul shape requirement unmet");
  }

  int64_t batch_size = input_1_shape[0];
  int64_t sequence_length = input_1_shape[1];
  int64_t hidden_dim = input_1_shape[2];

  int64_t proj_dim = input_2_shape[1];

  auto output_shape = input_1->Shape();
  output_shape[1] = 1;  // Middle dim is always 1
  output_shape[2] = input_2_shape[1];

  Tensor* output = ctx->Output(0, output_shape);

  // Bail out early if the output is going to be empty
  if (output->Shape().Size() == 0) {
    return Status::OK();
  }

  const CudaT alpha = ToCudaType<T>::FromFloat(1.0f);
  const CudaT zero = ToCudaType<T>::FromFloat(0.0f);
  auto& device_prop = GetDeviceProp();

  IAllocatorUniquePtr<void> sliced_input_buffer;

  // We only need to slice out the last token per batch if the sequence length
  // is greater than 1
  if (sequence_length != 1) {
    constexpr size_t element_size = sizeof(T);
    size_t sliced_input_buffer_size = static_cast<size_t>(batch_size) * hidden_dim * element_size;

    sliced_input_buffer = GetScratchBuffer<void>(sliced_input_buffer_size);
    ORT_ENFORCE(sliced_input_buffer.get() != nullptr);

    CudaT* dst = reinterpret_cast<CudaT*>(sliced_input_buffer.get());
    const CudaT* src = reinterpret_cast<const CudaT*>(input_1->Data<T>()) + (sequence_length - 1) * hidden_dim;
    size_t copy_size_in_bytes = hidden_dim * element_size;

    for (int i = 0; i < batch_size; ++i) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst, src, copy_size_in_bytes, cudaMemcpyDeviceToDevice, Stream()));
      src += sequence_length * hidden_dim;
      dst += hidden_dim;
    }
  }

  CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
      Base::CublasHandle(),
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      static_cast<int>(proj_dim),
      static_cast<int>(batch_size),
      static_cast<int>(hidden_dim),
      &alpha,
      reinterpret_cast<const CudaT*>(input_2->Data<T>()),
      static_cast<int>(proj_dim),
      // Re-use the input buffer if sequence_length == 1
      sequence_length == 1 ? reinterpret_cast<const CudaT*>(input_1->Data<T>())
                           : reinterpret_cast<const CudaT*>(sliced_input_buffer.get()),
      static_cast<int>(hidden_dim),
      &zero,
      reinterpret_cast<CudaT*>(output->MutableData<T>()),
      static_cast<int>(proj_dim),
      device_prop));

  return Status::OK();
}

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      LastTokenMatMul,                                            \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      LastTokenMatMul<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime