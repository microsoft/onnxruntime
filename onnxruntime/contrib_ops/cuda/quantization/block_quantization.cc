// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "block_quantization.h"
#include "block_quantization_impl.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/framework/tensor_shape.h"
#include "gsl/gsl"

#include <numeric>
#include <functional>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

ONNX_OPERATOR_KERNEL_EX(
    BlockQuantize,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("F", BuildKernelDefConstraints<float, MLFloat16>())
        .TypeConstraint("B", DataTypeImpl::GetTensorType<int32_t>())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("S", BuildKernelDefConstraints<float, MLFloat16>())
        .InputMemoryType(OrtMemTypeCPUInput, 1),  // block_size
    BlockQuantize);

Status BlockQuantize::ComputeInternal(OpKernelContext* context) const {
  static constexpr int MaxBlockSize = 1024;
  static constexpr int MinBlockSize = 8;

  const Tensor* block_size_tensor = context->Input<Tensor>(1);
  ORT_ENFORCE(IsScalarOr1ElementVector(block_size_tensor), "block_sizee must be a scalar or 1D tensor of size 1");
  auto block_size = static_cast<unsigned>(*block_size_tensor->Data<int32_t>());
  ORT_ENFORCE(block_size >= 8 && block_size <= MaxBlockSize && (block_size & -block_size) == 0,
              "block_size value must >= ", MinBlockSize, ", and  <= ", MaxBlockSize, ", and be power of 2.");

  const Tensor* x_tensor = context->Input<Tensor>(0);
  auto element_count = x_tensor->Shape().Size();
  ORT_ENFORCE(element_count > 0 && (element_count % block_size) == 0,
              "input element count should be of full blocks");
  auto block_count = element_count / block_size;

  TensorShape scale_shape{block_count};
  Tensor* scale_tensor = context->Output(1, scale_shape);
  Tensor* y_tensor = context->Output(0, x_tensor->Shape());

  if (x_tensor->IsDataType<MLFloat16>()) {  // half precision float
    typedef typename ToCudaType<MLFloat16>::MappedType CudaT;
    if (force_fp32_scale_) {
      return CudaBlockQuantize(
          Stream(context),
          GetDeviceProp(),
          (const CudaT*)(x_tensor->template Data<MLFloat16>()),
          block_size,
          static_cast<unsigned>(block_count),
          scale_tensor->template MutableData<float>(),
          y_tensor->MutableData<int8_t>());
    } else {
      return CudaBlockQuantize(
          Stream(context),
          GetDeviceProp(),
          (const CudaT*)(x_tensor->template Data<MLFloat16>()),
          block_size,
          static_cast<unsigned>(block_count),
          (CudaT*)(scale_tensor->template MutableData<MLFloat16>()),
          y_tensor->MutableData<int8_t>());
    }
  }

  ORT_ENFORCE(false, "Current BlockQuantize() do not support other float input except float16.");
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
