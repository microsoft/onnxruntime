// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <utility>

#include "orttraining/training_ops/cuda/quantization/fake_quant.h"
#include "orttraining/training_ops/cuda/quantization/fake_quant_impl.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_FAKEQUANT_KERNEL_TYPED(T)                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      FakeQuant,                                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .InputMemoryType(OrtMemTypeCPUInput, 1)                 \
          .InputMemoryType(OrtMemTypeCPUInput, 2)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      FakeQuant<T>);

REGISTER_FAKEQUANT_KERNEL_TYPED(float)

template <typename T>
Status FakeQuant<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  // Prepare the input, scale, zero point
  const auto* input_tensor = ctx->Input<Tensor>(0);
  const CudaT* input_data = reinterpret_cast<const CudaT*>(input_tensor->Data<T>());
  const auto* scale = ctx->Input<Tensor>(1);
  ORT_ENFORCE(IsScalarOr1ElementVector(scale), "Quantization scale must be a scalar or 1D tensor of size 1.");
  const CudaT* quant_scale = reinterpret_cast<const CudaT*>(scale->Data<T>());
  ORT_ENFORCE(*quant_scale != static_cast<const CudaT>(0),
              "Quantization scale cannot be 0. It may result in undefined behavior.");
  const auto* zero_point = ctx->Input<Tensor>(2);
  ORT_ENFORCE(IsScalarOr1ElementVector(zero_point), "Quantization zero point must be a scalar or 1D tensor of size 1.");
  const CudaT* quant_zero_point = reinterpret_cast<const CudaT*>(zero_point->Data<T>());

  // Prepare the output, mask for gradient computation
  auto* fake_quantized_tensor = ctx->Output(0, input_tensor->Shape());
  CudaT* fake_quantized_data = reinterpret_cast<CudaT*>(fake_quantized_tensor->MutableData<T>());
  bool* quantization_mask_data = ctx->Output(1, input_tensor->Shape())->MutableData<bool>();

  // Fake quantize the input tensor
  // TODO(bmeswani): Add support for FakeQuantPerChannel
  FakeQuantPerTensor(Stream(ctx), input_tensor->Shape().Size(), input_data, *quant_scale, *quant_zero_point, quant_min_,
                     quant_max_, fake_quantized_data, quantization_mask_data);

  return Status::OK();
}

#define REGISTER_FAKEQUANTGRAD_KERNEL_TYPED(T)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      FakeQuantGrad,                                              \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      FakeQuantGrad<T>);

REGISTER_FAKEQUANTGRAD_KERNEL_TYPED(float)

template <typename T>
Status FakeQuantGrad<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;

  // Prepare the gradient wrt the output and gradient mask input
  const auto* dY = ctx->Input<Tensor>(0);
  const CudaT* dY_data = reinterpret_cast<const CudaT*>(dY->Data<T>());
  const auto* gradient_mask = ctx->Input<Tensor>(1);
  const bool* gradient_mask_data = gradient_mask->Data<bool>();

  // Prepare the output
  auto* dX = ctx->Output(0, dY->Shape());
  CudaT* dX_data = reinterpret_cast<CudaT*>(dX->MutableData<T>());

  // Compute
  FakeQuantGradImpl(Stream(ctx), dY->Shape().Size(), dY_data, gradient_mask_data, dX_data);

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
