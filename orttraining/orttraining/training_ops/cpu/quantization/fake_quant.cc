// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cpu/quantization/fake_quant.h"
#include "core/common/narrow.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/element_wise_ops.h"

namespace onnxruntime {
namespace contrib {

namespace {

template <typename T>
void FakeQuantPerTensor(OpKernelContext* ctx, const int64_t num_elements, const T* input_data, T quant_scale,
                        T quant_zero_point, int64_t quant_min, int64_t quant_max, T* fake_quantized_data,
                        bool* quantization_mask_data) {
  const auto zero_point_int = static_cast<int64_t>(quant_zero_point);
  auto* tp = ctx->GetOperatorThreadPool();
  concurrency::ThreadPool::TryParallelFor(
      tp, narrow<ptrdiff_t>(num_elements), /* 1 Read, 2 Writes, 4 Computes */ TensorOpCost{1.0, 2.0, 4.0},
      [quant_scale, zero_point_int, quant_min, quant_max, &input_data, &fake_quantized_data, &quantization_mask_data](
          std::ptrdiff_t begin, std::ptrdiff_t end) {
        for (std::ptrdiff_t index = begin; index != end; ++index) {
          size_t idx = static_cast<size_t>(index);

          // Quantize
          const auto quantized_value = static_cast<int64_t>(std::nearbyint(input_data[idx] / quant_scale)) +
                                       zero_point_int;

          // Clamp and De-Quantize
          fake_quantized_data[idx] =
              (std::min(quant_max, std::max(quant_min, quantized_value)) - zero_point_int) * quant_scale;

          // Compute mask needed for gradient computation
          quantization_mask_data[idx] = (quant_min <= quantized_value && quantized_value <= quant_max);
        }
      });
}

template <typename T>
void FakeQuantGradImpl(const Tensor& dY, const Tensor& gradient_mask, Tensor& dX) {
  // If gradient_mask is true (i.e. quantization was in range), return dY, else return 0
  MakeEigenArrayMap<T>(dX) = MakeEigenArrayMap<T>(dY) * MakeEigenArrayMap<bool>(gradient_mask).template cast<T>();
}

}  // namespace

#define REGISTER_FAKEQUANT_KERNEL_TYPED(T)                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      FakeQuant,                                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      FakeQuant<T>);

REGISTER_FAKEQUANT_KERNEL_TYPED(float)

template <typename T>
Status FakeQuant<T>::Compute(OpKernelContext* ctx) const {
  // Prepare the input, scale, zero point
  const auto* input_tensor = ctx->Input<Tensor>(0);
  const T* input_data = input_tensor->Data<T>();
  const auto* scale = ctx->Input<Tensor>(1);
  ORT_ENFORCE(IsScalarOr1ElementVector(scale), "Quantization scale must be a scalar or 1D tensor of size 1.");
  const T* quant_scale = scale->Data<T>();
  ORT_ENFORCE(*quant_scale != static_cast<T>(0),
              "Quantization scale cannot be 0. It may result in undefined behavior.");
  const auto* zero_point = ctx->Input<Tensor>(2);
  ORT_ENFORCE(IsScalarOr1ElementVector(zero_point), "Quantization zero point must be a scalar or 1D tensor of size 1.");
  const T* quant_zero_point = zero_point->Data<T>();

  // Prepare the output, mask for gradient computation
  auto* fake_quantized_tensor = ctx->Output(0, input_tensor->Shape());
  T* fake_quantized_data = fake_quantized_tensor->MutableData<T>();
  bool* quantization_mask_data = ctx->Output(1, input_tensor->Shape())->MutableData<bool>();

  // Compute
  // TODO(bmeswani): Add support for FakeQuantPerChannel
  FakeQuantPerTensor(ctx, input_tensor->Shape().Size(), input_data, *quant_scale, *quant_zero_point, quant_min_,
                     quant_max_, fake_quantized_data, quantization_mask_data);

  return Status::OK();
}

#define REGISTER_FAKEQUANTGRAD_KERNEL_TYPED(T)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      FakeQuantGrad,                                              \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      (*KernelDefBuilder::Create())                               \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      FakeQuantGrad<T>);

REGISTER_FAKEQUANTGRAD_KERNEL_TYPED(float)

template <typename T>
Status FakeQuantGrad<T>::Compute(OpKernelContext* ctx) const {
  // Prepare the gradient wrt the output and gradient mask input
  const auto* dY = ctx->Input<Tensor>(0);
  const auto* gradient_mask = ctx->Input<Tensor>(1);

  // Prepare the output
  auto* dX = ctx->Output(0, dY->Shape());

  // Compute
  FakeQuantGradImpl<T>(*dY, *gradient_mask, *dX);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
