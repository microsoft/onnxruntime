// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qbias_gelu.h"

#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"
#include "core/quantization/quantization.h"

namespace onnxruntime {
namespace contrib {

#pragma warning(disable : 4189 4100)
namespace {

Status CheckQuantizedInputs(OpKernelContext* context, bool* is_signed_inputs) {
  const Tensor* input_quantized_tensor = context->Input<Tensor>(0);
  const Tensor* bias_quantized_tensor = context->Input<Tensor>(1);
  const Tensor* input_scale_tensor = context->Input<Tensor>(2);
  const Tensor* bias_scale_tensor = context->Input<Tensor>(3);
  const Tensor* input_zero_point_tensor = context->Input<Tensor>(4);
  const Tensor* bias_zero_point_tensor = context->Input<Tensor>(5);

  *is_signed_inputs = input_quantized_tensor->IsDataType<int8_t>();

  if (!IsScalarOr1ElementVector(input_scale_tensor)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input scale must be a scalar or 1D tensor of size 1");
  }
  if (!IsScalarOr1ElementVector(bias_scale_tensor)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Bias scale must be a scalar or 1D tensor of size 1");
  }
  if (!IsScalarOr1ElementVector(input_zero_point_tensor) &&
      input_zero_point_tensor->IsDataType<int8_t>() == *is_signed_inputs) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input zero point must be a scalar or 1D tensor of size 1");
  }
  if (!IsScalarOr1ElementVector(bias_zero_point_tensor) &&
      bias_zero_point_tensor->IsDataType<int8_t>() == *is_signed_inputs) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Bias zero point must be a scalar or 1D tensor of size 1");
  }

  return Status::OK();
}

// TODO(kreeger): Add support for no-approx...
template <typename T>
void AddBiasGeluNoApprox(const T* input,
                         const T* bias,
                         T* temp,
                         T* output,
                         int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    T value = input[i] + bias[i];
    output[i] = value * static_cast<T>(M_SQRT1_2);
    temp[i] = value * 0.5f;
  }

  // TODO  - looks like there is not a uint8 version of this?
  MlasComputeErf(output, output, count);

  for (int64_t i = 0; i < count; i++) {
    output[i] = temp[i] * (output[i] + 1.0f);
  }
}

template <typename T>
Status ComputeInternal(OpKernelContext* context) {
  const Tensor* input_quantized = context->Input<Tensor>(0);
  const Tensor* bias_quantized = context->Input<Tensor>(1);
  const Tensor* input_scale = context->Input<Tensor>(2);
  const Tensor* bias_scale = context->Input<Tensor>(3);
  const Tensor* input_zero_point = context->Input<Tensor>(4);
  const Tensor* bias_zero_point = context->Input<Tensor>(5);

  Tensor* output = context->Output(0, input_quantized->Shape());
  T* output_data = output->template MutableData<T>();

  int64_t element_length = input_quantized->Shape().Size();
  const T* input_data = input_quantized->template Data<T>();

  if (bias_quantized == nullptr) {
    // TODO(kreeger): handle bias conditionally here and checkinputs().
    // TODO(kreeger): add a unit test for this.
    return Status::OK();
  }

  const T* bias_data = bias_quantized->template Data<T>();
  int64_t bias_length = bias_quantized->Shape().Size();

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  BufferUniquePtr buffer = BufferUniquePtr(
      alloc->Alloc(SafeInt<size_t>(sizeof(T)) * element_length),
      BufferDeleter(alloc));

  T* tmp_data = static_cast<T*>(buffer.get());

  int64_t task_count = element_length / bias_length;

  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), static_cast<int32_t>(task_count),
      [&](ptrdiff_t task_idx) {
        const T* p_input = input_data + task_idx * bias_length;
        T* p_output = output_data + task_idx * bias_length;
        T* p_tmp = tmp_data + task_idx * bias_length;

        // Hack for now - dequant the list of output

        AddBiasGeluNoApprox(p_input, bias_data, p_tmp, p_output, bias_length);
      },
      0);

  return Status::OK();
}

}  // namespace

// This op is internal-only, so register outside of onnx:
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      QBiasGelu,                                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      QBiasGelu);
REGISTER_KERNEL_TYPED(uint8_t)

QBiasGelu::QBiasGelu(const OpKernelInfo& op_kernel_info)
    : OpKernel(op_kernel_info) {
}

Status QBiasGelu::Compute(OpKernelContext* context) const {
  bool is_signed_inputs = false;
  ORT_RETURN_IF_ERROR(CheckQuantizedInputs(context, &is_signed_inputs));

  if (is_signed_inputs) {
    return ComputeInternal<int8_t>(context);
  } else {
    return ComputeInternal<uint8_t>(context);
  }
}

#pragma warning(default: 4189 4100)

}  // namespace contrib
}  // namespace onnxruntime
