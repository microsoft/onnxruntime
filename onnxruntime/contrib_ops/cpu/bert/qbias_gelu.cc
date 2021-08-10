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
                         const quantization::Params<T>& input_params,
                         const quantization::Params<T>& bias_params,
                         float* temp,
                         float* output,
                         int64_t count) {
  for (int64_t i = 0; i < count; ++i) {
    // TODO(kreeger): Go back and update lookup tables.
    // TODO(kreeger): Consider writing a MlasCompteErf() for uint8/int8.
    //T value = input[i] + bias[i];
    float value = quantization::Dequantize<T>(input[i], input_params) +
                  quantization::Dequantize<T>(bias[i], bias_params);
    output[i] = value * static_cast<T>(M_SQRT1_2);
    temp[i] = value * 0.5f;
  }

  // TODO  - looks like there is not a uint8 version of this?
  MlasComputeErf(output, output, count);

  for (int64_t i = 0; i < count; i++) {
    // TODO(kreeger): When/how/where do I want to handle re-quantization?
    // TODO(kreeger): Double check the graph.
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
  float* output_data = output->template MutableData<float>();

  int64_t element_length = input_quantized->Shape().Size();
  const T* input_data = input_quantized->template Data<T>();

  quantization::Params<T> input_params =
    quantization::GetTensorQuantizationParams<T>(
          input_scale, input_zero_point);

  if (bias_quantized == nullptr) {
    // TODO(kreeger): handle bias conditionally here and checkinputs().
    // TODO(kreeger): add a unit test for this.
    return Status::OK();
  }

  const T* bias_data = bias_quantized->template Data<T>();
  int64_t bias_length = bias_quantized->Shape().Size();

  quantization::Params<T> bias_params =
      quantization::GetTensorQuantizationParams<T>(
          bias_scale, bias_zero_point);

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  BufferUniquePtr buffer = BufferUniquePtr(
      alloc->Alloc(SafeInt<size_t>(sizeof(float)) * element_length),
      BufferDeleter(alloc));

  float* tmp_data = static_cast<float*>(buffer.get());

  int64_t task_count = element_length / bias_length;

  concurrency::ThreadPool::TryBatchParallelFor(
      context->GetOperatorThreadPool(), static_cast<int32_t>(task_count),
      [&](ptrdiff_t task_idx) {
        const T* p_input = input_data + task_idx * bias_length;
        float* p_output = output_data + task_idx * bias_length;
        float* p_tmp = tmp_data + task_idx * bias_length;

        AddBiasGeluNoApprox<T>(p_input,
                               bias_data,
                               input_params,
                               bias_params,
                               p_tmp,
                               p_output,
                               bias_length);
      },
      0);

  return Status::OK();
}

}  // namespace

// This op is internal-only, so register outside of onnx:
ONNX_OPERATOR_TYPED_KERNEL_EX(
    QBiasGelu,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>()),
    QBiasGelu);

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
