// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/activation/activations.h"
#include "contrib_ops/cpu/activations.h"

#include "core/framework/allocator.h"

namespace onnxruntime {
namespace contrib {

ONNX_CPU_OPERATOR_KERNEL(
    ParametricSoftplus,
    1,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ParametricSoftplus<float>);

ONNX_CPU_OPERATOR_KERNEL(
    ScaledTanh,
    1,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ScaledTanh<float>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    ThresholdedRelu,
    1,
    9,
    KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ThresholdedRelu<float>);

// QuickGelu for MLFloat16 is computed in fp32 and converted back to fp16. This keeps the
// Swish/SiLU activation fused into a single kernel (instead of running as separate Sigmoid + Mul
// nodes), which is meaningfully faster on ARMv8.2-A CPUs, while remaining correct on CPUs without
// native fp16 support.
template <>
Status QuickGelu<MLFloat16>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const MLFloat16* input_data = input->Data<MLFloat16>();
  Tensor* output = context->Output(0, input->Shape());
  MLFloat16* output_data = output->MutableData<MLFloat16>();
  concurrency::ThreadPool* tp = context->GetOperatorThreadPool();
  int64_t elem_count = input->Shape().Size();
  if (elem_count == 0) {
    return Status::OK();
  }

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  const size_t count = onnxruntime::narrow<size_t>(elem_count);
  auto input_fp32 = IAllocator::MakeUniquePtr<float>(allocator, count);
  auto output_fp32 = IAllocator::MakeUniquePtr<float>(allocator, count);

  MlasConvertHalfToFloatBufferInParallel(input_data, input_fp32.get(), count, tp);

  const float alpha = alpha_;
  float* input_fp32_data = input_fp32.get();
  float* output_fp32_data = output_fp32.get();
  constexpr int64_t length_per_task = 4096;  // this number comes from FastGelu.
  int64_t task_count = (elem_count + length_per_task - 1) / length_per_task;
  concurrency::ThreadPool::TryBatchParallelFor(
      tp, static_cast<int32_t>(task_count),
      [&](ptrdiff_t task_idx) {
        const auto start = task_idx * length_per_task;
        const float* p_input = input_fp32_data + start;
        float* p_output = output_fp32_data + start;
        int64_t task_elems = std::min(length_per_task, elem_count - start);

        if (alpha == 1.0f) {
          MlasComputeSilu(p_input, p_output, onnxruntime::narrow<size_t>(task_elems));
          return;
        }

        for (int64_t i = 0; i < task_elems; i++) {
          p_output[i] = p_input[i] * alpha;
        }

        MlasComputeLogistic(p_output, p_output, onnxruntime::narrow<size_t>(task_elems));

        MlasEltwiseMul<float>(p_input, p_output, p_output, onnxruntime::narrow<size_t>(task_elems));
      },
      0);

  MlasConvertFloatToHalfBufferInParallel(output_fp32_data, output_data, count, tp);

  return Status::OK();
}

#define REGISTER_QUICKGELU_KERNEL(data_type)                                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                        \
      QuickGelu, kMSDomain, 1, data_type, kCpuExecutionProvider,                        \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      QuickGelu<data_type>);

REGISTER_QUICKGELU_KERNEL(float);
REGISTER_QUICKGELU_KERNEL(MLFloat16);

}  // namespace contrib
}  // namespace onnxruntime
