// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_quant.h"
#include "contrib_ops/cpu/bert/attention_helper.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#include "core/util/qmath.h"
#include "core/util/math_cpuonly.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"
#include "core/mlas/inc/mlas.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {
// These ops are internal-only, so register outside of onnx
#define REGISTER_KERNEL_TYPED(T, QInput, QWeight)                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      QAttention,                                                        \
      kMSDomain,                                                         \
      1,                                                                 \
      T##_##QInput##_##QWeight,                                          \
      kCpuExecutionProvider,                                             \
      KernelDefBuilder()                                                 \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<QInput>())   \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<QWeight>())  \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()), \
      QAttention<T, QInput, QWeight>);

REGISTER_KERNEL_TYPED(float, uint8_t, int8_t)
REGISTER_KERNEL_TYPED(float, uint8_t, uint8_t)

template <typename T, typename QInput, typename QWeight>
QAttention<T, QInput, QWeight>::QAttention(const OpKernelInfo& info) : OpKernel(info), AttentionCPUBase(info) {
}

template <typename T, typename QInput, typename QWeight>
Status QAttention<T, QInput, QWeight>::Compute(OpKernelContext* context) const {
  // Input and output shapes:
  //   Input  0 - input             : (batch_size, sequence_length, hidden_size)
  //   Input  1 - weights           : (hidden_size, 3 * hidden_size)
  //   Input  2 - bias              : (3 * hidden_size)
  //   Input  3 - input_scale       : scalar
  //   Input  4 - weight_scale      : scalar
  //   Input  5 - mask_index        : (batch_size)
  //   Input  6 - input_zero_point  : scalar
  //   Input  7 - weight_zero_point : scalar
  //   Input  8 - past              : (2, batch_size, num_heads, past_sequence_length, head_size)
  //   Output 0                     : (batch_size, sequence_length, hidden_size)
  //   Output 1 - present           : (2, batch_size, num_heads, past_sequence_length + sequence_length, head_size)
  //   ORT_RETURN_IF_ERROR(CheckInputs(context));
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* input_scale_tensor = context->Input<Tensor>(3);
  const Tensor* weight_scale_tensor = context->Input<Tensor>(4);
  const Tensor* mask_index = context->Input<Tensor>(5);
  const Tensor* i_zp_tensor = context->Input<Tensor>(6);
  const Tensor* w_zp_tensor = context->Input<Tensor>(7);
  const Tensor* past_tensor = context->Input<Tensor>(8);

  ORT_RETURN_IF_ERROR(AttentionBase::CheckInputs(input, weights, bias, mask_index, past_tensor));

  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(input_scale_tensor),
                    "input scale must be a scalar or 1D tensor of size 1");
  T input_scale = *(input_scale_tensor->template Data<T>());

  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(weight_scale_tensor),
                    "weight must be a scalar or 1D tensor of size 1");
  T weight_scale = *(weight_scale_tensor->template Data<T>());

  T dequant_scale = input_scale * weight_scale;

  QInput input_zero_point = 0;
  if (i_zp_tensor != nullptr) {
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(i_zp_tensor),
                      "input zero point must be a scalar or 1D tensor of size 1.");
    input_zero_point = *i_zp_tensor->template Data<QInput>();
  }

  QWeight weight_zero_point = 0;
  if (w_zp_tensor != nullptr) {
    // CUDA only support symmetric quantization for Attention
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(w_zp_tensor),
                      "weight zero point must be a scalar or 1D tensor of size 1.");
    weight_zero_point = *w_zp_tensor->template Data<QWeight>();
  }

  const auto& shape = input->Shape();
  const int batch_size = static_cast<int>(shape[0]);
  const int sequence_length = static_cast<int>(shape[1]);
  const int hidden_size = static_cast<int>(shape[2]);
  const int head_size = hidden_size / num_heads_;

  Tensor* output = context->Output(0, shape);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  constexpr size_t element_size = sizeof(T);

  auto* tp = context->GetOperatorThreadPool();
  // STEP.1: gemm_data(BS, 3NH) = Scale(input(BS, NH) x weights(NH, 3NH)) + bias(3NH)
  auto gemm_data = allocator->Alloc(SafeInt<size_t>(batch_size) * sequence_length * 3 * hidden_size * element_size);
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(allocator));

  auto Q = reinterpret_cast<T*>(gemm_data);
  auto K = Q + batch_size * sequence_length * hidden_size;
  auto V = K + batch_size * sequence_length * hidden_size;
  T* QKV[3] = {Q, K, V};

  {
    const int loop_len = 3 * batch_size * num_heads_;
    const auto input_data = input->template Data<QInput>();
    const auto weights_data = weights->template Data<QWeight>();
    const auto bias_data = bias->template Data<T>();

    const double cost =
        static_cast<double>(sequence_length) * static_cast<double>(head_size) * static_cast<double>(hidden_size);
    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int batch_index = static_cast<int>((i / 3) / num_heads_);
        const int head_index = static_cast<int>((i / 3) % num_heads_);
        const int qkv_index = static_cast<int>(i % 3);

        int input_offset = batch_index * sequence_length * hidden_size;
        int weights_offset = qkv_index * hidden_size + head_index * head_size;
        float* qkv_dest = QKV[qkv_index];
        int qkv_offset = (batch_index * num_heads_ + head_index) * (sequence_length * head_size);

        //                   original           transposed            iteration
        // A: input          (BxSxNxH)          (B.)S x NH            S x NH
        // B: weights        (NxHx3xNxH)        NH  x (3.N.)H         NH x H
        // C: QKV[qkv_index] (3xBxNxSxH)        (3.B.N.)S x H         S x H
        QGemm(sequence_length,                // M      = S
              head_size,                      // N      = H
              hidden_size,                    // K      = NH
              input_data + input_offset,      // A
              hidden_size,                    // lda    = NH
              input_zero_point,               // input zero point
              weights_data + weights_offset,  // B
              3 * hidden_size,                // ldb    = 3NH
              weight_zero_point,              // weight zero point
              qkv_dest + qkv_offset,          // C
              head_size,                      // ldc
              &dequant_scale,                 // output scale
              bias_data + weights_offset,     // bias
              nullptr                         // use single-thread
        );
      }
    });
  }

  // Compute the attention score and apply the score to V
  return ApplyAttention(Q, K, V, mask_index, past_tensor, output,
                        batch_size, sequence_length,
                        head_size, hidden_size, context);
}

}  // namespace contrib
}  // namespace onnxruntime
