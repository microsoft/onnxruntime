// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention.h"
#include "core/framework/tensorprotoutils.h"
#include "onnx/defs/schema.h"
#include "core/util/eigen_common_wrapper.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/math/gemm_helper.h"
#include "core/providers/cpu/math/softmax.h"
#include "core/providers/cpu/tensor/transpose.h"

namespace onnxruntime {
namespace contrib {
// These ops are internal-only, so register outside of onnx
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Attention,                                                  \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Attention<T>);

REGISTER_KERNEL_TYPED(float)

AttentionBase::AttentionBase(const OpKernelInfo& info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);
}

Status AttentionBase::CheckInputs(const OpKernelContext* context) const {
  // Input and output shapes:
  //   Input 0 - input       : (batch_size, sequence_length, hidden_size)
  //   Input 1 - weights     : (hidden_size, 3 * hidden_size)
  //   Input 2 - bias        : (3 * hidden_size)
  //   Input 3 - mask_index  : (batch_size)
  //   Output                : (batch_size, sequence_length, hidden_size)

  const Tensor* input = context->Input<Tensor>(0);
  const auto dims = input->Shape().GetDims();
  if (dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 0 is expected to have 3 dimensions, got ", dims.size());
  }
  int batch_size = static_cast<int>(dims[0]);
  int hidden_size = static_cast<int>(dims[2]);
  if (hidden_size % num_heads_ != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 0 dimension 2 should be divisiable by value of the num_heads attribute.");
  }

  const Tensor* weights = context->Input<Tensor>(1);
  const auto weights_dims = weights->Shape().GetDims();
  if (weights_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 1 is expected to have 2 dimensions, got ", weights_dims.size());
  }
  if (weights_dims[0] != dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 1 dimension 0 should have same length as dimension 2 of input 0");
  }
  if (weights_dims[1] != 3 * weights_dims[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 1 dimension 1 should be 3 times of dimension 0");
  }

  const Tensor* bias = context->Input<Tensor>(2);
  const auto bias_dims = bias->Shape().GetDims();
  if (bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 2 is expected to have 1 dimension, got ", bias_dims.size());
  }
  if (bias_dims[0] != weights_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 2 dimension 0 should have same length as dimension 1 of input 1");
  }

  const Tensor* mask_index = context->Input<Tensor>(3);
  const auto mask_dims = mask_index->Shape().GetDims();
  if (mask_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 3 is expected to have 1 dimension, got ", mask_dims.size());
  }
  if (static_cast<int>(mask_dims[0]) != batch_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Inputs 3 and 0 shall have same length at dimension 0");
  }

  return Status::OK();
}

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : OpKernel(info), AttentionBase(info) {}

template <typename T>
Status Attention<T>::Compute(OpKernelContext* context) const {
  ORT_RETURN_IF_ERROR(CheckInputs(context));

  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);

  const auto dims = input->Shape().GetDims();
  const int batch_size = static_cast<int>(dims[0]);
  const int sequence_length = static_cast<int>(dims[1]);
  const int hidden_size = static_cast<int>(dims[2]);
  const int head_size = hidden_size / num_heads_;

  TensorShape output_shape(dims);
  Tensor* output = context->Output(0, output_shape);

  constexpr size_t element_size = sizeof(T);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  // STEP.1: gemm_data(BS, 3NH) = input(BS, NH) x weights(NH, 3NH) + bias(3NH)
  auto gemm_data = allocator->Alloc(batch_size * sequence_length * 3 * hidden_size * element_size);
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(allocator));
  auto Q = reinterpret_cast<T*>(gemm_data);
  auto K = Q + batch_size * sequence_length * hidden_size;
  auto V = K + batch_size * sequence_length * hidden_size;

  T* QKV[3] = {Q, K, V};

  {
    const int loop_len = 3 * batch_size * num_heads_;
    const auto input_data = input->template Data<T>();
    const auto weights_data = weights->template Data<T>();
    const auto bias_data = bias->template Data<T>();

    concurrency::ThreadPool::TryParallelFor(context->GetOperatorThreadPool(), loop_len, [&](int32_t i) {
      const int batch_index = (i / 3) / num_heads_;
      const int head_index = (i / 3) % num_heads_;
      const int qkv_index = i % 3;

      int input_offset = batch_index * sequence_length * hidden_size;
      int weights_offset = qkv_index * hidden_size + head_index * head_size;
      T* qkv_dest = QKV[qkv_index];
      int qkv_offset = (batch_index * num_heads_ + head_index) * (sequence_length * head_size);

      // broadcast 3NH -> (3.B.N.S.H)
      const T* broadcast_data_src = bias_data + weights_offset;
      T* broadcast_data_dest = QKV[qkv_index] + qkv_offset;
      for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
        memcpy(broadcast_data_dest, broadcast_data_src, head_size * sizeof(T));
        broadcast_data_dest += head_size;
      }

      //                   original           transposed            iteration
      // A: input          (BxSxNxH)          (B.)S x NH            S x NH
      // B: weights        (NxHx3xNxH)        NH  x (3.N.)H         NH x H
      // C: QKV[qkv_index] (3xBxNxSxH)        (3.B.N.)S x H         S x H

      math::GemmEx<float, concurrency::ThreadPool>(CblasNoTrans,                   // TransA = no
                                                   CblasNoTrans,                   // TransB = no
                                                   sequence_length,                // M      = S
                                                   head_size,                      // N      = H
                                                   hidden_size,                    // K      = NH
                                                   1.0f,                           // alpha
                                                   input_data + input_offset,      // A
                                                   hidden_size,                    // lda    = NH
                                                   weights_data + weights_offset,  // B
                                                   3 * hidden_size,                // ldb    = 3NH
                                                   1.0f,                           // beta
                                                   qkv_dest + qkv_offset,          // C
                                                   head_size,                      // ldc
                                                   nullptr                         // use single-thread
      );
    });
  }

  // STEP.2: scratch(B, N, S, S) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, S, H -> B, N, H, S) + 1 x mask_index(B -> B, 1, 1, 1)
  auto scratch_data = allocator->Alloc(batch_size * num_heads_ * sequence_length * sequence_length * element_size);
  BufferUniquePtr scratch_buffer(scratch_data, BufferDeleter(allocator));

  {
    auto scratch_broadcast_data = allocator->Alloc(batch_size * sequence_length * element_size);
    BufferUniquePtr scratch_broadcast_buffer(scratch_broadcast_data, BufferDeleter(allocator));
    memset(scratch_broadcast_data, 0, batch_size * sequence_length * element_size);
    T* p_scratch_broadcast_current_data = reinterpret_cast<T*>(scratch_broadcast_data);
    for (int b_i = 0; b_i < batch_size; b_i++) {
      // TODO: mask_index can be used in softmax to save some calculation.
      int mask = mask_index->template Data<int32_t>()[b_i];
      for (int m_i = mask; m_i < sequence_length; m_i++) {
        p_scratch_broadcast_current_data[m_i] = static_cast<T>(-10000.0);
      }
      p_scratch_broadcast_current_data += sequence_length;
    }

    const int loop_len = batch_size * num_heads_;
    const float alpha = 1.0f / sqrt(static_cast<float>(head_size));

    concurrency::ThreadPool::TryParallelFor(context->GetOperatorThreadPool(), loop_len, [&](int32_t i) {
      const int batch_index = i / num_heads_;
      // broadcast masks (B) -> (B.N.)S.S
      const T* broadcast_data_src = reinterpret_cast<T*>(scratch_broadcast_data) + batch_index * sequence_length;
      T* broadcast_data_dest = reinterpret_cast<T*>(scratch_data) + sequence_length * sequence_length * i;
      for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
        memcpy(broadcast_data_dest, broadcast_data_src, sequence_length * sizeof(T));
        broadcast_data_dest += sequence_length;
      }

      // gemm

      //                   original           transposed            iteration
      // A: Q              (BxNxSxH)          (B.N.)S x H            S x H
      // B: K'             (BxNxSxH)          (B.N.)H x S            H x S
      // C: scratch_data   (BxNxSxS)          (B.N.)S x S            S x S

      math::Gemm<T, concurrency::ThreadPool>(
          CblasNoTrans,
          CblasTrans,
          sequence_length,
          sequence_length,
          head_size,
          alpha,
          Q + sequence_length * head_size * i,
          K + sequence_length * head_size * i,
          1.0,
          reinterpret_cast<T*>(scratch_data) + sequence_length * sequence_length * i,
          nullptr);
    });
  }

  // STEP.3: P(B, N, S, S) = Softmax(scratch)
  {
    const int N = batch_size * num_heads_ * sequence_length;
    const int D = sequence_length;

    concurrency::ThreadPool::TryBatchParallelFor(context->GetOperatorThreadPool(), N, [&](int j) {
      float* x = reinterpret_cast<T*>(scratch_data) + j * D;
      float* y = x;

      // e^x is represented as infinity if x is large enough, like 100.f.
      // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if one or more item are large enough.
      // a math transform as below is leveraged to get a stable softmax:
      // e^xi/(e^x1 + ...e^xn) = e^(xi - max) / (e^(x1 - max) + ... + e^(xn - max))
      float max = -std::numeric_limits<float>::infinity();
      for (int i = 0; i < D; i++) {
        if (max < x[i]) max = x[i];
      }
      for (int i = 0; i < D; i++) {
        y[i] = expf(x[i] - max);
      }

      double sum = 0.0;

      for (int i = 0; i < D; i++) {
        sum += x[i];
      }

      if (sum == 0) {
        for (int i = 0; i < D; i++) {
          y[i] = 1.0f / (float)D;
        }
      } else {
        for (int i = 0; i < D; i++) {
          y[i] = x[i] / (float)sum;
        }
      }
    });
  }

  // STEP.4: out_tmp(B, N, S, H) = P(B, N, S, S) x V(B, N, S, H)
  auto out_tmp_data = allocator->Alloc(batch_size * num_heads_ * sequence_length * head_size * element_size);
  BufferUniquePtr out_tmp_buffer(out_tmp_data, BufferDeleter(allocator));

  concurrency::ThreadPool::TryParallelFor(context->GetOperatorThreadPool(), batch_size * num_heads_, [&](int i) {
    T* current_tmp_data = reinterpret_cast<T*>(out_tmp_data) + sequence_length * head_size * i;
    math::MatMul<T>(
        sequence_length,
        head_size,
        sequence_length,
        reinterpret_cast<T*>(scratch_data) + sequence_length * sequence_length * i,
        V + sequence_length * head_size * i,
        current_tmp_data,
        nullptr);

    // transpose: out(B, S, N, H) = transpose out_tmp(B, N, S, H)
    const int batch_index = i / num_heads_;
    const int head_index = i % num_heads_;
    T* src = current_tmp_data;
    T* dest = output->template MutableData<T>() + (batch_index * sequence_length * num_heads_ + head_index) * head_size;
    for (int j = 0; j < sequence_length; j++) {
      memcpy(dest, src, head_size * sizeof(T));
      src += head_size;
      dest += hidden_size;
    }
  });

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
