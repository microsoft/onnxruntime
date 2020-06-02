// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention.h"
#include "attention_helper.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"
#include "core/mlas/inc/mlas.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {
// These ops are internal-only, so register outside of onnx
#define REGISTER_KERNEL_TYPED(T)                                                                          \
  ONNX_OPERATOR_TYPED_KERNEL_EX(Attention, kMSDomain, 1, T, kCpuExecutionProvider,                        \
                                KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
                                Attention<T>);

REGISTER_KERNEL_TYPED(float)

AttentionBase::AttentionBase(const OpKernelInfo& info) {
  int64_t num_heads = 0;
  ORT_ENFORCE(info.GetAttr("num_heads", &num_heads).IsOK() && num_heads > 0);
  num_heads_ = static_cast<int>(num_heads);
  is_unidirectional_ = info.GetAttrOrDefault<int64_t>("unidirectional", 0) == 1;
}

Status AttentionBase::CheckInputs(const Tensor* input,
                                  const Tensor* weights,
                                  const Tensor* bias,
                                  const Tensor* mask_index) const {
  // Input and output shapes:
  //   input       : (batch_size, sequence_length, hidden_size)
  //   weights     : (hidden_size, 3 * hidden_size)
  //   bias        : (3 * hidden_size)
  //   mask_index  : (batch_size) if presented

  const auto dims = input->Shape().GetDims();
  if (dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 0 is expected to have 3 dimensions, got ",
                           dims.size());
  }
  int batch_size = static_cast<int>(dims[0]);
  int hidden_size = static_cast<int>(dims[2]);
  if (hidden_size % num_heads_ != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 0 dimension 2 should be divisiable by value of the num_heads attribute.");
  }

  const auto weights_dims = weights->Shape().GetDims();
  if (weights_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 1 is expected to have 2 dimensions, got ",
                           weights_dims.size());
  }
  if (weights_dims[0] != dims[2]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 1 dimension 0 should have same length as dimension 2 of input 0");
  }
  if (weights_dims[1] != 3 * weights_dims[0]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 1 dimension 1 should be 3 times of dimension 0");
  }

  const auto bias_dims = bias->Shape().GetDims();
  if (bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 2 is expected to have 1 dimension, got ",
                           bias_dims.size());
  }
  if (bias_dims[0] != weights_dims[1]) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 2 dimension 0 should have same length as dimension 1 of input 1");
  }

  if (mask_index != nullptr) {  // mask_index is optional
    // unidirectional (like GPT2) does not need mask input. Here we do not allowed the input for unidirectional.
    if (is_unidirectional_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 3 (mask_index) is not allowed for unidirectional");
    }

    const auto mask_dims = mask_index->Shape().GetDims();
    if (mask_dims.size() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 3 is expected to have 1 dimension, got ",
                             mask_dims.size());
    }
    if (static_cast<int>(mask_dims[0]) != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 3 and 0 shall have same length at dimension 0");
    }
  }

  return Status::OK();
}

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : OpKernel(info), AttentionBase(info) {
}

template <typename T>
Status Attention<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* mask_index = context->Input<Tensor>(3);
  ORT_RETURN_IF_ERROR(CheckInputs(input, weights, bias, mask_index));

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

  auto* tp = context->GetOperatorThreadPool();
  // STEP.1: gemm_data(BS, 3NH) = input(BS, NH) x weights(NH, 3NH) + bias(3NH)
  auto gemm_data = allocator->Alloc(SafeInt<size_t>(batch_size) * sequence_length * 3 * hidden_size * element_size);
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

    const double cost =
        static_cast<double>(sequence_length) * static_cast<double>(head_size) * static_cast<double>(hidden_size);
    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int batch_index = static_cast<int>((i / 3) / num_heads_);
        const int head_index = static_cast<int>((i / 3) % num_heads_);
        const int qkv_index = static_cast<int>(i % 3);

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

        math::GemmEx<float, ThreadPool>(CblasNoTrans,                   // TransA = no
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
      }
    });
  }

  // STEP.2: compute the attention score. It does 2 things:
  //         I. attention_probs(B, N, S, S) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, S, H -> B, N, H, S) +
  //                                         1 x mask_data(B, N, S, S)
  //         II.attention_probs(B, N, S, S) = Softmax(attention_probs)
  size_t attention_probs_bytes = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * sequence_length * element_size;
  auto attention_probs = allocator->Alloc(attention_probs_bytes);
  BufferUniquePtr scratch_buffer(attention_probs, BufferDeleter(allocator));

  size_t mask_data_bytes = 0;
  if (mask_index != nullptr) {
    mask_data_bytes = SafeInt<size_t>(batch_size) * sequence_length * sequence_length * element_size;
  } else if (is_unidirectional_) {
    mask_data_bytes = SafeInt<size_t>(sequence_length) * sequence_length * element_size;
  }

  void* mask_data = nullptr;
  if (mask_data_bytes > 0) {
    mask_data = allocator->Alloc(mask_data_bytes);
    memset(mask_data, 0, mask_data_bytes);
  }
  BufferUniquePtr mask_data_buffer(mask_data, BufferDeleter(allocator));

  const int32_t* mask_index_data = mask_index != nullptr ? mask_index->template Data<int32_t>() : nullptr;

  ComputeAttentionProbs<T>(static_cast<T*>(attention_probs), Q, K, mask_index_data, static_cast<T*>(mask_data),
                           batch_size, sequence_length, head_size, num_heads_, is_unidirectional_, tp);

  // STEP.3: compute the attentionScore * Value. It does: out_tmp(B, N, S, H) = attention_probs(B, N, S, S) x V(B, N, S, H)
  auto out_tmp_data =
      allocator->Alloc(SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * head_size * element_size);
  BufferUniquePtr out_tmp_buffer(out_tmp_data, BufferDeleter(allocator));

  ComputeVxAttentionScore(output->template MutableData<T>(), static_cast<T*>(out_tmp_data), static_cast<T*>(attention_probs), V,
                          batch_size, sequence_length, head_size, num_heads_, hidden_size, tp);

  return Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
