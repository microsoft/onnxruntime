// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "rel_partial_learnable_attention_cpu_base.h"
#include "rel_partial_learnable_attention_helper.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

template <typename T>
class RelPartialLearnableAttention : public OpKernel, public RelPartialLearnableAttentionCPUBase {
 public:
  explicit RelPartialLearnableAttention(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;
};

// These ops are internal-only, so register outside of onnx
ONNX_OPERATOR_TYPED_KERNEL_EX(
    RelPartialLearnableAttention,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    RelPartialLearnableAttention<float>);

Status RelPartialLearnableAttentionBase::CheckInputs(const TensorShape& input_shape,
                                                     const TensorShape& pos_emb_shape,
                                                     const TensorShape& u_shape,
                                                     const TensorShape& v_shape,
                                                     const Tensor*& attn_mask,
                                                     const Tensor*& mems) const {
  // Input shapes:
  //   input       : (batch_size, sequence_length, d_model)
  //   pos_emb     : (batch_size, sequence_length, d_model)
  //   u           : (num_heads, head_size)
  //   v           : (batch_size, sequence_length, d_model)
  //   attn_mask   : nullptr, (sequence_length, sequence_length)
  //   mems        : nullptr

  const auto& dims = input_shape.GetDims();
  if (dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'input' is expected to have 3 dimensions, got ",
                           dims.size());
  }
  int batch_size = static_cast<int>(dims[0]);
  int sequence_length = static_cast<int>(dims[1]);
  int d_model = static_cast<int>(dims[2]);

  const auto& pos_emb_dims = pos_emb_shape.GetDims();
  if (pos_emb_dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'pos_emb' is expected to have 3 dimensions, got ",
                           pos_emb_dims.size());
  }
  if (static_cast<int>(pos_emb_dims[1]) != sequence_length) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'pos_emb' dimension 0 shall have same length as dimension 1 of input 0");
  }
  if (static_cast<int>(pos_emb_dims[2]) != d_model) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'pos_emb' dimension 0 shall have same length as dimension 2 of input 0");
  }

  const auto& u_dims = u_shape.GetDims();
  if (u_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'u' is expected to have 2 dimensions, got ",
                           u_dims.size());
  }
  if (static_cast<int>(u_dims[0]) != num_heads_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'u' dimension 0 shall have same length as num_heads");
  }
  if (static_cast<int>(u_dims[1]) != head_size_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'u' dimension 1 shall have same length as head_size");
  }

  const auto& v_dims = v_shape.GetDims();
  if (v_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'v' is expected to have 2 dimensions, got ",
                           v_dims.size());
  }
  if (static_cast<int>(v_dims[0]) != num_heads_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'v' dimension 0 shall have same length as num_heads");
  }
  if (static_cast<int>(v_dims[1]) != head_size_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'v' dimension 1 shall have same length as head_size");
  }

  if (attn_mask != nullptr) {  // attn_mask is optional
    const auto& attn_mask_dims = attn_mask->Shape().GetDims();
    if (attn_mask_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'attn_mask' is expected to have 2 dimensions, got ",
                             attn_mask_dims.size());
    }
    if (static_cast<int>(attn_mask_dims[0]) != sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'attn_mask' dimension 0 shall have same length as dimension 1 of input 0");
    }
    if (static_cast<int>(attn_mask_dims[1]) != sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'attn_mask' dimension 1 shall have same length as dimension 1 of input 0");
    }
  }

  if (mems != nullptr) {  // mems is optional
  }

  return Status::OK();
}

Status RelPartialLearnableAttentionBase::CheckInputs(const TensorShape& input_shape,
                                                     const TensorShape& pos_emb_shape,
                                                     const TensorShape& u_shape,
                                                     const TensorShape& v_shape,
                                                     const Tensor*& attn_mask,
                                                     const Tensor*& mems,
                                                     const int max_threads_per_block) const {
  if (num_heads_ > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
  }

  return CheckInputs(input_shape, pos_emb_shape, u_shape, v_shape, attn_mask, mems);
}

template <typename T>
RelPartialLearnableAttention<T>::RelPartialLearnableAttention(const OpKernelInfo& info) : OpKernel(info), RelPartialLearnableAttentionCPUBase(info) {
}

template <typename T>
Status RelPartialLearnableAttention<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* input_weight = context->Input<Tensor>(1);
  const Tensor* pos_emb = context->Input<Tensor>(2);
  const Tensor* pos_emb_weight = context->Input<Tensor>(3);
  const Tensor* u = context->Input<Tensor>(4);
  const Tensor* v = context->Input<Tensor>(5);
  const Tensor* output_weight = context->Input<Tensor>(6);

  const Tensor* attn_mask = context->Input<Tensor>(7);
  const Tensor* mems = context->Input<Tensor>(8);

  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  pos_emb->Shape(),
                                  u->Shape(),
                                  v->Shape(),
                                  attn_mask,
                                  mems));

  const auto& shape = input->Shape().GetDims();
  const int batch_size = static_cast<int>(shape[0]);
  const int sequence_length = static_cast<int>(shape[1]);
  const int d_model = static_cast<int>(shape[2]);

  std::vector<int64_t> output_shape(3);
  output_shape[0] = shape[0];
  output_shape[1] = shape[1];
  output_shape[2] = shape[2];
  Tensor* output = context->Output(0, output_shape);

  constexpr size_t element_size = sizeof(T);

  int q_hidden_size = 0;
  int k_hidden_size = 0;
  int v_hidden_size = 0;
  if (qkv_hidden_sizes_.size() == 0) {
    q_hidden_size = hidden_size;
    k_hidden_size = hidden_size;
    v_hidden_size = hidden_size;
  } else {
    q_hidden_size = static_cast<int>(qkv_hidden_sizes_[0]);
    k_hidden_size = static_cast<int>(qkv_hidden_sizes_[1]);
    v_hidden_size = static_cast<int>(qkv_hidden_sizes_[2]);
  }
  const int qkv_head_size[3] = {q_hidden_size / num_heads_, k_hidden_size / num_heads_, v_hidden_size / num_heads_};

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  auto* tp = context->GetOperatorThreadPool();
  // Compute Q, K, V
  // gemm_data(BS, NT) = input(BS, D) x weights(D, NT) + bias(NT)
  // D (input_hidden_size) is hidden dimension of input, where D could be larger than any of the hidden_sizes
  // (NH) when model is pruned. T = H1 + H2 + H3, where H1, H2, H3 are head sizes of Q, K, V respectively
  auto gemm_data = allocator->Alloc(SafeInt<size_t>(batch_size) * sequence_length * (q_hidden_size + k_hidden_size + v_hidden_size) * element_size);
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(allocator));

  auto Q = reinterpret_cast<T*>(gemm_data);
  auto K = Q + static_cast<size_t>(batch_size) * sequence_length * q_hidden_size;
  auto V = K + static_cast<size_t>(batch_size) * sequence_length * k_hidden_size;

  T* QKV[3] = {Q, K, V};

  {
    const int loop_len = 3 * batch_size * num_heads_;
    const auto* input_data = input->template Data<T>();
    const auto* weights_data = weights ? weights->template Data<T>() : nullptr;
    const auto* bias_data = bias->template Data<T>();

    const double cost =
        static_cast<double>(sequence_length) * static_cast<double>(head_size) * static_cast<double>(input_hidden_size);
    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int batch_index = static_cast<int>((i / 3) / num_heads_);
        const int head_index = static_cast<int>((i / 3) % num_heads_);
        const int qkv_index = static_cast<int>(i % 3);

        int input_offset = batch_index * sequence_length * input_hidden_size;

        T* qkv_dest = QKV[qkv_index];
        int head_size = qkv_head_size[qkv_index];
        int weights_offset = 0;
        int bias_offset = qkv_index * q_hidden_size + head_index * head_size;

        if (!is_prepack_) {
          weights_offset = bias_offset;
        } else {
          weights_offset = head_index * head_size;
        }

        int qkv_offset = (batch_index * num_heads_ + head_index) * (sequence_length * head_size);

        // TODO!! memcpy here makes it not worthwhile to use Gemm batch. Possible to post process?
        // broadcast NH -> (B.N.S.H) for each of Q, K, V
        const T* broadcast_data_src = bias_data + bias_offset;
        T* broadcast_data_dest = QKV[qkv_index] + qkv_offset;

        for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
          memcpy(broadcast_data_dest, broadcast_data_src, head_size * sizeof(T));
          broadcast_data_dest += head_size;
        }

        //                   original           transposed            iteration
        // A: input          (BxSxD)            (B.)S x D             S x D
        // B: weights        (DxNxT)             D x (N.)T            D x H
        // C: QKV[qkv_index] (BxNxSxT)          (B.N.)S x T           S x H
        if (is_prepack_) {
          uint8_t* packed_weight;
          packed_weight = static_cast<uint8_t*>(packed_weights_[qkv_index].get()) + packed_weights_size_[qkv_index] * (weights_offset / head_size);

          MlasGemm(
              CblasNoTrans,               // TransA = no
              sequence_length,            // M      = S
              head_size,                  // N      = H
              input_hidden_size,          // K      = D
              1.0f,                       // alpha
              input_data + input_offset,  // A
              input_hidden_size,          // lda    = D
              packed_weight,              // B
              1.0f,                       // beta
              qkv_dest + qkv_offset,      // C
              head_size,                  // ldc
              nullptr);                   // use single-thread
        } else {
          math::GemmEx<float, ThreadPool>(
              CblasNoTrans,                                   // TransA = no
              CblasNoTrans,                                   // TransB = no
              sequence_length,                                // M      = S
              head_size,                                      // N      = H
              input_hidden_size,                              // K      = D
              1.0f,                                           // alpha
              input_data + input_offset,                      // A
              input_hidden_size,                              // lda    = D
              weights_data + weights_offset,                  // B
              q_hidden_size + k_hidden_size + v_hidden_size,  // ldb = NH1 + NH2 + NH3
              1.0f,                                           // beta
              qkv_dest + qkv_offset,                          // C
              head_size,                                      // ldc
              nullptr                                         // use single-thread
          );
        }
      }
    });
  }

  // Compute the attention score and apply the score to V
  return ApplyRelPartialLearnableAttention(Q, K, V, mask_index, past, output,
                                           batch_size, sequence_length,
                                           qkv_head_size[0], qkv_head_size[2], v_hidden_size,
                                           extra_add_qk, context);
}
}  // namespace contrib
}  // namespace onnxruntime
