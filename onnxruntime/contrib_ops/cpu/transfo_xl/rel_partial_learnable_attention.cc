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
ONNX_OPERATOR_KERNEL_EX(
    RelPartialLearnableAttention,
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    RelPartialLearnableAttention<float>);

Status RelPartialLearnableAttentionBase::CheckInputs(const TensorShape& input_shape,
                                                     const TensorShape& input_weights_shape,
                                                     const TensorShape& pos_emb_shape,
                                                     const TensorShape& pos_emb_weights_shape,
                                                     const TensorShape& r_w_bias_shape,
                                                     const TensorShape& r_r_bias_shape,
                                                     const TensorShape& output_weights_shape,
                                                     const Tensor*& attn_mask,
                                                     const Tensor*& mems) const {
  // Input shapes:
  //   input               : (batch_size, sequence_length, d_model)
  //   input_weights       : (d_model, 3 * num_heads * head_size)
  //   pos_emb             : (batch_size, sequence_length, d_model)
  //   pos_emb_weights     : (d_model, num_heads * head_size)
  //   r_w_bias            : (num_heads, head_size)
  //   r_r_bias            : (num_heads, head_size)
  //   output_weights      : (num_heads * head_size, d_model)
  //   attn_mask           : nullptr, (sequence_length, sequence_length)
  //   mems                : nullptr, (batch_size, sequence_length + memory_length, d_model)

  const auto& dims = input_shape.GetDims();
  if (dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'input' is expected to have 3 dimensions, got ",
                           dims.size());
  }
  int batch_size = static_cast<int>(dims[0]);
  int sequence_length = static_cast<int>(dims[1]);
  int d_model = static_cast<int>(dims[2]);

  const auto& input_weights_dims = input_weights_shape.GetDims();
  if (input_weights_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'input_weights' is expected to have 2 dimensions, got ",
                           input_weights_dims.size());
  }
  if (static_cast<int>(input_weights_dims[0]) != d_model) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'input_weights' dimension 0 shall have same length as dimension 2 of input 0");
  }
  if (static_cast<int>(input_weights_dims[1]) != 3 * num_heads_ * head_size_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'input_weights' dimension 1 shall have same length as 3 * num_heads * head_size");
  }

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

  const auto& pos_emb_weights_dims = pos_emb_weights_shape.GetDims();
  if (pos_emb_weights_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'pos_emb_weights' is expected to have 2 dimensions, got ",
                           pos_emb_weights_dims.size());
  }
  if (static_cast<int>(pos_emb_weights_dims[0]) != d_model) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'pos_emb_weights' dimension 0 shall have same length as dimension 2 of input 0");
  }
  if (static_cast<int>(pos_emb_weights_dims[1]) != num_heads_ * head_size_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'pos_emb_weights' dimension 1 shall have same length as num_heads * head_size");
  }

  const auto& r_w_bias_dims = r_w_bias_shape.GetDims();
  if (r_w_bias_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'r_w_bias' is expected to have 2 dimensions, got ",
                           r_w_bias_dims.size());
  }
  if (static_cast<int>(r_w_bias_dims[0]) != num_heads_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'r_w_bias' dimension 0 shall have same length as num_heads");
  }
  if (static_cast<int>(r_w_bias_dims[1]) != head_size_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'r_w_bias' dimension 1 shall have same length as head_size");
  }

  const auto& r_r_bias_dims = r_r_bias_shape.GetDims();
  if (r_r_bias_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'r_r_bias' is expected to have 2 dimensions, got ",
                           r_r_bias_dims.size());
  }
  if (static_cast<int>(r_r_bias_dims[0]) != num_heads_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'r_r_bias' dimension 0 shall have same length as num_heads");
  }
  if (static_cast<int>(r_r_bias_dims[1]) != head_size_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'r_r_bias' dimension 1 shall have same length as head_size");
  }

  const auto& output_weights_dims = output_weights_shape.GetDims();
  if (output_weights_dims.size() != 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'output_weights' is expected to have 2 dimensions, got ",
                           output_weights_dims.size());
  }
  if (static_cast<int>(output_weights_dims[0]) != num_heads_ * head_size_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'output_weights' dimension 0 shall have same length as num_heads * head_size");
  }
  if (static_cast<int>(output_weights_dims[1]) != d_model) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'output_weights' dimension 1 shall have same length as dimension 2 of input 0");
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
    const auto& mems_dims = mems->Shape().GetDims();
    if (mems_dims.size() != 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'mems' is expected to have 3 dimensions, got ",
                             mems_dims.size());
    }
    if (static_cast<int>(mems_dims[0]) != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'mems' dimension 0 shall have same length as dimension 0 of input 0");
    }
  }

  return Status::OK();
}

Status RelPartialLearnableAttentionBase::CheckInputs(const TensorShape& input_shape,
                                                     const TensorShape& input_weights_shape,
                                                     const TensorShape& pos_emb_shape,
                                                     const TensorShape& pos_emb_weights_shape,
                                                     const TensorShape& r_w_bias_shape,
                                                     const TensorShape& r_r_bias_shape,
                                                     const TensorShape& output_weights_shape,
                                                     const Tensor*& attn_mask,
                                                     const Tensor*& mems,
                                                     const int max_threads_per_block) const {
  if (num_heads_ > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
  }

  return CheckInputs(input_shape, input_weights_shape, pos_emb_shape, pos_emb_weights_shape, r_w_bias_shape, r_r_bias_shape, output_weights_shape, attn_mask, mems);
}

template <typename T>
RelPartialLearnableAttention<T>::RelPartialLearnableAttention(const OpKernelInfo& info) : OpKernel(info), RelPartialLearnableAttentionCPUBase(info) {
}

template <typename T>
Status RelPartialLearnableAttention<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* input_weights = context->Input<Tensor>(1);
  const Tensor* pos_emb = context->Input<Tensor>(2);
  const Tensor* pos_emb_weights = context->Input<Tensor>(3);
  const Tensor* r_w_bias = context->Input<Tensor>(4);
  const Tensor* r_r_bias = context->Input<Tensor>(5);
  const Tensor* output_weights = context->Input<Tensor>(6);

  const Tensor* attn_mask = context->Input<Tensor>(7);
  const Tensor* mems = context->Input<Tensor>(8);

  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  input_weights->Shape(),
                                  pos_emb->Shape(),
                                  pos_emb_weights->Shape(),
                                  r_w_bias->Shape(),
                                  r_r_bias->Shape(),
                                  output_weights->Shape(),
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

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  auto* tp = context->GetOperatorThreadPool();
  // Compute Q, K, V
  // gemm_data(BS, NT) = input(BS, D) x weights(D, NT)
  // D (d_model) is hidden dimension of input
  // NT stands for 3 * num_heads * head_size
  auto gemm_data = allocator->Alloc(SafeInt<size_t>(batch_size) * sequence_length * (3 * d_model) * element_size);
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(allocator));

  auto Q = reinterpret_cast<T*>(gemm_data);
  auto K = Q + static_cast<size_t>(batch_size) * sequence_length * d_model;
  auto V = K + static_cast<size_t>(batch_size) * sequence_length * d_model;

  T* QKV[3] = {Q, K, V};

  {
    const int loop_len = 3 * batch_size * num_heads_;
    const auto* input_data = input->template Data<T>();
    const auto* input_weights_data = input_weights->template Data<T>();

    const double cost =
        static_cast<double>(sequence_length) * static_cast<double>(head_size_) * static_cast<double>(d_model);
    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int batch_index = static_cast<int>((i / 3) / num_heads_);
        const int head_index = static_cast<int>((i / 3) % num_heads_);
        const int qkv_index = static_cast<int>(i % 3);

        int input_offset = batch_index * sequence_length * d_model;

        T* qkv_dest = QKV[qkv_index];
        int weights_offset = 0;
        int bias_offset = qkv_index * d_model + head_index * head_size_;

        weights_offset = bias_offset;

        int qkv_offset = (batch_index * num_heads_ + head_index) * (sequence_length * head_size_);

        // TODO!! memcpy here makes it not worthwhile to use Gemm batch. Possible to post process?
        // broadcast NH -> (B.N.S.H) for each of Q, K, V
        // const T* broadcast_data_src = bias_offset;
        // T* broadcast_data_dest = QKV[qkv_index] + qkv_offset;

        // for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
        //   memcpy(broadcast_data_dest, broadcast_data_src, head_size_ * sizeof(T));
        //   broadcast_data_dest += head_size_;
        // }

        //                   original           transposed            iteration
        // A: input          (BxSxD)            (B.)S x D             S x D
        // B: weights        (DxNxT)             D x (N.)T            D x H
        // C: QKV[qkv_index] (BxNxSxT)          (B.N.)S x T           S x H
        math::GemmEx<float, ThreadPool>(
            CblasNoTrans,                                   // TransA = no
            CblasNoTrans,                                   // TransB = no
            sequence_length,                                // M      = S
            head_size_,                                      // N      = H
            d_model,                              // K      = D
            1.0f,                                           // alpha
            input_data + input_offset,                      // A
            d_model,                              // lda    = D
            input_weights_data + weights_offset,                  // B
            d_model + d_model + d_model,  // ldb = NH1 + NH2 + NH3
            1.0f,                                           // beta
            qkv_dest + qkv_offset,                          // C
            head_size_,                                      // ldc
            nullptr                                         // use single-thread
        );
      }
    });
  }

  // Compute the attention score and apply the score to V
  return ApplyRelPartialLearnableAttention(Q, K, V, pos_emb, pos_emb_weights, r_w_bias, r_r_bias,
                                           output_weights, attn_mask, mems, output,
                                           batch_size, sequence_length, d_model,
                                           num_heads_, head_size_, context);
}
}  // namespace contrib
}  // namespace onnxruntime
