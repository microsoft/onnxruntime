// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_cpu_base.h"
#include "attention_helper.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"

using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {

static void FreePackedWeights(BufferUniquePtr* array, size_t array_size) {
  for (size_t i = 0; i < array_size; i++) {
    array[i].reset();
  }
}

template <typename T>
class Attention : public OpKernel, public AttentionCPUBase {
 public:
  explicit Attention(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 /*out*/ bool& is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

 private:
  bool IsPackWeightsSuccessful(int qkv_index, AllocatorPtr alloc, size_t head_size,
                               size_t input_hidden_size, const T* weights_data,
                               size_t weight_matrix_col_size, PrePackedWeights* prepacked_weights);

  BufferUniquePtr packed_weights_[3];
  size_t packed_weights_size_[3] = {0, 0, 0};
  bool is_prepack_ = false;
  TensorShape weight_shape_;
};

// These ops are internal-only, so register outside of onnx
ONNX_OPERATOR_TYPED_KERNEL_EX(
    Attention,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Attention<float>);

Status AttentionBase::CheckInputs(const TensorShape& input_shape,
                                  const TensorShape* weights_shape,
                                  const TensorShape& bias_shape,
                                  const Tensor*& mask_index,
                                  const Tensor* past,
                                  const Tensor* extra_add_qk,
                                  const Tensor* key,
                                  const Tensor* value,
                                  void* parameters) const {
  // Abbreviation and Meanings:
  //   B:    batch_size
  //   S:    sequence_length (input sequence length of query)
  //   P:    past_sequence_length (past sequence length of key or value)
  //   L:    kv_sequence_length (input sequence length of key or value)
  //   M:    max_sequence_length
  //   T:    total_sequence_length = past_sequence_length + kv_sequence_length
  //   N:    num_heads
  //   H_q:  q_head_size (Note that q_head_size == k_head_size)
  //   H_k:  k_head_size
  //   H_v:  v_head_size
  //   D_i:  input hidden size
  //   D_q:  q_hidden_size = num_heads * q_head_size (Note that q_hidden_size == k_hidden_size)
  //   D_k:  k_hidden_size = num_heads * k_head_size
  //   D_v:  v_hidden_size = num_heads * v_head_size

  // When past state is used, Q, K and V should have same hidden size (unless we split it into past_key and past_value):
  //   H:    head_size
  //   D:    hidden_size, where D == N * H

  // Input shapes with weights:
  //   input        (Q/K/V)    : (B, S, D_i)
  //   weights      (Q/K/V)    : (D_i, D_q + D_k + D_v)
  //   bias         (Q/K/V)    : (D_q + D_k + D_v)
  //   mask_index              : see below
  //   past         (K/V)      : (2, B, N, P, H) or NULL
  //   extra_add_qk            : (B, N, S, L) or NULL
  //   key                     : NULL
  //   value                   : NULL

  // Input shapes without weights (only bias is provided):
  //   input         (Q)       : (B, S, D_q)
  //   weights                 : NULL
  //   bias          (Q/K/V)   : (D_q + D_k + D_v)
  //   mask_index              : see below
  //   past          (K/V)     : (2, B, N, P, H) or NULL
  //   extra_add_qk            : (B, N, S, L) or NULL
  //   key           (K)       : (B, L, D_k)
  //   value         (V)       : (B, L, D_v)

  // For mask_index, the following shapes are supported:
  //     NULL, (B, 1), (1, 1)
  //     (B), (2 * B),
  //     (B, L)
  //     (B, S, L)
  //     (B, 1, M, M)
  //
  // When a model is pruned (like some attention heads are removed in Q/K/V), input_hidden_size could be larger
  // than hidden dimension of Q, K and V.

  if (past != nullptr && extra_add_qk != nullptr) {
    // past is used on GPT-2 model with past state, we don't have a case for extra add qk yet
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Attention cannot have both past and extra_add_qk");
  }

  const auto& dims = input_shape.GetDims();
  if (dims.size() != 3) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'input' is expected to have 3 dimensions, got ",
                           dims.size());
  }

  auto& batch_size = dims[0];
  auto& sequence_length = dims[1];
  int64_t input_hidden_size = dims[2];

  const auto& bias_dims = bias_shape.GetDims();
  if (bias_dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'bias' is expected to have 1 dimension, got ",
                           bias_dims.size());
  }

  if (weights_shape != nullptr) {
    const auto& weights_dims = weights_shape->GetDims();
    if (weights_dims.size() != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'weights' is expected to have 2 dimensions, got ",
                            weights_dims.size());
    }
    if (weights_dims[0] != input_hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "Input 1 dimension 0 should have same length as dimension 2 of input 0");
    }

    if (bias_dims[0] != weights_dims[1]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                            "Input 'bias' dimension 0 should have same length as dimension 1 of input 'weights'");
    }
  }

  int64_t target_sequence_length = sequence_length;
  int64_t q_hidden_size = bias_dims[0] / static_cast<int64_t>(3);
  int64_t k_hidden_size = q_hidden_size;
  int64_t v_hidden_size = k_hidden_size;

  if (weights_shape == nullptr) { // no weights
    if (this->require_weights_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "This operator requires weights");
    }

    if (key == nullptr || value == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "When weights is not provided, key and value are required");
    }

    const auto& key_dims = key->Shape().GetDims();
    if (key_dims.size() != 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'key' is expected to have 3 dimensions, got ",
                             key_dims.size());
    }
    if (key_dims[0] != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'key' dimension 0 should have same length as dimension 0 of input 0");
    }

    const auto& value_dims = value->Shape().GetDims();
    if (value_dims.size() != 3) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'value' is expected to have 3 dimensions, got ",
                             value_dims.size());
    }
    if (value_dims[0] != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'value' dimension 0 should have same length as dimension 0 of input 0");
    }

    if (value_dims[1] != key_dims[1]) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'key' and 'value' dimension 1 should have same length");
    }

    input_hidden_size = -1;
    q_hidden_size = dims[2];
    k_hidden_size = key_dims[2];
    v_hidden_size = value_dims[1];
    target_sequence_length = key_dims[1];
  } else {  // with weights
    if (qkv_hidden_sizes_.size() != 0) {
      if (qkv_hidden_sizes_.size() != 3) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "qkv_hidden_sizes attribute should have 3 elements");
      }

      for (size_t i = 0; i < qkv_hidden_sizes_.size(); i++) {
        if (qkv_hidden_sizes_[i] % num_heads_ != 0) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "hidden_size should be divisible by num_heads:", qkv_hidden_sizes_[i]);
        }
      }

      q_hidden_size = qkv_hidden_sizes_[0];
      k_hidden_size = qkv_hidden_sizes_[1];
      v_hidden_size = qkv_hidden_sizes_[2];
    }
  }

  if (q_hidden_size != k_hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "qkv_hidden_sizes first element should be same as the second");
  }

  if (this->require_same_hidden_size_ && k_hidden_size != v_hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Hidden size of Q, K and V shall be same");
  }

  int64_t total_sequence_length = target_sequence_length;
  if (bias_dims[0] != q_hidden_size + k_hidden_size + v_hidden_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 'bias' dimension 0 should have same length as sum of Q/K/V hidden sizes");
  }

  int64_t past_sequence_length = 0;
  if (past != nullptr) {  // past is optional
    if (k_hidden_size != v_hidden_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'past' expect k_hidden_size == v_hidden_size");
    }

    const auto& past_dims = past->Shape().GetDims();
    if (past_dims.size() != 5) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'past' is expected to have 5 dimension, got ",
                             past_dims.size());
    }

    if (past_dims[0] != 2) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Inputs 'past' dimension 0 shall have length of 2");
    }

    if (past_dims[1] != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Inputs 'past' dimension 1 shall have same length as dimension 0 of input 0");
    }

    if (static_cast<int>(past_dims[2]) != num_heads_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Inputs 'past' dimension 2 shall have length of num_heads", num_heads_);
    }

    if (static_cast<int>(past_dims[4]) != k_hidden_size / num_heads_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Inputs 'past' dimension 2 shall have length of ", k_hidden_size / num_heads_);
    }

    past_sequence_length = past_dims[3];
    total_sequence_length += past_sequence_length;
  }

  int64_t max_sequence_length = -1;
  if (mask_index != nullptr) {  // mask_index is optional
    const auto& mask_dims = mask_index->Shape().GetDims();
    if (mask_dims.size() == 1) {
      if (mask_dims[0] != batch_size && mask_dims[0] != 2 * batch_size) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Inputs 'mask_index' with 1D data shall have length of batch_size or 2 * batch_size");
      }
    } else if (mask_dims.size() == 2) {
      if (mask_dims[0] != batch_size || mask_dims[1] != total_sequence_length) {
        // Add operator supports broadcasting. Here we handle a case with only one element in the 2nd dimension.
        if ((mask_dims[0] == batch_size || mask_dims[0] == 1) && mask_dims[1] == 1) {
          // Mask will have same value after propagation, which has same effect as no mask.
          mask_index = nullptr;
        } else {
          return ORT_MAKE_STATUS(
              ONNXRUNTIME, INVALID_ARGUMENT,
              "Inputs 'mask_index' with 2D data shall have shape "
              "batch_size x total_sequence_length");
        }
      }
    } else if (mask_dims.size() == 3) {
      if (mask_dims[0] != batch_size || mask_dims[1] != sequence_length || mask_dims[2] != total_sequence_length) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, INVALID_ARGUMENT,
            "Inputs 'mask_index' with 3D data shall have shape "
            "batch_size x sequence_length x total_sequence_length");
      }
    } else if (mask_dims.size() == 4) {
      if (mask_dims[0] != batch_size || mask_dims[1] != 1 || mask_dims[2] != mask_dims[3] ||
          mask_dims[2] < total_sequence_length) {
        return ORT_MAKE_STATUS(
            ONNXRUNTIME, INVALID_ARGUMENT,
            "Inputs 'mask_index' with 4D data shall have shape "
            "batch_size x 1 x max_sequence_length x max_sequence_length)");
      }
      max_sequence_length = mask_dims[3];

      if (is_unidirectional_ == true) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "Inputs 'mask_index' with 4D data shall have is_unidirectional_ set to false");
      }
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'mask_index' is expected to have 1, 2, 3 or 4 dimensions, got ",
                             mask_dims.size());
    }
  }

  if (extra_add_qk != nullptr) {
    const auto& extra_add_qk_dims = extra_add_qk->Shape().GetDims();

    if (extra_add_qk_dims.size() != 4) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'extra_add_qk' is expected to have 4 dimensions, got ",
                             extra_add_qk_dims.size());
    }

    if (extra_add_qk_dims[0] != batch_size) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'extra_add_qk' dimension 0 should be same as batch_size, got ",
                             extra_add_qk_dims[0]);
    }
    if (extra_add_qk_dims[1] != num_heads_) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'extra_add_qk' dimension 1 should be same as number of heads, got ",
                             extra_add_qk_dims[1]);
    }
    if (extra_add_qk_dims[2] != sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'extra_add_qk' dimension 2 should be same as sequence_length, got ",
                             extra_add_qk_dims[2]);
    }
    if (extra_add_qk_dims[3] != total_sequence_length) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input 'extra_add_qk' dimension 3 should be same as total_sequence_length, got ",
                             extra_add_qk_dims[3]);
    }
  }

  if (parameters != nullptr) {
    AttentionParameters* output_parameters = reinterpret_cast<AttentionParameters*>(parameters);
    output_parameters->batch_size = static_cast<int>(batch_size);
    output_parameters->sequence_length = static_cast<int>(sequence_length);
    output_parameters->past_sequence_length = static_cast<int>(past_sequence_length);
    output_parameters->target_sequence_length = static_cast<int>(target_sequence_length);
    output_parameters->total_sequence_length = static_cast<int>(total_sequence_length);
    output_parameters->max_sequence_length = static_cast<int>(max_sequence_length);
    output_parameters->input_hidden_size = static_cast<int>(input_hidden_size);
    output_parameters->q_hidden_size = static_cast<int>(q_hidden_size);
    output_parameters->k_hidden_size = static_cast<int>(k_hidden_size);
    output_parameters->v_hidden_size = static_cast<int>(v_hidden_size);
    output_parameters->q_head_size = static_cast<int>(q_hidden_size) / num_heads_;
    output_parameters->k_head_size = static_cast<int>(k_hidden_size) / num_heads_;
    output_parameters->v_head_size = static_cast<int>(v_hidden_size) / num_heads_;
    output_parameters->num_heads = num_heads_;
    output_parameters->is_unidirectional = is_unidirectional_;
  }

  return Status::OK();
}

Status AttentionBase::CheckInputs(const TensorShape& input_shape,
                                  const TensorShape* weights_shape,
                                  const TensorShape& bias_shape,
                                  const Tensor*& mask_index,
                                  const Tensor* past,
                                  const Tensor* extra_add_qk,
                                  const Tensor* key,
                                  const Tensor* value,
                                  void* parameters,
                                  const int max_threads_per_block) const {
  if (num_heads_ > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "num_heads should be no larger than ", max_threads_per_block);
  }

  return CheckInputs(input_shape, weights_shape, bias_shape, mask_index, past, extra_add_qk, key, value, parameters);
}

Tensor* AttentionBase::GetPresent(OpKernelContext* context,
                                  const Tensor* past,
                                  int batch_size,
                                  int head_size,
                                  int kv_sequence_length,
                                  int& past_sequence_length) const {
  // Input and output shapes:
  //   past        : (2, batch_size, num_heads, past_sequence_length, head_size)
  //   present     : (2, batch_size, num_heads, past_sequence_length + kv_sequence_length, head_size)

  past_sequence_length = (nullptr != past) ? static_cast<int>(past->Shape().GetDims()[3]) : 0;
  std::vector<int64_t> present_dims{2, batch_size, num_heads_, kv_sequence_length + past_sequence_length, head_size};

  TensorShape present_shape(present_dims);
  Tensor* present = context->Output(1, present_shape);
  if (nullptr != past && nullptr == present) {
    ORT_THROW("Expect to have present state output when past state input is given");
  }

  return present;
}

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : OpKernel(info), AttentionCPUBase(info, false, true) {
}

template <typename T>
bool Attention<T>::IsPackWeightsSuccessful(int qkv_index,
                                           AllocatorPtr alloc,
                                           size_t head_size,
                                           size_t input_hidden_size,
                                           const T* weights_data,
                                           size_t weight_matrix_col_size,
                                           /*out*/ PrePackedWeights* prepacked_weights) {
  size_t packb_size = MlasGemmPackBSize(head_size, input_hidden_size);
  if (packb_size == 0) {
    return false;
  }

  size_t loop_len = static_cast<size_t>(num_heads_);
  size_t packed_weights_data_size = packb_size * loop_len;  // The same size would be computed by AllocArray() below
  auto* packed_weights_data = static_cast<uint8_t*>(alloc->AllocArray(packb_size, loop_len));

  // Initialize memory to 0 as there could be some padding associated with pre-packed
  // buffer memory and we don not want it uninitialized and generate different hashes
  // if and when we try to cache this pre-packed buffer for sharing between sessions.
  memset(packed_weights_data, 0, packed_weights_data_size);
  packed_weights_[qkv_index] = BufferUniquePtr(packed_weights_data, BufferDeleter(std::move(alloc)));
  packed_weights_size_[qkv_index] = packb_size;

  for (size_t i = 0; i < loop_len; i++) {
    MlasGemmPackB(CblasNoTrans, head_size, input_hidden_size, weights_data, weight_matrix_col_size, packed_weights_data);
    packed_weights_data += packb_size;
    weights_data += head_size;
  }

  if (prepacked_weights != nullptr) {
    prepacked_weights->buffers_.push_back(std::move(packed_weights_[qkv_index]));
    prepacked_weights->buffer_sizes_.push_back(packed_weights_data_size);
  }
  return true;
}

template <typename T>
Status Attention<T>::PrePack(const Tensor& weights, int input_idx, AllocatorPtr alloc,
                             /*out*/ bool& is_packed,
                             /*out*/ PrePackedWeights* prepacked_weights) {
  /* The PrePack() massages the weights to speed up Compute(), there is an option to
   * use shared prepacked weights in which case prepacked_weights parameter would be non-null.
   *
   * We use an array of buffers to store prepacked Q, K, V weights for the sake of simplicity
   * and easy offset management in Compute(). They are packed one after the other. In case of failure,
   *    1. With shared pre-pack weights the caller of this fn() frees up the memory so far allocated.
   *    2. When weights are held by kernel, it will be freed before returning.
   */
  is_packed = false;

  if (1 != input_idx) {
    return Status::OK();
  }

  weight_shape_ = weights.Shape();
  const auto& weights_dims = weight_shape_.GetDims();
  if (weights_dims.size() != 2) {
    return Status::OK();
  }

  const auto* weights_data = weights.Data<T>();
  const size_t input_hidden_size = static_cast<size_t>(weights_dims[0]);
  size_t q_hidden_size, k_hidden_size, v_hidden_size;

  if (qkv_hidden_sizes_.size() != 0) {
    q_hidden_size = qkv_hidden_sizes_[0];
    k_hidden_size = qkv_hidden_sizes_[1];
    v_hidden_size = qkv_hidden_sizes_[2];

    if (q_hidden_size == 0 || k_hidden_size == 0 || v_hidden_size == 0) {
      return Status::OK();
    }

    if (q_hidden_size % num_heads_ != 0 || k_hidden_size % num_heads_ != 0 || v_hidden_size % num_heads_ != 0) {
      return Status::OK();
    }
  } else {
    const size_t hidden_size_x3 = static_cast<size_t>(weights_dims[1]);
    const size_t hidden_size = hidden_size_x3 / 3;

    if (hidden_size % num_heads_ != 0) {
      return Status::OK();
    }

    q_hidden_size = hidden_size;
    k_hidden_size = hidden_size;
    v_hidden_size = hidden_size;
  }

  const size_t qkv_head_size[3] = {q_hidden_size / num_heads_, k_hidden_size / num_heads_, v_hidden_size / num_heads_};
  const size_t weight_matrix_col_size = q_hidden_size + k_hidden_size + v_hidden_size;

  if (!IsPackWeightsSuccessful(0, alloc, qkv_head_size[0], input_hidden_size,
                               weights_data, weight_matrix_col_size, prepacked_weights) ||
      !IsPackWeightsSuccessful(1, alloc, qkv_head_size[1], input_hidden_size,
                               weights_data + (num_heads_ * qkv_head_size[0]),
                               weight_matrix_col_size, prepacked_weights) ||
      !IsPackWeightsSuccessful(2, alloc, qkv_head_size[2], input_hidden_size,
                               weights_data + (num_heads_ * (qkv_head_size[0] + qkv_head_size[1])),
                               weight_matrix_col_size, prepacked_weights)) {
    if (prepacked_weights == nullptr) {
      FreePackedWeights(packed_weights_, qkv_hidden_sizes_.size());
    }
    return Status::OK();
  }

  is_packed = true;
  is_prepack_ = true;
  return Status::OK();
}

template <typename T>
Status Attention<T>::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                               int input_idx,
                                               /*out*/ bool& used_shared_buffers) {
  if (1 != input_idx) {
    return Status::OK();
  }

  used_shared_buffers = true;
  packed_weights_[0] = std::move(prepacked_buffers[0]);
  packed_weights_[1] = std::move(prepacked_buffers[1]);
  packed_weights_[2] = std::move(prepacked_buffers[2]);

  return Status::OK();
}

template <typename T>
Status Attention<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = is_prepack_ ? nullptr : context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);

  const Tensor* mask_index = context->Input<Tensor>(3);
  const Tensor* past = context->Input<Tensor>(4);
  const Tensor* extra_add_qk = context->Input<Tensor>(5);

  const Tensor* key = context->Input<Tensor>(6);
  const Tensor* value = context->Input<Tensor>(7);

  const TensorShape& weights_shape = (weights ? weights->Shape() : weight_shape_);

  AttentionParameters parameters;
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  (nullptr != weights || is_prepack_) ? &weights_shape : nullptr,
                                  bias->Shape(),
                                  mask_index,
                                  past,
                                  extra_add_qk,
                                  key,
                                  value,
                                  &parameters));

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int input_hidden_size = parameters.input_hidden_size;

  std::vector<int64_t> output_shape(3);
  output_shape[0] = static_cast<int64_t>(batch_size);
  output_shape[1] = static_cast<int64_t>(sequence_length);
  output_shape[2] = static_cast<int64_t>(parameters.v_hidden_size);
  Tensor* output = context->Output(0, output_shape);

  constexpr size_t element_size = sizeof(T);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  auto* tp = context->GetOperatorThreadPool();
  // Compute Q, K, V
  // gemm_data(BS, D_t) = input(BS, D_i) x weights(D_i, D_t) + bias(D_t), where D_t = D_q + D_k + D_v
  // Hidden dimension of input could be larger than that of Q, K and V when model is pruned.
  int qkv_hidden_size = (parameters.q_hidden_size + parameters.k_hidden_size + parameters.v_hidden_size);
  auto gemm_data = allocator->Alloc(SafeInt<size_t>(batch_size) * sequence_length * qkv_hidden_size * element_size);
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(std::move(allocator)));

  auto Q = reinterpret_cast<T*>(gemm_data);
  auto K = Q + static_cast<size_t>(batch_size) * sequence_length * parameters.q_hidden_size;
  auto V = K + static_cast<size_t>(batch_size) * sequence_length * parameters.k_hidden_size;

  T* QKV[3] = {Q, K, V};
  const int qkv_head_size[3] = {parameters.q_head_size, parameters.k_head_size, parameters.v_head_size};

  {
    const int loop_len = 3 * batch_size * num_heads_;
    const auto* input_data = input->Data<T>();
    const auto* weights_data = weights ? weights->Data<T>() : nullptr;
    const auto* bias_data = bias->Data<T>();

    // We use Q/K head size to estimate the cost, this is not accurate when Q/K and V head sizes are different.
    const double cost = static_cast<double>(sequence_length) *
                        static_cast<double>(parameters.q_head_size) * static_cast<double>(input_hidden_size);

    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int batch_index = static_cast<int>((i / 3) / num_heads_);
        const int head_index = static_cast<int>((i / 3) % num_heads_);
        const int qkv_index = static_cast<int>(i % 3);

        int input_offset = batch_index * sequence_length * input_hidden_size;

        T* qkv_dest = QKV[qkv_index];
        int head_size = qkv_head_size[qkv_index];
        int weights_offset = 0;
        int bias_offset = qkv_index * parameters.q_hidden_size + head_index * head_size;

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
        // A: input          (BxSxD_i)          (B.)S x D_i           S x D_i
        // B: weights        (D_ixNxH_t)        D_i x (N.)H_t         D_i x H_t
        // C: QKV[qkv_index] (BxNxSxH_t)        (B.N.)S x H_t         S x H_t
        // Here H_t = H_q + H_k + H_v is size of one head of Q, K and V
        if (is_prepack_) {
          uint8_t* packed_weight;
          packed_weight = static_cast<uint8_t*>(packed_weights_[qkv_index].get()) +
                          packed_weights_size_[qkv_index] * (weights_offset / head_size);

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
              qkv_hidden_size,                                // ldb    = D_q + D_k + D_v
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
  return ApplyAttention(Q, K, V, mask_index, past, output,
                        batch_size, sequence_length,
                        qkv_head_size[0], qkv_head_size[2], v_hidden_size,
                        extra_add_qk, context);
}
}  // namespace contrib
}  // namespace onnxruntime
