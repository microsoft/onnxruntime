// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_cpu_base.h"
#include "attention_helper.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/common/safeint.h"
#include "core/common/span_utils.h"
#include "core/platform/threadpool.h"

using onnxruntime::narrow;
using onnxruntime::concurrency::ThreadPool;
namespace onnxruntime {
namespace contrib {

static void FreePackedWeights(gsl::span<IAllocatorUniquePtr<void>> array) {
  for (auto& weights : array) {
    weights.reset();
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
                                   gsl::span<const size_t> /*prepacked_buffer_sizes*/,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

 private:
  bool IsPackWeightsSuccessful(int qkv_index, AllocatorPtr alloc, size_t head_size,
                               size_t input_hidden_size, const T* weights_data,
                               size_t weight_matrix_col_size, PrePackedWeights* prepacked_weights);

  std::array<IAllocatorUniquePtr<void>, 3> packed_weights_;
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

template <typename T>
Attention<T>::Attention(const OpKernelInfo& info) : OpKernel(info), AttentionCPUBase(info, false) {
}

template <typename T>
bool Attention<T>::IsPackWeightsSuccessful(int qkv_index,
                                           AllocatorPtr alloc,
                                           size_t head_size,
                                           size_t input_hidden_size,
                                           const T* weights_data,
                                           size_t weight_matrix_col_size,
                                           /*out*/ PrePackedWeights* prepacked_weights) {
  size_t packb_size = MlasGemmPackBSize(CblasNoTrans, CblasNoTrans, head_size, input_hidden_size,
                                        &mlas_backend_kernel_selector_config_);
  if (packb_size == 0) {
    return false;
  }

  size_t loop_len = narrow<size_t>(num_heads_);
  size_t packed_weights_data_size = SafeInt<size_t>(packb_size) * loop_len;
  packed_weights_[qkv_index] = IAllocator::MakeUniquePtr<void>(alloc, packed_weights_data_size, true);
  packed_weights_size_[qkv_index] = packb_size;
  std::byte* packed_weights_data = static_cast<std::byte*>(packed_weights_[qkv_index].get());
  // Initialize memory to 0 as there could be some padding associated with pre-packed
  // buffer memory and we do not want it uninitialized and generate different hashes
  // if and when we try to cache this pre-packed buffer for sharing between sessions.
  memset(packed_weights_data, 0, packed_weights_data_size);

  for (size_t i = 0; i < loop_len; i++) {
    MlasGemmPackB(CblasNoTrans, CblasNoTrans, head_size, input_hidden_size, weights_data, weight_matrix_col_size,
                  packed_weights_data, &mlas_backend_kernel_selector_config_);
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
  std::array<int64_t, 3> hidden_sizes{};
  ORT_RETURN_IF_ERROR(attention::CheckAttentionWeights(weights_dims, qkv_hidden_sizes_, num_heads_,
                                                       require_same_hidden_size_, hidden_sizes));
  if (weight_shape_.Size() == 0) {
    return Status::OK();
  }

  const auto* weights_data = weights.Data<T>();
  const size_t input_hidden_size = narrow<size_t>(weights_dims[0]);
  const size_t q_hidden_size = narrow<size_t>(hidden_sizes[0]);
  const size_t k_hidden_size = narrow<size_t>(hidden_sizes[1]);
  const size_t v_hidden_size = narrow<size_t>(hidden_sizes[2]);
  const size_t v_weights_offset = SafeInt<size_t>(q_hidden_size) + k_hidden_size;
  const size_t qkv_head_size[3] = {q_hidden_size / num_heads_, k_hidden_size / num_heads_, v_hidden_size / num_heads_};
  const size_t weight_matrix_col_size = narrow<size_t>(weights_dims[1]);

  if (!IsPackWeightsSuccessful(0, alloc, qkv_head_size[0], input_hidden_size,
                               weights_data, weight_matrix_col_size, prepacked_weights) ||
      !IsPackWeightsSuccessful(1, alloc, qkv_head_size[1], input_hidden_size,
                               weights_data + q_hidden_size,
                               weight_matrix_col_size, prepacked_weights) ||
      !IsPackWeightsSuccessful(2, alloc, qkv_head_size[2], input_hidden_size,
                               weights_data + v_weights_offset,
                               weight_matrix_col_size, prepacked_weights)) {
    if (prepacked_weights == nullptr) {
      FreePackedWeights(packed_weights_);
    }
    return Status::OK();
  }

  is_packed = true;
  is_prepack_ = true;
  return Status::OK();
}

template <typename T>
Status Attention<T>::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                               gsl::span<const size_t> /*prepacked_buffer_sizes*/,
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
  const Tensor* attention_bias = context->Input<Tensor>(5);

  const TensorShape& weights_shape = (weights ? weights->Shape() : weight_shape_);
  ORT_RETURN_IF_NOT(weights_shape.NumDimensions() == 2, "Input 'weights' is expected to have 2 dimensions");
  const std::array<int64_t, 1> bias_dims{weights_shape[1]};
  const TensorShape default_bias_shape(bias_dims);

  AttentionParameters parameters;
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(),
                                  weights_shape,
                                  bias ? bias->Shape() : default_bias_shape,
                                  mask_index,
                                  past,
                                  attention_bias,
                                  &parameters));

  if (parameters.do_rotary) {
    ORT_NOT_IMPLEMENTED(
        "Rotary embedding is not supported in Attention CPU kernel. \
                        Please fuse the model with MHA + RotaryEmbedding.");
  }

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
  // gemm_data(BS, D_t) = input(BS, D_i) x weights(D_i, D_t) + bias(D_t), where D_t = D + D + D_v
  // Hidden dimension of input could be larger than that of Q, K and V when model is pruned.
  const int qkv_hidden_size = SafeInt<int>(parameters.hidden_size) * 2 + parameters.v_hidden_size;
  const size_t gemm_bytes = SafeInt<size_t>(batch_size) * sequence_length * qkv_hidden_size * element_size;
  auto gemm_data = gemm_bytes == 0 ? nullptr : allocator->Alloc(gemm_bytes);
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(std::move(allocator)));

  const size_t q_elements = SafeInt<size_t>(batch_size) * sequence_length * parameters.hidden_size;
  auto Q = reinterpret_cast<T*>(gemm_data);
  auto K = q_elements == 0 ? Q : Q + q_elements;
  auto V = q_elements == 0 ? K : K + q_elements;

  T* QKV[3] = {Q, K, V};
  const int qkv_head_size[3] = {parameters.head_size, parameters.head_size, parameters.v_head_size};

  if (batch_size > 0 && sequence_length > 0) {
    const ptrdiff_t loop_len = SafeInt<ptrdiff_t>(3) * batch_size * num_heads_;
    const auto* input_data = input->Data<T>();
    const auto* weights_data = weights ? weights->Data<T>() : nullptr;
    const auto* bias_data = bias ? bias->Data<T>() : nullptr;

    // We use Q/K head size to estimate the cost, this is not accurate when Q/K and V head sizes are different.
    const double cost = static_cast<double>(sequence_length) *
                        static_cast<double>(parameters.head_size) * static_cast<double>(input_hidden_size);

    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int batch_index = static_cast<int>((i / 3) / num_heads_);
        const int head_index = static_cast<int>((i / 3) % num_heads_);
        const int qkv_index = static_cast<int>(i % 3);

        const size_t input_offset = SafeInt<size_t>(batch_index) * sequence_length * input_hidden_size;

        T* qkv_dest = QKV[qkv_index];
        int head_size = qkv_head_size[qkv_index];
        size_t weights_offset = 0;
        const size_t bias_offset = SafeInt<size_t>(qkv_index) * parameters.hidden_size +
                                   SafeInt<size_t>(head_index) * head_size;

        if (!is_prepack_) {
          weights_offset = bias_offset;
        } else {
          weights_offset = SafeInt<size_t>(head_index) * head_size;
        }

        const size_t qkv_offset = (SafeInt<size_t>(batch_index) * num_heads_ + head_index) *
                                  sequence_length * head_size;

        // TODO!! memcpy here makes it not worthwhile to use Gemm batch. Possible to post process?
        // broadcast NH -> (B.N.S.H) for each of Q, K, V
        const T* broadcast_data_src = bias_data ? bias_data + bias_offset : nullptr;
        T* broadcast_data_dest = QKV[qkv_index] + qkv_offset;

        for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
          const size_t bias_bytes = SafeInt<size_t>(head_size) * sizeof(T);
          if (broadcast_data_src) {
            memcpy(broadcast_data_dest, broadcast_data_src, bias_bytes);
          } else {
            memset(broadcast_data_dest, 0, bias_bytes);
          }
          broadcast_data_dest += head_size;
        }

        if (input_hidden_size == 0) {
          continue;
        }

        //                   original           transposed            iteration
        // A: input          (BxSxD_i)          (B.)S x D_i           S x D_i
        // B: weights        (D_ixNxH_t)        D_i x (N.)H_t         D_i x H_t
        // C: QKV[qkv_index] (BxNxSxH_t)        (B.N.)S x H_t         S x H_t
        // Here H_t = H + H + H_v is size of one head of Q, K and V
        if (is_prepack_) {
          uint8_t* packed_weight;
          packed_weight = static_cast<uint8_t*>(packed_weights_[qkv_index].get()) +
                          packed_weights_size_[qkv_index] * (weights_offset / head_size);

          MlasGemm(
              CblasNoTrans,                            // TransA = no
              sequence_length,                         // M      = S
              head_size,                               // N      = H
              input_hidden_size,                       // K      = D
              1.0f,                                    // alpha
              input_data + input_offset,               // A
              input_hidden_size,                       // lda    = D
              packed_weight,                           // B
              1.0f,                                    // beta
              qkv_dest + qkv_offset,                   // C
              head_size,                               // ldc
              nullptr,                                 // use single-thread
              &mlas_backend_kernel_selector_config_);  // BackendKernelSelectorConfig
        } else {
          math::GemmEx<float, ThreadPool>(
              CblasNoTrans,                            // TransA = no
              CblasNoTrans,                            // TransB = no
              sequence_length,                         // M      = S
              head_size,                               // N      = H
              input_hidden_size,                       // K      = D
              1.0f,                                    // alpha
              input_data + input_offset,               // A
              input_hidden_size,                       // lda    = D
              weights_data + weights_offset,           // B
              qkv_hidden_size,                         // ldb    = D + D + D_v
              1.0f,                                    // beta
              qkv_dest + qkv_offset,                   // C
              head_size,                               // ldc
              nullptr,                                 // use single-thread
              &mlas_backend_kernel_selector_config_);  // BackendKernelSelectorConfig
        }
      }
    });
  }

  // Compute the attention score and apply the score to V
  return ApplyAttention(Q, K, V, mask_index, past, nullptr /* past_key */, nullptr /* past_value */,
                        output, nullptr /* present_key */, nullptr /* present_value */, nullptr /* output_qk */,
                        batch_size, sequence_length, sequence_length,
                        parameters.head_size, parameters.v_head_size, parameters.v_hidden_size,
                        attention_bias, context);
}
}  // namespace contrib
}  // namespace onnxruntime
