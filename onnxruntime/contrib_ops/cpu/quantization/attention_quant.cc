// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "contrib_ops/cpu/bert/attention_cpu_base.h"
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

template <typename T>
class QAttention : public OpKernel, public AttentionCPUBase {
 public:
  QAttention(const OpKernelInfo& info);

  Status Compute(OpKernelContext* context) const override;

  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& /*out*/ is_packed,
                 /*out*/ PrePackedWeights* prepacked_weights) override;

  Status UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                   int input_idx,
                                   /*out*/ bool& used_shared_buffers) override;

 private:
  BufferUniquePtr packed_weights_;
  size_t packed_weights_size_;
  TensorShape weight_shape_;
  bool weights_is_signed_;
};

// These ops are internal-only, so register outside of onnx
ONNX_OPERATOR_TYPED_KERNEL_EX(
    QAttention,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QAttention<float>);

template <typename T>
QAttention<T>::QAttention(const OpKernelInfo& info) : OpKernel(info), AttentionCPUBase(info) {}

template <typename T>
Status QAttention<T>::PrePack(const Tensor& weights, int input_idx, AllocatorPtr alloc,
                              /*out*/ bool& is_packed,
                              /*out*/ PrePackedWeights* prepacked_weights) {
  if (1 != input_idx) {
    return Status::OK();
  }

  weight_shape_ = weights.Shape();
  const auto& weights_dims = weight_shape_.GetDims();
  if (weights_dims.size() != 2) {
    return Status::OK();
  }

  const size_t input_hidden_size = static_cast<size_t>(weights_dims[0]);
  const size_t hidden_size_x3 = static_cast<size_t>(weights_dims[1]);
  const size_t hidden_size = hidden_size_x3 / 3;
  const size_t head_size = hidden_size / num_heads_;

  // Bail out if the weights shape has an expected shape.
  if ((hidden_size == 0) || ((hidden_size % num_heads_) != 0) || (hidden_size_x3 != 3 * hidden_size)) {
    return Status::OK();
  }

  const auto* weights_data = static_cast<const uint8_t*>(weights.DataRaw());
  weights_is_signed_ = weights.IsDataType<int8_t>();

  packed_weights_size_ = MlasGemmPackBSize(head_size, input_hidden_size, false /*AIsSigned*/, weights_is_signed_);
  if (packed_weights_size_ == 0) {
    return Status::OK();
  }

  const size_t loop_len = 3 * static_cast<size_t>(num_heads_);
  size_t packed_weights_data_size = packed_weights_size_ * loop_len;
  auto* packed_weights_data = static_cast<uint8_t*>(alloc->Alloc(packed_weights_data_size));

  // Initialize memory to 0 as there could be some padding associated with pre-packed
  // buffer memory and we don not want it uninitialized and generate different hashes
  // if and when we try to cache this pre-packed buffer for sharing between sessions.
  memset(packed_weights_data, 0, packed_weights_data_size);

  packed_weights_ = BufferUniquePtr(packed_weights_data, BufferDeleter(std::move(alloc)));

  for (size_t i = 0; i < loop_len; i++) {
    MlasGemmPackB(head_size, input_hidden_size, weights_data, hidden_size_x3, false /*AIsSigned*/, weights_is_signed_, packed_weights_data);
    packed_weights_data += packed_weights_size_;
    weights_data += head_size;
  }

  bool share_prepacked_weights = (prepacked_weights != nullptr);
  if (share_prepacked_weights) {
    prepacked_weights->buffers_.push_back(std::move(packed_weights_));
    prepacked_weights->buffer_sizes_.push_back(packed_weights_data_size);
  }

  is_packed = true;
  return Status::OK();
}

template <typename T>
Status QAttention<T>::UseSharedPrePackedBuffers(std::vector<BufferUniquePtr>& prepacked_buffers,
                                                int input_idx,
                                                /*out*/ bool& used_shared_buffers) {
  if (1 != input_idx) {
    return Status::OK();
  }

  used_shared_buffers = true;
  packed_weights_ = std::move(prepacked_buffers[0]);

  return Status::OK();
}

template <typename T>
Status QAttention<T>::Compute(OpKernelContext* context) const {
  // Input and output shapes:
  //   Input  0 - input             : (batch_size, sequence_length, input_hidden_size)
  //   Input  1 - weights           : (input_hidden_size, 3 * hidden_size)
  //   Input  2 - bias              : (3 * hidden_size)
  //   Input  3 - input_scale       : scalar
  //   Input  4 - weight_scale      : scalar for per tensor quantization, (3 * hidden_size) for per column quantization
  //   Input  5 - mask_index        : nullptr, (batch_size), (2 * batch_size), (batch_size, 1), (1, 1) or (batch_size, past_sequence_length + sequence_length)
  //   Input  6 - input_zero_point  : scalar
  //   Input  7 - weight_zero_point : scalar for per tensor quantization, (3 * hidden_size) for per column quantization
  //   Input  8 - past              : (2, batch_size, num_heads, past_sequence_length, head_size)
  //   Output 0                     : (batch_size, sequence_length, hidden_size)
  //   Output 1 - present           : (2, batch_size, num_heads, past_sequence_length + sequence_length, head_size)
  //   ORT_RETURN_IF_ERROR(CheckInputs(context));
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = packed_weights_ ? nullptr : context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* input_scale_tensor = context->Input<Tensor>(3);
  const Tensor* weight_scale_tensor = context->Input<Tensor>(4);
  const Tensor* mask_index = context->Input<Tensor>(5);
  const Tensor* i_zp_tensor = context->Input<Tensor>(6);
  const Tensor* w_zp_tensor = context->Input<Tensor>(7);
  const Tensor* past_tensor = context->Input<Tensor>(8);

  const TensorShape& weights_shape = (packed_weights_ ? weight_shape_ : weights->Shape());
  ORT_RETURN_IF_ERROR(AttentionBase::CheckInputs(input->Shape(),
                                                 weights_shape,
                                                 bias->Shape(),
                                                 mask_index,
                                                 past_tensor,
                                                 nullptr));

  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(input_scale_tensor),
                    "input scale must be a scalar or 1D tensor of size 1");
  T input_scale = *(input_scale_tensor->Data<T>());

  bool is_weight_scale_per_column = !IsScalarOr1ElementVector(weight_scale_tensor);
  const T* weight_scale_data = weight_scale_tensor->Data<T>();

  std::vector<T> dequant_scales(weight_scale_data, weight_scale_data + weight_scale_tensor->Shape().Size());
  std::for_each(dequant_scales.begin(), dequant_scales.end(), [&input_scale](float& dequant_scale) {
    return dequant_scale *= input_scale;
  });

  uint8_t input_zero_point = 0;
  if (i_zp_tensor != nullptr) {
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(i_zp_tensor),
                      "input zero point must be a scalar or 1D tensor of size 1.");
    input_zero_point = *i_zp_tensor->Data<uint8_t>();
  }

  bool is_weight_zp_per_column = false;
  uint8_t weight_zp_default = 0;
  const uint8_t* weight_zp_data = nullptr;
  if (w_zp_tensor != nullptr) {
    is_weight_zp_per_column = !IsScalarOr1ElementVector(w_zp_tensor);
    weight_zp_data = static_cast<const uint8_t*>(w_zp_tensor->DataRaw());
  }

  const auto& shape = input->Shape();
  const int batch_size = static_cast<int>(shape[0]);
  const int sequence_length = static_cast<int>(shape[1]);
  const int input_hidden_size = static_cast<int>(shape[2]);

  const auto hidden_size_x3 = weights_shape.GetDims()[1];
  const int hidden_size = static_cast<int>(hidden_size_x3) / 3;
  const int head_size = hidden_size / num_heads_;

  std::vector<int64_t> output_shape(3);
  output_shape[0] = shape[0];
  output_shape[1] = shape[1];
  output_shape[2] = static_cast<int64_t>(hidden_size);
  Tensor* output = context->Output(0, output_shape);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  constexpr size_t element_size = sizeof(T);

  auto* tp = context->GetOperatorThreadPool();
  // STEP.1: gemm_data(BS, 3NH) = Scale(input(BS, D) x weights(D, 3NH)) + bias(3NH)
  // D is hidden dimension of input, where input_hidden_size (D) could be larger than hidden_size (NH) when model is pruned.
  auto gemm_data = allocator->Alloc(SafeInt<size_t>(batch_size) * sequence_length * 3 * hidden_size * element_size);
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(std::move(allocator)));

  auto Q = reinterpret_cast<T*>(gemm_data);
  auto K = Q + static_cast<int64_t>(batch_size) * sequence_length * hidden_size;
  auto V = K + static_cast<int64_t>(batch_size) * sequence_length * hidden_size;
  T* QKV[3] = {Q, K, V};

  {
    const int loop_len = 3 * batch_size * num_heads_;
    const auto* input_data = input->Data<uint8_t>();
    const auto* bias_data = bias->Data<T>();

    const auto* weights_data = packed_weights_ ? nullptr : static_cast<const uint8_t*>(weights->DataRaw());
    const bool weights_is_signed = packed_weights_ ? weights_is_signed_ : weights->IsDataType<int8_t>();

    MLAS_GEMM_QUANT_SHAPE_PARAMS gemm_shape;
    gemm_shape.M = sequence_length;
    gemm_shape.N = head_size;
    gemm_shape.K = input_hidden_size;
    gemm_shape.BIsSigned = weights_is_signed;

    std::vector<MLAS_GEMM_QUANT_DATA_PARAMS> gemm_data_vec(loop_len);
    std::vector<MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR> scale_bias_procs;
    scale_bias_procs.reserve(loop_len);

    for (int i = 0; i < loop_len; i++) {
      const int batch_index = static_cast<int>((i / 3) / num_heads_);
      const int head_index = static_cast<int>((i / 3) % num_heads_);
      const int qkv_index = static_cast<int>(i % 3);

      int input_offset = batch_index * sequence_length * input_hidden_size;
      int weights_offset = qkv_index * hidden_size + head_index * head_size;
      int weights_scale_offset = is_weight_scale_per_column ? weights_offset : 0;
      int weights_zp_offset = is_weight_zp_per_column ? weights_offset : 0;
      float* qkv_dest = QKV[qkv_index];
      int qkv_offset = (batch_index * num_heads_ + head_index) * (sequence_length * head_size);

      //                   original           transposed            iteration
      // A: input          (BxSxD)            (B.)S x D             S x D
      // B: weights        (Dx3xNxH)          D  x (3.N.)H          D x H
      // C: QKV[qkv_index] (3xBxNxSxH)        (3.B.N.)S x H         S x H

      scale_bias_procs.emplace_back(qkv_dest + qkv_offset,
                                    head_size,
                                    dequant_scales.data() + weights_scale_offset,
                                    bias_data + weights_offset,
                                    MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
                                    is_weight_scale_per_column ? MLAS_QUANTIZATION_GRANULARITY::PerColumn : MLAS_QUANTIZATION_GRANULARITY::PerMatrix);

      auto& gemm_params = gemm_data_vec[i];
      gemm_params.A = input_data + input_offset;
      gemm_params.lda = input_hidden_size;
      gemm_params.ZeroPointA = input_zero_point;
      if (packed_weights_) {
        const auto* packed_weight =
            static_cast<const uint8_t*>(packed_weights_.get()) + packed_weights_size_ * (weights_offset / head_size);
        gemm_params.B = packed_weight;
        gemm_params.BIsPacked = true;
      } else {
        gemm_params.B = weights_data + weights_offset;
        gemm_params.ldb = static_cast<int64_t>(3) * hidden_size;
      }
      gemm_params.ZeroPointB = nullptr != weight_zp_data ? weight_zp_data + weights_zp_offset : &weight_zp_default;
      gemm_params.PerColumnZeroPoints = is_weight_zp_per_column;
      gemm_params.C = reinterpret_cast<int32_t*>(qkv_dest + qkv_offset);
      gemm_params.ldc = head_size;
      gemm_params.OutputProcessor = &(scale_bias_procs[i]);
    }

    MlasGemmBatch(gemm_shape, gemm_data_vec.data(), loop_len, tp);
  }

  // Compute the attention score and apply the score to V
  return ApplyAttention(Q, K, V, mask_index, past_tensor, output,
                        batch_size, sequence_length,
                        head_size, head_size, hidden_size, nullptr, context);
}

}  // namespace contrib
}  // namespace onnxruntime
