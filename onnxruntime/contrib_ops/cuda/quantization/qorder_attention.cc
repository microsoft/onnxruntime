// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qorder_common.h"
#include "qorder_common_impl.h"
#include "qorder_attention.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include <cmath>
#include <iostream>

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define DefineQOrderedAttentionInput(a, s, id) a = id
enum InputIds {
#include "./qorder_attention_input_enum.h"
};
#undef DefineQOrderedAttentionInput

static_assert((InputIds::K_Weight == 1 + InputIds::Q_Weight) && (InputIds::V_Weight == 2 + InputIds::Q_Weight));
static_assert((InputIds::Scale_K_Weight == 1 + InputIds::Scale_Q_Weight) && (InputIds::Scale_V_Weight == 2 + InputIds::Scale_Q_Weight));
static_assert((InputIds::K_Bias == 1 + InputIds::Q_Bias) && (InputIds::V_Bias == 2 + InputIds::Q_Bias));

ONNX_OPERATOR_KERNEL_EX(
    QOrderedAttention,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("S", BuildKernelDefConstraints<float>())
        .TypeConstraint("G", DataTypeImpl::GetTensorType<int32_t>())
        .InputMemoryType(OrtMemTypeCPUInput, InputIds::ScaleInput)
        .InputMemoryType(OrtMemTypeCPUInput, InputIds::Scale_Q_Gemm)
        .InputMemoryType(OrtMemTypeCPUInput, InputIds::Scale_K_Gemm)
        .InputMemoryType(OrtMemTypeCPUInput, InputIds::Scale_V_Gemm)
        .InputMemoryType(OrtMemTypeCPUInput, InputIds::Scale_QK_Gemm)
        .InputMemoryType(OrtMemTypeCPUInput, InputIds::Scale_QK_Softmax)
        .InputMemoryType(OrtMemTypeCPUInput, InputIds::Scale_Values_Gemm),
    QOrderedAttention);

QOrderedAttention::QOrderedAttention(const OpKernelInfo& info) : CudaKernel(info), AttentionBase(info), input_hidden_size_(0) {
  order_input_ = GetCublasLtOrderAttr(info, "order_input");
  order_weight_ = GetCublasLtOrderAttr(info, "order_weight");
  order_bias_ = GetCublasLtOrderAttr(info, "order_bias");
  order_output_ = GetCublasLtOrderAttr(info, "order_output");
  if (order_input_ == CUBLASLT_ORDER_ROW) {
    ORT_ENFORCE(order_weight_ == CUBLASLT_ORDER_COL, "Only CUBLASLT_ORDER_COL is supported for order_weight_");
    ORT_ENFORCE(order_bias_ == CUBLASLT_ORDER_ROW, "Only CUBLASLT_ORDER_ROW is supported for order_bias");
    ORT_ENFORCE(order_output_ == CUBLASLT_ORDER_ROW, "Only CUBLASLT_ORDER_ROW is supported for order_output");
  } else if (order_input_ == CUBLASLT_ORDER_COL32) {
    ORT_ENFORCE(order_weight_ == CUBLASLT_ORDER_COL4_4R2_8C || order_weight_ == CUBLASLT_ORDER_COL32_2R_4R4,
                "Only CUBLASLT_ORDER_COL4_4R2_8C, CUBLASLT_ORDER_COL32_2R_4R4 are supported for order_weight_");
    ORT_ENFORCE(order_bias_ == CUBLASLT_ORDER_COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_bias");
    ORT_ENFORCE(order_output_ == CUBLASLT_ORDER_COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_output");
  } else {
    ORT_ENFORCE(false, "Only CUBLASLT_ORDER_ROW or CUBLASLT_ORDER_COL32 are supported for order_input");
  }
  qkv_weight_const_count_ = scale_qkv_weight_const_count_ = qkv_bias_const_cout_ = 0;
  const_sacle_input_ = const_scale_qkv_layer_[0] = const_scale_qkv_layer_[1] = const_scale_qkv_layer_[2] = 0.0f;
}

Status QOrderedAttention::PutIntoMergedWeight(const Tensor& tensor, AllocatorPtr alloc, int qkv_index) {
  auto shape = tensor.Shape().GetDims();
  ORT_ENFORCE(shape.size() == 2, "QKV weight must be 2d tensors!");
  if (input_hidden_size_ == 0) {
    input_hidden_size_ = shape[0];
  } else {
    ORT_ENFORCE(input_hidden_size_ == shape[0], "QKV weight's shape[0] should be same!");
  }
  ORT_ENFORCE(qkv_hidden_sizes_.size() == 3, "qkv_hidden_sizes is needed!");
  ORT_ENFORCE(qkv_hidden_sizes_[qkv_index] == shape[1], "qkv hidden size not match on qkv_id:", qkv_index);
  ++qkv_weight_const_count_;
  int64_t qkv_total_hidden_size = qkv_hidden_sizes_[0] + qkv_hidden_sizes_[1] + qkv_hidden_sizes_[2];
  if (!merged_qkv_weight_) {
    auto* merged_qkv_weight_data = alloc->Alloc(input_hidden_size_ * qkv_total_hidden_size * sizeof(float));
    merged_qkv_weight_ = BufferUniquePtr(merged_qkv_weight_data, BufferDeleter(alloc));
  }
  auto column_offset = std::accumulate(&qkv_hidden_sizes_[0], &qkv_hidden_sizes_[qkv_index], 0LL);
  int8_t* start = ((int8_t*)merged_qkv_weight_.get()) + column_offset * input_hidden_size_;
  auto stream = Stream();
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(start, tensor.Data<int8_t>(), qkv_hidden_sizes_[qkv_index] * input_hidden_size_,
                                       cudaMemcpyDeviceToDevice, stream));
  return Status::OK();
}

Status QOrderedAttention::PutIntoMergedWeightScale(const Tensor& tensor, AllocatorPtr alloc, int qkv_index) {
  ORT_ENFORCE(tensor.Shape().IsScalar() || tensor.Shape().NumDimensions() == 1, "qkv gemm scale must be scalar or 1d vector");
  scale_qkv_weight_const_count_++;
  ORT_ENFORCE(qkv_hidden_sizes_.size() == 3, "qkv_hidden_sizes is needed!");
  ORT_ENFORCE(input_hidden_size_ > 0, "input_hidden_size_ can not be zero!");
  int64_t qkv_total_hidden_size = qkv_hidden_sizes_[0] + qkv_hidden_sizes_[1] + qkv_hidden_sizes_[2];

  if (!merged_qkv_alpha_) {
    auto* merged_alpha_data = alloc->Alloc(3 * qkv_total_hidden_size * sizeof(float));
    merged_qkv_alpha_ = BufferUniquePtr(merged_alpha_data, BufferDeleter(alloc));
  }
  auto column_offset = std::accumulate(&qkv_hidden_sizes_[0], &qkv_hidden_sizes_[qkv_index], 0LL);
  float* start = ((float*)merged_qkv_alpha_.get()) + column_offset;
  int incr = (tensor.Shape().IsScalar() ? 0 : 1);
  CUBLAS_RETURN_IF_ERROR(cublasScopy(CublasHandle(), gsl::narrow_cast<int>(qkv_hidden_sizes_[qkv_index]), tensor.Data<float>(), incr, start, 1));
  ORT_ENFORCE(const_sacle_input_ > 0.0f && const_scale_qkv_layer_[qkv_index] > 0.0f, "input scale and repective qkv gemm scale must be constant!");
  float scale = const_sacle_input_ / const_scale_qkv_layer_[qkv_index];
  CUBLAS_RETURN_IF_ERROR(cublasSscal(CublasHandle(), gsl::narrow_cast<int>(qkv_hidden_sizes_[qkv_index]), &scale, start, 1));
  return Status::OK();
}

Status QOrderedAttention::PutIntoMergedBias(const Tensor& tensor, AllocatorPtr alloc, int qkv_index) {
  ORT_ENFORCE(tensor.Shape().NumDimensions() == 1, "bias must be 1d vector");
  qkv_bias_const_cout_++;
  ORT_ENFORCE(qkv_hidden_sizes_.size() == 3, "qkv_hidden_sizes is needed!");
  ORT_ENFORCE(input_hidden_size_ > 0, "input_hidden_size_ can not be zero!");
  ORT_ENFORCE(qkv_hidden_sizes_[qkv_index] == tensor.Shape().GetDims().back(), "qkv hidden size is not matching");
  int64_t qkv_total_hidden_size = qkv_hidden_sizes_[0] + qkv_hidden_sizes_[1] + qkv_hidden_sizes_[2];

  if (!merged_qkv_bias_) {
    auto* merged_bias_data = alloc->Alloc(qkv_total_hidden_size * sizeof(float));
    merged_qkv_bias_ = BufferUniquePtr(merged_bias_data, BufferDeleter(alloc));
  }
  auto column_offset = std::accumulate(&qkv_hidden_sizes_[0], &qkv_hidden_sizes_[qkv_index], 0LL);
  float* start = ((float*)merged_qkv_bias_.get()) + column_offset;
  CUBLAS_RETURN_IF_ERROR(cublasScopy(CublasHandle(), gsl::narrow_cast<int>(qkv_hidden_sizes_[qkv_index]), tensor.Data<float>(), 1, start, 1));
  ORT_ENFORCE(const_scale_qkv_layer_[qkv_index] > 0.0f, "repective qkv gemm scale must be constant!");
  float scale = 1.0f / const_scale_qkv_layer_[qkv_index];
  CUBLAS_RETURN_IF_ERROR(cublasSscal(CublasHandle(), gsl::narrow_cast<int>(qkv_hidden_sizes_[qkv_index]), &scale, start, 1));
  return Status::OK();
}

Status QOrderedAttention::PrePack(const Tensor& tensor, int input_idx, /*out*/ AllocatorPtr alloc,
                                  /*out*/ bool& is_packed,
                                  /*out*/ PrePackedWeights* /*prepacked_weights*/) {
  is_packed = false;

  if (input_idx == InputIds::ScaleInput) {
    const_sacle_input_ = *tensor.Data<float>();
  }
  if (input_idx >= InputIds::Scale_Q_Gemm && input_idx < InputIds::Scale_Q_Gemm + 3) {
    const_scale_qkv_layer_[input_idx - InputIds::Scale_Q_Gemm] = *tensor.Data<float>();
  }

  if (input_idx >= InputIds::Q_Weight && input_idx < InputIds::Q_Weight + 3) {
    is_packed = true;
    ORT_RETURN_IF_ERROR(PutIntoMergedWeight(tensor, alloc, input_idx - InputIds::Q_Weight));
  }

  if (input_idx >= InputIds::Scale_Q_Weight && input_idx < InputIds::Scale_Q_Weight + 3) {
    is_packed = true;
    ORT_RETURN_IF_ERROR(PutIntoMergedWeightScale(tensor, alloc, input_idx - InputIds::Scale_Q_Weight));
  }

  if (input_idx >= InputIds::Q_Bias && input_idx < InputIds::Q_Bias + 3) {
    is_packed = true;
    ORT_RETURN_IF_ERROR(PutIntoMergedBias(tensor, alloc, input_idx - InputIds::Q_Bias));
  }

  if (input_idx == InputIds::Scale_QK_Gemm) {
    ORT_ENFORCE(qkv_hidden_sizes_.size() == 3, "Need it to calc head sizes!");
    float scale = *tensor.Data<float>();
    float base = std::exp(scale / sqrtf(qkv_hidden_sizes_[0] / num_heads_));
    auto* softmax_lookup_data = alloc->Alloc(256 * sizeof(float));
    softmax_lookup_ = BufferUniquePtr(softmax_lookup_data, BufferDeleter(alloc));
    BuildTableForSoftmaxPowerOf(Stream(), base, (float*)softmax_lookup_.get());
  }

  return Status::OK();
}

Status QOrderedAttention::ComputeInternal(OpKernelContext* context) const {
  ORT_ENFORCE(qkv_bias_const_cout_ == 3 && scale_qkv_weight_const_count_ == 3 && qkv_weight_const_count_ == 3,
              "qkv gemm weight and their scales, qkv gemm bias must all be constant!");
  ORT_ENFORCE(const_sacle_input_ > 0.0f, "input scale must be constant");
  ORT_ENFORCE(std::all_of(&const_scale_qkv_layer_[0], &const_scale_qkv_layer_[3], [](float v) -> bool { return v > 0.0f; }),
              "All qkv gemm output scale must be constant!");
  // inputs are column based
  const Tensor* input = context->Input<Tensor>(InputIds::Input);
  TensorShapeVector merged_shape = {input_hidden_size_, qkv_hidden_sizes_[0] + qkv_hidden_sizes_[1] + qkv_hidden_sizes_[2]};
  TensorShape merged_weights_shape(merged_shape);
  TensorShape merged_bias_shape{qkv_hidden_sizes_[0] + qkv_hidden_sizes_[1] + qkv_hidden_sizes_[2]};
  const Tensor* mask_index = context->Input<Tensor>(InputIds::Mask_Index);

  auto& device_prop = GetDeviceProp();
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(), merged_weights_shape, merged_bias_shape,
                                  mask_index, nullptr, nullptr, device_prop.maxThreadsPerBlock));

  const Tensor* tensor_scale_attn_scores = context->Input<Tensor>(InputIds::Scale_QK_Gemm);
  const float* scale_attn_scores_data = tensor_scale_attn_scores->Data<float>();

  const Tensor* tensor_scale_attn_probs = context->Input<Tensor>(InputIds::Scale_QK_Softmax);
  const float* scale_attn_probs_data = tensor_scale_attn_probs->Data<float>();

  const Tensor* scale_output_tensor = context->Input<Tensor>(InputIds::Scale_Values_Gemm);
  const float* scale_output_data = scale_output_tensor->template Data<float>();

  // input shape (batch_size, sequence_length, input_hidden_size)
  constexpr int max_threads_per_block = 256;
  const auto& shape = input->Shape();
  int batch_size = static_cast<int>(shape[0]);
  int sequence_length = static_cast<int>(shape[1]);
  int input_hidden_size = static_cast<int>(shape[2]);
  int hidden_size = qkv_hidden_sizes_[0];
  int head_size = hidden_size / num_heads_;

  TensorShapeVector output_shape(3);
  output_shape[0] = shape[0];
  output_shape[1] = shape[1];
  output_shape[2] = static_cast<int64_t>(hidden_size);
  Tensor* output = context->Output(0, output_shape);

  cublasLtHandle_t cublasLt = CublasLtHandle();
  // Use GEMM for fully connection.
  int m = batch_size * sequence_length;
  int n = 3 * hidden_size;
  int k = input_hidden_size;
  auto gemm_buffer_quantized = GetScratchBuffer<int8_t>(2 * m * n);  // merged and transposed

  cudaStream_t stream = Stream();
  // Gemm result (M, N) = alpha * input * weights + scale_bias.
  ORT_RETURN_IF_ERROR(QOrdered_MatMul(cublasLt, stream, device_prop,
                                      1, m, n, k,
                                      (const float*)merged_qkv_alpha_.get(), input->template Data<int8_t>(), (const int8_t*)merged_qkv_weight_.get(),
                                      (const float*)merged_qkv_bias_.get(), gemm_buffer_quantized.get(),
                                      (cublasLtOrder_t)order_weight_,
                                      CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO));

  int8_t* stacked_qkv_layers = gemm_buffer_quantized.get();
  int8_t* tranposed_qkv_layers = stacked_qkv_layers + ((int64_t)m * n);

  // BxSx3xNxH => 3xBxNxSxH, treat 4 consecutive int8 as float
  LaunchTransQkv(stream, 3, sequence_length, batch_size, head_size / sizeof(float), num_heads_,
                 max_threads_per_block, false, (const float*)stacked_qkv_layers, (float*)tranposed_qkv_layers);

  const int8_t* q_layer = tranposed_qkv_layers;
  const int8_t* k_layer = q_layer + ((int64_t)sequence_length * hidden_size);
  const int8_t* v_layer = k_layer + (k_layer - q_layer);
  int8_t* attention_scores = stacked_qkv_layers;                       // reuse it
  int8_t* attention_probs = stacked_qkv_layers + (k_layer - q_layer);  // attention_probs
  int8_t* context_layer = attention_probs + (k_layer - q_layer);

  const float q_mm_k_alpha = const_scale_qkv_layer_[0] * const_scale_qkv_layer_[1] / *scale_attn_scores_data;
  ORT_RETURN_IF_ERROR(QOrdered_MatMul(cublasLt, stream, device_prop,
                                      batch_size * num_heads_, sequence_length, sequence_length, head_size,
                                      &q_mm_k_alpha, q_layer, k_layer, nullptr, attention_scores,
                                      (cublasLtOrder_t)order_weight_));

  ORT_ENFORCE(softmax_lookup_.get() != nullptr, "qk_gemm scale must be constent");
  // the div sqrt(head_size) was processed when building the softmax lookup table
  QOrderMaskedSoftmax(stream, device_prop, attention_scores, (const float*)softmax_lookup_.get(),
                      mask_index->Data<int32_t>(), attention_probs, *scale_attn_probs_data,
                      batch_size, num_heads_, sequence_length);

  const float context_layer_alpha = *scale_attn_probs_data * *scale_attn_scores_data / *scale_output_data;
  ORT_RETURN_IF_ERROR(QOrdered_MatMul(cublasLt, stream, device_prop,
                                      batch_size * num_heads_, sequence_length, head_size, sequence_length,
                                      &context_layer_alpha, attention_probs, v_layer, nullptr, context_layer,
                                      (cublasLtOrder_t)order_weight_));

  // scratch3 is BxNxSxH, transpose to output SxBxNxH
  LaunchTransCtx(stream, sequence_length, batch_size, head_size / sizeof(float), num_heads_,
                 max_threads_per_block, false, (const float*)context_layer, (float*)output->MutableData<int8_t>());

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
