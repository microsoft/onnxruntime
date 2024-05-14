// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_attention.h"
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_attention_impl.h"
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_qdq_impl.h"
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_matmul_utils.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
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
#include "qordered_attention_input_enum.h"
};
#undef DefineQOrderedAttentionInput

// double check the definitions orders are correct
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

Status QOrderedAttention::PutIntoMergedWeight(const Tensor& tensor, AllocatorPtr alloc, int qkv_index, cudaStream_t cuda_stream) {
  ++qkv_weight_const_count_;
  ORT_ENFORCE(tensor.Shape().NumDimensions() == 2, "QKV weight must be 2d tensors!");
  input_hidden_size_ = (input_hidden_size_ == 0 ? tensor.Shape()[0] : input_hidden_size_);
  ORT_ENFORCE(input_hidden_size_ == tensor.Shape()[0] && input_hidden_size_ > 0, "QKV weight's shape[0] should be same positive value!");
  ORT_ENFORCE(qkv_hidden_sizes_[qkv_index] == tensor.Shape()[1], "qkv hidden size not match with qkv_hidden_sizes on qkv_id:", qkv_index);
  if (!merged_qkv_weight_) {
    merged_qkv_weight_ = BufferUniquePtr(alloc->Alloc(input_hidden_size_ * qkv_total_hidden_size_), BufferDeleter(alloc));
  }
  auto offset = std::accumulate(&qkv_hidden_sizes_[0], &qkv_hidden_sizes_[qkv_index], 0LL);
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(((int8_t*)merged_qkv_weight_.get()) + (offset * input_hidden_size_),
                                       tensor.Data<int8_t>(), qkv_hidden_sizes_[qkv_index] * input_hidden_size_,
                                       cudaMemcpyDeviceToDevice, cuda_stream));
  return Status::OK();
}

Status QOrderedAttention::PutIntoMergedWeightScale(const Tensor& tensor, AllocatorPtr alloc, int qkv_index) {
  ++scale_qkv_weight_const_count_;
  ORT_ENFORCE(tensor.Shape().IsScalar() || (tensor.Shape().NumDimensions() == 1 && qkv_hidden_sizes_[qkv_index] == tensor.Shape()[0]),
              "qkv gemm scale is not scalar or 1d vector, or not same dims as in qkv_hidden_sizes at qkv_index:", qkv_index);
  if (!merged_qkv_alpha_) {
    merged_qkv_alpha_ = BufferUniquePtr(alloc->Alloc(qkv_total_hidden_size_ * sizeof(float)), BufferDeleter(alloc));
  }
  auto offset = std::accumulate(&qkv_hidden_sizes_[0], &qkv_hidden_sizes_[qkv_index], 0LL);
  float* target = ((float*)merged_qkv_alpha_.get()) + offset;
  int count = gsl::narrow_cast<int>(qkv_hidden_sizes_[qkv_index]);
  CUBLAS_RETURN_IF_ERROR(cublasScopy(DefaultCublasHandle(), count, tensor.Data<float>(), tensor.Shape().IsScalar() ? 0 : 1, target, 1));
  ORT_ENFORCE(const_scale_input_ > 0.0f && const_scale_qkv_layer_[qkv_index] > 0.0f,
              "input scale and respective qkv gemm scale must be positive constant!");
  float scale = static_cast<float>((double)const_scale_input_ / const_scale_qkv_layer_[qkv_index]);
  CUBLAS_RETURN_IF_ERROR(cublasSscal(DefaultCublasHandle(), count, &scale, target, 1));
  return Status::OK();
}

Status QOrderedAttention::PutIntoMergedBias(const Tensor& tensor, AllocatorPtr alloc, int qkv_index) {
  ++qkv_bias_const_cout_;
  ORT_ENFORCE(tensor.Shape().NumDimensions() == 1, "bias must be 1d vector");
  ORT_ENFORCE(qkv_hidden_sizes_[qkv_index] == tensor.Shape()[0], "qkv hidden size is not matching qkv_hidden_sizes at qkv_index:", qkv_index);
  if (!merged_qkv_bias_) {
    merged_qkv_bias_ = BufferUniquePtr(alloc->Alloc(qkv_total_hidden_size_ * sizeof(float)), BufferDeleter(alloc));
  }
  auto offset = std::accumulate(&qkv_hidden_sizes_[0], &qkv_hidden_sizes_[qkv_index], 0LL);
  float* target = ((float*)merged_qkv_bias_.get()) + offset;
  int count = gsl::narrow_cast<int>(qkv_hidden_sizes_[qkv_index]);
  CUBLAS_RETURN_IF_ERROR(cublasScopy(DefaultCublasHandle(), count, tensor.Data<float>(), 1, target, 1));
  ORT_ENFORCE(const_scale_qkv_layer_[qkv_index] > 0.0f, "qkv gemm scale should be positive constant at qkv_index", qkv_index);
  float scale = static_cast<float>(1.0 / const_scale_qkv_layer_[qkv_index]);
  CUBLAS_RETURN_IF_ERROR(cublasSscal(DefaultCublasHandle(), count, &scale, target, 1));
  return Status::OK();
}

Status QOrderedAttention::PrePack(const Tensor& tensor, int input_idx, /*out*/ AllocatorPtr alloc,
                                  /*out*/ bool& is_packed,
                                  /*out*/ PrePackedWeights* /*prepacked_weights*/) {
  is_packed = false;

  if (input_idx == InputIds::ScaleInput) {
    const_scale_input_ = *tensor.Data<float>();
  } else if (input_idx >= InputIds::Scale_Q_Gemm && input_idx < InputIds::Scale_Q_Gemm + 3) {
    const_scale_qkv_layer_[input_idx - InputIds::Scale_Q_Gemm] = *tensor.Data<float>();
  } else if (input_idx >= InputIds::Q_Weight && input_idx < InputIds::Q_Weight + 3) {
    is_packed = true;
    ORT_RETURN_IF_ERROR(PutIntoMergedWeight(tensor, alloc, input_idx - InputIds::Q_Weight, nullptr));
  } else if (input_idx >= InputIds::Scale_Q_Weight && input_idx < InputIds::Scale_Q_Weight + 3) {
    is_packed = true;
    ORT_RETURN_IF_ERROR(PutIntoMergedWeightScale(tensor, alloc, input_idx - InputIds::Scale_Q_Weight));
  } else if (input_idx >= InputIds::Q_Bias && input_idx < InputIds::Q_Bias + 3) {
    is_packed = true;
    ORT_RETURN_IF_ERROR(PutIntoMergedBias(tensor, alloc, input_idx - InputIds::Q_Bias));
  } else if (input_idx == InputIds::Scale_QK_Gemm) {
    float scale = *tensor.Data<float>();
    double base = std::exp((double)scale / sqrt(qkv_hidden_sizes_[0] / static_cast<double>(num_heads_)));
    auto* softmax_lookup_data = alloc->Alloc(256 * sizeof(float));
    softmax_lookup_ = BufferUniquePtr(softmax_lookup_data, BufferDeleter(alloc));
    ORT_RETURN_IF_ERROR(BuildTableForSoftmaxPowerOf(nullptr, base, (float*)softmax_lookup_.get()));
  }

  cudaStreamSynchronize(nullptr);
  return Status::OK();
}

template <typename T>
inline void debug_print([[maybe_unused]] const T* arr,
                        [[maybe_unused]] const size_t sz,
                        [[maybe_unused]] const int w,
                        [[maybe_unused]] const char* name) {
#if defined(DEBUGPRINT_QORDERED_ATTENTION)

  cudaDeviceSynchronize();
  std::vector<T> buf(sz);
  cudaMemcpy(buf.data(), arr, sz * sizeof(T), cudaMemcpyDeviceToHost);

  std::cout << "========" << name << std::endl;
  for (size_t i = 0; i < sz; i++) {
    if (i % w == 0) std::cout << std::endl;
    if constepxr (std::is_same<T, int8_t>::value) {
      std::cout << (int)buf[i] << ", ";
    } else {
      std::cout << buf[i] << ", ";
    }
  }
  std::cout << std::endl;

#endif
}

QOrderedAttention::QOrderedAttention(const OpKernelInfo& info) : CudaKernel(info), AttentionBase(info, true) {
  input_hidden_size_ = 0;
  int cuda_runtime_version = 0;
  CUDA_CALL_THROW(cudaRuntimeGetVersion(&cuda_runtime_version));
  ORT_ENFORCE(cuda_runtime_version >= 11040, "QOrderedMatmul need cuda runtime higher than 11.4");
  auto& device_prop = GetDeviceProp();
  ORT_ENFORCE((device_prop.major * 10 + device_prop.minor) >= 75, "QOrderedMatmul need sm75 or highter");

  ORT_ENFORCE(qkv_hidden_sizes_.size() == 3, "qkv_hidden_sizes is needed and must be of shape [3]!");
  ORT_ENFORCE(std::all_of(qkv_hidden_sizes_.begin(), qkv_hidden_sizes_.end(),
                          [num_heads = this->num_heads_](int64_t v) { return (v > 0) && (v % num_heads) == 0; }),
              "All qkv hiddend_sizes must be positive and divisible by num_heads");
  ORT_ENFORCE(qkv_hidden_sizes_[0] == qkv_hidden_sizes_[1] && qkv_hidden_sizes_[0] == qkv_hidden_sizes_[2],
              "currently qkv hidden size should be same");
  qkv_total_hidden_size_ = qkv_hidden_sizes_[0] + qkv_hidden_sizes_[1] + qkv_hidden_sizes_[2];
  order_input_ = GetCublasLtOrderAttr(info, "order_input");
  order_weight_ = GetCublasLtOrderAttr(info, "order_weight");
  order_output_ = GetCublasLtOrderAttr(info, "order_output");

  ORT_ENFORCE(order_input_ == CUBLASLT_ORDER_ROW, "Currently only support ORDER_ROW input");
  ORT_ENFORCE(order_weight_ == CUBLASLT_ORDER_COL, "Only CUBLASLT_ORDER_COL is supported for order_weight_");
  ORT_ENFORCE(order_output_ == CUBLASLT_ORDER_ROW, "Only CUBLASLT_ORDER_ROW is supported for order_output");

  qkv_weight_const_count_ = scale_qkv_weight_const_count_ = qkv_bias_const_cout_ = 0;
  const_scale_input_ = const_scale_qkv_layer_[0] = const_scale_qkv_layer_[1] = const_scale_qkv_layer_[2] = 0.0f;
}

Status QOrderedAttention::ComputeInternal(OpKernelContext* context) const {
  ORT_ENFORCE(qkv_bias_const_cout_ == 3 && scale_qkv_weight_const_count_ == 3 && qkv_weight_const_count_ == 3,
              "qkv gemm weight and their scales, qkv gemm bias must all be constant!");
  ORT_ENFORCE(const_scale_input_ > 0.0f, "input scale must be constant");
  ORT_ENFORCE(std::all_of(&const_scale_qkv_layer_[0], &const_scale_qkv_layer_[3], [](float v) -> bool { return v > 0.0f; }),
              "All qkv gemm output scale must be constant!");
  ORT_ENFORCE(softmax_lookup_.get() != nullptr, "qk_gemm scale must be positive constant to make softmax prepared!");

  // inputs are column based
  const Tensor* input = context->Input<Tensor>(InputIds::Input);
  // TensorShapeVector{input_hidden_size_, qkv_total_hidden_size_}
  TensorShape merged_weights_shape{input_hidden_size_, qkv_total_hidden_size_};
  TensorShape merged_bias_shape{qkv_total_hidden_size_};
  const Tensor* mask_index = context->Input<Tensor>(InputIds::Mask_Index);

  auto& device_prop = GetDeviceProp();
  ORT_RETURN_IF_ERROR(CheckInputs(input->Shape(), merged_weights_shape, merged_bias_shape,
                                  mask_index,
                                  nullptr,  // past
                                  nullptr,  // relative_position_bias
                                  nullptr,  // parameters
                                  device_prop.maxThreadsPerBlock));

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
  int hidden_size = static_cast<int>(qkv_hidden_sizes_[0]);
  int head_size = hidden_size / num_heads_;
  ORT_ENFORCE(sequence_length % 16 == 0 && head_size % 16 == 0, "Unsupported size!");

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
  int64_t size_of_attention_scores = ((int64_t)batch_size) * num_heads_ * sequence_length * sequence_length;

  // transposed qkv_layer,  union(stacked, attention probs + attention scores)
  auto gemm_buffer_quantized = GetScratchBuffer<int8_t>((int64_t)m * n + std::max((int64_t)m * n, 2 * size_of_attention_scores), context->GetComputeStream());

  int8_t* stacked_qkv_layers = gemm_buffer_quantized.get() + ((int64_t)m * n);
  int8_t* tranposed_qkv_layers = gemm_buffer_quantized.get();
  int8_t* q_layer = tranposed_qkv_layers;
  int8_t* k_layer = tranposed_qkv_layers + ((int64_t)batch_size * sequence_length * hidden_size);
  int8_t* v_layer = tranposed_qkv_layers + ((int64_t)batch_size * sequence_length * hidden_size * 2);
  int8_t* v_layer_T = q_layer;                                               // rewrite q_layer after qk matmul
  int8_t* attention_scores = stacked_qkv_layers + size_of_attention_scores;  // rewrite to stacked_qkv_layers after they are transposed
  int8_t* attention_probs = stacked_qkv_layers;                              // rewrite to stacked_qkv_layers after they are transposed
  int8_t* context_layer = k_layer;                                           // rewrite k_layer after it is not used, it is float value

  cudaStream_t stream = Stream(context);
  // Gemm result (M, N) = alpha * input * weights + scale_bias.
  ORT_RETURN_IF_ERROR(QOrdered_MatMul(cublasLt, stream, device_prop,
                                      1, m, n, k,
                                      (const float*)merged_qkv_alpha_.get(), input->template Data<int8_t>(), (const int8_t*)merged_qkv_weight_.get(),
                                      (const float*)merged_qkv_bias_.get(), stacked_qkv_layers,
                                      CUBLASLT_ORDER_COL,
                                      (cublasLtPointerMode_t)4));  // CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST available after 11.4.2
  // debug_print(stacked_qkv_layers, m * n, hidden_size, "stacked_qkv_layer");

  // BxSx3xNxH => 3xBxNxSxH, treat 4 consecutive int8 as float
  ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 3, sequence_length, batch_size, head_size / sizeof(float), num_heads_,
                                     max_threads_per_block, false, (const float*)stacked_qkv_layers, (float*)tranposed_qkv_layers));
  // debug_print(q_layer, m * n / 3, head_size, "tranposed_q_layers BxNxSxH");
  // debug_print(k_layer, m * n / 3, head_size, "tranposed_k_layers BxNxSxH");
  // debug_print(v_layer, m * n / 3, head_size, "tranposed_v_layers BxNxSxH");

  const float q_mm_k_alpha = static_cast<float>((double)const_scale_qkv_layer_[0] * const_scale_qkv_layer_[1] / *scale_attn_scores_data);
  ORT_RETURN_IF_ERROR(QOrdered_MatMul(cublasLt, stream, device_prop,
                                      batch_size * num_heads_, sequence_length, sequence_length, head_size,
                                      &q_mm_k_alpha, q_layer, k_layer, batch_size * num_heads_,
                                      nullptr, nullptr, nullptr, 1, attention_scores,
                                      CUBLASLT_ORDER_COL));  // matrix B need extra transpose
  // debug_print(attention_scores, size_of_attention_scores, sequence_length, "attention_scores");

  // the div sqrt(head_size) was processed when building the softmax lookup table
  ORT_RETURN_IF_ERROR(QOrderMaskedSoftmax(stream, device_prop, attention_scores, (const float*)softmax_lookup_.get(),
                                          mask_index->Data<int32_t>(), attention_probs, *scale_attn_probs_data,
                                          batch_size, num_heads_, sequence_length));
  // debug_print(attention_probs, size_of_attention_scores, sequence_length, "attention_probs");

  // Transpose v_layer from BxNxSxH to BxNxHxS to use tensor core int8 matmul
  ORT_RETURN_IF_ERROR(QOrderBatchTransposeInt8Matrix(stream, device_prop, batch_size * num_heads_, sequence_length, head_size, v_layer, v_layer_T));

  const float context_layer_alpha = static_cast<float>((double)*scale_attn_probs_data * const_scale_qkv_layer_[2] / *scale_output_data);
  ORT_RETURN_IF_ERROR(QOrdered_MatMul(cublasLt, stream, device_prop,
                                      batch_size * num_heads_, sequence_length, head_size, sequence_length,
                                      &context_layer_alpha, attention_probs, v_layer_T, batch_size * num_heads_,
                                      nullptr, nullptr, nullptr, 1, context_layer,
                                      CUBLASLT_ORDER_COL));

  // scratch3 is BxNxSxH, transpose to output BxSxNxH
  ORT_RETURN_IF_ERROR(LaunchTransCtx(stream, sequence_length, batch_size, head_size / 4, num_heads_, device_prop.maxThreadsPerBlock,
                                     false, (const float*)context_layer, (float*)output->MutableData<int8_t>()));
  // debug_print(output->Data<int8_t>(), m * n / 3, head_size, "attention output");

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
