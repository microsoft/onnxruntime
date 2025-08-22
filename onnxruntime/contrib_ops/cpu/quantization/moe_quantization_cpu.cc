// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/quantization/moe_quantization_cpu.h"
#include "contrib_ops/cpu/moe/moe_utils.h"
#include "contrib_ops/cpu/quantization/moe_helper.h"
#include "core/framework/allocator.h"
#include "core/framework/buffer_deleter.h"
#include "core/mlas/inc/mlas.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/mlas/inc/mlas_float16.h"  // For MLAS_Half2Float function
#include <atomic>
#include "core/platform/threadpool.h"
#include <algorithm>
#include <cmath>    // For std::abs
#include <cstdlib>  // For std::getenv
#include <cstring>  // For std::strcmp
#include <cfloat>   // For FLT_MAX
#include <mutex>
#include <limits>  // For std::numeric_limits
#include <vector>

using namespace onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {

namespace {

// Helper function to perform top-k selection and bucket tokens by expert
void SelectTopKAndBucketTokens(
    const float* gating_output,
    int64_t num_rows,
    int64_t num_experts,
    int64_t k,
    std::vector<int>& route_expert,
    std::vector<float>& route_scale,
    std::vector<std::vector<int64_t>>& buckets) {
  route_expert.resize(static_cast<size_t>(num_rows * k));
  route_scale.resize(static_cast<size_t>(num_rows * k));
  buckets.assign(static_cast<size_t>(num_experts), std::vector<int64_t>());

  std::vector<std::pair<float, int64_t>> expert_scores;
  expert_scores.reserve(static_cast<size_t>(num_experts));

  for (int64_t row = 0; row < num_rows; ++row) {
    expert_scores.clear();

    std::vector<std::pair<float, int64_t>> all_experts;
    for (int64_t e = 0; e < num_experts; ++e) {
      float logit = gating_output[row * num_experts + e];
      all_experts.emplace_back(logit, e);
    }

    std::partial_sort(all_experts.begin(), all_experts.begin() + static_cast<ptrdiff_t>(std::min(k, num_experts)), all_experts.end(),
                      [](const std::pair<float, int64_t>& a, const std::pair<float, int64_t>& b) {
                        return a.first > b.first;
                      });

    const int64_t actual_k = std::min(k, num_experts);
    std::vector<float> top_logits(static_cast<size_t>(actual_k));
    for (int64_t i = 0; i < actual_k; ++i) {
      top_logits[static_cast<size_t>(i)] = all_experts[static_cast<size_t>(i)].first;
    }

    float max_logit = -std::numeric_limits<float>::infinity();
    if (actual_k > 0) {
      max_logit = *std::max_element(top_logits.begin(), top_logits.end());
    }

    float sum_exp = 0.0f;
    for (int64_t i = 0; i < actual_k; ++i) {
      top_logits[static_cast<size_t>(i)] = std::exp(top_logits[static_cast<size_t>(i)] - max_logit);
      sum_exp += top_logits[static_cast<size_t>(i)];
    }

    for (int64_t i = 0; i < actual_k; ++i) {
      float normalized_weight = sum_exp == 0.0f ? 0.0f : top_logits[static_cast<size_t>(i)] / sum_exp;
      expert_scores.emplace_back(normalized_weight, all_experts[static_cast<size_t>(i)].second);
    }
    const int64_t select = std::min(k, static_cast<int64_t>(expert_scores.size()));

    for (int64_t i = 0; i < select; ++i) {
      const int64_t off = row * k + i;
      route_expert[static_cast<size_t>(off)] = static_cast<int>(expert_scores[static_cast<size_t>(i)].second);
      route_scale[static_cast<size_t>(off)] = expert_scores[static_cast<size_t>(i)].first;
    }
    for (int64_t i = select; i < k; ++i) {
      const int64_t off = row * k + i;
      route_expert[static_cast<size_t>(off)] = 0;
      route_scale[static_cast<size_t>(off)] = 0.0f;
    }
  }

  for (int64_t row = 0; row < num_rows; ++row) {
    for (int64_t i = 0; i < k; ++i) {
      const int64_t off = row * k + i;
      const int e = route_expert[static_cast<size_t>(off)];
      const float s = route_scale[static_cast<size_t>(off)];
      if (s > 0.0f) buckets[static_cast<size_t>(e)].push_back(off);
    }
  }
}

// Helper function to dequantize weights for a single expert
void DequantizeExpertWeights(
    std::vector<float>& dequant_weights,
    const uint8_t* quantized_weights,
    const float* scales,
    size_t N, size_t K,
    int bit_width,
    uint8_t zero_point) {
  dequant_weights.resize(N * K);
  const size_t bytes_per_expert = (N * K * bit_width) / 8;

  if (bit_width == 8) {
    for (size_t n = 0; n < N; ++n) {
      const float sc = scales[n];
      for (size_t kx = 0; kx < K; ++kx) {
        size_t physical_idx = n * K + kx;
        if (physical_idx < bytes_per_expert) {
          uint8_t quantized_val = quantized_weights[physical_idx];
          float dequantized_val = sc * (static_cast<float>(quantized_val) - static_cast<float>(zero_point));
          dequant_weights[n * K + kx] = std::clamp(dequantized_val, -1e6f, 1e6f);
        } else {
          dequant_weights[n * K + kx] = 0.0f;
        }
      }
    }
  } else {  // 4-bit
    for (size_t n = 0; n < N; ++n) {
      const float sc = scales[n];
      for (size_t kx = 0; kx < K; kx += 2) {
        size_t byte_idx = (n * K + kx) / 2;
        if (byte_idx < bytes_per_expert) {
          const uint8_t packed_byte = quantized_weights[byte_idx];
          const uint8_t val_even = packed_byte & 0x0F;
          const uint8_t val_odd = (packed_byte >> 4) & 0x0F;
          float dequant_even = sc * (static_cast<float>(val_even) - static_cast<float>(zero_point));
          float dequant_odd = sc * (static_cast<float>(val_odd) - static_cast<float>(zero_point));
          dequant_weights[n * K + kx] = std::clamp(dequant_even, -1e6f, 1e6f);
          if (kx + 1 < K) {
            dequant_weights[n * K + kx + 1] = std::clamp(dequant_odd, -1e6f, 1e6f);
          }
        } else {
          dequant_weights[n * K + kx] = 0.0f;
          if (kx + 1 < K) {
            dequant_weights[n * K + kx + 1] = 0.0f;
          }
        }
      }
    }
  }
}

// Helper function to run the MLP for a single expert
void RunExpertMLP(
    float* output,
    const float* input_activations,
    const std::vector<int64_t>& routes,
    const std::vector<float>& route_scale,
    int64_t k,
    const float* B1_deq,
    const float* fc1_bias,
    const float* B2_deq,
    const float* fc2_bias,
    int64_t hidden_size,
    int64_t inter_size,
    float activation_alpha,
    float activation_beta,
    float swiglu_limit,
    onnxruntime::concurrency::ThreadPool* thread_pool) {
  const size_t Me = routes.size();
  if (Me == 0) return;

  const size_t K1 = static_cast<size_t>(hidden_size);
  const size_t N1 = static_cast<size_t>(2 * inter_size);
  const size_t K2 = static_cast<size_t>(inter_size);
  const size_t N2 = static_cast<size_t>(hidden_size);

  std::vector<float> A1(Me * K1);
  for (size_t r = 0; r < Me; ++r) {
    const int64_t off = routes[r];
    const int64_t row = off / k;
    const float* src_row = input_activations + row * hidden_size;
    std::copy(src_row, src_row + K1, A1.data() + r * K1);
  }

  std::vector<float> C1(Me * N1);
  MLAS_SGEMM_DATA_PARAMS d1{};
  d1.A = A1.data();
  d1.lda = K1;
  d1.B = B1_deq;
  d1.ldb = K1;
  d1.C = C1.data();
  d1.ldc = N1;
  d1.alpha = 1.0f;
  d1.beta = 0.0f;
  MlasGemm(CblasNoTrans, CblasTrans, Me, N1, K1, d1, thread_pool);

  if (fc1_bias) {
    for (size_t r = 0; r < Me; ++r) {
      float* rowp = C1.data() + r * N1;
      for (size_t c = 0; c < N1; ++c) rowp[c] += fc1_bias[c];
    }
  }

  for (size_t r = 0; r < Me; ++r) {
    contrib::ApplySwiGLUActivation(C1.data() + r * N1, inter_size, true, activation_alpha, activation_beta, swiglu_limit);
  }

  std::vector<float> A2(Me * K2);
  for (size_t r = 0; r < Me; ++r) {
    const float* src = C1.data() + r * N1;
    float* dst = A2.data() + r * K2;
    std::copy(src, src + K2, dst);
  }

  std::vector<float> C2(Me * N2);
  MLAS_SGEMM_DATA_PARAMS d2{};
  d2.A = A2.data();
  d2.lda = K2;
  d2.B = B2_deq;
  d2.ldb = K2;
  d2.C = C2.data();
  d2.ldc = N2;
  d2.alpha = 1.0f;
  d2.beta = 0.0f;
  MlasGemm(CblasNoTrans, CblasTrans, Me, N2, K2, d2, thread_pool);

  for (size_t r = 0; r < Me; ++r) {
    const int64_t off = routes[r];
    const int64_t row = off / k;
    const float scale = route_scale[static_cast<size_t>(off)];
    float* out_row = output + row * hidden_size;
    const float* c2_row = C2.data() + r * N2;

    if (fc2_bias) {
      for (size_t c = 0; c < N2; ++c) {
        out_row[c] += scale * (c2_row[c] + fc2_bias[c]);
      }
    } else {
      for (size_t c = 0; c < N2; ++c) {
        out_row[c] += scale * c2_row[c];
      }
    }
  }
}

static void run_moe_fc_cpu_grouped(
    const float* input_activations,
    const float* gating_output,
    const void* fc1_weights_q,
    const float* fc1_scales,
    const float* fc1_bias_f32,
    const void* fc2_weights_q,
    const float* fc2_scales,
    const float* fc2_bias_f32,
    int bit_width,
    int64_t num_rows,
    int64_t hidden_size,
    int64_t inter_size,
    int64_t num_experts,
    int64_t k,
    float activation_alpha,
    float activation_beta,
    float swiglu_limit,
    float* output,
    onnxruntime::concurrency::ThreadPool* thread_pool) {
  const int64_t fc1_out = 2 * inter_size;
  const uint8_t global_zero_point = static_cast<uint8_t>(bit_width == 8 ? 128 : 8);

  std::fill_n(output, static_cast<size_t>(num_rows * hidden_size), 0.0f);

  std::vector<int> route_expert;
  std::vector<float> route_scale;
  std::vector<std::vector<int64_t>> buckets;
  SelectTopKAndBucketTokens(gating_output, num_rows, num_experts, k, route_expert, route_scale, buckets);

  const uint8_t* fc1_wq = reinterpret_cast<const uint8_t*>(fc1_weights_q);
  const uint8_t* fc2_wq = reinterpret_cast<const uint8_t*>(fc2_weights_q);

  for (int64_t e = 0; e < num_experts; ++e) {
    const auto& routes = buckets[static_cast<size_t>(e)];
    if (routes.empty()) continue;

    std::vector<float> B1_deq, B2_deq;
    const size_t K1 = static_cast<size_t>(hidden_size);
    const size_t N1 = static_cast<size_t>(fc1_out);
    const size_t K2 = static_cast<size_t>(inter_size);
    const size_t N2 = static_cast<size_t>(hidden_size);

    const size_t bytes_per_expert_fc1 = (N1 * K1 * bit_width) / 8;
    const size_t bytes_per_expert_fc2 = (N2 * K2 * bit_width) / 8;

    DequantizeExpertWeights(B1_deq, fc1_wq + static_cast<size_t>(e) * bytes_per_expert_fc1,
                            fc1_scales + static_cast<size_t>(e) * N1, N1, K1, bit_width, global_zero_point);
    DequantizeExpertWeights(B2_deq, fc2_wq + static_cast<size_t>(e) * bytes_per_expert_fc2,
                            fc2_scales + static_cast<size_t>(e) * N2, N2, K2, bit_width, global_zero_point);

    RunExpertMLP(output, input_activations, routes, route_scale, k,
                 B1_deq.data(), fc1_bias_f32 ? fc1_bias_f32 + static_cast<size_t>(e) * N1 : nullptr,
                 B2_deq.data(), fc2_bias_f32 ? fc2_bias_f32 + static_cast<size_t>(e) * N2 : nullptr,
                 hidden_size, inter_size, activation_alpha, activation_beta, swiglu_limit, thread_pool);
  }
}

}  // namespace

template <typename T>
QMoE<T>::QMoE(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info), MoEBaseCPU(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
  ORT_ENFORCE(expert_weight_bits_ == 8 || expert_weight_bits_ == 4 || expert_weight_bits_ == 0,
              "expert_weight_bits must be 0 (FP32), 4, or 8, but got ", expert_weight_bits_);

  swiglu_fusion_ = op_kernel_info.GetAttrOrDefault<int64_t>("swiglu_fusion", 0);
  block_size_ = op_kernel_info.GetAttrOrDefault<int64_t>("block_size", -1);

  ORT_ENFORCE(activation_type_ == ActivationType::SwiGLU,
              "CPU QMoE kernel supports only 'swiglu' interleaved activation");
}

template <typename T>
Status QMoE<T>::Compute(OpKernelContext* ctx) const {
  const Tensor* input = ctx->Input<Tensor>(0);
  const Tensor* router_probs = ctx->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = ctx->Input<Tensor>(2);
  const Tensor* fc1_scales = ctx->Input<Tensor>(3);
  const Tensor* fc1_experts_bias_optional = ctx->Input<Tensor>(4);
  const Tensor* fc2_experts_weights = ctx->Input<Tensor>(5);
  const Tensor* fc2_scales = ctx->Input<Tensor>(6);
  const Tensor* fc2_experts_bias_optional = ctx->Input<Tensor>(7);
  const Tensor* fc3_experts_weights_optional = ctx->Input<Tensor>(8);
  const Tensor* fc3_scales_optional = ctx->Input<Tensor>(9);
  const Tensor* fc3_experts_bias_optional = ctx->Input<Tensor>(10);

  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs,
      fc1_experts_weights, fc1_experts_bias_optional, fc1_scales,
      fc2_experts_weights, fc2_experts_bias_optional, fc2_scales,
      fc3_experts_weights_optional, fc3_experts_bias_optional, fc3_scales_optional,
      expert_weight_bits_ == 4 ? 2 : 1,
      true));

  if (expert_weight_bits_ == 0) {
    return DirectFP32MoEImpl(ctx, moe_params, input, router_probs,
                             fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                             fc2_experts_bias_optional, fc3_experts_weights_optional,
                             fc3_experts_bias_optional);
  } else if (expert_weight_bits_ == 4) {
    return QuantizedMoEImpl<true>(ctx, moe_params, input, router_probs,
                                  fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                  fc2_experts_bias_optional, fc3_experts_weights_optional,
                                  fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional);
  } else {
    return QuantizedMoEImpl<false>(ctx, moe_params, input, router_probs,
                                   fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                   fc2_experts_bias_optional, fc3_experts_weights_optional,
                                   fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional);
  }
}

template <typename T>
template <bool UseUInt4x2>
Status QMoE<T>::QuantizedMoEImpl(OpKernelContext* context,
                                 MoEParameters& moe_params,
                                 const Tensor* input,
                                 const Tensor* router_probs,
                                 const Tensor* fc1_experts_weights,
                                 const Tensor* fc1_experts_bias_optional,
                                 const Tensor* fc2_experts_weights,
                                 const Tensor* fc2_experts_bias_optional,
                                 const Tensor* fc3_experts_weights_optional,
                                 const Tensor* fc3_experts_bias_optional,
                                 const Tensor* fc1_scales,
                                 const Tensor* fc2_scales,
                                 const Tensor* fc3_scales_optional) const {
  if (fc3_experts_weights_optional || fc3_experts_bias_optional || fc3_scales_optional) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "FC3 gating is not yet implemented on CPU for QMoE");
  }
  ORT_UNUSED_PARAMETER(fc3_experts_weights_optional);
  ORT_UNUSED_PARAMETER(fc3_experts_bias_optional);
  ORT_UNUSED_PARAMETER(fc3_scales_optional);

  auto* thread_pool = context->GetOperatorThreadPool();
  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  const T* input_data = input->Data<T>();
  const T* router_probs_data = router_probs->Data<T>();

  const size_t input_size = static_cast<size_t>(moe_params.num_rows * moe_params.hidden_size);
  const size_t router_size = static_cast<size_t>(moe_params.num_rows * moe_params.num_experts);
  const size_t output_size = static_cast<size_t>(moe_params.num_rows * moe_params.hidden_size);

  const T* fc1_bias_data = fc1_experts_bias_optional ? fc1_experts_bias_optional->Data<T>() : nullptr;
  const T* fc2_bias_data = fc2_experts_bias_optional ? fc2_experts_bias_optional->Data<T>() : nullptr;

  Tensor* output = context->Output(0, input->Shape());
  T* output_data = output->MutableData<T>();

  auto input_float = IAllocator::MakeUniquePtr<float>(allocator, input_size);
  auto router_float = IAllocator::MakeUniquePtr<float>(allocator, router_size);
  auto output_float = IAllocator::MakeUniquePtr<float>(allocator, output_size);

  if constexpr (std::is_same_v<T, MLFloat16>) {
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(input_data),
                                 input_float.get(), input_size);
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(router_probs_data),
                                 router_float.get(), router_size);
  } else {
    std::copy(input_data, input_data + input_size, input_float.get());
    std::copy(router_probs_data, router_probs_data + router_size, router_float.get());
  }

  std::unique_ptr<float[]> fc1_bias_float, fc2_bias_float;
  if (fc1_bias_data) {
    const size_t fc1_bias_size = static_cast<size_t>(moe_params.num_experts * (2 * moe_params.inter_size));
    fc1_bias_float = std::make_unique<float[]>(fc1_bias_size);
    if constexpr (std::is_same_v<T, MLFloat16>) {
      MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(fc1_bias_data), fc1_bias_float.get(), fc1_bias_size);
    } else {
      std::copy(fc1_bias_data, fc1_bias_data + fc1_bias_size, fc1_bias_float.get());
    }
  }
  if (fc2_bias_data) {
    const size_t fc2_bias_size = static_cast<size_t>(moe_params.num_experts * moe_params.hidden_size);
    fc2_bias_float = std::make_unique<float[]>(fc2_bias_size);
    if constexpr (std::is_same_v<T, MLFloat16>) {
      MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(fc2_bias_data), fc2_bias_float.get(), fc2_bias_size);
    } else {
      std::copy(fc2_bias_data, fc2_bias_data + fc2_bias_size, fc2_bias_float.get());
    }
  }

  const void* fc1_wq = fc1_experts_weights->DataRaw();
  const void* fc2_wq = fc2_experts_weights->DataRaw();

  const size_t fc1_scales_size = static_cast<size_t>(moe_params.num_experts) * (2 * static_cast<size_t>(moe_params.inter_size));
  const size_t fc2_scales_size = static_cast<size_t>(moe_params.num_experts) * static_cast<size_t>(moe_params.hidden_size);
  auto fc1_scales_float = IAllocator::MakeUniquePtr<float>(allocator, fc1_scales_size);
  auto fc2_scales_float = IAllocator::MakeUniquePtr<float>(allocator, fc2_scales_size);

  if (fc1_scales->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(fc1_scales->Data<MLFloat16>()), fc1_scales_float.get(), fc1_scales_size);
  } else {
    std::copy(fc1_scales->Data<float>(), fc1_scales->Data<float>() + fc1_scales_size, fc1_scales_float.get());
  }

  if (fc2_scales->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(fc2_scales->Data<MLFloat16>()), fc2_scales_float.get(), fc2_scales_size);
  } else {
    std::copy(fc2_scales->Data<float>(), fc2_scales->Data<float>() + fc2_scales_size, fc2_scales_float.get());
  }

  run_moe_fc_cpu_grouped(
      input_float.get(), router_float.get(),
      fc1_wq, fc1_scales_float.get(), fc1_bias_float.get(),
      fc2_wq, fc2_scales_float.get(), fc2_bias_float.get(),
      UseUInt4x2 ? 4 : 8,
      moe_params.num_rows, moe_params.hidden_size, moe_params.inter_size, moe_params.num_experts, k_,
      activation_alpha_, activation_beta_, swiglu_limit_, output_float.get(), thread_pool);

  if constexpr (std::is_same_v<T, MLFloat16>) {
    MlasConvertFloatToHalfBuffer(output_float.get(),
                                 reinterpret_cast<MLAS_FP16*>(output_data),
                                 output_size);
  } else {
    std::copy(output_float.get(), output_float.get() + output_size, output_data);
  }

  return Status::OK();
}

template <typename T>
Status QMoE<T>::DirectFP32MoEImpl(OpKernelContext* context,
                                  MoEParameters& moe_params,
                                  const Tensor* input,
                                  const Tensor* router_probs,
                                  const Tensor* fc1_experts_weights,
                                  const Tensor* fc1_experts_bias_optional,
                                  const Tensor* fc2_experts_weights,
                                  const Tensor* fc2_experts_bias_optional,
                                  const Tensor* fc3_experts_weights_optional,
                                  const Tensor* fc3_experts_bias_optional) const {
  if (fc3_experts_weights_optional || fc3_experts_bias_optional) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "FC3 gating is not yet implemented on CPU for FP32 MoE");
  }
  ORT_UNUSED_PARAMETER(fc3_experts_weights_optional);
  ORT_UNUSED_PARAMETER(fc3_experts_bias_optional);

  auto* thread_pool = context->GetOperatorThreadPool();
  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  const T* input_data = input->Data<T>();
  const T* router_probs_data = router_probs->Data<T>();
  const T* fc1_bias_data = fc1_experts_bias_optional ? fc1_experts_bias_optional->Data<T>() : nullptr;
  const T* fc2_bias_data = fc2_experts_bias_optional ? fc2_experts_bias_optional->Data<T>() : nullptr;

  const float* fc1_weights_data = fc1_experts_weights->Data<float>();
  const float* fc2_weights_data = fc2_experts_weights->Data<float>();

  Tensor* output = context->Output(0, input->Shape());
  T* output_data = output->MutableData<T>();

  const size_t input_size = static_cast<size_t>(moe_params.num_rows * moe_params.hidden_size);
  const size_t router_size = static_cast<size_t>(moe_params.num_rows * moe_params.num_experts);
  const size_t output_size = static_cast<size_t>(moe_params.num_rows * moe_params.hidden_size);

  auto input_float = IAllocator::MakeUniquePtr<float>(allocator, input_size);
  auto router_float = IAllocator::MakeUniquePtr<float>(allocator, router_size);
  auto output_float = IAllocator::MakeUniquePtr<float>(allocator, output_size);

  if constexpr (std::is_same_v<T, float>) {
    std::copy(input_data, input_data + input_size, input_float.get());
    std::copy(router_probs_data, router_probs_data + router_size, router_float.get());
  } else if constexpr (std::is_same_v<T, MLFloat16>) {
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(input_data), input_float.get(), input_size);
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(router_probs_data), router_float.get(), router_size);
  }

  std::fill(output_float.get(), output_float.get() + output_size, 0.0f);

  std::unique_ptr<float[]> fc1_bias_float, fc2_bias_float;
  if (fc1_bias_data) {
    const size_t fc1_bias_size = static_cast<size_t>(moe_params.num_experts * (2 * moe_params.inter_size));
    fc1_bias_float = std::make_unique<float[]>(fc1_bias_size);
    if constexpr (std::is_same_v<T, MLFloat16>) {
      MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(fc1_bias_data), fc1_bias_float.get(), fc1_bias_size);
    } else {
      std::copy(fc1_bias_data, fc1_bias_data + fc1_bias_size, fc1_bias_float.get());
    }
  }
  if (fc2_bias_data) {
    const size_t fc2_bias_size = static_cast<size_t>(moe_params.num_experts * moe_params.hidden_size);
    fc2_bias_float = std::make_unique<float[]>(fc2_bias_size);
    if constexpr (std::is_same_v<T, MLFloat16>) {
      MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(fc2_bias_data), fc2_bias_float.get(), fc2_bias_size);
    } else {
      std::copy(fc2_bias_data, fc2_bias_data + fc2_bias_size, fc2_bias_float.get());
    }
  }

  std::vector<int> route_expert;
  std::vector<float> route_scale;
  std::vector<std::vector<int64_t>> buckets;
  SelectTopKAndBucketTokens(router_float.get(), moe_params.num_rows, moe_params.num_experts, k_, route_expert, route_scale, buckets);

  const int64_t fc1_out = 2 * moe_params.inter_size;
  for (int64_t e = 0; e < moe_params.num_experts; ++e) {
    const auto& routes = buckets[static_cast<size_t>(e)];
    if (routes.empty()) continue;

    const float* B1_deq = fc1_weights_data + static_cast<size_t>(e * fc1_out * moe_params.hidden_size);
    const float* B2_deq = fc2_weights_data + static_cast<size_t>(e * moe_params.hidden_size * moe_params.inter_size);

    RunExpertMLP(output_float.get(), input_float.get(), routes, route_scale, k_,
                 B1_deq, fc1_bias_float ? fc1_bias_float.get() + static_cast<size_t>(e * fc1_out) : nullptr,
                 B2_deq, fc2_bias_float ? fc2_bias_float.get() + static_cast<size_t>(e * moe_params.hidden_size) : nullptr,
                 moe_params.hidden_size, moe_params.inter_size,
                 activation_alpha_, activation_beta_, swiglu_limit_, thread_pool);
  }

  if constexpr (std::is_same_v<T, float>) {
    std::copy(output_float.get(), output_float.get() + output_size, output_data);
  } else if constexpr (std::is_same_v<T, MLFloat16>) {
    MlasConvertFloatToHalfBuffer(output_float.get(), reinterpret_cast<MLAS_FP16*>(output_data), output_size);
  }

  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QMoE,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .MayInplace(0, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T1", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<float>()})
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<MLFloat16>()}),
    QMoE<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QMoE,
    kMSDomain,
    1,
    MLFloat16,
    kCpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .MayInplace(0, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T1", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<MLFloat16>()})
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<MLFloat16>()}),
    QMoE<MLFloat16>);

template class QMoE<float>;
template class QMoE<MLFloat16>;

}  // namespace contrib
}  // namespace onnxruntime