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

using namespace onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {

inline float quantize_router_probability(float prob, float quantization_step = 1e-5f) {
  return std::round(prob / quantization_step) * quantization_step;
}

inline bool use_robust_expert_selection() {
  static bool checked = false;
  static bool enabled = false;

  if (!checked) {
    const char* env_var = std::getenv("ORT_QMOE_ROBUST_SELECTION");
    enabled = (env_var != nullptr && std::strcmp(env_var, "1") == 0);
    checked = true;
  }

  return enabled;
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
    bool normalize_routing_weights,
    float activation_alpha,
    float activation_beta,
    float* output,
    onnxruntime::concurrency::ThreadPool* thread_pool) {
  ORT_UNUSED_PARAMETER(normalize_routing_weights);

  const int64_t fc1_out = 2 * inter_size;
  const uint8_t global_zero_point = static_cast<uint8_t>(bit_width == 8 ? 128 : 8);  // 8 for 4-bit symmetric, 128 for 8-bit

  std::fill_n(output, static_cast<size_t>(num_rows * hidden_size), 0.0f);

  std::vector<int> route_expert(static_cast<size_t>(num_rows * k));
  std::vector<float> route_scale(static_cast<size_t>(num_rows * k));

  std::vector<std::pair<float, int64_t>> expert_scores;
  expert_scores.reserve(static_cast<size_t>(num_experts));

  for (int64_t row = 0; row < num_rows; ++row) {
    expert_scores.clear();

    // Get all expert logits for this row
    std::vector<std::pair<float, int64_t>> all_experts;
    for (int64_t e = 0; e < num_experts; ++e) {
      float logit = gating_output[row * num_experts + e];
      all_experts.emplace_back(logit, e);
    }

    // Sort experts by logit value (descending)
    std::partial_sort(all_experts.begin(), all_experts.begin() + static_cast<ptrdiff_t>(std::min(k, num_experts)), all_experts.end(),
                      [](const std::pair<float, int64_t>& a, const std::pair<float, int64_t>& b) {
                        return a.first > b.first;
                      });

    // Get top-k experts
    const int64_t actual_k = std::min(k, num_experts);
    std::vector<float> top_logits(static_cast<size_t>(actual_k));
    for (int64_t i = 0; i < actual_k; ++i) {
      top_logits[static_cast<size_t>(i)] = all_experts[static_cast<size_t>(i)].first;
    }

    // Apply softmax to top-k logits for proper normalization (always, as per PyTorch)
    float max_logit = *std::max_element(top_logits.begin(), top_logits.end());
    float sum_exp = 0.0f;
    for (int64_t i = 0; i < actual_k; ++i) {
      top_logits[static_cast<size_t>(i)] = std::exp(top_logits[static_cast<size_t>(i)] - max_logit);
      sum_exp += top_logits[static_cast<size_t>(i)];
    }

    // Normalize weights (always apply softmax like PyTorch)
    for (int64_t i = 0; i < actual_k; ++i) {
      float normalized_weight = top_logits[static_cast<size_t>(i)] / sum_exp;
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

  std::vector<std::vector<int64_t>> buckets(static_cast<size_t>(num_experts));
  buckets.shrink_to_fit();
  for (int64_t row = 0; row < num_rows; ++row) {
    for (int64_t i = 0; i < k; ++i) {
      const int64_t off = row * k + i;
      const int e = route_expert[static_cast<size_t>(off)];
      const float s = route_scale[static_cast<size_t>(off)];
      if (s > 0.0f) buckets[static_cast<size_t>(e)].push_back(off);
    }
  }

  const size_t K1_logical = static_cast<size_t>(hidden_size);
  const size_t N1 = static_cast<size_t>(fc1_out);

  const size_t K2_logical = static_cast<size_t>(inter_size);
  const size_t N2 = static_cast<size_t>(hidden_size);

  const uint8_t* fc1_wq = reinterpret_cast<const uint8_t*>(fc1_weights_q);
  const uint8_t* fc2_wq = reinterpret_cast<const uint8_t*>(fc2_weights_q);

  // Process experts sequentially to avoid memory issues and thread safety problems
  for (int64_t e = 0; e < num_experts; ++e) {
    const auto& routes = buckets[static_cast<size_t>(e)];
    const size_t Me = routes.size();
    if (Me == 0) continue;

    std::vector<float> A1(Me * K1_logical);
    for (size_t r = 0; r < Me; ++r) {
      const int64_t off = routes[r];
      const int64_t row = off / k;
      const float* src_row = input_activations + row * hidden_size;
      std::copy(src_row, src_row + K1_logical, A1.data() + r * K1_logical);
    }

    std::vector<float> B1_deq(N1 * K1_logical);
    const float* s1 = fc1_scales + static_cast<size_t>(e) * N1;

    const size_t bytes_per_expert_fc1 = (N1 * K1_logical * bit_width) / 8;
    const size_t expert_base_fc1 = static_cast<size_t>(e) * bytes_per_expert_fc1;
    const uint8_t* expert_B1_q = fc1_wq + expert_base_fc1;

    if (bit_width == 8) {
      for (size_t n = 0; n < N1; ++n) {
        const float sc = s1[n];
        for (size_t kx = 0; kx < K1_logical; ++kx) {
          size_t physical_idx = n * K1_logical + kx;

          if (physical_idx < bytes_per_expert_fc1) {
            uint8_t quantized_val = expert_B1_q[physical_idx];
            float dequantized_val = sc * (static_cast<float>(quantized_val) - static_cast<float>(global_zero_point));
            // Clamp to prevent numerical overflow with very large scales
            dequantized_val = std::clamp(dequantized_val, -1e6f, 1e6f);
            B1_deq[n * K1_logical + kx] = dequantized_val;
          } else {
            B1_deq[n * K1_logical + kx] = 0.0f;
          }
        }
      }
    } else {
      for (size_t n = 0; n < N1; ++n) {
        const float sc = s1[n];
        for (size_t kx = 0; kx < K1_logical; kx += 2) {
          size_t byte_idx = (n * K1_logical + kx) / 2;

          if (byte_idx < bytes_per_expert_fc1) {
            const uint8_t packed_byte = expert_B1_q[byte_idx];

            const uint8_t val_even = packed_byte & 0x0F;
            const uint8_t val_odd = (packed_byte >> 4) & 0x0F;

            float dequant_even = sc * (static_cast<float>(val_even) - static_cast<float>(global_zero_point));
            float dequant_odd = sc * (static_cast<float>(val_odd) - static_cast<float>(global_zero_point));

            // Clamp to prevent numerical overflow with very large scales
            dequant_even = std::clamp(dequant_even, -1e6f, 1e6f);
            dequant_odd = std::clamp(dequant_odd, -1e6f, 1e6f);

            B1_deq[n * K1_logical + kx] = dequant_even;
            if (kx + 1 < K1_logical) {
              B1_deq[n * K1_logical + kx + 1] = dequant_odd;
            }

          } else {
            B1_deq[n * K1_logical + kx] = 0.0f;
            if (kx + 1 < K1_logical) {
              B1_deq[n * K1_logical + kx + 1] = 0.0f;
            }
          }
        }
      }
    }

    std::vector<float> C1(Me * N1);

    MLAS_SGEMM_DATA_PARAMS d1{};
    d1.A = A1.data();
    d1.lda = K1_logical;
    d1.B = B1_deq.data();
    d1.ldb = K1_logical;
    d1.C = C1.data();
    d1.ldc = N1;
    d1.alpha = 1.0f;
    d1.beta = 0.0f;
    MlasGemm(CblasNoTrans, CblasTrans, Me, N1, K1_logical, d1, thread_pool);

    if (fc1_bias_f32) {
      const float* b = fc1_bias_f32 + static_cast<size_t>(e) * N1;
      for (size_t r = 0; r < Me; ++r) {
        float* rowp = C1.data() + r * N1;
        for (size_t c = 0; c < N1; ++c) rowp[c] += b[c];
      }
    }

    for (size_t r = 0; r < Me; ++r) {
      contrib::ApplySwiGLUActivation(C1.data() + r * N1, inter_size, true, activation_alpha, activation_beta);
    }

    std::vector<float> A2(Me * K2_logical);
    for (size_t r = 0; r < Me; ++r) {
      const float* src = C1.data() + r * N1;
      float* dst = A2.data() + r * K2_logical;
      std::copy(src, src + K2_logical, dst);
    }

    std::vector<float> B2_deq(N2 * K2_logical);
    const float* s2 = fc2_scales + static_cast<size_t>(e) * N2;

    const size_t bytes_per_expert_fc2 = (N2 * K2_logical * bit_width) / 8;
    const size_t expert_base_fc2 = static_cast<size_t>(e) * bytes_per_expert_fc2;
    const uint8_t* expert_B2_q = fc2_wq + expert_base_fc2;

    if (bit_width == 8) {
      for (size_t n = 0; n < N2; ++n) {
        const float sc = s2[n];
        for (size_t kx = 0; kx < K2_logical; ++kx) {
          size_t physical_idx = n * K2_logical + kx;

          if (physical_idx < bytes_per_expert_fc2) {
            uint8_t quantized_val = expert_B2_q[physical_idx];
            float dequantized_val = sc * (static_cast<float>(quantized_val) - static_cast<float>(global_zero_point));
            // Clamp to prevent numerical overflow with very large scales
            dequantized_val = std::clamp(dequantized_val, -1e6f, 1e6f);
            B2_deq[n * K2_logical + kx] = dequantized_val;
          } else {
            B2_deq[n * K2_logical + kx] = 0.0f;
          }
        }
      }
    } else {
      for (size_t n = 0; n < N2; ++n) {
        const float sc = s2[n];
        for (size_t kx = 0; kx < K2_logical; kx += 2) {
          size_t byte_idx = (n * K2_logical + kx) / 2;

          if (byte_idx < bytes_per_expert_fc2) {
            const uint8_t packed_byte = expert_B2_q[byte_idx];

            const uint8_t val_even = packed_byte & 0x0F;
            const uint8_t val_odd = (packed_byte >> 4) & 0x0F;

            float dequant_even = sc * (static_cast<float>(val_even) - static_cast<float>(global_zero_point));
            float dequant_odd = sc * (static_cast<float>(val_odd) - static_cast<float>(global_zero_point));

            // Clamp to prevent numerical overflow with very large scales
            dequant_even = std::clamp(dequant_even, -1e6f, 1e6f);
            dequant_odd = std::clamp(dequant_odd, -1e6f, 1e6f);

            B2_deq[n * K2_logical + kx] = dequant_even;
            if (kx + 1 < K2_logical) {
              B2_deq[n * K2_logical + kx + 1] = dequant_odd;
            }
          } else {
            B2_deq[n * K2_logical + kx] = 0.0f;
            if (kx + 1 < K2_logical) {
              B2_deq[n * K2_logical + kx + 1] = 0.0f;
            }
          }
        }
      }
    }

    std::vector<float> C2(Me * N2);
    MLAS_SGEMM_DATA_PARAMS d2{};
    d2.A = A2.data();
    d2.lda = K2_logical;
    d2.B = B2_deq.data();
    d2.ldb = K2_logical;
    d2.C = C2.data();
    d2.ldc = N2;
    d2.alpha = 1.0f;
    d2.beta = 0.0f;
    MlasGemm(CblasNoTrans, CblasTrans, Me, N2, K2_logical, d2, thread_pool);

    for (size_t r = 0; r < Me; ++r) {
      const int64_t off = routes[r];
      const int64_t row = off / k;
      const float scale = route_scale[static_cast<size_t>(off)];
      float* out_row = output + row * hidden_size;
      const float* c2_row = C2.data() + r * N2;

      if (fc2_bias_f32) {
        const float* b2 = fc2_bias_f32 + static_cast<size_t>(e) * N2;
        for (size_t c = 0; c < N2; ++c) {
          out_row[c] += scale * (c2_row[c] + b2[c]);
        }
      } else {
        for (size_t c = 0; c < N2; ++c) {
          out_row[c] += scale * c2_row[c];
        }
      }
    }
  }
}

void finalize_moe_routing_cpu(const float* expert_outputs, float* final_output,
                              const float* fc2_bias_float, const float* expert_scales,
                              const int* expert_indices, int64_t num_rows, int64_t hidden_size, int64_t k) {
  for (int64_t row = 0; row < num_rows; ++row) {
    float* output_row = final_output + row * hidden_size;

    std::fill_n(output_row, hidden_size, 0.0f);

    for (int64_t k_idx = 0; k_idx < k; ++k_idx) {
      const int64_t expert_offset = row * k + k_idx;
      const int64_t expert_idx = expert_indices[static_cast<size_t>(expert_offset)];
      const float expert_scale = expert_scales[static_cast<size_t>(expert_offset)];

      const float* expert_output_row = expert_outputs + expert_offset * hidden_size;
      const float* bias_ptr = fc2_bias_float ? fc2_bias_float + expert_idx * hidden_size : nullptr;

      for (int64_t col = 0; col < hidden_size; ++col) {
        const float bias_value = bias_ptr ? bias_ptr[col] : 0.0f;
        output_row[col] += expert_scale * (expert_output_row[col] + bias_value);
      }
    }
  }
}

void run_moe_fc_cpu(const float* input_activations, const float* gating_output,
                    const float* fc1_expert_weights, const float* fc1_scales, const float* fc1_expert_biases,
                    const float* fc2_expert_weights, const float* fc2_scales,
                    int64_t num_rows, int64_t hidden_size, int64_t inter_size, int64_t num_experts, int64_t k,
                    bool normalize_routing_weights,
                    float* expert_outputs, float* expert_scales_output, int* expert_indices_output,
                    onnxruntime::concurrency::ThreadPool* thread_pool,
                    float activation_alpha = 1.702f, float activation_beta = 1.0f) {
  ORT_UNUSED_PARAMETER(fc1_scales);
  ORT_UNUSED_PARAMETER(fc2_scales);

  const int64_t fc1_output_size = 2 * inter_size;

  std::vector<std::pair<float, int64_t>> expert_scores;
  std::vector<float> routing_weights(static_cast<size_t>(num_rows * k));
  std::vector<int> expert_indices(static_cast<size_t>(num_rows * k));

  for (int64_t row = 0; row < num_rows; ++row) {
    expert_scores.clear();
    expert_scores.reserve(static_cast<size_t>(num_experts));

    for (int64_t expert_idx = 0; expert_idx < num_experts; ++expert_idx) {
      const int64_t router_idx = row * num_experts + expert_idx;
      float routing_weight = gating_output[router_idx];

      if (use_robust_expert_selection()) {
        routing_weight = quantize_router_probability(routing_weight);
      }

      expert_scores.emplace_back(routing_weight, expert_idx);
    }

    auto robust_comparator = [](const std::pair<float, int64_t>& a, const std::pair<float, int64_t>& b) {
      constexpr float epsilon = 1e-6f;

      if (std::abs(a.first - b.first) < epsilon) {
        return a.second < b.second;
      }

      return a.first > b.first;
    };

    std::partial_sort(expert_scores.begin(), expert_scores.begin() + static_cast<ptrdiff_t>(k),
                      expert_scores.end(), robust_comparator);
    float selected_sum = 0.0f;
    for (int64_t k_idx = 0; k_idx < k; ++k_idx) {
      selected_sum += expert_scores[static_cast<size_t>(k_idx)].first;
    }

    for (int64_t k_idx = 0; k_idx < k; ++k_idx) {
      const int64_t offset = row * k + k_idx;
      expert_indices[static_cast<size_t>(offset)] = static_cast<int>(expert_scores[static_cast<size_t>(k_idx)].second);

      if (normalize_routing_weights) {
        routing_weights[static_cast<size_t>(offset)] = (selected_sum > 0.0f) ? (expert_scores[static_cast<size_t>(k_idx)].first / selected_sum) : 0.0f;
      } else {
        routing_weights[static_cast<size_t>(offset)] = expert_scores[static_cast<size_t>(k_idx)].first;
      }
    }
  }

  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(num_rows * k),
      static_cast<double>(hidden_size * inter_size * 0.1),
      [&](std::ptrdiff_t start, std::ptrdiff_t end) {
        std::vector<float> fc1_output(static_cast<size_t>(fc1_output_size));

        for (std::ptrdiff_t idx = start; idx < end; ++idx) {
          const int64_t row = idx / k;
          const int64_t expert_idx = expert_indices[static_cast<size_t>(idx)];

          const float* input_row = input_activations + row * hidden_size;
          float* output_row = expert_outputs + idx * hidden_size;

          const float* fc1_weights = fc1_expert_weights + expert_idx * hidden_size * fc1_output_size;

          MlasGemm(CblasNoTrans, CblasTrans,
                   1, static_cast<size_t>(fc1_output_size), static_cast<size_t>(hidden_size),
                   1.0f,
                   input_row, static_cast<size_t>(hidden_size),
                   fc1_weights, static_cast<size_t>(hidden_size),
                   0.0f,
                   fc1_output.data(), static_cast<size_t>(fc1_output_size),
                   nullptr);

          if (fc1_expert_biases) {
            const float* fc1_bias = fc1_expert_biases + expert_idx * fc1_output_size;
            for (int64_t i = 0; i < fc1_output_size; ++i) {
              fc1_output[static_cast<size_t>(i)] += fc1_bias[i];
            }
          }

          contrib::ApplySwiGLUActivation(fc1_output.data(), fc1_output_size / 2, true,
                                         activation_alpha, activation_beta);

          const int64_t actual_inter_size = fc1_output_size / 2;
          const float* fc2_weights = fc2_expert_weights + expert_idx * actual_inter_size * hidden_size;

          MlasGemm(CblasNoTrans, CblasTrans,
                   1, static_cast<size_t>(hidden_size), static_cast<size_t>(actual_inter_size),
                   1.0f,
                   fc1_output.data(), static_cast<size_t>(actual_inter_size),
                   fc2_weights, static_cast<size_t>(actual_inter_size),
                   0.0f,
                   output_row, static_cast<size_t>(hidden_size),
                   nullptr);
        }
      });

  std::copy(routing_weights.begin(), routing_weights.end(), expert_scales_output);
  std::copy(expert_indices.begin(), expert_indices.end(), expert_indices_output);
}

template <typename T>
QMoE<T>::QMoE(const OpKernelInfo& op_kernel_info) : OpKernel(op_kernel_info), MoEBaseCPU(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
  ORT_ENFORCE(expert_weight_bits_ == 8 || expert_weight_bits_ == 4 || expert_weight_bits_ == 0,
              "expert_weight_bits must be 0 (FP32), 4, or 8, but got ", expert_weight_bits_);

  // Handle new attributes from updated specification
  swiglu_fusion_ = op_kernel_info.GetAttrOrDefault<int64_t>("swiglu_fusion", 0);
  swiglu_limit_ = op_kernel_info.GetAttrOrDefault<float>("swiglu_limit", std::numeric_limits<float>::infinity());
  block_size_ = op_kernel_info.GetAttrOrDefault<int64_t>("block_size", -1);  // -1 means not specified

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
      /*is_fused_swiglu*/ true));

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
Status QMoE<T>::PrepackAndDequantizeWeights(OpKernelContext* context,
                                            MoEParameters& moe_params,
                                            const Tensor* fc1_experts_weights,
                                            const Tensor* fc2_experts_weights,
                                            const Tensor* fc1_scales,
                                            const Tensor* fc2_scales,
                                            bool is_swiglu) {
  if (!weights_allocator_) {
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&weights_allocator_));
  }

  const int64_t fc1_output_size = is_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size;
  const size_t fc1_weights_size = static_cast<size_t>(moe_params.num_experts * moe_params.hidden_size * fc1_output_size);
  const size_t fc2_weights_size = static_cast<size_t>(moe_params.num_experts * moe_params.inter_size * moe_params.hidden_size);

  prepacked_fc1_weights_ = IAllocator::MakeUniquePtr<float>(weights_allocator_, fc1_weights_size);
  prepacked_fc2_weights_ = IAllocator::MakeUniquePtr<float>(weights_allocator_, fc2_weights_size);
  prepacked_fc1_weights_data_ = prepacked_fc1_weights_.get();
  prepacked_fc2_weights_data_ = prepacked_fc2_weights_.get();

  if (expert_weight_bits_ == 0) {
    if (fc1_experts_weights->IsDataType<float>() && fc2_experts_weights->IsDataType<float>()) {
      const float* fc1_weights_data = fc1_experts_weights->Data<float>();
      const float* fc2_weights_data = fc2_experts_weights->Data<float>();

      std::copy(fc1_weights_data, fc1_weights_data + fc1_weights_size, prepacked_fc1_weights_data_);

      std::copy(fc2_weights_data, fc2_weights_data + fc2_weights_size, prepacked_fc2_weights_data_);

    } else {
      expert_weight_bits_ = 8;
    }
  }

  if (expert_weight_bits_ != 0) {
    const uint8_t* fc1_weights_data = fc1_experts_weights->Data<uint8_t>();
    const uint8_t* fc2_weights_data = fc2_experts_weights->Data<uint8_t>();
    const T* fc1_scales_data = fc1_scales->Data<T>();
    const T* fc2_scales_data = fc2_scales->Data<T>();

    if constexpr (UseUInt4x2) {
      for (int64_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
        for (int64_t col = 0; col < fc1_output_size; ++col) {
          const float scale = static_cast<float>(fc1_scales_data[expert_idx * fc1_output_size + col]);
          for (int64_t row = 0; row < moe_params.hidden_size; row += 2) {
            const size_t byte_idx = static_cast<size_t>(expert_idx * fc1_output_size * (moe_params.hidden_size / 2) +
                                                        col * (moe_params.hidden_size / 2) + (row / 2));
            const uint8_t packed_val = fc1_weights_data[byte_idx];

            const uint8_t val_even = packed_val & 0x0F;
            const uint8_t val_odd = (packed_val >> 4) & 0x0F;

            const float dequantized_val_even = scale * (static_cast<float>(val_even) - 8.0f);
            const float dequantized_val_odd = scale * (static_cast<float>(val_odd) - 8.0f);

            prepacked_fc1_weights_data_[static_cast<size_t>(expert_idx * fc1_output_size * moe_params.hidden_size + col * moe_params.hidden_size + row)] = dequantized_val_even;
            if (row + 1 < moe_params.hidden_size) {
              prepacked_fc1_weights_data_[static_cast<size_t>(expert_idx * fc1_output_size * moe_params.hidden_size + col * moe_params.hidden_size + row + 1)] = dequantized_val_odd;
            }
          }
        }
      }
    } else {
      for (int64_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
        for (int64_t col = 0; col < fc1_output_size; ++col) {
          const float scale = static_cast<float>(fc1_scales_data[expert_idx * fc1_output_size + col]);
          for (int64_t row = 0; row < moe_params.hidden_size; ++row) {
            const size_t idx = static_cast<size_t>(expert_idx * fc1_output_size * moe_params.hidden_size + col * moe_params.hidden_size + row);
            const uint8_t val = fc1_weights_data[idx];

            const float dequantized_val = scale * (static_cast<float>(val) - 128.0f);
            prepacked_fc1_weights_data_[idx] = dequantized_val;
          }
        }
      }
    }
    if constexpr (UseUInt4x2) {
      for (int64_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
        for (int64_t col = 0; col < moe_params.hidden_size; ++col) {
          const float scale = static_cast<float>(fc2_scales_data[expert_idx * moe_params.hidden_size + col]);
          for (int64_t row = 0; row < moe_params.inter_size; row += 2) {
            const size_t byte_idx = static_cast<size_t>(expert_idx * moe_params.hidden_size * (moe_params.inter_size / 2) +
                                                        col * (moe_params.inter_size / 2) + (row / 2));
            const uint8_t packed_val = fc2_weights_data[byte_idx];

            const uint8_t val_even = packed_val & 0x0F;
            const uint8_t val_odd = (packed_val >> 4) & 0x0F;

            const float dequantized_val_even = scale * (static_cast<float>(val_even) - 8.0f);
            const float dequantized_val_odd = scale * (static_cast<float>(val_odd) - 8.0f);

            prepacked_fc2_weights_data_[static_cast<size_t>(expert_idx * moe_params.hidden_size * moe_params.inter_size + col * moe_params.inter_size + row)] = dequantized_val_even;
            if (row + 1 < moe_params.inter_size) {
              prepacked_fc2_weights_data_[static_cast<size_t>(expert_idx * moe_params.hidden_size * moe_params.inter_size + col * moe_params.inter_size + row + 1)] = dequantized_val_odd;
            }
          }
        }
      }
    } else {
      for (int64_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
        for (int64_t col = 0; col < moe_params.hidden_size; ++col) {
          const float scale = static_cast<float>(fc2_scales_data[expert_idx * moe_params.hidden_size + col]);
          for (int64_t row = 0; row < moe_params.inter_size; ++row) {
            const size_t idx = static_cast<size_t>(expert_idx * moe_params.hidden_size * moe_params.inter_size + col * moe_params.inter_size + row);
            const uint8_t val = fc2_weights_data[idx];

            const float dequantized_val = scale * (static_cast<float>(val) - 128.0f);
            prepacked_fc2_weights_data_[idx] = dequantized_val;
          }
        }
      }
    }
  }

  cached_num_experts_ = moe_params.num_experts;
  cached_hidden_size_ = moe_params.hidden_size;
  cached_inter_size_ = moe_params.inter_size;
  cached_is_swiglu_ = is_swiglu;
  is_prepacked_ = true;

  return Status::OK();
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
  // If any FC3 input is present, return not implemented error for CPU
  if (fc3_experts_weights_optional || fc3_experts_bias_optional || fc3_scales_optional) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "FC3 gating is not yet implemented");
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

  const int64_t correct_k = k_;

  std::unique_ptr<float[]> fc1_bias_float, fc2_bias_float;
  if (fc1_bias_data) {
    const size_t fc1_bias_size = static_cast<size_t>(moe_params.num_experts * (2 * moe_params.inter_size));
    fc1_bias_float = std::make_unique<float[]>(fc1_bias_size);
    std::unique_ptr<MLAS_FP16[]> fc1_bias_fp16 = std::make_unique<MLAS_FP16[]>(fc1_bias_size);
    if constexpr (std::is_same_v<T, MLFloat16>) {
      for (size_t i = 0; i < fc1_bias_size; ++i) fc1_bias_fp16[i] = reinterpret_cast<const MLAS_FP16*>(fc1_bias_data)[i];
    } else {
      MlasConvertFloatToHalfBuffer(fc1_bias_data, fc1_bias_fp16.get(), fc1_bias_size);
    }
    MlasConvertHalfToFloatBuffer(fc1_bias_fp16.get(), fc1_bias_float.get(), fc1_bias_size);
  }
  if (fc2_bias_data) {
    const size_t fc2_bias_size = static_cast<size_t>(moe_params.num_experts * moe_params.hidden_size);
    fc2_bias_float = std::make_unique<float[]>(fc2_bias_size);
    std::unique_ptr<MLAS_FP16[]> fc2_bias_fp16 = std::make_unique<MLAS_FP16[]>(fc2_bias_size);
    if constexpr (std::is_same_v<T, MLFloat16>) {
      for (size_t i = 0; i < fc2_bias_size; ++i) fc2_bias_fp16[i] = reinterpret_cast<const MLAS_FP16*>(fc2_bias_data)[i];
    } else {
      MlasConvertFloatToHalfBuffer(fc2_bias_data, fc2_bias_fp16.get(), fc2_bias_size);
    }
    MlasConvertHalfToFloatBuffer(fc2_bias_fp16.get(), fc2_bias_float.get(), fc2_bias_size);
  }

  const void* fc1_wq = fc1_experts_weights->DataRaw();
  const void* fc2_wq = fc2_experts_weights->DataRaw();

  const size_t fc1_scales_size = static_cast<size_t>(moe_params.num_experts) * (2 * static_cast<size_t>(moe_params.inter_size));
  const size_t fc2_scales_size = static_cast<size_t>(moe_params.num_experts) * static_cast<size_t>(moe_params.hidden_size);
  std::unique_ptr<float[]> fc1_scales_float = std::make_unique<float[]>(fc1_scales_size);
  std::unique_ptr<float[]> fc2_scales_float = std::make_unique<float[]>(fc2_scales_size);
  {
    std::unique_ptr<MLAS_FP16[]> fc1_scales_fp16 = std::make_unique<MLAS_FP16[]>(fc1_scales_size);
    std::unique_ptr<MLAS_FP16[]> fc2_scales_fp16 = std::make_unique<MLAS_FP16[]>(fc2_scales_size);
    if (fc1_scales->GetElementType() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
      const MLFloat16* src1 = fc1_scales->Data<MLFloat16>();
      const MLFloat16* src2 = fc2_scales->Data<MLFloat16>();
      for (size_t i = 0; i < fc1_scales_size; ++i) fc1_scales_fp16[i] = src1[i];
      for (size_t i = 0; i < fc2_scales_size; ++i) fc2_scales_fp16[i] = src2[i];
    } else {
      const float* src1 = fc1_scales->Data<float>();
      const float* src2 = fc2_scales->Data<float>();
      MlasConvertFloatToHalfBuffer(src1, fc1_scales_fp16.get(), fc1_scales_size);
      MlasConvertFloatToHalfBuffer(src2, fc2_scales_fp16.get(), fc2_scales_size);
    }
    MlasConvertHalfToFloatBuffer(fc1_scales_fp16.get(), fc1_scales_float.get(), fc1_scales_size);
    MlasConvertHalfToFloatBuffer(fc2_scales_fp16.get(), fc2_scales_float.get(), fc2_scales_size);
  }
  const float* fc1_scales_data = fc1_scales_float.get();
  const float* fc2_scales_data = fc2_scales_float.get();

  run_moe_fc_cpu_grouped(
      input_float.get(), router_float.get(),
      fc1_wq, fc1_scales_data, fc1_bias_float.get(),
      fc2_wq, fc2_scales_data, fc2_bias_float.get(),
      UseUInt4x2 ? 4 : 8,
      moe_params.num_rows, moe_params.hidden_size, moe_params.inter_size, moe_params.num_experts, correct_k,
      normalize_routing_weights_, activation_alpha_, activation_beta_, output_float.get(), thread_pool);

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
  ORT_UNUSED_PARAMETER(fc3_experts_weights_optional);
  ORT_UNUSED_PARAMETER(fc3_experts_bias_optional);

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
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(input_data),
                                 input_float.get(), input_size);
    MlasConvertHalfToFloatBuffer(reinterpret_cast<const MLAS_FP16*>(router_probs_data),
                                 router_float.get(), router_size);
  }

  std::fill(output_float.get(), output_float.get() + output_size, 0.0f);

  for (int64_t token_idx = 0; token_idx < moe_params.num_rows; ++token_idx) {
    const float* token_router_probs = router_float.get() + token_idx * moe_params.num_experts;

    std::vector<std::pair<float, int64_t>> expert_scores;
    for (int64_t expert_idx = 0; expert_idx < moe_params.num_experts; ++expert_idx) {
      expert_scores.emplace_back(token_router_probs[expert_idx], expert_idx);
    }
    std::sort(expert_scores.rbegin(), expert_scores.rend());

    std::vector<float> topk_weights;
    std::vector<int64_t> topk_experts;
    float weight_sum = 0.0f;

    for (int64_t k_idx = 0; k_idx < std::min(static_cast<int64_t>(k_), moe_params.num_experts); ++k_idx) {
      float weight = expert_scores[static_cast<size_t>(k_idx)].first;
      int64_t expert_idx = expert_scores[static_cast<size_t>(k_idx)].second;

      if (weight <= 0.0f) break;

      topk_weights.push_back(weight);
      topk_experts.push_back(expert_idx);
      weight_sum += weight;
    }

    if (weight_sum > 0.0f) {
      for (auto& w : topk_weights) {
        w /= weight_sum;
      }
    }

    for (size_t k_idx = 0; k_idx < topk_weights.size(); ++k_idx) {
      float weight = topk_weights[k_idx];
      int64_t expert_idx = topk_experts[k_idx];

      const float* token_input = input_float.get() + token_idx * moe_params.hidden_size;
      float* token_output = output_float.get() + token_idx * moe_params.hidden_size;

      const int64_t fc1_output_size = 2 * moe_params.inter_size;

      auto intermediate = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(fc1_output_size));

      const float* expert_fc1_weights = fc1_weights_data + expert_idx * moe_params.hidden_size * fc1_output_size;

      for (int64_t out_idx = 0; out_idx < fc1_output_size; ++out_idx) {
        float sum = 0.0f;
        for (int64_t in_idx = 0; in_idx < moe_params.hidden_size; ++in_idx) {
          sum += token_input[in_idx] * expert_fc1_weights[out_idx * moe_params.hidden_size + in_idx];
        }
        if (fc1_bias_data) {
          sum += static_cast<float>(fc1_bias_data[expert_idx * fc1_output_size + out_idx]);
        }
        intermediate.get()[out_idx] = sum;
      }

      contrib::ApplySwiGLUActivation(intermediate.get(), moe_params.inter_size, true,
                                     activation_alpha_, activation_beta_);

      const float* expert_fc2_weights = fc2_weights_data + expert_idx * moe_params.inter_size * moe_params.hidden_size;

      for (int64_t out_idx = 0; out_idx < moe_params.hidden_size; ++out_idx) {
        float sum = 0.0f;
        for (int64_t in_idx = 0; in_idx < moe_params.inter_size; ++in_idx) {
          sum += intermediate.get()[in_idx] * expert_fc2_weights[out_idx * moe_params.inter_size + in_idx];
        }
        if (fc2_bias_data) {
          sum += static_cast<float>(fc2_bias_data[expert_idx * moe_params.hidden_size + out_idx]);
        }

        token_output[out_idx] += weight * sum;
      }
    }
  }

  if constexpr (std::is_same_v<T, float>) {
    std::copy(output_float.get(), output_float.get() + output_size, output_data);
  } else if constexpr (std::is_same_v<T, MLFloat16>) {
    MlasConvertFloatToHalfBuffer(output_float.get(),
                                 reinterpret_cast<MLAS_FP16*>(output_data),
                                 output_size);
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
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<float>()),
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
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<MLFloat16>()),
    QMoE<MLFloat16>);

template class QMoE<float>;
template class QMoE<MLFloat16>;

}  // namespace contrib
}  // namespace onnxruntime
