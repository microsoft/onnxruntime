// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

#include <gsl/gsl>

#include "core/common/common.h"
#include "core/common/float16.h"
#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"
#include "contrib_ops/cpu/moe/moe_base_cpu.h"

#ifdef SHARED_PROVIDER
#include "core/providers/shared_library/provider_wrappedtypes.h"
#endif

namespace onnxruntime::contrib {

inline float ApplyMoeCpuOffloadActivation(float value, ActivationType activation_type) {
  switch (activation_type) {
    case ActivationType::Relu:
      return std::max(0.0f, value);
    case ActivationType::Gelu:
      return 0.5f * value *
             (1.0f + std::tanh(0.7978845608f * (value + 0.044715f * value * value * value)));
    case ActivationType::Silu:
      return value / (1.0f + std::exp(-value));
    case ActivationType::Identity:
    case ActivationType::SwiGLU:
    default:
      return value;
  }
}

inline void RunMoeCpuOffloadHalfGemm(
    size_t M, size_t N, size_t K,
    const MLAS_HALF_GEMM_DATA_PARAMS& parameters,
    concurrency::ThreadPool* thread_pool) {
#if defined(SHARED_PROVIDER) && !defined(ORT_MINIMAL_BUILD)
  g_host->MlasHalfGemmBatch__Run(M, N, K, 1, &parameters, thread_pool);
#elif !defined(SHARED_PROVIDER)
  MlasHalfGemmBatch(M, N, K, 1, &parameters, thread_pool);
#else
  ORT_UNUSED_PARAMETER(M);
  ORT_UNUSED_PARAMETER(N);
  ORT_UNUSED_PARAMETER(K);
  ORT_UNUSED_PARAMETER(parameters);
  ORT_UNUSED_PARAMETER(thread_pool);
#endif
}

inline void RunMoeCpuOffloadFloatGemm(
    size_t M, size_t N, size_t K,
    const MLAS_SGEMM_DATA_PARAMS& parameters,
    concurrency::ThreadPool* thread_pool) {
#if defined(SHARED_PROVIDER) && !defined(ORT_MINIMAL_BUILD)
  g_host->MlasGemmBatch__Run(M, N, K, 1, &parameters, thread_pool);
#elif !defined(SHARED_PROVIDER)
  MlasGemmBatch(CblasNoTrans, CblasNoTrans, M, N, K, &parameters, 1, thread_pool, nullptr);
#else
  ORT_UNUSED_PARAMETER(M);
  ORT_UNUSED_PARAMETER(N);
  ORT_UNUSED_PARAMETER(K);
  ORT_UNUSED_PARAMETER(parameters);
  ORT_UNUSED_PARAMETER(thread_pool);
#endif
}

struct MoeCpuOffloadParameters {
  ActivationType activation_type;
  float activation_alpha;
  float activation_beta;
  float swiglu_limit;
  bool fused_swiglu;
};

template <typename T>
inline float MoeCpuOffloadToFloat(T value) {
  if constexpr (std::is_same_v<T, float>) {
    return value;
  } else {
    return value.ToFloat();
  }
}

// Converts one canonical QMoE expert from [N, K / pack] into the KxN layout
// consumed by the dense CPU offload GEMMs.
template <typename TScale, typename TDense>
inline Status DequantizeMoeCpuOffloadExpert(
    gsl::span<const uint8_t> weights,
    gsl::span<const TScale> scales,
    gsl::span<const uint8_t> zero_points,
    int64_t weight_bits,
    int64_t block_size,
    int64_t output_size,
    int64_t input_size,
    gsl::span<TDense> dense_weights) {
  ORT_RETURN_IF_NOT(weight_bits == 4 || weight_bits == 8,
                    "QMoE CPU offload supports only INT4 and INT8 weights.");
  const int64_t pack = 8 / weight_bits;
  const int64_t packed_input_size = (input_size + pack - 1) / pack;
  const int64_t blocks_per_row =
      block_size > 0 ? (input_size + block_size - 1) / block_size : 1;
  ORT_RETURN_IF_NOT(
      weights.size() == static_cast<size_t>(SafeInt<int64_t>(output_size) * packed_input_size) &&
          scales.size() == static_cast<size_t>(SafeInt<int64_t>(output_size) * blocks_per_row) &&
          dense_weights.size() == static_cast<size_t>(
                                      SafeInt<int64_t>(output_size) * input_size),
      "Invalid QMoE CPU-offload expert buffer sizes.");

  const size_t zero_points_per_row =
      zero_points.empty() ? 0 : static_cast<size_t>((blocks_per_row + pack - 1) / pack);
  ORT_RETURN_IF_NOT(
      zero_points.empty() ||
          zero_points.size() == static_cast<size_t>(output_size) * zero_points_per_row,
      "Invalid QMoE CPU-offload zero-point buffer size.");

  const uint8_t value_mask = static_cast<uint8_t>((1u << weight_bits) - 1u);
  const uint8_t default_zero_point = static_cast<uint8_t>(1u << (weight_bits - 1));
  for (int64_t output = 0; output < output_size; ++output) {
    for (int64_t input = 0; input < input_size; ++input) {
      const int64_t packed_index =
          output * packed_input_size + input / pack;
      const int shift = static_cast<int>((input % pack) * weight_bits);
      const uint8_t value =
          static_cast<uint8_t>((weights[static_cast<size_t>(packed_index)] >> shift) & value_mask);
      const int64_t block = block_size > 0 ? input / block_size : 0;
      uint8_t zero_point = default_zero_point;
      if (!zero_points.empty()) {
        const size_t zp_index =
            static_cast<size_t>(output) * zero_points_per_row + static_cast<size_t>(block / pack);
        const int zp_shift = static_cast<int>((block % pack) * weight_bits);
        zero_point = static_cast<uint8_t>((zero_points[zp_index] >> zp_shift) & value_mask);
      }
      const float scale =
          MoeCpuOffloadToFloat(scales[static_cast<size_t>(output * blocks_per_row + block)]);
      dense_weights[static_cast<size_t>(input * output_size + output)] =
          TDense((static_cast<float>(value) - static_cast<float>(zero_point)) * scale);
    }
  }
  return Status::OK();
}

inline Status ComputeMoeCpuOffloadedExpertsFp16(
    gsl::span<const MLFloat16> input,
    gsl::span<const int> route_experts,
    gsl::span<const float> route_scales,
    gsl::span<const int> cpu_expert_map,
    gsl::span<const MLFloat16> fc1_weights,
    gsl::span<const MLFloat16> fc1_bias,
    gsl::span<const MLFloat16> fc2_weights,
    gsl::span<const MLFloat16> fc2_bias,
    int64_t num_rows,
    int64_t hidden_size,
    int64_t inter_size,
    int64_t num_experts,
    int64_t experts_per_token,
    const MoeCpuOffloadParameters& parameters,
    gsl::span<MLFloat16> output,
    concurrency::ThreadPool* thread_pool) {
  const int64_t fc1_output_size = parameters.fused_swiglu ? 2 * inter_size : inter_size;
  ORT_RETURN_IF(parameters.activation_type == ActivationType::SwiGLU && !parameters.fused_swiglu,
                "FP16 MoE CPU offload requires fused_swiglu for SwiGLU activation.");
  const size_t num_cpu_experts =
      static_cast<size_t>(std::count_if(cpu_expert_map.begin(), cpu_expert_map.end(),
                                        [](int slot) { return slot >= 0; }));
  ORT_RETURN_IF_NOT(input.size() == static_cast<size_t>(SafeInt<int64_t>(num_rows) * hidden_size) &&
                        route_experts.size() == static_cast<size_t>(SafeInt<int64_t>(num_rows) * experts_per_token) &&
                        route_scales.size() == route_experts.size() &&
                        cpu_expert_map.size() == static_cast<size_t>(num_experts) &&
                        fc1_weights.size() ==
                            static_cast<size_t>(SafeInt<size_t>(num_cpu_experts) * fc1_output_size * hidden_size) &&
                        (fc1_bias.empty() ||
                         fc1_bias.size() ==
                             static_cast<size_t>(SafeInt<size_t>(num_cpu_experts) * fc1_output_size)) &&
                        fc2_weights.size() ==
                            static_cast<size_t>(SafeInt<size_t>(num_cpu_experts) * hidden_size * inter_size) &&
                        (fc2_bias.empty() ||
                         fc2_bias.size() ==
                             static_cast<size_t>(SafeInt<size_t>(num_cpu_experts) * hidden_size)) &&
                        output.size() == input.size(),
                    "Invalid FP16 MoE CPU-offload buffer sizes.");

  std::vector<float> accumulated(output.size(), 0.0f);
  std::vector<size_t> route_counts(static_cast<size_t>(num_experts), 0);
  for (size_t route = 0; route < route_experts.size(); ++route) {
    const int expert = route_experts[route];
    if (expert >= 0 && expert < num_experts &&
        cpu_expert_map[static_cast<size_t>(expert)] >= 0 && route_scales[route] > 0.0f) {
      ++route_counts[static_cast<size_t>(expert)];
    }
  }

  std::vector<size_t> route_offsets(static_cast<size_t>(num_experts) + 1, 0);
  size_t max_route_count = 0;
  for (size_t expert = 0; expert < static_cast<size_t>(num_experts); ++expert) {
    route_offsets[expert + 1] = route_offsets[expert] + route_counts[expert];
    max_route_count = std::max(max_route_count, route_counts[expert]);
  }

  std::vector<int64_t> routes(route_offsets.back());
  std::vector<size_t> route_cursors(route_offsets.begin(), route_offsets.end() - 1);
  for (size_t route = 0; route < route_experts.size(); ++route) {
    const int expert = route_experts[route];
    if (expert >= 0 && expert < num_experts &&
        cpu_expert_map[static_cast<size_t>(expert)] >= 0 && route_scales[route] > 0.0f) {
      routes[route_cursors[static_cast<size_t>(expert)]++] = static_cast<int64_t>(route);
    }
  }

  std::vector<MLFloat16> gathered(max_route_count * static_cast<size_t>(hidden_size));
  std::vector<MLFloat16> fc1_output(max_route_count * static_cast<size_t>(fc1_output_size));
  std::vector<MLFloat16> activated(max_route_count * static_cast<size_t>(inter_size));
  std::vector<MLFloat16> expert_output(max_route_count * static_cast<size_t>(hidden_size));

  for (int64_t expert = 0; expert < num_experts; ++expert) {
    const int cpu_slot = cpu_expert_map[static_cast<size_t>(expert)];
    if (cpu_slot < 0) {
      continue;
    }

    const size_t route_begin = route_offsets[static_cast<size_t>(expert)];
    const size_t route_count = route_counts[static_cast<size_t>(expert)];
    if (route_count == 0) {
      continue;
    }

    for (size_t row = 0; row < route_count; ++row) {
      const int64_t token = routes[route_begin + row] / experts_per_token;
      std::copy_n(input.data() + token * hidden_size, hidden_size,
                  gathered.data() + row * static_cast<size_t>(hidden_size));
    }

    MLAS_HALF_GEMM_DATA_PARAMS fc1_params{};
    fc1_params.A = gathered.data();
    fc1_params.lda = static_cast<size_t>(hidden_size);
    fc1_params.B = fc1_weights.data() + cpu_slot * fc1_output_size * hidden_size;
    fc1_params.ldb = static_cast<size_t>(fc1_output_size);
    fc1_params.C = fc1_output.data();
    fc1_params.ldc = static_cast<size_t>(fc1_output_size);
    RunMoeCpuOffloadHalfGemm(route_count, static_cast<size_t>(fc1_output_size),
                             static_cast<size_t>(hidden_size), fc1_params, thread_pool);

    const MLFloat16* expert_fc1_bias =
        fc1_bias.empty() ? nullptr : fc1_bias.data() + cpu_slot * fc1_output_size;
    if (expert_fc1_bias != nullptr) {
      for (size_t row = 0; row < route_count; ++row) {
        for (int64_t column = 0; column < fc1_output_size; ++column) {
          const size_t index = row * static_cast<size_t>(fc1_output_size) + static_cast<size_t>(column);
          fc1_output[index] = MLFloat16(fc1_output[index].ToFloat() + expert_fc1_bias[column].ToFloat());
        }
      }
    }

    if (parameters.fused_swiglu) {
      for (size_t row = 0; row < route_count; ++row) {
        const MLFloat16* source = fc1_output.data() + row * static_cast<size_t>(fc1_output_size);
        MLFloat16* destination = activated.data() + row * static_cast<size_t>(inter_size);
        for (int64_t column = 0; column < inter_size; ++column) {
          float gate = std::min(source[2 * column].ToFloat(), parameters.swiglu_limit);
          const float linear = std::clamp(source[2 * column + 1].ToFloat(),
                                          -parameters.swiglu_limit, parameters.swiglu_limit);
          const float sigmoid_argument = parameters.activation_alpha * gate;
          const float sigmoid = sigmoid_argument > 0.0f
                                    ? 1.0f / (1.0f + std::exp(-sigmoid_argument))
                                    : std::exp(sigmoid_argument) / (1.0f + std::exp(sigmoid_argument));
          destination[column] =
              MLFloat16(gate * sigmoid * (linear + parameters.activation_beta));
        }
      }
    } else {
      const size_t activated_element_count = route_count * static_cast<size_t>(inter_size);
      for (size_t index = 0; index < activated_element_count; ++index) {
        activated[index] = MLFloat16(ApplyMoeCpuOffloadActivation(
            fc1_output[index].ToFloat(), parameters.activation_type));
      }
    }

    MLAS_HALF_GEMM_DATA_PARAMS fc2_params{};
    fc2_params.A = activated.data();
    fc2_params.lda = static_cast<size_t>(inter_size);
    fc2_params.B = fc2_weights.data() + cpu_slot * hidden_size * inter_size;
    fc2_params.ldb = static_cast<size_t>(hidden_size);
    fc2_params.C = expert_output.data();
    fc2_params.ldc = static_cast<size_t>(hidden_size);
    RunMoeCpuOffloadHalfGemm(route_count, static_cast<size_t>(hidden_size),
                             static_cast<size_t>(inter_size), fc2_params, thread_pool);

    const MLFloat16* expert_fc2_bias =
        fc2_bias.empty() ? nullptr : fc2_bias.data() + cpu_slot * hidden_size;
    for (size_t row = 0; row < route_count; ++row) {
      const int64_t route = routes[route_begin + row];
      const int64_t token = route / experts_per_token;
      const float scale = route_scales[static_cast<size_t>(route)];
      for (int64_t column = 0; column < hidden_size; ++column) {
        float value = expert_output[row * static_cast<size_t>(hidden_size) +
                                    static_cast<size_t>(column)]
                          .ToFloat();
        if (expert_fc2_bias != nullptr) {
          value += expert_fc2_bias[column].ToFloat();
        }
        accumulated[static_cast<size_t>(token * hidden_size + column)] += scale * value;
      }
    }
  }

  for (size_t index = 0; index < output.size(); ++index) {
    output[index] = MLFloat16(accumulated[index]);
  }
  return Status::OK();
}

inline Status ComputeMoeCpuOffloadedExpertsBFloat16(
    gsl::span<const BFloat16> input,
    gsl::span<const int> route_experts,
    gsl::span<const float> route_scales,
    gsl::span<const int> cpu_expert_map,
    gsl::span<const float> fc1_weights,
    gsl::span<const BFloat16> fc1_bias,
    gsl::span<const float> fc2_weights,
    gsl::span<const BFloat16> fc2_bias,
    int64_t num_rows,
    int64_t hidden_size,
    int64_t inter_size,
    int64_t num_experts,
    int64_t experts_per_token,
    const MoeCpuOffloadParameters& parameters,
    gsl::span<BFloat16> output,
    concurrency::ThreadPool* thread_pool) {
  const int64_t fc1_output_size = parameters.fused_swiglu ? 2 * inter_size : inter_size;
  ORT_RETURN_IF(parameters.activation_type == ActivationType::SwiGLU && !parameters.fused_swiglu,
                "BF16 MoE CPU offload requires fused_swiglu for SwiGLU activation.");
  const size_t num_cpu_experts =
      static_cast<size_t>(std::count_if(cpu_expert_map.begin(), cpu_expert_map.end(),
                                        [](int slot) { return slot >= 0; }));
  ORT_RETURN_IF_NOT(input.size() == static_cast<size_t>(SafeInt<int64_t>(num_rows) * hidden_size) &&
                        route_experts.size() == static_cast<size_t>(SafeInt<int64_t>(num_rows) * experts_per_token) &&
                        route_scales.size() == route_experts.size() &&
                        cpu_expert_map.size() == static_cast<size_t>(num_experts) &&
                        fc1_weights.size() ==
                            static_cast<size_t>(SafeInt<size_t>(num_cpu_experts) * fc1_output_size * hidden_size) &&
                        (fc1_bias.empty() ||
                         fc1_bias.size() ==
                             static_cast<size_t>(SafeInt<size_t>(num_cpu_experts) * fc1_output_size)) &&
                        fc2_weights.size() ==
                            static_cast<size_t>(SafeInt<size_t>(num_cpu_experts) * hidden_size * inter_size) &&
                        (fc2_bias.empty() ||
                         fc2_bias.size() ==
                             static_cast<size_t>(SafeInt<size_t>(num_cpu_experts) * hidden_size)) &&
                        output.size() == input.size(),
                    "Invalid BF16 MoE CPU-offload buffer sizes.");

  std::vector<float> accumulated(output.size(), 0.0f);
  std::vector<size_t> route_counts(static_cast<size_t>(num_experts), 0);
  for (size_t route = 0; route < route_experts.size(); ++route) {
    const int expert = route_experts[route];
    if (expert >= 0 && expert < num_experts &&
        cpu_expert_map[static_cast<size_t>(expert)] >= 0 && route_scales[route] > 0.0f) {
      ++route_counts[static_cast<size_t>(expert)];
    }
  }

  std::vector<size_t> route_offsets(static_cast<size_t>(num_experts) + 1, 0);
  size_t max_route_count = 0;
  for (size_t expert = 0; expert < static_cast<size_t>(num_experts); ++expert) {
    route_offsets[expert + 1] = route_offsets[expert] + route_counts[expert];
    max_route_count = std::max(max_route_count, route_counts[expert]);
  }

  std::vector<int64_t> routes(route_offsets.back());
  std::vector<size_t> route_cursors(route_offsets.begin(), route_offsets.end() - 1);
  for (size_t route = 0; route < route_experts.size(); ++route) {
    const int expert = route_experts[route];
    if (expert >= 0 && expert < num_experts &&
        cpu_expert_map[static_cast<size_t>(expert)] >= 0 && route_scales[route] > 0.0f) {
      routes[route_cursors[static_cast<size_t>(expert)]++] = static_cast<int64_t>(route);
    }
  }

  std::vector<float> gathered(max_route_count * static_cast<size_t>(hidden_size));
  std::vector<float> fc1_output(max_route_count * static_cast<size_t>(fc1_output_size));
  std::vector<float> activated(max_route_count * static_cast<size_t>(inter_size));
  std::vector<float> expert_output(max_route_count * static_cast<size_t>(hidden_size));

  for (int64_t expert = 0; expert < num_experts; ++expert) {
    const int cpu_slot = cpu_expert_map[static_cast<size_t>(expert)];
    if (cpu_slot < 0) {
      continue;
    }

    const size_t route_begin = route_offsets[static_cast<size_t>(expert)];
    const size_t route_count = route_counts[static_cast<size_t>(expert)];
    if (route_count == 0) {
      continue;
    }

    for (size_t row = 0; row < route_count; ++row) {
      const int64_t token = routes[route_begin + row] / experts_per_token;
      for (int64_t column = 0; column < hidden_size; ++column) {
        gathered[row * static_cast<size_t>(hidden_size) + static_cast<size_t>(column)] =
            input[static_cast<size_t>(token * hidden_size + column)].ToFloat();
      }
    }

    MLAS_SGEMM_DATA_PARAMS fc1_params{};
    fc1_params.A = gathered.data();
    fc1_params.lda = static_cast<size_t>(hidden_size);
    fc1_params.B = fc1_weights.data() + cpu_slot * fc1_output_size * hidden_size;
    fc1_params.ldb = static_cast<size_t>(fc1_output_size);
    fc1_params.C = fc1_output.data();
    fc1_params.ldc = static_cast<size_t>(fc1_output_size);
    fc1_params.alpha = 1.0f;
    fc1_params.beta = 0.0f;
    RunMoeCpuOffloadFloatGemm(route_count, static_cast<size_t>(fc1_output_size),
                              static_cast<size_t>(hidden_size), fc1_params, thread_pool);

    const BFloat16* expert_fc1_bias =
        fc1_bias.empty() ? nullptr : fc1_bias.data() + cpu_slot * fc1_output_size;
    if (expert_fc1_bias != nullptr) {
      for (size_t row = 0; row < route_count; ++row) {
        for (int64_t column = 0; column < fc1_output_size; ++column) {
          fc1_output[row * static_cast<size_t>(fc1_output_size) + static_cast<size_t>(column)] +=
              expert_fc1_bias[column].ToFloat();
        }
      }
    }

    if (parameters.fused_swiglu) {
      for (size_t row = 0; row < route_count; ++row) {
        const float* source = fc1_output.data() + row * static_cast<size_t>(fc1_output_size);
        float* destination = activated.data() + row * static_cast<size_t>(inter_size);
        for (int64_t column = 0; column < inter_size; ++column) {
          float gate = std::min(source[2 * column], parameters.swiglu_limit);
          const float linear = std::clamp(source[2 * column + 1],
                                          -parameters.swiglu_limit, parameters.swiglu_limit);
          const float sigmoid_argument = parameters.activation_alpha * gate;
          const float sigmoid = sigmoid_argument > 0.0f
                                    ? 1.0f / (1.0f + std::exp(-sigmoid_argument))
                                    : std::exp(sigmoid_argument) / (1.0f + std::exp(sigmoid_argument));
          destination[column] = gate * sigmoid * (linear + parameters.activation_beta);
        }
      }
    } else {
      const size_t activated_element_count = route_count * static_cast<size_t>(inter_size);
      for (size_t index = 0; index < activated_element_count; ++index) {
        activated[index] = ApplyMoeCpuOffloadActivation(fc1_output[index], parameters.activation_type);
      }
    }

    MLAS_SGEMM_DATA_PARAMS fc2_params{};
    fc2_params.A = activated.data();
    fc2_params.lda = static_cast<size_t>(inter_size);
    fc2_params.B = fc2_weights.data() + cpu_slot * hidden_size * inter_size;
    fc2_params.ldb = static_cast<size_t>(hidden_size);
    fc2_params.C = expert_output.data();
    fc2_params.ldc = static_cast<size_t>(hidden_size);
    fc2_params.alpha = 1.0f;
    fc2_params.beta = 0.0f;
    RunMoeCpuOffloadFloatGemm(route_count, static_cast<size_t>(hidden_size),
                              static_cast<size_t>(inter_size), fc2_params, thread_pool);

    const BFloat16* expert_fc2_bias =
        fc2_bias.empty() ? nullptr : fc2_bias.data() + cpu_slot * hidden_size;
    for (size_t row = 0; row < route_count; ++row) {
      const int64_t route = routes[route_begin + row];
      const int64_t token = route / experts_per_token;
      const float scale = route_scales[static_cast<size_t>(route)];
      for (int64_t column = 0; column < hidden_size; ++column) {
        float value = expert_output[row * static_cast<size_t>(hidden_size) +
                                    static_cast<size_t>(column)];
        if (expert_fc2_bias != nullptr) {
          value += expert_fc2_bias[column].ToFloat();
        }
        accumulated[static_cast<size_t>(token * hidden_size + column)] += scale * value;
      }
    }
  }

  for (size_t index = 0; index < output.size(); ++index) {
    output[index] = BFloat16(accumulated[index]);
  }
  return Status::OK();
}

}  // namespace onnxruntime::contrib
