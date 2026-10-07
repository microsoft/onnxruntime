// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
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
#ifdef SHARED_PROVIDER
  g_host->MlasHalfGemmBatch__Run(M, N, K, 1, &parameters, thread_pool);
#else
  MlasHalfGemmBatch(M, N, K, 1, &parameters, thread_pool);
#endif
}

inline void RunMoeCpuOffloadFloatGemm(
    size_t M, size_t N, size_t K,
    const MLAS_SGEMM_DATA_PARAMS& parameters,
    concurrency::ThreadPool* thread_pool) {
#ifdef SHARED_PROVIDER
  g_host->MlasGemmBatch__Run(M, N, K, 1, &parameters, thread_pool);
#else
  MlasGemmBatch(CblasNoTrans, CblasNoTrans, M, N, K, &parameters, 1, thread_pool, nullptr);
#endif
}

struct MoeCpuOffloadParameters {
  ActivationType activation_type;
  float activation_alpha;
  float activation_beta;
  float swiglu_limit;
  bool fused_swiglu;
};

inline Status ComputeMoeCpuOffloadedExpertsFp16(
    gsl::span<const MLFloat16> input,
    gsl::span<const int> route_experts,
    gsl::span<const float> route_scales,
    gsl::span<const int> cuda_expert_map,
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
  ORT_RETURN_IF_NOT(input.size() == static_cast<size_t>(SafeInt<int64_t>(num_rows) * hidden_size) &&
                        route_experts.size() == static_cast<size_t>(SafeInt<int64_t>(num_rows) * experts_per_token) &&
                        route_scales.size() == route_experts.size() &&
                        cuda_expert_map.size() == static_cast<size_t>(num_experts) &&
                        fc1_weights.size() ==
                            static_cast<size_t>(SafeInt<int64_t>(num_experts) * fc1_output_size * hidden_size) &&
                        (fc1_bias.empty() ||
                         fc1_bias.size() ==
                             static_cast<size_t>(SafeInt<int64_t>(num_experts) * fc1_output_size)) &&
                        fc2_weights.size() ==
                            static_cast<size_t>(SafeInt<int64_t>(num_experts) * hidden_size * inter_size) &&
                        (fc2_bias.empty() ||
                         fc2_bias.size() ==
                             static_cast<size_t>(SafeInt<int64_t>(num_experts) * hidden_size)) &&
                        output.size() == input.size(),
                    "Invalid FP16 MoE CPU-offload buffer sizes.");

  std::vector<float> accumulated(output.size(), 0.0f);
  std::vector<size_t> route_counts(static_cast<size_t>(num_experts), 0);
  for (size_t route = 0; route < route_experts.size(); ++route) {
    const int expert = route_experts[route];
    if (expert >= 0 && expert < num_experts &&
        cuda_expert_map[static_cast<size_t>(expert)] < 0 && route_scales[route] > 0.0f) {
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
        cuda_expert_map[static_cast<size_t>(expert)] < 0 && route_scales[route] > 0.0f) {
      routes[route_cursors[static_cast<size_t>(expert)]++] = static_cast<int64_t>(route);
    }
  }

  std::vector<MLFloat16> gathered(max_route_count * static_cast<size_t>(hidden_size));
  std::vector<MLFloat16> fc1_output(max_route_count * static_cast<size_t>(fc1_output_size));
  std::vector<MLFloat16> activated(max_route_count * static_cast<size_t>(inter_size));
  std::vector<MLFloat16> expert_output(max_route_count * static_cast<size_t>(hidden_size));

  for (int64_t expert = 0; expert < num_experts; ++expert) {
    if (cuda_expert_map[static_cast<size_t>(expert)] >= 0) {
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
    fc1_params.B = fc1_weights.data() + expert * fc1_output_size * hidden_size;
    fc1_params.ldb = static_cast<size_t>(fc1_output_size);
    fc1_params.C = fc1_output.data();
    fc1_params.ldc = static_cast<size_t>(fc1_output_size);
    RunMoeCpuOffloadHalfGemm(route_count, static_cast<size_t>(fc1_output_size),
                             static_cast<size_t>(hidden_size), fc1_params, thread_pool);

    const MLFloat16* expert_fc1_bias =
        fc1_bias.empty() ? nullptr : fc1_bias.data() + expert * fc1_output_size;
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
    fc2_params.B = fc2_weights.data() + expert * hidden_size * inter_size;
    fc2_params.ldb = static_cast<size_t>(hidden_size);
    fc2_params.C = expert_output.data();
    fc2_params.ldc = static_cast<size_t>(hidden_size);
    RunMoeCpuOffloadHalfGemm(route_count, static_cast<size_t>(hidden_size),
                             static_cast<size_t>(inter_size), fc2_params, thread_pool);

    const MLFloat16* expert_fc2_bias =
        fc2_bias.empty() ? nullptr : fc2_bias.data() + expert * hidden_size;
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
    gsl::span<const int> cuda_expert_map,
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
  ORT_RETURN_IF_NOT(input.size() == static_cast<size_t>(SafeInt<int64_t>(num_rows) * hidden_size) &&
                        route_experts.size() == static_cast<size_t>(SafeInt<int64_t>(num_rows) * experts_per_token) &&
                        route_scales.size() == route_experts.size() &&
                        cuda_expert_map.size() == static_cast<size_t>(num_experts) &&
                        fc1_weights.size() ==
                            static_cast<size_t>(SafeInt<int64_t>(num_experts) * fc1_output_size * hidden_size) &&
                        (fc1_bias.empty() ||
                         fc1_bias.size() ==
                             static_cast<size_t>(SafeInt<int64_t>(num_experts) * fc1_output_size)) &&
                        fc2_weights.size() ==
                            static_cast<size_t>(SafeInt<int64_t>(num_experts) * hidden_size * inter_size) &&
                        (fc2_bias.empty() ||
                         fc2_bias.size() ==
                             static_cast<size_t>(SafeInt<int64_t>(num_experts) * hidden_size)) &&
                        output.size() == input.size(),
                    "Invalid BF16 MoE CPU-offload buffer sizes.");

  std::vector<float> accumulated(output.size(), 0.0f);
  std::vector<size_t> route_counts(static_cast<size_t>(num_experts), 0);
  for (size_t route = 0; route < route_experts.size(); ++route) {
    const int expert = route_experts[route];
    if (expert >= 0 && expert < num_experts &&
        cuda_expert_map[static_cast<size_t>(expert)] < 0 && route_scales[route] > 0.0f) {
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
        cuda_expert_map[static_cast<size_t>(expert)] < 0 && route_scales[route] > 0.0f) {
      routes[route_cursors[static_cast<size_t>(expert)]++] = static_cast<int64_t>(route);
    }
  }

  std::vector<float> gathered(max_route_count * static_cast<size_t>(hidden_size));
  std::vector<float> fc1_output(max_route_count * static_cast<size_t>(fc1_output_size));
  std::vector<float> activated(max_route_count * static_cast<size_t>(inter_size));
  std::vector<float> expert_output(max_route_count * static_cast<size_t>(hidden_size));

  for (int64_t expert = 0; expert < num_experts; ++expert) {
    if (cuda_expert_map[static_cast<size_t>(expert)] >= 0) {
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
    fc1_params.B = fc1_weights.data() + expert * fc1_output_size * hidden_size;
    fc1_params.ldb = static_cast<size_t>(fc1_output_size);
    fc1_params.C = fc1_output.data();
    fc1_params.ldc = static_cast<size_t>(fc1_output_size);
    fc1_params.alpha = 1.0f;
    fc1_params.beta = 0.0f;
    RunMoeCpuOffloadFloatGemm(route_count, static_cast<size_t>(fc1_output_size),
                              static_cast<size_t>(hidden_size), fc1_params, thread_pool);

    const BFloat16* expert_fc1_bias =
        fc1_bias.empty() ? nullptr : fc1_bias.data() + expert * fc1_output_size;
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
    fc2_params.B = fc2_weights.data() + expert * hidden_size * inter_size;
    fc2_params.ldb = static_cast<size_t>(hidden_size);
    fc2_params.C = expert_output.data();
    fc2_params.ldc = static_cast<size_t>(hidden_size);
    fc2_params.alpha = 1.0f;
    fc2_params.beta = 0.0f;
    RunMoeCpuOffloadFloatGemm(route_count, static_cast<size_t>(hidden_size),
                              static_cast<size_t>(inter_size), fc2_params, thread_pool);

    const BFloat16* expert_fc2_bias =
        fc2_bias.empty() ? nullptr : fc2_bias.data() + expert * hidden_size;
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
