/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "core/common/common.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/common.h"
#include "contrib_ops/cuda/llm/common/env_utils.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_utils.cuh"
#include "contrib_ops/cuda/llm/kernels/quantization.cuh"
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/array.h>
#include <cutlass/numeric_conversion.h>

namespace onnxruntime::llm::kernels::cutlass_kernels {

constexpr static int ACTIVATION_THREADS_PER_BLOCK = 256;

struct QuantParams;

template <class ActivationOutputType, class GemmOutputType, template <class> class ActFn>
__global__ void doGatedActivationKernel(ActivationOutputType* output, GemmOutputType const* gemm_result,
                                        int64_t const* num_valid_tokens_ptr, int64_t inter_size,
                                        ActivationType activation_type,
                                        ActivationParams activation_params = {}) {
  int64_t const tid = threadIdx.x;
  int64_t const token = blockIdx.x;
  if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr) {
    return;
  }

  output = output + token * inter_size;
  gemm_result = gemm_result + token * inter_size * 2;

  constexpr int64_t ACTIVATION_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<ActivationOutputType>::value;

  using OutputElem = cutlass::Array<ActivationOutputType, ACTIVATION_ELEM_PER_THREAD>;
  using GemmResultElem = cutlass::Array<GemmOutputType, ACTIVATION_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
  auto gemm_result_vec = reinterpret_cast<GemmResultElem const*>(gemm_result);
  auto output_vec = reinterpret_cast<OutputElem*>(output);
  int64_t const start_offset = tid;
  int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
  assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
  int64_t const num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;
  int64_t const inter_size_vec = inter_size / ACTIVATION_ELEM_PER_THREAD;

  ActFn<ComputeElem> fn{};
  bool const use_custom_swiglu = std::is_same_v<ActFn<float>, cutlass::epilogue::thread::SiLu<float>> &&
                                 (activation_params.alpha != 1.0f || activation_params.beta != 0.0f || isfinite(activation_params.limit));
  bool const is_swiglu_interleaved = activation_params.swiglu_fusion == 1;

  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    ComputeElem gate_part;
    ComputeElem linear_part;

    if (is_swiglu_interleaved) {
      auto* scalar_gemm = reinterpret_cast<GemmOutputType const*>(gemm_result);
      for (int i = 0; i < ACTIVATION_ELEM_PER_THREAD; ++i) {
        int64_t global_elem = elem_index * ACTIVATION_ELEM_PER_THREAD + i;
        if (global_elem >= inter_size) continue;
        // Interleaved Layout [Gate, Linear, Gate, Linear] matches Python swiglu(view(..., 2))
        gate_part[i] = static_cast<float>(scalar_gemm[2 * global_elem]);
        linear_part[i] = static_cast<float>(scalar_gemm[2 * global_elem + 1]);
      }

    } else {
      gate_part = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index]);
      linear_part = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index + inter_size_vec]);
    }

    ComputeElem gate_act;
    if (use_custom_swiglu) {
      for (int i = 0; i < ACTIVATION_ELEM_PER_THREAD; ++i) {
        float g = gate_part[i];
        if (isfinite(activation_params.limit)) {
          g = fminf(g, activation_params.limit);
        }
        float sigmoid = 1.0f / (1.0f + expf(-activation_params.alpha * g));
        float l = linear_part[i];
        if (isfinite(activation_params.limit)) {
          l = fminf(fmaxf(l, -activation_params.limit), activation_params.limit);
        }
        l += activation_params.beta;
        gate_act[i] = g * sigmoid * l;
      }
    } else {
      gate_act = fn(gate_part) * linear_part;
    }

    output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(gate_act);
  }
}

template <typename ActivationOutputType, typename GemmOutputType>
void doGatedActivation(ActivationOutputType* output, GemmOutputType const* gemm_result,
                       int64_t const* num_valid_tokens_ptr, int64_t inter_size, int64_t num_tokens, ActivationType activation_type,
                       cudaStream_t stream, ActivationParams activation_params) {
  int64_t const blocks = num_tokens;
  int64_t const threads = ACTIVATION_THREADS_PER_BLOCK;

  using namespace cutlass::epilogue::thread;
  // Select kernel based on activation type (matches TRT-LLM pattern)
  auto* fn = [&]() -> void (*)(ActivationOutputType*, GemmOutputType const*, int64_t const*,
                               int64_t, ActivationType, ActivationParams) {
    switch (activation_type) {
      case ActivationType::Swiglu:
      case ActivationType::SwigluBias:
        return &doGatedActivationKernel<ActivationOutputType, GemmOutputType, SiLu>;
      case ActivationType::Geglu:
        return &doGatedActivationKernel<ActivationOutputType, GemmOutputType, GELU>;
      case ActivationType::Silu:
        return &doGatedActivationKernel<ActivationOutputType, GemmOutputType, SiLu>;
      case ActivationType::Gelu:
        return &doGatedActivationKernel<ActivationOutputType, GemmOutputType, GELU>;
      case ActivationType::Relu:
      case ActivationType::Relu2:
        return &doGatedActivationKernel<ActivationOutputType, GemmOutputType, ReLu>;
      case ActivationType::Identity:
      default:
        return &doGatedActivationKernel<ActivationOutputType, GemmOutputType, Identity>;
    }
  }();
  fn<<<blocks, threads, 0, stream>>>(output, gemm_result, num_valid_tokens_ptr, inter_size, activation_type, activation_params);
}

// ============================== Activation =================================

template <class T, class GemmOutputType, class ScaleBiasType, template <class> class ActFn,
          TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType BlockScalingType>
__global__ void doActivationKernel(T* output, GemmOutputType const* gemm_result, float const* fp8_quant,
                                   ScaleBiasType const* bias_ptr, bool bias_is_broadcast, int64_t const* expert_first_token_offset,
                                   int num_experts_per_node, int64_t inter_size, bool gated, float const* fc2_act_global_scale,
                                   bool use_per_expert_act_scale, TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_act_sf_flat,

                                   ActivationParams activation_params = {}) {
#ifdef ENABLE_FP4
  constexpr bool IsNVFP4 = std::is_same_v<T, __nv_fp4_e2m1> && BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4;
  constexpr bool IsMXFP8 = std::is_same_v<T, __nv_fp8_e4m3> && BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX;
#else
  constexpr bool IsNVFP4 = cute::dependent_false<T>;
  constexpr bool IsMXFP8 = cute::dependent_false<T>;
#endif

  int64_t const tid = threadIdx.x;
  size_t const gated_size_mul = gated ? 2 : 1;
  size_t const gated_off = gated ? inter_size : 0;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
#endif

  constexpr int64_t VecSize = IsNVFP4 ? TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize
                                      : TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize;
  // Load 128-bits per thread, according to the smallest data type we read/write
  constexpr int64_t ACTIVATION_ELEM_PER_THREAD = (IsNVFP4 || IsMXFP8)
                                                     ? CVT_FP4_ELTS_PER_THREAD
                                                     : (128 / std::min(cutlass::sizeof_bits<T>::value, cutlass::sizeof_bits<GemmOutputType>::value));

  // This should be VecSize * 4 elements
  // We assume at least VecSize alignment or the quantization will fail
  int64_t const min_k_dim_alignment = IsNVFP4 ? TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4
                                              : TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX;
  int64_t const padded_inter_size = onnxruntime::llm::common::ceilDiv(inter_size, min_k_dim_alignment) * min_k_dim_alignment;

  int64_t const num_valid_tokens = expert_first_token_offset[num_experts_per_node];

  for (int64_t token = blockIdx.x; token < num_valid_tokens; token += gridDim.x) {
    size_t gemm_result_offset = token * inter_size * gated_size_mul;
    size_t output_offset = token * inter_size;

    int64_t expert = 0;
    if (bias_ptr || IsNVFP4 || IsMXFP8 || use_per_expert_act_scale) {
      // TODO this is almost certainly faster as a linear scan
      expert = findTotalEltsLessThanTarget(expert_first_token_offset, num_experts_per_node, token + 1) - 1;
    }

    size_t act_scale_idx = use_per_expert_act_scale ? expert : 0;
    float const quant_scale = fp8_quant ? fp8_quant[act_scale_idx] : 1.f;

    // Some globals for FP4
    float global_scale_val = fc2_act_global_scale ? fc2_act_global_scale[act_scale_idx] : 1.0f;
    int64_t num_tokens_before_expert = (IsNVFP4 || IsMXFP8) ? expert_first_token_offset[expert] : 0;

    size_t bias_offset = 0;
    if (bias_ptr) {
      bias_offset = (bias_is_broadcast ? expert * inter_size * gated_size_mul : gemm_result_offset);
    }

    using BiasElem = cutlass::Array<ScaleBiasType, ACTIVATION_ELEM_PER_THREAD>;
    using GemmResultElem = cutlass::Array<GemmOutputType, ACTIVATION_ELEM_PER_THREAD>;
    using OutputElem = std::conditional_t<IsNVFP4, uint32_t,
                                          std::conditional_t<IsMXFP8, uint64_t, cutlass::Array<T, ACTIVATION_ELEM_PER_THREAD>>>;
    using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
    // Aliases gemm_result for non-gated, non-fp8 cases
    auto gemm_result_vec = reinterpret_cast<GemmResultElem const*>(gemm_result + gemm_result_offset);
    auto output_vec = reinterpret_cast<OutputElem*>(safe_inc_ptr(output, output_offset));
    auto bias_ptr_vec = reinterpret_cast<BiasElem const*>(bias_ptr + bias_offset);
    int64_t const start_offset = tid;
    int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
    assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;
    assert(gated_off % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const gated_off_vec = gated_off / ACTIVATION_ELEM_PER_THREAD;

    ActFn<ComputeElem> fn{};
    constexpr bool IsSiLu = std::is_same_v<ActFn<float>, cutlass::epilogue::thread::SiLu<float>>;
    bool const use_custom_swiglu = gated && IsSiLu && (activation_params.alpha != 1.0f || activation_params.beta != 0.0f || isfinite(activation_params.limit));
    bool const is_interleaved = gated && activation_params.swiglu_fusion == 1;

    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
      ComputeElem gate_part;
      ComputeElem linear_part;

      if (is_interleaved) {
        auto* scalar_gemm = reinterpret_cast<GemmOutputType const*>(gemm_result + gemm_result_offset);
        for (int i = 0; i < ACTIVATION_ELEM_PER_THREAD; ++i) {
          int64_t global_elem = elem_index * ACTIVATION_ELEM_PER_THREAD + i;
          gate_part[i] = static_cast<float>(scalar_gemm[2 * global_elem]);
          linear_part[i] = static_cast<float>(scalar_gemm[2 * global_elem + 1]);
        }
      } else {
        // If not gated, gate_part reads from elem_index (gated_off_vec is 0).
        // If gated, gate_part reads from elem_index + inter_size_vec (chunk 1).
        gate_part = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index + gated_off_vec]);
        if (gated) {
          linear_part = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index]);
        }
      }

      if (bias_ptr) {
        if (is_interleaved) {
          auto* scalar_bias = reinterpret_cast<ScaleBiasType const*>(bias_ptr + bias_offset);
          for (int i = 0; i < ACTIVATION_ELEM_PER_THREAD; ++i) {
            int64_t global_elem = elem_index * ACTIVATION_ELEM_PER_THREAD + i;
            gate_part[i] += static_cast<float>(scalar_bias[2 * global_elem]);
            linear_part[i] += static_cast<float>(scalar_bias[2 * global_elem + 1]);
          }
        } else {
          gate_part = gate_part + arrayConvert<BiasElem, ComputeElem>(bias_ptr_vec[elem_index + gated_off_vec]);
          if (gated) {
            linear_part = linear_part + arrayConvert<BiasElem, ComputeElem>(bias_ptr_vec[elem_index]);
          }
        }
      }

      ComputeElem gate_act;
      if (use_custom_swiglu) {
        for (int i = 0; i < ACTIVATION_ELEM_PER_THREAD; ++i) {
          float g = gate_part[i];
          if (isfinite(activation_params.limit)) {
            g = fminf(g, activation_params.limit);
          }
          float sigmoid = 1.0f / (1.0f + expf(-activation_params.alpha * g));
          float l = linear_part[i];
          if (isfinite(activation_params.limit)) {
            l = fminf(fmaxf(l, -activation_params.limit), activation_params.limit);
          }
          l += activation_params.beta;
          gate_act[i] = g * sigmoid * l;
        }
      } else {
        gate_act = fn(gate_part);
        if (gated) {
          gate_act = gate_act * linear_part;
        }
      }

      auto post_act_val = gate_act * quant_scale;

      if constexpr (IsNVFP4 || IsMXFP8) {
        // We use GemmOutputType as the intermediate compute type as that should always be unquantized
        auto res = quantizePackedFPXValue<GemmOutputType, T, ComputeElem, VecSize>(post_act_val,
                                                                                   global_scale_val, num_tokens_before_expert, expert, token, elem_index, inter_size, fc2_act_sf_flat,
                                                                                   IsNVFP4 ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
                                                                                           : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);
        static_assert(
            sizeof(res) == sizeof(*output_vec), "Quantized value must be the same size as the output");
        output_vec[elem_index] = res;
      } else {
        output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(post_act_val);
      }
    }

    // Pad zeros in the extra SFs along the K dimension, we do this to ensure there are no nan values in the padded
    // SF atom
    if constexpr (IsNVFP4 || IsMXFP8) {
      // Use VecSize per thread since we are just writing out zeros so every thread can process a whole vector
      size_t padding_start_offset = inter_size / VecSize + start_offset;
      size_t padding_elems_in_col = padded_inter_size / VecSize;
      for (int64_t elem_index = padding_start_offset; elem_index < padding_elems_in_col; elem_index += stride) {
        writeSF<VecSize, VecSize>(num_tokens_before_expert, expert, /*source_row*/ -1, token, elem_index,
                                  padded_inter_size, fc2_act_sf_flat, /* input_sf */ nullptr);  // Pass nulltpr input_sf so we write 0
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;");
#endif

  // Pad zeros in the extra SFs along the N dimension, we do this to ensure there are no nan values in the padded SF
  // atom
  if constexpr (IsNVFP4 || IsMXFP8) {
    int64_t const start_offset = threadIdx.x;
    int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
    // Use VecSize per thread since we are just writing out zeros so every thread can process a whole vector
    int64_t const padded_num_elems_in_col = padded_inter_size / VecSize;
    assert(padded_inter_size % VecSize == 0);

    constexpr int64_t min_num_tokens_alignment = IsNVFP4
                                                     ? TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4
                                                     : TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX;
    static_assert((min_num_tokens_alignment & (min_num_tokens_alignment - 1)) == 0,
                  "Min num tokens alignment must be a power of two");
    // Since we don't know a priori how much padding is needed we assume the max per expert
    // NOTE: we don't (min_num_tokens_alignment-1) to have power of two divisions
    int64_t num_padding_tokens = min_num_tokens_alignment * num_experts_per_node;

    for (int64_t padding_token = blockIdx.x; padding_token < num_padding_tokens; padding_token += gridDim.x) {
      int64_t expert = padding_token / min_num_tokens_alignment;
      int64_t num_tokens_before_expert = expert_first_token_offset[expert];
      int64_t num_tokens_after_expert = expert_first_token_offset[expert + 1];
      int64_t tokens_to_expert = num_tokens_after_expert - num_tokens_before_expert;
      int64_t padding_to_expert = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(tokens_to_expert, min_num_tokens_alignment) - tokens_to_expert;
      int64_t expert_pad_idx = padding_token % min_num_tokens_alignment;
      if (expert_pad_idx < padding_to_expert) {
        for (int64_t elem_index = start_offset; elem_index < padded_num_elems_in_col; elem_index += stride) {
          // The SF buffer is padded to a multiple of MinNDimAlignment for each expert
          // This means we can safely write to offset num_tokens_after_expert + padded_token, since the next
          // expert will leave space for the padding
          writeSF<VecSize, VecSize>(num_tokens_before_expert, expert, /*source_row*/ -1,
                                    num_tokens_after_expert + expert_pad_idx, elem_index, padded_inter_size, fc2_act_sf_flat,
                                    /* input_sf */ nullptr);  // Pass nulltpr input_sf so we write 0
        }
      }
    }
  }
}

template <class T, class GemmOutputType, class ScaleBiasType>
void doActivation(T* output, GemmOutputType const* gemm_result, float const* fp8_quant, ScaleBiasType const* bias,
                  bool bias_is_broadcast, int64_t const* expert_first_token_offset, int num_experts_per_node, int64_t inter_size,
                  int64_t expanded_num_tokens, ActivationType activation_type, QuantParams const& quant_params,
                  bool use_per_expert_act_scale, TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_act_sf_flat, cudaStream_t stream,
                  ActivationParams activation_params) {
#ifdef ENABLE_FP4
  constexpr int64_t min_num_tokens_alignment = std::is_same_v<T, __nv_fp4_e2m1>
                                                   ? TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4
                                                   : TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX;
  int64_t num_padding_tokens = min_num_tokens_alignment * num_experts_per_node;
#else
  int64_t num_padding_tokens = 0;
#endif

  static int64_t const smCount = onnxruntime::llm::common::getMultiProcessorCount();
  // Note: Launching 8 blocks per SM can fully leverage the memory bandwidth (tested on B200).
  int64_t const blocks = std::min(smCount * 8, std::max(expanded_num_tokens, num_padding_tokens));
  int64_t const threads = ACTIVATION_THREADS_PER_BLOCK;

  auto fn = [&]() {
    auto fn = [&](auto block_scaling_type) {
      using namespace cutlass::epilogue::thread;
      // Switch dispatch for new enum order (matches TRT-LLM)
      switch (activation_type) {
        case ActivationType::Identity:
          return &doActivationKernel<T, GemmOutputType, ScaleBiasType, Identity, decltype(block_scaling_type)::value>;
        case ActivationType::Gelu:
        case ActivationType::Geglu:
          return &doActivationKernel<T, GemmOutputType, ScaleBiasType, GELU, decltype(block_scaling_type)::value>;
        case ActivationType::Relu:
        case ActivationType::Relu2:
          return &doActivationKernel<T, GemmOutputType, ScaleBiasType, ReLu, decltype(block_scaling_type)::value>;
        case ActivationType::Silu:
        case ActivationType::Swiglu:
        case ActivationType::SwigluBias:
        default:
          return &doActivationKernel<T, GemmOutputType, ScaleBiasType, SiLu, decltype(block_scaling_type)::value>;
      }
    };
    auto NVFP4 = onnxruntime::llm::common::ConstExprWrapper<TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType,
                                                            TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4>{};
    auto MXFPX = onnxruntime::llm::common::ConstExprWrapper<TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType,
                                                            TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX>{};
    auto NONE = onnxruntime::llm::common::ConstExprWrapper<TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType,
                                                           TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE>{};
#ifdef ENABLE_FP4
    if constexpr (std::is_same_v<T, __nv_fp4_e2m1>) {
      ORT_ENFORCE(
          quant_params.fp4.fc2.weight_block_scale, "NVFP4 block scaling is expected for FP4xFP4");
      return fn(NVFP4);
    } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      return quant_params.mxfp8_mxfp4.fc2.weight_block_scale ? fn(MXFPX) : fn(NONE);
    } else
#endif
    {
      return fn(NONE);
    }
  }();

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = onnxruntime::llm::common::getEnvEnablePDL();
  config.numAttrs = 1;
  config.attrs = attrs;
  auto const* fc2_act_global_scale = quant_params.fp4.fc2.act_global_scale
                                         ? quant_params.fp4.fc2.act_global_scale
                                     : quant_params.mxfp8_mxfp4.fc2.global_scale
                                         ? quant_params.mxfp8_mxfp4.fc2.global_scale
                                         : quant_params.fp8_mxfp4.fc2.act_global_scale;
  cudaLaunchKernelEx(&config, fn, output, gemm_result, fp8_quant, bias, bias_is_broadcast, expert_first_token_offset,

                     num_experts_per_node, inter_size, isGatedActivation(activation_type), fc2_act_global_scale,
                     use_per_expert_act_scale, fc2_act_sf_flat, activation_params);
}

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
