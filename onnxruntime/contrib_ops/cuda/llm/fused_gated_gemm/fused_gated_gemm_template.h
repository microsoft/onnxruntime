/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // __GNUC__

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/packed_stride.hpp"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm_configs.h"

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif  // __GNUC

#include "contrib_ops/cuda/llm/common/logger.h"
#include "contrib_ops/cuda/llm/fused_gated_gemm/fused_gated_gemm.h"
#include "contrib_ops/cuda/llm/fused_gated_gemm/fused_gated_gemm_kernel_template_sm90.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/common/quantization.h"
#include "contrib_ops/cuda/llm/cutlass_heuristic.h"
#include "contrib_ops/cuda/llm/cutlass_type_conversion.h"

#include <algorithm>
#include <vector>

namespace onnxruntime::llm {
namespace kernels {
namespace cutlass_kernels {
namespace tk = onnxruntime::llm::common;
namespace tkc = onnxruntime::llm::cutlass_extensions;

using namespace cute;

template <typename Gemm>
size_t typedGemmGatedKernelLauncher(Gemm gemm, typename Gemm::Arguments args, void* D, void const* A, void const* B,
                                    void const* C_bias, char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr) {
  ORT_LLM_LOG_ENTRY();

  using ElementT = typename Gemm::ElementA;

  // Check shared memory size; throw when SMEM exceeds
  int smem_size = int(sizeof(typename Gemm::GemmKernel::SharedStorage));
  static int mMaxSmemSize = tk::getMaxSharedMemoryPerBlockOptin();
  if (smem_size > mMaxSmemSize) {
    std::string errMsg = "SMEM size exceeds maximum allowed. Required " + std::to_string(smem_size) + ", got " + std::to_string(mMaxSmemSize);
    ORT_THROW("[LLM Error][fusedGatedGemm Runner] " + errMsg);
  }

  // Return workspace size
  if (!A && !B && !C_bias && !D) {
    return gemm.get_workspace_size(args);
  }

  if (gemm.get_workspace_size(args) > workspaceBytes) {
    std::string errMsg("Requested workspace size insufficient. Required " + std::to_string(gemm.get_workspace_size(args)) + ", got " + std::to_string(workspaceBytes));
    ORT_THROW("[LLM Error][fusedGatedGemm Runner] " + errMsg);
  }

  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string errMsg = "fusedGatedGemm cutlass kernel not implemented given the params. Error: " + std::string(cutlassGetStatusString(can_implement));
    ORT_THROW("[LLM Error][fusedGatedGemm Runner] " + errMsg);
  }

  auto initStatus = gemm.initialize(args, workspace, stream);
  if (initStatus != cutlass::Status::kSuccess) {
    std::string errMsg = "Failed to initialize. Error: " + std::string(cutlassGetStatusString(initStatus));
    ORT_THROW("[LLM Error][fusedGatedGemm Runner] " + errMsg);
  }

  auto runStatus = gemm.run(stream);
  if (runStatus != cutlass::Status::kSuccess) {
    std::string errMsg = "Failed to run gemm. Error: " + std::string(cutlassGetStatusString(runStatus));
    ORT_THROW("[LLM Error][fusedGatedGemm Runner] " + errMsg);
  }
  return gemm.get_workspace_size(args);
}

template <typename Gemm, bool SwapAB>
typename Gemm::Arguments prepareGemmArgsSm90(void* D, void const* A, void const* B, void const* C_bias,
                                             tk::QuantMode quantOption, int m, int n, int k, float scale_d0, float scale_d1, float scale_output,
                                             tkc::CutlassGemmConfig gemmConfig) {
  using ElementT = typename Gemm::ElementA;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  int arg_m = m;
  int arg_n = n / 2;
  ElementT const* ptr_A = reinterpret_cast<ElementT const*>(A);
  ElementT const* ptr_B = reinterpret_cast<ElementT const*>(B);
  if constexpr (SwapAB) {
    arg_m = n / 2;
    arg_n = m;
    ptr_A = reinterpret_cast<ElementT const*>(B);
    ptr_B = reinterpret_cast<ElementT const*>(A);
  }
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(arg_m, k, 1));
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(arg_n, k, 1));
  StrideC stride_C;
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(arg_m, arg_n, 1));
  typename Gemm::Arguments args = {cutlass::gemm::GemmUniversalMode::kGemm, {arg_m, arg_n, k, 1}, {ptr_A, stride_A, ptr_B, stride_B, scale_d0, scale_d1}, {{},  // epilogue.thread
                                                                                                                                                           nullptr,
                                                                                                                                                           stride_C,
                                                                                                                                                           reinterpret_cast<ElementT*>(D),
                                                                                                                                                           stride_D}};
  args.epilogue.thread.alpha = scale_output;
  return args;
}

template <typename T, typename CTAShape, typename ClusterShape,
          template <class> typename Activation = cutlass::epilogue::thread::SiLu, bool SwapAB = true>
size_t genericGemmGatedKernelLauncherSm90(void* D, void const* A, void const* B, void const* C_bias,
                                          tk::QuantMode quantOption, int m, int n, int k, float scale_d0, float scale_d1, float scale_output,
                                          tkc::CutlassGemmConfig gemmConfig, char* workspace, size_t workspaceBytes, cudaStream_t stream,
                                          int* occupancy = nullptr) {
  ORT_LLM_LOG_ENTRY();

#ifdef COMPILE_HOPPER_TMA_GEMMS
  using ElementT = typename CudaToCutlassTypeAdapter<T>::type;
  using AccumElementType = float;
  using MainloopScheduleType = cute::conditional_t<size<0>(CTAShape{}) == Int<64>{},
                                                   cutlass::gemm::KernelTmaWarpSpecializedPingpongFP8FastAccum,
                                                   cutlass::gemm::KernelTmaWarpSpecializedCooperativeFP8FastAccum>;
  using EpilogueScheduleType = cute::conditional_t<size<0>(CTAShape{}) == Int<64>{},
                                                   cutlass::epilogue::TmaWarpSpecialized, cutlass::epilogue::TmaWarpSpecializedCooperative>;
  using TileSchedulerType = void;
  using Gemm = typename DeviceGemmGatedSm90<ElementT, AccumElementType, CTAShape, ClusterShape, MainloopScheduleType,
                                            EpilogueScheduleType, TileSchedulerType, Activation, SwapAB>::Gemm;
  auto args = prepareGemmArgsSm90<Gemm, SwapAB>(
      D, A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, scale_output, gemmConfig);
  return typedGemmGatedKernelLauncher(Gemm{}, args, D, A, B, C_bias, workspace, workspaceBytes, stream, occupancy);
#else   // COMPILE_HOPPER_TMA_GEMMS
  ORT_THROW(
      "[LLM Error][GemmGatedKernelLauncherSm90] Please recompile with support for hopper by passing 90-real "
      "as an arch to build_wheel.py.");
#endif  // COMPILE_HOPPER_TMA_GEMMS
}

template <typename T, typename CTAShape>
size_t dispatchGemmConfigSm90(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption,
                              int m, int n, int k, float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig,
                              char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr) {
  ORT_LLM_LOG_ENTRY();
  switch (gemmConfig.cluster_shape) {
    case tkc::ClusterShape::ClusterShape_1x1x1:
      return genericGemmGatedKernelLauncherSm90<T, CTAShape, Shape<_1, _1, _1>>(D, A, B, C_bias, quantOption, m, n, k,
                                                                                scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
      break;
    case tkc::ClusterShape::ClusterShape_2x1x1:
      return genericGemmGatedKernelLauncherSm90<T, CTAShape, Shape<_2, _1, _1>>(D, A, B, C_bias, quantOption, m, n, k,
                                                                                scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
      break;
    case tkc::ClusterShape::ClusterShape_1x2x1:
      return genericGemmGatedKernelLauncherSm90<T, CTAShape, Shape<_1, _2, _1>>(D, A, B, C_bias, quantOption, m, n, k,
                                                                                scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
      break;
    case tkc::ClusterShape::ClusterShape_2x2x1:
      return genericGemmGatedKernelLauncherSm90<T, CTAShape, Shape<_2, _2, _1>>(D, A, B, C_bias, quantOption, m, n, k,
                                                                                scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
      break;
    case tkc::ClusterShape::ClusterShape_1x8x1:
      return genericGemmGatedKernelLauncherSm90<T, CTAShape, Shape<_1, _8, _1>>(D, A, B, C_bias, quantOption, m, n, k,
                                                                                scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
      break;
    case tkc::ClusterShape::ClusterShape_8x1x1:
      return genericGemmGatedKernelLauncherSm90<T, CTAShape, Shape<_8, _1, _1>>(D, A, B, C_bias, quantOption, m, n, k,
                                                                                scale_d0, scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
      break;
    default:
      ORT_THROW(
          "[LLM Error][CutlassFusedGatedGemmRunner][dispatchGemmConfigSm90] Config is invalid for fused "
          "gated GEMM.");
      break;
  }
}

template <typename T>
size_t dispatchGemmToCutlassSm90(void* D, void const* A, void const* B, void const* C_bias, tk::QuantMode quantOption,
                                 int m, int n, int k, float scale_d0, float scale_d1, float scale_output, tkc::CutlassGemmConfig gemmConfig,
                                 char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr) {
  ORT_LLM_LOG_ENTRY();
  static_assert(std::is_same_v<T, __nv_fp8_e4m3>, "fusedGatedGemmSm90 only support FP8(e4m3)");
  constexpr int Ktile = 128 / sizeof(T);
  using _Ktile = Int<Ktile>;
  switch (gemmConfig.tile_config_sm90) {
    case tkc::CutlassTileConfigSM90::CtaShape64x16x128B:
      return dispatchGemmConfigSm90<T, Shape<_64, _16, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
                                                                scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
      break;
    case tkc::CutlassTileConfigSM90::CtaShape64x32x128B:
      return dispatchGemmConfigSm90<T, Shape<_64, _32, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
                                                                scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
      break;
    case tkc::CutlassTileConfigSM90::CtaShape64x64x128B:
      return dispatchGemmConfigSm90<T, Shape<_64, _64, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
                                                                scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
      break;
    case tkc::CutlassTileConfigSM90::CtaShape64x128x128B:
      return dispatchGemmConfigSm90<T, Shape<_64, _128, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
                                                                 scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
      break;
    case tkc::CutlassTileConfigSM90::CtaShape128x16x128B:
      return dispatchGemmConfigSm90<T, Shape<_128, _16, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
                                                                 scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
      break;
    case tkc::CutlassTileConfigSM90::CtaShape128x32x128B:
      return dispatchGemmConfigSm90<T, Shape<_128, _32, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
                                                                 scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
      break;
    case tkc::CutlassTileConfigSM90::CtaShape128x64x128B:
      return dispatchGemmConfigSm90<T, Shape<_128, _64, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
                                                                 scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
      break;
    case tkc::CutlassTileConfigSM90::CtaShape128x128x128B:
      return dispatchGemmConfigSm90<T, Shape<_128, _128, _Ktile>>(D, A, B, C_bias, quantOption, m, n, k, scale_d0,
                                                                  scale_d1, scale_output, gemmConfig, workspace, workspaceBytes, stream, occupancy);
      break;
    case tkc::CutlassTileConfigSM90::Undefined:
      ORT_THROW(
          "[LLM Error][CutlassFusedGatedGemmRunner][dispatchGemmToCutlassSm90] gemm config undefined.");
      break;
    case tkc::CutlassTileConfigSM90::ChooseWithHeuristic:
      ORT_THROW(
          "[LLM Error][CutlassFusedGatedGemmRunner][dispatchGemmToCutlassSm90] gemm config should have "
          "already been set by "
          "heuristic.");
      break;
    default:
      ORT_THROW(
          "[LLM Error][CutlassFusedGatedGemmRunner][dispatchGemmToCutlassSm90] Config is invalid for fused "
          "gated GEMM.");
      break;
  }
}

template <typename T>
CutlassFusedGatedGemmRunner<T>::CutlassFusedGatedGemmRunner() {
  ORT_LLM_LOG_ENTRY();
  mSm = tk::getSMVersion();
}

template <typename T>
CutlassFusedGatedGemmRunner<T>::~CutlassFusedGatedGemmRunner() {
  ORT_LLM_LOG_ENTRY();
}

template <typename T>
size_t CutlassFusedGatedGemmRunner<T>::dispatchToArch(void* D, void const* A, void const* B, void const* C_bias,
                                                      tk::QuantMode quantOption, int m, int n, int k, float scale_d0, float scale_d1, float scale_output,
                                                      tkc::CutlassGemmConfig gemmConfig, char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy) {
  ORT_LLM_LOG_ENTRY();
  if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
#ifndef PLACEHOLDER_KERNELS
    if (mSm == 90) {
      return dispatchGemmToCutlassSm90<T>(D, A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, scale_output,
                                          gemmConfig, workspace, workspaceBytes, stream, occupancy);
    } else
#endif
    {
      ORT_THROW(
          "[LLM Error][CutlassFusedGatedGemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS fused "
          "gated GEMM");
    }
  } else {
    ORT_THROW(
        "[LLM Error][CutlassFusedGatedGemmRunner][GEMM Dispatch] dtype unsupported for CUTLASS fused "
        "gated "
        "GEMM");
  }
  return 0;
}

template <typename T>
void CutlassFusedGatedGemmRunner<T>::gemm(void* D, void const* A, void const* B, void const* C_bias,
                                          tk::QuantMode quantOption, int m, int n, int k, float scale_d0, float scale_d1, float scale_output,
                                          tkc::CutlassGemmConfig gemmConfig, char* workspace, size_t workspaceBytes, cudaStream_t stream, int* occupancy) {
  ORT_LLM_LOG_ENTRY();
  dispatchToArch(D, A, B, C_bias, quantOption, m, n, k, scale_d0, scale_d1, scale_output, gemmConfig, workspace,
                 workspaceBytes, stream, occupancy);
}

template <typename T>
std::vector<tkc::CutlassGemmConfig> CutlassFusedGatedGemmRunner<T>::getConfigs() const {
  using tkc::CutlassGemmConfig;
  using tkc::CutlassTileConfig;
  using tkc::SplitKStyle;

  std::vector<CutlassGemmConfig> candidateConfigs;

  if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
    if (mSm != 90) {
      ORT_THROW(
          "[LLM Error][CutlassFusedGatedGemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS fused "
          "gated GEMM");
    }
    tkc::CutlassGemmConfig::CandidateConfigTypeParam config_type_param = tkc::CutlassGemmConfig::CandidateConfigTypeParam::HOPPER;
    std::vector<CutlassGemmConfig> commonConfigs = get_candidate_configs(mSm, 2, config_type_param);
    candidateConfigs.insert(candidateConfigs.end(), commonConfigs.begin(), commonConfigs.end());
    // registers are not enough when N_tile is 256, remove some configs
    candidateConfigs.erase(std::remove_if(candidateConfigs.begin(), candidateConfigs.end(),
                                          [](auto const& config) {
                                            return config.tile_config_sm90 == tkc::CutlassTileConfigSM90::CtaShape64x256x128B || config.tile_config_sm90 == tkc::CutlassTileConfigSM90::CtaShape128x256x128B;
                                          }),
                           candidateConfigs.end());
    std::vector<tkc::CutlassTileConfigSM90> tilesSm90 = {tkc::CutlassTileConfigSM90::CtaShape64x16x128B, tkc::CutlassTileConfigSM90::CtaShape64x32x128B,
                                                         tkc::CutlassTileConfigSM90::CtaShape64x64x128B, tkc::CutlassTileConfigSM90::CtaShape64x128x128B,
                                                         tkc::CutlassTileConfigSM90::CtaShape128x16x128B, tkc::CutlassTileConfigSM90::CtaShape128x32x128B,
                                                         tkc::CutlassTileConfigSM90::CtaShape128x64x128B, tkc::CutlassTileConfigSM90::CtaShape128x128x128B};
    for (auto const& tile_config : tilesSm90) {
      {
        CutlassGemmConfig config(tile_config, tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO,
                                 tkc::ClusterShape::ClusterShape_1x8x1);
        candidateConfigs.push_back(config);
      }
      {
        CutlassGemmConfig config(tile_config, tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO,
                                 tkc::ClusterShape::ClusterShape_8x1x1);
        candidateConfigs.push_back(config);
      }
    }
  } else {
    ORT_THROW(
        "[LLM Error][CutlassFusedGatedGemmRunner][GEMM Dispatch] dtype unsupported for CUTLASS fused "
        "gated "
        "GEMM");
  }
  return candidateConfigs;
}

// Note: can be quite heavyweight; when possible, call once
template <typename T>
size_t CutlassFusedGatedGemmRunner<T>::getWorkspaceSizeImpl(int const m, int const n, int const k) {
  ORT_LLM_LOG_ENTRY();
  size_t workspace_size = 0;
  auto gemmConfigs = CutlassFusedGatedGemmRunner<T>{}.getConfigs();
  for (auto const& gemmConfig : gemmConfigs) {
    try {
      size_t curr_workspace_size = CutlassFusedGatedGemmRunner<T>::dispatchToArch(
          nullptr, nullptr, nullptr, nullptr, tk::QuantMode{}, m, n, k, 1.0, 1.0, 1.0, gemmConfig, nullptr, 0, 0);
      workspace_size = std::max(workspace_size, curr_workspace_size);
    } catch (std::runtime_error& e) {
      // Swallow errors when SMEM exceeds maximum allowed
      continue;
    }
  }

  return workspace_size;
}

template <typename T>
size_t CutlassFusedGatedGemmRunner<T>::getWorkspaceSize(int const m, int const n, int const k) {
  ORT_LLM_LOG_ENTRY();

  // Custom hash function for the MNK type
  using MNK = std::tuple<int, int, int>;

  struct MNKHash {
    size_t operator()(const MNK& mnk) const {
      auto h1 = std::hash<int>{}(std::get<0>(mnk));
      auto h2 = std::hash<int>{}(std::get<1>(mnk));
      auto h3 = std::hash<int>{}(std::get<2>(mnk));
      return h1 ^ h2 ^ h3;
    }
  };

  static std::unordered_map<MNK, size_t, MNKHash> workspace_hashmap;

  size_t workspace_size = 0;
  if (workspace_hashmap.find(std::make_tuple(m, n, k)) == workspace_hashmap.end()) {
    workspace_size = CutlassFusedGatedGemmRunner<T>::getWorkspaceSizeImpl(m, n, k);
    workspace_hashmap[std::make_tuple(m, n, k)] = workspace_size;
  } else {
    workspace_size = workspace_hashmap[std::make_tuple(m, n, k)];
  }
  return workspace_size;
}

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace onnxruntime::llm
