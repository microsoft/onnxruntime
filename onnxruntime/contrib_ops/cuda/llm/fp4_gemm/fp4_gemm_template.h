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

#ifdef ENABLE_FP4
#include "contrib_ops/cuda/llm/common/logger.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

#ifdef _WIN32
#pragma nv_diag_suppress 177
#endif

#ifndef _WIN32
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif  // #ifndef _WIN32

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

#include "cutlass/arch/arch.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/gemm.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm_configs.h"

#ifndef _WIN32
#pragma GCC diagnostic pop
#endif  // #ifndef _WIN32

#include "contrib_ops/cuda/llm/common/logger.h"
#include "contrib_ops/cuda/llm/fp4_gemm/fp4_gemm.h"
#include "contrib_ops/cuda/llm/fp4_gemm/mxfp8_mxfp4_gemm_template_sm100.h"
#include "contrib_ops/cuda/llm/fp4_gemm/nvfp4_nvfp4_gemm_template_sm100.h"
#include "contrib_ops/cuda/llm/fp4_gemm/nvfp4_nvfp4_gemm_template_sm120.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

#ifdef _WIN32
#pragma warning(pop)
#endif

namespace onnxruntime::llm {
namespace kernels {
namespace cutlass_kernels {
using namespace cute;

namespace tkc = onnxruntime::llm::cutlass_extensions;

template <typename T, typename CTA_M_, typename CTA_N_, typename CTA_K_>
size_t dispatchNVFP4xNVFP4GemmClusterShapeSm100(T* D, void const* A, void const* B, void const* input_sf,
                                                void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
                                                tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,
                                                int* occupancy = nullptr) {
  ORT_LLM_LOG_ENTRY();

  switch (gemmConfig.cluster_shape) {
    case tkc::ClusterShape::ClusterShape_1x1x1:
      return genericFp4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<1>, cute::Int<1>, cute::Int<1>, _1SM>(
          D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case tkc::ClusterShape::ClusterShape_2x1x1:
      return genericFp4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<2>, cute::Int<1>, cute::Int<1>, _2SM>(
          D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case tkc::ClusterShape::ClusterShape_1x2x1:
      return genericFp4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<1>, cute::Int<2>, cute::Int<1>, _1SM>(
          D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case tkc::ClusterShape::ClusterShape_2x2x1:
      return genericFp4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<2>, cute::Int<2>, cute::Int<1>, _2SM>(
          D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case tkc::ClusterShape::ClusterShape_1x4x1:
      return genericFp4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<1>, cute::Int<4>, cute::Int<1>, _1SM>(
          D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case tkc::ClusterShape::ClusterShape_4x2x1:
      return genericFp4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<4>, cute::Int<2>, cute::Int<1>, _2SM>(
          D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case tkc::ClusterShape::ClusterShape_2x4x1:
      return genericFp4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<2>, cute::Int<4>, cute::Int<1>, _2SM>(
          D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    case tkc::ClusterShape::ClusterShape_4x4x1:
      return genericFp4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<4>, cute::Int<4>, cute::Int<1>, _2SM>(
          D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
          stream, occupancy);
      break;
    default:
      ORT_THROW(
          "[LLM Error][FP4][dispatch_gemm_cluster_shape] Config is invalid for FP4 GEMM.");
      break;
  }
}

template <typename T>
size_t dispatchNVFP4xNVFP4GemmCTAShapeSm100(T* D, void const* A, void const* B, void const* input_sf,
                                            void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
                                            tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,
                                            int* occupancy = nullptr) {
  ORT_LLM_LOG_ENTRY();
  // Several constraints:
  // Cta N should be one of 128/192/256.
  // M-mode size should be 128 or 256 for 2 CTA cluster MMA;
  // M-mode size should be 128 for 1 CTA cluster OMMA.
  // K256 looks to be better than K128
  switch (gemmConfig.tile_config_sm100) {
    case tkc::CutlassTileConfigSM100::CtaShape128x64x128B:
      return dispatchNVFP4xNVFP4GemmClusterShapeSm100<T, cute::Int<128>, cute::Int<64>, cute::Int<128>>(D, A, B,
                                                                                                        input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
                                                                                                        occupancy);
      break;
    case tkc::CutlassTileConfigSM100::CtaShape128x256x128B:
      return dispatchNVFP4xNVFP4GemmClusterShapeSm100<T, cute::Int<128>, cute::Int<256>, cute::Int<128>>(D, A, B,
                                                                                                         input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
                                                                                                         occupancy);
      break;
    case tkc::CutlassTileConfigSM100::CtaShape128x128x256B:
      return dispatchNVFP4xNVFP4GemmClusterShapeSm100<T, cute::Int<128>, cute::Int<128>, cute::Int<256>>(D, A, B,
                                                                                                         input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
                                                                                                         occupancy);
      break;
    case tkc::CutlassTileConfigSM100::CtaShape128x256x256B:
      return dispatchNVFP4xNVFP4GemmClusterShapeSm100<T, cute::Int<128>, cute::Int<256>, cute::Int<256>>(D, A, B,
                                                                                                         input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
                                                                                                         occupancy);
      break;
    case tkc::CutlassTileConfigSM100::Undefined:
      ORT_THROW("[LLM Error][FP4][dispatch_gemm_cta_shape] Gemm config undefined.");
      break;
    case tkc::CutlassTileConfigSM100::ChooseWithHeuristic:
      ORT_THROW(
          "[LLM Error][FP4][dispatch_gemm_cta_shape] Gemm config should have already been set by "
          "heuristic.");
      break;
    default:
      ORT_THROW("[LLM Error][FP4][dispatch_gemm_cta_shape] Config is invalid for FP4 GEMM.");
      break;
  }
}

template <typename T, typename CTA_M_, typename CTA_N_, typename CTA_K_>
size_t dispatchNVFP4xNVFP4GemmClusterShapeSm120(T* D, void const* A, void const* B, void const* input_sf,
                                                void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
                                                tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,
                                                int* occupancy = nullptr) {
  ORT_LLM_LOG_ENTRY();

  switch (gemmConfig.cluster_shape) {
    case tkc::ClusterShape::ClusterShape_1x1x1:
      return genericFp4GemmKernelLauncherSm120<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<1>, cute::Int<1>, cute::Int<1>>(D,
                                                                                                                    A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
                                                                                                                    occupancy);
      break;
    default:
      ORT_THROW(
          "[LLM Error][FP4][dispatch_gemm_cluster_shape] Config is invalid for FP4 GEMM.");
      break;
  }
}

template <typename T>
size_t dispatchNVFP4xNVFP4GemmCTAShapeSm120(T* D, void const* A, void const* B, void const* input_sf,
                                            void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
                                            tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,
                                            int* occupancy = nullptr) {
  ORT_LLM_LOG_ENTRY();
  ORT_LLM_LOG_DEBUG(onnxruntime::MakeString("gemmConfig.tile_config_sm120: ", gemmConfig.tile_config_sm120));

  switch (gemmConfig.tile_config_sm120) {
    case tkc::CutlassTileConfigSM120::CtaShape128x128x256B:
      return dispatchNVFP4xNVFP4GemmClusterShapeSm120<T, cute::Int<128>, cute::Int<128>, cute::Int<256>>(D, A, B,
                                                                                                         input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
                                                                                                         occupancy);
      break;
    case tkc::CutlassTileConfigSM120::CtaShape256x128x128B:
      return dispatchNVFP4xNVFP4GemmClusterShapeSm120<T, cute::Int<256>, cute::Int<128>, cute::Int<128>>(D, A, B,
                                                                                                         input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
                                                                                                         occupancy);
      break;
    case tkc::CutlassTileConfigSM120::Undefined:
      ORT_THROW("[LLM Error][FP4][sm120][dispatch_gemm_cta_shape] Gemm config undefined.");
      break;
    case tkc::CutlassTileConfigSM120::ChooseWithHeuristic:
      ORT_THROW(
          "[LLM Error][FP4][sm120][dispatch_gemm_cta_shape] Gemm config should have already been set by "
          "heuristic.");
      break;
    default:
      ORT_THROW(
          "[LLM Error][FP4][sm120][dispatch_gemm_cta_shape] Config is invalid for FP4 GEMM.");
      break;
  }
}

template <typename T, typename CTA_M_, typename CTA_N_, typename CTA_K_>
size_t dispatchMXFP8xMXFP4GemmClusterShapeSm100(T* D, void const* A, void const* B, void const* input_sf,
                                                void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
                                                tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,
                                                int* occupancy = nullptr) {
  ORT_LLM_LOG_ENTRY();

  switch (gemmConfig.cluster_shape) {
    case tkc::ClusterShape::ClusterShape_2x1x1:
      return genericMXFP8xMXFP4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<2>, cute::Int<1>, cute::Int<1>,
                                                  __2SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
                                                         stream, occupancy);
      break;
    case tkc::ClusterShape::ClusterShape_2x2x1:
      return genericMXFP8xMXFP4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<2>, cute::Int<2>, cute::Int<1>,
                                                  __2SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
                                                         stream, occupancy);
      break;
    case tkc::ClusterShape::ClusterShape_4x2x1:
      return genericMXFP8xMXFP4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<4>, cute::Int<2>, cute::Int<1>,
                                                  __2SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
                                                         stream, occupancy);
      break;
    case tkc::ClusterShape::ClusterShape_2x4x1:
      return genericMXFP8xMXFP4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<2>, cute::Int<4>, cute::Int<1>,
                                                  __2SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
                                                         stream, occupancy);
      break;
    case tkc::ClusterShape::ClusterShape_4x4x1:
      return genericMXFP8xMXFP4GemmKernelLauncher<T, CTA_M_, CTA_N_, CTA_K_, cute::Int<4>, cute::Int<4>, cute::Int<1>,
                                                  __2SM>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes,
                                                         stream, occupancy);
      break;
    default:
      ORT_THROW(
          "[LLM Error][FP4][dispatch_gemm_cluster_shape] Config is invalid for FP4 GEMM.");
      break;
  }
}

template <typename T>
size_t dispatchMXFP8xMXFP4GemmCTAShapeSm100(T* D, void const* A, void const* B, void const* input_sf,
                                            void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
                                            tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,
                                            int* occupancy = nullptr) {
  ORT_LLM_LOG_ENTRY();
  switch (gemmConfig.tile_config_sm100) {
    case tkc::CutlassTileConfigSM100::CtaShape128x64x128B:
      return dispatchMXFP8xMXFP4GemmClusterShapeSm100<T, cute::Int<128>, cute::Int<64>, cute::Int<128>>(D, A, B,
                                                                                                        input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
                                                                                                        occupancy);
      break;
    case tkc::CutlassTileConfigSM100::CtaShape128x256x128B:
      return dispatchMXFP8xMXFP4GemmClusterShapeSm100<T, cute::Int<128>, cute::Int<256>, cute::Int<128>>(D, A, B,
                                                                                                         input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
                                                                                                         occupancy);
      break;
    case tkc::CutlassTileConfigSM100::CtaShape128x128x256B:
      return dispatchMXFP8xMXFP4GemmClusterShapeSm100<T, cute::Int<128>, cute::Int<128>, cute::Int<256>>(D, A, B,
                                                                                                         input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
                                                                                                         occupancy);
      break;
    case tkc::CutlassTileConfigSM100::CtaShape128x256x256B:
      return dispatchMXFP8xMXFP4GemmClusterShapeSm100<T, cute::Int<128>, cute::Int<256>, cute::Int<256>>(D, A, B,
                                                                                                         input_sf, weight_sf, global_sf, m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream,
                                                                                                         occupancy);
      break;
    case tkc::CutlassTileConfigSM100::Undefined:
      ORT_THROW("[LLM Error][FP4][dispatch_gemm_cta_shape] Gemm config undefined.");
      break;
    case tkc::CutlassTileConfigSM100::ChooseWithHeuristic:
      ORT_THROW(
          "[LLM Error][FP4][dispatch_gemm_cta_shape] Gemm config should have already been set by "
          "heuristic.");
      break;
    default:
      ORT_THROW("[LLM Error][FP4][dispatch_gemm_cta_shape] Config is invalid for FP4 GEMM.");
      break;
  }
}

template <typename T, FP4GemmType fp4GemmType>
CutlassFp4GemmRunner<T, fp4GemmType>::CutlassFp4GemmRunner() {
  ORT_LLM_LOG_ENTRY();
  int device{-1};
  CUDA_CALL_THROW(cudaGetDevice(&device));
  mSm = onnxruntime::llm::common::getSMVersion();
  CUDA_CALL_THROW(cudaDeviceGetAttribute(&mMultiProcessorCount, cudaDevAttrMultiProcessorCount, device));
}

template <typename T, FP4GemmType fp4GemmType>
CutlassFp4GemmRunner<T, fp4GemmType>::~CutlassFp4GemmRunner() {
  ORT_LLM_LOG_ENTRY();
}

template <typename T, FP4GemmType fp4GemmType>
size_t CutlassFp4GemmRunner<T, fp4GemmType>::dispatchToArch(T* D, void const* A, void const* B, void const* input_sf,
                                                            void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
                                                            tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream,
                                                            int* occupancy) {
  if constexpr (fp4GemmType == FP4GemmType::W4A8_MXFP4_MXFP8) {
    if (mSm == 100) {
      return dispatchMXFP8xMXFP4GemmCTAShapeSm100<T>(D, A, B, input_sf, weight_sf, global_sf, m, n, k,
                                                     batch_count, gemmConfig, workspace, workspaceBytes, stream, occupancy);
    } else {
      ORT_THROW(
          "[LLM Error][CutlassFp4GemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS FP4 GEMM");
    }
  } else if constexpr (fp4GemmType == FP4GemmType::W4A4_NVFP4_NVFP4) {
    if (mSm == 100) {
      return dispatchNVFP4xNVFP4GemmCTAShapeSm100<T>(D, A, B, input_sf, weight_sf, global_sf, m, n, k,
                                                     batch_count, gemmConfig, workspace, workspaceBytes, stream, occupancy);
    } else if (mSm == 120 || mSm == 121) {
      return dispatchNVFP4xNVFP4GemmCTAShapeSm120<T>(D, A, B, input_sf, weight_sf, global_sf, m, n, k,
                                                     batch_count, gemmConfig, workspace, workspaceBytes, stream, occupancy);
    } else {
      ORT_THROW(
          "[LLM Error][CutlassFp4GemmRunner][GEMM Dispatch] Arch unsupported for CUTLASS FP4 GEMM");
    }
  } else {
    ORT_THROW(
        "[LLM Error][CutlassFp4GemmRunner][GEMM Dispatch] FP4 Gemm type unsupported for CUTLASS FP4 GEMM");
  }
}

template <typename T, FP4GemmType fp4GemmType>
void CutlassFp4GemmRunner<T, fp4GemmType>::gemm(void* D, void const* A, void const* B, void const* input_sf,
                                                void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
                                                tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream) {
  ORT_LLM_LOG_ENTRY();
  CutlassFp4GemmRunner<T, fp4GemmType>::dispatchToArch(reinterpret_cast<T*>(D), A, B, input_sf, weight_sf, global_sf,
                                                       m, n, k, batch_count, gemmConfig, workspace, workspaceBytes, stream);
}

template <typename T, FP4GemmType fp4GemmType>
std::vector<tkc::CutlassGemmConfig> CutlassFp4GemmRunner<T, fp4GemmType>::getConfigs() const {
  using tkc::CutlassGemmConfig;
  using tkc::CutlassTileConfig;

  std::vector<CutlassGemmConfig> candidateConfigs;

  if (mSm == 100) {
    std::vector<tkc::CutlassTileConfigSM100> tilesSm100 = {
        tkc::CutlassTileConfigSM100::CtaShape128x64x128B,
        tkc::CutlassTileConfigSM100::CtaShape128x256x128B,
        tkc::CutlassTileConfigSM100::CtaShape128x128x256B,
        tkc::CutlassTileConfigSM100::CtaShape128x256x256B,
    };
    std::vector<tkc::ClusterShape> clusterShapes = {
        tkc::ClusterShape::ClusterShape_1x1x1,
        tkc::ClusterShape::ClusterShape_1x2x1,
        tkc::ClusterShape::ClusterShape_2x1x1,
        tkc::ClusterShape::ClusterShape_2x2x1,
        tkc::ClusterShape::ClusterShape_1x4x1,
        tkc::ClusterShape::ClusterShape_4x2x1,
        tkc::ClusterShape::ClusterShape_2x4x1,
        tkc::ClusterShape::ClusterShape_4x4x1,
    };
    for (auto const& tile_config : tilesSm100) {
      for (auto const& cluster_config : clusterShapes) {
        if constexpr (fp4GemmType == FP4GemmType::W4A8_MXFP4_MXFP8) {
          // Skip for high smem usage.
          if (cluster_config == tkc::ClusterShape::ClusterShape_1x1x1 || cluster_config == tkc::ClusterShape::ClusterShape_1x2x1 || cluster_config == tkc::ClusterShape::ClusterShape_1x4x1) {
            continue;
          }
        }
        CutlassGemmConfig config(
            tile_config, tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO, cluster_config);
        candidateConfigs.push_back(config);
      }
    }
  } else if (mSm == 120 || mSm == 121) {
    std::vector<tkc::CutlassTileConfigSM120> tilesSm120 = {
        // tkc::CutlassTileConfigSM120::CtaShape128x128x128B,
        tkc::CutlassTileConfigSM120::CtaShape128x128x256B,
        tkc::CutlassTileConfigSM120::CtaShape256x128x128B,

    };
    tkc::ClusterShape clusterShape = tkc::ClusterShape::ClusterShape_1x1x1;
    for (auto const& tile_config : tilesSm120) {
      CutlassGemmConfig config(
          tile_config, tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO, clusterShape);
      candidateConfigs.push_back(config);
    }
  }

  return candidateConfigs;
}

template <typename T, FP4GemmType fp4GemmType>
size_t CutlassFp4GemmRunner<T, fp4GemmType>::getWorkspaceSizeImpl(
    int const m, int const n, int const k, int const batch_count) {
  ORT_LLM_LOG_ENTRY();
  size_t workspace_size = 0;
  auto gemmConfigs = CutlassFp4GemmRunner<T, fp4GemmType>{}.getConfigs();
  for (auto const& gemmConfig : gemmConfigs) {
    try {
      size_t curr_workspace_size = CutlassFp4GemmRunner<T, fp4GemmType>::dispatchToArch(
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, m, n, k, batch_count, gemmConfig, nullptr, 0, 0);
      workspace_size = std::max(workspace_size, curr_workspace_size);
    } catch (std::runtime_error& e) {
      // Swallow errors when SMEM exceeds maximum allowed
      continue;
    }
  }
  return workspace_size;
}

template <typename T, FP4GemmType fp4GemmType>
size_t CutlassFp4GemmRunner<T, fp4GemmType>::getWorkspaceSize(
    int const m, int const n, int const k, int const batch_count) {
  ORT_LLM_LOG_ENTRY();

  // Custom hash function for the MNKB type
  using MNK = std::tuple<int, int, int, int>;

  struct MNKHash {
    size_t operator()(const MNK& mnk) const {
      auto h1 = std::hash<int>{}(std::get<0>(mnk));
      auto h2 = std::hash<int>{}(std::get<1>(mnk));
      auto h3 = std::hash<int>{}(std::get<2>(mnk));
      auto h4 = std::hash<int>{}(std::get<3>(mnk));
      return h1 ^ h2 ^ h3 ^ h4;
    }
  };

  static std::unordered_map<MNK, size_t, MNKHash> workspace_hashmap;

  size_t workspace_size = 0;
  if (workspace_hashmap.find(std::make_tuple(m, n, k, batch_count)) == workspace_hashmap.end()) {
    workspace_size = CutlassFp4GemmRunner<T, fp4GemmType>::getWorkspaceSizeImpl(m, n, k, batch_count);
    workspace_hashmap[std::make_tuple(m, n, k, batch_count)] = workspace_size;
  } else {
    workspace_size = workspace_hashmap[std::make_tuple(m, n, k, batch_count)];
  }
  return workspace_size;
}

}  // namespace cutlass_kernels
}  // namespace kernels
}  // namespace onnxruntime::llm

#endif
