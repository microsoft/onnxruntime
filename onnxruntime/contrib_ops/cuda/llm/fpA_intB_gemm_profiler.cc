/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#if USE_FPA_INTB_GEMM
#include "contrib_ops/cuda/llm/fpA_intB_gemm_profiler.h"
#include "contrib_ops/cuda/llm/common/workspace.h"
#include "core/platform/env_var_utils.h"

#include <algorithm>
#include <set>
#include <sstream>

using namespace onnxruntime::llm::common;
using namespace onnxruntime::llm::kernels::cutlass_kernels;

namespace onnxruntime::llm::kernels::weight_only {

void WeightOnlyGroupwiseQuantGemmPluginProfiler::runTactic(
    int m, int n, int k,
    WeightOnlyGroupwiseQuantGemmPluginProfiler::Config const& tactic, char* workspace, cudaStream_t const& stream) {
  int const originalN = mQuantBits == 8 ? n * FP16_INT8_RATIO : n * FP16_INT4_RATIO;
  half* actPtr = reinterpret_cast<half*>(workspace);
  void* weightPtr = nextWorkspacePtr(reinterpret_cast<int8_t*>(actPtr), m * k * sizeof(half));
  half* inputScalesPtr = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(weightPtr), n * k * sizeof(half)));
  half* zerosPtr = reinterpret_cast<half*>(
      nextWorkspacePtr(reinterpret_cast<int8_t*>(inputScalesPtr), k * originalN * sizeof(half) / mGroupSize));
  half* biasesPtr = reinterpret_cast<half*>(
      nextWorkspacePtr(reinterpret_cast<int8_t*>(zerosPtr), k * originalN * sizeof(half) / mGroupSize));
  half* outputPtr = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(biasesPtr), n * sizeof(half)));
  char* workspacePtr = reinterpret_cast<char*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(outputPtr), m * originalN * sizeof(half)));

  if (!mHasZeros) {
    zerosPtr = nullptr;
  }

  if (!mHasBiases) {
    biasesPtr = nullptr;
  }

  if (tactic.enableCudaKernel) {
    // run CUDA kernel
    void const* pre_quant_scale_ptr = nullptr;
    bool apply_alpha_in_advance = false;
    float alpha = 1.0f;
    onnxruntime::llm::kernels::fpA_intB_gemv::Params params(
        actPtr, pre_quant_scale_ptr, weightPtr,
        inputScalesPtr, zerosPtr,
        biasesPtr, outputPtr,
        alpha, m, originalN, k, mGroupSize, mCudaKernelType, apply_alpha_in_advance);
    onnxruntime::llm::kernels::fpA_intB_gemv::kernel_launcher(mArch, params, stream);
  } else {
    // run CUTLASS kernel
    size_t const wsSize = mRunner->getWorkspaceSize(m, originalN, k);
    if (mQuantBits == 8) {
      mRunner->gemm(actPtr, reinterpret_cast<int8_t*>(weightPtr), inputScalesPtr, zerosPtr, biasesPtr, outputPtr,
                    m, originalN, k, mGroupSize, tactic, workspacePtr, wsSize, stream);
    } else {
      mRunner->gemm(actPtr, reinterpret_cast<cutlass::uint4b_t*>(weightPtr), inputScalesPtr, zerosPtr, biasesPtr,
                    outputPtr, m, originalN, k, mGroupSize, tactic, workspacePtr, wsSize, stream);
    }
  }
}

size_t WeightOnlyGroupwiseQuantGemmPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k) {
  // Quantized weights are packed in FP16 format (INT4*4 -> FP16, INT8*2 -> FP16)
  int const originalN = mQuantBits == 8 ? static_cast<int>(n) * FP16_INT8_RATIO : static_cast<int>(n) * FP16_INT4_RATIO;
  std::vector<size_t> workspaces = {
      maxM * k * sizeof(half),                                                           // A
      k * n * sizeof(half),                                                              // B
      k * originalN * sizeof(half) / mGroupSize,                                         // scales
      k * originalN * sizeof(half) / mGroupSize,                                         // zeros
      originalN * sizeof(half),                                                          // biases
      maxM * originalN * sizeof(half),                                                   // C
      mRunner->getWorkspaceSize(static_cast<int>(maxM), originalN, static_cast<int>(k))  // workspace
  };
  return calculateTotalWorkspaceSize(workspaces.data(), static_cast<int>(workspaces.size()));
}

std::vector<WeightOnlyGroupwiseQuantGemmPluginProfiler::Config> WeightOnlyGroupwiseQuantGemmPluginProfiler::getTactics(
    int /*m*/, int /*n*/, int /*k*/) const {
  return mRunner->getConfigs();
}

bool WeightOnlyGroupwiseQuantGemmPluginProfiler::checkTactic(int m, int /*n*/, int /*k*/, Config const& tactic) const {
  // stop to profile Cuda kernel for m >= 16
  if (tactic.enableCudaKernel) {
    return m < 16;
  }
  return true;
}

std::vector<int> WeightOnlyGroupwiseQuantGemmPluginProfiler::ParseProfileMOverride() {
  const std::string value = onnxruntime::ParseEnvironmentVariableWithDefault<std::string>(kEnvProfileM, "");
  std::vector<int> result;
  if (value.empty()) {
    return result;
  }
  std::stringstream ss(value);
  std::string token;
  std::set<int> unique;
  while (std::getline(ss, token, ',')) {
    // Trim surrounding whitespace.
    size_t start = token.find_first_not_of(" \t");
    size_t end = token.find_last_not_of(" \t");
    if (start == std::string::npos) {
      continue;
    }
    token = token.substr(start, end - start + 1);
    try {
      int m = std::stoi(token);
      if (m > 0) {
        unique.insert(m);
      }
    } catch (const std::exception&) {
      // Ignore malformed entries.
    }
  }
  result.assign(unique.begin(), unique.end());
  return result;
}

int WeightOnlyGroupwiseQuantGemmPluginProfiler::ProfileMaxM() {
  auto override_ms = ParseProfileMOverride();
  if (!override_ms.empty()) {
    return override_ms.back();  // sorted ascending
  }
  return kDefaultProfileMaxM;
}

std::vector<int> WeightOnlyGroupwiseQuantGemmPluginProfiler::getProfileMBuckets(
    int minM, int maxM, bool /*hasWeightOnlyCudaKernel*/) const {
  int const lo = std::max(1, minM);
  int const hi = std::max(lo, maxM);

  std::set<int> buckets;

  auto override_ms = ParseProfileMOverride();
  if (!override_ms.empty()) {
    for (int m : override_ms) {
      buckets.insert(std::min(std::max(lo, m), hi));
    }
  } else {
    // Small default bucket set clamped to [lo, hi].
    static const int kDefault[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    for (int m : kDefault) {
      if (m >= lo && m <= hi) {
        buckets.insert(m);
      }
    }
  }

  // Always include the decode bucket (M=1) and the top bucket so both extremes are tuned.
  buckets.insert(lo);
  buckets.insert(hi);

  return std::vector<int>(buckets.begin(), buckets.end());
}

onnxruntime::llm::gemm_cache::MatMulNBitsKey WeightOnlyGroupwiseQuantGemmPluginProfiler::makeCacheKey(
    GemmIdCore const& gemmId, bool hasWeightOnlyCudaKernel) const {
  onnxruntime::llm::gemm_cache::MatMulNBitsKey key;
  key.n_16b = gemmId.n;
  key.k = gemmId.k;
  key.activation_dtype = (gemmId.dtype == onnxruntime::llm::nvinfer::DataType::kBF16) ? "bfloat16" : "half";
  key.weight_type = (mQuantBits == 8) ? "uint8_t" : "uint4b_t";
  key.bits = mQuantBits;
  key.block_size = mGroupSize;
  key.has_zero_points = mHasZeros;
  key.zero_point_dtype = mHasZeros ? key.weight_type : "none";
  key.gemv_enabled = hasWeightOnlyCudaKernel;
  key.packing_sm = mArch;
  return key;
}

void WeightOnlyGroupwiseQuantGemmPluginProfiler::loadPersistentCache(
    GemmIdCore const& gemmId, MProfileMap& map, bool hasWeightOnlyCudaKernel) {
  if (mCache == nullptr) {
    return;
  }
  auto key = makeCacheKey(gemmId, hasWeightOnlyCudaKernel);
  auto buckets = mCache->GetAll(key);
  if (buckets.empty()) {
    return;
  }

  // Validate CUTLASS tactics loaded from disk against the tactics this runner can actually
  // dispatch. A parseable-but-incompatible cache row (e.g. hand-edited, or written by a build
  // whose signature happens to match but whose tactic set differs) would otherwise be handed
  // straight to the kernel. Non-matching CUTLASS tactics are dropped so the bucket is re-profiled.
  // The synthetic CUDA-GEMV tactic (enableCudaKernel) is not part of getConfigs(); its validity
  // is already keyed by gemv_enabled in the cache key, so it is accepted as-is.
  auto const valid_configs = getTactics(0, gemmId.n, gemmId.k);
  auto is_valid_cutlass = [&valid_configs](Config const& c) {
    for (auto const& v : valid_configs) {
      if (v.sm_version == c.sm_version && v.is_tma_warp_specialized == c.is_tma_warp_specialized &&
          v.tile_config_sm80 == c.tile_config_sm80 && v.tile_config_sm90 == c.tile_config_sm90 &&
          v.tile_config_sm100 == c.tile_config_sm100 && v.tile_config_sm120 == c.tile_config_sm120 &&
          v.split_k_style == c.split_k_style && v.split_k_factor == c.split_k_factor &&
          v.stages == c.stages && v.cluster_shape == c.cluster_shape &&
          v.mainloop_schedule == c.mainloop_schedule && v.epilogue_schedule == c.epilogue_schedule) {
        return true;
      }
    }
    return false;
  };

  for (auto const& [m, config] : buckets) {
    if (config.has_value()) {
      if (!checkTactic(m, gemmId.n, gemmId.k, *config)) {
        ORT_LLM_LOG_WARNING("Dropping unsupported cached fpA_intB tactic from the tactic cache; re-profiling.");
        continue;
      }
      if (!config->enableCudaKernel && !is_valid_cutlass(*config)) {
        ORT_LLM_LOG_WARNING("Dropping incompatible cached fpA_intB tactic from the tactic cache; re-profiling.");
        continue;
      }
    }
    // Do not clobber tactics already selected in-process this session.
    map.emplace(m, config);
  }
}

bool WeightOnlyGroupwiseQuantGemmPluginProfiler::stageProfiledTactics(
    GemmIdCore const& gemmId, MProfileMap const& map, bool hasWeightOnlyCudaKernel) {
  if (mCache == nullptr) {
    return false;
  }
  auto key = makeCacheKey(gemmId, hasWeightOnlyCudaKernel);
  bool added = false;
  for (auto const& [m, config] : map) {
    // Only stage buckets that are not already recorded (skips re-staging cache hits).
    if (!mCache->Get(key, m).has_value()) {
      mCache->Put(key, m, config);
      added = true;
    }
  }
  return added;
}

void WeightOnlyGroupwiseQuantGemmPluginProfiler::storePersistentCache(
    GemmIdCore const& gemmId, MProfileMap const& map, bool hasWeightOnlyCudaKernel) {
  // Construction-time sweep: stage and flush immediately so the cache file exists while the session
  // is alive (the offline tuning tool reads it before the process exits).
  if (stageProfiledTactics(gemmId, map, hasWeightOnlyCudaKernel)) {
    auto status = mCache->Flush();
    if (!status.IsOK()) {
      ORT_LLM_LOG_WARNING("Failed to flush MatMulNBits gemm tactic cache: " + status.ErrorMessage());
    }
  }
}

void WeightOnlyGroupwiseQuantGemmPluginProfiler::stagePersistentCache(
    GemmIdCore const& gemmId, MProfileMap const& map, bool hasWeightOnlyCudaKernel) {
  // Teardown path: stage only (no disk write). Every MatMulNBits kernel destructor calls this, so
  // flushing here would rewrite the whole cache file once per node. The staged tactics are written
  // to disk a single time when the process-global cache table is destroyed (see matmul_nbits.cc).
  stageProfiledTactics(gemmId, map, hasWeightOnlyCudaKernel);
}

}  // namespace onnxruntime::llm::kernels::weight_only
#endif
