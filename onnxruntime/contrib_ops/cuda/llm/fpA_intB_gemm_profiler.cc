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

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#include "contrib_ops/cuda/llm/common/workspace.h"
#include "core/common/safeint.h"
#include "core/common/parse_string.h"
#include "core/common/string_utils.h"

using namespace onnxruntime::llm::common;
using namespace onnxruntime::llm::kernels::cutlass_kernels;

namespace onnxruntime::llm::kernels::weight_only {

std::optional<std::array<size_t, 7>> ComputeWeightOnlyGemmProfilerBufferSizes(
    size_t max_m, size_t packed_n, size_t k, int quant_bits,
    size_t group_size, size_t runner_workspace_bytes, size_t streaming_l2_bytes) {
  if (max_m == 0 || packed_n == 0 || k == 0 ||
      (quant_bits != INT2_BITS && quant_bits != INT4_BITS && quant_bits != INT8_BITS) || group_size == 0) {
    return std::nullopt;
  }

  try {
    const SafeInt<size_t> original_n =
        SafeInt<size_t>(packed_n) * (FP16_BITS / quant_bits);
    constexpr size_t kElementBytes = sizeof(uint16_t);
    const SafeInt<size_t> weight_bytes = SafeInt<size_t>(k) * packed_n * kElementBytes;
    const SafeInt<size_t> scale_bytes = SafeInt<size_t>(k) * original_n * kElementBytes / group_size;
    const SafeInt<size_t> matrix_bytes = weight_bytes + scale_bytes;
    const size_t copies = streaming_l2_bytes != 0 && matrix_bytes <= streaming_l2_bytes
                              ? static_cast<size_t>(SafeInt<size_t>(streaming_l2_bytes) * 2 / matrix_bytes + 1)
                              : 1;
    return std::array<size_t, 7>{
        static_cast<size_t>(SafeInt<size_t>(max_m) * k * kElementBytes),
        static_cast<size_t>(weight_bytes * copies),
        static_cast<size_t>(scale_bytes * copies),
        static_cast<size_t>(scale_bytes),
        static_cast<size_t>(original_n * kElementBytes),
        static_cast<size_t>(SafeInt<size_t>(max_m) * original_n * kElementBytes),
        runner_workspace_bytes};
  } catch (const OnnxRuntimeException&) {
    return std::nullopt;
  }
}

std::optional<size_t> ComputeWeightOnlyGemmProfilerScratchSize(
    size_t max_m, size_t packed_n, size_t k, int quant_bits,
    size_t group_size, size_t runner_workspace_bytes, size_t streaming_l2_bytes) {
  const auto workspaces = ComputeWeightOnlyGemmProfilerBufferSizes(
      max_m, packed_n, k, quant_bits, group_size, runner_workspace_bytes, streaming_l2_bytes);
  if (!workspaces) {
    return std::nullopt;
  }
  try {
    SafeInt<size_t> total = 0;
    for (const size_t workspace : *workspaces) {
      SafeInt<size_t> aligned_workspace = workspace;
      const size_t remainder = workspace % kCudaMemAlign;
      if (remainder != 0) {
        aligned_workspace += kCudaMemAlign - remainder;
      }
      total += aligned_workspace;
    }
    return static_cast<size_t>(total);
  } catch (const OnnxRuntimeException&) {
    return std::nullopt;
  }
}

void WeightOnlyGroupwiseQuantGemmPluginProfiler::runTactic(
    int m, int n, int k,
    WeightOnlyGroupwiseQuantGemmPluginProfiler::Config const& tactic, char* workspace, cudaStream_t const& stream) {
  int const originalN = n * (FP16_BITS / mQuantBits);
  const auto buffer_sizes = ComputeWeightOnlyGemmProfilerBufferSizes(
      m, n, k, mQuantBits, mGroupSize, 0, canProfileInt4Decode(m) ? mL2CacheBytes : 0);
  ORT_ENFORCE(buffer_sizes.has_value(), "Failed to compute fpA_intB tactic-profiler buffer sizes.");
  const auto& sizes = *buffer_sizes;
  half* actPtr = reinterpret_cast<half*>(workspace);
  void* weightPtr = nextWorkspacePtr(reinterpret_cast<int8_t*>(actPtr), sizes[0]);
  half* inputScalesPtr = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(weightPtr), sizes[1]));
  half* zerosPtr = reinterpret_cast<half*>(
      nextWorkspacePtr(reinterpret_cast<int8_t*>(inputScalesPtr), sizes[2]));
  half* biasesPtr = reinterpret_cast<half*>(
      nextWorkspacePtr(reinterpret_cast<int8_t*>(zerosPtr), sizes[3]));
  half* outputPtr = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(biasesPtr), sizes[4]));
  char* workspacePtr = reinterpret_cast<char*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(outputPtr), sizes[5]));

  if (canProfileInt4Decode(m)) {
    const size_t weight_bytes = static_cast<size_t>(k) * n * sizeof(half);
    const size_t scale_bytes = static_cast<size_t>(k) * originalN * sizeof(half) / mGroupSize;
    const size_t copies = sizes[1] / weight_bytes;
    const size_t index = mProfileWeightIndex.fetch_add(1, std::memory_order_relaxed) % copies;
    weightPtr = static_cast<char*>(weightPtr) + index * weight_bytes;
    inputScalesPtr = reinterpret_cast<half*>(reinterpret_cast<char*>(inputScalesPtr) + index * scale_bytes);
  }

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
    params.decode_variant = tactic.cudaKernelVariant >= 2 ? tactic.cudaKernelVariant : 0;
    params.paired_k = tactic.cudaKernelVariant == 1;
    params.wave_aware = mWaveAwareGemv;
    onnxruntime::llm::kernels::fpA_intB_gemv::kernel_launcher(mKernelArch, params, stream);
  } else {
    // run CUTLASS kernel
    int const wsSize = static_cast<int>(mRunner->getWorkspaceSize(m, originalN, k));
    if (mQuantBits == INT8_BITS) {
      mRunner->gemm(actPtr, reinterpret_cast<int8_t*>(weightPtr), inputScalesPtr, zerosPtr, biasesPtr, outputPtr,
                    m, originalN, k, mGroupSize, tactic, workspacePtr, wsSize, stream);
    } else if (mQuantBits == INT2_BITS) {
      mRunner->gemm(actPtr, reinterpret_cast<cutlass::uint2b_t*>(weightPtr), inputScalesPtr, zerosPtr, biasesPtr,
                    outputPtr, m, originalN, k, mGroupSize, tactic, workspacePtr, wsSize, stream);
    } else {
      mRunner->gemm(actPtr, reinterpret_cast<cutlass::uint4b_t*>(weightPtr), inputScalesPtr, zerosPtr, biasesPtr,
                    outputPtr, m, originalN, k, mGroupSize, tactic, workspacePtr, wsSize, stream);
    }
  }
}

size_t WeightOnlyGroupwiseQuantGemmPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k) {
  maxM = std::max<size_t>(1, maxM);
  const int original_n = static_cast<int>(n * (FP16_BITS / mQuantBits));
  const auto scratch_size = ComputeWeightOnlyGemmProfilerScratchSize(
      maxM, n, k, mQuantBits, mGroupSize,
      mRunner->getWorkspaceSize(static_cast<int>(maxM), original_n, static_cast<int>(k)),
      canProfileInt4Decode(static_cast<int>(maxM)) ? mL2CacheBytes : 0);
  ORT_ENFORCE(scratch_size.has_value(), "Failed to compute fpA_intB tactic-profiler scratch size.");
  return *scratch_size;
}

void WeightOnlyGroupwiseQuantGemmPluginProfiler::initTmpData(
    int m, int /*n*/, int /*k*/, char* workspace, size_t size, cudaStream_t stream) {
  if (canProfileInt4Decode(m)) {
    CUDA_CALL_THROW(cudaMemsetAsync(workspace, 0, size, stream));
    mProfileWeightIndex.store(0, std::memory_order_relaxed);
  }
}

std::vector<WeightOnlyGroupwiseQuantGemmPluginProfiler::Config> WeightOnlyGroupwiseQuantGemmPluginProfiler::getTactics(
    int m, int n, int k) const {
  auto tactics = mRunner->getConfigs();
  if (canProfileInt4Decode(m)) {
    const auto default_gemv = std::find_if(tactics.begin(), tactics.end(), [](const auto& tactic) {
      return tactic.enableCudaKernel && tactic.cudaKernelVariant == 0;
    });
    if (default_gemv != tactics.end()) {
      const auto base = *default_gemv;
      const int original_n = SafeInt<int>(n) * (FP16_BITS / mQuantBits);
      for (int variant = 2; variant <= 7; ++variant) {
        if (fpA_intB_gemv::IsInt4DecodeGeometryLegal(variant, original_n, k, mDecodeInterleave)) {
          auto decode = base;
          decode.cudaKernelVariant = variant;
          tactics.push_back(decode);
        }
      }
    }
  }
  if (mPairedGemvMode != 0 && m >= 5 && m <= 8) {
    for (auto const& tactic : tactics) {
      if (tactic.enableCudaKernel) {
        auto paired = tactic;
        paired.cudaKernelVariant = 1;
        if (mPairedGemvMode == 2) {
          return {paired};
        }
        tactics.push_back(paired);
        break;
      }
    }
  }
  return tactics;
}

bool WeightOnlyGroupwiseQuantGemmPluginProfiler::checkTactic(int m, int n, int k, Config const& tactic) const {
  // stop to profile Cuda kernel for m >= 16
  if (tactic.enableCudaKernel) {
    if (tactic.cudaKernelVariant >= 2) {
      return canProfileInt4Decode(m) &&
             fpA_intB_gemv::IsInt4DecodeGeometryLegal(tactic.cudaKernelVariant,
                                                      SafeInt<int>(n) * (FP16_BITS / mQuantBits), k, mDecodeInterleave);
    }
    return m < 16 && (tactic.cudaKernelVariant == 0 ||
                      (tactic.cudaKernelVariant == 1 && m >= 5 && m <= 8));
  }
  return true;
}

std::optional<WeightOnlyGroupwiseQuantGemmPluginProfiler::Config>
WeightOnlyGroupwiseQuantGemmPluginProfiler::getDeterministicConfig(int m) const {
  const auto configs = mRunner->getConfigs();
  if (m < 16) {
    for (const auto& config : configs) {
      if (config.enableCudaKernel) {
        return config;
      }
    }
  }
  for (const auto& config : configs) {
    if (!config.enableCudaKernel && config.split_k_style == cutlass_extensions::SplitKStyle::NO_SPLIT_K) {
      return config;
    }
  }
  return std::nullopt;
}

float GetWeightOnlyGemmSelectionTime(int m, size_t weight_bytes, size_t l2_cache_bytes,
                                     bool is_cuda_kernel, float time) {
  // The profiler replays one synthetic weight matrix back to back, so a matrix that fits in L2 is
  // timed L2-resident. In decode every weight matrix streams from DRAM once per step, where the CUDA
  // GEMV (a pure weight stream) keeps its measured speed but the CUTLASS kernels lose much of their
  // L2 advantage. A matrix larger than L2 is already timed from DRAM, so no bias is applied.
  constexpr float kCutlassPenaltyWhenGemvEligible = 1.1f;
  if (!is_cuda_kernel && m < 16 && weight_bytes <= l2_cache_bytes) {
    return time * kCutlassPenaltyWhenGemvEligible;
  }
  return time;
}

float WeightOnlyGroupwiseQuantGemmPluginProfiler::getSelectionTime(int m, int n, int k, Config const& tactic,
                                                                   float time) const {
  if (canProfileInt4Decode(m) && mL2CacheBytes != 0) {
    return time;
  }
  // n counts the 16-bit elements of one packed weight row (see runTactic).
  const size_t weight_bytes = SafeInt<size_t>(n) * k * sizeof(half);
  return GetWeightOnlyGemmSelectionTime(m, weight_bytes, mL2CacheBytes, tactic.enableCudaKernel, time);
}

std::vector<int> WeightOnlyGroupwiseQuantGemmPluginProfiler::ParseProfileMList(const std::string& value) {
  std::vector<int> result;
  if (value.empty()) {
    return result;
  }
  std::set<int> unique;
  for (const auto token : onnxruntime::utils::SplitString(value, ",", true)) {
    const std::string trimmed_token = onnxruntime::utils::TrimString(token);
    if (trimmed_token.empty()) {
      continue;
    }
    int m = 0;
    if (TryParseStringWithClassicLocale(trimmed_token, m) && m > 0) {
      unique.insert(m);
    }
  }
  result.assign(unique.begin(), unique.end());
  return result;
}

std::vector<int> WeightOnlyGroupwiseQuantGemmPluginProfiler::getProfileMBuckets(
    int minM, int maxM, bool /*hasWeightOnlyCudaKernel*/) const {
  return GetInitialProfileMBuckets(minM, maxM, mProfileMOverride);
}

std::vector<int> WeightOnlyGroupwiseQuantGemmPluginProfiler::GetInitialProfileMBuckets(
    int min_m, int max_m, const std::vector<int>& profile_m_override) {
  int const lo = std::max(1, min_m);
  int const hi = std::max(lo, max_m);

  std::set<int> buckets;

  if (!profile_m_override.empty()) {
    for (int m : profile_m_override) {
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

}  // namespace onnxruntime::llm::kernels::weight_only
#endif
