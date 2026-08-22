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

std::optional<size_t> ComputeWeightOnlyGemmProfilerScratchSize(
    size_t max_m, size_t packed_n, size_t k, int quant_bits,
    size_t group_size, size_t runner_workspace_bytes) {
  if (max_m == 0 || packed_n == 0 || k == 0 ||
      (quant_bits != INT4_BITS && quant_bits != INT8_BITS) || group_size == 0) {
    return std::nullopt;
  }

  try {
    const SafeInt<size_t> original_n =
        SafeInt<size_t>(packed_n) * (FP16_BITS / quant_bits);
    constexpr size_t kElementBytes = sizeof(uint16_t);
    const size_t workspaces[] = {
        static_cast<size_t>(SafeInt<size_t>(max_m) * k * kElementBytes),
        static_cast<size_t>(SafeInt<size_t>(k) * packed_n * kElementBytes),
        static_cast<size_t>(SafeInt<size_t>(k) * original_n * kElementBytes / group_size),
        static_cast<size_t>(SafeInt<size_t>(k) * original_n * kElementBytes / group_size),
        static_cast<size_t>(original_n * kElementBytes),
        static_cast<size_t>(SafeInt<size_t>(max_m) * original_n * kElementBytes),
        runner_workspace_bytes};

    SafeInt<size_t> total = 0;
    for (const size_t workspace : workspaces) {
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
    int const wsSize = static_cast<int>(mRunner->getWorkspaceSize(m, originalN, k));
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
  const int original_n =
      static_cast<int>(mQuantBits == 8 ? n * FP16_INT8_RATIO : n * FP16_INT4_RATIO);
  const auto scratch_size = ComputeWeightOnlyGemmProfilerScratchSize(
      maxM, n, k, mQuantBits, mGroupSize,
      mRunner->getWorkspaceSize(static_cast<int>(maxM), original_n, static_cast<int>(k)));
  ORT_ENFORCE(scratch_size.has_value(), "Failed to compute fpA_intB tactic-profiler scratch size.");
  return *scratch_size;
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
  int const lo = std::max(1, minM);
  int const hi = std::max(lo, maxM);

  std::set<int> buckets;

  if (!mProfileMOverride.empty()) {
    for (int m : mProfileMOverride) {
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
