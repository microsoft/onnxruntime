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
#pragma once

#include <array>
#include <atomic>
#include <cassert>
#include <cutlass/numeric_types.h>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "contrib_ops/cuda/llm/gemm_profiler.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm/fpA_intB_gemm.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/fpA_intB_gemv.h"

using WeightOnlyGemmRunner = onnxruntime::llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface;
using WeightOnlyGemmRunnerPtr = std::shared_ptr<WeightOnlyGemmRunner>;
using KernelType = onnxruntime::llm::kernels::fpA_intB_gemv::KernelType;

namespace onnxruntime::llm::kernels::weight_only {
enum class WeightTypeId {
  INT8 = 1,
  INT4 = 2,
};

constexpr int32_t FP16_BITS = 16;
constexpr int32_t INT8_BITS = 8;
constexpr int32_t INT4_BITS = 4;
constexpr int32_t INT2_BITS = 2;

// Comma-separated list of M buckets to profile for MatMulNBits/fpA_intB. Overrides the default
// reduced bucket set. Example: ORT_FPA_INTB_PROFILE_M="1,8,64,512".
constexpr const char* kEnvProfileM = "ORT_FPA_INTB_PROFILE_M";

// Default top M that bounds the initial profile sweep when no override is given. Larger runtime
// M values are handled by lazy single-bucket profiling.
constexpr int kDefaultProfileMaxM = 2048;

// Computes the single temporary CUDA allocation used while profiling tactics.
// `packed_n` is the number of 16-bit elements that hold one packed weight row.
// This pure-math helper is shared by the runtime profiler and partition-time
// memory estimation so their allocation formulas cannot drift.
// Maps a measured tactic time to the value compared during tactic selection. For small M, CUTLASS
// must beat the CUDA GEMV by 10% when the weight fits in L2, because the profiler then times it
// L2-resident while decode streams it from DRAM.
float GetWeightOnlyGemmSelectionTime(int m, size_t weight_bytes, size_t l2_cache_bytes,
                                     bool is_cuda_kernel, float time);

std::optional<std::array<size_t, 7>> ComputeWeightOnlyGemmProfilerBufferSizes(
    size_t max_m, size_t packed_n, size_t k, int quant_bits,
    size_t group_size, size_t runner_workspace_bytes, size_t streaming_l2_bytes = 0);

std::optional<size_t> ComputeWeightOnlyGemmProfilerScratchSize(
    size_t max_m, size_t packed_n, size_t k, int quant_bits,
    size_t group_size, size_t runner_workspace_bytes, size_t streaming_l2_bytes = 0);

class WeightOnlyGroupwiseQuantGemmPluginProfiler
    : public GemmPluginProfiler<onnxruntime::llm::cutlass_extensions::CutlassGemmConfig, WeightOnlyGemmRunnerPtr,
                                GemmIdCore, GemmIdCoreHash> {
 public:
  using Config = onnxruntime::llm::cutlass_extensions::CutlassGemmConfig;

  // Parses a comma-separated list of M buckets (e.g. "1,8,64,512") into a sorted, de-duplicated,
  // positive list (empty when the string is empty/blank). Used for the ep.cuda.fpa_intb_profile_m
  // session-config key and the ORT_FPA_INTB_PROFILE_M env var, both resolved by the kernel.
  static std::vector<int> ParseProfileMList(const std::string& value);

  // Returns the exact initial bucket set used by construction-time profiling.
  static std::vector<int> GetInitialProfileMBuckets(
      int min_m, int max_m, const std::vector<int>& profile_m_override);

  // Overrides the initial profile M-bucket set for this profiler instance (per session). An empty
  // list keeps the built-in default bucket set. Resolved by the kernel from session config / env.
  void setProfileMOverride(std::vector<int> ms) {
    mProfileMOverride = std::move(ms);
  }

  void setQuant(int bits, bool has_bias, bool has_zeros) {
    mQuantBits = bits;
    mHasBiases = has_bias;
    mHasZeros = has_zeros;
  }

  void setGroupSize(int groupSize) {
    mGroupSize = groupSize;
  }

  void setDecodeInterleave(int interleave) {
    mDecodeInterleave = interleave;
  }

  void setCudaKernelType(KernelType cudaKernelType, int arch) {
    mCudaKernelType = cudaKernelType;
    mArch = arch;
  }

  void setL2CacheBytes(size_t l2CacheBytes) {
    mL2CacheBytes = l2CacheBytes;
  }

  // Paired-K fp16 int4 GEMV tactic for the M = 8 bucket (M = 5..8 at run time). Mode 1 adds it as an extra
  // candidate, so it is kept only where it beats the default GEMV and the CUTLASS kernels; mode 2 offers
  // only that tactic (testing and benchmarking); mode 0 disables it.
  void setPairedGemvMode(int mode) {
    mPairedGemvMode = mode;
  }

  void setWaveAwareGemv(bool enabled) {
    mWaveAwareGemv = enabled;
  }

 protected:
  bool canProfileInt4Decode(int m) const {
    return m == 1 && mDecodeInterleave == 4 && mQuantBits == INT4_BITS && mGroupSize == 32 &&
           !mHasBiases && !mHasZeros &&
           (mCudaKernelType == KernelType::FP16Int4Groupwise ||
            mCudaKernelType == KernelType::BF16Int4Groupwise);
  }

  void runTactic(int m, int n, int k, Config const& tactic,
                 char* workspace, cudaStream_t const& stream) override;

  size_t computeTmpSize(size_t maxM, size_t n, size_t k) override;

  void initTmpData(int m, int n, int k, char* workspace, size_t size, cudaStream_t stream) override;

  std::vector<Config> getTactics(int m, int n, int k) const override;

  bool checkTactic(int m, int n, int k, Config const& tactic) const override;

  float getSelectionTime(int m, int n, int k, Config const& tactic, float time) const override;

  std::vector<int> getProfileMBuckets(int minM, int maxM, bool hasWeightOnlyCudaKernel) const override;

 private:
  bool mHasBiases = false;
  bool mHasZeros = false;
  int mQuantBits = 0;
  int mGroupSize = 0;
  KernelType mCudaKernelType = KernelType::FP16Int4Groupwise;
  int mArch = 0;
  int mDecodeInterleave = 0;
  size_t mL2CacheBytes = 0;
  std::atomic<size_t> mProfileWeightIndex{0};
  int mPairedGemvMode = 0;
  bool mWaveAwareGemv = false;
  std::vector<int> mProfileMOverride;
};

}  // namespace onnxruntime::llm::kernels::weight_only
