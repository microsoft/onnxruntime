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

#include "contrib_ops/cuda/llm/gemm_profiler.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm/fpA_intB_gemm.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/fpA_intB_gemv.h"

#include <cassert>
#include <cutlass/numeric_types.h>
#include <memory>
#include <set>
#include <string>
#include <vector>

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
constexpr int32_t FP16_INT4_RATIO = FP16_BITS / INT4_BITS;
constexpr int32_t FP16_INT8_RATIO = FP16_BITS / INT8_BITS;

class WeightOnlyGroupwiseQuantGemmPluginProfiler
    : public GemmPluginProfiler<onnxruntime::llm::cutlass_extensions::CutlassGemmConfig, WeightOnlyGemmRunnerPtr,
                                GemmIdCore, GemmIdCoreHash> {
 public:
  using Config = onnxruntime::llm::cutlass_extensions::CutlassGemmConfig;

  void setQuant(int bits, bool has_bias, bool has_zeros) {
    mQuantBits = bits;
    mHasBiases = has_bias;
    mHasZeros = has_zeros;
  }

  void setGroupSize(int groupSize) {
    mGroupSize = groupSize;
  }

  void setCudaKernelType(KernelType cudaKernelType, int arch) {
    mCudaKernelType = cudaKernelType;
    mArch = arch;
  }

 protected:
  void runTactic(int m, int n, int k, Config const& tactic,
                 char* workspace, cudaStream_t const& stream) override;

  void computeTmpSize(size_t maxM, size_t n, size_t k) override;

  std::vector<Config> getTactics(int m, int n, int k) const override;

  bool checkTactic(int m, int n, int k, Config const& tactic) const override;

 private:
  bool mHasBiases;
  bool mHasZeros;
  int mQuantBits;
  int mGroupSize;
  KernelType mCudaKernelType;
  int mArch;
};

}  // namespace onnxruntime::llm::kernels::weight_only
