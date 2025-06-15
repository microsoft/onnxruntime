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
#include "contrib_ops/cuda/llm/fpA_intB_gemm_profiler.h"
#include "contrib_ops/cuda/llm/common/workspace.h"
#include "contrib_ops/cuda/llm/common/logger.h"

using namespace onnxruntime::llm::common;
using namespace onnxruntime::llm::kernels::cutlass_kernels;

namespace onnxruntime::llm::kernels::weight_only {

void WeightOnlyGroupwiseQuantGemmPluginProfiler::runTactic(
    int m, int n, int k,
    WeightOnlyGroupwiseQuantGemmPluginProfiler::Config const& tactic, char* workspace, cudaStream_t const& stream) {
  ORT_LLM_LOG_ENTRY();
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
    int const wsSize = mRunner->getWorkspaceSize(m, originalN, k);
    if (mQuantBits == 8) {
      mRunner->gemm(actPtr, reinterpret_cast<int8_t*>(weightPtr), inputScalesPtr, zerosPtr, biasesPtr, outputPtr,
                    m, originalN, k, mGroupSize, tactic, workspacePtr, wsSize, stream);
    } else {
      mRunner->gemm(actPtr, reinterpret_cast<cutlass::uint4b_t*>(weightPtr), inputScalesPtr, zerosPtr, biasesPtr,
                    outputPtr, m, originalN, k, mGroupSize, tactic, workspacePtr, wsSize, stream);
    }
  }
  ORT_LLM_LOG_EXIT();
}

size_t WeightOnlyGroupwiseQuantGemmPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k) {
  // Quantized weights are packed in FP16 format (INT4*4 -> FP16, INT8*2 -> FP16)
  int const originalN = mQuantBits == 8 ? n * FP16_INT8_RATIO : n * FP16_INT4_RATIO;
  std::vector<size_t> workspaces = {
      maxM * k * sizeof(half),                       // A
      k * n * sizeof(half),                          // B
      k * originalN * sizeof(half) / mGroupSize,     // scales
      k * originalN * sizeof(half) / mGroupSize,     // zeros
      originalN * sizeof(half),                      // biases
      maxM * originalN * sizeof(half),               // C
      mRunner->getWorkspaceSize(maxM, originalN, k)  // workspace
  };
  size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
  return bytes;
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

}  // namespace onnxruntime::llm::kernels::weight_only
