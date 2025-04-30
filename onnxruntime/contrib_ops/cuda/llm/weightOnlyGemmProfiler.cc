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
#include "contrib_ops/cuda/llm/weightOnlyGemmProfiler.h"
#include "contrib_ops/cuda/llm/common/workspace.h"

using namespace ort_llm::common;
using namespace ort_llm::kernels::cutlass_kernels;
//using ort_llm::plugins::read;
// using ort_llm::plugins::WeightOnlyQuantGemmPluginProfiler;
//using ort_llm::plugins::WeightOnlyQuantMatmulPlugin;
//using ort_llm::plugins::WeightOnlyQuantMatmulPluginCreator;
//using ort_llm::plugins::write;

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

namespace ort_llm::kernels::weight_only {

void WeightOnlyQuantGemmPluginProfiler::runTactic(int m, int n, int k,
                                                  WeightOnlyQuantGemmPluginProfiler::Config const& tactic,
                                                  char* workspace,
                                                  cudaStream_t const& stream) {
  int const originalN = n * getWeightTypeMultiplier(mWeightTypeId);
  half* actPtr = reinterpret_cast<half*>(workspace);
  int8_t* weightPtr = reinterpret_cast<int8_t*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(actPtr), m * k * sizeof(half)));
  half* scalesPtr = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(weightPtr), n * k * sizeof(int8_t)));
  half* outputPtr = reinterpret_cast<half*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(scalesPtr), originalN * sizeof(half)));
  char* workspacePtr = reinterpret_cast<char*>(nextWorkspacePtr(reinterpret_cast<int8_t*>(outputPtr), m * originalN * sizeof(half)));

  int const wsSize = mRunner->getWorkspaceSize(m, originalN, k);

  /*if (tactic.enableCudaKernel) {
    // run CUDA kernel
    ort_llm::kernels::weight_only::Params params{actPtr, nullptr, weightPtr, scalesPtr, nullptr, nullptr,
                                                 outputPtr, 1.f, m, originalN, k, 0, mCudaKernelType};
    ort_llm::kernels::weight_only::kernel_launcher(mArch, params, stream);
  } else */ {
    // run CUTLASS kernel
    if (mWeightTypeId == WeightTypeId::INT8) {
      mRunner->gemm(
          actPtr, weightPtr, scalesPtr, outputPtr, m, originalN, k, tactic, workspacePtr, wsSize, stream);
    } else {
      mRunner->gemm(actPtr, reinterpret_cast<cutlass::uint4b_t*>(weightPtr), scalesPtr, outputPtr, m, originalN,
                    k, tactic, workspacePtr, wsSize, stream);
    }
  }
}

void WeightOnlyQuantGemmPluginProfiler::computeTmpSize(size_t maxM, size_t n, size_t k) {
  int const originalN = n * getWeightTypeMultiplier(mWeightTypeId);
  std::vector<size_t> workspaces = {
      maxM * k * sizeof(half),                       // A
      n * k * sizeof(int8_t),                        // B
      originalN * sizeof(half),                      // scales
      maxM * originalN * sizeof(half),               // C
      mRunner->getWorkspaceSize(maxM, originalN, k)  // workspace
  };
  size_t bytes = calculateTotalWorkspaceSize(workspaces.data(), workspaces.size());
  setTmpWorkspaceSizeInBytes(bytes);
}

std::vector<WeightOnlyQuantGemmPluginProfiler::Config> WeightOnlyQuantGemmPluginProfiler::getTactics(
    int m, int n, int k) const {
  return mRunner->getConfigs();
}

bool WeightOnlyQuantGemmPluginProfiler::checkTactic(int m, int n, int k, Config const& tactic) const {
  // stop to profile Cuda kernel for m >= 16
  if (tactic.enableCudaKernel) {
    return m < 16;
  }
  return true;
}

}  // namespace ort_llm::kernels::weight_only

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif