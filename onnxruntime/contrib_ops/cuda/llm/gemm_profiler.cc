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

#include "contrib_ops/cuda/llm/gemm_profiler.h"
#include "contrib_ops/cuda/llm/common/logger.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm/fpA_intB_gemm.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

#include <cstddef>

namespace onnxruntime::llm::kernels::weight_only {

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::GemmPluginProfiler() {
  mMNKProfileMap = std::make_shared<MNKProfileMap>();

  // set SKIP_GEMM_PLUGIN_PROFILINGS=1 to avoid tactics profilings
  auto const skipEnv = std::getenv("SKIP_GEMM_PLUGIN_PROFILINGS");
  mSkip = (skipEnv != NULL && std::stoi(skipEnv));
  if (mSkip) {
    ORT_LLM_LOG_DEBUG(
        "SKIP_GEMM_PLUGIN_PROFILINGS is set. Skipping GEMM plugin profilings. It could result in runtime error "
        "if default tactic is not defined.");
  }
}

// template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
// void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::serialize(
//     char*& buffer, GemmIdType const& gemmId) const
// {
//     auto mProfileMap = mMNKProfileMap->getMProfileMap(gemmId);

//     // Save number of profiles for given GEMM ID
//     write(buffer, static_cast<int>(mProfileMap->size()));
//     for (auto const& pair : *mProfileMap)
//     {
//         // Save pair of M to the best GEMM config
//         write(buffer, pair);
//     }
// }

// template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
// void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::deserialize(
//     char const*& data, GemmDims& dims, GemmIdType const& gemmId)
// {
//     // NOTE: this mutex is not needed since each thread owns its private map, but will put here for
//     // consistency
//     writer_lock lock(mMNKProfileMap->mutex);

//     mDims = dims;

//     // GemmId gemmId(dims.n, dims.k);
//     if (!mMNKProfileMap->existsMProfileMap(gemmId))
//     {
//         // Create GEMM with GEMM ID if it does not exist
//         mMNKProfileMap->createMProfileMap(gemmId);
//     }
//     // Populate map with profiles of GEMM ID
//     auto profileMap = mMNKProfileMap->getMProfileMap(gemmId);
//     int selectedMapSize;
//     read(data, selectedMapSize);
//     for (int ii = 0; ii < selectedMapSize; ++ii)
//     {
//         std::pair<int, std::optional<Config>> config;
//         read(data, config);
//         profileMap->insert(config);
//     }
// }

// template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
// size_t GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::getSerializationSize(
//     GemmIdType const& gemmId) const
// {
//     reader_lock lock(mMNKProfileMap->mutex);
//     return sizeof(int) +                                 // size of the tactics map
//         mMNKProfileMap->getMProfileMap(gemmId)->size()
//         * sizeof(std::pair<int, std::optional<Config>>); // size of the tactics map
// }

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
int GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::getMaxProfileM() const {
  return 8192;
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::initTmpData(
    int /*m*/, int /*n*/, int /*k*/, char* /*workspace*/, size_t /*size*/, cudaStream_t /*stream*/) {
  /* Do nothing */
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTactics(
    RunnerPtr const& runner, nvinfer::DataType const& type, GemmDims const& dims, GemmIdType const& gemmId,
    bool hasWeightOnlyCudaKernel) {
  writer_lock lock(mMNKProfileMap->mutex);

  if (!dims.isInitialized()) {
    return;
  }

  mRunner = runner;
  mType = type;

  int const maxM = std::min(nextPowerOfTwo(dims.maxM), getMaxProfileM());
  computeTmpSize(maxM, dims.n, dims.k);

  if (!mMNKProfileMap->existsMProfileMap(gemmId)) {
    // Create map for GEMM ID
    mMNKProfileMap->createMProfileMap(gemmId);
  }

  if (mSkip) {
    return;
  }

  auto mProfileMap = mMNKProfileMap->getMProfileMap(gemmId);
  bool isAllocated{false};

  auto profileTactics = [&mProfileMap, &isAllocated, this](int m, int n, int k) {
    if (mProfileMap->count(m) == 0) {
      if (!isAllocated) {
        // Allocate tmp data to run GEMMs
        allocateTmpData();
        isAllocated = true;
      }
      initTmpData(m, n, k, mWorkspaceTmp, mTmpWorkspaceSizeInBytes, mStream);
      auto tactics = this->getTactics(m, n, k);

      // Profile different tactics for particular m and insert best config to the map
      mProfileMap->insert({m, this->profileTacticsForProblem(m, n, k, tactics)});
    }
  };

  CUDA_CALL_THROW(cudaStreamCreate(&mStream));

  int const startMinMRounded = nextPowerOfTwo(dims.minM);

  if (hasWeightOnlyCudaKernel) {
    // Profile tactics for finer granularity of M,
    // if CUDA kernel is enabled for weight-only plugins
    int minM = dims.minM;
    for (int m = std::max(1, minM); m < std::min(16, maxM); m += 1) {
      profileTactics(m, dims.n, dims.k);
    }

    for (int m = 16; m < maxM; m *= 2) {
      profileTactics(m, dims.n, dims.k);
    }
  } else {
    // Profile tactics for CUTLASS kernel only
    for (int m = std::max(1, startMinMRounded); m < maxM; m *= 2) {
      profileTactics(m, dims.n, dims.k);
    }
  }

  profileTactics(maxM, dims.n, dims.k);

  if (isAllocated) {
    // Free tmp data
    freeTmpData();
  }
  CUDA_CALL_THROW(cudaStreamDestroy(mStream));
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
std::optional<Config> GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::getBestConfig(
    int m, GemmIdType const& gemmId) const {
  reader_lock lock(mMNKProfileMap->mutex);

  if (mSkip) {
    ORT_LLM_LOG_TRACE("Skip is set, no best config is set for this instance");
    return std::nullopt;
  }

  int const mRounded = std::min(std::max(1, nextPowerOfTwo(m)), getMaxProfileM());
  fflush(stdout);

  if (mMNKProfileMap->getMProfileMap(gemmId)->count(m) > 0) {
    return mMNKProfileMap->getMProfileMap(gemmId)->at(m);
  } else if (mMNKProfileMap->getMProfileMap(gemmId)->count(mRounded) > 0) {
    return mMNKProfileMap->getMProfileMap(gemmId)->at(mRounded);
  } else {
    std::ostringstream msg;
    msg << "Cannot find best tactic for m=" << m << " and GEMM ID " << gemmId;
    ORT_LLM_LOG_WARNING(msg.str());
    return std::nullopt;
  }
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::allocateTmpData() {
  ORT_ENFORCE(mTmpWorkspaceSizeInBytes > 0, "tmpWorkspaceSizeInBytes must be larger than 0");
  auto const status = cudaMalloc(&mWorkspaceTmp, mTmpWorkspaceSizeInBytes);
  ORT_ENFORCE(status == cudaSuccess, "Can't allocate tmp workspace for GEMM tactics profiling.");
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
void GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::freeTmpData() {
  auto const status = cudaFree(mWorkspaceTmp);
  ORT_ENFORCE(status == cudaSuccess, "Can't free tmp workspace for GEMM tactics profiling.");
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
std::optional<Config> GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTacticsForProblem(
    int m, int n, int k, std::vector<Config> const& tactics) {
  ORT_LLM_LOG_DEBUG(__PRETTY_FUNCTION__);

  float bestTime = std::numeric_limits<float>::max();
  Config bestConfig;
  bool foundOne = false;

  // Iterate over all tactics for given M, N and K
  for (size_t ii = 0; ii < tactics.size(); ++ii) {
    Config const& candidateConfig = tactics[ii];
    float time = std::numeric_limits<float>::max();
    try {
      if (!checkTactic(m, n, k, candidateConfig)) {
        continue;
      }
      // Profile particular tactic for given M, N and K
      time = profileTacticForProblem(m, n, k, candidateConfig);
      foundOne = true;
    } catch (std::exception const& e) {
      std::ostringstream msg;
      msg << "Cannot profile configuration " << ii;
      if constexpr (std::is_same_v<Config, onnxruntime::llm::cutlass_extensions::CutlassGemmConfig>) {
        msg << ": " << candidateConfig.toString();
      }
      msg << "\n (for"
          << " m=" << m << ", n=" << n << ", k=" << k << ")"
          << ", reason: \"" << e.what() << "\". Skipped";
      ORT_LLM_LOG_TRACE(msg.str());
      cudaGetLastError();  // Reset the last cudaError to cudaSuccess.
      continue;
    }

    // Choose the fastest tactic
    if (time < bestTime) {
      bestConfig = candidateConfig;
      bestTime = time;
    }
  }

  if (!foundOne) {
    std::ostringstream msg;
    msg << "Have not found any valid GEMM config for shape ("
        << "m=" << m << ", n=" << n << ", k=" << k << "). Will try to use default or fail at runtime";
    ORT_LLM_LOG_WARNING(msg.str());
    return std::nullopt;
  }

  return {bestConfig};
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
float GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTacticForProblem(
    int m, int n, int k, Config const& tactic) {
  constexpr int warmup = 5;
  constexpr int runs = 10;

  cudaStream_t stream = mStream;

  // Warmup the execution
  for (int i = 0; i < warmup; ++i) {
    runTactic(m, n, k, tactic, mWorkspaceTmp, stream);
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CALL_THROW(cudaEventCreate(&start));
  CUDA_CALL_THROW(cudaEventCreate(&stop));
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  CUDA_CALL_THROW(cudaEventRecord(start, stream));

  // Profile GEMM
  for (int i = 0; i < runs; ++i) {
    runTactic(m, n, k, tactic, mWorkspaceTmp, stream);
  }

  CUDA_CALL_THROW(cudaEventRecord(stop, stream));

  CUDA_CALL_THROW(cudaEventSynchronize(stop));

  float elapsed;
  CUDA_CALL_THROW(cudaEventElapsedTime(&elapsed, start, stop));

  CUDA_CALL_THROW(cudaEventDestroy(start));
  CUDA_CALL_THROW(cudaEventDestroy(stop));

  return elapsed / runs;
}

template class GemmPluginProfiler<onnxruntime::llm::cutlass_extensions::CutlassGemmConfig,
                                  std::shared_ptr<onnxruntime::llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunnerInterface>, GemmIdCore,
                                  GemmIdCoreHash>;

}  // namespace onnxruntime::llm::kernels::weight_only
