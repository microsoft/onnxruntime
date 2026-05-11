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

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <sstream>
#include <unordered_map>
#include <vector>

#include "contrib_ops/cuda/llm/nv_infer_datatype.h"
#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "contrib_ops/cuda/llm/common/logger.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/gemm_configs.h"

namespace onnxruntime::llm::kernels::weight_only {

struct GemmDims {
  int64_t minM;
  int64_t maxM;
  int64_t n;
  int64_t k;

  GemmDims()
      : minM(-1), maxM(-1), n(-1), k(-1) {
  }

  GemmDims(int64_t minM_, int64_t maxM_, int64_t n_, int64_t k_)
      : minM(minM_), maxM(maxM_), n(n_), k(k_) {
  }

  [[nodiscard]] bool isInitialized() const {
    return minM >= 0 && maxM >= 0 && n >= 0 && k >= 0;
  }
};

// Unique ID of GEMM
// In our case GEMM is uniqly identified by N and K
class GemmIdCore {
 public:
  int n;
  int k;
  nvinfer::DataType dtype;

  GemmIdCore(int n_, int k_, nvinfer::DataType const& dtype_)
      : n(n_), k(k_), dtype(dtype_) {
  }

  GemmIdCore()
      : n(-1), k(-1), dtype(nvinfer::DataType::kFLOAT)  // dtype does not matter here
  {
  }

  bool operator==(GemmIdCore const& id) const {
    return isEqual(id);
  }

  friend std::ostream& operator<<(std::ostream& out, GemmIdCore const& id) {
    out << "(N;K)=(" << id.n << ";" << id.k << "),";
    out << " type=" << static_cast<int>(id.dtype);
    return out;
  }

 protected:
  bool isEqual(GemmIdCore const& id) const {
    return n == id.n && k == id.k && dtype == id.dtype;
  }
};

// Hash of GemmId
struct GemmIdCoreHash {
  std::size_t operator()(GemmIdCore const& id) const {
    auto h1 = std::hash<int>{}(id.n);
    auto h2 = std::hash<int>{}(id.k);
    auto h3 = std::hash<int>{}(static_cast<int>(id.dtype));
    return h1 ^ h2 ^ h3;
  }
};

// class GemmIdCublas : public GemmIdCore {
//  public:
//   bool transA{};
//   bool transB{};
//   nvinfer::DataType outputDtype;

//   GemmIdCublas(int n_, int k_, nvinfer::DataType const& dtype_, bool transA_, bool transB_,
//                nvinfer::DataType const& output_dtype_)
//       : GemmIdCore(n_, k_, dtype_), transA(transA_), transB(transB_), outputDtype(output_dtype_) {
//   }

//   GemmIdCublas() {}

//   bool operator==(GemmIdCublas const& id) const {
//     return isEqual(id) && transA == id.transA && transB == id.transB && outputDtype == id.outputDtype;
//   }

//   friend std::ostream& operator<<(std::ostream& out, GemmIdCublas const& id) {
//     out << "(N;K)=(" << id.n << ";" << id.k << "),";
//     out << " type=" << static_cast<int>(id.dtype);
//     out << " transA=" << id.transA;
//     out << " transB=" << id.transB;
//     out << " outputDtype=" << static_cast<int>(id.outputDtype);
//     return out;
//   }
// };

// // Hash of GemmIdCublas
// struct GemmIdCublasHash {
//   std::size_t operator()(GemmIdCublas const& id) const {
//     auto h1 = std::hash<int>{}(id.n);
//     auto h2 = std::hash<int>{}(id.k);
//     auto h3 = std::hash<int>{}(static_cast<int>(id.dtype));
//     auto h4 = std::hash<bool>{}(id.transA);
//     auto h5 = std::hash<bool>{}(id.transB);
//     auto h6 = std::hash<bool>{}(static_cast<int>(id.outputDtype));
//     return h1 ^ h2 ^ h3 ^ h4 ^ h5 ^ h6;
//   }
// };

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
class GemmPluginProfiler {
 public:
  // Map for single GEMM for different Ms (GEMM dimension) to the best config for particular M
  using MProfileMap = std::unordered_map<int, std::optional<Config>>;
  using MProfileMapPtr = std::shared_ptr<MProfileMap>;

  // requires exclusive ownership to write to *this
  using reader_lock = std::unique_lock<std::shared_timed_mutex>;
  // requires shared ownership to read from other
  using writer_lock = std::shared_lock<std::shared_timed_mutex>;

  // Struct of continuing map if GEMMs to the best profiles for different Ms
  struct MNKProfileMap {
    // Mutex guarding map
    std::shared_timed_mutex mutex;
    // Map from GEMM Id to profile for particular GEMM
    std::unordered_map<GemmIdType, MProfileMapPtr, GemmIdHashType> profileMap;

    bool existsMProfileMap(GemmIdType const& id) {
      auto const iter = profileMap.find(id);
      return iter != profileMap.end();
    }

    void createMProfileMap(GemmIdType const& id) {
      profileMap[id] = std::make_shared<MProfileMap>();
    }

    MProfileMapPtr getMProfileMap(GemmIdType const& id) {
      auto const iter = profileMap.find(id);
      if (iter == profileMap.end()) {
        ORT_THROW("Cannot find ID (", id, ") in the profile map. Abort.");
      }
      return iter->second;
    }
  };

  using MNKProfileMapPtr = std::shared_ptr<MNKProfileMap>;

  GemmPluginProfiler();

  virtual ~GemmPluginProfiler() = default;

  // void serialize(char*& buffer, GemmIdType const& gemmId) const;

  // void deserialize(char const*& data, GemmDims& dims, GemmIdType const& gemmId);
  // size_t getSerializationSize(GemmIdType const& gemmId) const;

  void profileTactics(RunnerPtr const& runner, nvinfer::DataType const& type, GemmDims const& dims,
                      GemmIdType const& gemmId, bool hasWeightOnlyCudaKernel = false);

  void setSelectionTactics(MNKProfileMapPtr const& map) {
    mMNKProfileMap = map;
  }

  void setSkip(bool skip) {
    mSkip = mSkip || skip;
  }

  void setAllocator(onnxruntime::AllocatorPtr allocator) {
    mAllocator = std::move(allocator);
  }

  std::optional<Config> getBestConfig(int m, GemmIdType const& gemmId) const;

  virtual int getMaxProfileM() const;

 protected:
  virtual void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) = 0;

  virtual size_t computeTmpSize(size_t maxM, size_t n, size_t k) = 0;

  virtual bool checkTactic(int /*m*/, int /*n*/, int /*k*/, Config const& /*tactic*/) const {
    return true;
  }

  virtual std::vector<Config> getTactics(int m, int n, int k) const = 0;

  virtual void initTmpData(int m, int n, int k, char* workspace, size_t size, cudaStream_t stream);

 private:
  std::optional<Config> profileTacticsForProblem(int m, int n, int k, std::vector<Config> const& tactics);

  float profileTacticForProblem(int m, int n, int k, Config const& tactic);

  int nextPowerOfTwo(int v) const {
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    return ++v;
  }

 protected:
  RunnerPtr mRunner{nullptr};

  nvinfer::DataType mType{};

 private:
  MNKProfileMapPtr mMNKProfileMap{};

  onnxruntime::IAllocatorUniquePtr<char> mWorkspaceTmp{nullptr};

  cudaStream_t mStream;

  GemmDims mDims{};

  bool mSkip{false};

  onnxruntime::AllocatorPtr mAllocator;
};

template <typename GemmPluginProfilerType>
class GemmPluginProfilerManager {
 public:
  using MNKProfileMap = typename GemmPluginProfilerType::MNKProfileMap;
  using MNKProfileMapPtr = typename GemmPluginProfilerType::MNKProfileMapPtr;
  using GemmPluginProfilerPtr = std::shared_ptr<GemmPluginProfilerType>;

  GemmPluginProfilerManager() {
    mMNKProfileMap = std::make_shared<MNKProfileMap>();
  }

  GemmPluginProfilerPtr createGemmPluginProfiler(bool inference, bool skip = false) {
    auto profiler = std::make_shared<GemmPluginProfilerType>();
    profiler->setSkip(skip);
    // If the profiler is created during the engine build,
    // mMNKProfileMap is shared between different profilers to minimize the time spent on the profiling
    // and do not repeat profiling for the GEMMs of the same shape.
    if (!inference) {
      profiler->setSelectionTactics(mMNKProfileMap);
    }
    return profiler;
  }

 private:
  MNKProfileMapPtr mMNKProfileMap{};
};

}  // namespace onnxruntime::llm::kernels::weight_only

namespace onnxruntime::llm::kernels::weight_only {

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::GemmPluginProfiler() {
  mMNKProfileMap = std::make_shared<MNKProfileMap>();
}

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
  ORT_LLM_LOG_ENTRY();
  writer_lock lock(mMNKProfileMap->mutex);

  if (!dims.isInitialized()) {
    return;
  }

  mRunner = runner;
  mType = type;

  int const maxM = std::min(nextPowerOfTwo(dims.maxM), getMaxProfileM());

  size_t workspace_bytes = computeTmpSize(maxM, dims.n, dims.k);

  if (!mMNKProfileMap->existsMProfileMap(gemmId)) {
    // Create map for GEMM ID
    mMNKProfileMap->createMProfileMap(gemmId);
  }

  if (mSkip) {
    return;
  }

  auto mProfileMap = mMNKProfileMap->getMProfileMap(gemmId);
  bool isAllocated{false};

  auto profileTactics = [&](int m, int n, int k) {
    if (mProfileMap->count(m) == 0) {
      if (!isAllocated) {
        this->mWorkspaceTmp = onnxruntime::IAllocator::MakeUniquePtr<char>(mAllocator, workspace_bytes, true);
#if ORT_LLM_VERBOSE
        AllocatorStats stats;
        this->mAllocator->GetStats(&stats);
        std::cout << "Allocator state after " << workspace_bytes << " bytes gemm profiler workspace:" << std::endl
                  << stats.DebugString() << std::endl;
#endif
        isAllocated = true;
      }

      initTmpData(m, n, k, this->mWorkspaceTmp.get(), workspace_bytes, this->mStream);

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
    mWorkspaceTmp.reset();
  }
  CUDA_CALL_THROW(cudaStreamDestroy(mStream));
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
std::optional<Config> GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::getBestConfig(
    int m, GemmIdType const& gemmId) const {
  ORT_LLM_LOG_ENTRY();
  reader_lock lock(mMNKProfileMap->mutex);

  if (mSkip) {
    ORT_LLM_LOG_DEBUG("Skip is set, no best config is set for this instance");
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
std::optional<Config> GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTacticsForProblem(
    int m, int n, int k, std::vector<Config> const& tactics) {
  ORT_LLM_LOG_ENTRY();

  float bestTime = std::numeric_limits<float>::max();
  Config bestConfig;
  bool foundOne = false;

#if ORT_LLM_VERBOSE > 1
  std::cout << "Total configs to profile:" << tactics.size() << std::endl;
#endif

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

#if ORT_LLM_VERBOSE > 1
      if constexpr (std::is_same_v<Config, onnxruntime::llm::cutlass_extensions::CutlassGemmConfig>) {
        std::cout << "Time=" << time << " for config: " << candidateConfig.toString() << std::endl;
      }
#endif

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
      ORT_LLM_LOG_DEBUG(msg.str());
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

#if ORT_LLM_VERBOSE > 1
  std::cout << "Best config:" << bestConfig.toString() << std::endl;
#endif

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
    runTactic(m, n, k, tactic, mWorkspaceTmp.get(), stream);
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CALL_THROW(cudaEventCreate(&start));
  CUDA_CALL_THROW(cudaEventCreate(&stop));
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  CUDA_CALL_THROW(cudaEventRecord(start, stream));

  // Profile GEMM
  for (int i = 0; i < runs; ++i) {
    runTactic(m, n, k, tactic, mWorkspaceTmp.get(), stream);
  }

  CUDA_CALL_THROW(cudaEventRecord(stop, stream));

  CUDA_CALL_THROW(cudaEventSynchronize(stop));

  float elapsed;
  CUDA_CALL_THROW(cudaEventElapsedTime(&elapsed, start, stop));

  CUDA_CALL_THROW(cudaEventDestroy(start));
  CUDA_CALL_THROW(cudaEventDestroy(stop));

  return elapsed / runs;
}

}  // namespace onnxruntime::llm::kernels::weight_only
