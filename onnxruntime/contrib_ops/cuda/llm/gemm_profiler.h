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

#include <algorithm>
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
// In our case GEMM is uniquely identified by N and K, plus the target SM architecture (so the
// SM80-compatibility and native SM90 kernels for the same shape do not share profiled configs).
class GemmIdCore {
 public:
  int n;
  int k;
  nvinfer::DataType dtype;
  int sm;

  GemmIdCore(int n_, int k_, nvinfer::DataType const& dtype_, int sm_ = 0)
      : n(n_), k(k_), dtype(dtype_), sm(sm_) {
  }

  GemmIdCore()
      : n(-1), k(-1), dtype(nvinfer::DataType::kFLOAT),  // dtype does not matter here
        sm(0) {
  }

  bool operator==(GemmIdCore const& id) const {
    return isEqual(id);
  }

  friend std::ostream& operator<<(std::ostream& out, GemmIdCore const& id) {
    out << "(N;K)=(" << id.n << ";" << id.k << "),";
    out << " type=" << static_cast<int>(id.dtype);
    out << " sm=" << id.sm;
    return out;
  }

 protected:
  bool isEqual(GemmIdCore const& id) const {
    return n == id.n && k == id.k && dtype == id.dtype && sm == id.sm;
  }
};

// Hash of GemmId
struct GemmIdCoreHash {
  std::size_t operator()(GemmIdCore const& id) const {
    auto h1 = std::hash<int>{}(id.n);
    auto h2 = std::hash<int>{}(id.k);
    auto h3 = std::hash<int>{}(static_cast<int>(id.dtype));
    auto h4 = std::hash<int>{}(id.sm);
    return h1 ^ h2 ^ h3 ^ h4;
  }
};

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
class GemmPluginProfiler {
 public:
  // Map for single GEMM for different Ms (GEMM dimension) to the best config for particular M
  using MProfileMap = std::unordered_map<int, std::optional<Config>>;
  using MProfileMapPtr = std::shared_ptr<MProfileMap>;

  // requires shared ownership to read from *this
  using reader_lock = std::shared_lock<std::shared_timed_mutex>;
  // requires exclusive ownership to write to *this
  using writer_lock = std::unique_lock<std::shared_timed_mutex>;

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

  // Like getBestConfig, but if the requested M bucket has not been profiled yet, profiles it
  // lazily (single bucket) and inserts it into the in-process map. This briefly blocks the caller
  // but guarantees a tuned tactic for any runtime M, which is what makes the reduced first-time M
  // sweep safe. Must not be called while the compute stream is being captured into a CUDA graph
  // (the caller is responsible for using getBestConfig instead during capture).
  std::optional<Config> getBestConfigOrProfile(int m, GemmIdType const& gemmId);

  virtual int getMaxProfileM() const;

 protected:
  virtual void runTactic(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t const& stream) = 0;

  virtual size_t computeTmpSize(size_t maxM, size_t n, size_t k) = 0;

  virtual bool checkTactic(int /*m*/, int /*n*/, int /*k*/, Config const& /*tactic*/) const {
    return true;
  }

  virtual std::vector<Config> getTactics(int m, int n, int k) const = 0;

  virtual void initTmpData(int m, int n, int k, char* workspace, size_t size, cudaStream_t stream);

  // Returns the ordered set of M buckets to profile during the initial sweep, given the
  // (rounded) profile range [minM, maxM]. The default reproduces the historical dense sweep.
  // Subclasses may override to profile a smaller, configurable bucket set.
  virtual std::vector<int> getProfileMBuckets(int minM, int maxM, bool hasWeightOnlyCudaKernel) const;

 private:
  std::optional<Config> profileTacticsForProblem(int m, int n, int k, std::vector<Config> const& tactics,
                                                 char* workspace, cudaStream_t stream);

  float profileTacticForProblem(int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t stream);

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

  GemmDims mDims{};

  bool mSkip{false};

  // Remembered from the initial profileTactics call so lazy single-bucket profiling can
  // reproduce the same tactic candidate set.
  bool mHasWeightOnlyCudaKernel{false};

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
  mDims = dims;
  mHasWeightOnlyCudaKernel = hasWeightOnlyCudaKernel;

  int const maxM = std::min(nextPowerOfTwo(static_cast<int>(dims.maxM)), getMaxProfileM());

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
  onnxruntime::IAllocatorUniquePtr<char> workspace_tmp{nullptr};
  cudaStream_t stream;

  auto profileTactics = [&](int m, int n, int k) {
    if (mProfileMap->count(m) == 0) {
      if (!isAllocated) {
        workspace_tmp = onnxruntime::IAllocator::MakeUniquePtr<char>(mAllocator, workspace_bytes, true);
#if ORT_LLM_VERBOSE
        AllocatorStats stats;
        this->mAllocator->GetStats(&stats);
        std::cout << "Allocator state after " << workspace_bytes << " bytes gemm profiler workspace:" << std::endl
                  << stats.DebugString() << std::endl;
#endif
        isAllocated = true;
      }

      initTmpData(m, n, k, workspace_tmp.get(), workspace_bytes, stream);

      auto tactics = this->getTactics(m, n, k);
      // Profile different tactics for particular m and insert best config to the map
      mProfileMap->insert({m, this->profileTacticsForProblem(m, n, k, tactics, workspace_tmp.get(), stream)});
    }
  };

  CUDA_CALL_THROW(cudaStreamCreate(&stream));

  // Profile the (possibly reduced) set of M buckets. Any unprofiled runtime M is handled
  // later by lazy single-bucket profiling in getBestConfigOrProfile.
  for (int m : getProfileMBuckets(static_cast<int>(dims.minM), maxM, hasWeightOnlyCudaKernel)) {
    profileTactics(m, static_cast<int>(dims.n), static_cast<int>(dims.k));
  }

  if (isAllocated) {
    // Free tmp data
    workspace_tmp.reset();
  }
  CUDA_CALL_THROW(cudaStreamDestroy(stream));
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
std::vector<int> GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::getProfileMBuckets(
    int minM, int maxM, bool hasWeightOnlyCudaKernel) const {
  // Default: reproduce the historical dense sweep so any other users of this template keep
  // their behavior. Subclasses may override this to profile a smaller bucket set.
  std::vector<int> buckets;
  if (hasWeightOnlyCudaKernel) {
    for (int m = std::max(1, minM); m < std::min(16, maxM); m += 1) {
      buckets.push_back(m);
    }
    for (int m = 16; m < maxM; m *= 2) {
      buckets.push_back(m);
    }
  } else {
    for (int m = std::max(1, nextPowerOfTwo(minM)); m < maxM; m *= 2) {
      buckets.push_back(m);
    }
  }
  buckets.push_back(maxM);
  return buckets;
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
std::optional<Config> GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::getBestConfigOrProfile(
    int m, GemmIdType const& gemmId) {
  if (mSkip) {
    return std::nullopt;
  }

  int const target = std::min(std::max(1, nextPowerOfTwo(m)), getMaxProfileM());

  // Fast path: an already-profiled (exact or rounded) bucket under a shared read lock.
  {
    reader_lock lock(mMNKProfileMap->mutex);
    if (mMNKProfileMap->existsMProfileMap(gemmId)) {
      auto mProfileMap = mMNKProfileMap->getMProfileMap(gemmId);
      if (mProfileMap->count(m) > 0) {
        return mProfileMap->at(m);
      }
      if (mProfileMap->count(target) > 0) {
        return mProfileMap->at(target);
      }
    }
  }

  // We can only profile lazily if the profiling context from construction is available.
  if (mRunner == nullptr || mAllocator == nullptr || !mDims.isInitialized()) {
    ORT_LLM_LOG_WARNING("Cannot lazily profile an unprofiled M bucket: profiler context is unavailable.");
    return std::nullopt;
  }

  int const n = static_cast<int>(mDims.n);
  int const k = static_cast<int>(mDims.k);
  size_t const workspace_bytes = computeTmpSize(target, n, k);

  cudaStream_t stream;
  CUDA_CALL_THROW(cudaStreamCreate(&stream));
  auto workspace_tmp = onnxruntime::IAllocator::MakeUniquePtr<char>(mAllocator, workspace_bytes, true);
  initTmpData(target, n, k, workspace_tmp.get(), workspace_bytes, stream);

  auto tactics = this->getTactics(target, n, k);
  auto best = this->profileTacticsForProblem(target, n, k, tactics, workspace_tmp.get(), stream);

  workspace_tmp.reset();
  CUDA_CALL_THROW(cudaStreamDestroy(stream));

  writer_lock lock(mMNKProfileMap->mutex);
  if (!mMNKProfileMap->existsMProfileMap(gemmId)) {
    mMNKProfileMap->createMProfileMap(gemmId);
  }
  auto mProfileMap = mMNKProfileMap->getMProfileMap(gemmId);

  if (mProfileMap->count(m) > 0) {
    return mProfileMap->at(m);
  }
  if (mProfileMap->count(target) > 0) {
    return mProfileMap->at(target);
  }

  mProfileMap->insert({target, best});

  return best;
}

template <typename Config, typename RunnerPtr, typename GemmIdType, typename GemmIdHashType>
std::optional<Config> GemmPluginProfiler<Config, RunnerPtr, GemmIdType, GemmIdHashType>::profileTacticsForProblem(
    int m, int n, int k, std::vector<Config> const& tactics, char* workspace, cudaStream_t stream) {
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
      time = profileTacticForProblem(m, n, k, candidateConfig, workspace, stream);

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
    int m, int n, int k, Config const& tactic, char* workspace, cudaStream_t stream) {
  constexpr int warmup = 5;
  constexpr int runs = 10;

  // Warmup the execution
  for (int i = 0; i < warmup; ++i) {
    runTactic(m, n, k, tactic, workspace, stream);
  }

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CALL_THROW(cudaEventCreate(&start));
  CUDA_CALL_THROW(cudaEventCreate(&stop));
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  CUDA_CALL_THROW(cudaEventRecord(start, stream));

  // Profile GEMM
  for (int i = 0; i < runs; ++i) {
    runTactic(m, n, k, tactic, workspace, stream);
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
