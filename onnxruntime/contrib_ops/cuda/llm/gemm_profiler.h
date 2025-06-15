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

  void setTmpWorkspaceSizeInBytes(size_t bytes) {
    mTmpWorkspaceSizeInBytes = bytes;
  }

  void setSkip(bool skip) {
    mSkip = mSkip || skip;
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
  void allocateTmpData();

  void freeTmpData();

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

  size_t mTmpWorkspaceSizeInBytes{0};

  char* mWorkspaceTmp{nullptr};

  cudaStream_t mStream;

  GemmDims mDims{};

  bool mSkip{false};
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
