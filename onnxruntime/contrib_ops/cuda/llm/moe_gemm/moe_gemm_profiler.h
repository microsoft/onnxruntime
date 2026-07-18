// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/llm/gemm_profiler.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include <cutlass/numeric_types.h>
#include <optional>
#include <unordered_map>

namespace onnxruntime::llm::kernels::cutlass_kernels {

// Define MoeGemmId - includes weight type for proper buffer sizing
class MoeGemmId {
 public:
  enum class GemmType {
    Gemm1 = 0,
    Gemm2 = 1,
  };

  int n{0};
  int k{0};
  nvinfer::DataType dtype{nvinfer::DataType::kHALF};
  nvinfer::DataType wtype{nvinfer::DataType::kHALF};  // Weight type
  GemmType gemm_type{GemmType::Gemm1};

  MoeGemmId() = default;

  MoeGemmId(int n_, int k_, nvinfer::DataType dtype_, nvinfer::DataType wtype_, GemmType gemm_type_)
      : n(n_), k(k_), dtype(dtype_), wtype(wtype_), gemm_type(gemm_type_) {}

  // Legacy constructor for backward compatibility (assumes wtype == dtype)
  MoeGemmId(int n_, int k_, nvinfer::DataType dtype_, GemmType gemm_type_)
      : n(n_), k(k_), dtype(dtype_), wtype(dtype_), gemm_type(gemm_type_) {}

  bool operator==(MoeGemmId const& id) const {
    return n == id.n && k == id.k && dtype == id.dtype && wtype == id.wtype && gemm_type == id.gemm_type;
  }

  bool operator!=(MoeGemmId const& id) const {
    return !(*this == id);
  }

  friend std::ostream& operator<<(std::ostream& out, MoeGemmId const& id) {
    out << "(N;K)=(" << id.n << ";" << id.k << "),";
    out << " dtype=" << static_cast<int>(id.dtype);
    out << " wtype=" << static_cast<int>(id.wtype);
    out << " gemm_type=" << static_cast<int>(id.gemm_type);
    return out;
  }
};

struct MoeGemmIdHash {
  std::size_t operator()(MoeGemmId const& id) const {
    auto h1 = std::hash<int>{}(id.n);
    auto h2 = std::hash<int>{}(id.k);
    auto h3 = std::hash<int>{}(static_cast<int>(id.dtype));
    auto h4 = std::hash<int>{}(static_cast<int>(id.wtype));
    auto h5 = std::hash<int>{}(static_cast<int>(id.gemm_type));
    return h1 ^ h2 ^ h3 ^ h4 ^ h5;
  }
};

// MoeGemmProfiler using GemmProfilerBackend for proper grouped GEMM profiling
class MoeGemmProfiler {
 public:
  using Config = cutlass_extensions::CutlassGemmConfig;

  MoeGemmProfiler() = default;

  void setAllocator(AllocatorPtr allocator) {
    allocator_ = allocator;
  }

  // Set profiler parameters including weight type for quantized weights
  void setProfilerParams(int num_experts, int k, int64_t hidden_size, int64_t inter_size, int64_t group_size,
                         ActivationType activation_type, bool bias,
                         bool need_weights, MOEParallelismConfig parallelism_config,
                         int sm) {
    num_experts_ = num_experts;
    k_ = k;
    hidden_size_ = hidden_size;
    inter_size_ = inter_size;
    group_size_ = group_size;
    activation_type_ = activation_type;
    bias_ = bias;
    need_weights_ = need_weights;
    parallelism_config_ = parallelism_config;
    sm_ = sm;
  }

  // Profile tactics for a GEMM problem using GemmProfilerBackend.
  // Profiles (and caches) the best config for the M bucket that contains dims.maxM. The first
  // call for a new (GemmId, M-bucket) pair runs the profiler; subsequent calls return immediately.
  // The data/weight types are taken from gemmId, so no separate dtype argument is needed.
  // `timing_stream` (the ORT compute stream) is required: the profiler runs its kernels on it so
  // they are strictly ordered with the surrounding compute-stream work and share the same temp
  // arena, avoiding the cross-stream scratch race that a private side stream would introduce.
  // Must not be called while `timing_stream` is being captured into a CUDA graph.
  void profileTactics(CutlassMoeFCRunnerInterface* runner,
                      weight_only::GemmDims const& dims, MoeGemmId const& gemmId,
                      cudaStream_t timing_stream);

  // Get best config for a given M and GemmId. Selects the config profiled for the M bucket that
  // contains m, so small-M (decode) GEMMs use a decode-tuned tile instead of a prefill-tuned one.
  std::optional<Config> getBestConfig(int m, MoeGemmId const& id) const;

  // Snap a row count M to a representative profiling bucket. Decode (small M) and prefill
  // (large M) favor very different CUTLASS tile shapes, so we keep a separate best config per
  // bucket rather than reusing a single shape-only config. Buckets are powers of two.
  static int bucketM(int64_t m);

 private:
  // Initialize backend for profiling
  void initBackend(CutlassMoeFCRunnerInterface* runner, MoeGemmId const& gemmId);

  // Run profiling for all tactics
  std::optional<Config> runProfiling(int maxM, MoeGemmId const& gemmId, cudaStream_t timing_stream);

  AllocatorPtr allocator_;
  GemmProfilerBackend backend_;
  CutlassMoeFCRunnerInterface* runner_{nullptr};

  // Cached results: GemmId -> (M bucket -> best config)
  mutable std::unordered_map<MoeGemmId, std::unordered_map<int, std::optional<Config>>, MoeGemmIdHash> config_cache_;

  // Profiler parameters
  int num_experts_{0};
  int k_{0};
  int64_t hidden_size_{0};
  int64_t inter_size_{0};
  int64_t group_size_{0};
  ActivationType activation_type_{ActivationType::Gelu};
  bool bias_{false};
  bool need_weights_{false};
  MOEParallelismConfig parallelism_config_{};
  int sm_{0};
};

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
