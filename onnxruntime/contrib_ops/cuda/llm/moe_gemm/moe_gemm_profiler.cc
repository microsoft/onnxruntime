// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_profiler.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/moe_gemm/common.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels.h"
#include "core/platform/env_var_utils.h"

#include <atomic>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <typeindex>
#include <typeinfo>

namespace onnxruntime::llm::kernels::cutlass_kernels {

namespace {

// Identical-shape MoE layers (e.g. the 48 QMoE nodes of one model) profile the same problem. Share each
// (runner type and tactic-relevant state, device, problem, GemmId, M bucket) result across profiler instances so
// it is timed once per process instead of once per node. ORT_DISABLE_MOE_SHARED_TACTIC_CACHE=1 restores per-node
// profiling.
bool SharedTacticCacheDisabled() {
  static const bool disabled =
      onnxruntime::ParseEnvironmentVariableWithDefault<int>("ORT_DISABLE_MOE_SHARED_TACTIC_CACHE", 0) == 1;
  return disabled;
}

// Everything a profiling run reads, so a shared result is always one this runner could produce: the runner type, runner
// state that changes the candidate tactics (SM80 FP4 family) or the profiled GEMM2 epilogue (deterministic Hopper
// reduce), the device, every GemmProfilerBackend::init parameter, the GEMM and the M bucket.
struct SharedTacticKey {
  std::type_index runner_type = typeid(void);
  bool use_sm80_fp4 = false;
  bool use_deterministic_hopper_reduce = false;
  int device_id = 0;
  int sm = 0;
  int num_experts = 0;
  int k = 0;
  int64_t hidden_size = 0;
  int64_t inter_size = 0;
  int64_t group_size = 0;
  ActivationType activation_type = ActivationType::Gelu;
  bool bias = false;
  bool need_weights = false;
  int tp_size = 0;
  int tp_rank = 0;
  int ep_size = 0;
  int ep_rank = 0;
  int cluster_size = 0;
  int cluster_rank = 0;
  int gemm_n = 0;
  int gemm_k = 0;
  nvinfer::DataType dtype = nvinfer::DataType::kHALF;
  nvinfer::DataType wtype = nvinfer::DataType::kHALF;
  MoeGemmId::GemmType gemm_type = MoeGemmId::GemmType::Gemm1;
  int bucket = 0;

  auto Tie() const {
    return std::tie(runner_type, use_sm80_fp4, use_deterministic_hopper_reduce, device_id, sm, num_experts, k,
                    hidden_size, inter_size, group_size, activation_type, bias, need_weights, tp_size, tp_rank, ep_size,
                    ep_rank, cluster_size, cluster_rank, gemm_n, gemm_k, dtype, wtype, gemm_type, bucket);
  }

  bool operator<(const SharedTacticKey& other) const { return Tie() < other.Tie(); }
};

// The entry mutex is held while its key is profiled: concurrent first uses of one key time it once, while other keys
// (other problems, M buckets or devices) profile in parallel. Only successful results are kept, so a failed run (for
// example a transient allocation failure) is retried by the next profiler instead of being reused process-wide.
struct SharedTacticEntry {
  std::mutex mutex;
  std::optional<MoeGemmProfiler::Config> config;
};

class SharedTacticCache {
 public:
  SharedTacticEntry& Get(const SharedTacticKey& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_[key];  // std::map nodes never move, so the entry stays valid after the lock is released.
  }

 private:
  std::mutex mutex_;
  std::map<SharedTacticKey, SharedTacticEntry> entries_;
};

SharedTacticCache& GetSharedTacticCache() {
  // Leaked on purpose, like the MatMulNBits tactic cache registry: a CUDA EP owned by another static may be
  // destroyed after this library's statics.
  static auto* cache = new SharedTacticCache();
  return *cache;
}

std::atomic<size_t> g_profiling_runs{0};

}  // namespace

size_t MoeGemmProfiler::ProfilingRunCountForTesting() {
  return g_profiling_runs.load();
}

void MoeGemmProfiler::initBackend(CutlassMoeFCRunnerInterface* runner, const MoeGemmId& gemmId) {
  runner_ = runner;

  auto gemm_to_profile = (gemmId.gemm_type == MoeGemmId::GemmType::Gemm1)
                             ? GemmProfilerBackend::GemmToProfile::GEMM_1
                             : GemmProfilerBackend::GemmToProfile::GEMM_2;

  // Infer output type - same as dtype for non-FP8/FP4
  nvinfer::DataType otype = gemmId.dtype;

  backend_.init(*runner, gemm_to_profile, gemmId.dtype, gemmId.wtype, otype,
                num_experts_, k_, hidden_size_, inter_size_, group_size_,
                activation_type_, bias_,
                need_weights_, parallelism_config_);
}

std::optional<MoeGemmProfiler::Config> MoeGemmProfiler::runProfiling(int maxM, const MoeGemmId& gemmId,
                                                                     cudaStream_t timing_stream) {
  ORT_LLM_LOG_ENTRY();
  ORT_LLM_LOG_DEBUG(onnxruntime::MakeString("MoeGemmProfiler::runProfiling for M=", maxM, " ", gemmId));
  g_profiling_runs.fetch_add(1);

  // Get tactics from runner
  auto tactics = runner_->getTactics();

  if (tactics.empty()) {
    ORT_LLM_LOG_WARNING("No tactics available for MoE GEMM profiling");
    return std::nullopt;
  }

  // Allocate workspace
  size_t workspace_size = backend_.getWorkspaceSize(maxM);
  if (workspace_size == 0) {
    ORT_LLM_LOG_WARNING("Workspace size is 0 for MoE GEMM profiling");
    return std::nullopt;
  }

  // RAII guards so any throw between allocation and the end of this function still releases
  // the workspace, the profiling stream, and the timing events. Without these, exceptions
  // escaping backend_.prepare(), cudaEventRecord(), or cudaEventSynchronize() would leak.
  // On the built-in CUDA EP an out-of-memory Alloc() throws instead of returning nullptr. Both mean no config, so the
  // caller falls back to tactics[0] and a later profiler of the same problem retries.
  void* workspace = nullptr;
  try {
    workspace = allocator_->Alloc(workspace_size);
  } catch (const std::exception& e) {
    ORT_LLM_LOG_WARNING(onnxruntime::MakeString(
        "Failed to allocate workspace (", workspace_size, " bytes) for MoE GEMM profiling: ", e.what()));
    return std::nullopt;
  }
  if (!workspace) {
    ORT_LLM_LOG_WARNING("Failed to allocate workspace for MoE GEMM profiling");
    return std::nullopt;
  }
  std::unique_ptr<void, std::function<void(void*)>> workspace_guard(
      workspace, [a = allocator_](void* p) { if (p) a->Free(p); });
  auto* workspace_ptr = static_cast<char*>(workspace);

  // Run profiling on the caller-supplied stream (the ORT compute stream) when one is provided,
  // so every profiler kernel is strictly ordered with the surrounding compute-stream work and
  // shares its stream context with the temp allocator. Profiling on a private side stream instead
  // races with the compute stream: the temp arena is stream-aware and, because the side-stream
  // usage is invisible to it, can hand the same scratch block to a later compute-stream allocation
  // (e.g. the real MoE workspace) while the profiler's grouped-GEMM kernels are still in flight.
  // The resulting overlapping access corrupts the profiler's routing/GEMM buffers and surfaces as a
  // sticky CUDA 700 (illegal memory access) at a downstream MoE kernel launch. Only create and
  // destroy a private stream when the caller does not supply one (e.g. standalone profiling).
  cudaStream_t stream = timing_stream;
  std::unique_ptr<CUstream_st, void (*)(cudaStream_t)> stream_guard(
      nullptr, [](cudaStream_t s) { if (s) cudaStreamDestroy(s); });
  if (stream == nullptr) {
    CUDA_CALL_THROW(cudaStreamCreate(&stream));
    stream_guard.reset(stream);
  }

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CALL_THROW(cudaEventCreate(&start));
  std::unique_ptr<CUevent_st, void (*)(cudaEvent_t)> start_guard(
      start, [](cudaEvent_t e) { if (e) cudaEventDestroy(e); });
  CUDA_CALL_THROW(cudaEventCreate(&stop));
  std::unique_ptr<CUevent_st, void (*)(cudaEvent_t)> stop_guard(
      stop, [](cudaEvent_t e) { if (e) cudaEventDestroy(e); });

  // Prepare backend (may throw; guards above will release stream/events/workspace).
  backend_.prepare(maxM, workspace_ptr, nullptr /* expert_weights */, stream);

  // Profile each tactic
  float best_time = std::numeric_limits<float>::max();
  Config best_config;
  bool found_one = false;

  constexpr int warmup_iters = 3;
  constexpr int profile_iters = 10;

  for (size_t i = 0; i < tactics.size(); ++i) {
    const auto& tactic = tactics[i];
    try {
      // Warmup
      for (int j = 0; j < warmup_iters; ++j) {
        backend_.runProfiler(maxM, tactic, workspace_ptr, nullptr, stream);
      }
      CUDA_CALL_THROW(cudaStreamSynchronize(stream));

      // Profile
      CUDA_CALL_THROW(cudaEventRecord(start, stream));
      for (int k = 0; k < profile_iters; ++k) {
        backend_.runProfiler(maxM, tactic, workspace_ptr, nullptr, stream);
      }
      CUDA_CALL_THROW(cudaEventRecord(stop, stream));
      CUDA_CALL_THROW(cudaEventSynchronize(stop));

      float elapsed_ms = 0;
      CUDA_CALL_THROW(cudaEventElapsedTime(&elapsed_ms, start, stop));
      float avg_time = elapsed_ms / profile_iters;

      if (avg_time < best_time) {
        best_time = avg_time;
        best_config = tactic;
        found_one = true;
      }
    } catch (const std::exception& e) {
      ORT_LLM_LOG_DEBUG(onnxruntime::MakeString("Tactic failed: ", e.what(), " ", tactic.toString()));
      cudaGetLastError();  // Clear error
      continue;
    }
  }

  // RAII guards above release stream/events/workspace on the way out.

  if (!found_one) {
    ORT_LLM_LOG_WARNING(onnxruntime::MakeString("No valid GEMM config found for ", gemmId));
    return std::nullopt;
  }

  ORT_LLM_LOG_DEBUG(onnxruntime::MakeString("Best config for ", gemmId, ": ", best_config.toString(), ", time=", best_time, "ms"));
  return best_config;
}

void MoeGemmProfiler::profileTactics(CutlassMoeFCRunnerInterface* runner,
                                     const weight_only::GemmDims& dims, const MoeGemmId& gemmId,
                                     cudaStream_t timing_stream) {
  ORT_LLM_LOG_ENTRY();

  // Profile per M bucket: decode (small M) and prefill (large M) prefer different tile shapes,
  // so cache a separate best config for each bucket instead of a single shape-only config.
  const int bucket = bucketM(dims.maxM);
  auto& bucket_map = config_cache_[gemmId];
  if (bucket_map.find(bucket) != bucket_map.end()) {
    return;  // Already profiled for this (GemmId, M bucket).
  }

  if (SharedTacticCacheDisabled()) {
    initBackend(runner, gemmId);
    bucket_map[bucket] = runProfiling(bucket, gemmId, timing_stream);
    return;
  }

  SharedTacticKey key;
  key.runner_type = typeid(*runner);
  key.use_sm80_fp4 = runner->useSm80Fp4();
  key.use_deterministic_hopper_reduce = runner->use_deterministic_hopper_reduce_;
  CUDA_CALL_THROW(cudaGetDevice(&key.device_id));
  key.sm = sm_;
  key.num_experts = num_experts_;
  key.k = k_;
  key.hidden_size = hidden_size_;
  key.inter_size = inter_size_;
  key.group_size = group_size_;
  key.activation_type = activation_type_;
  key.bias = bias_;
  key.need_weights = need_weights_;
  key.tp_size = parallelism_config_.tp_size;
  key.tp_rank = parallelism_config_.tp_rank;
  key.ep_size = parallelism_config_.ep_size;
  key.ep_rank = parallelism_config_.ep_rank;
  key.cluster_size = parallelism_config_.cluster_size;
  key.cluster_rank = parallelism_config_.cluster_rank;
  key.gemm_n = gemmId.n;
  key.gemm_k = gemmId.k;
  key.dtype = gemmId.dtype;
  key.wtype = gemmId.wtype;
  key.gemm_type = gemmId.gemm_type;
  key.bucket = bucket;

  auto& entry = GetSharedTacticCache().Get(key);
  std::lock_guard<std::mutex> lock(entry.mutex);
  if (!entry.config.has_value()) {
    initBackend(runner, gemmId);
    entry.config = runProfiling(bucket, gemmId, timing_stream);
  }
  bucket_map[bucket] = entry.config;
}

int MoeGemmProfiler::bucketM(int64_t m) {
  // Snap M up to the next power of two so a handful of buckets cover the full range. M=1 (the
  // common batch-1 decode case) gets its own bucket and therefore its own decode-tuned tactic.
  if (m <= 1) {
    return 1;
  }
  // Saturate large M values into one bucket to keep the cache bounded.
  constexpr int64_t kMaxBucket = 8192;
  if (m >= kMaxBucket) {
    return static_cast<int>(kMaxBucket);
  }
  int64_t bucket = 1;
  while (bucket < m) {
    bucket <<= 1;
  }
  return static_cast<int>(bucket);
}

std::optional<MoeGemmProfiler::Config> MoeGemmProfiler::getBestConfig(int m, const MoeGemmId& id) const {
  ORT_LLM_LOG_ENTRY();
  auto it = config_cache_.find(id);
  if (it == config_cache_.end()) {
    return std::nullopt;
  }
  const auto& bucket_map = it->second;
  const int bucket = bucketM(m);

  // Exact bucket profiled: use it.
  auto exact = bucket_map.find(bucket);
  if (exact != bucket_map.end()) {
    return exact->second;
  }

  // Not profiled for this exact bucket. Fall back to the nearest profiled bucket: prefer the
  // smallest profiled bucket >= requested (tuned for at least this much work), otherwise the
  // largest profiled bucket below it.
  std::optional<Config> best_ge;
  int best_ge_bucket = std::numeric_limits<int>::max();
  std::optional<Config> best_lt;
  int best_lt_bucket = -1;
  for (const auto& kv : bucket_map) {
    if (kv.first >= bucket) {
      if (kv.first < best_ge_bucket) {
        best_ge_bucket = kv.first;
        best_ge = kv.second;
      }
    } else {
      if (kv.first > best_lt_bucket) {
        best_lt_bucket = kv.first;
        best_lt = kv.second;
      }
    }
  }
  if (best_ge_bucket != std::numeric_limits<int>::max()) {
    return best_ge;
  }
  if (best_lt_bucket >= 0) {
    return best_lt;
  }
  return std::nullopt;
}

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
