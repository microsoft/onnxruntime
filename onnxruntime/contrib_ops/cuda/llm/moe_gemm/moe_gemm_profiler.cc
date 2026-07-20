// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_profiler.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/moe_gemm/common.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels.h"

#include <functional>
#include <limits>
#include <memory>

namespace onnxruntime::llm::kernels::cutlass_kernels {

void MoeGemmProfiler::initBackend(CutlassMoeFCRunnerInterface* runner, MoeGemmId const& gemmId) {
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

std::optional<MoeGemmProfiler::Config> MoeGemmProfiler::runProfiling(int maxM, MoeGemmId const& gemmId,
                                                                     cudaStream_t timing_stream) {
  ORT_LLM_LOG_ENTRY();
  ORT_LLM_LOG_DEBUG(onnxruntime::MakeString("MoeGemmProfiler::runProfiling for M=", maxM, " ", gemmId));

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
  void* workspace = allocator_->Alloc(workspace_size);
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
    auto const& tactic = tactics[i];
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
    } catch (std::exception const& e) {
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
                                     weight_only::GemmDims const& dims, MoeGemmId const& gemmId,
                                     cudaStream_t timing_stream) {
  ORT_LLM_LOG_ENTRY();

  // Profile per M bucket: decode (small M) and prefill (large M) prefer different tile shapes,
  // so cache a separate best config for each bucket instead of a single shape-only config.
  int const bucket = bucketM(dims.maxM);
  auto& bucket_map = config_cache_[gemmId];
  if (bucket_map.find(bucket) != bucket_map.end()) {
    return;  // Already profiled for this (GemmId, M bucket).
  }

  // Initialize backend with correct types
  initBackend(runner, gemmId);

  // Run profiling at the bucket's representative M.
  auto result = runProfiling(bucket, gemmId, timing_stream);

  // Cache result for this bucket
  bucket_map[bucket] = result;
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

std::optional<MoeGemmProfiler::Config> MoeGemmProfiler::getBestConfig(int m, MoeGemmId const& id) const {
  ORT_LLM_LOG_ENTRY();
  auto it = config_cache_.find(id);
  if (it == config_cache_.end()) {
    return std::nullopt;
  }
  auto const& bucket_map = it->second;
  int const bucket = bucketM(m);

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
  for (auto const& kv : bucket_map) {
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
