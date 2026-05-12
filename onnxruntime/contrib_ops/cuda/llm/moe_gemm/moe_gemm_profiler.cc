// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_profiler.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/moe_gemm/common.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels.h"

#include <functional>
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

std::optional<MoeGemmProfiler::Config> MoeGemmProfiler::runProfiling(int maxM, MoeGemmId const& gemmId) {
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

  cudaStream_t stream = nullptr;
  CUDA_CALL_THROW(cudaStreamCreate(&stream));
  std::unique_ptr<CUstream_st, void (*)(cudaStream_t)> stream_guard(
      stream, [](cudaStream_t s) { if (s) cudaStreamDestroy(s); });

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

void MoeGemmProfiler::profileTactics(CutlassMoeFCRunnerInterface* runner, nvinfer::DataType dtype,
                                     weight_only::GemmDims const& dims, MoeGemmId const& gemmId) {
  ORT_LLM_LOG_ENTRY();
  // Check if already cached
  (void)dtype;
  auto it = config_cache_.find(gemmId);
  if (it != config_cache_.end()) {
    return;  // Already profiled
  }

  // Initialize backend with correct types
  initBackend(runner, gemmId);

  // Run profiling
  int maxM = static_cast<int>(dims.maxM);
  auto result = runProfiling(maxM, gemmId);

  // Cache result
  config_cache_[gemmId] = result;
}

std::optional<MoeGemmProfiler::Config> MoeGemmProfiler::getBestConfig(int m, MoeGemmId const& id) const {
  ORT_LLM_LOG_ENTRY();
  (void)m;  // M is already factored into profiling
  auto it = config_cache_.find(id);
  if (it != config_cache_.end()) {
    return it->second;
  }
  return std::nullopt;
}

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
