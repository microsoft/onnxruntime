// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/moe/moe_base.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_profiler.h"
#include "core/common/common.h"
#include "core/framework/kernel_pilot_moe_expert_usage.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <array>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class MoE final : public CudaKernel, public MoEBase
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
    ,
                  public IKernelPilotMoeExpertUsage
#endif
{
 public:
  explicit MoE(const OpKernelInfo& op_kernel_info);
  ~MoE() override;
  Status ComputeInternal(OpKernelContext* ctx) const override;
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  Status InitializeKernelPilot(KernelPilot* pilot) override;
#endif

 private:
  static constexpr bool kCpuOffloadSupported =
      std::is_same_v<T, MLFloat16> || std::is_same_v<T, BFloat16>;

  struct PackedTensor {
    TensorShape shape;
    std::vector<T> cpu_data;
    std::vector<T> cpu_gemm_data;
    std::vector<float> cpu_gemm_float_data;
    IAllocatorUniquePtr<void> cuda_data;
    size_t bytes{0};
    bool present{false};
  };

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  Status InitializeCudaExpertWeights(gsl::span<const int> cuda_experts);
  struct CpuTask;
  Status PrepareUsage(OpKernelContext* context, KernelPilot* pilot, size_t expert_count,
                      size_t num_rows, size_t hidden_size, size_t inter_size, bool fused_swiglu,
                      bool prepare_cpu_task) const;
  Status ComputeCpuTask() const;
  void RunCpuWorker() const;
  static void CUDART_CB NotifyCpuWorker(void* context);
  void ReleaseUsageResources() noexcept;
  bool HasPendingUsage() const noexcept override { return usage_pending_; }
  Status CompleteUsage(KernelPilotMoeExpertUsageSnapshot& snapshot) override;

#endif

  mutable onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler mGemmProfiler;
  mutable std::mutex mGemmProfilerMutex;
  bool cpu_offload_enabled_{false};
  AllocatorPtr cuda_allocator_;
  std::array<PackedTensor, 8> packed_inputs_;
  InlinedVector<int> cuda_experts_;
  InlinedVector<int> expert_map_;
  IAllocatorUniquePtr<void> device_expert_map_;
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  cudaStream_t input_copy_stream_{nullptr};
  int device_id_{0};
  mutable std::mutex input_copy_mutex_;
  bool async_cpu_supported_{false};
  mutable bool usage_pending_{false};
  mutable bool usage_recorded_{false};
  mutable cudaStream_t usage_stream_{nullptr};
  cudaEvent_t usage_complete_event_{nullptr};
  mutable IAllocatorUniquePtr<void> device_usage_;
  mutable IAllocatorUniquePtr<void> host_usage_;
  mutable std::unique_ptr<CpuTask> cpu_task_;
  mutable std::thread cpu_worker_;
  mutable std::mutex cpu_worker_mutex_;
  mutable std::condition_variable cpu_worker_condition_;
  mutable bool cpu_task_ready_{false};
  mutable bool cpu_task_finished_{true};
  mutable bool cpu_callback_scheduled_{false};
  mutable bool stop_cpu_worker_{false};
#endif
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
