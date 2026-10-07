// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/moe/moe_base.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_profiler.h"
#include "core/common/common.h"
#include "core/framework/kernel_pilot_moe_expert_cache.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <array>
#include <mutex>
#include <vector>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class MoE final : public CudaKernel, public MoEBase
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
    ,
                  public IKernelPilotMoeExpertCache
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
    size_t expert_bytes{0};
    size_t swap_offset{0};
    size_t gemm_input_size{0};
    size_t gemm_output_size{0};
    bool present{false};
  };

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  Status InitializeCudaExpertWeights(gsl::span<const int> cuda_experts);
  Status PrepareExpertSwapForInvocation(cudaStream_t stream, KernelPilot* pilot) const;
  void ReleaseSwapResources() noexcept;
  void PrepareIncomingExpert() noexcept;
  static void CUDART_CB PrepareIncomingExpertCallback(void* context);

  int DeviceId() const noexcept override { return device_id_; }
  bool HasPendingSwap() const noexcept override;
  Status ReclaimCompletedSwap() override;
  Status StartSwap(int cuda_expert_id, int cpu_expert_id) override;
#endif

  mutable onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler mGemmProfiler;
  mutable std::mutex mGemmProfilerMutex;
  bool cpu_offload_enabled_{false};
  AllocatorPtr cuda_allocator_;
  std::array<PackedTensor, 8> packed_inputs_;
  mutable InlinedVector<int> cuda_experts_;
  mutable InlinedVector<int> expert_map_;
  IAllocatorUniquePtr<void> device_expert_map_;
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  cudaStream_t input_copy_stream_{nullptr};
  cudaStream_t swap_d2h_stream_{nullptr};
  cudaStream_t swap_h2d_stream_{nullptr};
  cudaEvent_t last_expert_use_event_{nullptr};
  cudaEvent_t swap_cpu_ready_event_{nullptr};
  cudaEvent_t swap_transfer_complete_event_{nullptr};
  cudaEvent_t swap_publication_complete_event_{nullptr};
  enum class SwapPhase {
    Idle,
    TransferInFlight,
    PublicationInFlight,
  };
  mutable SwapPhase swap_phase_{SwapPhase::Idle};
  mutable int swap_cuda_expert_{-1};
  mutable int swap_cpu_expert_{-1};
  mutable size_t swap_cuda_slot_{0};
  size_t swap_staging_bytes_{0};
  int device_id_{0};
  mutable IAllocatorUniquePtr<void> swap_pinned_buffer_;
  mutable IAllocatorUniquePtr<void> swap_cuda_staging_;
  mutable std::mutex input_copy_mutex_;
#endif
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
