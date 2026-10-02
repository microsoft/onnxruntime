// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "contrib_ops/cuda/moe/moe_base.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_profiler.h"
#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"

#include <array>
#include <mutex>
#include <vector>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class MoE final : public CudaKernel, public MoEBase {
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
  struct PackedTensor {
    TensorShape shape;
    IAllocatorUniquePtr<void> cpu_data;
    IAllocatorUniquePtr<void> cuda_data;
    size_t bytes{0};
    bool present{false};
  };

#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  Status InitializeCudaExpertWeights(gsl::span<const int> cuda_experts);
#endif

  mutable onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler mGemmProfiler;
  mutable std::mutex mGemmProfilerMutex;
  bool cpu_offload_enabled_{false};
  AllocatorPtr cpu_allocator_;
  AllocatorPtr cuda_allocator_;
  std::array<PackedTensor, 8> packed_inputs_;
  InlinedVector<int> cuda_experts_;
  InlinedVector<int> expert_map_;
  IAllocatorUniquePtr<void> device_expert_map_;
#if !defined(BUILD_CUDA_EP_AS_PLUGIN) && !defined(ORT_MINIMAL_BUILD)
  cudaStream_t input_copy_stream_{nullptr};
  mutable std::mutex input_copy_mutex_;
#endif
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
