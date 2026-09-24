// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_external_data_loader.h"
#include "contrib_ops/cuda/bert/attention_kernel_options.h"
#include "contrib_ops/cuda/bert/group_query_attention_workspace_estimate.h"
#include "contrib_ops/cuda/bert/packed_attention_workspace_estimate.h"
#include "contrib_ops/cuda/quantization/matmul_nbits_workspace_estimate.h"
#include "test/providers/cuda/test_cases/matmul_nbits_workspace_test_probe.h"

namespace onnxruntime::test {

// Test-only entry points into the provider's header world. The includer supplies
// its own Node declaration, just as it does for the estimator headers.
struct CudaInternalTestApi {
  std::unique_ptr<IExternalDataLoader> (*create_external_data_loader)(
      int, size_t, cuda::ExternalDataLoader::AllocatePinnedBufferFn, cuda::ExternalDataLoader::CreateStreamFn);
#if !defined(USE_CUDA_MINIMAL)
  void (*initialize_attention_options)(AttentionKernelOptions&, int, bool, bool);

#if !defined(DISABLE_CONTRIB_OPS)
  decltype(&contrib::cuda::BuildPackedAttentionProblem) build_packed_problem;
  decltype(&contrib::cuda::BuildPackedMultiHeadAttentionProblem) build_packed_mha_problem;
  decltype(&contrib::cuda::GetPackedAttentionWorkspaceRecipe) packed_recipe;
  decltype(&contrib::cuda::GetPackedMultiHeadAttentionWorkspaceRecipe) packed_mha_recipe;
  decltype(&contrib::cuda::GetPackedAttentionWorkspaceAggregateForBounds) packed_aggregate;
  decltype(&contrib::cuda::GetPackedMultiHeadAttentionWorkspaceAggregateForBounds) packed_mha_aggregate;
  decltype(&contrib::cuda::GetPackedAttentionReachableBackendsForBounds) packed_reachable_backends;
  decltype(&contrib::cuda::GetPackedMultiHeadAttentionReachableBackendsForBounds) packed_mha_reachable_backends;
  std::optional<contrib::cuda::PackedAttentionWorkspaceAggregate> (*estimate_packed_config)(
      const contrib::cuda::PackedAttentionWorkspaceEstimateConfig&, gsl::span<const WorkspaceInputShape>,
      const cudaDeviceProp&, const AttentionKernelOptions&);
  std::optional<contrib::cuda::PackedAttentionWorkspaceAggregate> (*estimate_packed_node)(
      const Node&, gsl::span<const WorkspaceInputShape>, const cudaDeviceProp&, const AttentionKernelOptions&);
  decltype(&contrib::cuda::SetPackedAttentionWorkspaceRequirements) set_packed_requirements;

  decltype(&contrib::cuda::GetGQAFlashWorkspaceRecipe) gqa_flash_recipe;
  decltype(&contrib::cuda::GetGQACompleteWorkspaceRecipe) gqa_complete_recipe;
  decltype(&contrib::cuda::GetGQAWorkspaceAggregateForBounds) gqa_aggregate;
  std::optional<contrib::cuda::GQAWorkspaceAggregate> (*estimate_gqa_config)(
      const contrib::cuda::GQAWorkspaceEstimateConfig&, gsl::span<const WorkspaceInputShape>,
      const cudaDeviceProp&, const AttentionKernelOptions&);
  std::optional<contrib::cuda::GQAWorkspaceAggregate> (*estimate_gqa_node)(
      const Node&, gsl::span<const WorkspaceInputShape>, const cudaDeviceProp&, const AttentionKernelOptions&, bool);
  decltype(&contrib::cuda::SetGroupQueryAttentionWorkspaceRequirements) set_gqa_requirements;
  decltype(&contrib::cuda::SetGroupQueryAttentionLevel1MemoryEstimate) set_gqa_level1_estimate;

#if defined(USE_FPA_INTB_GEMM) && USE_FPA_INTB_GEMM
  decltype(&contrib::cuda::ComputeMatMulNBitsLeadingDimProduct) matmul_leading_dim_product;
  std::optional<Level1MemoryEstimate> (*estimate_matmul_node)(
      const Node&, const cudaDeviceProp&, contrib::cuda::MatMulNBitsMemoryEstimateOptions);
  std::optional<Level1MemoryEstimate> (*estimate_matmul_shape)(
      const Node&, gsl::span<const int64_t>, const cudaDeviceProp&, contrib::cuda::MatMulNBitsMemoryEstimateOptions);
  std::optional<size_t> (*matmul_workspace_node)(const Node&, const cudaDeviceProp&);
  std::optional<size_t> (*matmul_workspace_shape)(const Node&, gsl::span<const int64_t>, const cudaDeviceProp&);
  decltype(&GetMatMulNBitsLastComputeWorkspaceBytes) matmul_last_workspace_bytes;
  decltype(&GetMatMulNBitsLastComputeUsedPreallocatedWorkspace) matmul_last_used_preallocated;
#endif
#endif
#endif
};

}  // namespace onnxruntime::test
