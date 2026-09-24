// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "test/providers/cuda/internal_testing/cuda_internal_test_api.h"

extern "C" __declspec(dllexport) const onnxruntime::test::CudaInternalTestApi* GetCudaInternalTestApi() {
  using namespace onnxruntime;
#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS)
  using namespace onnxruntime::contrib::cuda;
#endif
  static const test::CudaInternalTestApi api{
      [](int device, size_t readers, cuda::ExternalDataLoader::AllocatePinnedBufferFn allocate,
         cuda::ExternalDataLoader::CreateStreamFn create_stream) -> std::unique_ptr<IExternalDataLoader> {
        return std::make_unique<cuda::ExternalDataLoader>(device, readers, allocate, create_stream);
      },
#if !defined(USE_CUDA_MINIMAL)
      [](AttentionKernelOptions& options, int kernel, bool use_build_flag, bool check_cudnn_version) {
        options.InitializeOnce(kernel, use_build_flag, check_cudnn_version);
      },
#if !defined(DISABLE_CONTRIB_OPS)
      BuildPackedAttentionProblem,
      BuildPackedMultiHeadAttentionProblem,
      GetPackedAttentionWorkspaceRecipe,
      GetPackedMultiHeadAttentionWorkspaceRecipe,
      GetPackedAttentionWorkspaceAggregateForBounds,
      GetPackedMultiHeadAttentionWorkspaceAggregateForBounds,
      GetPackedAttentionReachableBackendsForBounds,
      GetPackedMultiHeadAttentionReachableBackendsForBounds,
      EstimatePackedAttentionWorkspace,
      EstimatePackedAttentionWorkspace,
      SetPackedAttentionWorkspaceRequirements,
      GetGQAFlashWorkspaceRecipe,
      GetGQACompleteWorkspaceRecipe,
      GetGQAWorkspaceAggregateForBounds,
      EstimateGroupQueryAttentionWorkspace,
      EstimateGroupQueryAttentionWorkspace,
      SetGroupQueryAttentionWorkspaceRequirements,
      SetGroupQueryAttentionLevel1MemoryEstimate,
#if defined(USE_FPA_INTB_GEMM) && USE_FPA_INTB_GEMM
      ComputeMatMulNBitsLeadingDimProduct,
      EstimateMatMulNBitsMemory,
      EstimateMatMulNBitsMemory,
      EstimateMatMulNBitsWorkspace,
      EstimateMatMulNBitsWorkspace,
      test::GetMatMulNBitsLastComputeWorkspaceBytes,
      test::GetMatMulNBitsLastComputeUsedPreallocatedWorkspace,
#endif
#endif
#endif
  };
  return &api;
}
