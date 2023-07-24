// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_provider_factory.h"
#include "core/providers/cuda/cuda_provider_factory_creator.h"
#include "core/providers/cuda/cuda_provider_options.h"

#include <memory>
#include <chrono>

#include "core/common/gsl.h"
#include "gtest/gtest.h"

#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/gpu_data_transfer.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"

#ifdef ENABLE_NVTX_PROFILE
#include "core/providers/cuda/nvtx_profile.h"
#endif

using namespace onnxruntime;

#if defined(USE_CUDA) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P) && defined(ENABLE_TRAINING)
namespace cuda {
cuda::INcclService& GetINcclService();
}
#endif

namespace onnxruntime {

void InitializeRegistry();
void DeleteRegistry();

struct ProviderInfo_CUDA_TestImpl : ProviderInfo_CUDA {
  OrtStatus* SetCurrentGpuDeviceId(_In_ int) override {
    return nullptr;
  }

  OrtStatus* GetCurrentGpuDeviceId(_In_ int*) override {
    return nullptr;
  }

  std::unique_ptr<IAllocator> CreateCUDAAllocator(int16_t, const char*) override {
    return nullptr;
  }

  std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(const char*) override {
    return nullptr;
  }

  std::unique_ptr<IDataTransfer> CreateGPUDataTransfer() override {
    return nullptr;
  }

  void cuda__Impl_Cast(void*, const int64_t*, int32_t*, size_t) override {}

  void cuda__Impl_Cast(void*, const int32_t*, int64_t*, size_t) override {}

  void cuda__Impl_Cast(void*, const double*, float*, size_t) override {}

  void cuda__Impl_Cast(void*, const float*, double*, size_t) override {}

  Status CudaCall_false(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) override { return CudaCall<cudaError, false>(cudaError(retCode), exprString, libName, cudaError(successCode), msg, file, line); }
  void CudaCall_true(int retCode, const char* exprString, const char* libName, int successCode, const char* msg, const char* file, const int line) override { CudaCall<cudaError, true>(cudaError(retCode), exprString, libName, cudaError(successCode), msg, file, line); }

  void CopyGpuToCpu(void*, const void*, const size_t, const OrtMemoryInfo&, const OrtMemoryInfo&) override {}

  void cudaMemcpy_HostToDevice(void*, const void*, size_t) override {}

  // Used by onnxruntime_pybind_state.cc
  void cudaMemcpy_DeviceToHost(void*, const void*, size_t) override {}

  int cudaGetDeviceCount() override { return 0; }

  void CUDAExecutionProviderInfo__FromProviderOptions(const ProviderOptions&, CUDAExecutionProviderInfo&) override {}

#if defined(USE_CUDA) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P) && defined(ENABLE_TRAINING)
  cuda::INcclService& GetINcclService() override {
    return cuda::GetINcclService();
  }
#endif

#ifdef ENABLE_NVTX_PROFILE
  void NvtxRangeCreator__BeginImpl(profile::NvtxRangeCreator* p) override { p->BeginImpl(); }
  void NvtxRangeCreator__EndImpl(profile::NvtxRangeCreator* p) override { p->EndImpl(); }
#endif

  std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory(const CUDAExecutionProviderInfo&) override {
    return nullptr;
  }

  std::shared_ptr<IAllocator> CreateCudaAllocator(int16_t, size_t, onnxruntime::ArenaExtendStrategy, onnxruntime::CUDAExecutionProviderExternalAllocatorInfo&, const OrtArenaCfg*) override {
    return nullptr;
  }

  void TestAll() override {
    // TestAll is the entry point of CUDA EP's insternal tests.
    // Those internal tests are not directly callable from onnxruntime_test_all
    // because CUDA EP is a shared library now.
    // Instead, this is a test provider that implements all the test cases.
    // onnxruntime_test_all is calling this function through TryGetProviderInfo_CUDA_Test.
    int argc = 1;
    std::string mock_exe_name = "onnxruntime_providers_cuda_ut";
    char* argv[] = {const_cast<char*>(mock_exe_name.data())};
    ::testing::InitGoogleTest(&argc, argv);
    ORT_ENFORCE(RUN_ALL_TESTS() == 0);
  }
};
ProviderInfo_CUDA_TestImpl g_test_info;

struct CUDA_Test_Provider : Provider {
  void* GetInfo() override { return &g_test_info; }

  void Initialize() override {
    InitializeRegistry();
  }

  void Shutdown() override {
    DeleteRegistry();
  }
};

CUDA_Test_Provider g_test_provider;

}  // namespace onnxruntime

extern "C" {
// This is the entry point of libonnxruntime_providers_cuda_ut.so/dll.
ORT_API(onnxruntime::Provider*, GetProvider) {
  return &onnxruntime::g_test_provider;
}
}
