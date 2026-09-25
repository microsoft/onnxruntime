// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_external_data_loader.h"

namespace onnxruntime::test {

#ifdef _WIN32
std::shared_ptr<CUDAExecutionProvider> CreateCudaInternalTestExecutionProvider(const CUDAExecutionProviderInfo& info);
std::unique_ptr<IExternalDataLoader> CreateCudaInternalTestExternalDataLoader(
    int device, size_t readers,
    cuda::ExternalDataLoader::AllocatePinnedBufferFn allocate = cudaMallocHost,
    cuda::ExternalDataLoader::CreateStreamFn create_stream = cudaStreamCreateWithFlags);
#else
inline std::shared_ptr<CUDAExecutionProvider> CreateCudaInternalTestExecutionProvider(
    const CUDAExecutionProviderInfo& info) {
  return std::make_shared<CUDAExecutionProvider>(info);
}

inline std::unique_ptr<IExternalDataLoader> CreateCudaInternalTestExternalDataLoader(
    int device, size_t readers,
    cuda::ExternalDataLoader::AllocatePinnedBufferFn allocate = cudaMallocHost,
    cuda::ExternalDataLoader::CreateStreamFn create_stream = cudaStreamCreateWithFlags) {
  return std::make_unique<cuda::ExternalDataLoader>(device, readers, allocate, create_stream);
}
#endif

}  // namespace onnxruntime::test
