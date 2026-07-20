// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Regression tests for GH issue #29713:
// On the CUDA EP, a graph whose weights are retained as in-memory-external OrtValue
// initializers can schedule a device copy whose source and destination are the same
// physical host buffer. The host->host branch of GPUDataTransfer::CopyTensor{,Async}
// must treat such a self-copy as a benign no-op (matching CPUDataTransfer and the
// GPU->GPU branch), instead of failing ORT_ENFORCE(dst_data != src_data) or issuing a
// cudaStreamSynchronize that is illegal during CUDA-graph capture.
//
// This file is compiled into the CUDA EP shared-library test module, so it uses the
// provider-bridge API (provider_api.h first, Tensor::Create) like cuda_test_provider.cc.

#include "core/providers/shared_library/provider_api.h"

#include "gtest/gtest.h"
#include <memory>

#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_execution_provider_info.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/cuda_stream_handle.h"
#include "core/providers/cuda/gpu_data_transfer.h"

namespace onnxruntime {
namespace cuda {
namespace test {

namespace {
// A HOST_ACCESSIBLE (pinned) tensor. Passing the same Tensor as both copy source and
// destination gives src_data == dst_data (a single physical allocation) — the aliased
// in-memory initializer condition from the issue.
// 64 * sizeof(float) = 256 bytes > kSmallTensorExternalDataThreshold (127).
constexpr int64_t kNumElements = 64;

std::unique_ptr<Tensor> MakePinnedTensor(const AllocatorPtr& pinned_alloc) {
  return Tensor::Create(DataTypeImpl::GetType<float>(), TensorShape{kNumElements}, pinned_alloc);
}
}  // namespace

// Variant A (sync): a same-buffer host->host copy must not fail ORT_ENFORCE(dst != src).
TEST(GPUDataTransferSameBufferTest, CopyTensorSelfCopyIsNoOp) {
  CUDAExecutionProviderInfo info;
  CUDAExecutionProvider ep(info);
  AllocatorPtr pinned_alloc = ep.CreatePreferredAllocators()[1];

  auto tensor = MakePinnedTensor(pinned_alloc);

  GPUDataTransfer data_transfer;
  // src and dst are the same Tensor -> src_data == dst_data.
  auto status = data_transfer.CopyTensor(*tensor, *tensor);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

// Variant A (async): a same-buffer host->host copy on a stream must not fail.
TEST(GPUDataTransferSameBufferTest, CopyTensorAsyncSelfCopyIsNoOp) {
  CUDAExecutionProviderInfo info;
  CUDAExecutionProvider ep(info);
  AllocatorPtr gpu_allocator = ep.CreatePreferredAllocators()[0];
  AllocatorPtr pinned_alloc = ep.CreatePreferredAllocators()[1];

  cudaStream_t cuda_stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  CudaStream stream(cuda_stream, gpu_allocator->Info().device, pinned_alloc, false, false, nullptr, nullptr, info);

  auto tensor = MakePinnedTensor(pinned_alloc);

  GPUDataTransfer data_transfer;
  auto status = data_transfer.CopyTensorAsync(*tensor, *tensor, stream);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  ASSERT_EQ(cudaStreamSynchronize(cuda_stream), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(cuda_stream), cudaSuccess);
}

// Variant B: a same-buffer host->host copy inside a CUDA-graph capture region must not
// issue the (capture-illegal) cudaStreamSynchronize, so capture stays valid.
TEST(GPUDataTransferSameBufferTest, CopyTensorAsyncSelfCopyDuringCaptureKeepsCaptureValid) {
  CUDAExecutionProviderInfo info;
  CUDAExecutionProvider ep(info);
  AllocatorPtr gpu_allocator = ep.CreatePreferredAllocators()[0];
  AllocatorPtr pinned_alloc = ep.CreatePreferredAllocators()[1];

  cudaStream_t cuda_stream = nullptr;
  ASSERT_EQ(cudaStreamCreate(&cuda_stream), cudaSuccess);
  CudaStream stream(cuda_stream, gpu_allocator->Info().device, pinned_alloc, false, false, nullptr, nullptr, info);

  auto tensor = MakePinnedTensor(pinned_alloc);

  GPUDataTransfer data_transfer;
  ASSERT_EQ(cudaStreamBeginCapture(cuda_stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
  auto status = data_transfer.CopyTensorAsync(*tensor, *tensor, stream);
  cudaGraph_t graph = nullptr;
  cudaError_t end_capture = cudaStreamEndCapture(cuda_stream, &graph);

  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(end_capture, cudaSuccess) << "capture was invalidated: " << cudaGetErrorString(end_capture);

  if (graph != nullptr) {
    cudaGraphDestroy(graph);
  }
  ASSERT_EQ(cudaStreamDestroy(cuda_stream), cudaSuccess);
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
