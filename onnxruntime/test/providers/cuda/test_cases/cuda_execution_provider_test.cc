// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This test is built only under DEBUG mode because it requires
// extra code in the core of CUDA EP and that code may
//  1. slow down performance critical applications and
//  2. increase binary size of ORT.

#include "gtest/gtest.h"
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include <cuda_runtime_api.h>

#include "core/framework/run_options.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/providers/cuda/cuda_stream_handle.h"

namespace onnxruntime {
namespace cuda {
namespace test {
// TODO: Since the "DeferredRelease" has been migrated to CudaStream class,
// we should migrate this test from CudaEP unit test to CudaStream unit test.
TEST(TestDeferredRelease, WithArena) {
  // Create CUDA EP.
  CUDAExecutionProviderInfo info;
  CUDAExecutionProvider ep(info);
  AllocatorPtr gpu_allocator = ep.CreatePreferredAllocators()[0];

  onnxruntime::RunOptions run_opts;
  run_opts.run_tag = "log1";
  // Allocator for call cudaMallocHost and cudaFreeHost
  // For details, see CUDAPinnedAllocator in cuda_allocator.cc.
  AllocatorPtr cpu_pinned_alloc = ep.CreatePreferredAllocators()[1];
  // let the CudaStream instance "own" the default stream, so we can avoid the
  // work to initialize cublas/cudnn/... It is ok since it is just a customized unit test.
  CudaStream stream(nullptr, gpu_allocator->Info().device, cpu_pinned_alloc, false, true, nullptr, nullptr, info);
  // 10 MB
  const size_t n_bytes = 10 * 1000000;
  const int64_t n_allocs = 64;
  ORT_THROW_IF_ERROR(ep.OnRunStart(run_opts));
  for (size_t i = 0; i < n_allocs; ++i) {
    // Allocate 10MB CUDA pinned memory.
    auto pinned_buffer = IAllocator::MakeUniquePtr<void>(cpu_pinned_alloc, n_bytes);
    // Release it using CUDA callback.
    stream.EnqueDeferredCPUBuffer(pinned_buffer.release());
  }

  // Memory stats
  AllocatorStats stats;
  cpu_pinned_alloc->GetStats(&stats);
  ASSERT_EQ(stats.num_allocs, n_allocs);
  ORT_THROW_IF_ERROR(stream.CleanUpOnRunEnd());
  ORT_THROW_IF_ERROR(ep.OnRunEnd(true, run_opts));
}

TEST(TestDeferredRelease, WithoutArena) {
  // Create CUDA EP.
  CUDAExecutionProviderInfo info;
  CUDAExecutionProvider ep(info);

  onnxruntime::RunOptions run_opts;
  run_opts.run_tag = "log1";

  OrtDevice pinned_device{OrtDevice::GPU, OrtDevice::MemType::HOST_ACCESSIBLE, OrtDevice::VendorIds::NVIDIA,
                          info.device_id};
  // Create allocator without BFCArena
  AllocatorCreationInfo pinned_memory_info(
      [](OrtDevice::DeviceId device_id) {
        return std::make_unique<CUDAPinnedAllocator>(device_id, CUDA_PINNED);
      },
      pinned_device.Id(),
      false /* no arena */);
  auto cuda_pinned_alloc = CreateAllocator(pinned_memory_info);
  AllocatorPtr gpu_allocator = ep.CreatePreferredAllocators()[0];
  // Allocator for call cudaMallocHost and cudaFreeHost
  // For details, see CUDAPinnedAllocator in cuda_allocator.cc.
  // let the CudaStream instance "own" the default stream, so we can avoid the
  // work to initialize cublas/cudnn/... It is ok since it is just a customized unit test.
  CudaStream stream(nullptr, gpu_allocator->Info().device, cuda_pinned_alloc, false, true, nullptr, nullptr, info);
  // 10 MB
  const size_t n_bytes = 10 * 1000000;
  const int64_t n_allocs = 64;
  ORT_THROW_IF_ERROR(ep.OnRunStart(run_opts));
  for (size_t i = 0; i < n_allocs; ++i) {
    // Allocate 10MB CUDA pinned memory.
    auto pinned_buffer = IAllocator::MakeUniquePtr<void>(cuda_pinned_alloc, n_bytes);
    // Release it using CUDA callback.
    stream.EnqueDeferredCPUBuffer(pinned_buffer.release());
  }

  ORT_THROW_IF_ERROR(stream.CleanUpOnRunEnd());
  ORT_THROW_IF_ERROR(ep.OnRunEnd(true, run_opts));
}

namespace {

constexpr const char* kMinNumRunsKey = "min_num_runs_before_cuda_graph_capture";

std::unique_ptr<CUDAExecutionProvider> CreateGraphCaptureProvider(int min_num_runs_before_capture) {
  CUDAExecutionProviderInfo info;
  info.enable_cuda_graph = true;
  info.min_num_runs_before_cuda_graph_capture = min_num_runs_before_capture;
  auto ep = std::make_unique<CUDAExecutionProvider>(info);
  // Starting a capture logs through the provider's logger, which a session normally sets.
  ep->SetLogger(&logging::LoggingManager::DefaultLogger());
  return ep;
}

// InferenceSession reports a run that did not complete to the providers through this run config entry.
onnxruntime::RunOptions RunOptionsForOutcome(bool run_failed) {
  onnxruntime::RunOptions run_options;
  if (run_failed) {
    ORT_THROW_IF_ERROR(run_options.config_options.AddConfigEntry("__ort.run_outcome", "failed"));
  }
  return run_options;
}

bool IsCapturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  EXPECT_EQ(cudaStreamIsCapturing(stream, &status), cudaSuccess);
  return status != cudaStreamCaptureStatusNone;
}

// Runs the provider once the way InferenceSession::Run does: OnRunStart, the graph executes (`enqueue_work` issues its
// work on the provider's stream), then OnRunEnd. Returns whether OnRunStart began a graph capture.
bool RunProviderOnce(CUDAExecutionProvider& ep, bool run_failed,
                     const std::function<void(cudaStream_t)>& enqueue_work = {}) {
  ORT_THROW_IF_ERROR(ep.OnRunStart(onnxruntime::RunOptions{}));
  const bool capturing = IsCapturing(ep.ComputeStream());
  if (enqueue_work) {
    enqueue_work(ep.ComputeStream());
  }
  ORT_THROW_IF_ERROR(ep.OnRunEnd(/*sync_stream=*/true, RunOptionsForOutcome(run_failed)));
  return capturing;
}

class ZeroedDeviceBuffer {
 public:
  explicit ZeroedDeviceBuffer(size_t bytes) : bytes_(bytes) {
    ORT_ENFORCE(cudaMalloc(&data_, bytes_) == cudaSuccess);
    ORT_ENFORCE(cudaMemset(data_, 0, bytes_) == cudaSuccess);
  }
  ~ZeroedDeviceBuffer() { static_cast<void>(cudaFree(data_)); }
  ZeroedDeviceBuffer(const ZeroedDeviceBuffer&) = delete;
  ZeroedDeviceBuffer& operator=(const ZeroedDeviceBuffer&) = delete;

  void FillAsync(uint8_t value, cudaStream_t stream) const {
    EXPECT_EQ(cudaMemsetAsync(data_, value, bytes_, stream), cudaSuccess);
  }

  std::vector<uint8_t> Read() const {
    std::vector<uint8_t> host(bytes_);
    ORT_ENFORCE(cudaMemcpy(host.data(), data_, bytes_, cudaMemcpyDeviceToHost) == cudaSuccess);
    return host;
  }

 private:
  size_t bytes_;
  void* data_ = nullptr;
};

constexpr size_t kBufferBytes = 64;

}  // namespace

// A warm-up run that failed did not execute the whole graph, so it must not bring the first capture closer.
TEST(CudaGraphCaptureRunOutcomeTest, FailedWarmupRunsDoNotCountTowardsCapture) {
  const auto ep_owner = CreateGraphCaptureProvider(/*min_num_runs_before_capture=*/2);
  CUDAExecutionProvider& ep = *ep_owner;
  ZeroedDeviceBuffer buffer(kBufferBytes);
  const auto enqueue_work = [&](cudaStream_t stream) { buffer.FillAsync(0x11, stream); };

  std::vector<bool> capture_started;
  for (const bool run_failed : {true, false, true, false, false}) {
    capture_started.push_back(RunProviderOnce(ep, run_failed, enqueue_work));
  }

  // Only the two successful runs count as warm-up, so the capture starts at the fifth run rather than the third.
  EXPECT_EQ(capture_started, (std::vector<bool>{false, false, false, false, true}));
  EXPECT_TRUE(ep.IsGraphCaptured(0));
  EXPECT_EQ(buffer.Read(), std::vector<uint8_t>(kBufferBytes, 0x11));
}

// Work recorded by a run that failed while capturing is incomplete. It must be dropped instead of being instantiated and
// replayed, and a later successful run must capture again.
TEST(CudaGraphCaptureRunOutcomeTest, FailedRunDuringCaptureIsDiscardedInsteadOfReplayed) {
  const auto ep_owner = CreateGraphCaptureProvider(/*min_num_runs_before_capture=*/1);
  CUDAExecutionProvider& ep = *ep_owner;
  ZeroedDeviceBuffer buffer(kBufferBytes);

  EXPECT_FALSE(RunProviderOnce(ep, /*run_failed=*/false));

  EXPECT_TRUE(RunProviderOnce(ep, /*run_failed=*/true,
                              [&](cudaStream_t stream) { buffer.FillAsync(0xCD, stream); }));
  EXPECT_FALSE(ep.IsGraphCaptured(0));
  EXPECT_FALSE(IsCapturing(ep.ComputeStream()));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_EQ(buffer.Read(), std::vector<uint8_t>(kBufferBytes, 0)) << "The recorded work of the failed run was replayed.";

  EXPECT_TRUE(RunProviderOnce(ep, /*run_failed=*/false,
                              [&](cudaStream_t stream) { buffer.FillAsync(0x5A, stream); }));
  EXPECT_TRUE(ep.IsGraphCaptured(0));
  EXPECT_EQ(buffer.Read(), std::vector<uint8_t>(kBufferBytes, 0x5A));
}

// A failed run can leave the capture invalidated. Discarding it must neither throw nor leave the CUDA error behind
// for unrelated calls to trip over.
TEST(CudaGraphCaptureRunOutcomeTest, InvalidatedCaptureOfFailedRunIsDiscardedWithoutLeavingAnError) {
  const auto ep_owner = CreateGraphCaptureProvider(/*min_num_runs_before_capture=*/1);
  CUDAExecutionProvider& ep = *ep_owner;

  EXPECT_FALSE(RunProviderOnce(ep, /*run_failed=*/false));

  EXPECT_TRUE(RunProviderOnce(ep, /*run_failed=*/true, [](cudaStream_t stream) {
    // Synchronizing a capturing stream is not allowed and invalidates the capture.
    EXPECT_NE(cudaStreamSynchronize(stream), cudaSuccess);
    static_cast<void>(cudaGetLastError());
  }));
  EXPECT_FALSE(IsCapturing(ep.ComputeStream()));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  EXPECT_FALSE(ep.IsGraphCaptured(0));

  EXPECT_TRUE(RunProviderOnce(ep, /*run_failed=*/false));
  EXPECT_TRUE(ep.IsGraphCaptured(0));
}

TEST(CudaProviderInfoTest, MinNumRunsBeforeCudaGraphCaptureIsParsedAndSerialized) {
  EXPECT_EQ(CUDAExecutionProviderInfo{}.min_num_runs_before_cuda_graph_capture, 2);

  const auto info = CUDAExecutionProviderInfo::FromProviderOptions({{kMinNumRunsKey, "8"}});
  EXPECT_EQ(info.min_num_runs_before_cuda_graph_capture, 8);

  const auto options = CUDAExecutionProviderInfo::ToProviderOptions(info);
  ASSERT_EQ(options.count(kMinNumRunsKey), 1u);
  EXPECT_EQ(options.at(kMinNumRunsKey), "8");
  EXPECT_EQ(CUDAExecutionProviderInfo::FromProviderOptions(options).min_num_runs_before_cuda_graph_capture, 8);

  CUDAExecutionProviderInfo other = info;
  other.min_num_runs_before_cuda_graph_capture = 3;
  EXPECT_NE(std::hash<CUDAExecutionProviderInfo>{}(info), std::hash<CUDAExecutionProviderInfo>{}(other));
}

TEST(CudaProviderInfoTest, InvalidMinNumRunsBeforeCudaGraphCaptureIsRejected) {
  EXPECT_THROW(CUDAExecutionProviderInfo::FromProviderOptions({{kMinNumRunsKey, "-1"}}), std::exception);
  EXPECT_THROW(CUDAExecutionProviderInfo::FromProviderOptions({{kMinNumRunsKey, "many"}}), std::exception);
  EXPECT_EQ(CUDAExecutionProviderInfo::FromProviderOptions({{kMinNumRunsKey, "0"}}).min_num_runs_before_cuda_graph_capture,
            0);
}

TEST(CudaProviderInfoTest, V2ProviderOptionsCarryMinNumRunsBeforeCudaGraphCapture) {
  OrtCUDAProviderOptionsV2 v2_options;
  EXPECT_EQ(v2_options.min_num_runs_before_cuda_graph_capture, 2);

  auto options = CUDAExecutionProviderInfo::ToProviderOptions(v2_options);
  ASSERT_EQ(options.count(kMinNumRunsKey), 1u);
  EXPECT_EQ(options.at(kMinNumRunsKey), "2");

  v2_options.min_num_runs_before_cuda_graph_capture = 5;
  options = CUDAExecutionProviderInfo::ToProviderOptions(v2_options);
  ASSERT_EQ(options.count(kMinNumRunsKey), 1u);
  EXPECT_EQ(options.at(kMinNumRunsKey), "5");
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
