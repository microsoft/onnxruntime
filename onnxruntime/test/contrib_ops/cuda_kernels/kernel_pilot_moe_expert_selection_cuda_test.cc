// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <array>
#include <chrono>
#include <condition_variable>
#include <mutex>

#include "contrib_ops/cuda/moe/kernel_pilot_moe_expert_selection_cuda.h"
#include "contrib_ops/cuda/moe/moe_kernels.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "gtest/gtest.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {
namespace {

class StreamGate {
 public:
  static void CUDART_CB Wait(void* context) {
    auto& gate = *static_cast<StreamGate*>(context);
    std::unique_lock lock(gate.mutex_);
    // A stream-wide wait in Consume must fail the test rather than hang the test process.
    gate.timed_out = !gate.condition_.wait_for(lock, std::chrono::seconds(10), [&gate]() {
      return gate.released_;
    });
  }

  void Release() {
    std::lock_guard lock(mutex_);
    released_ = true;
    condition_.notify_one();
  }

  std::atomic<bool> timed_out{false};

 private:
  std::mutex mutex_;
  std::condition_variable condition_;
  bool released_{false};
};

class KernelPilotMoeExpertSelectionCudaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
    if (device_count == 0) {
      GTEST_SKIP() << "No CUDA device is available.";
    }
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&device_ids_, 8 * sizeof(int)), cudaSuccess);
  }

  void TearDown() override {
    if (stream_ != nullptr) {
      EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
      EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
    }
    if (device_ids_ != nullptr) {
      EXPECT_EQ(cudaFree(device_ids_), cudaSuccess);
    }
  }

  Status Capture(contrib::cuda::KernelPilotMoeExpertSelectionCuda& snapshot, gsl::span<const int> ids) {
    ORT_RETURN_IF(ids.size() > 8, "Test routing buffer is too small.");
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(device_ids_, ids.data(), ids.size_bytes(),
                                         cudaMemcpyHostToDevice, stream_));
    return snapshot.Capture(device_ids_, ids.size(), stream_);
  }

  AllocatorPtr PinnedAllocator() {
    int device = 0;
    CUDA_CALL_THROW(cudaGetDevice(&device));
    return std::make_shared<CUDAPinnedAllocator>(static_cast<OrtDevice::DeviceId>(device), CUDA_PINNED);
  }

  void ExpectSelectedExperts(const IKernelPilotMoeExpertSelection& usage,
                             const InlinedVector<int>& expected) {
    gsl::span<const int> selected;
    ASSERT_STATUS_OK(usage.GetSelectedExperts(selected));
    EXPECT_EQ((InlinedVector<int>{selected.begin(), selected.end()}), expected);
  }

  cudaStream_t stream_{nullptr};
  int* device_ids_{nullptr};
};

TEST_F(KernelPilotMoeExpertSelectionCudaTest, GpuCountersMatchBinaryAndWeightedHostUpdates) {
  struct Snapshot {
    double counters[4];
    int64_t first_routes[4];
    float probabilities[4];
  };
  void* device_snapshot = nullptr;
  float* device_scales = nullptr;
  ASSERT_EQ(cudaMalloc(&device_snapshot, sizeof(Snapshot)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_scales, 4 * sizeof(float)), cudaSuccess);
  auto release = gsl::finally([&]() {
    EXPECT_EQ(cudaFree(device_snapshot), cudaSuccess);
    EXPECT_EQ(cudaFree(device_scales), cudaSuccess);
  });
  const std::array<int, 4> ids{1, 3, 1, 0};
  const std::array<float, 4> scales{0.2f, 0.7f, 0.8f, 0.1f};
  const std::array<double, 4> initial{0.4, 0.25, 0.6, 0.0};
  ASSERT_EQ(cudaMemcpyAsync(device_ids_, ids.data(), sizeof(ids), cudaMemcpyHostToDevice, stream_), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_scales, scales.data(), sizeof(scales), cudaMemcpyHostToDevice, stream_), cudaSuccess);
  auto* counters = static_cast<double*>(device_snapshot);
  auto* first_routes = reinterpret_cast<int64_t*>(counters + 4);
  auto* probabilities = reinterpret_cast<float*>(first_routes + 4);
  for (double alpha : {0.0, 0.9, 1.0}) {
    for (double weight : {0.0, 0.5, 1.0}) {
      SCOPED_TRACE(MakeString("alpha=", alpha, ", weight=", weight));
      ASSERT_EQ(cudaMemcpyAsync(counters, initial.data(), sizeof(initial), cudaMemcpyHostToDevice, stream_), cudaSuccess);
      contrib::cuda::LaunchUpdateMoeExpertCounters(device_ids_, device_scales, ids.size(),
                                                   counters, probabilities, first_routes, 4, alpha, weight, stream_);
      Snapshot snapshot;
      ASSERT_EQ(cudaMemcpyAsync(&snapshot, device_snapshot, sizeof(snapshot), cudaMemcpyDeviceToHost, stream_), cudaSuccess);
      ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
      const std::array<float, 4> expected_probabilities{0.1f, 0.8f, -1.0f, 0.7f};
      const std::array<int64_t, 4> expected_first_routes{3, 0, -1, 1};
      for (size_t expert = 0; expert < 4; ++expert) {
        const double contribution = expected_probabilities[expert] >= 0.0f
                                        ? (1.0 - weight) + weight * expected_probabilities[expert]
                                        : 0.0;
        EXPECT_NEAR(snapshot.counters[expert], alpha * initial[expert] + (1.0 - alpha) * contribution, 1e-12);
        EXPECT_EQ(snapshot.probabilities[expert], expected_probabilities[expert]);
        EXPECT_EQ(snapshot.first_routes[expert], expected_first_routes[expert]);
      }
    }
  }
}

TEST_F(KernelPilotMoeExpertSelectionCudaTest, EmptyDecodeCpuRoutingReleasesDeviceWaitWithoutHostWorker) {
#if CUDA_VERSION < 11070
  CUdevice device;
  ASSERT_STATUS_OK(CU_CALL(cuDeviceGet(&device, 0)));
  int supported = 0;
  ASSERT_STATUS_OK(CU_CALL(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS, device)));
  if (!supported) {
    GTEST_SKIP() << "CUDA stream memory operations are unavailable.";
  }
#endif
  auto completion = IAllocator::MakeUniquePtr<uint32_t>(PinnedAllocator(), 1);
  ASSERT_NE(completion, nullptr);
  *completion = 0;
  void* device_completion = nullptr;
  ASSERT_EQ(cudaHostGetDevicePointer(&device_completion, completion.get(), 0), cudaSuccess);
  half* device_input = nullptr;
  float* device_scales = nullptr;
  ASSERT_EQ(cudaMalloc(&device_input, 32 * sizeof(half)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&device_scales, 2 * sizeof(float)), cudaSuccess);
  auto release = gsl::finally([&]() {
    EXPECT_EQ(cudaFree(device_input), cudaSuccess);
    EXPECT_EQ(cudaFree(device_scales), cudaSuccess);
  });
  const std::array<int, 8> ids_map{0, 0, -1, 0, 0, 0, 0, 0};
  const float scale = 1.0f;
  ASSERT_EQ(cudaMemcpyAsync(device_ids_, ids_map.data(), sizeof(ids_map), cudaMemcpyHostToDevice, stream_), cudaSuccess);
  ASSERT_EQ(cudaMemcpyAsync(device_scales, &scale, sizeof(scale), cudaMemcpyHostToDevice, stream_), cudaSuccess);
  contrib::cuda::LaunchPackMoeCpuRows(device_input, device_ids_, device_scales, device_ids_ + 1,
                                      device_input + 16, device_ids_ + 5, device_scales + 1,
                                      device_ids_ + 4, reinterpret_cast<uint32_t*>(device_ids_ + 3),
                                      static_cast<uint32_t*>(device_completion), 1, 16, 1, stream_);
  ASSERT_STATUS_OK(CU_CALL(cuStreamWaitValue32(reinterpret_cast<CUstream>(stream_),
                                               reinterpret_cast<CUdeviceptr>(device_completion), 1,
                                               CU_STREAM_WAIT_VALUE_GEQ)));
  ASSERT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  EXPECT_EQ(*completion, 1U);
}

TEST_F(KernelPilotMoeExpertSelectionCudaTest, CollectsWhileLaterDeviceWorkIsPending) {
  KernelPilotMoeExpertSelection usage;
  contrib::cuda::KernelPilotMoeExpertSelectionCuda snapshot(PinnedAllocator());
  ASSERT_STATUS_OK(snapshot.BeginInvocation(usage, 4));
  EXPECT_TRUE(snapshot.IsInitialized());
  EXPECT_EQ(snapshot.ExpertCount(), 4U);
  const InlinedVector<int> ids{0, 0, 2};
  ASSERT_STATUS_OK(Capture(snapshot, ids));

  StreamGate gate;
  auto release_gate = gsl::finally([&]() {
    gate.Release();
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  });
  cudaEvent_t expert_done{};
  ASSERT_EQ(cudaEventCreateWithFlags(&expert_done, cudaEventDisableTiming), cudaSuccess);
  auto destroy_event = gsl::finally([&]() { EXPECT_EQ(cudaEventDestroy(expert_done), cudaSuccess); });
  ASSERT_EQ(cudaLaunchHostFunc(stream_, StreamGate::Wait, &gate), cudaSuccess);
  ASSERT_EQ(cudaMemsetAsync(device_ids_, 0, ids.size() * sizeof(int), stream_), cudaSuccess);
  ASSERT_EQ(cudaEventRecord(expert_done, stream_), cudaSuccess);

  ASSERT_STATUS_OK(snapshot.Consume());
  ExpectSelectedExperts(snapshot, {0, 2});
  EXPECT_FALSE(gate.timed_out);
  EXPECT_EQ(cudaEventQuery(expert_done), cudaErrorNotReady);
}

TEST_F(KernelPilotMoeExpertSelectionCudaTest, EnqueuesWithoutWaitingForRouting) {
  KernelPilotMoeExpertSelection usage;
  contrib::cuda::KernelPilotMoeExpertSelectionCuda snapshot(PinnedAllocator());
  const InlinedVector<int> ids{1, 3};
  ASSERT_STATUS_OK(snapshot.BeginInvocation(usage, 4));
  ASSERT_STATUS_OK(Capture(snapshot, ids));
  ASSERT_STATUS_OK(snapshot.Consume());

  ASSERT_STATUS_OK(snapshot.BeginInvocation(usage, 4));
  StreamGate gate;
  auto release_gate = gsl::finally([&]() {
    gate.Release();
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  });
  ASSERT_EQ(cudaLaunchHostFunc(stream_, StreamGate::Wait, &gate), cudaSuccess);
  ASSERT_STATUS_OK(snapshot.Capture(device_ids_, ids.size(), stream_));
  EXPECT_FALSE(gate.timed_out);
  EXPECT_EQ(cudaStreamQuery(stream_), cudaErrorNotReady);
  gate.Release();
  ASSERT_STATUS_OK(snapshot.Consume());
  ExpectSelectedExperts(usage, ids);
}

TEST_F(KernelPilotMoeExpertSelectionCudaTest, UnionsTilesAndReusesBuffersAcrossInvocations) {
  KernelPilotMoeExpertSelection usage;
  contrib::cuda::KernelPilotMoeExpertSelectionCuda snapshot(PinnedAllocator());
  ASSERT_STATUS_OK(snapshot.BeginInvocation(usage, 4));
  for (const InlinedVector<int>& ids : {InlinedVector<int>{0, 0, 2}, InlinedVector<int>{2, 3}}) {
    ASSERT_STATUS_OK(Capture(snapshot, ids));
    ASSERT_EQ(cudaMemsetAsync(device_ids_, 0, ids.size() * sizeof(int), stream_), cudaSuccess);
    ASSERT_STATUS_OK(snapshot.Consume());
  }
  ExpectSelectedExperts(usage, {0, 2, 3});

  for (const InlinedVector<int>& ids : {InlinedVector<int>{1, 1, 1, 1, 1}, InlinedVector<int>{1}}) {
    ASSERT_STATUS_OK(snapshot.BeginInvocation(usage, 4));
    ASSERT_STATUS_OK(Capture(snapshot, ids));
    ASSERT_STATUS_OK(snapshot.Consume());
    ExpectSelectedExperts(usage, {1});
  }
}

TEST_F(KernelPilotMoeExpertSelectionCudaTest, DiscardsAnUnconsumedSnapshotAfterAnAbortedInvocation) {
  KernelPilotMoeExpertSelection usage;
  contrib::cuda::KernelPilotMoeExpertSelectionCuda snapshot(PinnedAllocator());
  ASSERT_STATUS_OK(snapshot.BeginInvocation(usage, 4));
  const InlinedVector<int> previous{0, 2};
  ASSERT_STATUS_OK(Capture(snapshot, previous));
  EXPECT_FALSE(snapshot.Capture(device_ids_, previous.size(), stream_).IsOK());
  ExpectSelectedExperts(usage, {});
  ASSERT_STATUS_OK(snapshot.BeginInvocation(usage, 4));
  const InlinedVector<int> current{1, 3, 1};
  ASSERT_STATUS_OK(Capture(snapshot, current));
  ASSERT_STATUS_OK(snapshot.Consume());
  ExpectSelectedExperts(usage, {1, 3});
}

TEST_F(KernelPilotMoeExpertSelectionCudaTest, RejectsInvalidRoutingAndRecovers) {
  KernelPilotMoeExpertSelection usage;
  contrib::cuda::KernelPilotMoeExpertSelectionCuda snapshot(PinnedAllocator());
  for (const int invalid : {-1, 4}) {
    ASSERT_STATUS_OK(snapshot.BeginInvocation(usage, 4));
    const InlinedVector<int> ids{0, invalid};
    ASSERT_STATUS_OK(Capture(snapshot, ids));
    EXPECT_FALSE(snapshot.Consume().IsOK());
    ExpectSelectedExperts(usage, {});
  }
  ASSERT_STATUS_OK(snapshot.BeginInvocation(usage, 4));
  const InlinedVector<int> ids{2};
  ASSERT_STATUS_OK(Capture(snapshot, ids));
  ASSERT_STATUS_OK(snapshot.Consume());
  ExpectSelectedExperts(usage, ids);
}

TEST_F(KernelPilotMoeExpertSelectionCudaTest, RejectsUninitializedCollection) {
  KernelPilotMoeExpertSelection usage;
  contrib::cuda::KernelPilotMoeExpertSelectionCuda snapshot(PinnedAllocator());
  EXPECT_FALSE(snapshot.BeginInvocation(4).IsOK());
  EXPECT_FALSE(snapshot.IsInitialized());
  EXPECT_EQ(snapshot.ExpertCount(), 0U);
  EXPECT_FALSE(snapshot.BeginInvocation(usage, 0).IsOK());
  EXPECT_FALSE(snapshot.Capture(device_ids_, 1, stream_).IsOK());
  EXPECT_FALSE(snapshot.Consume().IsOK());
}

}  // namespace
}  // namespace onnxruntime::test
