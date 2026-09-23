// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

#include "contrib_ops/cuda/moe/cuda_routing_snapshot.h"
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

class CudaRoutingSnapshotTest : public ::testing::Test {
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

  Status Capture(contrib::cuda::CudaRoutingSnapshot& snapshot, gsl::span<const int> ids) {
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

  void ExpectSelectedExperts(const KernelUsage& usage,
                             const InlinedVector<int>& expected) {
    gsl::span<const int> selected;
    ASSERT_STATUS_OK(usage.GetSelectedExperts(selected));
    EXPECT_EQ((InlinedVector<int>{selected.begin(), selected.end()}), expected);
  }

  cudaStream_t stream_{nullptr};
  int* device_ids_{nullptr};
};

TEST_F(CudaRoutingSnapshotTest, CollectsWhileLaterDeviceWorkIsPending) {
  KernelUsage usage;
  contrib::cuda::CudaRoutingSnapshot snapshot(PinnedAllocator());
  ASSERT_STATUS_OK(snapshot.BeginInvocation(usage, 4));
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
  ExpectSelectedExperts(usage, {0, 2});
  EXPECT_FALSE(gate.timed_out);
  EXPECT_EQ(cudaEventQuery(expert_done), cudaErrorNotReady);
}

TEST_F(CudaRoutingSnapshotTest, EnqueuesWithoutWaitingForRouting) {
  KernelUsage usage;
  contrib::cuda::CudaRoutingSnapshot snapshot(PinnedAllocator());
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

TEST_F(CudaRoutingSnapshotTest, UnionsTilesAndReusesBuffersAcrossInvocations) {
  KernelUsage usage;
  contrib::cuda::CudaRoutingSnapshot snapshot(PinnedAllocator());
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

TEST_F(CudaRoutingSnapshotTest, DiscardsAnUnconsumedSnapshotAfterAnAbortedInvocation) {
  KernelUsage usage;
  contrib::cuda::CudaRoutingSnapshot snapshot(PinnedAllocator());
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

TEST_F(CudaRoutingSnapshotTest, RejectsInvalidRoutingAndRecovers) {
  KernelUsage usage;
  contrib::cuda::CudaRoutingSnapshot snapshot(PinnedAllocator());
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

TEST_F(CudaRoutingSnapshotTest, RejectsUninitializedCollection) {
  KernelUsage usage;
  contrib::cuda::CudaRoutingSnapshot snapshot(PinnedAllocator());
  EXPECT_FALSE(snapshot.BeginInvocation(usage, 0).IsOK());
  EXPECT_FALSE(snapshot.Capture(device_ids_, 1, stream_).IsOK());
  EXPECT_FALSE(snapshot.Consume().IsOK());
}

}  // namespace
}  // namespace onnxruntime::test
