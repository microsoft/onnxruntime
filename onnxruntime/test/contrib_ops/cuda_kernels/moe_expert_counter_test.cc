// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

#include "contrib_ops/cuda/moe/moe_expert_counter.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "gtest/gtest.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {
namespace {

class CollectedUsage final : public MoeExpertUsage {
 public:
  Status RecordUsage(gsl::span<const int> ids) override {
    selected.assign(ids.begin(), ids.end());
    ++records;
    return Status::OK();
  }

  Status GetCounters(InlinedVector<double>&) const override {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "This test only collects selected expert IDs.");
  }

  InlinedVector<int> selected;
  int records{0};
};

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

class CudaMoeExpertCounterTest : public ::testing::Test {
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

  Status Capture(contrib::cuda::CudaMoeExpertCounter& counter, gsl::span<const int> ids) {
    ORT_RETURN_IF(ids.size() > 8, "Test routing buffer is too small.");
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(device_ids_, ids.data(), ids.size_bytes(),
                                         cudaMemcpyHostToDevice, stream_));
    return counter.Capture(device_ids_, ids.size(), stream_);
  }

  AllocatorPtr PinnedAllocator() {
    int device = 0;
    CUDA_CALL_THROW(cudaGetDevice(&device));
    return std::make_shared<CUDAPinnedAllocator>(static_cast<OrtDevice::DeviceId>(device), CUDA_PINNED);
  }

  cudaStream_t stream_{nullptr};
  int* device_ids_{nullptr};
};

TEST_F(CudaMoeExpertCounterTest, RecordsWhileLaterDeviceWorkIsPending) {
  CollectedUsage usage;
  contrib::cuda::CudaMoeExpertCounter counter(PinnedAllocator());
  ASSERT_STATUS_OK(counter.BeginInvocation(usage, 4));
  const InlinedVector<int> ids{0, 0, 2};
  ASSERT_STATUS_OK(Capture(counter, ids));

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

  ASSERT_STATUS_OK(counter.Consume());
  ASSERT_STATUS_OK(counter.Record());
  EXPECT_EQ(usage.selected, (InlinedVector<int>{0, 2}));
  EXPECT_EQ(usage.records, 1);
  EXPECT_FALSE(gate.timed_out);
  EXPECT_EQ(cudaEventQuery(expert_done), cudaErrorNotReady);
}

TEST_F(CudaMoeExpertCounterTest, EnqueuesWithoutWaitingForRouting) {
  CollectedUsage usage;
  contrib::cuda::CudaMoeExpertCounter counter(PinnedAllocator());
  const InlinedVector<int> ids{1, 3};
  ASSERT_STATUS_OK(counter.BeginInvocation(usage, 4));
  ASSERT_STATUS_OK(Capture(counter, ids));
  ASSERT_STATUS_OK(counter.Consume());

  ASSERT_STATUS_OK(counter.BeginInvocation(usage, 4));
  StreamGate gate;
  auto release_gate = gsl::finally([&]() {
    gate.Release();
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  });
  ASSERT_EQ(cudaLaunchHostFunc(stream_, StreamGate::Wait, &gate), cudaSuccess);
  ASSERT_STATUS_OK(counter.Capture(device_ids_, ids.size(), stream_));
  EXPECT_FALSE(gate.timed_out);
  EXPECT_EQ(cudaStreamQuery(stream_), cudaErrorNotReady);
  gate.Release();
  ASSERT_STATUS_OK(counter.Consume());
  ASSERT_STATUS_OK(counter.Record());
  EXPECT_EQ(usage.selected, ids);
  EXPECT_EQ(usage.records, 1);
}

TEST_F(CudaMoeExpertCounterTest, UnionsTilesAndReusesBuffersAcrossInvocations) {
  CollectedUsage usage;
  contrib::cuda::CudaMoeExpertCounter counter(PinnedAllocator());
  ASSERT_STATUS_OK(counter.BeginInvocation(usage, 4));
  for (const InlinedVector<int>& ids : {InlinedVector<int>{0, 0, 2}, InlinedVector<int>{2, 3}}) {
    ASSERT_STATUS_OK(Capture(counter, ids));
    ASSERT_EQ(cudaMemsetAsync(device_ids_, 0, ids.size() * sizeof(int), stream_), cudaSuccess);
    ASSERT_STATUS_OK(counter.Consume());
    EXPECT_EQ(usage.records, 0);
  }
  ASSERT_STATUS_OK(counter.Record());
  EXPECT_EQ(usage.selected, (InlinedVector<int>{0, 2, 3}));
  EXPECT_EQ(usage.records, 1);

  for (const InlinedVector<int>& ids : {InlinedVector<int>{1, 1, 1, 1, 1}, InlinedVector<int>{1}}) {
    ASSERT_STATUS_OK(counter.BeginInvocation(usage, 4));
    ASSERT_STATUS_OK(Capture(counter, ids));
    ASSERT_STATUS_OK(counter.Consume());
    ASSERT_STATUS_OK(counter.Record());
    EXPECT_EQ(usage.selected, (InlinedVector<int>{1}));
  }
  EXPECT_EQ(usage.records, 3);
}

TEST_F(CudaMoeExpertCounterTest, DiscardsAnUnconsumedSnapshotAfterAnAbortedInvocation) {
  CollectedUsage usage;
  contrib::cuda::CudaMoeExpertCounter counter(PinnedAllocator());
  ASSERT_STATUS_OK(counter.BeginInvocation(usage, 4));
  const InlinedVector<int> previous{0, 2};
  ASSERT_STATUS_OK(Capture(counter, previous));
  EXPECT_FALSE(counter.Record().IsOK());
  ASSERT_STATUS_OK(counter.BeginInvocation(usage, 4));
  const InlinedVector<int> current{1, 3, 1};
  ASSERT_STATUS_OK(Capture(counter, current));
  ASSERT_STATUS_OK(counter.Consume());
  ASSERT_STATUS_OK(counter.Record());
  EXPECT_EQ(usage.selected, (InlinedVector<int>{1, 3}));
  EXPECT_EQ(usage.records, 1);
}

TEST_F(CudaMoeExpertCounterTest, RejectsInvalidRoutingAndRecovers) {
  CollectedUsage usage;
  contrib::cuda::CudaMoeExpertCounter counter(PinnedAllocator());
  for (const int invalid : {-1, 4}) {
    ASSERT_STATUS_OK(counter.BeginInvocation(usage, 4));
    const InlinedVector<int> ids{0, invalid};
    ASSERT_STATUS_OK(Capture(counter, ids));
    EXPECT_FALSE(counter.Consume().IsOK());
    EXPECT_EQ(usage.records, 0);
  }
  ASSERT_STATUS_OK(counter.BeginInvocation(usage, 4));
  const InlinedVector<int> ids{2};
  ASSERT_STATUS_OK(Capture(counter, ids));
  ASSERT_STATUS_OK(counter.Consume());
  ASSERT_STATUS_OK(counter.Record());
  EXPECT_EQ(usage.selected, ids);
}

}  // namespace
}  // namespace onnxruntime::test
