// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <atomic>
#include <chrono>
#include <future>
#include <utility>

#include <cuda_runtime_api.h>
#include <gsl/gsl>
#include "gtest/gtest.h"

#include "core/framework/execution_provider.h"
#include "core/providers/tensorrt/tensorrt_provider_options.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

struct ProducerGate {
  explicit ProducerGate(std::future<void> release_signal) : release(std::move(release_signal)) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ProducerGate);

  std::future<void> release;
  std::atomic<bool> timed_out{false};
};

void CUDART_CB WaitForProducerRelease(void* data) {
  auto& gate = *static_cast<ProducerGate*>(data);
  gate.timed_out = gate.release.wait_for(std::chrono::seconds(10)) != std::future_status::ready;
}

void TestSyncWaitsForExternalStream(int provider_device, bool use_user_stream) {
  int original_device;
  ASSERT_EQ(cudaGetDevice(&original_device), cudaSuccess);
  const auto restore_device = gsl::finally([&]() { EXPECT_EQ(cudaSetDevice(original_device), cudaSuccess); });
  ASSERT_EQ(cudaSetDevice(provider_device), cudaSuccess);

  cudaStream_t producer_stream;
  ASSERT_EQ(cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking), cudaSuccess);
  const auto destroy_producer_stream = gsl::finally([&]() { EXPECT_EQ(cudaStreamDestroy(producer_stream), cudaSuccess); });

  cudaStream_t compute_stream = nullptr;
  const auto destroy_compute_stream = gsl::finally([&]() {
    if (compute_stream) {
      EXPECT_EQ(cudaStreamDestroy(compute_stream), cudaSuccess);
    }
  });
  if (use_user_stream) {
    ASSERT_EQ(cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking), cudaSuccess);
  }

  OrtTensorRTProviderOptionsV2 options{};
  options.device_id = provider_device;
  options.has_user_compute_stream = use_user_stream;
  options.user_compute_stream = compute_stream;
  auto provider = TensorrtExecutionProviderWithOptions(&options);
  ASSERT_NE(provider, nullptr);

  cudaEvent_t producer_done;
  ASSERT_EQ(cudaEventCreateWithFlags(&producer_done, cudaEventDisableTiming), cudaSuccess);
  const auto destroy_event = gsl::finally([&]() { EXPECT_EQ(cudaEventDestroy(producer_done), cudaSuccess); });

  std::promise<void> release_producer;
  ProducerGate producer_gate{release_producer.get_future()};
  std::promise<void> sync_started;
  auto sync_started_future = sync_started.get_future();
  cudaError_t set_device_status = cudaSuccess;
  cudaError_t event_query_status = cudaSuccess;
  cudaError_t get_device_status = cudaSuccess;
  int device_after_sync = -1;
  std::future<Status> sync_result;
  {
    // Release pending work even on an assertion failure, before joining the Sync thread.
    const auto unblock_producer = gsl::finally([&]() {
      release_producer.set_value();
      EXPECT_EQ(cudaStreamSynchronize(producer_stream), cudaSuccess);
    });
    ASSERT_EQ(cudaLaunchHostFunc(producer_stream, WaitForProducerRelease, &producer_gate), cudaSuccess);
    ASSERT_EQ(cudaEventRecord(producer_done, producer_stream), cudaSuccess);
    ASSERT_EQ(cudaEventQuery(producer_done), cudaErrorNotReady);

    sync_result = std::async(std::launch::async, [&]() {
      // A new caller thread need not have the provider's device selected.
      set_device_status = cudaSetDevice(0);
      sync_started.set_value();
      const auto status = provider->Sync();
      event_query_status = cudaEventQuery(producer_done);
      get_device_status = cudaGetDevice(&device_after_sync);
      return status;
    });

    EXPECT_EQ(sync_started_future.wait_for(std::chrono::seconds(10)), std::future_status::ready);
    EXPECT_EQ(sync_result.wait_for(std::chrono::milliseconds(100)), std::future_status::timeout)
        << "Sync must wait for work on an external nonblocking stream";
  }
  EXPECT_FALSE(producer_gate.timed_out.load());
  EXPECT_EQ(sync_result.wait_for(std::chrono::seconds(10)), std::future_status::ready);
  const auto status = sync_result.get();
  EXPECT_EQ(set_device_status, cudaSuccess);
  EXPECT_EQ(event_query_status, cudaSuccess);
  EXPECT_EQ(get_device_status, cudaSuccess);
  EXPECT_EQ(device_after_sync, 0);
  ASSERT_STATUS_OK(status);
}

}  // namespace

class TensorrtExecutionProviderSyncTest : public testing::TestWithParam<bool> {};

TEST_P(TensorrtExecutionProviderSyncTest, WaitsForExternalStream) {
  TestSyncWaitsForExternalStream(0, GetParam());
}

TEST_P(TensorrtExecutionProviderSyncTest, SelectsProviderDeviceAndRestoresCallerDevice) {
  int device_count;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  if (device_count < 2) {
    GTEST_SKIP() << "Requires two CUDA devices";
  }
  TestSyncWaitsForExternalStream(1, GetParam());
}

INSTANTIATE_TEST_SUITE_P(TensorrtExecutionProviderTests, TensorrtExecutionProviderSyncTest,
                         testing::Values(false, true));

TEST(TensorrtExecutionProviderTest, SyncReportsActiveCaptureError) {
  int original_device;
  ASSERT_EQ(cudaGetDevice(&original_device), cudaSuccess);
  const auto restore_device = gsl::finally([&]() { EXPECT_EQ(cudaSetDevice(original_device), cudaSuccess); });
  auto provider = DefaultTensorrtExecutionProvider();
  ASSERT_NE(provider, nullptr);

  cudaStream_t stream;
  ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
  const auto destroy_stream = gsl::finally([&]() { EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess); });
  ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), cudaSuccess);

  const auto status = provider->Sync();
  cudaGraph_t graph = nullptr;
  const auto end_capture_status = cudaStreamEndCapture(stream, &graph);
  if (graph) {
    EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  }
  EXPECT_STATUS_NOT_OK_AND_HAS_SUBSTR(status, "cudaDeviceSynchronize");
  EXPECT_EQ(end_capture_status, cudaErrorStreamCaptureInvalidated);
  // Clear the expected capture error so subsequent tests can use this CUDA context.
  cudaGetLastError();
  ASSERT_STATUS_OK(provider->Sync());
}

}  // namespace test
}  // namespace onnxruntime
