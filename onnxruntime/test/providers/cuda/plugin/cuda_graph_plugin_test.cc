// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Host-only unit tests for CUDA plugin graph retirement (CudaGraphSet,
// CudaGraphManager, and CudaEpGraphController).
//
// Limitation: these tests use host mock callbacks (DestroyGraphExec, SyncStream)
// to validate handle ownership, stream synchronization ordering, error
// propagation, per-ID warm-up bookkeeping, skip-ID (-1) exclusion, active-capture
// rejection, non-lazy context lookup, and thread-ownership enforcement without
// a CUDA device; physical GPU stream capture, graph instantiation, replay, and
// device memory reclamation require a CUDA device test.

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "core/providers/cuda/plugin/cuda_graph_plugin.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace test {

std::optional<std::string> ReadEnvironmentVariableOptional(const char* name);
bool IsCudaVisibleDevicesExplicitlyDisabled(const std::optional<std::string>& cuda_visible_devices);
bool ShouldAutoRegisterCudaPluginEp(const std::optional<std::string>& dynamic_plugin_ep_config_json,
                                    const std::optional<std::string>& cuda_visible_devices);

namespace {

struct MockCudaGraphCallbacks {
  static inline std::vector<cudaGraphExec_t> destroyed_handles;
  static inline std::vector<cudaStream_t> synced_streams;
  static inline std::vector<cudaStream_t> queried_streams;
  static inline std::vector<cudaGraphExec_t> launched_handles;
  static inline std::vector<cudaGraph_t> destroyed_graphs;
  static inline cudaError_t next_sync_result = cudaSuccess;
  static inline cudaError_t next_query_result = cudaSuccess;
  static inline cudaError_t next_launch_result = cudaSuccess;
  static inline cudaError_t next_destroy_result = cudaSuccess;
  static inline cudaError_t next_end_capture_result = cudaSuccess;
  static inline cudaGraph_t next_end_capture_graph = nullptr;
  static inline size_t end_capture_calls = 0;
  static inline size_t get_last_error_calls = 0;

  static void Reset() {
    destroyed_handles.clear();
    synced_streams.clear();
    queried_streams.clear();
    launched_handles.clear();
    destroyed_graphs.clear();
    next_sync_result = cudaSuccess;
    next_query_result = cudaSuccess;
    next_launch_result = cudaSuccess;
    next_destroy_result = cudaSuccess;
    next_end_capture_result = cudaSuccess;
    next_end_capture_graph = nullptr;
    end_capture_calls = 0;
    get_last_error_calls = 0;
  }

  static cudaError_t DestroyGraphExec(cudaGraphExec_t exec) {
    destroyed_handles.push_back(exec);
    return next_destroy_result;
  }

  static cudaError_t SyncStream(cudaStream_t stream) {
    synced_streams.push_back(stream);
    return next_sync_result;
  }

  static cudaError_t QueryStream(cudaStream_t stream) {
    queried_streams.push_back(stream);
    return next_query_result;
  }

  static cudaError_t LaunchGraph(cudaGraphExec_t exec, cudaStream_t /*stream*/) {
    launched_handles.push_back(exec);
    return next_launch_result;
  }

  static cudaError_t EndCapture(cudaStream_t /*stream*/, cudaGraph_t* out_graph) {
    ++end_capture_calls;
    if (out_graph != nullptr) {
      *out_graph = next_end_capture_graph;
    }
    return next_end_capture_result;
  }

  static cudaError_t DestroyGraph(cudaGraph_t graph) {
    destroyed_graphs.push_back(graph);
    return cudaSuccess;
  }

  static cudaError_t GetLastError() {
    ++get_last_error_calls;
    return cudaSuccess;
  }
};

struct ThreadSafeMockCudaGraphCallbacks {
  static inline std::mutex mutex;
  static inline std::vector<cudaGraphExec_t> destroyed_handles;
  static inline std::vector<cudaStream_t> synced_streams;

  static void Reset() {
    std::lock_guard<std::mutex> lock(mutex);
    destroyed_handles.clear();
    synced_streams.clear();
  }

  static cudaError_t DestroyGraphExec(cudaGraphExec_t exec) {
    std::lock_guard<std::mutex> lock(mutex);
    destroyed_handles.push_back(exec);
    return cudaSuccess;
  }

  static cudaError_t SyncStream(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(mutex);
    synced_streams.push_back(stream);
    return cudaSuccess;
  }

  static size_t DestroyCountForHandle(cudaGraphExec_t target) {
    std::lock_guard<std::mutex> lock(mutex);
    return static_cast<size_t>(
        std::count(destroyed_handles.begin(), destroyed_handles.end(), target));
  }

  static size_t TotalDestroyCount() {
    std::lock_guard<std::mutex> lock(mutex);
    return destroyed_handles.size();
  }
};

}  // namespace

TEST(CudaPluginGraphRetirementHostTest, ReleaseRetiresOwnedHandleExactlyOnceAndRemovesRunCount) {
  MockCudaGraphCallbacks::Reset();
  auto fake_stream = reinterpret_cast<cudaStream_t>(uintptr_t{0xCAFE0001});
  auto fake_exec_1 = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0001});
  auto fake_exec_1_recaptured = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0002});

  onnxruntime::cuda_plugin::CudaGraphManager mgr(
      fake_stream, &MockCudaGraphCallbacks::DestroyGraphExec, &MockCudaGraphCallbacks::SyncStream);

  // Warm-up counter tracking before capture
  EXPECT_FALSE(mgr.IsGraphCaptureAllowed(10, 2));
  mgr.IncrementRegularRunCount(10);
  EXPECT_FALSE(mgr.IsGraphCaptureAllowed(10, 2));
  mgr.IncrementRegularRunCount(10);
  EXPECT_TRUE(mgr.IsGraphCaptureAllowed(10, 2));
  EXPECT_EQ(mgr.RunCountEntryCount(), 1u);

  // Register captured executable
  mgr.PutGraphExecForTesting(10, fake_exec_1);
  EXPECT_TRUE(mgr.IsGraphCaptured(10));
  EXPECT_TRUE(mgr.HasGraphOrRunCount(10));
  EXPECT_EQ(mgr.CapturedGraphCount(), 1u);

  // Release ID 10: must synchronize stream once, destroy handle once,
  // and remove the warm-up run counter so recapture requires fresh warm-up runs.
  OrtStatus* st = mgr.Release(10, /*sync_stream=*/true);
  ASSERT_EQ(st, nullptr);
  EXPECT_FALSE(mgr.IsGraphCaptured(10));
  EXPECT_FALSE(mgr.HasGraphOrRunCount(10));
  EXPECT_FALSE(mgr.IsGraphCaptureAllowed(10, 2));
  EXPECT_EQ(mgr.CapturedGraphCount(), 0u);
  EXPECT_EQ(mgr.RunCountEntryCount(), 0u);
  ASSERT_EQ(MockCudaGraphCallbacks::synced_streams.size(), 1u);
  EXPECT_EQ(MockCudaGraphCallbacks::synced_streams[0], fake_stream);
  ASSERT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 1u);
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles[0], fake_exec_1);

  // Recapture on the same ID 10 must succeed after warm-up
  mgr.IncrementRegularRunCount(10);
  mgr.IncrementRegularRunCount(10);
  EXPECT_TRUE(mgr.IsGraphCaptureAllowed(10, 2));
  ASSERT_NO_THROW(mgr.PutGraphExecForTesting(10, fake_exec_1_recaptured));
  EXPECT_TRUE(mgr.IsGraphCaptured(10));
  EXPECT_EQ(mgr.CapturedGraphCount(), 1u);
}

TEST(CudaPluginGraphRetirementHostTest, RepeatedAndMissingReleaseAreIdempotentNoOpsAndOtherIdsSurvive) {
  MockCudaGraphCallbacks::Reset();
  auto fake_stream = reinterpret_cast<cudaStream_t>(uintptr_t{0xCAFE0002});
  auto fake_exec_1 = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0011});
  auto fake_exec_2 = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0022});

  onnxruntime::cuda_plugin::CudaGraphManager mgr(
      fake_stream, &MockCudaGraphCallbacks::DestroyGraphExec, &MockCudaGraphCallbacks::SyncStream);

  mgr.IncrementRegularRunCount(1);
  mgr.PutGraphExecForTesting(1, fake_exec_1);
  mgr.IncrementRegularRunCount(2);
  mgr.PutGraphExecForTesting(2, fake_exec_2);
  // ID 3 has only warm-up run count (not yet captured)
  mgr.IncrementRegularRunCount(3);
  EXPECT_EQ(mgr.CapturedGraphCount(), 2u);
  EXPECT_EQ(mgr.RunCountEntryCount(), 3u);

  // Release ID 1: ID 2 and ID 3 survive
  ASSERT_EQ(mgr.Release(1, /*sync_stream=*/true), nullptr);
  EXPECT_FALSE(mgr.IsGraphCaptured(1));
  EXPECT_TRUE(mgr.IsGraphCaptured(2));
  EXPECT_TRUE(mgr.HasGraphOrRunCount(3));
  EXPECT_EQ(mgr.CapturedGraphCount(), 1u);
  EXPECT_EQ(mgr.RunCountEntryCount(), 2u);
  ASSERT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 1u);
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles[0], fake_exec_1);

  // Repeated release of ID 1, release of missing ID 999, and release of skip ID -1 are no-ops
  ASSERT_EQ(mgr.Release(1, /*sync_stream=*/true), nullptr);
  ASSERT_EQ(mgr.Release(999, /*sync_stream=*/true), nullptr);
  ASSERT_EQ(mgr.Release(onnxruntime::cuda_plugin::kCudaGraphAnnotationSkip, /*sync_stream=*/true), nullptr);
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 1u);
  EXPECT_EQ(MockCudaGraphCallbacks::synced_streams.size(), 1u);

  // Release of warm-up-only ID 3 clears run_count_ without syncing or destroying any handle
  ASSERT_EQ(mgr.Release(3, /*sync_stream=*/false), nullptr);
  EXPECT_FALSE(mgr.HasGraphOrRunCount(3));
  EXPECT_EQ(mgr.RunCountEntryCount(), 1u);
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 1u);
  EXPECT_EQ(MockCudaGraphCallbacks::synced_streams.size(), 1u);
}

TEST(CudaPluginGraphRetirementHostTest, TeardownDoesNotDoubleDestroyAlreadyReleasedGraphs) {
  MockCudaGraphCallbacks::Reset();
  auto fake_stream = reinterpret_cast<cudaStream_t>(uintptr_t{0xCAFE0003});
  auto fake_exec_1 = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0101});
  auto fake_exec_2 = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0202});

  {
    onnxruntime::cuda_plugin::CudaGraphManager mgr(
        fake_stream, &MockCudaGraphCallbacks::DestroyGraphExec, &MockCudaGraphCallbacks::SyncStream);
    mgr.PutGraphExecForTesting(1, fake_exec_1);
    mgr.PutGraphExecForTesting(2, fake_exec_2);

    ASSERT_EQ(mgr.Release(1, /*sync_stream=*/true), nullptr);
    ASSERT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 1u);
    EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles[0], fake_exec_1);
  }

  // Manager destructor must destroy only the remaining unreleased ID 2 once, never ID 1 again.
  ASSERT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 2u);
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles[0], fake_exec_1);
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles[1], fake_exec_2);
}

TEST(CudaPluginGraphRetirementHostTest, StreamSyncFailurePreservesGraphAndReturnsEpFail) {
  MockCudaGraphCallbacks::Reset();
  auto fake_stream = reinterpret_cast<cudaStream_t>(uintptr_t{0xCAFE0004});
  auto fake_exec_1 = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0303});

  onnxruntime::cuda_plugin::CudaGraphManager mgr(
      fake_stream, &MockCudaGraphCallbacks::DestroyGraphExec, &MockCudaGraphCallbacks::SyncStream);
  mgr.IncrementRegularRunCount(7);
  mgr.PutGraphExecForTesting(7, fake_exec_1);

  MockCudaGraphCallbacks::next_sync_result = cudaErrorLaunchFailure;
  OrtStatus* st = mgr.Release(7, /*sync_stream=*/true);
  ASSERT_NE(st, nullptr);
  Ort::GetApi().ReleaseStatus(st);

  // Graph and run_count must remain intact if stream sync failed before destruction
  EXPECT_TRUE(mgr.IsGraphCaptured(7));
  EXPECT_TRUE(mgr.HasGraphOrRunCount(7));
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 0u);

  // Subsequent release when stream sync succeeds retires the graph cleanly
  MockCudaGraphCallbacks::next_sync_result = cudaSuccess;
  ASSERT_EQ(mgr.Release(7, /*sync_stream=*/true), nullptr);
  EXPECT_FALSE(mgr.IsGraphCaptured(7));
  ASSERT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 1u);
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles[0], fake_exec_1);
}

TEST(CudaPluginGraphRetirementHostTest, PutAllocationFailureDuplicateIdAndDestroyErrorNeverDoubleDestroyOrClaimSuccess) {
  MockCudaGraphCallbacks::Reset();
  auto fake_stream = reinterpret_cast<cudaStream_t>(uintptr_t{0xCAFE0005});
  auto fake_exec_bad_alloc = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0501});
  auto fake_exec_valid = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0502});
  auto fake_exec_dup = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0503});
  auto fake_exec_destroy_err = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0504});

  {
    onnxruntime::cuda_plugin::CudaGraphManager mgr(
        fake_stream, &MockCudaGraphCallbacks::DestroyGraphExec, &MockCudaGraphCallbacks::SyncStream);

    // 1. Simulated std::bad_alloc inside CudaGraphSet::Put: handle must be destroyed exactly once.
    mgr.SetInjectPutAllocFailureForTesting(true);
    EXPECT_THROW(mgr.PutGraphExecForTesting(11, fake_exec_bad_alloc), std::bad_alloc);
    mgr.SetInjectPutAllocFailureForTesting(false);
    EXPECT_FALSE(mgr.IsGraphCaptured(11));
    EXPECT_EQ(mgr.CapturedGraphCount(), 0u);
    ASSERT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 1u);
    EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles[0], fake_exec_bad_alloc);

    // 2. Insert valid ID 11, then attempt duplicate Put on ID 11: duplicate handle must be destroyed
    //    exactly once while the existing ID 11 handle remains live.
    ASSERT_NO_THROW(mgr.PutGraphExecForTesting(11, fake_exec_valid));
    EXPECT_TRUE(mgr.IsGraphCaptured(11));
    EXPECT_THROW(mgr.PutGraphExecForTesting(11, fake_exec_dup), std::runtime_error);
    EXPECT_TRUE(mgr.IsGraphCaptured(11));
    EXPECT_EQ(mgr.CapturedGraphCount(), 1u);
    ASSERT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 2u);
    EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles[1], fake_exec_dup);

    // 3. Explicit release where destroy_fn returns an error must return ORT_EP_FAIL and NOT
    //    double-destroy on teardown.
    ASSERT_NO_THROW(mgr.PutGraphExecForTesting(12, fake_exec_destroy_err));
    MockCudaGraphCallbacks::next_destroy_result = cudaErrorInvalidValue;
    OrtStatus* dst_st = mgr.Release(12, /*sync_stream=*/true);
    MockCudaGraphCallbacks::next_destroy_result = cudaSuccess;
    ASSERT_NE(dst_st, nullptr);
    Ort::GetApi().ReleaseStatus(dst_st);
    EXPECT_FALSE(mgr.IsGraphCaptured(12));
    ASSERT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 3u);
    EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles[2], fake_exec_destroy_err);
  }

  // Teardown destroys only fake_exec_valid once (total 4 destroys across all 4 distinct handles).
  ASSERT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 4u);
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles[3], fake_exec_valid);
}

TEST(CudaPluginGraphRetirementHostTest, CudaEpEnforcesThreadOwnershipRejectsActiveCaptureAndAvoidsLazyContextCreation) {
  MockCudaGraphCallbacks::Reset();
  auto fake_stream = reinterpret_cast<cudaStream_t>(uintptr_t{0xCAFE0006});
  onnxruntime::cuda_plugin::CudaEpGraphController controller(
      /*device_id=*/-1,
      /*enable_cuda_graph=*/true,
      /*min_num_runs_before_cuda_graph_capture=*/2,
      /*has_user_compute_stream=*/true,
      /*user_compute_stream=*/fake_stream);

  EXPECT_FALSE(controller.HasPerThreadContextForCurrentThreadForTesting());

  // 1. Releasing an uncaptured ID on a thread without a PerThreadContext must succeed without
  //    lazily creating a PerThreadContext or CUDA stream.
  ASSERT_EQ(controller.ReleaseCapturedGraph(999), nullptr);
  EXPECT_FALSE(controller.HasPerThreadContextForCurrentThreadForTesting());

  // 2. Register graph ID 42 on the current thread (Thread A)
  auto fake_exec_42 = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF4242});
  controller.RegisterGraphForTestingOnCurrentThread(
      42, fake_exec_42, /*warmup_runs=*/2,
      &MockCudaGraphCallbacks::DestroyGraphExec, &MockCudaGraphCallbacks::SyncStream);
  EXPECT_TRUE(controller.HasPerThreadContextForCurrentThreadForTesting());
  EXPECT_TRUE(controller.IsGraphCaptured(42));

  // 3. Releasing while stream capture is active on the current thread must fail with ORT_EP_FAIL
  ::IsThreadCapturingCudaGraph() = true;
  OrtStatus* cap_st = controller.ReleaseCapturedGraph(42);
  ::IsThreadCapturingCudaGraph() = false;
  ASSERT_NE(cap_st, nullptr);
  EXPECT_EQ(Ort::GetApi().GetErrorCode(cap_st), ORT_EP_FAIL);
  Ort::GetApi().ReleaseStatus(cap_st);
  EXPECT_TRUE(controller.IsGraphCaptured(42));
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 0u);

  // 4. Attempting to release Thread A's graph ID 42 from another thread (Thread B) must NOT
  //    silently return success after looking in Thread B's empty map; it must return ORT_EP_FAIL
  //    without creating a PerThreadContext on Thread B.
  bool thread_b_had_context = true;
  OrtErrorCode thread_b_err_code = ORT_OK;
  std::string thread_b_err_msg;
  OrtStatus* thread_b_uncaptured_st = nullptr;
  std::thread other_thread([&]() {
    thread_b_uncaptured_st = controller.ReleaseCapturedGraph(777);
    OrtStatus* st = controller.ReleaseCapturedGraph(42);
    thread_b_had_context = controller.HasPerThreadContextForCurrentThreadForTesting();
    if (st != nullptr) {
      thread_b_err_code = Ort::GetApi().GetErrorCode(st);
      thread_b_err_msg = Ort::GetApi().GetErrorMessage(st);
      Ort::GetApi().ReleaseStatus(st);
    }
  });
  other_thread.join();

  EXPECT_EQ(thread_b_uncaptured_st, nullptr);
  EXPECT_FALSE(thread_b_had_context);
  EXPECT_EQ(thread_b_err_code, ORT_EP_FAIL);
  EXPECT_NE(thread_b_err_msg.find("another thread"), std::string::npos);
  EXPECT_TRUE(controller.IsGraphCaptured(42));
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 0u);

  // 5. Releasing ID 42 on the owning thread (Thread A) retires it cleanly; subsequent release on
  //    either thread is now a safe no-op.
  ASSERT_EQ(controller.ReleaseCapturedGraph(42), nullptr);
  EXPECT_FALSE(controller.IsGraphCaptured(42));
  ASSERT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 1u);
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles[0], fake_exec_42);
  ASSERT_EQ(controller.ReleaseCapturedGraph(42), nullptr);
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 1u);
}

TEST(CudaPluginGraphRetirementHostTest, SkipAnnotationIdDoesNotAccumulateWarmupRunCount) {
  MockCudaGraphCallbacks::Reset();
  auto fake_stream = reinterpret_cast<cudaStream_t>(uintptr_t{0xCAFE0007});
  auto fake_exec_1 = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0701});

  onnxruntime::cuda_plugin::CudaGraphManager mgr(
      fake_stream, &MockCudaGraphCallbacks::DestroyGraphExec, &MockCudaGraphCallbacks::SyncStream);

  // 1. Direct calls to IncrementRegularRunCount(kCudaGraphAnnotationSkip) must be ignored
  //    and never insert -1 into run_count_.
  mgr.IncrementRegularRunCount(onnxruntime::cuda_plugin::kCudaGraphAnnotationSkip);
  mgr.IncrementRegularRunCount(onnxruntime::cuda_plugin::kCudaGraphAnnotationSkip);
  EXPECT_EQ(mgr.RunCountEntryCount(), 0u);
  EXPECT_FALSE(mgr.HasGraphOrRunCount(onnxruntime::cuda_plugin::kCudaGraphAnnotationSkip));
  EXPECT_FALSE(mgr.IsGraphCaptureAllowed(onnxruntime::cuda_plugin::kCudaGraphAnnotationSkip, 1));

  // 2. Interleaving skip-ID runs with a valid captured graph ID 1 and then releasing ID 1
  //    leaves both CapturedGraphCount() and RunCountEntryCount() at 0.
  mgr.IncrementRegularRunCount(1);
  mgr.IncrementRegularRunCount(onnxruntime::cuda_plugin::kCudaGraphAnnotationSkip);
  mgr.IncrementRegularRunCount(1);
  mgr.PutGraphExecForTesting(1, fake_exec_1);
  EXPECT_EQ(mgr.CapturedGraphCount(), 1u);
  EXPECT_EQ(mgr.RunCountEntryCount(), 1u);

  ASSERT_EQ(mgr.Release(1, /*sync_stream=*/true), nullptr);
  EXPECT_EQ(mgr.CapturedGraphCount(), 0u);
  EXPECT_EQ(mgr.RunCountEntryCount(), 0u);

  // 3. Through CudaEpGraphController::OnRunStart / OnRunEnd with kCudaGraphAnnotationSkip (-1)
  //    and sync_stream=true: skip-ID runs must NOT leave a sentinel entry in run_count_, AND
  //    OnRunEnd's final stream synchronization must still execute.
  MockCudaGraphCallbacks::Reset();
  onnxruntime::cuda_plugin::CudaEpGraphController controller(
      /*device_id=*/-1,
      /*enable_cuda_graph=*/true,
      /*min_num_runs_before_cuda_graph_capture=*/2,
      /*has_user_compute_stream=*/true,
      /*user_compute_stream=*/fake_stream);

  auto fake_exec_ep = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0702});
  controller.RegisterGraphForTestingOnCurrentThread(
      1, fake_exec_ep, /*warmup_runs=*/2,
      &MockCudaGraphCallbacks::DestroyGraphExec, &MockCudaGraphCallbacks::SyncStream);
  EXPECT_EQ(controller.CapturedGraphCountOnCurrentThreadForTesting(), 1u);
  EXPECT_EQ(controller.RunCountEntryCountOnCurrentThreadForTesting(), 1u);
  EXPECT_EQ(MockCudaGraphCallbacks::synced_streams.size(), 0u);

  ASSERT_EQ(controller.OnRunStart(onnxruntime::cuda_plugin::kCudaGraphAnnotationSkip), nullptr);
  ASSERT_EQ(controller.OnRunEnd(onnxruntime::cuda_plugin::kCudaGraphAnnotationSkip, /*sync_stream=*/true), nullptr);
  EXPECT_EQ(controller.RunCountEntryCountOnCurrentThreadForTesting(), 1u);
  ASSERT_EQ(MockCudaGraphCallbacks::synced_streams.size(), 1u);
  EXPECT_EQ(MockCudaGraphCallbacks::synced_streams[0], fake_stream);

  ASSERT_EQ(controller.ReleaseCapturedGraph(1), nullptr);
  EXPECT_EQ(controller.CapturedGraphCountOnCurrentThreadForTesting(), 0u);
  EXPECT_EQ(controller.RunCountEntryCountOnCurrentThreadForTesting(), 0u);
}

TEST(CudaPluginGraphRetirementHostTest,
     ConcurrentForeignReleaseWhileOwnerMutatesOtherControllersInSharedThreadCache) {
  ThreadSafeMockCudaGraphCallbacks::Reset();
  auto fake_stream = reinterpret_cast<cudaStream_t>(uintptr_t{0xCAFE0008});
  auto fake_exec_a = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0801});
  auto fake_exec_b_survivor = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0802});

  onnxruntime::cuda_plugin::CudaEpGraphController controller_a(
      /*device_id=*/-1,
      /*enable_cuda_graph=*/true,
      /*min_num_runs_before_cuda_graph_capture=*/2,
      /*has_user_compute_stream=*/true,
      /*user_compute_stream=*/fake_stream);

  onnxruntime::cuda_plugin::CudaEpGraphController controller_b_survivor(
      /*device_id=*/-1,
      /*enable_cuda_graph=*/true,
      /*min_num_runs_before_cuda_graph_capture=*/2,
      /*has_user_compute_stream=*/true,
      /*user_compute_stream=*/fake_stream);

  std::mutex coord_mutex;
  std::condition_variable coord_cv;
  bool t1_registered_a = false;
  bool t2_ready = false;
  bool start_overlap = false;
  bool t2_finished_foreign_probes = false;
  std::atomic<bool> t2_completed_first_foreign_probe{false};
  std::atomic<bool> t1_finished_cache_mutations{false};
  std::atomic<bool> abort_flag{false};

  constexpr size_t kWave1TransientControllers = 64;
  constexpr size_t kWave2TransientControllers = 64;
  constexpr size_t kMinForeignReleaseProbes = 64;
  constexpr auto kHandshakeTimeout = std::chrono::seconds(10);

  // Results recorded by T1 and T2 (inspected on the main thread after joining)
  bool t1_phase1_a_alive_after_t2_probes = false;
  bool t1_phase1_b_alive_after_t2_probes = false;
  bool t1_a_survived_during_overlap = false;
  bool t1_b_survivor_survived = false;
  bool t1_owner_release_a_ok = false;
  size_t t1_post_release_a_captured_count = 999;
  size_t t1_post_release_a_run_count = 999;
  size_t t1_post_release_b_captured_count = 0;
  bool t1_owner_release_b_ok = false;

  size_t t2_foreign_attempts = 0;
  size_t t2_expected_foreign_failures = 0;
  bool t2_had_context_on_a = true;

  std::thread t1([&]() {
    try {
      // 1. Register controller_a (ID 101) and controller_b_survivor (ID 202) in T1's shared cache.
      controller_a.RegisterGraphForTestingOnCurrentThread(
          101, fake_exec_a, /*warmup_runs=*/2,
          &ThreadSafeMockCudaGraphCallbacks::DestroyGraphExec,
          &ThreadSafeMockCudaGraphCallbacks::SyncStream);
      controller_b_survivor.RegisterGraphForTestingOnCurrentThread(
          202, fake_exec_b_survivor, /*warmup_runs=*/2,
          &ThreadSafeMockCudaGraphCallbacks::DestroyGraphExec,
          &ThreadSafeMockCudaGraphCallbacks::SyncStream);

      {
        std::lock_guard<std::mutex> lk(coord_mutex);
        t1_registered_a = true;
      }
      coord_cv.notify_all();

      {
        std::unique_lock<std::mutex> lk(coord_mutex);
        if (!coord_cv.wait_for(lk, kHandshakeTimeout, [&]() { return t2_ready || abort_flag.load(); })) {
          abort_flag.store(true, std::memory_order_release);
          coord_cv.notify_all();
          return;
        }
        if (abort_flag.load(std::memory_order_relaxed)) {
          return;
        }
        start_overlap = true;
      }
      coord_cv.notify_all();

      const auto first_probe_deadline = std::chrono::steady_clock::now() + kHandshakeTimeout;
      while (!t2_completed_first_foreign_probe.load(std::memory_order_acquire) &&
             !abort_flag.load(std::memory_order_relaxed)) {
        if (std::chrono::steady_clock::now() >= first_probe_deadline) {
          abort_flag.store(true, std::memory_order_release);
          coord_cv.notify_all();
          return;
        }
        std::this_thread::yield();
      }

      // 2. While T2 repeatedly scans T1's shared PerThreadContextMap via controller_a.ReleaseCapturedGraph(101),
      //    insert, rehash, look up, and destroy transient controllers B_i in T1's SAME shared cache.
      std::vector<std::unique_ptr<onnxruntime::cuda_plugin::CudaEpGraphController>> transient_b;
      transient_b.reserve(kWave1TransientControllers);
      for (size_t i = 0; i < kWave1TransientControllers; ++i) {
        auto ctrl = std::make_unique<onnxruntime::cuda_plugin::CudaEpGraphController>(
            /*device_id=*/-1,
            /*enable_cuda_graph=*/true,
            /*min_num_runs_before_cuda_graph_capture=*/2,
            /*has_user_compute_stream=*/true,
            /*user_compute_stream=*/fake_stream);
        auto exec_handle = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF1000 + i});
        ctrl->RegisterGraphForTestingOnCurrentThread(
            static_cast<int>(300 + i), exec_handle, /*warmup_runs=*/1,
            &ThreadSafeMockCudaGraphCallbacks::DestroyGraphExec,
            &ThreadSafeMockCudaGraphCallbacks::SyncStream);
        transient_b.push_back(std::move(ctrl));
      }

      // Destroy even-indexed controllers (exercising ~CudaEpGraphController erase on T1's shared cache)
      for (size_t i = 0; i < transient_b.size(); i += 2) {
        transient_b[i].reset();
      }

      // Insert a second wave of controllers to force additional map growth/rehash after erasures
      for (size_t i = 0; i < kWave2TransientControllers; ++i) {
        auto ctrl = std::make_unique<onnxruntime::cuda_plugin::CudaEpGraphController>(
            /*device_id=*/-1,
            /*enable_cuda_graph=*/true,
            /*min_num_runs_before_cuda_graph_capture=*/2,
            /*has_user_compute_stream=*/true,
            /*user_compute_stream=*/fake_stream);
        auto exec_handle = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF2000 + i});
        ctrl->RegisterGraphForTestingOnCurrentThread(
            static_cast<int>(500 + i), exec_handle, /*warmup_runs=*/1,
            &ThreadSafeMockCudaGraphCallbacks::DestroyGraphExec,
            &ThreadSafeMockCudaGraphCallbacks::SyncStream);
        transient_b.push_back(std::move(ctrl));
      }

      // Tear down all remaining transient controllers while T2 is still probing controller_a
      transient_b.clear();
      t1_finished_cache_mutations.store(true, std::memory_order_release);
      coord_cv.notify_all();

      // Keep T1's thread_local PerThreadContextMap and its controller_a / controller_b_survivor
      // graphs alive until T2 confirms all foreign probes and inspections have finished.
      {
        std::unique_lock<std::mutex> lk(coord_mutex);
        if (!coord_cv.wait_for(lk, kHandshakeTimeout,
                               [&]() { return t2_finished_foreign_probes || abort_flag.load(); })) {
          abort_flag.store(true, std::memory_order_release);
          coord_cv.notify_all();
          return;
        }
      }

      t1_phase1_a_alive_after_t2_probes =
          controller_a.IsGraphCaptured(101) &&
          (ThreadSafeMockCudaGraphCallbacks::DestroyCountForHandle(fake_exec_a) == 0u);
      t1_phase1_b_alive_after_t2_probes =
          controller_b_survivor.IsGraphCaptured(202) &&
          (ThreadSafeMockCudaGraphCallbacks::DestroyCountForHandle(fake_exec_b_survivor) == 0u);
    } catch (...) {
      abort_flag.store(true, std::memory_order_release);
      coord_cv.notify_all();
    }
  });

  std::thread t2([&]() {
    struct T2CompletionSignal {
      std::mutex& mu;
      std::condition_variable& cv;
      bool& done_flag;
      ~T2CompletionSignal() {
        {
          std::lock_guard<std::mutex> lk(mu);
          done_flag = true;
        }
        cv.notify_all();
      }
    } completion_signal{coord_mutex, coord_cv, t2_finished_foreign_probes};

    try {
      {
        std::unique_lock<std::mutex> lk(coord_mutex);
        if (!coord_cv.wait_for(lk, kHandshakeTimeout, [&]() { return t1_registered_a || abort_flag.load(); })) {
          abort_flag.store(true, std::memory_order_release);
          coord_cv.notify_all();
          return;
        }
        t2_ready = true;
        coord_cv.notify_all();
        if (!coord_cv.wait_for(lk, kHandshakeTimeout, [&]() { return start_overlap || abort_flag.load(); })) {
          abort_flag.store(true, std::memory_order_release);
          coord_cv.notify_all();
          return;
        }
      }

      const auto probe_loop_deadline = std::chrono::steady_clock::now() + kHandshakeTimeout;
      while (!abort_flag.load(std::memory_order_relaxed)) {
        OrtStatus* st = controller_a.ReleaseCapturedGraph(101);
        ++t2_foreign_attempts;
        if (st != nullptr) {
          const OrtErrorCode code = Ort::GetApi().GetErrorCode(st);
          const std::string msg = Ort::GetApi().GetErrorMessage(st);
          Ort::GetApi().ReleaseStatus(st);
          if (code == ORT_EP_FAIL && msg.find("another thread") != std::string::npos) {
            ++t2_expected_foreign_failures;
          }
        }
        t2_completed_first_foreign_probe.store(true, std::memory_order_release);
        if (t1_finished_cache_mutations.load(std::memory_order_acquire) &&
            t2_foreign_attempts >= kMinForeignReleaseProbes) {
          break;
        }
        if (std::chrono::steady_clock::now() >= probe_loop_deadline) {
          abort_flag.store(true, std::memory_order_release);
          break;
        }
      }
      t2_had_context_on_a = controller_a.HasPerThreadContextForCurrentThreadForTesting();
    } catch (...) {
      abort_flag.store(true, std::memory_order_release);
      coord_cv.notify_all();
    }
  });

  t2.join();
  t1.join();

  ASSERT_FALSE(abort_flag.load());
  EXPECT_GE(t2_foreign_attempts, kMinForeignReleaseProbes);
  EXPECT_EQ(t2_expected_foreign_failures, t2_foreign_attempts);
  EXPECT_FALSE(t2_had_context_on_a);
  EXPECT_TRUE(t1_phase1_a_alive_after_t2_probes);
  EXPECT_TRUE(t1_phase1_b_alive_after_t2_probes);

  // T1 has now exited (destroying T1's thread_local PerThreadContextMap and releasing any remaining
  // contexts owned by T1's cache). Verify that:
  // - Before T1 exited, all 128 transient B handles were destroyed once during transient_b teardown,
  //   and when T1 exited, fake_exec_a and fake_exec_b_survivor were destroyed once by ~PerThreadContext().
  EXPECT_EQ(ThreadSafeMockCudaGraphCallbacks::DestroyCountForHandle(fake_exec_a), 1u);
  EXPECT_EQ(ThreadSafeMockCudaGraphCallbacks::DestroyCountForHandle(fake_exec_b_survivor), 1u);
  EXPECT_EQ(ThreadSafeMockCudaGraphCallbacks::TotalDestroyCount(),
            kWave1TransientControllers + kWave2TransientControllers + 2u);

  // Also verify owner-thread release on a live thread after concurrent foreign-release + B-mutation overlap:
  ThreadSafeMockCudaGraphCallbacks::Reset();
  std::atomic<bool> p2_t1_registered{false};
  std::atomic<bool> p2_t2_done{false};
  std::atomic<bool> p2_abort{false};
  size_t p2_t2_foreign_attempts = 0;
  size_t p2_t2_expected_foreign_failures = 0;

  std::thread p2_t1([&]() {
    try {
      controller_a.RegisterGraphForTestingOnCurrentThread(
          101, fake_exec_a, /*warmup_runs=*/2,
          &ThreadSafeMockCudaGraphCallbacks::DestroyGraphExec,
          &ThreadSafeMockCudaGraphCallbacks::SyncStream);
      controller_b_survivor.RegisterGraphForTestingOnCurrentThread(
          202, fake_exec_b_survivor, /*warmup_runs=*/2,
          &ThreadSafeMockCudaGraphCallbacks::DestroyGraphExec,
          &ThreadSafeMockCudaGraphCallbacks::SyncStream);
      p2_t1_registered.store(true, std::memory_order_release);

      const auto p2_deadline = std::chrono::steady_clock::now() + kHandshakeTimeout;
      while (!p2_t2_done.load(std::memory_order_acquire) &&
             !p2_abort.load(std::memory_order_relaxed)) {
        if (std::chrono::steady_clock::now() >= p2_deadline) {
          p2_abort.store(true, std::memory_order_release);
          return;
        }
        onnxruntime::cuda_plugin::CudaEpGraphController temp_b(
            /*device_id=*/-1, /*enable_cuda_graph=*/true,
            /*min_num_runs_before_cuda_graph_capture=*/2,
            /*has_user_compute_stream=*/true, /*user_compute_stream=*/fake_stream);
        temp_b.RegisterGraphForTestingOnCurrentThread(
            900, reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF9000}), /*warmup_runs=*/1,
            &ThreadSafeMockCudaGraphCallbacks::DestroyGraphExec,
            &ThreadSafeMockCudaGraphCallbacks::SyncStream);
      }

      t1_a_survived_during_overlap =
          controller_a.IsGraphCaptured(101) &&
          (ThreadSafeMockCudaGraphCallbacks::DestroyCountForHandle(fake_exec_a) == 0u);
      t1_b_survivor_survived =
          controller_b_survivor.IsGraphCaptured(202) &&
          (ThreadSafeMockCudaGraphCallbacks::DestroyCountForHandle(fake_exec_b_survivor) == 0u);

      OrtStatus* rel_a = controller_a.ReleaseCapturedGraph(101);
      t1_owner_release_a_ok = (rel_a == nullptr);
      if (rel_a != nullptr) {
        Ort::GetApi().ReleaseStatus(rel_a);
      }
      t1_post_release_a_captured_count = controller_a.CapturedGraphCountOnCurrentThreadForTesting();
      t1_post_release_a_run_count = controller_a.RunCountEntryCountOnCurrentThreadForTesting();
      t1_post_release_b_captured_count =
          controller_b_survivor.CapturedGraphCountOnCurrentThreadForTesting();

      OrtStatus* rel_b = controller_b_survivor.ReleaseCapturedGraph(202);
      t1_owner_release_b_ok = (rel_b == nullptr);
      if (rel_b != nullptr) {
        Ort::GetApi().ReleaseStatus(rel_b);
      }
    } catch (...) {
      p2_abort.store(true, std::memory_order_release);
      p2_t2_done.store(true, std::memory_order_release);
    }
  });

  std::thread p2_t2([&]() {
    struct P2T2DoneGuard {
      std::atomic<bool>& done;
      ~P2T2DoneGuard() { done.store(true, std::memory_order_release); }
    } done_guard{p2_t2_done};

    try {
      const auto p2_deadline = std::chrono::steady_clock::now() + kHandshakeTimeout;
      while (!p2_t1_registered.load(std::memory_order_acquire) &&
             !p2_abort.load(std::memory_order_relaxed)) {
        if (std::chrono::steady_clock::now() >= p2_deadline) {
          p2_abort.store(true, std::memory_order_release);
          return;
        }
        std::this_thread::yield();
      }
      for (size_t i = 0; i < kMinForeignReleaseProbes && !p2_abort.load(std::memory_order_relaxed); ++i) {
        OrtStatus* st = controller_a.ReleaseCapturedGraph(101);
        ++p2_t2_foreign_attempts;
        if (st != nullptr) {
          const OrtErrorCode code = Ort::GetApi().GetErrorCode(st);
          const std::string msg = Ort::GetApi().GetErrorMessage(st);
          Ort::GetApi().ReleaseStatus(st);
          if (code == ORT_EP_FAIL && msg.find("another thread") != std::string::npos) {
            ++p2_t2_expected_foreign_failures;
          }
        }
      }
    } catch (...) {
      p2_abort.store(true, std::memory_order_release);
    }
  });

  p2_t2.join();
  p2_t1.join();

  ASSERT_FALSE(p2_abort.load());
  EXPECT_EQ(p2_t2_foreign_attempts, kMinForeignReleaseProbes);
  EXPECT_EQ(p2_t2_expected_foreign_failures, kMinForeignReleaseProbes);
  EXPECT_TRUE(t1_a_survived_during_overlap);
  EXPECT_TRUE(t1_b_survivor_survived);
  EXPECT_TRUE(t1_owner_release_a_ok);
  EXPECT_EQ(t1_post_release_a_captured_count, 0u);
  EXPECT_EQ(t1_post_release_a_run_count, 0u);
  EXPECT_EQ(t1_post_release_b_captured_count, 1u);
  EXPECT_TRUE(t1_owner_release_b_ok);
  EXPECT_EQ(ThreadSafeMockCudaGraphCallbacks::DestroyCountForHandle(fake_exec_a), 1u);
  EXPECT_EQ(ThreadSafeMockCudaGraphCallbacks::DestroyCountForHandle(fake_exec_b_survivor), 1u);
}

TEST(CudaPluginGraphRetirementHostTest,
     FailedCaptureDiscardAndFailedWarmupPreserveRetirementInvariants) {
  MockCudaGraphCallbacks::Reset();
  auto fake_stream = reinterpret_cast<cudaStream_t>(uintptr_t{0xCAFE0090});
  auto fake_partial_graph = reinterpret_cast<cudaGraph_t>(uintptr_t{0xDEAD0091});
  auto fake_exec_90 = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0090});

  onnxruntime::cuda_plugin::CudaEpGraphController controller(
      /*device_id=*/-1,
      /*enable_cuda_graph=*/true,
      /*min_num_runs_before_cuda_graph_capture=*/2,
      /*has_user_compute_stream=*/true,
      /*user_compute_stream=*/fake_stream);

  // Initialize host mock context with 1 warm-up run (below min_runs=2).
  controller.RegisterGraphForTestingOnCurrentThread(
      90, /*graph_exec=*/nullptr, /*warmup_runs=*/1,
      &MockCudaGraphCallbacks::DestroyGraphExec,
      &MockCudaGraphCallbacks::SyncStream);
  {
    auto locked_ctx = controller.GetLockedPerThreadContext();
    locked_ctx->cuda_graph.SetRuntimeCallbacksForTesting(
        &MockCudaGraphCallbacks::QueryStream,
        &MockCudaGraphCallbacks::LaunchGraph,
        &MockCudaGraphCallbacks::EndCapture,
        &MockCudaGraphCallbacks::DestroyGraph,
        &MockCudaGraphCallbacks::GetLastError);
  }

  // 1. A failed warm-up run (when capture_allowed is false) must not increment run_count_[90].
  OrtStatus* st = controller.OnRunEnd(90, /*sync_stream=*/false, nullptr, /*run_failed=*/true);
  ASSERT_EQ(st, nullptr);
  EXPECT_EQ(controller.RunCountEntryCountOnCurrentThreadForTesting(), 1u);
  {
    auto locked_ctx = controller.GetLockedPerThreadContext();
    EXPECT_FALSE(locked_ctx->cuda_graph.IsGraphCaptureAllowed(90, 2));
  }

  // 2. A successful warm-up run increments run_count_[90] to 2 so capture_allowed becomes true.
  st = controller.OnRunEnd(90, /*sync_stream=*/false, nullptr, /*run_failed=*/false);
  ASSERT_EQ(st, nullptr);
  {
    auto locked_ctx = controller.GetLockedPerThreadContext();
    EXPECT_TRUE(locked_ctx->cuda_graph.IsGraphCaptureAllowed(90, 2));
  }

  // 3. Simulate a failed capture run where cudaStreamEndCapture returns a partial graph:
  //    OnRunEnd(..., run_failed=true) must clear IsThreadCapturingCudaGraph(), invoke DiscardCapture(),
  //    destroy the partial cudaGraph_t without storing a cudaGraphExec_t, and keep the warm-up count.
  IsThreadCapturingCudaGraph() = true;
  MockCudaGraphCallbacks::next_end_capture_result = cudaSuccess;
  MockCudaGraphCallbacks::next_end_capture_graph = fake_partial_graph;
  st = controller.OnRunEnd(90, /*sync_stream=*/false, nullptr, /*run_failed=*/true);
  ASSERT_EQ(st, nullptr);
  EXPECT_FALSE(IsThreadCapturingCudaGraph());
  EXPECT_EQ(MockCudaGraphCallbacks::end_capture_calls, 1u);
  ASSERT_EQ(MockCudaGraphCallbacks::destroyed_graphs.size(), 1u);
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_graphs[0], fake_partial_graph);
  EXPECT_EQ(MockCudaGraphCallbacks::get_last_error_calls, 0u);
  EXPECT_FALSE(controller.IsGraphCaptured(90));
  EXPECT_EQ(controller.CapturedGraphCountOnCurrentThreadForTesting(), 0u);

  // 4. Simulate a failed capture run where cudaStreamEndCapture reports cudaErrorStreamCaptureInvalidated:
  //    DiscardCapture() must clear the sticky CUDA error via cudaGetLastError().
  IsThreadCapturingCudaGraph() = true;
  MockCudaGraphCallbacks::next_end_capture_result = cudaErrorStreamCaptureInvalidated;
  MockCudaGraphCallbacks::next_end_capture_graph = nullptr;
  st = controller.OnRunEnd(90, /*sync_stream=*/false, nullptr, /*run_failed=*/true);
  ASSERT_EQ(st, nullptr);
  EXPECT_FALSE(IsThreadCapturingCudaGraph());
  EXPECT_EQ(MockCudaGraphCallbacks::end_capture_calls, 2u);
  EXPECT_EQ(MockCudaGraphCallbacks::get_last_error_calls, 1u);
  EXPECT_FALSE(controller.IsGraphCaptured(90));

  // 5. Because warm-up count was preserved across failed capture attempts, registering/capturing 90
  //    and then retiring it via ReleaseCapturedGraph(90) succeeds and destroys fake_exec_90 once.
  controller.RegisterGraphForTestingOnCurrentThread(
      90, fake_exec_90, /*warmup_runs=*/0,
      &MockCudaGraphCallbacks::DestroyGraphExec,
      &MockCudaGraphCallbacks::SyncStream);
  EXPECT_TRUE(controller.IsGraphCaptured(90));
  st = controller.ReleaseCapturedGraph(90);
  ASSERT_EQ(st, nullptr);
  EXPECT_FALSE(controller.IsGraphCaptured(90));
  EXPECT_EQ(controller.CapturedGraphCountOnCurrentThreadForTesting(), 0u);
  EXPECT_EQ(controller.RunCountEntryCountOnCurrentThreadForTesting(), 0u);
  ASSERT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 1u);
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles[0], fake_exec_90);
}

TEST(CudaPluginGraphRetirementHostTest,
     GraphGenerationInvalidationAndAsyncReplayStreamQueryWithRetirement) {
  MockCudaGraphCallbacks::Reset();
  auto fake_stream = reinterpret_cast<cudaStream_t>(uintptr_t{0xCAFE0095});
  auto fake_exec_10 = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0010});
  auto fake_exec_20 = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0020});

  onnxruntime::cuda_plugin::CudaEpGraphController controller(
      /*device_id=*/-1,
      /*enable_cuda_graph=*/true,
      /*min_num_runs_before_cuda_graph_capture=*/2,
      /*has_user_compute_stream=*/true,
      /*user_compute_stream=*/fake_stream);

  controller.RegisterGraphForTestingOnCurrentThread(
      10, fake_exec_10, /*warmup_runs=*/2,
      &MockCudaGraphCallbacks::DestroyGraphExec,
      &MockCudaGraphCallbacks::SyncStream);
  controller.RegisterGraphForTestingOnCurrentThread(
      20, fake_exec_20, /*warmup_runs=*/2,
      &MockCudaGraphCallbacks::DestroyGraphExec,
      &MockCudaGraphCallbacks::SyncStream);
  {
    auto locked_ctx = controller.GetLockedPerThreadContext();
    locked_ctx->cuda_graph.SetRuntimeCallbacksForTesting(
        &MockCudaGraphCallbacks::QueryStream,
        &MockCudaGraphCallbacks::LaunchGraph,
        &MockCudaGraphCallbacks::EndCapture,
        &MockCudaGraphCallbacks::DestroyGraph,
        &MockCudaGraphCallbacks::GetLastError);
    // Verify upstream has_captured_any_ warm-up floor: once any graph has been captured,
    // IsGraphCaptureAllowed(id, min_runs=0) requires std::max(1, 0) == 1 warm-up run.
    EXPECT_TRUE(locked_ctx->cuda_graph.HasCapturedAny());
    EXPECT_FALSE(locked_ctx->cuda_graph.IsGraphCaptureAllowed(30, /*min_runs=*/0));
    locked_ctx->cuda_graph.IncrementRegularRunCount(30);
    EXPECT_TRUE(locked_ctx->cuda_graph.IsGraphCaptureAllowed(30, /*min_runs=*/0));
    OrtStatus* rel_30 = locked_ctx->cuda_graph.Release(30, /*sync_stream=*/false);
    ASSERT_EQ(rel_30, nullptr);
  }

  // 1. ReplayGraph(10, gen=0) uses async Replay(10, sync=false), which flushes via QueryStream.
  //    Both cudaSuccess and cudaErrorNotReady are treated as non-error completion states.
  MockCudaGraphCallbacks::next_query_result = cudaErrorNotReady;
  OrtStatus* st = controller.ReplayGraph(10, /*qmoe_graph_gen=*/0);
  ASSERT_EQ(st, nullptr);
  EXPECT_EQ(MockCudaGraphCallbacks::launched_handles.size(), 1u);
  EXPECT_EQ(MockCudaGraphCallbacks::queried_streams.size(), 1u);

  // 2. A real stream query error during async ReplayGraph is propagated as ORT_EP_FAIL.
  MockCudaGraphCallbacks::next_query_result = cudaErrorIllegalAddress;
  st = controller.ReplayGraph(10, /*qmoe_graph_gen=*/0);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(Ort::GetApi().GetErrorCode(st), ORT_EP_FAIL);
  EXPECT_NE(std::string(Ort::GetApi().GetErrorMessage(st)).find("CUDA graph stream query error"),
            std::string::npos);
  Ort::GetApi().ReleaseStatus(st);
  MockCudaGraphCallbacks::next_query_result = cudaSuccess;

  // 3. Retire graph 10 selectively at generation 0; graph 20 remains captured.
  st = controller.ReleaseCapturedGraph(10);
  ASSERT_EQ(st, nullptr);
  EXPECT_FALSE(controller.IsGraphCaptured(10, /*qmoe_graph_gen=*/0));
  EXPECT_TRUE(controller.IsGraphCaptured(20, /*qmoe_graph_gen=*/0));
  ASSERT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 1u);
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles[0], fake_exec_10);

  // 4. Advance QMoE graph generation to 1 during ReplayGraph(20, gen=1):
  //    must reset remaining captured graph 20 (destroying fake_exec_20 once) and return ORT_FAIL.
  st = controller.ReplayGraph(20, /*qmoe_graph_gen=*/1);
  ASSERT_NE(st, nullptr);
  EXPECT_EQ(Ort::GetApi().GetErrorCode(st), ORT_FAIL);
  EXPECT_NE(std::string(Ort::GetApi().GetErrorMessage(st)).find("invalidated by state reclamation"),
            std::string::npos);
  Ort::GetApi().ReleaseStatus(st);
  EXPECT_FALSE(controller.IsGraphCaptured(20, /*qmoe_graph_gen=*/1));
  EXPECT_EQ(controller.CapturedGraphCountOnCurrentThreadForTesting(), 0u);
  EXPECT_EQ(controller.RunCountEntryCountOnCurrentThreadForTesting(), 0u);
  ASSERT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 2u);
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles[1], fake_exec_20);

  // 5. Subsequent ReleaseCapturedGraph(20) after generation-invalidation Reset() is a safe no-op
  //    and does not double-destroy fake_exec_20.
  st = controller.ReleaseCapturedGraph(20);
  ASSERT_EQ(st, nullptr);
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 2u);
}

TEST(CudaPluginGraphRetirementHostTest,
     LockedPerThreadContextDeletesMoveAssignmentAndUnlocksBeforeContextTeardown) {
  using LockedCtx = onnxruntime::cuda_plugin::CudaEpGraphController::LockedPerThreadContext;
  static_assert(std::is_move_constructible_v<LockedCtx>,
                "LockedPerThreadContext must remain move-constructible");
  static_assert(!std::is_move_assignable_v<LockedCtx>,
                "LockedPerThreadContext move-assignment must be deleted to avoid destroying "
                "PerThreadContext::mutex before unlocking it");
  static_assert(!std::is_copy_constructible_v<LockedCtx>);
  static_assert(!std::is_copy_assignable_v<LockedCtx>);

  MockCudaGraphCallbacks::Reset();
  auto fake_stream = reinterpret_cast<cudaStream_t>(uintptr_t{0xCAFE00B0});
  auto fake_exec_77 = reinterpret_cast<cudaGraphExec_t>(uintptr_t{0xBEEF0077});

  onnxruntime::cuda_plugin::CudaEpGraphController controller(
      /*device_id=*/-1,
      /*enable_cuda_graph=*/true,
      /*min_num_runs_before_cuda_graph_capture=*/2,
      /*has_user_compute_stream=*/true,
      /*user_compute_stream=*/fake_stream);

  controller.RegisterGraphForTestingOnCurrentThread(
      77, fake_exec_77, /*warmup_runs=*/2,
      &MockCudaGraphCallbacks::DestroyGraphExec,
      &MockCudaGraphCallbacks::SyncStream);

  {
    // Acquire LockedPerThreadContext via move construction, then clear the controller's
    // per-thread cache entry while locked_moved still holds the last shared_ptr<PerThreadContext>
    // and its unique_lock<std::mutex>. When locked_moved leaves scope, ~LockedPerThreadContext()
    // must unlock lock_ first and then destroy context_ (which retires fake_exec_77 once).
    LockedCtx locked_initial = controller.GetLockedPerThreadContext();
    LockedCtx locked_moved(std::move(locked_initial));
    EXPECT_TRUE(locked_moved->cuda_graph.IsGraphCaptured(77));

    controller.ClearPerThreadContexts();
    EXPECT_FALSE(controller.HasPerThreadContextForCurrentThreadForTesting());
    EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 0u);
  }

  ASSERT_EQ(MockCudaGraphCallbacks::destroyed_handles.size(), 1u);
  EXPECT_EQ(MockCudaGraphCallbacks::destroyed_handles[0], fake_exec_77);
}

TEST(CudaPluginGraphRetirementHostTest, CudaVisibleDevicesAutoRegistrationPolicy) {
  // Unset CUDA_VISIBLE_DEVICES must not classify CUDA as explicitly disabled.
  EXPECT_FALSE(IsCudaVisibleDevicesExplicitlyDisabled(std::nullopt));
  EXPECT_TRUE(ShouldAutoRegisterCudaPluginEp(std::nullopt, std::nullopt));

  // Explicitly empty "" or exact "-1" must preserve intentional disable behavior.
  EXPECT_TRUE(IsCudaVisibleDevicesExplicitlyDisabled(std::optional<std::string>{""}));
  EXPECT_FALSE(ShouldAutoRegisterCudaPluginEp(std::nullopt, std::optional<std::string>{""}));
  EXPECT_TRUE(IsCudaVisibleDevicesExplicitlyDisabled(std::optional<std::string>{"-1"}));
  EXPECT_FALSE(ShouldAutoRegisterCudaPluginEp(std::nullopt, std::optional<std::string>{"-1"}));

  // Device indices / UUIDs and non-exact strings must preserve auto-registration.
  EXPECT_FALSE(IsCudaVisibleDevicesExplicitlyDisabled(std::optional<std::string>{"0"}));
  EXPECT_TRUE(ShouldAutoRegisterCudaPluginEp(std::nullopt, std::optional<std::string>{"0"}));
  EXPECT_FALSE(IsCudaVisibleDevicesExplicitlyDisabled(std::optional<std::string>{"0,1"}));
  EXPECT_TRUE(ShouldAutoRegisterCudaPluginEp(std::nullopt, std::optional<std::string>{"0,1"}));
  EXPECT_FALSE(IsCudaVisibleDevicesExplicitlyDisabled(std::optional<std::string>{"GPU-1be37795-b43f-34ae-4871-f249699e949c"}));
  EXPECT_TRUE(ShouldAutoRegisterCudaPluginEp(
      std::nullopt, std::optional<std::string>{"GPU-1be37795-b43f-34ae-4871-f249699e949c"}));
  EXPECT_FALSE(IsCudaVisibleDevicesExplicitlyDisabled(std::optional<std::string>{"-10"}));
  EXPECT_TRUE(ShouldAutoRegisterCudaPluginEp(std::nullopt, std::optional<std::string>{"-10"}));

  // An explicit dynamic plugin configuration retains precedence over default auto-registration.
  const std::optional<std::string> explicit_config{"{\"selected_ep_name\":\"CUDAExecutionProvider\"}"};
  EXPECT_FALSE(ShouldAutoRegisterCudaPluginEp(explicit_config, std::nullopt));
  EXPECT_FALSE(ShouldAutoRegisterCudaPluginEp(explicit_config, std::optional<std::string>{"0"}));
  EXPECT_FALSE(ShouldAutoRegisterCudaPluginEp(explicit_config, std::optional<std::string>{""}));
  EXPECT_FALSE(ShouldAutoRegisterCudaPluginEp(explicit_config, std::optional<std::string>{"-1"}));

  // Reading a guaranteed-unset variable name returns std::nullopt without mutating process env.
  EXPECT_FALSE(ReadEnvironmentVariableOptional("ORT_UNIT_TEST_UNSET_ENV_VAR_33927_CHECK").has_value());
}

}  // namespace test
}  // namespace onnxruntime

#endif  // defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
