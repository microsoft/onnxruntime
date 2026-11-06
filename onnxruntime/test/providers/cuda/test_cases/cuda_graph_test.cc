// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <chrono>
#include <cstdint>
#include <future>
#include <latch>
#include <memory>
#include <gsl/gsl>

#include "gtest/gtest.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/cuda_graph.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_run_options_config_keys.h"

namespace onnxruntime {
namespace test {

class CudaGraphCaptureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    CUDA_CALL_THROW(cudaSetDevice(0));
    CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }

  void TearDown() override {
    EXPECT_EQ(cudaStreamDestroy(stream_), cudaSuccess);
  }

  cudaStream_t stream_ = nullptr;
};

TEST_F(CudaGraphCaptureTest, CaptureOwnershipTransitions) {
  CUDAGraphManager graph(stream_);
  EXPECT_EQ(graph.CapturingGraphId(), kCudaGraphAnnotationSkip);
  EXPECT_THROW(graph.CaptureBegin(kCudaGraphAnnotationSkip), OnnxRuntimeException);
  EXPECT_EQ(graph.CapturingGraphId(), kCudaGraphAnnotationSkip);
  graph.CaptureBegin(7);
  EXPECT_EQ(graph.CapturingGraphId(), 7);
  EXPECT_THROW(graph.CaptureBegin(8), OnnxRuntimeException);
  EXPECT_EQ(graph.CapturingGraphId(), 7);
  EXPECT_THROW(graph.CaptureEnd(8), OnnxRuntimeException);
  EXPECT_EQ(graph.CapturingGraphId(), 7);
  EXPECT_NO_THROW(graph.CaptureEnd(7));
  EXPECT_EQ(graph.CapturingGraphId(), kCudaGraphAnnotationSkip);
  EXPECT_TRUE(graph.IsGraphCaptured(7));
  EXPECT_FALSE(graph.IsGraphCaptured(8));
  const auto status = graph.Release(7);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
}

TEST_F(CudaGraphCaptureTest, CaptureEndDoesNotEndExternalCapture) {
  CUDAGraphManager graph(stream_);
  cudaGraph_t external_graph = nullptr;
  const auto cleanup = gsl::finally([&] {
    if (external_graph != nullptr) {
      EXPECT_EQ(cudaGraphDestroy(external_graph), cudaSuccess);
    }
  });
  CUDA_CALL_THROW(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
  EXPECT_THROW(graph.CaptureEnd(kCudaGraphAnnotationSkip), OnnxRuntimeException);
  EXPECT_NO_THROW(CUDA_CALL_THROW(cudaStreamEndCapture(stream_, &external_graph)));
  EXPECT_FALSE(graph.IsGraphCaptured(kCudaGraphAnnotationSkip));
}

TEST_F(CudaGraphCaptureTest, FailedCaptureEndClearsOwner) {
  CUDAGraphManager graph(stream_);
  graph.CaptureBegin(7);
  EXPECT_EQ(cudaStreamSynchronize(stream_), cudaErrorStreamCaptureUnsupported);
  EXPECT_THROW(graph.CaptureEnd(7), OnnxRuntimeException);
  EXPECT_EQ(graph.CapturingGraphId(), kCudaGraphAnnotationSkip);
  EXPECT_FALSE(graph.IsGraphCaptured(7));
  graph.CaptureBegin(7);
  graph.CaptureEnd(7);
  EXPECT_TRUE(graph.IsGraphCaptured(7));
}

TEST_F(CudaGraphCaptureTest, ReleaseWaitsForAsynchronousReplay) {
  struct PendingReplay {
    std::promise<void> started;
    std::latch finish{1};
  } pending;
  auto started = pending.started.get_future();
  CUDAGraphManager graph(stream_);
  std::future<Status> release;
  std::promise<void> release_started;
  auto releasing = release_started.get_future();
  const auto unblock = gsl::finally([&] {
    if (!pending.finish.try_wait()) {
      pending.finish.count_down();
    }
    if (release.valid()) {
      release.wait();
    }
    EXPECT_EQ(cudaStreamSynchronize(stream_), cudaSuccess);
  });

  graph.CaptureBegin(7);
  CUDA_CALL_THROW(cudaLaunchHostFunc(
      stream_, [](void* data) {
        auto& pending = *static_cast<PendingReplay*>(data);
        pending.started.set_value();
        pending.finish.wait();
      },
      &pending));
  graph.CaptureEnd(7);
  const auto replay_status = graph.Replay(7, false);
  ASSERT_TRUE(replay_status.IsOK()) << replay_status.ErrorMessage();
  ASSERT_EQ(started.wait_for(std::chrono::seconds(5)), std::future_status::ready);

  release = std::async(std::launch::async, [&] {
    release_started.set_value();
    return graph.Release(7);
  });
  releasing.wait();
  EXPECT_EQ(release.wait_for(std::chrono::milliseconds(50)), std::future_status::timeout);
  pending.finish.count_down();
  const auto release_status = release.get();
  EXPECT_TRUE(release_status.IsOK()) << release_status.ErrorMessage();
  EXPECT_FALSE(graph.IsGraphCaptured(7));
}

TEST_F(CudaGraphCaptureTest, ExternalBuffersRetainProviderLifetime) {
  for (const bool enable_cuda_graph : {false, true}) {
    SCOPED_TRACE(enable_cuda_graph);
    std::weak_ptr<int> retained;
    {
      CUDAExecutionProviderInfo info;
      info.enable_cuda_graph = enable_cuda_graph;
      info.has_user_compute_stream = true;
      info.user_compute_stream = stream_;
      CUDAExecutionProvider provider(info);
      EXPECT_FALSE(provider.IsGraphCaptured(0));

      auto buffer = std::make_shared<int>(17);
      retained = buffer;
      cudaGraph_t external_graph = nullptr;
      const auto cleanup = gsl::finally([&] {
        if (external_graph != nullptr) {
          EXPECT_EQ(cudaGraphDestroy(external_graph), cudaSuccess);
        }
      });
      CUDA_CALL_THROW(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
      EXPECT_NO_THROW(provider.RetainBufferForGraphCapture(buffer));
      CUDA_CALL_THROW(cudaStreamEndCapture(stream_, &external_graph));
      buffer.reset();
      EXPECT_FALSE(retained.expired());

      for (const int graph_id : {0, 7, kCudaGraphAnnotationSkip}) {
        const auto status = provider.ReleaseCapturedGraph(graph_id);
        ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
        EXPECT_FALSE(retained.expired());
      }
    }
    EXPECT_TRUE(retained.expired());
  }
}

TEST(CudaGraphCaptureDeviceTest, SharedContextSelectsDeviceForSerializedThreads) {
  int device_count = 0;
  CUDA_CALL_THROW(cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    GTEST_SKIP() << "Requires two visible CUDA devices";
  }

  int original_device = 0;
  CUDA_CALL_THROW(cudaGetDevice(&original_device));
  const auto restore_device = gsl::finally([&] {
    EXPECT_EQ(cudaSetDevice(original_device), cudaSuccess);
  });
  CUDAExecutionProviderInfo info;
  info.device_id = 1;
  info.enable_cuda_graph = true;
  info.min_num_runs_before_cuda_graph_capture = 1;
  CUDAExecutionProvider provider(info);
  provider.SetLogger(&logging::LoggingManager::DefaultLogger());
  EXPECT_FALSE(provider.IsGraphCaptured(0));

  constexpr size_t kBufferBytes = 16;
  void* buffer = nullptr;
  CUDA_CALL_THROW(cudaMalloc(&buffer, kBufferBytes));
  const auto release_buffer = gsl::finally([&] {
    EXPECT_EQ(cudaSetDevice(info.device_id), cudaSuccess);
    EXPECT_EQ(cudaFree(buffer), cudaSuccess);
  });
  const auto read_buffer = [&] {
    std::array<uint8_t, kBufferBytes> output{};
    CUDA_CALL_THROW(cudaMemcpy(output.data(), buffer, kBufferBytes, cudaMemcpyDeviceToHost));
    return output;
  };

  uint8_t value = 0;
  std::array<uint8_t, kBufferBytes> expected{};
  for (const int graph_id : {0, kCudaGraphAnnotationSkip, 0}) {
    expected.fill(++value);
    auto run = std::async(std::launch::async, [&, graph_id, value] {
      CUDA_CALL_THROW(cudaSetDevice(0));
      Ort::RunOptions options;
      options.AddConfigEntry(kOrtRunOptionsConfigCudaGraphAnnotation, std::to_string(graph_id).c_str());
      ORT_THROW_IF_ERROR(provider.OnRunStart(*options));
      int current_device = -1;
      CUDA_CALL_THROW(cudaGetDevice(&current_device));
      EXPECT_EQ(current_device, info.device_id);
      CUDA_CALL_THROW(cudaMemsetAsync(buffer, value, kBufferBytes, provider.ComputeStream()));
      ORT_THROW_IF_ERROR(provider.OnRunEnd(/*sync_stream=*/true, *options));
      return read_buffer();
    });
    EXPECT_EQ(run.get(), expected);
  }

  EXPECT_TRUE(provider.IsGraphCaptured(0));
  CUDA_CALL_THROW(cudaMemset(buffer, 0, kBufferBytes));
  auto replay = std::async(std::launch::async, [&] {
    CUDA_CALL_THROW(cudaSetDevice(0));
    ORT_THROW_IF_ERROR(provider.ReplayGraph(0));
    int current_device = -1;
    CUDA_CALL_THROW(cudaGetDevice(&current_device));
    EXPECT_EQ(current_device, info.device_id);
    return read_buffer();
  });
  EXPECT_EQ(replay.get(), expected);
  ORT_THROW_IF_ERROR(provider.ReleaseCapturedGraph(0));
}

}  // namespace test
}  // namespace onnxruntime

#ifdef ORT_CUDA_GRAPH_FAILURE_TESTS

#define ORT_CUDA_REAL_IMPL(name) __real_##name
#define ORT_CUDA_REAL(name) ORT_CUDA_REAL_IMPL(name)
#define ORT_CUDA_WRAP_IMPL(name) __wrap_##name
#define ORT_CUDA_WRAP(name) ORT_CUDA_WRAP_IMPL(name)

extern "C" cudaError_t __real_cudaGraphDestroy(cudaGraph_t graph);
extern "C" cudaError_t __real_cudaGraphExecDestroy(cudaGraphExec_t graph);

namespace {

enum class GraphFailure {
  Instantiate,
  DestroyGraph,
  InstantiateAndDestroyGraph,
};

struct GraphFaultState {
  GraphFailure mode = GraphFailure::Instantiate;
  bool active = false;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  int instantiations = 0;
  int graph_destructions = 0;
  int executable_destructions = 0;
};

thread_local GraphFaultState graph_fault;

class ScopedGraphFailure {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScopedGraphFailure);

  explicit ScopedGraphFailure(GraphFailure mode) {
    graph_fault = {};
    graph_fault.mode = mode;
  }

  ~ScopedGraphFailure() { CleanLeaks(); }

  void Arm() { graph_fault.active = true; }

  void CleanLeaks() {
    graph_fault.active = false;
    if (graph_fault.executable != nullptr) {
      EXPECT_EQ(__real_cudaGraphExecDestroy(graph_fault.executable), cudaSuccess);
      graph_fault.executable = nullptr;
    }
    if (graph_fault.graph != nullptr) {
      EXPECT_EQ(__real_cudaGraphDestroy(graph_fault.graph), cudaSuccess);
      graph_fault.graph = nullptr;
    }
  }
};

template <typename Instantiate>
cudaError_t InstantiateGraphWithFailure(Instantiate instantiate, cudaGraphExec_t* executable, cudaGraph_t graph) {
  if (graph_fault.active) {
    graph_fault.graph = graph;
    ++graph_fault.instantiations;
    if (graph_fault.mode != GraphFailure::DestroyGraph) {
      return cudaErrorMemoryAllocation;
    }
  }
  const auto status = instantiate();
  if (graph_fault.active && status == cudaSuccess) {
    graph_fault.executable = *executable;
  }
  return status;
}

}  // namespace

// CUDA 12 changed the exported ABI; the five-argument C++ overload now forwards to it.
#if CUDART_VERSION >= 12000
extern "C" cudaError_t ORT_CUDA_REAL(cudaGraphInstantiate)(cudaGraphExec_t*, cudaGraph_t, unsigned long long);
extern "C" cudaError_t ORT_CUDA_WRAP(cudaGraphInstantiate)(
    cudaGraphExec_t* executable, cudaGraph_t graph, unsigned long long flags) {
  return InstantiateGraphWithFailure(
      [&] { return ORT_CUDA_REAL(cudaGraphInstantiate)(executable, graph, flags); }, executable, graph);
}
#else
extern "C" cudaError_t ORT_CUDA_REAL(cudaGraphInstantiate)(
    cudaGraphExec_t*, cudaGraph_t, cudaGraphNode_t*, char*, std::size_t);
extern "C" cudaError_t ORT_CUDA_WRAP(cudaGraphInstantiate)(
    cudaGraphExec_t* executable, cudaGraph_t graph, cudaGraphNode_t* error_node, char* log, std::size_t size) {
  return InstantiateGraphWithFailure(
      [&] { return ORT_CUDA_REAL(cudaGraphInstantiate)(executable, graph, error_node, log, size); }, executable, graph);
}
#endif

#if CUDART_VERSION >= 11040
extern "C" cudaError_t __real_cudaGraphInstantiateWithFlags(cudaGraphExec_t*, cudaGraph_t, unsigned long long);
extern "C" cudaError_t __wrap_cudaGraphInstantiateWithFlags(
    cudaGraphExec_t* executable, cudaGraph_t graph, unsigned long long flags) {
  return InstantiateGraphWithFailure(
      [&] { return __real_cudaGraphInstantiateWithFlags(executable, graph, flags); }, executable, graph);
}
#endif

extern "C" cudaError_t __wrap_cudaGraphDestroy(cudaGraph_t graph) {
  const bool tracked = graph_fault.active && graph == graph_fault.graph;
  if (tracked)
    ++graph_fault.graph_destructions;
  const auto status = __real_cudaGraphDestroy(graph);
  if (tracked && status == cudaSuccess) {
    graph_fault.graph = nullptr;
    if (graph_fault.mode != GraphFailure::Instantiate) {
      return cudaErrorInvalidValue;
    }
  }
  return status;
}

extern "C" cudaError_t __wrap_cudaGraphExecDestroy(cudaGraphExec_t executable) {
  const bool tracked = graph_fault.active && executable == graph_fault.executable;
  if (tracked)
    ++graph_fault.executable_destructions;
  const auto status = __real_cudaGraphExecDestroy(executable);
  if (tracked && status == cudaSuccess) {
    graph_fault.executable = nullptr;
  }
  return status;
}

#undef ORT_CUDA_REAL_IMPL
#undef ORT_CUDA_REAL
#undef ORT_CUDA_WRAP_IMPL
#undef ORT_CUDA_WRAP

namespace onnxruntime {
namespace test {

class CudaGraphCleanupFailureTest : public CudaGraphCaptureTest {
 protected:
  void CheckFailure(GraphFailure mode) {
    CUDAGraphManager graph(stream_);
    ScopedGraphFailure injection(mode);
    graph.CaptureBegin(7);
    injection.Arm();
    try {
      graph.CaptureEnd(7);
      ADD_FAILURE() << "CaptureEnd unexpectedly succeeded";
    } catch (const OnnxRuntimeException& error) {
      const auto expected_error = mode == GraphFailure::DestroyGraph ? cudaErrorInvalidValue : cudaErrorMemoryAllocation;
      const auto expected = "CUDA failure " + std::to_string(expected_error) + ":";
      EXPECT_NE(std::string(error.what()).find(expected), std::string::npos) << error.what();
    }
    EXPECT_EQ(graph_fault.instantiations, 1);
    EXPECT_EQ(graph_fault.graph_destructions, 1);
    EXPECT_EQ(graph_fault.executable_destructions, mode == GraphFailure::DestroyGraph ? 1 : 0);
    EXPECT_EQ(graph_fault.graph, nullptr);
    EXPECT_EQ(graph_fault.executable, nullptr);
    EXPECT_EQ(graph.CapturingGraphId(), kCudaGraphAnnotationSkip);
    EXPECT_FALSE(graph.IsGraphCaptured(7));
    if (graph.IsGraphCaptured(7)) {
      EXPECT_TRUE(graph.Release(7).IsOK());
    }
    injection.CleanLeaks();
    EXPECT_NO_THROW(graph.CaptureBegin(7));
    EXPECT_NO_THROW(graph.CaptureEnd(7));
    EXPECT_TRUE(graph.IsGraphCaptured(7));
    EXPECT_TRUE(graph.Release(7).IsOK());
  }
};

TEST_F(CudaGraphCleanupFailureTest, InstantiationFailureDestroysTemporaryGraph) {
  CheckFailure(GraphFailure::Instantiate);
}

TEST_F(CudaGraphCleanupFailureTest, TemporaryGraphDestroyFailureDestroysExecutable) {
  CheckFailure(GraphFailure::DestroyGraph);
}

TEST_F(CudaGraphCleanupFailureTest, CleanupErrorDoesNotMaskInstantiationFailure) {
  CheckFailure(GraphFailure::InstantiateAndDestroyGraph);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_CUDA_GRAPH_FAILURE_TESTS
