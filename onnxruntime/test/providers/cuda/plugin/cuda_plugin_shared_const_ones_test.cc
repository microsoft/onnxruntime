// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2026 CERN
// Licensed under the MIT License.

// Tests for the buffer of ones used by Gemm to broadcast its bias (CudaKernel::GetConstOnes).
//
// The CUDA plugin EP keeps one such buffer per device, shared by all the sessions, streams and threads of the process
// (see GetConstOnesBufferForDevice in core/providers/cuda/plugin/cuda_kernel_adapter.h). These tests verify:
//   1. Concurrent sessions, each with its own stream and thread, that keep growing the buffer get correct results.
//      Before the buffer was made thread safe, this crashed (double cudaFree, use after free) or gave wrong results
//      (a stream read the buffer before another stream had filled it).
//   2. A CUDA graph captured by one session keeps giving correct results when it is replayed while another session
//      grows the buffer: the buffer used by the graph must not be freed, and capturing and replaying the graph must not
//      be affected by the synchronisation of the buffer.

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)

#include <array>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/file_util.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
namespace {

constexpr const char* kCudaPluginEpRegistrationName = "CudaPluginSharedConstOnesTest";
constexpr const char* kCudaPluginEpName = "CUDAExecutionProvider";

// Y[M, N] = A[M, K] @ B[K, N] + C[N], a single Gemm node with a dynamic M, B = ones and C = [0, 1, 2, 3]. The bias C is
// broadcast over the M rows using a buffer of M ones. With A filled with ones, Y[m][n] = K + n for every row.
// Generated with onnx 1.22.0 (opset 17, ir_version 9):
//   A = helper.make_tensor_value_info("A", TensorProto.FLOAT, ["M", 4])
//   Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, ["M", 4])
//   B = helper.make_tensor("B", TensorProto.FLOAT, [4, 4], np.ones(16, np.float32))
//   C = helper.make_tensor("C", TensorProto.FLOAT, [4], np.arange(4, dtype=np.float32))
//   node = helper.make_node("Gemm", ["A", "B", "C"], ["Y"], alpha=1.0, beta=1.0, transA=0, transB=0)
//   graph = helper.make_graph([node], "gemm_bias", [A], [Y], [B, C])
//   model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
constexpr int64_t kK = 4;
constexpr int64_t kN = 4;
// clang-format off
alignas(4) constexpr unsigned char kGemmBiasModel[] = {
    0x08, 0x09, 0x3a, 0xf2, 0x01, 0x0a, 0x51, 0x0a, 0x01, 0x41, 0x0a, 0x01, 0x42, 0x0a, 0x01, 0x43,
    0x12, 0x01, 0x59, 0x22, 0x04, 0x47, 0x65, 0x6d, 0x6d, 0x2a, 0x0f, 0x0a, 0x05, 0x61, 0x6c, 0x70,
    0x68, 0x61, 0x15, 0x00, 0x00, 0x80, 0x3f, 0xa0, 0x01, 0x01, 0x2a, 0x0e, 0x0a, 0x04, 0x62, 0x65,
    0x74, 0x61, 0x15, 0x00, 0x00, 0x80, 0x3f, 0xa0, 0x01, 0x01, 0x2a, 0x0d, 0x0a, 0x06, 0x74, 0x72,
    0x61, 0x6e, 0x73, 0x41, 0x18, 0x00, 0xa0, 0x01, 0x02, 0x2a, 0x0d, 0x0a, 0x06, 0x74, 0x72, 0x61,
    0x6e, 0x73, 0x42, 0x18, 0x00, 0xa0, 0x01, 0x02, 0x12, 0x09, 0x67, 0x65, 0x6d, 0x6d, 0x5f, 0x62,
    0x69, 0x61, 0x73, 0x2a, 0x4b, 0x08, 0x04, 0x08, 0x04, 0x10, 0x01, 0x22, 0x40, 0x00, 0x00, 0x80,
    0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80,
    0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80,
    0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80,
    0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x80, 0x3f, 0x42, 0x01, 0x42,
    0x2a, 0x19, 0x08, 0x04, 0x10, 0x01, 0x22, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, 0x3f,
    0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40, 0x42, 0x01, 0x43, 0x5a, 0x14, 0x0a, 0x01, 0x41,
    0x12, 0x0f, 0x0a, 0x0d, 0x08, 0x01, 0x12, 0x09, 0x0a, 0x03, 0x12, 0x01, 0x4d, 0x0a, 0x02, 0x08,
    0x04, 0x62, 0x14, 0x0a, 0x01, 0x59, 0x12, 0x0f, 0x0a, 0x0d, 0x08, 0x01, 0x12, 0x09, 0x0a, 0x03,
    0x12, 0x01, 0x4d, 0x0a, 0x02, 0x08, 0x04, 0x42, 0x04, 0x0a, 0x00, 0x10, 0x11,
};
// clang-format on

// Resolve the CUDA plugin EP shared library path.
std::filesystem::path GetCudaPluginLibraryPath() {
  return GetSharedLibraryFileName(ORT_TSTR("onnxruntime_providers_cuda"));
}

// RAII handle that registers/unregisters the CUDA plugin EP library.
class ScopedCudaPluginRegistration {
 public:
  ScopedCudaPluginRegistration(Ort::Env& env, const char* registration_name)
      : env_(env), name_(registration_name) {
    auto lib_path = GetCudaPluginLibraryPath();
    if (!std::filesystem::exists(lib_path)) {
      available_ = false;
      return;
    }
    env_.RegisterExecutionProviderLibrary(name_.c_str(), lib_path.c_str());
    available_ = true;
  }

  ~ScopedCudaPluginRegistration() {
    if (available_) {
      try {
        env_.UnregisterExecutionProviderLibrary(name_.c_str());
      } catch (...) {
      }
    }
  }

  bool IsAvailable() const { return available_; }

  ScopedCudaPluginRegistration(const ScopedCudaPluginRegistration&) = delete;
  ScopedCudaPluginRegistration& operator=(const ScopedCudaPluginRegistration&) = delete;

 private:
  Ort::Env& env_;
  std::string name_;
  bool available_ = false;
};

// Find the CUDA plugin EP device after registration.
Ort::ConstEpDevice FindCudaPluginDevice(Ort::Env& env) {
  auto ep_devices = env.GetEpDevices();
  for (const auto& device : ep_devices) {
    if (strcmp(device.EpName(), kCudaPluginEpName) == 0) {
      return device;
    }
  }
  return Ort::ConstEpDevice{nullptr};
}

// Returns an empty string if every row of y is [K, K + 1, ..., K + N - 1], or a description of the first mismatch.
std::string CheckGemmBiasOutput(const std::vector<float>& y, int64_t m) {
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < kN; ++col) {
      const float expected = static_cast<float>(kK + col);
      const float actual = y[row * kN + col];
      if (actual != expected) {
        std::ostringstream message;
        message << "M = " << m << ": Y[" << row << "][" << col << "] = " << actual << ", expected " << expected;
        return message.str();
      }
    }
  }
  return {};
}

}  // namespace

class CudaPluginSharedConstOnesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available.";
    }

    registration_ = std::make_unique<ScopedCudaPluginRegistration>(
        *ort_env, kCudaPluginEpRegistrationName);
    if (!registration_->IsAvailable()) {
      GTEST_SKIP() << "CUDA plugin EP library not found.";
    }

    cuda_device_ = FindCudaPluginDevice(*ort_env);
    if (!cuda_device_) {
      GTEST_SKIP() << "No CUDA plugin EP device found after registration.";
    }
  }

  void TearDown() override {
    registration_.reset();
    cudaDeviceSynchronize();
  }

  // Session options that select the plugin EP with its own compute stream. Graph optimizations are disabled to keep
  // the Gemm node, and with it the bias broadcast through the buffer of ones.
  Ort::SessionOptions CreateSessionOptions(cudaStream_t stream, bool enable_cuda_graph) {
    Ort::SessionOptions so;
    so.SetIntraOpNumThreads(1);
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    std::unordered_map<std::string, std::string> provider_options = {
        {"user_compute_stream", std::to_string(reinterpret_cast<uintptr_t>(stream))},
    };
    if (enable_cuda_graph) {
      provider_options["enable_cuda_graph"] = "1";
    }
    so.AppendExecutionProvider_V2(*ort_env, {cuda_device_}, provider_options);
    return so;
  }

  // Run the model with M rows, using CPU input and output, and check the result. Returns an empty string on success,
  // or a description of the failure.
  static std::string RunAndCheck(Ort::Session& session, int64_t m) {
    try {
      auto cpu_memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      std::vector<float> a(m * kK, 1.0f);
      std::vector<float> y(m * kN, 0.0f);
      const std::array<int64_t, 2> a_shape = {m, kK};
      const std::array<int64_t, 2> y_shape = {m, kN};
      Ort::Value a_tensor = Ort::Value::CreateTensor(cpu_memory_info, a.data(), a.size(), a_shape.data(), 2);
      Ort::Value y_tensor = Ort::Value::CreateTensor(cpu_memory_info, y.data(), y.size(), y_shape.data(), 2);
      const char* input_names[] = {"A"};
      const char* output_names[] = {"Y"};
      session.Run(Ort::RunOptions{}, input_names, &a_tensor, 1, output_names, &y_tensor, 1);
      return CheckGemmBiasOutput(y, m);
    } catch (const std::exception& e) {
      return std::string("M = ") + std::to_string(m) + ": exception: " + e.what();
    }
  }

  std::unique_ptr<ScopedCudaPluginRegistration> registration_;
  Ort::ConstEpDevice cuda_device_{nullptr};
};

// Several sessions, each with its own stream and thread, run in lockstep; in every round each thread asks for more
// ones than any thread did in the previous round, so the shared buffer keeps growing while the other threads use it.
TEST_F(CudaPluginSharedConstOnesTest, ConcurrentSessionsGrowTheSharedBuffer) {
  constexpr int kSessions = 4;
  constexpr int kRounds = 50;

  std::array<cudaStream_t, kSessions> streams{};
  std::vector<std::unique_ptr<Ort::Session>> sessions;
  for (int i = 0; i < kSessions; ++i) {
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));
    sessions.push_back(std::make_unique<Ort::Session>(*ort_env, kGemmBiasModel, sizeof(kGemmBiasModel),
                                                      CreateSessionOptions(streams[i], false)));
  }

  std::barrier sync_point(kSessions);
  std::array<std::string, kSessions> first_failure;
  std::array<int, kSessions> failures{};
  std::vector<std::thread> threads;
  for (int t = 0; t < kSessions; ++t) {
    threads.emplace_back([&, t] {
      for (int round = 0; round < kRounds; ++round) {
        sync_point.arrive_and_wait();
        const int64_t m = 1 + static_cast<int64_t>(round) * kSessions + t;
        std::string failure = RunAndCheck(*sessions[t], m);
        if (!failure.empty()) {
          if (failures[t]++ == 0) {
            first_failure[t] = failure;
          }
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }

  for (int t = 0; t < kSessions; ++t) {
    EXPECT_EQ(failures[t], 0) << "thread " << t << " failed " << failures[t] << " of " << kRounds
                              << " runs, the first one with " << first_failure[t];
  }

  sessions.clear();
  for (auto stream : streams) {
    ASSERT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  }
}

// One session captures a CUDA graph that reads the shared buffer, and replays it while another session, in another
// thread and stream, grows the buffer well beyond the size used by the graph.
TEST_F(CudaPluginSharedConstOnesTest, GraphReplayWhileAnotherSessionGrowsTheSharedBuffer) {
  constexpr int64_t kGraphRows = 3;
  constexpr int kWarmupAndCaptureRuns = 3;  // min_num_runs_before_cuda_graph_capture = 2, plus the capture
  constexpr int kReplays = 200;
  constexpr int64_t kMaxGrowRows = int64_t{1} << 18;

  cudaStream_t graph_stream = nullptr;
  cudaStream_t grow_stream = nullptr;
  ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&graph_stream, cudaStreamNonBlocking));
  ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&grow_stream, cudaStreamNonBlocking));

  {
    Ort::Session graph_session(*ort_env, kGemmBiasModel, sizeof(kGemmBiasModel),
                               CreateSessionOptions(graph_stream, true));
    Ort::Session grow_session(*ort_env, kGemmBiasModel, sizeof(kGemmBiasModel),
                              CreateSessionOptions(grow_stream, false));

    // A CUDA graph needs fixed input and output addresses: bind device buffers.
    auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
    auto allocator = ort_env->GetSharedAllocator(device_memory_info);
    ASSERT_NE(allocator, nullptr);
    const std::array<int64_t, 2> a_shape = {kGraphRows, kK};
    const std::array<int64_t, 2> y_shape = {kGraphRows, kN};
    const size_t a_bytes = kGraphRows * kK * sizeof(float);
    const size_t y_bytes = kGraphRows * kN * sizeof(float);
    void* a_gpu = allocator.Alloc(a_bytes);
    void* y_gpu = allocator.Alloc(y_bytes);
    ASSERT_NE(a_gpu, nullptr);
    ASSERT_NE(y_gpu, nullptr);

    // Upload the input once, before any capture.
    const std::vector<float> a(kGraphRows * kK, 1.0f);
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(a_gpu, a.data(), a_bytes, cudaMemcpyHostToDevice, graph_stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(graph_stream));

    Ort::Value a_tensor = Ort::Value::CreateTensor(device_memory_info, static_cast<float*>(a_gpu),
                                                   kGraphRows * kK, a_shape.data(), a_shape.size());
    Ort::Value y_tensor = Ort::Value::CreateTensor(device_memory_info, static_cast<float*>(y_gpu),
                                                   kGraphRows * kN, y_shape.data(), y_shape.size());
    Ort::IoBinding binding(graph_session);
    binding.BindInput("A", a_tensor);
    binding.BindOutput("Y", y_tensor);

    // Run the graph session and check its output. Returns an empty string on success.
    auto run_graph = [&]() -> std::string {
      try {
        graph_session.Run(Ort::RunOptions{}, binding);
        if (cudaStreamSynchronize(graph_stream) != cudaSuccess) {
          return "cudaStreamSynchronize failed";
        }
        std::vector<float> y(kGraphRows * kN, 0.0f);
        if (cudaMemcpy(y.data(), y_gpu, y_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
          return "cudaMemcpy failed";
        }
        return CheckGemmBiasOutput(y, kGraphRows);
      } catch (const std::exception& e) {
        return std::string("exception: ") + e.what();
      }
    };

    // Warm up and capture the graph, with the buffer of ones as it is now.
    for (int i = 0; i < kWarmupAndCaptureRuns; ++i) {
      std::string failure = run_graph();
      ASSERT_TRUE(failure.empty()) << "warm-up or capture run " << i << ": " << failure;
    }

    // Grow the buffer from another thread while the graph is replayed.
    std::atomic<bool> growing{true};
    std::string grow_failure;
    std::thread grow_thread([&] {
      for (int64_t m = 1; m <= kMaxGrowRows && grow_failure.empty(); m = 2 * m + 1) {
        grow_failure = RunAndCheck(grow_session, m);
      }
      growing = false;
    });

    int replays = 0;
    std::string replay_failure;
    while ((growing || replays < kReplays) && replay_failure.empty()) {
      replay_failure = run_graph();
      ++replays;
    }
    grow_thread.join();

    EXPECT_TRUE(grow_failure.empty()) << "growing session: " << grow_failure;
    EXPECT_TRUE(replay_failure.empty()) << "graph replay " << replays << ": " << replay_failure;

    // Replay once more, after the buffer has stopped growing.
    std::string failure = run_graph();
    EXPECT_TRUE(failure.empty()) << "final graph replay: " << failure;

    binding.ClearBoundInputs();
    binding.ClearBoundOutputs();
    allocator.Free(a_gpu);
    allocator.Free(y_gpu);
  }

  ASSERT_EQ(cudaSuccess, cudaStreamDestroy(graph_stream));
  ASSERT_EQ(cudaSuccess, cudaStreamDestroy(grow_stream));
}

}  // namespace test
}  // namespace onnxruntime

#endif  // defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
