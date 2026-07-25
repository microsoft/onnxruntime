// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests that the CUDA plugin EP supports combining a user-provided compute stream
// (user_compute_stream) with CUDA graph capture/replay (enable_cuda_graph).
//
// Historically the plugin EP rejected this combination with ORT_INVALID_ARGUMENT.
// It now captures and replays the CUDA graph on the user-provided stream (the same
// stream the kernels are issued to), matching the bundled CUDA EP behavior. These
// tests verify:
//   1. Session creation succeeds with both options set (regression for the removed
//      validation).
//   2. Capture + replay on the user stream produce correct results.
//   3. Replay after an in-place input update (on the user stream) is correct.

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
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

constexpr const char* kCudaPluginEpRegistrationName = "CudaPluginUserStreamGraphTest";
constexpr const char* kCudaPluginEpName = "CUDAExecutionProvider";

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

// Dummy external allocator callbacks. They are only used to make the external-allocator
// configuration non-null; the plugin EP rejects the combination with user_compute_stream
// before either is ever invoked.
void* DummyExternalAlloc(size_t /*size*/) { return nullptr; }
void DummyExternalFree(void* /*ptr*/) {}

}  // namespace

class CudaPluginUserStreamGraphTest : public ::testing::Test {
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

  // Build session options that select the plugin EP with CUDA graph capture enabled
  // and the user-provided stream supplied as a pointer-sized address string.
  Ort::SessionOptions CreateUserStreamGraphSessionOptions(cudaStream_t user_stream) {
    Ort::SessionOptions so;
    std::unordered_map<std::string, std::string> provider_options = {
        {"enable_cuda_graph", "1"},
        {"user_compute_stream",
         std::to_string(reinterpret_cast<uintptr_t>(user_stream))},
    };
    so.AppendExecutionProvider_V2(*ort_env, {cuda_device_}, provider_options);
    return so;
  }

  // Allocate device input/output, bind them, and run `iterations` times on `stream`, verifying
  // Y = X * W each run. The input is uploaded once up front and then left constant: when CUDA graph
  // capture is enabled, issuing host->device work on the stream immediately before the capture run
  // would interfere with cudaStreamBeginCapture, so the buffers are populated and synchronized
  // before any capture happens. When `graph_ids` is non-empty, run i sets gpu_graph_id to
  // graph_ids[i % size] to exercise CUDA graph annotation-id switching. mul_1.onnx computes
  // Y = X * W with W = [1..6] (shape 3x2).
  void RunAndVerifyOnStream(Ort::Session& session, cudaStream_t stream, int iterations,
                            const std::vector<std::string>& graph_ids = {}) {
    auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
    auto allocator = ort_env->GetSharedAllocator(device_memory_info);
    ASSERT_NE(allocator, nullptr);

    constexpr size_t kNumElements = 6;
    constexpr size_t kBytes = kNumElements * sizeof(float);
    const std::array<int64_t, 2> shape = {3, 2};
    const std::array<float, kNumElements> w_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::array<float, kNumElements> x_values = {2.0f, 3.0f, 5.0f, 7.0f, 11.0f, 13.0f};

    // Fixed device buffers so captured CUDA graphs keep valid IO addresses across replays.
    void* input_gpu = allocator.Alloc(kBytes);
    void* output_gpu = allocator.Alloc(kBytes);
    ASSERT_NE(input_gpu, nullptr);
    ASSERT_NE(output_gpu, nullptr);

    // Populate the input once and synchronize, so no host-issued work is pending on `stream`
    // when graph capture begins on a later run.
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(input_gpu, x_values.data(), kBytes, cudaMemcpyHostToDevice, stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

    Ort::Value input_tensor = Ort::Value::CreateTensor(
        device_memory_info, reinterpret_cast<float*>(input_gpu), kNumElements,
        shape.data(), shape.size());
    Ort::Value output_tensor = Ort::Value::CreateTensor(
        device_memory_info, reinterpret_cast<float*>(output_gpu), kNumElements,
        shape.data(), shape.size());

    Ort::IoBinding binding(session);
    binding.BindInput("X", input_tensor);
    binding.BindOutput("Y", output_tensor);

    for (int i = 0; i < iterations; ++i) {
      Ort::RunOptions run_options;
      if (!graph_ids.empty()) {
        run_options.AddConfigEntry("gpu_graph_id", graph_ids[i % graph_ids.size()].c_str());
      }
      session.Run(run_options, binding);

      // Kernels run on `stream`; wait for them before copying the result back.
      ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
      std::array<float, kNumElements> y{};
      ASSERT_EQ(cudaSuccess, cudaMemcpy(y.data(), output_gpu, kBytes, cudaMemcpyDeviceToHost));
      for (size_t j = 0; j < kNumElements; ++j) {
        EXPECT_FLOAT_EQ(y[j], x_values[j] * w_values[j]) << "mismatch at iteration " << i << " index " << j;
      }
    }

    binding.ClearBoundInputs();
    binding.ClearBoundOutputs();
    allocator.Free(input_gpu);
    allocator.Free(output_gpu);
  }

  std::unique_ptr<ScopedCudaPluginRegistration> registration_;
  Ort::ConstEpDevice cuda_device_{nullptr};
};

// Regression: creating a session with both user_compute_stream and enable_cuda_graph
// used to fail with ORT_INVALID_ARGUMENT. It must now succeed.
TEST_F(CudaPluginUserStreamGraphTest, SessionCreatesWithUserStreamAndCudaGraph) {
  cudaStream_t user_stream = nullptr;
  ASSERT_EQ(cudaSuccess, cudaStreamCreate(&user_stream));

  {
    Ort::SessionOptions so = CreateUserStreamGraphSessionOptions(user_stream);
    ASSERT_NO_THROW({
      Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), so);
      (void)session;
    });
  }

  ASSERT_EQ(cudaSuccess, cudaStreamDestroy(user_stream));
}

// Full capture + replay on the user stream, including replay after an in-place input
// update. mul_1.onnx computes Y = X * W with W = [1, 2, 3, 4, 5, 6] (shape 3x2).
TEST_F(CudaPluginUserStreamGraphTest, CaptureAndReplayOnUserStream) {
  cudaStream_t user_stream = nullptr;
  ASSERT_EQ(cudaSuccess, cudaStreamCreate(&user_stream));

  Ort::SessionOptions so = CreateUserStreamGraphSessionOptions(user_stream);
  Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), so);

  // Device allocator backing the plugin EP's default memory.
  auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  auto allocator = ort_env->GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);

  constexpr size_t kNumElements = 6;
  constexpr size_t kBytes = kNumElements * sizeof(float);
  const std::array<int64_t, 2> shape = {3, 2};
  const std::array<float, kNumElements> w_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

  // Pre-allocate device input/output buffers (required for CUDA graph IO binding).
  void* input_gpu = allocator.Alloc(kBytes);
  void* output_gpu = allocator.Alloc(kBytes);
  ASSERT_NE(input_gpu, nullptr);
  ASSERT_NE(output_gpu, nullptr);

  auto upload_input = [&](const std::array<float, kNumElements>& host_values) {
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(input_gpu, host_values.data(), kBytes,
                              cudaMemcpyHostToDevice, user_stream));
  };

  auto read_output = [&](std::array<float, kNumElements>& host_values) {
    // Kernels run on the user stream; wait for them before copying the result back.
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(user_stream));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpy(host_values.data(), output_gpu, kBytes, cudaMemcpyDeviceToHost));
  };

  Ort::Value input_tensor = Ort::Value::CreateTensor(
      device_memory_info, reinterpret_cast<float*>(input_gpu), kNumElements,
      shape.data(), shape.size());
  Ort::Value output_tensor = Ort::Value::CreateTensor(
      device_memory_info, reinterpret_cast<float*>(output_gpu), kNumElements,
      shape.data(), shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("X", input_tensor);
  binding.BindOutput("Y", output_tensor);

  // First run: warmup + capture + first replay on the user stream.
  const std::array<float, kNumElements> x0 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  upload_input(x0);
  session.Run(Ort::RunOptions{}, binding);

  std::array<float, kNumElements> y{};
  read_output(y);
  for (size_t i = 0; i < kNumElements; ++i) {
    EXPECT_FLOAT_EQ(y[i], x0[i] * w_values[i]) << "capture mismatch at " << i;
  }

  // Second run: pure graph replay (same inputs) on the user stream.
  session.Run(Ort::RunOptions{}, binding);
  read_output(y);
  for (size_t i = 0; i < kNumElements; ++i) {
    EXPECT_FLOAT_EQ(y[i], x0[i] * w_values[i]) << "replay mismatch at " << i;
  }

  // Update the input in place on the user stream and replay again.
  const std::array<float, kNumElements> x1 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
  upload_input(x1);
  session.Run(Ort::RunOptions{}, binding);
  read_output(y);
  for (size_t i = 0; i < kNumElements; ++i) {
    EXPECT_FLOAT_EQ(y[i], x1[i] * w_values[i]) << "updated-input replay mismatch at " << i;
  }

  binding.ClearBoundInputs();
  binding.ClearBoundOutputs();
  allocator.Free(input_gpu);
  allocator.Free(output_gpu);

  ASSERT_EQ(cudaSuccess, cudaStreamDestroy(user_stream));
}

// Negative: a user_compute_stream combined with an external GPU allocator
// (gpu_external_alloc/gpu_external_free) is not supported and must be rejected at session
// creation with an error rather than silently ignored.
TEST_F(CudaPluginUserStreamGraphTest, RejectsUserStreamWithExternalAllocator) {
  cudaStream_t user_stream = nullptr;
  ASSERT_EQ(cudaSuccess, cudaStreamCreate(&user_stream));

  Ort::SessionOptions so;
  std::unordered_map<std::string, std::string> provider_options = {
      {"user_compute_stream", std::to_string(reinterpret_cast<uintptr_t>(user_stream))},
      {"gpu_external_alloc", std::to_string(reinterpret_cast<uintptr_t>(&DummyExternalAlloc))},
      {"gpu_external_free", std::to_string(reinterpret_cast<uintptr_t>(&DummyExternalFree))},
  };
  so.AppendExecutionProvider_V2(*ort_env, {cuda_device_}, provider_options);

  EXPECT_THROW(
      {
        Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), so);
        (void)session;
      },
      Ort::Exception);

  ASSERT_EQ(cudaSuccess, cudaStreamDestroy(user_stream));
}

// Edge case: cudaStream_t(0) (the CUDA default stream) is a valid user-provided stream. Because
// user_compute_stream parses to nullptr, the caller must set has_user_compute_stream explicitly,
// otherwise the stream would be treated as "not provided". Session creation must succeed and
// inference must run correctly on the default stream.
//
// Note: CUDA graph capture is intentionally NOT enabled here. The legacy default stream (stream 0)
// cannot be captured (cudaStreamBeginCapture returns cudaErrorStreamCaptureUnsupported), so this
// test exercises only that stream 0 is honored as the compute stream for non-graph execution.
TEST_F(CudaPluginUserStreamGraphTest, DefaultStreamAsUserStream) {
  Ort::SessionOptions so;
  std::unordered_map<std::string, std::string> provider_options = {
      {"has_user_compute_stream", "1"},
      {"user_compute_stream", "0"},
  };
  so.AppendExecutionProvider_V2(*ort_env, {cuda_device_}, provider_options);

  Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), so);

  // Run several iterations on the default stream (stream 0) and verify correctness.
  RunAndVerifyOnStream(session, /*stream=*/nullptr, /*iterations=*/4);
}

// Switching the CUDA graph annotation id (gpu_graph_id) between runs while using a user stream
// must capture/replay a distinct graph per id without crashing and keep producing correct results.
TEST_F(CudaPluginUserStreamGraphTest, GraphAnnotationIdSwitchingWithUserStream) {
  cudaStream_t user_stream = nullptr;
  ASSERT_EQ(cudaSuccess, cudaStreamCreate(&user_stream));

  Ort::SessionOptions so = CreateUserStreamGraphSessionOptions(user_stream);
  Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), so);

  // Alternate between annotation ids "1" and "2". With min_num_runs_before_cuda_graph_capture == 2,
  // 8 iterations let each id accumulate warmup runs, capture, and then replay on the user stream.
  RunAndVerifyOnStream(session, user_stream, /*iterations=*/8, /*graph_ids=*/{"1", "2"});

  ASSERT_EQ(cudaSuccess, cudaStreamDestroy(user_stream));
}

}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP
