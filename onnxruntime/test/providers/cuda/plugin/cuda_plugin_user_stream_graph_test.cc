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
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include "core/graph/onnx_protobuf.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/util/include/file_util.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
namespace {

constexpr const char* kCudaPluginEpRegistrationName = "CudaPluginUserStreamGraphTest";
constexpr const char* kCudaPluginEpName = "CUDAExecutionProvider";

// Resolve the CUDA plugin EP shared library path as an absolute path in the current working
// directory so std::filesystem::exists() and RegisterExecutionProviderLibrary() (dlopen)
// always inspect and load the exact same file.
std::filesystem::path GetCudaPluginLibraryPath() {
  return std::filesystem::absolute(GetSharedLibraryFileName(ORT_TSTR("onnxruntime_providers_cuda")));
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

std::string BuildGatherNDModel() {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  auto* graph = model.mutable_graph();
  graph->set_name("gather_nd");
  auto add_value_info = [graph](const char* name, int32_t element_type,
                                std::initializer_list<int64_t> shape, bool is_input) {
    auto* value_info = is_input ? graph->add_input() : graph->add_output();
    value_info->set_name(name);
    auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(element_type);
    for (const auto dim : shape) {
      tensor_type->mutable_shape()->add_dim()->set_dim_value(dim);
    }
  };
  add_value_info("data", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {2, 2}, true);
  add_value_info("indices", ONNX_NAMESPACE::TensorProto_DataType_INT64, {1, 1}, true);
  add_value_info("output", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, {1, 2}, false);

  auto* node = graph->add_node();
  node->set_name("gather_nd");
  node->set_op_type("GatherND");
  node->add_input("data");
  node->add_input("indices");
  node->add_output("output");

  std::string serialized;
  ORT_ENFORCE(model.SerializeToString(&serialized));
  return serialized;
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

TEST_F(CudaPluginUserStreamGraphTest, GatherNDCudaGraphZeroFillsInvalidIndicesAcrossReplay) {
  Ort::SessionOptions so;
  so.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  const std::unordered_map<std::string, std::string> provider_options{{"enable_cuda_graph", "1"}};
  so.AppendExecutionProvider_V2(*ort_env, {cuda_device_}, provider_options);
  const auto model = BuildGatherNDModel();
  Ort::Session session(*ort_env, model.data(), model.size(), so);

  auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
  auto allocator = ort_env->GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);

  const std::array<int64_t, 2> data_shape{2, 2};
  const std::array<int64_t, 2> indices_shape{1, 1};
  const std::array<int64_t, 2> output_shape{1, 2};
  const std::array<float, 4> data{1.0f, 2.0f, 3.0f, 4.0f};
  std::array<int64_t, 1> indices{1};
  const size_t data_bytes = data.size() * sizeof(float);
  const size_t indices_bytes = indices.size() * sizeof(int64_t);
  constexpr size_t output_element_count = 2;
  constexpr size_t output_bytes = output_element_count * sizeof(float);

  void* data_gpu = allocator.Alloc(data_bytes);
  void* indices_gpu = allocator.Alloc(indices_bytes);
  void* output_gpu = allocator.Alloc(output_bytes);
  ASSERT_NE(data_gpu, nullptr);
  ASSERT_NE(indices_gpu, nullptr);
  ASSERT_NE(output_gpu, nullptr);

  ASSERT_EQ(cudaSuccess, cudaMemcpy(data_gpu, data.data(), data_bytes, cudaMemcpyHostToDevice));
  ASSERT_EQ(cudaSuccess, cudaMemcpy(indices_gpu, indices.data(), indices_bytes, cudaMemcpyHostToDevice));

  Ort::Value data_tensor = Ort::Value::CreateTensor(
      device_memory_info, static_cast<float*>(data_gpu), data.size(),
      data_shape.data(), data_shape.size());
  Ort::Value indices_tensor = Ort::Value::CreateTensor(
      device_memory_info, static_cast<int64_t*>(indices_gpu), indices.size(),
      indices_shape.data(), indices_shape.size());
  Ort::Value output_tensor = Ort::Value::CreateTensor(
      device_memory_info, static_cast<float*>(output_gpu), output_element_count,
      output_shape.data(), output_shape.size());

  Ort::IoBinding binding(session);
  binding.BindInput("data", data_tensor);
  binding.BindInput("indices", indices_tensor);
  binding.BindOutput("output", output_tensor);

  const auto run_and_read_output = [&]() {
    session.Run(Ort::RunOptions{}, binding);
    EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
    std::array<float, output_element_count> output{};
    EXPECT_EQ(cudaSuccess, cudaMemcpy(output.data(), output_gpu, output_bytes, cudaMemcpyDeviceToHost));
    return output;
  };

  auto output = run_and_read_output();
  EXPECT_FLOAT_EQ(output[0], 3.0f);
  EXPECT_FLOAT_EQ(output[1], 4.0f);

  indices[0] = 2;
  ASSERT_EQ(cudaSuccess, cudaMemcpy(indices_gpu, indices.data(), indices_bytes, cudaMemcpyHostToDevice));
  for (int i = 0; i < 3; ++i) {
    output = run_and_read_output();
    EXPECT_FLOAT_EQ(output[0], 0.0f) << "mismatch at invalid-index iteration " << i;
    EXPECT_FLOAT_EQ(output[1], 0.0f) << "mismatch at invalid-index iteration " << i;
  }

  binding.ClearBoundInputs();
  binding.ClearBoundOutputs();
  allocator.Free(data_gpu);
  allocator.Free(indices_gpu);
  allocator.Free(output_gpu);
}

// Full capture + replay on the user stream, including replay after an in-place input
// update. mul_1.onnx computes Y = X * W with W = [1, 2, 3, 4, 5, 6] (shape 3x2).
TEST_F(CudaPluginUserStreamGraphTest, CaptureAndReplayOnUserStream) {
  cudaStream_t user_stream = nullptr;
  ASSERT_EQ(cudaSuccess, cudaStreamCreate(&user_stream));

  {
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
  }

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

  {
    Ort::SessionOptions so = CreateUserStreamGraphSessionOptions(user_stream);
    Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), so);

    // Alternate between annotation ids "1" and "2". With min_num_runs_before_cuda_graph_capture == 2,
    // 8 iterations let each id accumulate warmup runs, capture, and then replay on the user stream.
    RunAndVerifyOnStream(session, user_stream, /*iterations=*/8, /*graph_ids=*/{"1", "2"});
  }

  ASSERT_EQ(cudaSuccess, cudaStreamDestroy(user_stream));
}

// GPU integration test: observably proves capture of IDs 1 and 2, retirement of ID 1 via
// Session::ReleaseCapturedGraph(1) (distinguishing retirement from a no-op release), preservation
// of ID 2's captured graph across ID 1's release, and recapture of ID 1 on newly bound device
// buffer addresses while keeping all device buffers and bindings alive until teardown.
TEST_F(CudaPluginUserStreamGraphTest, ReleaseCapturedGraphAndRecaptureOnUserStream) {
#ifdef __linux__
  {
    const auto expected_plugin_realpath = std::filesystem::canonical(GetCudaPluginLibraryPath()).string();
    std::ifstream maps("/proc/self/maps");
    ASSERT_TRUE(maps.is_open());
    bool found_expected_plugin = false;
    std::string line;
    while (std::getline(maps, line)) {
      if (line.find("libonnxruntime_providers_cuda.so") == std::string::npos) {
        continue;
      }
      if (line.find(expected_plugin_realpath) != std::string::npos) {
        found_expected_plugin = true;
      } else {
        FAIL() << "Unexpected duplicate/mismatched libonnxruntime_providers_cuda.so mapping in /proc/self/maps: "
               << line << " (expected " << expected_plugin_realpath << ")";
      }
    }
    ASSERT_TRUE(found_expected_plugin)
        << "Expected candidate plugin not found in /proc/self/maps: " << expected_plugin_realpath;
  }
#endif

  cudaStream_t user_stream = nullptr;
  ASSERT_EQ(cudaSuccess, cudaStreamCreate(&user_stream));

  {
    Ort::SessionOptions so = CreateUserStreamGraphSessionOptions(user_stream);
    Ort::Session session(*ort_env, ORT_TSTR("testdata/mul_1.onnx"), so);

    auto device_memory_info = cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);
    auto allocator = ort_env->GetSharedAllocator(device_memory_info);
    ASSERT_NE(allocator, nullptr);

    constexpr size_t kNumElements = 6;
    constexpr size_t kBytes = kNumElements * sizeof(float);
    const std::array<int64_t, 2> shape = {3, 2};
    const std::array<float, kNumElements> w_values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::array<float, kNumElements> x_1a = {2.0f, 3.0f, 5.0f, 7.0f, 11.0f, 13.0f};
    const std::array<float, kNumElements> x_2 = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f};
    const std::array<float, kNumElements> x_1b = {100.0f, 200.0f, 300.0f, 400.0f, 500.0f, 600.0f};

    // Allocate three distinct device buffer pairs and keep all of them alive until test teardown
    // so that even in a negative-control no-op release case, replaying a retained graph never
    // accesses freed device memory.
    void* input_gpu_1a = allocator.Alloc(kBytes);
    void* output_gpu_1a = allocator.Alloc(kBytes);
    void* input_gpu_2 = allocator.Alloc(kBytes);
    void* output_gpu_2 = allocator.Alloc(kBytes);
    void* input_gpu_1b = allocator.Alloc(kBytes);
    void* output_gpu_1b = allocator.Alloc(kBytes);
    ASSERT_NE(input_gpu_1a, nullptr);
    ASSERT_NE(output_gpu_1a, nullptr);
    ASSERT_NE(input_gpu_2, nullptr);
    ASSERT_NE(output_gpu_2, nullptr);
    ASSERT_NE(input_gpu_1b, nullptr);
    ASSERT_NE(output_gpu_1b, nullptr);
    ASSERT_NE(output_gpu_1a, output_gpu_2);
    ASSERT_NE(output_gpu_1a, output_gpu_1b);
    ASSERT_NE(output_gpu_2, output_gpu_1b);

    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(input_gpu_1a, x_1a.data(), kBytes, cudaMemcpyHostToDevice, user_stream));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(input_gpu_2, x_2.data(), kBytes, cudaMemcpyHostToDevice, user_stream));
    ASSERT_EQ(cudaSuccess,
              cudaMemcpyAsync(input_gpu_1b, x_1b.data(), kBytes, cudaMemcpyHostToDevice, user_stream));
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(user_stream));

    auto make_tensor = [&](void* ptr) {
      return Ort::Value::CreateTensor(
          device_memory_info, reinterpret_cast<float*>(ptr), kNumElements, shape.data(), shape.size());
    };
    Ort::Value in_tensor_1a = make_tensor(input_gpu_1a);
    Ort::Value out_tensor_1a = make_tensor(output_gpu_1a);
    Ort::Value in_tensor_2 = make_tensor(input_gpu_2);
    Ort::Value out_tensor_2 = make_tensor(output_gpu_2);
    Ort::Value in_tensor_1b = make_tensor(input_gpu_1b);
    Ort::Value out_tensor_1b = make_tensor(output_gpu_1b);

    Ort::IoBinding binding_1a(session);
    binding_1a.BindInput("X", in_tensor_1a);
    binding_1a.BindOutput("Y", out_tensor_1a);

    Ort::IoBinding binding_2(session);
    binding_2.BindInput("X", in_tensor_2);
    binding_2.BindOutput("Y", out_tensor_2);

    Ort::IoBinding binding_1b(session);
    binding_1b.BindInput("X", in_tensor_1b);
    binding_1b.BindOutput("Y", out_tensor_1b);

    auto fill_device_buffer = [&](void* dev_ptr, float sentinel) {
      std::array<float, kNumElements> fill_vals{};
      fill_vals.fill(sentinel);
      ASSERT_EQ(cudaSuccess,
                cudaMemcpyAsync(dev_ptr, fill_vals.data(), kBytes, cudaMemcpyHostToDevice, user_stream));
      ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(user_stream));
    };

    auto read_device_buffer = [&](void* dev_ptr) {
      std::array<float, kNumElements> host_vals{};
      EXPECT_EQ(cudaSuccess, cudaMemcpy(host_vals.data(), dev_ptr, kBytes, cudaMemcpyDeviceToHost));
      return host_vals;
    };

    auto run_with_id = [&](const char* graph_id, Ort::IoBinding& binding) {
      Ort::RunOptions run_options;
      run_options.AddConfigEntry("gpu_graph_id", graph_id);
      session.Run(run_options, binding);
      ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(user_stream));
    };

    // 1. Capture ID 1 on binding_1a and ID 2 on binding_2 (InferenceSession::RunImpl internally
    //    loops warm-up and capture runs within the first Session::Run call when enable_cuda_graph == 1).
    fill_device_buffer(output_gpu_1a, -1.0f);
    run_with_id("1", binding_1a);
    auto y_1a = read_device_buffer(output_gpu_1a);
    for (size_t j = 0; j < kNumElements; ++j) {
      EXPECT_FLOAT_EQ(y_1a[j], x_1a[j] * w_values[j]);
    }

    fill_device_buffer(output_gpu_2, -1.0f);
    run_with_id("2", binding_2);
    auto y_2 = read_device_buffer(output_gpu_2);
    for (size_t j = 0; j < kNumElements; ++j) {
      EXPECT_FLOAT_EQ(y_2[j], x_2[j] * w_values[j]);
    }

    // Prove ID 1 and ID 2 are both captured: when IsGraphCaptured(id) is true, InferenceSession::RunImpl
    // takes the ReplayGraph(id) branch (which replays the baked device addresses and ignores the passed
    // IoBinding's tensor addresses).
    fill_device_buffer(output_gpu_1a, -111.0f);
    fill_device_buffer(output_gpu_2, -222.0f);
    run_with_id("1", binding_2);
    y_1a = read_device_buffer(output_gpu_1a);
    y_2 = read_device_buffer(output_gpu_2);
    for (size_t j = 0; j < kNumElements; ++j) {
      EXPECT_FLOAT_EQ(y_1a[j], x_1a[j] * w_values[j]) << "ID 1 did not replay into its captured buffer 1a";
      EXPECT_FLOAT_EQ(y_2[j], -222.0f) << "ID 1 was not captured (executed on binding_2 instead of replaying 1a)";
    }

    fill_device_buffer(output_gpu_2, -222.0f);
    fill_device_buffer(output_gpu_1a, -111.0f);
    run_with_id("2", binding_1a);
    y_2 = read_device_buffer(output_gpu_2);
    y_1a = read_device_buffer(output_gpu_1a);
    for (size_t j = 0; j < kNumElements; ++j) {
      EXPECT_FLOAT_EQ(y_2[j], x_2[j] * w_values[j]) << "ID 2 did not replay into its captured buffer 2";
      EXPECT_FLOAT_EQ(y_1a[j], -111.0f) << "ID 2 was not captured (executed on binding_1a instead of replaying 2)";
    }

    // 2. Release ID 1 and prove ID 2 stayed captured (replaying ID 2 with binding_1a still writes
    //    to output_gpu_2 and leaves output_gpu_1a untouched).
    session.ReleaseCapturedGraph(1);

    fill_device_buffer(output_gpu_2, -333.0f);
    fill_device_buffer(output_gpu_1a, -444.0f);
    run_with_id("2", binding_1a);
    y_2 = read_device_buffer(output_gpu_2);
    y_1a = read_device_buffer(output_gpu_1a);
    for (size_t j = 0; j < kNumElements; ++j) {
      EXPECT_FLOAT_EQ(y_2[j], x_2[j] * w_values[j]) << "ID 2 failed to replay after releasing ID 1";
      EXPECT_FLOAT_EQ(y_1a[j], -444.0f) << "ID 2 lost its captured state when ID 1 was released";
    }

    // 3. Prove ID 1 was retired (not a no-op) and recaptures on binding_1b:
    //    Run ID 1 with binding_1b (input_gpu_1b -> output_gpu_1b). If ReleaseCapturedGraph(1) was a
    //    no-op, ReplayGraph(1) would replay the old graph into output_gpu_1a and leave output_gpu_1b
    //    at -999.0f. With real retirement, output_gpu_1a stays at -777.0f and output_gpu_1b receives
    //    x_1b * w_values.
    fill_device_buffer(output_gpu_1a, -777.0f);
    fill_device_buffer(output_gpu_1b, -999.0f);
    run_with_id("1", binding_1b);
    y_1a = read_device_buffer(output_gpu_1a);
    auto y_1b = read_device_buffer(output_gpu_1b);
    for (size_t j = 0; j < kNumElements; ++j) {
      EXPECT_FLOAT_EQ(y_1a[j], -777.0f)
          << "Stale graph for ID 1 was replayed into output_gpu_1a after ReleaseCapturedGraph(1)";
      EXPECT_FLOAT_EQ(y_1b[j], x_1b[j] * w_values[j])
          << "Released ID 1 failed to execute and recapture on new binding_1b";
    }

    // Prove ID 1 is now recaptured on binding_1b: running ID 1 while passing binding_1a must take
    // the ReplayGraph(1) fast path and write x_1b * w_values into output_gpu_1b while leaving
    // output_gpu_1a untouched at -777.0f.
    fill_device_buffer(output_gpu_1a, -777.0f);
    fill_device_buffer(output_gpu_1b, -888.0f);
    run_with_id("1", binding_1a);
    y_1a = read_device_buffer(output_gpu_1a);
    y_1b = read_device_buffer(output_gpu_1b);
    for (size_t j = 0; j < kNumElements; ++j) {
      EXPECT_FLOAT_EQ(y_1b[j], x_1b[j] * w_values[j]) << "Recaptured ID 1 did not replay into output_gpu_1b";
      EXPECT_FLOAT_EQ(y_1a[j], -777.0f) << "Recaptured ID 1 did not take ReplayGraph(1) fast path";
    }

    // 4. Release remaining captured graphs (including idempotent repeat release) and free resources.
    session.ReleaseCapturedGraph(1);
    session.ReleaseCapturedGraph(1);
    session.ReleaseCapturedGraph(2);

    binding_1a.ClearBoundInputs();
    binding_1a.ClearBoundOutputs();
    binding_2.ClearBoundInputs();
    binding_2.ClearBoundOutputs();
    binding_1b.ClearBoundInputs();
    binding_1b.ClearBoundOutputs();
    allocator.Free(input_gpu_1a);
    allocator.Free(output_gpu_1a);
    allocator.Free(input_gpu_2);
    allocator.Free(output_gpu_2);
    allocator.Free(input_gpu_1b);
    allocator.Free(output_gpu_1b);
  }

  ASSERT_EQ(cudaSuccess, cudaStreamDestroy(user_stream));
}

}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP
