// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA) || defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)

#include <chrono>
#include <cstring>
#include <filesystem>
#include <future>
#include <string>
#include <unordered_map>
#include <cuda_runtime_api.h>
#include "gtest/gtest.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/util/include/file_util.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime::test {
namespace {

std::string BuildGemmBiasModel() {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  model.add_opset_import()->set_version(13);
  auto& graph = *model.mutable_graph();
  graph.set_name("gemm_row_and_column_bias");
  auto set_value = [](ONNX_NAMESPACE::ValueInfoProto* value, const char* name,
                      std::initializer_list<const char*> dimensions) {
    value->set_name(name);
    auto* type = value->mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    for (const char* dimension : dimensions) {
      auto* dim = type->mutable_shape()->add_dim();
      if (strcmp(dimension, "1") == 0) {
        dim->set_dim_value(1);
      } else {
        dim->set_dim_param(dimension);
      }
    }
  };
  set_value(graph.add_input(), "A", {"M", "K"});
  set_value(graph.add_input(), "B", {"K", "N"});
  set_value(graph.add_input(), "row_bias", {"N"});
  set_value(graph.add_input(), "column_bias", {"M", "1"});
  for (const char* bias : {"row_bias", "column_bias"}) {
    auto* node = graph.add_node();
    node->set_op_type("Gemm");
    node->add_input("A");
    node->add_input("B");
    node->add_input(bias);
    std::string output = std::string(bias) + "_output";
    node->add_output(output);
    set_value(graph.add_output(), output.c_str(), {"M", "N"});
    for (const auto& [name, value] : {std::pair{"alpha", 0.5f}, {"beta", -2.0f}}) {
      auto* attribute = node->add_attribute();
      attribute->set_name(name);
      attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
      attribute->set_f(value);
    }
  }
  return model.SerializeAsString();
}

class ScopedGemmBiasStream {
 public:
  ScopedGemmBiasStream() {
    ORT_ENFORCE(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess);
  }
  ~ScopedGemmBiasStream() { (void)cudaStreamDestroy(stream); }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScopedGemmBiasStream);
  cudaStream_t stream = nullptr;
};

// Each captured shape keeps its own device IO alive, including when other graph IDs or sessions run.
class GemmBiasBinding {
 public:
  GemmBiasBinding(Ort::Session& session, OrtAllocator* allocator, cudaStream_t stream, int64_t m, int64_t n)
      : session_(session), m_(m), n_(n), binding_(session) {
    tensors_.reserve(6);
    for (const auto& shape : {TensorShapeVector{m, 4}, {4, n}, {n}, {m, 1}, {m, n}, {m, n}}) {
      tensors_.push_back(Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size()));
    }
    InlinedVector<float> b, row_bias, column_bias;
    b.reserve(4 * n);
    row_bias.reserve(n);
    column_bias.reserve(m);
    for (int64_t i = 0; i < 4 * n; ++i) b.push_back(static_cast<float>((i % n) % 5 - 2));
    for (int64_t i = 0; i < n; ++i) row_bias.push_back(static_cast<float>(i % 7) + 0.25f);
    for (int64_t i = 0; i < m; ++i) column_bias.push_back(static_cast<float>(i % 5) + 0.5f);
    Upload(1, b, stream);
    Upload(2, row_bias, stream);
    Upload(3, column_bias, stream);
    ORT_ENFORCE(cudaStreamSynchronize(stream) == cudaSuccess);
    binding_.BindInput("A", tensors_[0]);
    binding_.BindInput("B", tensors_[1]);
    binding_.BindInput("row_bias", tensors_[2]);
    binding_.BindInput("column_bias", tensors_[3]);
    binding_.BindOutput("row_bias_output", tensors_[4]);
    binding_.BindOutput("column_bias_output", tensors_[5]);
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GemmBiasBinding);

  void RunAndVerify(cudaStream_t stream, const char* graph_id, float offset) {
    InlinedVector<float> a;
    a.reserve(m_ * 4);
    for (int64_t i = 0; i < m_ * 4; ++i) a.push_back(static_cast<float>((i / 4) % 3 - 1) + offset);
    Upload(0, a, stream);
    for (size_t output : {4u, 5u}) {
      ASSERT_EQ(cudaSuccess, cudaMemsetAsync(tensors_[output].GetTensorMutableData<float>(), 0xff,
                                             m_ * n_ * sizeof(float), stream));
    }
    // Capture must start with no pending host-issued copies on the user stream.
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    Ort::RunOptions options;
    options.AddConfigEntry("gpu_graph_id", graph_id);
    session_.Run(options, binding_);
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    for (size_t output : {4u, 5u}) {
      InlinedVector<float> actual(static_cast<size_t>(m_ * n_));
      ASSERT_EQ(cudaSuccess, cudaMemcpy(actual.data(), tensors_[output].GetTensorMutableData<float>(),
                                        actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
      for (int64_t row = 0; row < m_; ++row) {
        for (int64_t col = 0; col < n_; ++col) {
          const float product = 2.0f * (static_cast<float>(row % 3 - 1) + offset) *
                                static_cast<float>(col % 5 - 2);
          const float bias = output == 4 ? static_cast<float>(col % 7) + 0.25f
                                         : static_cast<float>(row % 5) + 0.5f;
          ASSERT_FLOAT_EQ(actual[row * n_ + col], product - 2.0f * bias)
              << "graph=" << graph_id << " output=" << output << " M=" << m_ << " N=" << n_
              << " row=" << row << " column=" << col;
        }
      }
    }
  }

 private:
  void Upload(size_t index, gsl::span<const float> data, cudaStream_t stream) {
    ORT_ENFORCE(cudaMemcpyAsync(tensors_[index].GetTensorMutableData<float>(), data.data(),
                                data.size_bytes(), cudaMemcpyHostToDevice, stream) == cudaSuccess);
  }
  Ort::Session& session_;
  int64_t m_, n_;
  InlinedVector<Ort::Value> tensors_;
  Ort::IoBinding binding_;
};

class CudaGemmBiasSessionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available.";
    }
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
    auto path = GetSharedLibraryFileName(ORT_TSTR("onnxruntime_providers_cuda"));
    if (!std::filesystem::exists(path)) {
      GTEST_SKIP() << "CUDA plugin EP library not found.";
    }
    ort_env->RegisterExecutionProviderLibrary(kRegistrationName, path.c_str());
    registered_ = true;
    for (const auto& device : ort_env->GetEpDevices()) {
      if (strcmp(device.EpName(), "CUDAExecutionProvider") == 0) {
        cuda_device_ = device;
        break;
      }
    }
    ASSERT_TRUE(cuda_device_) << "CUDA plugin did not expose a device.";
#endif
  }

  void TearDown() override {
#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
    if (registered_) {
      EXPECT_NO_THROW(ort_env->UnregisterExecutionProviderLibrary(kRegistrationName));
    }
#endif
  }

  Ort::Session CreateSession(cudaStream_t stream, bool enable_graph) {
    Ort::SessionOptions options;
    options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    std::unordered_map<std::string, std::string> provider_options = {
        {"enable_cuda_graph", enable_graph ? "1" : "0"}};
#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
    provider_options["user_compute_stream"] = std::to_string(reinterpret_cast<uintptr_t>(stream));
    options.AppendExecutionProvider_V2(*ort_env, {cuda_device_}, provider_options);
#else
    Ort::CUDAProviderOptions cuda_options;
    cuda_options.Update(provider_options);
    cuda_options.UpdateWithValue("user_compute_stream", stream);
    options.AppendExecutionProvider_CUDA_V2(*cuda_options);
#endif
    const auto model = BuildGemmBiasModel();
    return Ort::Session(*ort_env, model.data(), model.size(), options);
  }

  auto CreateAllocator(Ort::Session& session) {
#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
    (void)session;
    return ort_env->GetSharedAllocator(cuda_device_.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT));
#else
    Ort::MemoryInfo info("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);
    return Ort::Allocator(session, info);
#endif
  }

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP)
  static constexpr const char* kRegistrationName = "CudaGemmBiasSessionTest";
  bool registered_ = false;
  Ort::ConstEpDevice cuda_device_{nullptr};
#endif
};

TEST_F(CudaGemmBiasSessionTest, ConcurrentSessionsGrowingDimensions) {
  ScopedGemmBiasStream first_stream, second_stream;
  auto first = CreateSession(first_stream.stream, false);
  auto second = CreateSession(second_stream.stream, false);
  auto allocator = CreateAllocator(first);
  ASSERT_NE(allocator, nullptr);
  std::promise<void> start;
  auto ready = start.get_future().share();
  auto run = [&](Ort::Session& session, cudaStream_t stream, int64_t delta) {
    ready.wait();
    ORT_ENFORCE(cudaSetDevice(0) == cudaSuccess);
    for (int64_t size : {17, 65, 257, 1025}) {
      GemmBiasBinding binding(session, allocator, stream, size + delta, size + 4 + delta);
      for (int iteration = 0; iteration < 3; ++iteration) {
        binding.RunAndVerify(stream, "-1", static_cast<float>(iteration + delta));
      }
    }
  };
  auto first_run = std::async(std::launch::async, [&] { run(first, first_stream.stream, 0); });
  auto second_run = std::async(std::launch::async, [&] { run(second, second_stream.stream, 8); });
  start.set_value();
  EXPECT_NO_THROW(first_run.get());
  EXPECT_NO_THROW(second_run.get());
}

TEST_F(CudaGemmBiasSessionTest, GraphReplayAfterLargerSession) {
  ScopedGemmBiasStream capture_stream, other_stream;
  auto captured = CreateSession(capture_stream.stream, true);
  auto other = CreateSession(other_stream.stream, false);
  auto allocator = CreateAllocator(captured);
  ASSERT_NE(allocator, nullptr);
  GemmBiasBinding small_binding(captured, allocator, capture_stream.stream, 17, 19);
  GemmBiasBinding medium(captured, allocator, capture_stream.stream, 65, 67);
  for (int iteration = 0; iteration < 4; ++iteration) {
    small_binding.RunAndVerify(capture_stream.stream, "1", static_cast<float>(iteration));
  }
  for (int iteration = 0; iteration < 4; ++iteration) {
    medium.RunAndVerify(capture_stream.stream, "2", static_cast<float>(iteration));
  }
  GemmBiasBinding large(other, allocator, other_stream.stream, 2049, 2051);
  large.RunAndVerify(other_stream.stream, "-1", 7.0f);
  // Old captures must neither reference freed shared ones buffers nor depend on another stream's initialization.
  small_binding.RunAndVerify(capture_stream.stream, "1", 5.0f);
  medium.RunAndVerify(capture_stream.stream, "2", 6.0f);
  small_binding.RunAndVerify(capture_stream.stream, "1", -1.0f);

  std::promise<void> start;
  auto ready = start.get_future();
  auto grow = std::async(std::launch::async, [&] {
    ready.wait();
    ORT_ENFORCE(cudaSetDevice(0) == cudaSuccess);
    constexpr int64_t size = 2053;
    GemmBiasBinding growing(other, allocator, other_stream.stream, size, size + 2);
    growing.RunAndVerify(other_stream.stream, "-1", 3.0f);
  });
  start.set_value();
  // CUDA graph caches are per-thread: replay on the capturing thread while the other session grows.
  do {
    small_binding.RunAndVerify(capture_stream.stream, "1", 2.0f);
    medium.RunAndVerify(capture_stream.stream, "2", -2.0f);
  } while (grow.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready);
  EXPECT_NO_THROW(grow.get());
  small_binding.RunAndVerify(capture_stream.stream, "1", 4.0f);
}

}  // namespace
}  // namespace onnxruntime::test

#endif  // USE_CUDA || ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP
