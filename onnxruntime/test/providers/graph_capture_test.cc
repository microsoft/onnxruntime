// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_WEBGPU

#include <numeric>

#include "gtest/gtest.h"

#include "core/graph/constants.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_run_options_config_keys.h"

using namespace Ort;

extern std::unique_ptr<Ort::Env> ort_env;

namespace {

// Append WebGPU EP to session options, handling both built-in and plugin builds.
void AppendWebGpuEp(Env& env, SessionOptions& session_options,
                    const std::unordered_map<std::string, std::string>& provider_options) {
#if defined(ORT_USE_EP_API_ADAPTERS)
  // Plugin build: find the WebGPU EpDevice and use V2 API
  auto ep_devices = env.GetEpDevices();
  std::vector<ConstEpDevice> webgpu_devices;
  for (const auto& device : ep_devices) {
    if (std::string(device.EpName()) == onnxruntime::kWebGpuExecutionProvider) {
      webgpu_devices.push_back(device);
      break;
    }
  }
  ASSERT_FALSE(webgpu_devices.empty()) << "No WebGPU EP device found after plugin registration";
  session_options.AppendExecutionProvider_V2(env, webgpu_devices, provider_options);
#else
  static_cast<void>(env);
  session_options.AppendExecutionProvider("WebGPU", provider_options);
#endif
}

// Build a model: Y = MatMul(Relu(MatMul(A, B)), C)
// All shapes are unspecified (free dimensions) to keep it simple.
static Model CreateMatMulReluMatMulModel() {
  Graph graph;

  // Inputs: A[3x4], B[4x3], C[3x2] — float tensors
  std::vector<int64_t> dims_a_shape = {3, 4};
  std::vector<int64_t> dims_b_shape = {4, 3};
  std::vector<int64_t> dims_c_shape = {3, 2};
  std::vector<int64_t> dims_y_shape = {3, 2};

  TensorTypeAndShapeInfo a_info(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dims_a_shape);
  TensorTypeAndShapeInfo b_info(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dims_b_shape);
  TensorTypeAndShapeInfo c_info(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dims_c_shape);
  TensorTypeAndShapeInfo y_info(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, dims_y_shape);

  auto a_type = TypeInfo::CreateTensorInfo(a_info.GetConst());
  auto b_type = TypeInfo::CreateTensorInfo(b_info.GetConst());
  auto c_type = TypeInfo::CreateTensorInfo(c_info.GetConst());
  auto y_type = TypeInfo::CreateTensorInfo(y_info.GetConst());

  std::vector<ValueInfo> inputs;
  inputs.emplace_back("A", a_type.GetConst());
  inputs.emplace_back("B", b_type.GetConst());
  inputs.emplace_back("C", c_type.GetConst());

  std::vector<ValueInfo> outputs;
  outputs.emplace_back("Y", y_type.GetConst());

  graph.SetInputs(inputs);
  graph.SetOutputs(outputs);

  // MatMul(A, B) -> T1
  Node matmul1("MatMul", onnxruntime::kOnnxDomain, "matmul1", {"A", "B"}, {"T1"});
  graph.AddNode(matmul1);

  // Relu(T1) -> T2
  Node relu("Relu", onnxruntime::kOnnxDomain, "relu", {"T1"}, {"T2"});
  graph.AddNode(relu);

  // MatMul(T2, C) -> Y
  Node matmul2("MatMul", onnxruntime::kOnnxDomain, "matmul2", {"T2", "C"}, {"Y"});
  graph.AddNode(matmul2);

  std::vector<Model::DomainOpsetPair> opsets{{onnxruntime::kOnnxDomain, 13}};
  Model model(opsets);
  model.AddGraph(graph);
  return model;
}

TEST(GraphCaptureTests, TestReleaseCapturedGraph) {
  Env& env = *ort_env;

  // Create session with WebGPU EP and graph capture enabled
  SessionOptions session_options;
  session_options.DisableMemPattern();
  std::unordered_map<std::string, std::string> provider_options;
  provider_options["enableGraphCapture"] = "1";
  AppendWebGpuEp(env, session_options, provider_options);

  auto model = CreateMatMulReluMatMulModel();
  Session session(env, model, session_options);

  // Get GPU allocator from session
  MemoryInfo gpu_mem_info("WebGPU_Buffer", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemTypeDefault);
  Allocator gpu_allocator(session, gpu_mem_info);

  // Model: Y = MatMul(Relu(MatMul(A[3x4], B[4x3])), C[3x2]) => Y[3x2]
  std::vector<int64_t> dims_a = {3, 4};
  std::vector<int64_t> dims_b = {4, 3};
  std::vector<int64_t> dims_c = {3, 2};
  std::vector<int64_t> dims_y = {3, 2};

  // Input set 1
  std::vector<float> values_a1(12);
  std::iota(values_a1.begin(), values_a1.end(), 0.0f);  // 0..11
  std::vector<float> values_b(12);
  std::iota(values_b.begin(), values_b.end(), 0.0f);  // 0..11
  std::vector<float> values_c = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};

  // Input set 2
  std::vector<float> values_a2 = {12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};

  MemoryInfo cpu_mem_info = MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Pre-compute expected outputs on CPU:
  // T1 = MatMul(A, B), T2 = Relu(T1), Y = MatMul(T2, C)
  auto compute_expected = [&](const std::vector<float>& a) -> std::vector<float> {
    // T1 = A[3x4] * B[4x3]
    std::vector<float> t1(9, 0.0f);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        for (int k = 0; k < 4; ++k)
          t1[i * 3 + j] += a[i * 4 + k] * values_b[k * 3 + j];

    // T2 = Relu(T1)
    std::vector<float> t2(9);
    for (int i = 0; i < 9; ++i)
      t2[i] = std::max(0.0f, t1[i]);

    // Y = T2[3x3] * C[3x2]
    std::vector<float> y(6, 0.0f);
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 2; ++j)
        for (int k = 0; k < 3; ++k)
          y[i * 2 + j] += t2[i * 3 + k] * values_c[k * 2 + j];

    return y;
  };

  auto expected_y1 = compute_expected(values_a1);
  auto expected_y2 = compute_expected(values_a2);

  // Allocate GPU tensors
  Value gpu_a = Value::CreateTensor<float>(gpu_allocator, dims_a.data(), dims_a.size());
  Value gpu_b = Value::CreateTensor<float>(gpu_allocator, dims_b.data(), dims_b.size());
  Value gpu_c = Value::CreateTensor<float>(gpu_allocator, dims_c.data(), dims_c.size());
  Value gpu_y = Value::CreateTensor<float>(gpu_allocator, dims_y.data(), dims_y.size());

  // Helper: copy CPU tensor to GPU tensor via Env::CopyTensor
  auto copy_to_gpu = [&](float* data, size_t count, const int64_t* shape, size_t shape_len, Value& gpu_tensor) {
    Value cpu_tensor = Value::CreateTensor<float>(cpu_mem_info, data, count, shape, shape_len);
    auto status = env.CopyTensor(cpu_tensor, gpu_tensor, nullptr);
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  };

  // Upload initial inputs (B and C are constant across all runs)
  copy_to_gpu(values_a1.data(), values_a1.size(), dims_a.data(), dims_a.size(), gpu_a);
  copy_to_gpu(values_b.data(), values_b.size(), dims_b.data(), dims_b.size(), gpu_b);
  copy_to_gpu(values_c.data(), values_c.size(), dims_c.data(), dims_c.size(), gpu_c);

  // Set up IoBinding
  IoBinding io_binding(session);
  io_binding.BindInput("A", gpu_a);
  io_binding.BindInput("B", gpu_b);
  io_binding.BindInput("C", gpu_c);
  io_binding.BindOutput("Y", gpu_y);
  io_binding.SynchronizeInputs();

  // Helper: verify GPU output matches expected values
  auto verify_output = [&](const std::vector<float>& expected) {
    std::vector<float> result(expected.size(), 0.0f);
    Value cpu_result = Value::CreateTensor<float>(cpu_mem_info, result.data(), result.size(),
                                                  dims_y.data(), dims_y.size());
    auto status = env.CopyTensor(gpu_y, cpu_result, nullptr);
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
    for (size_t i = 0; i < expected.size(); ++i) {
      ASSERT_FLOAT_EQ(result[i], expected[i]) << "Mismatch at index " << i;
    }
  };

  // Helper: upload new A values to GPU
  auto set_input_a = [&](std::vector<float>& values) {
    copy_to_gpu(values.data(), values.size(), dims_a.data(), dims_a.size(), gpu_a);
  };

  // Helper: run with a given annotation ID
  auto run_with_id = [&](const char* id) {
    RunOptions run_options;
    run_options.AddConfigEntry(kOrtRunOptionsConfigCudaGraphAnnotation, id);
    session.Run(run_options, io_binding);
  };

  // Capture graph for annotation ID 1 (input set 1)
  run_with_id("1");
  verify_output(expected_y1);

  // Replay ID 1
  run_with_id("1");
  verify_output(expected_y1);

  // Capture graph for annotation ID 2 (input set 2)
  set_input_a(values_a2);
  run_with_id("2");
  verify_output(expected_y2);

  // Replay ID 2
  run_with_id("2");
  verify_output(expected_y2);

  // Replay ID 1 again (cross-ID isolation)
  set_input_a(values_a1);
  run_with_id("1");
  verify_output(expected_y1);

  // Release ID 1 using the C++ API
  session.ReleaseCapturedGraph(1);

  // Replay ID 2 (unaffected by ID 1 release)
  set_input_a(values_a2);
  run_with_id("2");
  verify_output(expected_y2);

  // Re-capture ID 1 after release
  run_with_id("1");
  verify_output(expected_y2);

  // Release both using the C++ API
  session.ReleaseCapturedGraph(1);
  session.ReleaseCapturedGraph(2);
}

}  // namespace

#endif  // USE_WEBGPU
