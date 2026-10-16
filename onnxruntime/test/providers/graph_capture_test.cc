// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_WEBGPU

#include <filesystem>
#include <numeric>

#include <gsl/gsl>

#include "gtest/gtest.h"

#include "core/graph/constants.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_run_options_config_keys.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

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

TEST(WebGpuDispatchBatchingTests, ProfilingDispatchCountStaysBoundedAcrossAllocatorClears) {
  Env& env = *ort_env;

  SessionOptions session_options;
  session_options.DisableMemPattern();
  session_options.EnableProfiling(ORT_TSTR("webgpu_dispatch_batching_profiling_test"));

  std::unordered_map<std::string, std::string> provider_options;
  provider_options["maxNumPendingDispatches"] = "2";
  provider_options["validationMode"] = "full";
  AppendWebGpuEp(env, session_options, provider_options);

  auto model = CreateMatMulReluMatMulModel();
  Session session(env, model, session_options);

  MemoryInfo cpu_mem_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  std::vector<int64_t> dims_a = {3, 4};
  std::vector<int64_t> dims_b = {4, 3};
  std::vector<int64_t> dims_c = {3, 2};
  std::vector<int64_t> dims_y = {3, 2};
  std::vector<float> values_a(12, 1.0f);
  std::vector<float> values_b(12, 1.0f);
  std::vector<float> values_c(6, 1.0f);
  std::vector<float> values_y(6, 0.0f);

  std::vector<Value> inputs;
  inputs.emplace_back(Value::CreateTensor<float>(cpu_mem_info, values_a.data(), values_a.size(),
                                                 dims_a.data(), dims_a.size()));
  inputs.emplace_back(Value::CreateTensor<float>(cpu_mem_info, values_b.data(), values_b.size(),
                                                 dims_b.data(), dims_b.size()));
  inputs.emplace_back(Value::CreateTensor<float>(cpu_mem_info, values_c.data(), values_c.size(),
                                                 dims_c.data(), dims_c.size()));
  Value output = Value::CreateTensor<float>(cpu_mem_info, values_y.data(), values_y.size(),
                                            dims_y.data(), dims_y.size());
  std::vector<const char*> input_names = {"A", "B", "C"};
  std::vector<const char*> output_names = {"Y"};

  std::string run_error;
  std::vector<float> warm_output;
  std::vector<float> second_output;
  try {
    session.Run(RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                output_names.data(), &output, 1);
    warm_output = values_y;

    std::fill(values_y.begin(), values_y.end(), 0.0f);
    session.Run(RunOptions{nullptr}, input_names.data(), inputs.data(), inputs.size(),
                output_names.data(), &output, 1);
    second_output = values_y;
  } catch (const std::exception& ex) {
    run_error = ex.what();
  }

  AllocatorWithDefaultOptions allocator;
  auto profile_file = session.EndProfilingAllocated(allocator);
  std::filesystem::path profile_file_path = profile_file.get();
  auto cleanup_profile = gsl::finally([&profile_file_path] {
    std::error_code ec;
    std::filesystem::remove(profile_file_path, ec);
  });

  ASSERT_TRUE(run_error.empty()) << run_error;
  const std::vector<float> expected_output(6, 12.0f);
  ASSERT_EQ(warm_output.size(), expected_output.size());
  ASSERT_EQ(second_output.size(), expected_output.size());
  for (size_t i = 0; i < expected_output.size(); ++i) {
    EXPECT_FLOAT_EQ(warm_output[i], expected_output[i]) << "Warm run output mismatch at index " << i;
    EXPECT_FLOAT_EQ(second_output[i], expected_output[i]) << "Second run output mismatch at index " << i;
  }
}

class WebGpuScatterGraphCaptureTests : public testing::TestWithParam<bool> {};

void TestScatterReplayCopiesInput(const char* op_type, const std::vector<int64_t>& indices_shape,
                                  bool enable_graph_capture, bool surrounding_dispatches = false) {
  Env& env = *ort_env;
  SessionOptions session_options;
  session_options.DisableMemPattern();
  session_options.SetGraphOptimizationLevel(ORT_DISABLE_ALL);
  session_options.SetLogSeverityLevel(1);
  session_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
  if (surrounding_dispatches) {
    session_options.EnableProfiling(ORT_TSTR("webgpu_scatter_capture_test"));
  }
  ASSERT_NO_FATAL_FAILURE(AppendWebGpuEp(
      env, session_options, {{"enableGraphCapture", enable_graph_capture ? "1" : "0"}}));

  const std::vector<int64_t> data_shape{3};
  const std::vector<int64_t> updates_shape{1};
  TensorTypeAndShapeInfo data_info(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, data_shape);
  TensorTypeAndShapeInfo indices_info(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, indices_shape);
  TensorTypeAndShapeInfo updates_info(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, updates_shape);
  auto data_type = TypeInfo::CreateTensorInfo(data_info.GetConst());
  auto indices_type = TypeInfo::CreateTensorInfo(indices_info.GetConst());
  auto updates_type = TypeInfo::CreateTensorInfo(updates_info.GetConst());
  std::vector<ValueInfo> inputs;
  inputs.emplace_back("X", data_type.GetConst());
  inputs.emplace_back("indices", indices_type.GetConst());
  inputs.emplace_back("updates", updates_type.GetConst());
  std::vector<ValueInfo> outputs;
  outputs.emplace_back("Y", data_type.GetConst());
  if (surrounding_dispatches) {
    outputs.emplace_back("T", data_type.GetConst());
    outputs.emplace_back("Z", data_type.GetConst());
  }
  Graph graph;
  graph.SetInputs(inputs);
  graph.SetOutputs(outputs);
  if (surrounding_dispatches) {
    Node before("Relu", onnxruntime::kOnnxDomain, "before", {"X"}, {"T"});
    graph.AddNode(before);
  }
  Node scatter(op_type, onnxruntime::kOnnxDomain, "scatter",
               {surrounding_dispatches ? "T" : "X", "indices", "updates"}, {"Y"});
  graph.AddNode(scatter);
  if (surrounding_dispatches) {
    Node after("Relu", onnxruntime::kOnnxDomain, "after", {"Y"}, {"Z"});
    graph.AddNode(after);
  }
  Model model(std::vector<Model::DomainOpsetPair>{{onnxruntime::kOnnxDomain, 18}});
  model.AddGraph(graph);
  Session session(env, model, session_options);

  MemoryInfo gpu_mem_info("WebGPU_Buffer", OrtDeviceAllocator, 0, OrtMemTypeDefault);
  Allocator gpu_allocator(session, gpu_mem_info);
  auto gpu_x = Value::CreateTensor<float>(gpu_allocator, data_shape.data(), data_shape.size());
  auto gpu_indices = Value::CreateTensor<int64_t>(gpu_allocator, indices_shape.data(), indices_shape.size());
  auto gpu_updates = Value::CreateTensor<float>(gpu_allocator, updates_shape.data(), updates_shape.size());
  auto gpu_y = Value::CreateTensor<float>(gpu_allocator, data_shape.data(), data_shape.size());
  Value gpu_t{nullptr};
  Value gpu_z{nullptr};
  // Separate bound output storage forces the copy path despite the kernel's MayInplace registration.
  ASSERT_NE(gpu_x.GetTensorRawData(), gpu_y.GetTensorRawData());

  IoBinding binding(session);
  binding.BindInput("X", gpu_x);
  binding.BindInput("indices", gpu_indices);
  binding.BindInput("updates", gpu_updates);
  binding.BindOutput("Y", gpu_y);
  if (surrounding_dispatches) {
    gpu_t = Value::CreateTensor<float>(gpu_allocator, data_shape.data(), data_shape.size());
    gpu_z = Value::CreateTensor<float>(gpu_allocator, data_shape.data(), data_shape.size());
    ASSERT_NE(gpu_t.GetTensorRawData(), gpu_y.GetTensorRawData());
    binding.BindOutput("T", gpu_t);
    binding.BindOutput("Z", gpu_z);
  }

  MemoryInfo cpu_mem_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  std::vector<float> x{1.0f, 2.0f, 3.0f};
  std::vector<int64_t> indices{1};
  std::vector<float> updates{9.0f};
  std::vector<float> result(3);
  std::vector<float> input_after_run(3);
  auto cpu_x = Value::CreateTensor<float>(cpu_mem_info, x.data(), x.size(), data_shape.data(), data_shape.size());
  auto cpu_indices = Value::CreateTensor<int64_t>(
      cpu_mem_info, indices.data(), indices.size(), indices_shape.data(), indices_shape.size());
  auto cpu_updates = Value::CreateTensor<float>(
      cpu_mem_info, updates.data(), updates.size(), updates_shape.data(), updates_shape.size());
  auto cpu_result = Value::CreateTensor<float>(
      cpu_mem_info, result.data(), result.size(), data_shape.data(), data_shape.size());
  auto cpu_input_after_run = Value::CreateTensor<float>(
      cpu_mem_info, input_after_run.data(), input_after_run.size(), data_shape.data(), data_shape.size());
  auto copy = [&](const Value& src, Value& dst) {
    auto status = env.CopyTensor(src, dst, nullptr);
    ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();
  };

  ASSERT_NO_FATAL_FAILURE(copy(cpu_indices, gpu_indices));
  RunOptions run_options;
  run_options.AddConfigEntry(kOrtRunOptionsConfigCudaGraphAnnotation, "1");
  for (int run = 0; run < 4; ++run) {
    SCOPED_TRACE(testing::Message() << op_type << ", capture=" << enable_graph_capture << ", run=" << run);
    if (enable_graph_capture && run == 3) {
      session.ReleaseCapturedGraph(1);
    }
    x[0] = 1.0f + 3.0f * run;
    x[1] = 2.0f + 3.0f * run;
    x[2] = 3.0f + 3.0f * run;
    updates[0] = 9.0f - run;
    ASSERT_NO_FATAL_FAILURE(copy(cpu_x, gpu_x));
    ASSERT_NO_FATAL_FAILURE(copy(cpu_updates, gpu_updates));
    binding.SynchronizeInputs();
    session.Run(run_options, binding);
    binding.SynchronizeOutputs();
    ASSERT_NO_FATAL_FAILURE(copy(gpu_y, cpu_result));
    ASSERT_NO_FATAL_FAILURE(copy(gpu_x, cpu_input_after_run));
    EXPECT_EQ(input_after_run, x) << "Scatter must not overwrite the separately bound input";
    EXPECT_EQ(result, (std::vector<float>{x[0], updates[0], x[2]}))
        << "Replay must copy the new input values at positions not written by scatter";
    if (surrounding_dispatches) {
      ASSERT_NO_FATAL_FAILURE(copy(gpu_z, cpu_result));
      EXPECT_EQ(result, (std::vector<float>{x[0], updates[0], x[2]}))
          << "The dispatch following scatter must observe the copy and the scatter update";
    }
  }
  if (enable_graph_capture) {
    session.ReleaseCapturedGraph(1);
  }
  if (surrounding_dispatches) {
    AllocatorWithDefaultOptions allocator;
    auto profile_path = session.EndProfilingAllocated(allocator);
    std::filesystem::remove(profile_path.get());
  }
}

TEST_P(WebGpuScatterGraphCaptureTests, ScatterNDNonInplaceCopiesUpdatedInput) {
  TestScatterReplayCopiesInput("ScatterND", {1, 1}, GetParam());
}

TEST_P(WebGpuScatterGraphCaptureTests, ScatterElementsNonInplaceCopiesUpdatedInput) {
  TestScatterReplayCopiesInput("ScatterElements", {1}, GetParam());
}

TEST_P(WebGpuScatterGraphCaptureTests, ScatterNDCopyBetweenProfiledDispatches) {
  TestScatterReplayCopiesInput("ScatterND", {1, 1}, GetParam(), true);
}

INSTANTIATE_TEST_SUITE_P(CaptureDisabledAndEnabled, WebGpuScatterGraphCaptureTests, testing::Bool());

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
