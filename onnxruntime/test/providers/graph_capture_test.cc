// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_WEBGPU

#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"

#include <sstream>

#include "core/graph/model.h"
#include "core/framework/tensorprotoutils.h"
#include "core/session/IOBinding.h"
#include "core/session/onnxruntime_run_options_config_keys.h"

#include "test/unittest_util/framework_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/test_environment.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

// Build a three-op model: Y = MatMul(Relu(MatMul(A, B)), C)
// The two intermediate tensors (T1 from MatMul, T2 from Relu) exercise the per-graph buffer manager.
static void CreateMatMulReluMatMulModel(std::unique_ptr<onnxruntime::Model>& p_model, ProviderType provider_type) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[onnxruntime::kOnnxDomain] = 7;
  std::vector<ONNX_NAMESPACE::FunctionProto> model_specific_functions;
  p_model = std::make_unique<Model>("test", true, ModelMetaData(), PathString(),
                                    IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
                                    model_specific_functions, DefaultLoggingManager().DefaultLogger(),
                                    ModelOptions(true, true));
  onnxruntime::Graph& graph = p_model->MainGraph();

  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  auto& input_a = graph.GetOrCreateNodeArg("A", &tensor_float);
  auto& input_b = graph.GetOrCreateNodeArg("B", &tensor_float);
  auto& input_c = graph.GetOrCreateNodeArg("C", &tensor_float);
  auto& t1 = graph.GetOrCreateNodeArg("T1", &tensor_float);
  auto& t2 = graph.GetOrCreateNodeArg("T2", &tensor_float);
  auto& output_y = graph.GetOrCreateNodeArg("Y", &tensor_float);

  auto& matmul1 = graph.AddNode("matmul1", "MatMul", "MatMul",
                                {&input_a, &input_b}, {&t1},
                                nullptr, onnxruntime::kOnnxDomain);
  auto& relu = graph.AddNode("relu", "Relu", "Relu",
                             {&t1}, {&t2},
                             nullptr, onnxruntime::kOnnxDomain);
  auto& matmul2 = graph.AddNode("matmul2", "MatMul", "MatMul",
                                {&t2, &input_c}, {&output_y},
                                nullptr, onnxruntime::kOnnxDomain);

  matmul1.SetExecutionProviderType(provider_type);
  relu.SetExecutionProviderType(provider_type);
  matmul2.SetExecutionProviderType(provider_type);

  ASSERT_STATUS_OK(graph.Resolve());
}

TEST(InferenceSessionTests, TestReleaseCapturedGraph) {
  SessionOptions so;
  so.session_logid = "InferenceSessionTests.TestReleaseCapturedGraph";
  so.session_log_verbosity_level = 1;
  InferenceSession session_object{so, GetEnvironment()};

  ConfigOptions config_options{};
  ORT_ENFORCE(config_options.AddConfigEntry(webgpu::options::kEnableGraphCapture,
                                            webgpu::options::kEnableGraphCapture_ON)
                  .IsOK());
  auto provider = WebGpuExecutionProviderWithOptions(config_options);
  auto* gpu_provider = provider.get();
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(provider)));

  std::unique_ptr<Model> p_model;
  CreateMatMulReluMatMulModel(p_model, kWebGpuExecutionProvider);
  std::string s1;
  p_model->ToProto().SerializeToString(&s1);
  std::istringstream str(s1);
  ASSERT_STATUS_OK(session_object.Load(str));
  ASSERT_STATUS_OK(session_object.Initialize());

  auto cpu_alloc = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  OrtMemoryInfo mem_info(WEBGPU_BUFFER, OrtAllocatorType::OrtDeviceAllocator,
                         OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, 0));
  auto gpu_alloc = session_object.GetAllocator(mem_info);

  // Model: Y = MatMul(Relu(MatMul(A[3x4], B[4x3])), C[3x2]) => Y[3x2]
  //
  // Input set 1: A1 = {0..11}
  //   MatMul(A1, B) = {42,48,54, 114,136,158, 186,224,262}
  //   Relu = same (all positive)
  //   Y1 = MatMul(Relu, C) = {96,102, 272,294, 448,486}
  //
  // Input set 2: A2 = {12,11,...,1}
  //   MatMul(A2, B) = {174,216,258, 102,128,154, 30,40,50}
  //   Relu = same (all positive)
  //   Y2 = MatMul(Relu, C) = {432,474, 256,282, 80,90}

  std::vector<float> values_a1 = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  std::vector<float> values_b = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  std::vector<float> values_c = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
  std::vector<float> expected_y1 = {96, 102, 272, 294, 448, 486};

  std::vector<float> values_a2 = {12.0f, 11.0f, 10.0f, 9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
  std::vector<float> expected_y2 = {432, 474, 256, 282, 80, 90};

  std::vector<int64_t> dims_a = {3, 4};
  std::vector<int64_t> dims_b = {4, 3};
  std::vector<int64_t> dims_c = {3, 2};
  std::vector<int64_t> dims_y = {3, 2};

  // Prepare GPU input A (will be overwritten between capture and replay)
  OrtValue ml_a1_cpu;
  CreateMLValue<float>(cpu_alloc, dims_a, values_a1, &ml_a1_cpu);
  Tensor gpu_a(ml_a1_cpu.Get<Tensor>().DataType(), ml_a1_cpu.Get<Tensor>().Shape(), gpu_alloc);
  ASSERT_STATUS_OK(gpu_provider->GetDataTransfer()->CopyTensor(ml_a1_cpu.Get<Tensor>(), gpu_a));
  OrtValue ml_a;
  Tensor::InitOrtValue(std::move(gpu_a), ml_a);

  // Prepare GPU input B
  OrtValue ml_b_cpu;
  CreateMLValue<float>(cpu_alloc, dims_b, values_b, &ml_b_cpu);
  Tensor gpu_b(ml_b_cpu.Get<Tensor>().DataType(), ml_b_cpu.Get<Tensor>().Shape(), gpu_alloc);
  ASSERT_STATUS_OK(gpu_provider->GetDataTransfer()->CopyTensor(ml_b_cpu.Get<Tensor>(), gpu_b));
  OrtValue ml_b;
  Tensor::InitOrtValue(std::move(gpu_b), ml_b);

  // Prepare GPU input C
  OrtValue ml_c_cpu;
  CreateMLValue<float>(cpu_alloc, dims_c, values_c, &ml_c_cpu);
  Tensor gpu_c(ml_c_cpu.Get<Tensor>().DataType(), ml_c_cpu.Get<Tensor>().Shape(), gpu_alloc);
  ASSERT_STATUS_OK(gpu_provider->GetDataTransfer()->CopyTensor(ml_c_cpu.Get<Tensor>(), gpu_c));
  OrtValue ml_c;
  Tensor::InitOrtValue(std::move(gpu_c), ml_c);

  // Prepare GPU output
  OrtValue ml_y;
  AllocateMLValue<float>(gpu_alloc, dims_y, &ml_y);

  // Bind inputs/outputs
  std::unique_ptr<IOBinding> io_binding;
  ASSERT_STATUS_OK(session_object.NewIOBinding(&io_binding));
  ASSERT_STATUS_OK(io_binding->BindInput("A", ml_a));
  ASSERT_STATUS_OK(io_binding->BindInput("B", ml_b));
  ASSERT_STATUS_OK(io_binding->BindInput("C", ml_c));
  ASSERT_STATUS_OK(io_binding->BindOutput("Y", ml_y));
  ASSERT_TRUE(io_binding->SynchronizeInputs().IsOK());

  // Helper: copy GPU output to CPU and verify against expected values
  auto verify_output = [&](const std::vector<float>& expected) {
    std::vector<OrtValue>& outputs = io_binding->GetOutputs();
    ASSERT_EQ(1u, outputs.size());
    auto& rtensor = outputs.front().Get<Tensor>();
    Tensor cpu_tensor(rtensor.DataType(), rtensor.Shape(), cpu_alloc);
    ASSERT_STATUS_OK(gpu_provider->GetDataTransfer()->CopyTensor(rtensor, cpu_tensor));
    OrtValue ml_value;
    Tensor::InitOrtValue(std::move(cpu_tensor), ml_value);
    VerifySingleOutput({ml_value}, dims_y, expected);
  };

  // Helper: overwrite GPU tensor A with new CPU values
  auto set_input_a = [&](const std::vector<float>& values) {
    OrtValue cpu_val;
    CreateMLValue<float>(cpu_alloc, dims_a, values, &cpu_val);
    ASSERT_STATUS_OK(gpu_provider->GetDataTransfer()->CopyTensor(
        cpu_val.Get<Tensor>(), const_cast<Tensor&>(io_binding->GetInputs()[0].Get<Tensor>())));
  };

  // Helper: run with a given annotation ID
  auto run_with_id = [&](const char* id) {
    RunOptions run_options;
    ORT_ENFORCE(run_options.config_options.AddConfigEntry(kOrtRunOptionsConfigCudaGraphAnnotation, id).IsOK());
    return session_object.Run(run_options, *io_binding);
  };

  // Capture ID 1 with input set 1
  ASSERT_STATUS_OK(run_with_id("1"));
  verify_output(expected_y1);

  // Replay ID 1
  ASSERT_STATUS_OK(run_with_id("1"));
  verify_output(expected_y1);

  // Capture ID 2 with input set 2
  set_input_a(values_a2);
  ASSERT_STATUS_OK(run_with_id("2"));
  verify_output(expected_y2);

  // Replay ID 2
  ASSERT_STATUS_OK(run_with_id("2"));
  verify_output(expected_y2);

  // Replay ID 1 again (cross-ID isolation: ID 1 should still replay with its captured buffers)
  set_input_a(values_a1);
  ASSERT_STATUS_OK(run_with_id("1"));
  verify_output(expected_y1);

  // Release ID 1 only
  ASSERT_STATUS_OK(session_object.ReleaseCapturedGraph(1));

  // Replay ID 2 (unaffected by ID 1 release)
  set_input_a(values_a2);
  ASSERT_STATUS_OK(run_with_id("2"));
  verify_output(expected_y2);

  // Re-capture ID 1 (input A still has set 2 values)
  ASSERT_STATUS_OK(run_with_id("1"));
  verify_output(expected_y2);

  // Release both
  ASSERT_STATUS_OK(session_object.ReleaseCapturedGraph(1));
  ASSERT_STATUS_OK(session_object.ReleaseCapturedGraph(2));
}

}  // namespace test
}  // namespace onnxruntime

#endif  // USE_WEBGPU
