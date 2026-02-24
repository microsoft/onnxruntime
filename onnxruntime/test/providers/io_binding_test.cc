// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"

#include <sstream>

#include "core/graph/model.h"
#include "core/framework/tensorprotoutils.h"
#include "core/session/IOBinding.h"

#include "test/unittest_util/framework_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/test_environment.h"

using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

static void CreateMatMulModel(std::unique_ptr<onnxruntime::Model>& p_model, ProviderType provider_type) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[onnxruntime::kOnnxDomain] = 7;
  // Generate the input & output def lists
  std::vector<ONNX_NAMESPACE::FunctionProto> model_specific_functions;
  p_model = std::make_unique<Model>("test", true, ModelMetaData(), PathString(),
                                    IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
                                    model_specific_functions, DefaultLoggingManager().DefaultLogger(),
                                    ModelOptions(true, true));
  onnxruntime::Graph& graph = p_model->MainGraph();

  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  std::vector<onnxruntime::NodeArg*> input_defs;
  auto& input_arg_a = graph.GetOrCreateNodeArg("A", &tensor_float);
  input_defs.push_back(&input_arg_a);

  auto& input_arg_b = graph.GetOrCreateNodeArg("B", &tensor_float);
  input_defs.push_back(&input_arg_b);

  std::vector<onnxruntime::NodeArg*> output_defs;
  auto& output_arg = graph.GetOrCreateNodeArg("Y", &tensor_float);
  output_defs.push_back(&output_arg);

  // Create a simple model
  auto& node = graph.AddNode("node1", "MatMul", "MatMul", input_defs, output_defs, nullptr, onnxruntime::kOnnxDomain);
  if (provider_type == kCpuExecutionProvider) {
    node.SetExecutionProviderType(provider_type);
  } else {
#if defined(USE_CUDA) || defined(USE_WEBGPU)
    node.SetExecutionProviderType(provider_type);
#endif
  }
  Status status = graph.Resolve();
  ASSERT_STATUS_OK(status);
}

void RunModelWithBindingMatMul(InferenceSession& session_object,
                               const RunOptions& run_options,
                               ProviderType bind_provider_type,
                               bool is_preallocate_output_vec,
                               ProviderType allocation_provider,
                               IExecutionProvider* gpu_provider,
                               OrtDevice* output_device,
                               bool enable_graph_capture) {
  std::unique_ptr<IOBinding> io_binding;
  Status st = session_object.NewIOBinding(&io_binding);
  ASSERT_TRUE(st.IsOK());

  // bind a value to A with input that will produce invalid output in order to test replacement of a feed
  std::vector<float> values_mul_x_tmp = {12.f, 11.f, 10.f, 9.f, 8.f, 7.f, 6.f, 5.f, 4.f, 3.f, 2.f, 1.f};
  std::vector<int64_t> dims_mul_x_A_tmp = {3, 4};
  std::vector<float> values_mul_x = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  std::vector<int64_t> dims_mul_x_A = {3, 4};
  std::vector<int64_t> dims_mul_x_B = {4, 3};

  auto cpu_alloc = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  onnxruntime::AllocatorPtr gpu_alloc = nullptr;
  if (allocation_provider == kWebGpuExecutionProvider) {
    // Use session_object.GetAllocator to get the OrtAllocator for WebGPU.
    // Otherwise, gpu_provider->CreatePreferredAllocators() will create a new OrtAllocator which will go to the create UMA path.
    // And it can't be used for copying buffer to buffer since the target buffer is still in mapped state.
    OrtMemoryInfo mem_info(WEBGPU_BUFFER, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, 0));
    gpu_alloc = session_object.GetAllocator(mem_info);
  } else if (allocation_provider == kCudaExecutionProvider) {
    gpu_alloc = gpu_provider->CreatePreferredAllocators()[0];
  }
  if (enable_graph_capture) {
    // For graph capture, all inputs/outputs should be in preallocated gpu memory.
    ASSERT_TRUE(is_preallocate_output_vec);
    OrtValue input_ml_value_A_cpu;
    CreateMLValue<float>(cpu_alloc, dims_mul_x_A, values_mul_x, &input_ml_value_A_cpu);
    auto& cpu_tensor_a = input_ml_value_A_cpu.Get<Tensor>();
    Tensor gpu_tensor_a(cpu_tensor_a.DataType(), cpu_tensor_a.Shape(), gpu_alloc);
    st = gpu_provider->GetDataTransfer()->CopyTensor(cpu_tensor_a, gpu_tensor_a);
    ASSERT_STATUS_OK(st);
    OrtValue input_ml_value_A;
    Tensor::InitOrtValue(std::move(gpu_tensor_a), input_ml_value_A);

    OrtValue input_ml_value_B_cpu;
    CreateMLValue<float>(cpu_alloc, dims_mul_x_B, values_mul_x, &input_ml_value_B_cpu);
    auto& cpu_tensor_b = input_ml_value_B_cpu.Get<Tensor>();
    Tensor gpu_tensor_b(cpu_tensor_b.DataType(), cpu_tensor_b.Shape(), gpu_alloc);
    st = gpu_provider->GetDataTransfer()->CopyTensor(cpu_tensor_b, gpu_tensor_b);
    ASSERT_STATUS_OK(st);
    OrtValue input_ml_value_B;
    Tensor::InitOrtValue(std::move(gpu_tensor_b), input_ml_value_B);

    ASSERT_STATUS_OK(io_binding->BindInput("A", input_ml_value_A));
    ASSERT_STATUS_OK(io_binding->BindInput("B", input_ml_value_B));
  } else {
    auto input_allocator = io_binding->GetCPUAllocator(bind_provider_type);
    OrtValue input_tmp;
    CreateMLValue<float>(input_allocator, dims_mul_x_A_tmp, values_mul_x_tmp, &input_tmp);
    ASSERT_STATUS_OK(io_binding->BindInput("A", input_tmp));
    const void* tmp_A = io_binding->GetInputs()[0].Get<Tensor>().DataRaw();  // location of data post binding

    // prepare inputs
    /*
        0 1 2 3     0 1 2
        4 5 6 7     3 4 5
        8 9 10 11   6 7 8
        9 10 11
        */
    // bind one input to cpu allocator from bind_provider_type, and another on user provided CPU memory
    // so both code pathes are covered
    OrtValue input_ml_value_A;
    CreateMLValue<float>(input_allocator, dims_mul_x_A, values_mul_x, &input_ml_value_A);

    OrtValue input_ml_value_B;
    CreateMLValue<float>(cpu_alloc, dims_mul_x_B, values_mul_x, &input_ml_value_B);

    ASSERT_STATUS_OK(io_binding->BindInput("A", input_ml_value_A));
    ASSERT_STATUS_OK(io_binding->BindInput("B", input_ml_value_B));

    // check location of 'A' post-binding has changed to validate that the previous value was replaced
    ASSERT_TRUE(io_binding->GetInputs()[0].Get<Tensor>().DataRaw() != tmp_A);
  }
  // prepare outputs
  std::vector<int64_t> expected_output_dims = {3, 3};
  OrtValue output_ml_value;
  if (is_preallocate_output_vec) {
    if (allocation_provider == kCpuExecutionProvider) {
      AllocateMLValue<float>(cpu_alloc, expected_output_dims, &output_ml_value);
    } else if (allocation_provider == kCudaExecutionProvider || allocation_provider == kWebGpuExecutionProvider) {
      AllocateMLValue<float>(gpu_alloc, expected_output_dims, &output_ml_value);
    } else {
      ORT_THROW("Unsupported provider");
    }
  }

  if (output_device) {
    // output should be allocated on specified device (if not preallocated here)
    ASSERT_STATUS_OK(io_binding->BindOutput("Y", *output_device));
  } else {
    ASSERT_STATUS_OK(io_binding->BindOutput("Y", output_ml_value));
  }

  ASSERT_TRUE(io_binding->SynchronizeInputs().IsOK());

  // prepare expected inputs and outputs
  std::vector<float> expected_values_mul_y = {42, 48, 54, 114, 136, 158, 186, 224, 262};
  std::vector<float> expected_values_mul_y_2 = {174, 216, 258, 102, 128, 154, 30, 40, 50};

  // Now run
  ASSERT_STATUS_OK(session_object.Run(run_options, *io_binding));

  if ((is_preallocate_output_vec && (allocation_provider == kCudaExecutionProvider || allocation_provider == kWebGpuExecutionProvider)) ||
      (output_device && output_device->Type() == OrtDevice::GPU)) {
#if defined(USE_CUDA) || defined(USE_WEBGPU)
    // in this case we need to copy the tensor from GPU to CPU
    std::vector<OrtValue>& outputs = io_binding->GetOutputs();
    ASSERT_EQ(1u, outputs.size());
    auto& rtensor = outputs.front().Get<Tensor>();
    auto element_type = rtensor.DataType();
    auto& shape = rtensor.Shape();
    Tensor cpu_tensor(element_type, shape, cpu_alloc);
    st = gpu_provider->GetDataTransfer()->CopyTensor(rtensor, cpu_tensor);
    ASSERT_STATUS_OK(st);
    OrtValue ml_value;
    Tensor::InitOrtValue(std::move(cpu_tensor), ml_value);
    VerifySingleOutput({ml_value}, expected_output_dims, expected_values_mul_y);
#endif
  } else {
    if (allocation_provider == kCudaExecutionProvider || allocation_provider == kWebGpuExecutionProvider) {
      ASSERT_STATUS_OK(gpu_provider->Sync());
    }
    VerifySingleOutput(io_binding->GetOutputs(), expected_output_dims, expected_values_mul_y);
  }

  if (enable_graph_capture) {
    // Update input_a's value. Run again. Replay the captured graph
    OrtValue input_a2;
    CreateMLValue<float>(cpu_alloc, dims_mul_x_A_tmp, values_mul_x_tmp, &input_a2);
    auto& cpu_tensor_a2 = input_a2.Get<Tensor>();
    st = gpu_provider->GetDataTransfer()->CopyTensor(cpu_tensor_a2, const_cast<Tensor&>(io_binding->GetInputs()[0].Get<Tensor>()));
    ASSERT_STATUS_OK(st);

    st = session_object.Run(run_options, *io_binding.get());
    ASSERT_STATUS_OK(st);

    // Copy the tensor from gpu to cpu
    std::vector<OrtValue>& outputs = io_binding->GetOutputs();
    ASSERT_EQ(1u, outputs.size());
    auto& rtensor = outputs.front().Get<Tensor>();
    auto element_type = rtensor.DataType();
    auto& shape = rtensor.Shape();
    std::unique_ptr<Tensor> cpu_tensor = std::make_unique<Tensor>(element_type, shape, cpu_alloc);
    st = gpu_provider->GetDataTransfer()->CopyTensor(rtensor, *cpu_tensor.get());
    ASSERT_STATUS_OK(st);
    OrtValue ml_value;
    ml_value.Init(cpu_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
    VerifySingleOutput({ml_value}, expected_output_dims, expected_values_mul_y_2);
  }
}

static void TestBindHelper(const std::string& log_str,
                           ProviderType bind_provider_type,
                           ProviderType run_provider_type,
                           bool preallocate_output,
                           ProviderType allocation_provider = kCpuExecutionProvider,
                           OrtDevice* output_device = nullptr,
                           bool enable_graph_capture = false) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests." + log_str;
  so.session_log_verbosity_level = 1;  // change to 1 for detailed logging
  InferenceSession session_object{so, GetEnvironment()};
  IExecutionProvider* gpu_provider{};

  if (bind_provider_type == kCudaExecutionProvider || bind_provider_type == kWebGpuExecutionProvider) {
#ifdef USE_CUDA
    {
      auto provider = DefaultCudaExecutionProvider();
      gpu_provider = provider.get();
      ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(provider)));
    }
#endif
#ifdef USE_WEBGPU
    {
      ConfigOptions config_options{};
      ORT_ENFORCE(config_options.AddConfigEntry(webgpu::options::kEnableGraphCapture,
                                                enable_graph_capture ? webgpu::options::kEnableGraphCapture_ON : webgpu::options::kEnableGraphCapture_OFF)
                      .IsOK());
      auto provider = WebGpuExecutionProviderWithOptions(config_options);
      gpu_provider = provider.get();
      ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(provider)));
    }
#endif
  }

  std::unique_ptr<Model> p_model;
  CreateMatMulModel(p_model, run_provider_type);

  std::string s1;
  p_model->ToProto().SerializeToString(&s1);
  std::stringstream sstr(s1);
  ASSERT_STATUS_OK(session_object.Load(sstr));
  ASSERT_STATUS_OK(session_object.Initialize());

  RunOptions run_options;
  run_options.run_log_verbosity_level = so.session_log_verbosity_level;
  run_options.run_tag = so.session_logid;

  RunModelWithBindingMatMul(session_object,
                            run_options,
                            bind_provider_type,
                            preallocate_output,
                            allocation_provider,
                            gpu_provider,
                            output_device,
                            enable_graph_capture);
}

TEST(InferenceSessionTests, TestBindCpu) {
  TestBindHelper("TestBindCpu",
                 kCpuExecutionProvider,
                 kCpuExecutionProvider,
                 false /* don't preallocate output */);
}

TEST(InferenceSessionTests, TestIOBindingReuse) {
  SessionOptions so;
  InferenceSession session_object(so, GetEnvironment());
  std::unique_ptr<Model> p_model;
  CreateMatMulModel(p_model, kCpuExecutionProvider);

  std::string s1;
  p_model->ToProto().SerializeToString(&s1);
  std::stringstream sstr(s1);
  ASSERT_STATUS_OK(session_object.Load(sstr));
  ASSERT_STATUS_OK(session_object.Initialize());
  std::unique_ptr<IOBinding> io_binding;
  Status st = session_object.NewIOBinding(&io_binding);
  ASSERT_STATUS_OK(st);

  OrtValue ml_value1;
  const std::vector<float> v1{2.f};
  const int64_t shape[] = {1};
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], shape, v1, &ml_value1);
  ASSERT_STATUS_OK(io_binding->BindOutput("foo", ml_value1));
  ASSERT_TRUE(io_binding->GetOutputs().size() == 1);
  auto span = io_binding->GetOutputs()[0].Get<Tensor>().DataAsSpan<float>();
  ASSERT_TRUE(static_cast<size_t>(span.size()) == v1.size());
  for (size_t i = 0; i < v1.size(); ++i) {
    ASSERT_TRUE(v1[i] == span[i]);
  }

  OrtValue ml_value2;
  const std::vector<float> v2{3.f};
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], shape, v2, &ml_value2);
  ASSERT_STATUS_OK(io_binding->BindOutput("foo", ml_value2));
  ASSERT_TRUE(io_binding->GetOutputs().size() == 1);
  span = io_binding->GetOutputs()[0].Get<Tensor>().DataAsSpan<float>();
  ASSERT_TRUE(static_cast<size_t>(span.size()) == v2.size());
  for (size_t i = 0; i < v2.size(); ++i) {
    ASSERT_TRUE(v2[i] == span[i]);
  }
}

#if defined(USE_CUDA) || defined(USE_WEBGPU)
#if defined(USE_CUDA)
constexpr const char* kGpuExecutionProvider = kCudaExecutionProvider;
#elif defined(USE_WEBGPU)
constexpr const char* kGpuExecutionProvider = kWebGpuExecutionProvider;
#endif

TEST(InferenceSessionTests, TestBindGpu) {
  TestBindHelper("TestBindGpu",
                 kGpuExecutionProvider,
                 kGpuExecutionProvider,
                 false /* don't preallocate output */);
}

TEST(InferenceSessionTests, TestBindGpuPreallocateOutputOnGpu) {
  TestBindHelper("TestBindGpuPreallocateOutputOnGpu",
                 kGpuExecutionProvider,
                 kGpuExecutionProvider,
                 true /* preallocate output on GPU */,
                 kGpuExecutionProvider);
}

TEST(InferenceSessionTests, TestBindGpuPreallocateOutputOnCpu) {
  TestBindHelper("TestBindGpuPreallocateOutputOnCpu",
                 kGpuExecutionProvider,
                 kGpuExecutionProvider,
                 true /* preallocate output on CPU */,
                 kCpuExecutionProvider);
}

TEST(InferenceSessionTests, TestBindGpuPreallocateOutputOnCpu2) {
  TestBindHelper("TestBindGpuPreallocateOutputOnCpu2",
                 kGpuExecutionProvider,
                 kCpuExecutionProvider,
                 true /* preallocate output on CPU */,
                 kCpuExecutionProvider);
}
#ifndef USE_WEBGPU
TEST(InferenceSessionTests, TestBindGpuSpecifyOutputDeviceOnGpu) {
  OrtDevice device(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA, 0);

  TestBindHelper("TestBindGpuSpecifyOutputDeviceOnGpu",
                 kGpuExecutionProvider,
                 kGpuExecutionProvider,
                 false /* preallocate output on GPU */,
                 kGpuExecutionProvider,
                 &device /* specify output device */);
}
#else
TEST(InferenceSessionTests, TestGraphCapture) {
  TestBindHelper("TestGraphCapture",
                 kGpuExecutionProvider,
                 kGpuExecutionProvider,
                 true /* preallocate output on GPU */,
                 kGpuExecutionProvider,
                 nullptr,
                 true /* enable graph capture*/);
}
#endif  // !USE_WEBGPU
#endif

}  // namespace test
}  // namespace onnxruntime
