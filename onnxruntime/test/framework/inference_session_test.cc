// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"

#include <algorithm>
#include <cfloat>
#include <functional>
#include <iterator>
#include <thread>
#include <fstream>

#include <google/protobuf/io/zero_copy_stream_impl.h>
#include "core/common/logging/logging.h"
#include "core/common/profiler.h"
#include "core/framework/compute_capability.h"
#include "core/framework/execution_provider.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/platform/env.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/providers/cpu/math/element_wise_ops.h"
#include "core/session/IOBinding.h"
#include "dummy_provider.h"
#include "test_utils.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "test/providers/provider_test_utils.h"
#include "test/optimizer/dummy_graph_transformer.h"
#include "core/optimizer/rule_based_graph_transformer.h"

#include "gtest/gtest.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

namespace onnxruntime {
class FuseAdd : public OpKernel {
 public:
  FuseAdd(const OpKernelInfo& info) : OpKernel(info) {}

  Status Compute(OpKernelContext* context) const override {
    auto X = context->Input<Tensor>(0);
    auto Y = context->Input<Tensor>(1);
    auto Z = context->Input<Tensor>(2);
    auto& shape = X->Shape();
    auto M = context->Output(0, shape)->template MutableData<float>();
    for (int i = 0; i < shape.Size(); ++i) {
      *(M + i) = *(X->template Data<float>() + i) + *(Y->template Data<float>() + i) + *(Z->template Data<float>() + i);
    }
    return Status::OK();
  }
};
std::string kFuseTest = "FuseTest";
std::string kFuseExecutionProvider = "FuseExecutionProvider";
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kFuseExecutionProvider, kFuseTest, 1, FuseAdd);
ONNX_OPERATOR_KERNEL_EX(FuseAdd,
                        kFuseTest,
                        1,
                        kFuseExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        FuseAdd);

void RegisterOperatorKernels(KernelRegistry& kernel_registry) {
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kFuseExecutionProvider, kFuseTest, 1, FuseAdd)>());
}

std::shared_ptr<KernelRegistry> GetFusedKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  RegisterOperatorKernels(*kernel_registry);
  return kernel_registry;
}

class FuseExecutionProvider : public IExecutionProvider {
 public:
  explicit FuseExecutionProvider() : IExecutionProvider{kFuseExecutionProvider} {
    DeviceAllocatorRegistrationInfo device_info({OrtMemTypeDefault,
                                                 [](int) { return std::make_unique<CPUAllocator>(); },
                                                 std::numeric_limits<size_t>::max()});
    InsertAllocator(std::shared_ptr<IArenaAllocator>(
        std::make_unique<DummyArena>(device_info.factory(0))));
  }

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override {
    // Fuse two add into one.
    std::vector<std::unique_ptr<ComputeCapability>> result;
    std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
    for (auto& node : graph.Nodes()) {
      sub_graph->nodes.push_back(node.Index());
    }
    auto meta_def = std::make_unique<IndexedSubGraph::MetaDef>();
    meta_def->name = "FuseAdd";
    meta_def->domain = "FuseTest";
    meta_def->inputs = {"X", "Y", "Z"};
    meta_def->outputs = {"M"};
    meta_def->since_version = 1;
    meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;
    sub_graph->SetMetaDef(meta_def);
    result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
    return result;
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override {
    static std::shared_ptr<KernelRegistry> kernel_registry = GetFusedKernelRegistry();
    return kernel_registry;
  }

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override {
    ORT_UNUSED_PARAMETER(src);
    ORT_UNUSED_PARAMETER(dst);
    return Status::OK();
  }

  const void* GetExecutionHandle() const noexcept override {
    return nullptr;
  }
};

namespace test {
static void VerifyOutputs(const std::vector<MLValue>& fetches,
                          const std::vector<int64_t>& expected_dims,
                          const std::vector<float>& expected_values);
static const std::string MODEL_URI = "testdata/mul_1.pb";
static const std::string MODEL_URI_NO_OPSET = "testdata/mul_1.pb.noopset";
//static const std::string MODEL_URI = "./testdata/squeezenet/model.onnx"; // TODO enable this after we've weights?

static void CreateMatMulModel(std::unique_ptr<onnxruntime::Model>& p_model, ProviderType provider_type) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[onnxruntime::kOnnxDomain] = 7;
  // Generate the input & output def lists
  p_model = std::make_unique<onnxruntime::Model>("test", true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(),
                                                 domain_to_version);
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
#ifdef USE_CUDA
    node.SetExecutionProviderType(provider_type);
#endif
  }
  Status status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
}

void VerifyOutputs(const std::vector<MLValue>& fetches,
                   const std::vector<int64_t>& expected_dims,
                   const std::vector<float>& expected_values) {
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<float> found(rtensor.template Data<float>(),
                                 rtensor.template Data<float>() + expected_values.size());
  ASSERT_EQ(expected_values, found);
}

void RunModel(InferenceSession& session_object,
              const RunOptions& run_options,
              bool is_preallocate_output_vec = false) {
  // prepare inputs
  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  MLValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<MLValue> fetches;

  if (is_preallocate_output_vec) {
    fetches.resize(output_names.size());
    for (auto& elem : fetches) {
      CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                           &elem);
    }
  }

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_y = {3, 2};
  std::vector<float> expected_values_mul_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(st.IsOK());
  VerifyOutputs(fetches, expected_dims_mul_y, expected_values_mul_y);
}

void RunModelWithBindingMatMul(InferenceSession& session_object,
                               const RunOptions& run_options,
                               ProviderType bind_provider_type,
                               bool is_preallocate_output_vec = false,
                               ProviderType allocation_provider = kCpuExecutionProvider) {
  unique_ptr<IOBinding> io_binding;
  Status st = session_object.NewIOBinding(&io_binding);
  ASSERT_TRUE(st.IsOK());
  auto input_allocator = io_binding->GetCPUAllocator(0, bind_provider_type);

  // prepare inputs
  std::vector<float> values_mul_x = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f};
  /*
      0 1 2 3     0 1 2
      4 5 6 7     3 4 5
      8 9 10 11   6 7 8
      9 10 11
      */
  // bind one input to cpu allocator from bind_provider_type, and another on user provided CPU memory
  // so both code pathes are covered
  MLValue input_ml_value_A;
  std::vector<int64_t> dims_mul_x_A = {3, 4};
  CreateMLValue<float>(input_allocator, dims_mul_x_A, values_mul_x, &input_ml_value_A);

  MLValue input_ml_value_B;
  std::vector<int64_t> dims_mul_x_B = {4, 3};
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x_B, values_mul_x,
                       &input_ml_value_B);

  io_binding->BindInput("A", input_ml_value_A);
  io_binding->BindInput("B", input_ml_value_B);

  // prepare outputs
  std::vector<int64_t> expected_output_dims = {3, 3};
  MLValue output_ml_value;
  if (is_preallocate_output_vec) {
    if (allocation_provider == kCpuExecutionProvider) {
      AllocateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), expected_output_dims,
                             &output_ml_value);
    } else if (allocation_provider == kCudaExecutionProvider) {
#ifdef USE_CUDA
      AllocateMLValue<float>(TestCudaExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), expected_output_dims,
                             &output_ml_value);
#endif
    } else {
      ORT_THROW("Unsupported provider");
    }
  }
  io_binding->BindOutput("Y", output_ml_value);
  ASSERT_TRUE(io_binding->SynchronizeInputs().IsOK());

  // prepare expected inputs and outputs
  std::vector<float> expected_values_mul_y = {42, 48, 54, 114, 136, 158, 186, 224, 262};

  // Now run
  st = session_object.Run(run_options, *io_binding.get());

  std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  ASSERT_TRUE(st.IsOK());

  if (is_preallocate_output_vec &&
      allocation_provider == kCudaExecutionProvider) {
#ifdef USE_CUDA
    // in this case we need to copy the tensor from cuda to cpu
    vector<MLValue>& outputs = io_binding->GetOutputs();
    ASSERT_EQ(1, outputs.size());
    auto& rtensor = outputs.front().Get<Tensor>();
    auto element_type = rtensor.DataType();
    auto& shape = rtensor.Shape();
    auto cpu_allocator = TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault);
    std::unique_ptr<Tensor> cpu_tensor = std::make_unique<Tensor>(element_type,
                                                                  shape,
                                                                  cpu_allocator);
    st = TestCudaExecutionProvider()->CopyTensor(rtensor, *cpu_tensor.get());
    ASSERT_TRUE(st.IsOK());
    MLValue ml_value;
    ml_value.Init(cpu_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
    VerifyOutputs({ml_value}, expected_output_dims, expected_values_mul_y);
#endif
  } else {
    if (allocation_provider == kCudaExecutionProvider) {
#ifdef USE_CUDA
      TestCudaExecutionProvider()->Sync();
#endif
    }
    VerifyOutputs(io_binding->GetOutputs(), expected_output_dims, expected_values_mul_y);
  }
}

TEST(InferenceSessionTests, NoTimeout) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.NoTimeout";

  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";
  RunModel(session_object, run_options);
}

TEST(InferenceSessionTests, DisableCPUArena) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.DisableCPUArena";
  so.enable_cpu_mem_arena = false;

  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";
  RunModel(session_object, run_options);
}

#ifdef ORT_RUN_EXTERNAL_ONNX_TESTS
static bool Compare(const InputDefList& f_arg, const InputDefList& s_arg) {
  if (f_arg.size() != s_arg.size()) {
    cout << "Sizes differ: f_arg size: " << f_arg.size() << " s_arg size: " << s_arg.size() << endl;
    return false;
  }

  for (size_t i = 0; i < f_arg.size(); ++i) {
    const onnxruntime::NodeArg* x = f_arg[i];
    const onnxruntime::NodeArg* y = s_arg[i];
    if ((x->Shape() == nullptr) ^ (y->Shape() == nullptr)) {
      return false;
    }
    if (!x->Shape()) {
      continue;
    }
    vector<int64_t> x_shape = utils::GetTensorShapeFromTensorShapeProto(*x->Shape());
    vector<int64_t> y_shape = utils::GetTensorShapeFromTensorShapeProto(*y->Shape());
    if (x->Name() == y->Name() && x_shape == y_shape && *x->Type() == *y->Type()) {
      continue;
    }
    return false;
  }

  return true;
}

TEST(InferenceSessionTests, ModelMetadata) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.ModelMetadata";
  InferenceSession session_object{so, &DefaultLoggingManager()};
  string model_uri = "../models/opset8/test_squeezenet/model.onnx";
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<onnxruntime::Model> p_model;
  Status st = onnxruntime::Model::Load(model_uri, p_model);
  ASSERT_TRUE(st.IsOK());
  const onnxruntime::Graph& graph = p_model->MainGraph();

  // 1. first test the model meta
  {
    auto retval = session_object.GetModelMetadata();
    ASSERT_TRUE(retval.first.IsOK());
    const ModelMetadata* m = retval.second;
    ASSERT_TRUE(m->custom_metadata_map == p_model->MetaData() &&
                m->description == p_model->DocString() &&
                m->domain == p_model->Domain() &&
                m->graph_name == graph.Name() &&
                m->producer_name == p_model->ProducerName() &&
                m->version == p_model->ModelVersion());
  }

  {
    // 2. test inputs
    auto& inputs = graph.GetInputs();
    auto weights = graph.GetAllInitializedTensors();

    // skip the weights
    InputDefList inputs_no_weights;
    for (auto& elem : inputs) {
      if (weights.find(elem->Name()) != weights.end()) {
        continue;
      } else {
        inputs_no_weights.push_back(elem);
      }
    }

    auto retval = session_object.GetModelInputs();
    cout << "weights size: " << weights.size()
         << " inputs.size(): " << inputs.size()
         << " from session: " << retval.second->size() << endl;
    ASSERT_TRUE(retval.first.IsOK());
    ASSERT_TRUE(Compare(inputs_no_weights, *retval.second));
  }

  // 3. test outputs
  {
    auto retval = session_object.GetModelOutputs();
    ASSERT_TRUE(retval.first.IsOK());

    auto& outputs = graph.GetOutputs();
    retval = session_object.GetModelOutputs();
    ASSERT_TRUE(retval.first.IsOK());
    ASSERT_TRUE(Compare(outputs, *retval.second));
  }
}
#endif
TEST(InferenceSessionTests, CheckRunLogger) {
  SessionOptions so;

  so.session_logid = "CheckRunLogger";

  // create CapturingSink. LoggingManager will own it, but as long as the logging_manager
  // is around our pointer stays valid.
  auto capturing_sink = new CapturingSink();

  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<ISink>(capturing_sink), logging::Severity::kVERBOSE, false,
      LoggingManager::InstanceType::Temporal);

  InferenceSession session_object{so, logging_manager.get()};
  ASSERT_TRUE(session_object.Load(MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "RunTag";
  RunModel(session_object, run_options);

#ifndef NDEBUG
  // check for some VLOG output to make sure tag was correct. VLOG is not enabled in release build
  auto& msgs = capturing_sink->Messages();
  std::copy(msgs.begin(), msgs.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
  bool have_log_entry_with_run_tag =
      (std::find_if(msgs.begin(), msgs.end(),
                    [&run_options](std::string msg) {
                      return msg.find(run_options.run_tag) != string::npos;
                    }) != msgs.end());

  ASSERT_TRUE(have_log_entry_with_run_tag);
#endif
}

TEST(InferenceSessionTests, CheckRunProfilerWithSessionOptions) {
  SessionOptions so;

  so.session_logid = "CheckRunProfiler";
  so.enable_profiling = true;
  so.profile_file_prefix = ORT_TSTR("onnxprofile_profile_test");

  InferenceSession session_object(so);
  ASSERT_TRUE(session_object.Load(MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "RunTag";

  RunModel(session_object, run_options);
  std::string profile_file = session_object.EndProfiling();

  std::ifstream profile(profile_file);
  ASSERT_TRUE(profile);
  std::string line;

  std::vector<std::string> tags = {"pid", "dur", "ts", "ph", "X", "name", "args"};
  int count = 0;
  while (std::getline(profile, line)) {
    if (count == 0) {
      ASSERT_TRUE(line.find("[") != string::npos);
    } else if (count <= 7) {
      for (auto& s : tags) {
        ASSERT_TRUE(line.find(s) != string::npos);
      }
    } else {
      ASSERT_TRUE(line.find("]") != string::npos);
    }

    if (count == 1) {
      ASSERT_TRUE(line.find("model_loading_uri") != string::npos);
    }
    count++;
  }
}

TEST(InferenceSessionTests, CheckRunProfilerWithStartProfile) {
  SessionOptions so;

  so.session_logid = "CheckRunProfiler";

  InferenceSession session_object(so);
  ASSERT_TRUE(session_object.Load(MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "RunTag";

  session_object.StartProfiling("onnxruntime_profile_custom");
  RunModel(session_object, run_options);
  std::string profile_file = session_object.EndProfiling();

  std::ifstream profile(profile_file);
  std::string line;

  std::vector<std::string> tags = {"pid", "dur", "ts", "ph", "X", "name", "args"};
  int count = 0;
  while (std::getline(profile, line)) {
    if (count == 0) {
      ASSERT_TRUE(line.find("[") != string::npos);
    } else if (count <= 5) {
      for (auto& s : tags) {
        ASSERT_TRUE(line.find(s) != string::npos);
      }
    } else {
      ASSERT_TRUE(line.find("]") != string::npos);
    }

    if (count == 1) {
      ASSERT_TRUE(line.find("mul_1_fence_before") != string::npos);
    }
    count++;
  }
}

TEST(InferenceSessionTests, MultipleSessionsNoTimeout) {
  SessionOptions session_options;

  session_options.session_logid = "InferenceSessionTests.MultipleSessionsNoTimeout";
  InferenceSession session_object{session_options, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  std::thread thread1{[&session_object]() {
    RunOptions run_options;
    run_options.run_tag = "one session/thread 1";
    RunModel(session_object, run_options);
  }};

  std::thread thread2{[&session_object]() {
    RunOptions run_options;
    run_options.run_tag = "one session/thread 2";
    RunModel(session_object, run_options);
  }};

  thread1.join();
  thread2.join();
}

TEST(InferenceSessionTests, PreAllocateOutputVector) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.PreAllocateOutputVector";

  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "InferenceSessionTests.PreAllocateOutputVector";
  bool is_preallocate_output_vec = true;
  RunModel(session_object, run_options, is_preallocate_output_vec);
}

TEST(InferenceSessionTests, ConfigureVerbosityLevel) {
  SessionOptions so;

  so.session_logid = "ConfigureVerbosityLevel";
  so.session_log_verbosity_level = 1;

  // create CapturingSink. LoggingManager will own it, but as long as the logging_manager
  // is around our pointer stays valid.
  auto capturing_sink = new CapturingSink();

  auto logging_manager = std::make_unique<logging::LoggingManager>(
      std::unique_ptr<ISink>(capturing_sink),
      logging::Severity::kVERBOSE,
      false,
      LoggingManager::InstanceType::Temporal);

  InferenceSession session_object{so, logging_manager.get()};
  ASSERT_TRUE(session_object.Load(MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "ConfigureVerbosityLevel";
  run_options.run_log_verbosity_level = 1;
  RunModel(session_object, run_options);

#ifndef NDEBUG
  // check for some VLOG output to make sure tag was correct. VLOG is not enabled in release build
  auto& msgs = capturing_sink->Messages();
  std::copy(msgs.begin(), msgs.end(), std::ostream_iterator<std::string>(std::cout, "\n"));
  bool have_log_entry_with_vlog_session_msg =
      (std::find_if(msgs.begin(), msgs.end(),
                    [&](std::string msg) { return msg.find("Added input argument with name") != string::npos; }) !=
       msgs.end());

  ASSERT_TRUE(have_log_entry_with_vlog_session_msg);

  bool have_log_entry_with_vlog_run_msg =
      (std::find_if(msgs.begin(), msgs.end(),
                    [&](std::string msg) { return msg.find("Size of execution plan vector") != string::npos; }) !=
       msgs.end());

  ASSERT_TRUE(have_log_entry_with_vlog_run_msg);
#endif
}

TEST(InferenceSessionTests, TestWithIstream) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.TestWithIstream";

  InferenceSession session_object{so};

  std::ifstream model_file_stream(MODEL_URI, ios::in | ios::binary);
  ASSERT_TRUE(model_file_stream.good());
  ASSERT_TRUE(session_object.Load(model_file_stream).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "InferenceSessionTests.TestWithIstream";
  RunModel(session_object, run_options);
}

TEST(InferenceSessionTests, TestRegisterExecutionProvider) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.TestWithIstream";

  InferenceSession session_object{so};
  CPUExecutionProviderInfo epi;
  ASSERT_TRUE(session_object.RegisterExecutionProvider(std::make_unique<CPUExecutionProvider>(epi)).IsOK());

  std::ifstream model_file_stream(MODEL_URI, ios::in | ios::binary);
  ASSERT_TRUE(model_file_stream.good());
  ASSERT_TRUE(session_object.Load(model_file_stream).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "InferenceSessionTests.TestWithIstream";
  RunModel(session_object, run_options);
}

static void TestBindHelper(const std::string& log_str,
                           ProviderType bind_provider_type,
                           ProviderType run_provider_type,
                           bool preallocate_output,
                           ProviderType allocation_provider = kCpuExecutionProvider) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests." + log_str;
  so.session_log_verbosity_level = 1;  // change to 1 for detailed logging

  InferenceSession session_object{so, &DefaultLoggingManager()};

  if (bind_provider_type == kCudaExecutionProvider || run_provider_type == kCudaExecutionProvider) {
#ifdef USE_CUDA
    CUDAExecutionProviderInfo epi;
    epi.device_id = 0;
    EXPECT_TRUE(session_object.RegisterExecutionProvider(std::make_unique<CUDAExecutionProvider>(epi)).IsOK());
#endif
  }

  std::unique_ptr<Model> p_model;
  CreateMatMulModel(p_model, run_provider_type);

  std::string s1;
  p_model->ToProto().SerializeToString(&s1);
  std::stringstream sstr(s1);
  ASSERT_TRUE(session_object.Load(sstr).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_log_verbosity_level = so.session_log_verbosity_level;
  run_options.run_tag = so.session_logid;
  RunModelWithBindingMatMul(session_object,
                            run_options,
                            bind_provider_type,
                            preallocate_output,
                            allocation_provider);
}

TEST(InferenceSessionTests, TestBindCpu) {
  TestBindHelper("TestBindCpu",
                 kCpuExecutionProvider,
                 kCpuExecutionProvider,
                 false /* don't preallocate output */);
}

TEST(InferenceSessionTests, TestIOBindingReuse) {
  SessionOptions so;
  InferenceSession session_object(so);
  std::unique_ptr<Model> p_model;
  CreateMatMulModel(p_model, kCpuExecutionProvider);

  std::string s1;
  p_model->ToProto().SerializeToString(&s1);
  std::stringstream sstr(s1);
  ASSERT_TRUE(session_object.Load(sstr).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());
  unique_ptr<IOBinding> io_binding;
  Status st = session_object.NewIOBinding(&io_binding);
  ASSERT_TRUE(st.IsOK());

  MLValue ml_value1;
  vector<float> v1{2.f};
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {1}, v1, &ml_value1);
  io_binding->BindOutput("foo", ml_value1);
  ASSERT_TRUE(io_binding->GetOutputs().size() == 1);
  auto span = io_binding->GetOutputs()[0].Get<Tensor>().DataAsSpan<float>();
  ASSERT_TRUE(static_cast<size_t>(span.size()) == v1.size());
  for (size_t i = 0; i < v1.size(); ++i) {
    ASSERT_TRUE(v1[i] == span[i]);
  }

  MLValue ml_value2;
  vector<float> v2{3.f};
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), {1}, v2, &ml_value2);
  io_binding->BindOutput("foo", ml_value2);
  ASSERT_TRUE(io_binding->GetOutputs().size() == 1);
  span = io_binding->GetOutputs()[0].Get<Tensor>().DataAsSpan<float>();
  ASSERT_TRUE(static_cast<size_t>(span.size()) == v2.size());
  for (size_t i = 0; i < v2.size(); ++i) {
    ASSERT_TRUE(v2[i] == span[i]);
  }
}

TEST(InferenceSessionTests, InvalidInputTypeOfTensorElement) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.InvalidInputTypeOfTensorElement";

  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  // prepare inputs
  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<int64_t> values_mul_x = {1, 2, 3, 4, 5, 6};
  MLValue ml_value;
  CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                         &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<MLValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_y = {3, 2};
  std::vector<float> expected_values_mul_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(!st.IsOK());
}

#ifdef USE_CUDA

TEST(InferenceSessionTests, TestBindCuda) {
  TestBindHelper("TestBindCuda",
                 kCudaExecutionProvider,
                 kCudaExecutionProvider,
                 false /* don't preallocate output */);
}

TEST(InferenceSessionTests, TestBindCudaPreallocateOutputOnCuda) {
  TestBindHelper("TestBindCudaPreallocateOutputOnCuda",
                 kCudaExecutionProvider,
                 kCudaExecutionProvider,
                 true /* preallocate output on GPU */,
                 kCudaExecutionProvider);
}

TEST(InferenceSessionTests, TestBindCudaPreallocateOutputOnCpu) {
  TestBindHelper("TestBindCudaPreallocateOutputOnCpu",
                 kCudaExecutionProvider,
                 kCudaExecutionProvider,
                 true /* preallocate output on CPU */,
                 kCpuExecutionProvider);
}

TEST(InferenceSessionTests, TestBindCudaPreallocateOutputOnCpu2) {
  TestBindHelper("TestBindCudaPreallocateOutputOnCpu2",
                 kCudaExecutionProvider,
                 kCpuExecutionProvider,
                 true /* preallocate output on CPU */,
                 kCpuExecutionProvider);
}

#endif

TEST(InferenceSessionTests, ModelWithoutOpset) {
  SessionOptions so;

  so.session_logid = "InferenceSessionTests.ModelWithoutOpset";

  InferenceSession session_object{so, &DefaultLoggingManager()};
  Status retval = session_object.Load(MODEL_URI_NO_OPSET);
  ASSERT_FALSE(retval.IsOK());
  if (!retval.IsOK()) {
    ASSERT_TRUE(retval.ErrorMessage().find("Missing opset in the model") != std::string::npos);
  }
}

static ONNX_NAMESPACE::ModelProto CreateModelWithOptionalInputs() {
  Model model("ModelWithOptionalInputs");
  auto& graph = model.MainGraph();

  // create an initializer, which is an optional input that can be overridden
  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.add_dims(1);
  tensor_proto.set_data_type(TensorProto_DataType_FLOAT);
  tensor_proto.add_float_data(1.f);
  tensor_proto.set_name("optional_input");

  graph.AddInitializedTensor(tensor_proto);

  TypeProto single_float;
  single_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  single_float.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto& required_input = graph.GetOrCreateNodeArg("required_input", &single_float);
  auto& optional_input = graph.GetOrCreateNodeArg("optional_input", nullptr);
  auto& add_output = graph.GetOrCreateNodeArg("add_output", &single_float);

  EXPECT_TRUE(optional_input.Shape() != nullptr) << "AddInitializedTensor should have created the NodeArg with shape.";

  graph.AddNode("add", "Add", "Add required and optional inputs", {&required_input, &optional_input}, {&add_output});

  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  auto model_proto = model.ToProto();

  return model_proto;
}

static common::Status RunOptionalInputTest(bool add_required_input,
                                           bool add_optional_input,
                                           bool add_invalid_input) {
  auto model_proto = CreateModelWithOptionalInputs();

  SessionOptions so;
  so.session_logid = "InferenceSessionTests.TestOptionalInputs";

  InferenceSession session_object{so, &DefaultLoggingManager()};

  std::string s1;
  model_proto.SerializeToString(&s1);
  std::stringstream sstr(s1);
  auto status = session_object.Load(sstr);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  status = session_object.Initialize();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  // prepare inputs
  std::vector<int64_t> dims = {1};
  std::vector<float> required_input_val = {1.f};
  std::vector<float> optional_input_val = {10.f};  // override initializer value of 1
  std::vector<float> unknown_input_val = {20.f};

  MLValue required_input_mlvalue;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault),
                       dims, required_input_val, &required_input_mlvalue);

  MLValue optional_input_mlvalue;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault),
                       dims, optional_input_val, &optional_input_mlvalue);

  MLValue unknown_input_mlvalue;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault),
                       dims, unknown_input_val, &unknown_input_mlvalue);

  NameMLValMap feeds;

  if (add_required_input)
    feeds.insert(std::make_pair("required_input", required_input_mlvalue));

  if (add_optional_input)
    feeds.insert(std::make_pair("optional_input", optional_input_mlvalue));

  if (add_invalid_input)
    feeds.insert(std::make_pair("unknown_input", unknown_input_mlvalue));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("add_output");
  std::vector<MLValue> fetches;

  float expected_value = required_input_val[0];
  expected_value += add_optional_input ? optional_input_val[0] : 1.f;

  status = session_object.Run(run_options, feeds, output_names, &fetches);

  if (status.IsOK()) {
    MLValue& output = fetches.front();
    const auto& tensor = output.Get<Tensor>();
    float output_value = *tensor.Data<float>();
    if (output_value != expected_value) {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Output of ", output_value, " != ", expected_value);
    }
  }

  return status;
}

TEST(InferenceSessionTests, TestOptionalInputs) {
  // required input only
  auto status = RunOptionalInputTest(true, false, false);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  // required and optional input
  status = RunOptionalInputTest(true, true, false);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  // required, optional and invalid input
  status = RunOptionalInputTest(true, true, true);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("Invalid Feed Input Name"));

  // missing required
  status = RunOptionalInputTest(false, true, false);
  ASSERT_FALSE(status.IsOK());
  EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr("Missing Input:"));
}

TEST(ExecutionProviderTest, FunctionTest) {
  onnxruntime::Model model("graph_1");
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor.
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
  inputs.push_back(&input_arg_1);
  inputs.push_back(&input_arg_2);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

  auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&input_arg_3);
  auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &float_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  std::string model_file_name = "execution_provider_test_graph.onnx";
  status = onnxruntime::Model::Save(model, model_file_name);

  SessionOptions so;
  so.session_logid = "ExecutionProviderTest.FunctionTest";
  InferenceSession session_object{so};
  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());

  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  CPUExecutionProviderInfo epi;
  auto testCPUExecutionProvider = std::make_unique<::onnxruntime::CPUExecutionProvider>(epi);

  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  MLValue ml_value_x;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value_x);
  MLValue ml_value_y;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value_y);
  MLValue ml_value_z;
  CreateMLValue<float>(testCPUExecutionProvider->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("M");
  std::vector<MLValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_m = {3, 2};
  std::vector<float> expected_values_mul_m = {3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f};

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
  VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);

  InferenceSession session_object_2{so};
  session_object_2.RegisterExecutionProvider(std::move(testCPUExecutionProvider));
  session_object_2.RegisterExecutionProvider(std::make_unique<::onnxruntime::FuseExecutionProvider>());
  status = session_object_2.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object_2.Initialize();
  ASSERT_TRUE(status.IsOK());
  status = session_object_2.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
  VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);
}

TEST(ExecutionProviderTest, FunctionInlineTest) {
  onnxruntime::Model model("graph_1");

  ONNX_NAMESPACE::FunctionProto fc_proto;
  fc_proto.set_name("FC");
  fc_proto.set_doc_string("this is a full connection function.");
  fc_proto.set_since_version(7);
  fc_proto.add_input("w");
  fc_proto.add_input("x");
  fc_proto.add_input("b");
  fc_proto.add_output("y");
  NodeProto* node0 = fc_proto.add_node();
  node0->set_name("node0");
  node0->set_domain("");
  node0->set_doc_string("This is a matmul testing node ");
  node0->set_op_type("MatMul");
  node0->add_input("w");
  node0->add_input("x");
  node0->add_output("y_1");
  NodeProto* node1 = fc_proto.add_node();
  node1->set_name("node1");
  node1->set_domain("");
  node1->set_doc_string("This is a add testing node ");
  node1->set_op_type("Add");
  node1->add_input("y_1");
  node1->add_input("b");
  node1->add_output("y");
  model.AddFunction(fc_proto);

  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor.
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
  float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
  auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
  inputs.push_back(&input_arg_1);
  inputs.push_back(&input_arg_2);
  inputs.push_back(&input_arg_3);
  auto& output_arg = graph.GetOrCreateNodeArg("M", &float_tensor);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "FC", "node 1.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  std::string model_file_name = "inline_test_graph.onnx";
  status = onnxruntime::Model::Save(model, model_file_name);

  SessionOptions so;
  so.session_logid = "ExecutionProviderTest.FunctionInlineTest";
  InferenceSession session_object{so};
  status = session_object.Load(model_file_name);
  ASSERT_TRUE(status.IsOK());
  status = session_object.Initialize();
  ASSERT_TRUE(status.IsOK());

  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  std::vector<int64_t> dims_mul_x = {2, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f};
  MLValue ml_value_x;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &ml_value_x);
  MLValue ml_value_y;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &ml_value_y);
  MLValue ml_value_z;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("M");
  std::vector<MLValue> fetches;

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_m = {2, 2};
  std::vector<float> expected_values_mul_m = {8.0f, 12.0f, 18.0f, 26.0f};

  // Now run
  status = session_object.Run(run_options, feeds, output_names, &fetches);
  ASSERT_TRUE(status.IsOK());
  VerifyOutputs(fetches, expected_dims_mul_m, expected_values_mul_m);
}

TEST(InferenceSessionTests, TestTruncatedSequence) {
  // model/data generated by <repo>/onnxruntime/test/testdata/CNTK/gen.py GenScan()
  static const std::string LSTM_MODEL_URI = "testdata/scan_1.pb";
  // This model is a 4x forward LSTM. Parse it to find out mapping between init_state input/output
  ONNX_NAMESPACE::ModelProto model_proto;
  int model_fd;
  auto status = Env::Default().FileOpenRd(LSTM_MODEL_URI, model_fd);
  ASSERT_TRUE(status.IsOK());
  google::protobuf::io::FileInputStream f(model_fd);
  f.SetCloseOnDelete(true);
  ASSERT_TRUE(model_proto.ParseFromZeroCopyStream(&f));
  GraphProto& graph_proto = *model_proto.mutable_graph();

  auto find_attr = [&](const NodeProto& node, const std::string& attr_name) -> const AttributeProto* {
    for (int i = 0; i < node.attribute_size(); ++i) {
      auto& attr = node.attribute(i);
      if (attr.name() == attr_name)
        return &attr;
    }
    return nullptr;
  };

  std::unordered_map<std::string, std::string> init_state_map;
  for (int i_node = 0; i_node < graph_proto.node_size(); ++i_node) {
    auto& node = *graph_proto.mutable_node(i_node);
    if (node.op_type() == "Scan") {
      // only works in forward, and do not allow bidirection
      auto attr_directions = find_attr(node, "scan_input_directions");
      if (attr_directions != nullptr) {
        ASSERT_TRUE(attr_directions->ints_size() == 1);

        if (attr_directions->ints(0) == 1)
          continue;  // skip backward Scan
      }

      // input 0 is optional sequence length, 1..N are for initial states
      // and N+1..N+num_scan_inputs are actual inputs
      // output 0..N-1 are for output states, and N.. are actual outputs
      auto attr_num_scan_inputs = find_attr(node, "num_scan_inputs");
      ASSERT_TRUE(attr_num_scan_inputs != nullptr);
      int num_scan_inputs = gsl::narrow_cast<int>(attr_num_scan_inputs->i());
      ASSERT_TRUE(node.input_size() - num_scan_inputs < node.output_size());
      for (int i = 0; i < node.input_size() - num_scan_inputs; ++i) {
        init_state_map.insert(std::make_pair(node.output(i), node.input(i)));
      }
    }
  }

  // now run the truncated model
  SessionOptions so;
  InferenceSession session_object(so);
  ASSERT_TRUE(session_object.Load(LSTM_MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  std::vector<int64_t> X_dims = {5, 1, 3};
  std::vector<float> X = {0.5488135f, 0.71518934f, 0.60276335f,
                          0.5448832f, 0.4236548f, 0.6458941f,
                          0.4375872f, 0.891773f, 0.96366274f,
                          0.3834415f, 0.79172504f, 0.5288949f,
                          0.56804454f, 0.92559665f, 0.07103606f};

  std::vector<int64_t> Y_dims = {5, 1, 2};
  std::vector<float> Y_data = {-1.1730184e-04f, -3.1204990e-04f,
                               -2.9978977e-04f, -1.0602647e-03f,
                               -3.8115133e-04f, -2.0684483e-03f,
                               -2.5120965e-04f, -2.9920202e-03f,
                               3.0980256e-05f, -3.5933927e-03f};

  MLValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), X_dims, X, &ml_value);

  std::string input_name = "Input13165";
  NameMLValMap feeds = {{input_name, ml_value}};

  // prepare outputs for whole sequence
  std::string final_output_name = "";
  int final_output_index = -1;
  for (int i = 0; i < graph_proto.output_size(); ++i) {
    if (init_state_map.find(graph_proto.output(i).name()) == init_state_map.end()) {
      ASSERT_TRUE(final_output_name.empty());
      final_output_name = graph_proto.output(i).name();
      final_output_index = i;
    }
  }

  std::vector<std::string> output_names = {final_output_name};
  std::vector<MLValue> fetches;

  // Now run the full sequence
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(st.IsOK());
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(Y_dims);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  for (size_t i = 0; i < Y_data.size(); ++i)
    EXPECT_NEAR(Y_data[i], rtensor.template Data<float>()[i], FLT_EPSILON);

  // run truncated sequence
  output_names.clear();
  for (int i = 0; i < graph_proto.output_size(); ++i) {
    output_names.push_back(graph_proto.output(i).name());
  }
  fetches.clear();

  std::vector<int> truncated_lengths = {2, 2, 1};              // sums to non-truncated length
  auto seq_stride = TensorShape(X_dims).SizeFromDimension(1);  // sequence is the first dimension of input shape
  int seq_start = 0;
  for (auto truncated_len : truncated_lengths) {
    std::vector<int64_t> truncated_input_dims = X_dims;
    truncated_input_dims[0] = truncated_len;
    MLValue truncated_ml_value;
    std::vector<float> truncated_input(X.begin() + seq_start * seq_stride, X.begin() + (seq_start + truncated_len) * seq_stride);
    CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), truncated_input_dims, truncated_input, &truncated_ml_value);
    NameMLValMap truncated_feeds = {{input_name, truncated_ml_value}};
    if (seq_start > 0) {
      // continue from truncated sequence
      ASSERT_TRUE(fetches.size() == output_names.size());
      for (size_t i_output = 0; i_output < output_names.size(); ++i_output) {
        auto iter = init_state_map.find(output_names[i_output]);
        if (iter != init_state_map.end())
          truncated_feeds.insert(std::make_pair(iter->second, fetches[i_output]));
      }
    }
    std::vector<MLValue> truncated_fetches;
    st = session_object.Run(run_options, truncated_feeds, output_names, &truncated_fetches);
    if (!st.IsOK()) {
      std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
    }
    ASSERT_TRUE(st.IsOK());

    // check truncated output
    auto& truncated_rtensor = truncated_fetches[final_output_index].Get<Tensor>();
    std::vector<int64_t> truncated_output_dims = Y_dims;
    truncated_output_dims[0] = truncated_len;
    TensorShape truncated_shape(truncated_output_dims);
    ASSERT_EQ(truncated_shape, truncated_rtensor.Shape());
    auto seq_output_stride = truncated_shape.SizeFromDimension(1);
    for (int i = 0; i < truncated_shape.Size(); ++i)
      EXPECT_NEAR(Y_data[i + seq_start * seq_output_stride], truncated_rtensor.template Data<float>()[i], FLT_EPSILON);

    // prepare for next truncated input
    fetches = truncated_fetches;
    seq_start += truncated_len;
  }
}

// create the feeds and fetches using the dummy allocator so that we have to copy to CPU to execute, and from
// CPU to return in utils::ExecuteGraph. Call InferenceSession::Run twice to test the caching of the copy logic.
TEST(InferenceSessionTests, TestCopyToFromDevices) {
  SessionOptions so;
  so.session_logid = "InferenceSessionTests.TestCopyToFromDevices";
  InferenceSession session_object{so, &DefaultLoggingManager()};

  ASSERT_TRUE(session_object.Load(MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  auto dummy_provider = std::make_unique<DummyExecutionProvider>();
  auto* p_dummy_provider = dummy_provider.get();
  session_object.RegisterExecutionProvider(std::move(dummy_provider));

  // prepare inputs
  std::vector<int64_t> dims_mul_x = {3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  MLValue ml_value;
  CreateMLValue<float>(p_dummy_provider->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                       &ml_value);

  std::vector<std::string> feed_names;
  std::vector<MLValue> feeds;
  feed_names.push_back("X");
  feeds.push_back(ml_value);

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_y = {3, 2};
  std::vector<float> expected_values_mul_y = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};

  auto run_test = [&](int run_num) {
    // prepare outputs
    std::vector<std::string> output_names;
    std::vector<MLValue> fetches;
    output_names.push_back("Y");

    fetches.resize(output_names.size());
    for (auto& elem : fetches) {
      CreateMLValue<float>(p_dummy_provider->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x,
                           &elem);
    }

    // Now run
    RunOptions run_options;
    run_options.run_tag = "run:" + std::to_string(run_num);

    common::Status st = session_object.Run(run_options, feed_names, feeds, output_names, &fetches);
    ASSERT_TRUE(st.IsOK()) << st.ErrorMessage();

    VerifyOutputs(fetches, expected_dims_mul_y, expected_values_mul_y);
  };

  int run_number = 0;
  run_test(run_number++);
  run_test(run_number++);
}

// This test validates the RegisterTransformer API
// It creates and registers a dummy transformer and after session initialize
// validates that this transformer was called regardless of the graph optimization level set.
TEST(InferenceSessionTests, TestRegisterTransformers) {
  string model_uri = "testdata/transform/fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";

  for (int i = static_cast<int>(TransformerLevel::Default); i < static_cast<int>(TransformerLevel::MaxTransformerLevel); i++) {
    SessionOptions so;
    so.session_logid = "InferenceSessionTests.TestL1AndL2Transformers";
    so.graph_optimization_level = static_cast<TransformerLevel>(i);
    InferenceSession session_object{so, &DefaultLoggingManager()};

    // Create and register dummy graph transformer
    auto dummy_transformer_unique_ptr = std::make_unique<DummyGraphTransformer>("DummyTransformer");
    const auto* dummy_transformer = dummy_transformer_unique_ptr.get();
    session_object.RegisterGraphTransformer(std::move(dummy_transformer_unique_ptr));

    session_object.Load(model_uri);
    ASSERT_TRUE(session_object.Initialize().IsOK());

    // Validate transformer was called after Session.Initialize
    ASSERT_TRUE(dummy_transformer->IsTransformerInvoked());
  }
}

// This test validates session initialize is successful when all the pre-defined
// L1 and L2 transformers are enabled.
TEST(InferenceSessionTests, TestL1AndL2Transformers) {
  // Models which cover all transformers.
  std::vector<std::string> test_model_uris = {"testdata/transform/fusion/fuse-conv-bn-mul-add-unsqueeze.onnx",
                                              "testdata/transform/abs-id-max.onnx",
                                              "testdata/transform/slice-elim.onnx",
                                              "testdata/transform/matmul_add_fusion/2Input/model.onnx",
                                              "testdata/transform/matmul_add_fusion/3Input/gemm_relu.onnx",
                                              "testdata/transform/fusion/fuse-conv-bn-add-mul-float16.onnx"};

  for (const auto& model_uri : test_model_uris) {
    SessionOptions so;
    so.session_logid = "InferenceSessionTests.TestL1AndL2Transformers";
    so.graph_optimization_level = TransformerLevel::Level2;
    InferenceSession session_object{so, &DefaultLoggingManager()};
    ASSERT_TRUE(session_object.Load(model_uri).IsOK());
    ASSERT_TRUE(session_object.Initialize().IsOK());
  }
}

#ifdef USE_CUDA

TEST(InferenceSessionTests, TestParallelExecutionWithCudaProvider) {
  string model_uri = "testdata/transform/fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";

  SessionOptions so;
  so.enable_sequential_execution = false;
  so.session_logid = "InferenceSessionTests.TestParallelExecutionWithCudaProvider";
  InferenceSession session_object{so};
  
  CUDAExecutionProviderInfo epi;
  epi.device_id = 0;
  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::make_unique<CUDAExecutionProvider>(epi)).IsOK());

  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  auto status = session_object.Initialize();

  ASSERT_TRUE(!status.IsOK());
}

#endif

}  // namespace test
}  // namespace onnxruntime
