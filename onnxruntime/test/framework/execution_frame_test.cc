// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/make_unique.h"
#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/model.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "test_utils.h"
#include "test/test_environment.h"
#include "test/framework/TestAllocatorManager.h"
#include "asserts.h"
#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ONNX_NAMESPACE;
using namespace std;

namespace onnxruntime {
namespace test {
typedef std::vector<onnxruntime::NodeArg*> ArgMap;

std::unique_ptr<IExecutionProvider> CreateCPUExecutionProvider() {
  CPUExecutionProviderInfo info;
  return onnxruntime::make_unique<CPUExecutionProvider>(info);
}

class ExecutionFrameTest : public ::testing::Test {
 protected:
  concurrency::ThreadPool tp_;
  ExecutionFrameTest() : tp_(&onnxruntime::Env::Default(), ThreadOptions(), ORT_TSTR("ExecutionFrameTest"), 2, true) {
  }
};

TEST_F(ExecutionFrameTest, TensorAllocationTest) {
  onnxruntime::Model model("test", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}}, {}, DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model.MainGraph();
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  onnxruntime::NodeArg input_def("X", &tensor_float), output_def("Y", &tensor_float);

  onnxruntime::Node* node = &graph.AddNode("node1", "Relu", "Relu operator", ArgMap{&input_def}, ArgMap{&output_def});
  node->SetExecutionProviderType(kCpuExecutionProvider);
  ASSERT_STATUS_OK(graph.Resolve());

  auto cpu_xp = CreateCPUExecutionProvider();
  auto xp_typ = cpu_xp->Type();
  ExecutionProviders execution_providers;
  execution_providers.Add(xp_typ, std::move(cpu_xp));
  KernelRegistryManager kernel_registry_manager;
  ASSERT_STATUS_OK(kernel_registry_manager.RegisterKernels(execution_providers));

  DataTransferManager dtm;
  profiling::Profiler profiler;
  SessionState state(graph, execution_providers, true, &tp_, nullptr, dtm,
                     DefaultLoggingManager().DefaultLogger(), profiler);

  node->SetExecutionProviderType(xp_typ);

  ASSERT_STATUS_OK(state.FinalizeSessionState(ORT_TSTR(""), kernel_registry_manager));

  vector<OrtValue> outputs;
  ExecutionFrame frame({}, {}, {}, outputs, {}, state);

  int start_index = frame.GetNodeOffset(node->Index());
  ASSERT_EQ(start_index, 0);

  TensorShape shape(std::vector<int64_t>{2, 3});
  OrtValue& mlvalue0 = *frame.GetMutableNodeInputOrOutputMLValue(start_index);
  const auto& memory_info = execution_providers.Get(xp_typ)->GetAllocator(0, OrtMemTypeDefault)->Info();
  ASSERT_STATUS_OK(frame.AllocateMLValueTensorSelfOwnBuffer(mlvalue0, start_index, DataTypeImpl::GetType<float>(),
                                                            memory_info, shape));

  OrtValue* p_ml_value = frame.GetMutableNodeInputOrOutputMLValue(0);
  ASSERT_TRUE(p_ml_value != nullptr);
  Tensor* p_tensor = p_ml_value->GetMutable<Tensor>();
  ASSERT_TRUE(p_tensor != nullptr);
  //Use reinterpret_cast to bypass a gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51213
  ASSERT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&p_tensor->Shape()),
            *reinterpret_cast<const std::vector<int64_t>*>(&shape));
  ASSERT_EQ(p_tensor->DataType(), DataTypeImpl::GetType<float>());

  //test share memory from tensor
  TensorShape shape2(std::vector<int64_t>{3, 2});
  OrtValue& mlvalue1 = *frame.GetMutableNodeInputOrOutputMLValue(start_index + 1);
  ASSERT_STATUS_OK(frame.AllocateMLValueTensorPreAllocateBuffer(mlvalue1,
                                                                start_index,
                                                                DataTypeImpl::GetType<float>(),
                                                                p_tensor->Location(),
                                                                shape2));

  const OrtValue* p_ml_value_const = frame.GetNodeInputOrOutputMLValue(1);
  auto tensor2 = p_ml_value_const ? &(p_ml_value_const->Get<Tensor>()) : nullptr;
  ASSERT_TRUE(tensor2);
  //Use reinterpret_cast to bypass a gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51213
  ASSERT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&tensor2->Shape()),
            *reinterpret_cast<const std::vector<int64_t>*>(&shape2));
  ASSERT_EQ(tensor2->template Data<float>(), p_tensor->template Data<float>());
}

TEST_F(ExecutionFrameTest, FeedInDataTest) {
  onnxruntime::Model model("test", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                           std::unordered_map<std::string, int>{{"", 10}}, {},
                           DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model.MainGraph();
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  onnxruntime::NodeArg input_def("X", &tensor_float), output_def("Y", &tensor_float);

  graph.AddNode("node1", "Clip", "Clip operator", ArgMap{&input_def}, ArgMap{&output_def})
      .SetExecutionProviderType(kCpuExecutionProvider);
  graph.Resolve();
  auto element_type = DataTypeImpl::GetType<float>();
  TensorShape shape({3, 2});
  std::vector<float> fdata(static_cast<size_t>(shape.Size()));
  //create fake ml value with owned buffer.
  OrtMemoryInfo cpuinfo(kCpuExecutionProvider, OrtDeviceAllocator);
  std::unique_ptr<Tensor> p_tensor = onnxruntime::make_unique<Tensor>(element_type, shape, fdata.data(), cpuinfo);
  OrtValue value;
  value.Init(p_tensor.release(),
             DataTypeImpl::GetType<Tensor>(),
             DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  auto cpu_xp = CreateCPUExecutionProvider();
  auto xp_typ = cpu_xp->Type();

  KernelRegistryManager kernel_registry_manager;
  ExecutionProviders execution_providers;
  execution_providers.Add(xp_typ, std::move(cpu_xp));
  ASSERT_STATUS_OK(kernel_registry_manager.RegisterKernels(execution_providers));

  DataTransferManager dtm;
  profiling::Profiler profiler;
  SessionState state(graph, execution_providers, true, &tp_, nullptr, dtm,
                     DefaultLoggingManager().DefaultLogger(), profiler);

  ASSERT_STATUS_OK(state.FinalizeSessionState(ORT_TSTR(""), kernel_registry_manager));
  
  const OrtValueNameIdxMap& mlvalue_name_idx_map = state.GetOrtValueNameIdxMap();
  int x_idx = -1, y_idx = -1;
  ASSERT_TRUE(mlvalue_name_idx_map.GetIdx("X", x_idx).IsOK());
  ASSERT_TRUE(mlvalue_name_idx_map.GetIdx("Y", y_idx).IsOK());

  vector<OrtValue> outputs;
  ExecutionFrame frame({x_idx}, {value}, {y_idx}, outputs, {}, state);

  OrtValue* p_ml_value = frame.GetMutableNodeInputOrOutputMLValue(0);
  Tensor* p_tensor_arg_0 = p_ml_value ? p_ml_value->GetMutable<Tensor>() : nullptr;
  ASSERT_TRUE(p_tensor_arg_0);
  //Use reinterpret_cast to bypass a gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51213
  ASSERT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&p_tensor_arg_0->Shape()),
            *reinterpret_cast<const std::vector<int64_t>*>(&shape));
  ASSERT_EQ(p_tensor_arg_0->DataType(), DataTypeImpl::GetType<float>());
  ASSERT_EQ(p_tensor_arg_0->MutableData<float>(), value.GetMutable<Tensor>()->MutableData<float>());
}

TEST_F(ExecutionFrameTest, MemPatternTest) {
  auto cpu_xp = CreateCPUExecutionProvider();
  auto xp_type = cpu_xp->Type();
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[onnxruntime::kOnnxDomain] = 7;
  onnxruntime::Model model("test", true, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
                           domain_to_version, {}, DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model.MainGraph();
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  onnxruntime::NodeArg input_def1("X1", &tensor_float),
      input_def2("X2", &tensor_float),
      input_def3("X3", &tensor_float),
      gemm1_out_def("T1", &tensor_float),
      gemm2_out_def("T2", &tensor_float),
      clip_out_def("T3", &tensor_float);

  graph.AddNode("node1", "MatMul", "gemm1", ArgMap{&input_def1, &input_def2}, ArgMap{&gemm1_out_def})
      .SetExecutionProviderType(xp_type);
  graph.AddNode("node2", "MatMul", "gemm2", ArgMap{&gemm1_out_def, &input_def3}, ArgMap{&gemm2_out_def})
      .SetExecutionProviderType(xp_type);
  graph.AddNode("node3", "Clip", "clip1", ArgMap{&gemm2_out_def}, ArgMap{&clip_out_def})
      .SetExecutionProviderType(xp_type);

  ASSERT_STATUS_OK(graph.Resolve());

  KernelRegistryManager kernel_registry_manager;

  ExecutionProviders execution_providers;
  execution_providers.Add(xp_type, std::move(cpu_xp));
  ASSERT_STATUS_OK(kernel_registry_manager.RegisterKernels(execution_providers));
  //1. prepare input

  DataTransferManager dtm;
  profiling::Profiler profiler;
  SessionState state(graph, execution_providers, true, &tp_, nullptr, dtm,
                     DefaultLoggingManager().DefaultLogger(), profiler);

  ASSERT_STATUS_OK(state.FinalizeSessionState(ORT_TSTR(""), kernel_registry_manager));

  const OrtValueNameIdxMap& mlvalue_name_idx_map(state.GetOrtValueNameIdxMap());

  int x1_idx = -1, x2_idx = -1, x3_idx = -1;
  int t1_idx = -1, t2_idx = -1, t3_idx = -1;
  ASSERT_TRUE(mlvalue_name_idx_map.GetIdx("X1", x1_idx).IsOK());
  ASSERT_TRUE(mlvalue_name_idx_map.GetIdx("X2", x2_idx).IsOK());
  ASSERT_TRUE(mlvalue_name_idx_map.GetIdx("X3", x3_idx).IsOK());

  ASSERT_TRUE(mlvalue_name_idx_map.GetIdx("T1", t1_idx).IsOK());
  ASSERT_TRUE(mlvalue_name_idx_map.GetIdx("T2", t2_idx).IsOK());
  ASSERT_TRUE(mlvalue_name_idx_map.GetIdx("T3", t3_idx).IsOK());

  auto cpu_allocator = execution_providers.Get(xp_type)->GetAllocator(0, OrtMemTypeDefault);

  OrtValue v1, v2, v3;
  CreateMLValue<float>(cpu_allocator,
                       std::vector<int64_t>{1, 2},
                       std::vector<float>{1.0f, 1.0f}, &v1);
  CreateMLValue<float>(cpu_allocator,
                       std::vector<int64_t>{2, 2},
                       std::vector<float>(4, 1.0f), &v2);
  CreateMLValue<float>(cpu_allocator,
                       std::vector<int64_t>{2, 3},
                       std::vector<float>(6, 1.0f), &v3);

  vector<OrtValue> outputs;
  ExecutionFrame frame({x1_idx, x2_idx, x3_idx}, {v1, v2, v3}, {t3_idx}, outputs, {}, state);

  OrtValue& mlvalue3 = *frame.GetMutableNodeInputOrOutputMLValue(3);
  OrtValue& mlvalue4 = *frame.GetMutableNodeInputOrOutputMLValue(4);
  OrtValue& mlvalue5 = *frame.GetMutableNodeInputOrOutputMLValue(5);

  ASSERT_STATUS_OK(frame.AllocateMLValueTensorSelfOwnBuffer(mlvalue3, 3,
                                                            DataTypeImpl::GetType<float>(),
                                                            cpu_allocator->Info(),
                                                            TensorShape(std::vector<int64_t>{2, 2})));

  ASSERT_STATUS_OK(frame.AllocateMLValueTensorSelfOwnBuffer(mlvalue4, 4,
                                                            DataTypeImpl::GetType<float>(),
                                                            cpu_allocator->Info(),
                                                            TensorShape(std::vector<int64_t>{2, 3})));

  ASSERT_STATUS_OK(frame.AllocateMLValueTensorSelfOwnBuffer(mlvalue5, 5,
                                                            DataTypeImpl::GetType<float>(),
                                                            cpu_allocator->Info(),
                                                            TensorShape(std::vector<int64_t>{2, 3})));
  MemoryPatternGroup pattern;
  ASSERT_STATUS_OK(frame.GeneratePatterns(&pattern));

  ASSERT_EQ(pattern.patterns.size(), pattern.locations.size());
  ASSERT_EQ(pattern.patterns.size(), 1u);
  auto p = pattern.GetPatterns(cpu_allocator->Info());
  ASSERT_EQ(p->PeakSize(), 2u * 64u);  // each allocation is 64-byte aligned
  ASSERT_EQ(p->GetBlock(3)->offset_, 0u);
  ASSERT_EQ(p->GetBlock(4)->offset_, 64u);
}

TEST(ExecutionFrameTestWithoutSessionState, BadModelInvalidDimParamUsage) {
  // load model with 2 Scan ops that both incorrectly use shapes of { 'None', 'None' } for their outputs.
  // as 'None' is not a special value it's treated as a variable name, leading to a runtime error when we
  // attempt to re-use the output from the first Scan node for the second. validate we detect this and error out.
  SessionOptions so;
  so.session_logid = "BadModelInvalidDimParamUsage";

  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load("testdata/invalid_dim_param_value_repetition.onnx"));
  ASSERT_STATUS_OK(session_object.Initialize());

  std::vector<int64_t> dims_X = {10, 6};
  std::vector<float> values_X;
  values_X.reserve(60);
  for (int i = 0; i < 60; ++i) {
    values_X.push_back(float(i));
  }

  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_X, values_X, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<OrtValue> fetches;

  // Now run
  RunOptions run_options;
  auto st = session_object.Run(run_options, feeds, output_names, &fetches);

  EXPECT_FALSE(st.IsOK()) << st;
  EXPECT_THAT(st.ErrorMessage(), testing::HasSubstr("Shape mismatch attempting to re-use buffer."));
}

// Test that when an initializer is a graph output it is handled correctly
TEST(ExecutionFrameTestInit, InitializerAsOutput) {
  const std::vector<float> expected{
      1.764052391052246f, 0.40015721321105957f, 0.978738009929657f, 2.2408931255340576f, 1.8675580024719238f,
      -0.9772778749465942f, 0.9500884413719177f, -0.15135720372200012f, -0.10321885347366333f, 0.4105985164642334f,
      0.14404356479644775f, 1.4542734622955322f, 0.7610377073287964f, 0.12167501449584961f, 0.44386324286460876f,
      0.3336743414402008f, 1.4940791130065918f, -0.2051582634449005f, 0.3130677044391632f, -0.8540957570075989f,
      -2.5529897212982178f, 0.653618574142456f, 0.8644362092018127f, -0.7421650290489197f, 2.269754648208618f};

  SessionOptions so;

  // test if pre-allocated fetch is provided the initializer values are copied into that buffer
  {
    InferenceSession session(so, GetEnvironment());
    ASSERT_STATUS_OK(session.Load(ORT_TSTR("testdata/initializer_as_output.onnx")));
    ASSERT_STATUS_OK(session.Initialize());

    auto allocator = test::AllocatorManager::Instance().GetAllocator(CPU);
    auto p_tensor = onnxruntime::make_unique<Tensor>(DataTypeImpl::GetType<float>(), TensorShape({5, 5}), allocator);
    const void* orig_buffer = p_tensor->DataRaw();

    std::vector<OrtValue> results;
    results.resize(1);
    results[0].Init(p_tensor.release(), DataTypeImpl::GetType<Tensor>(),
                    DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
    RunOptions ro;
    ASSERT_STATUS_OK(session.Run(ro, {}, {}, {"values"}, &results, nullptr));

    EXPECT_EQ(results[0].Get<Tensor>().DataRaw(), orig_buffer);
    EXPECT_THAT(results[0].Get<Tensor>().DataAsSpan<float>(), ::testing::ContainerEq(gsl::make_span(expected)));
  }

  // test that if no pre-allocated fetch is provided a new OrtValue is allocated for the results
  {
    class TestInferenceSesssion : public InferenceSession {
     public:
      TestInferenceSesssion(const SessionOptions& session_options,
                            const Environment& session_env)
          : InferenceSession(session_options, session_env) {
      }

      const SessionState& GetSessionState() const { return InferenceSession::GetSessionState(); }
    };

    TestInferenceSesssion session(so, GetEnvironment());
    ASSERT_STATUS_OK(session.Load(ORT_TSTR("testdata/initializer_as_output.onnx")));
    ASSERT_STATUS_OK(session.Initialize());

    std::vector<OrtValue> results;
    RunOptions ro;
    ASSERT_STATUS_OK(session.Run(ro, {}, {}, {"values"}, &results, nullptr));

    // output buffer should not be the same as the initializer in SessionState
    const auto& initializers = session.GetSessionState().GetInitializedTensors();
    EXPECT_NE(results[0].Get<Tensor>().DataRaw(), initializers.at(0).Get<Tensor>().DataRaw());
    EXPECT_THAT(results[0].Get<Tensor>().DataAsSpan<float>(), ::testing::ContainerEq(gsl::make_span(expected)));
  }
}

}  // namespace test
}  // namespace onnxruntime
