// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/model.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "test_utils.h"
#include "test/test_environment.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ONNX_NAMESPACE;
using namespace std;

namespace onnxruntime {
namespace test {
typedef std::vector<onnxruntime::NodeArg*> ArgMap;

std::shared_ptr<onnxruntime::Model> DummyGraphWithClip() {
  auto model = std::make_shared<onnxruntime::Model>("test", false, DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  onnxruntime::NodeArg input_def("X", &tensor_float), output_def("Y", &tensor_float);

  graph.AddNode("node1", "Clip", "clip operator", ArgMap{&input_def}, ArgMap{&output_def});
  return model;
}

std::unique_ptr<IExecutionProvider> CreateCPUExecutionProvider() {
  CPUExecutionProviderInfo info;
  return onnxruntime::make_unique<CPUExecutionProvider>(info);
}

class ExecutionFrameTest : public ::testing::Test {
 protected:
  concurrency::ThreadPool tp_{"test", 1};
};

TEST_F(ExecutionFrameTest, TensorAllocationTest) {
  onnxruntime::Model model("test", false, DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model.MainGraph();
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  onnxruntime::NodeArg input_def("X", &tensor_float), output_def("Y", &tensor_float);

  onnxruntime::Node* node = &graph.AddNode("node1", "Relu", "Relu operator", ArgMap{&input_def}, ArgMap{&output_def});
  node->SetExecutionProviderType(kCpuExecutionProvider);
  Status status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  auto cpu_xp = CreateCPUExecutionProvider();
  auto xp_typ = cpu_xp->Type();
  ExecutionProviders execution_providers;
  execution_providers.Add(xp_typ, std::move(cpu_xp));
  KernelRegistryManager kernel_registry_manager;
  status = kernel_registry_manager.RegisterKernels(execution_providers);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  SessionState state{execution_providers, true, &tp_, nullptr};
  status = state.SetGraphAndCreateKernels(graph, kernel_registry_manager);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  node->SetExecutionProviderType(xp_typ);

  std::unique_ptr<SequentialExecutionPlan> p_seq_exec_plan;
  // TODO below line is for testing only. In production use SequentialPlanner::CreatePlan()
  SequentialPlannerContext context(ExecutionMode::ORT_SEQUENTIAL);
  status = SequentialPlanner::CreatePlan(nullptr, GraphViewer(graph), {}, execution_providers, kernel_registry_manager,
                                         state.GetOrtValueNameIdxMap(), context, p_seq_exec_plan);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  state.SetExecutionPlan(std::move(p_seq_exec_plan));

  vector<OrtValue> outputs;
  ExecutionFrame frame({}, {}, {}, outputs, {}, state);

  int start_index = frame.GetNodeOffset(node->Index());
  EXPECT_EQ(start_index, 0);

  TensorShape shape(std::vector<int64_t>{2, 3});
  OrtValue& mlvalue0 = *frame.GetMutableNodeInputOrOutputMLValue(start_index);
  status = frame.AllocateMLValueTensorSelfOwnBuffer(mlvalue0, start_index, DataTypeImpl::GetType<float>(),
                                                    execution_providers.Get(xp_typ)->GetAllocator(0, OrtMemTypeDefault)->Info(), shape);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  OrtValue* p_ml_value = frame.GetMutableNodeInputOrOutputMLValue(0);
  Tensor* p_tensor = p_ml_value ? p_ml_value->GetMutable<Tensor>() : nullptr;
  EXPECT_TRUE(p_tensor);
  //Use reinterpret_cast to bypass a gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51213
  EXPECT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&p_tensor->Shape()), *reinterpret_cast<const std::vector<int64_t>*>(&shape));
  EXPECT_EQ(p_tensor->DataType(), DataTypeImpl::GetType<float>());

  //test share memory from tensor
  TensorShape shape2(std::vector<int64_t>{3, 2});
  OrtValue& mlvalue1 = *frame.GetMutableNodeInputOrOutputMLValue(start_index + 1);
  status = frame.AllocateMLValueTensorPreAllocateBuffer(mlvalue1,
                                                        start_index,
                                                        DataTypeImpl::GetType<float>(),
                                                        p_tensor->Location(),
                                                        shape2);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  const OrtValue* p_ml_value_const = frame.GetNodeInputOrOutputMLValue(1);
  auto tensor2 = p_ml_value_const ? &(p_ml_value_const->Get<Tensor>()) : nullptr;
  EXPECT_TRUE(tensor2);
  //Use reinterpret_cast to bypass a gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51213
  EXPECT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&tensor2->Shape()), *reinterpret_cast<const std::vector<int64_t>*>(&shape2));
  EXPECT_EQ(tensor2->template Data<float>(), p_tensor->template Data<float>());
}

TEST_F(ExecutionFrameTest, FeedInDataTest) {
  onnxruntime::Model model("test", false, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(),
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
  EXPECT_TRUE(kernel_registry_manager.RegisterKernels(execution_providers).IsOK());
  SessionState state{execution_providers, true, &tp_, nullptr};
  auto status = state.SetGraphAndCreateKernels(graph, kernel_registry_manager);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  const OrtValueNameIdxMap& mlvalue_name_idx_map = state.GetOrtValueNameIdxMap();
  int x_idx, y_idx;
  ASSERT_TRUE(mlvalue_name_idx_map.GetIdx("X", x_idx).IsOK());
  ASSERT_TRUE(mlvalue_name_idx_map.GetIdx("Y", y_idx).IsOK());

  vector<OrtValue> outputs;
  ExecutionFrame frame({x_idx}, {value}, {y_idx}, outputs, {}, state);

  OrtValue* p_ml_value = frame.GetMutableNodeInputOrOutputMLValue(0);
  Tensor* p_tensor_arg_0 = p_ml_value ? p_ml_value->GetMutable<Tensor>() : nullptr;
  EXPECT_TRUE(p_tensor_arg_0);
  //Use reinterpret_cast to bypass a gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51213
  EXPECT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&p_tensor_arg_0->Shape()), *reinterpret_cast<const std::vector<int64_t>*>(&shape));
  EXPECT_EQ(p_tensor_arg_0->DataType(), DataTypeImpl::GetType<float>());
  EXPECT_EQ(p_tensor_arg_0->MutableData<float>(), value.GetMutable<Tensor>()->MutableData<float>());
}

TEST_F(ExecutionFrameTest, MemPatternTest) {
  auto cpu_xp = CreateCPUExecutionProvider();
  auto xp_type = cpu_xp->Type();
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[onnxruntime::kOnnxDomain] = 7;
  onnxruntime::Model model("test", true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {}, DefaultLoggingManager().DefaultLogger());
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

  auto status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  KernelRegistryManager kernel_registry_manager;

  ExecutionProviders execution_providers;
  execution_providers.Add(xp_type, std::move(cpu_xp));
  kernel_registry_manager.RegisterKernels(execution_providers);
  //1. prepare input
  SessionState state{execution_providers, true, &tp_, nullptr};
  status = state.SetGraphAndCreateKernels(graph, kernel_registry_manager);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  const OrtValueNameIdxMap& mlvalue_name_idx_map(state.GetOrtValueNameIdxMap());

  int x1_idx, x2_idx, x3_idx;
  int t1_idx, t2_idx, t3_idx;
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

  std::unique_ptr<SequentialExecutionPlan> p_seq_exec_plan = onnxruntime::make_unique<SequentialExecutionPlan>();
  SequentialPlannerContext context(ExecutionMode::ORT_SEQUENTIAL);
  status = SequentialPlanner::CreatePlan(nullptr, GraphViewer(graph), {}, execution_providers, kernel_registry_manager,
                                         mlvalue_name_idx_map, context, p_seq_exec_plan);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  state.SetExecutionPlan(std::move(p_seq_exec_plan));

  vector<OrtValue> outputs;
  ExecutionFrame frame({x1_idx, x2_idx, x3_idx}, {v1, v2, v3}, {t3_idx}, outputs, {}, state);

  OrtValue& mlvalue3 = *frame.GetMutableNodeInputOrOutputMLValue(3);
  OrtValue& mlvalue4 = *frame.GetMutableNodeInputOrOutputMLValue(4);
  OrtValue& mlvalue5 = *frame.GetMutableNodeInputOrOutputMLValue(5);

  status = frame.AllocateMLValueTensorSelfOwnBuffer(mlvalue3, 3,
                                                    DataTypeImpl::GetType<float>(),
                                                    cpu_allocator->Info(),
                                                    TensorShape(std::vector<int64_t>{2, 2}));
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  status = frame.AllocateMLValueTensorSelfOwnBuffer(mlvalue4, 4,
                                                    DataTypeImpl::GetType<float>(),
                                                    cpu_allocator->Info(),
                                                    TensorShape(std::vector<int64_t>{2, 3}));
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  status = frame.AllocateMLValueTensorSelfOwnBuffer(mlvalue5, 5,
                                                    DataTypeImpl::GetType<float>(),
                                                    cpu_allocator->Info(),
                                                    TensorShape(std::vector<int64_t>{2, 3}));
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  MemoryPatternGroup pattern;
  status = frame.GeneratePatterns(&pattern);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  EXPECT_EQ(pattern.patterns.size(), pattern.locations.size());
  EXPECT_EQ(pattern.patterns.size(), 1u);
  auto p = pattern.GetPatterns(cpu_allocator->Info());
  EXPECT_EQ(p->PeakSize(), 2u * 64u);  // each allocation is 64-byte aligned
  EXPECT_EQ(p->GetBlock(3)->offset_, 0u);
  EXPECT_EQ(p->GetBlock(4)->offset_, 64u);
}

TEST(ExecutionFrameTestWithoutSessionState, BadModelInvalidDimParamUsage) {
  // load model with 2 Scan ops that both incorrectly use shapes of { 'None', 'None' } for their outputs.
  // as 'None' is not a special value it's treated as a variable name, leading to a runtime error when we
  // attempt to re-use the output from the first Scan node for the second. validate we detect this and error out.
  SessionOptions so;
  so.session_logid = "BadModelInvalidDimParamUsage";

  InferenceSession session_object{so, &DefaultLoggingManager()};
  Status st;
  ASSERT_TRUE((st = session_object.Load("testdata/invalid_dim_param_value_repetition.onnx")).IsOK()) << st;
  ASSERT_TRUE((st = session_object.Initialize()).IsOK()) << st;

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
  st = session_object.Run(run_options, feeds, output_names, &fetches);

  EXPECT_FALSE(st.IsOK()) << st;
  EXPECT_THAT(st.ErrorMessage(), testing::HasSubstr("Shape mismatch attempting to re-use buffer."));
}

}  // namespace test
}  // namespace onnxruntime
