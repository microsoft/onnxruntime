// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/execution_frame.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/model.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "test_utils.h"
#include "gtest/gtest.h"

using namespace ONNX_NAMESPACE;
using namespace std;

namespace onnxruntime {
namespace test {
typedef std::vector<onnxruntime::NodeArg*> ArgMap;

std::shared_ptr<onnxruntime::Model> DummyGraphWithClip() {
  auto model = std::make_shared<onnxruntime::Model>("test");
  onnxruntime::Graph& graph = model->MainGraph();
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  onnxruntime::NodeArg input_def("X", &tensor_float), output_def("Y", &tensor_float);

  graph.AddNode("node1", "Clip", "clip operator", ArgMap{&input_def}, ArgMap{&output_def});
  return model;
}

std::unique_ptr<IExecutionProvider> CreateCPUExecutionProvider() {
  CPUExecutionProviderInfo info;
  return std::make_unique<CPUExecutionProvider>(info);
}

TEST(ExecutionFrameTest, TensorAllocationTest) {
  onnxruntime::Model model("test");
  onnxruntime::Graph& graph = model.MainGraph();
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  onnxruntime::NodeArg input_def("X", &tensor_float), output_def("Y", &tensor_float);

  graph.AddNode("node1", "Clip", "Clip operator", ArgMap{&input_def}, ArgMap{&output_def});
  onnxruntime::Node* node = graph.GetNode(graph.NumberOfNodes() - 1);

  Status status = graph.Resolve();
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  auto cpu_xp = CreateCPUExecutionProvider();
  auto xp_typ = cpu_xp->Type();
  ExecutionProviders execution_providers;
  execution_providers.Add(xp_typ, std::move(cpu_xp));
  KernelRegistryManager kernel_registry_manager;
  kernel_registry_manager.RegisterKernels(execution_providers);

  SessionState state{execution_providers};
  state.SetGraphViewer(std::make_unique<GraphViewer>(graph));

  MLValueNameIdxMap& mlvalue_name_idx_map{state.GetMLValueNameIdxMap()};
  mlvalue_name_idx_map.Add("X");
  mlvalue_name_idx_map.Add("Y");

  node->SetExecutionProviderType(xp_typ);

  std::unique_ptr<SequentialExecutionPlan> p_seq_exec_plan;
  // TODO below line is for testing only. In production use SequentialPlanner::CreatePlan()
  status = SequentialPlanner::CreatePlan(GraphViewer(graph), {}, execution_providers, kernel_registry_manager, mlvalue_name_idx_map,
                                         p_seq_exec_plan);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  state.SetExecutionPlan(std::move(p_seq_exec_plan));

  state.CalculateNodeIndexInfo();

  vector<MLValue> outputs;
  ExecutionFrame frame(std::unordered_map<std::string, MLValue>{}, std::vector<std::string>{}, outputs, {}, state);

  int start_index = frame.GetNodeOffset(node->Index());
  EXPECT_EQ(start_index, 0);

  TensorShape shape(std::vector<int64_t>{2, 3});
  status = frame.AllocateMLValueTensorSelfOwnBuffer(start_index, DataTypeImpl::GetType<float>(),
                                                    execution_providers.Get(xp_typ)->GetAllocator(0, OrtMemTypeDefault)->Info(), shape);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  MLValue* p_ml_value = frame.GetMutableNodeInputOrOutputMLValue(0);
  Tensor* p_tensor = p_ml_value ? p_ml_value->GetMutable<Tensor>() : nullptr;
  EXPECT_TRUE(p_tensor);
  EXPECT_EQ(p_tensor->Shape(), shape);
  EXPECT_EQ(p_tensor->DataType(), DataTypeImpl::GetType<float>());

  //test share memory from tensor
  TensorShape shape2(std::vector<int64_t>{3, 2});
  status = frame.AllocateMLValueTensorPreAllocateBuffer(start_index + 1,
                                                        start_index,
                                                        DataTypeImpl::GetType<float>(),
                                                        p_tensor->Location(),
                                                        shape2);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  const MLValue* p_ml_value_const = frame.GetNodeInputOrOutputMLValue(1);
  auto tensor2 = p_ml_value_const ? &(p_ml_value_const->Get<Tensor>()) : nullptr;
  EXPECT_TRUE(tensor2);
  EXPECT_EQ(tensor2->Shape(), shape2);
  EXPECT_EQ(tensor2->template Data<float>(), p_tensor->template Data<float>());
}

TEST(ExecutionFrameTest, FeedInDataTest) {
  onnxruntime::Model model("test");
  onnxruntime::Graph& graph = model.MainGraph();
  TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  onnxruntime::NodeArg input_def("X", &tensor_float), output_def("Y", &tensor_float);

  graph.AddNode("node1", "Clip", "Clip operator", ArgMap{&input_def}, ArgMap{&output_def});
  graph.Resolve();
  auto cpu_allocator = TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault);
  auto element_type = DataTypeImpl::GetType<float>();
  TensorShape shape({3, 2});
  void* buffer = cpu_allocator->Alloc(element_type->Size() * shape.Size());
  //create fake ml value with owned buffer.
  std::unique_ptr<Tensor> p_tensor = std::make_unique<Tensor>(element_type,
                                                              shape,
                                                              buffer,
                                                              cpu_allocator->Info(),
                                                              cpu_allocator);
  MLValue value;
  value.Init(p_tensor.release(),
             DataTypeImpl::GetType<Tensor>(),
             DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  auto cpu_xp = CreateCPUExecutionProvider();
  auto xp_typ = cpu_xp->Type();

  KernelRegistryManager kernel_registry_manager;
  ExecutionProviders execution_providers;
  execution_providers.Add(xp_typ, std::move(cpu_xp));
  kernel_registry_manager.RegisterKernels(execution_providers);

  SessionState state{execution_providers};
  state.SetGraphViewer(std::make_unique<GraphViewer>(graph));

  MLValueNameIdxMap& mlvalue_name_idx_map{state.GetMLValueNameIdxMap()};
  mlvalue_name_idx_map.Add("X");
  mlvalue_name_idx_map.Add("Y");

  state.CalculateNodeIndexInfo();

  vector<MLValue> outputs;
  ExecutionFrame frame(std::unordered_map<std::string, MLValue>{{"X", value}},
                       std::vector<std::string>{}, outputs, {}, state);

  MLValue* p_ml_value = frame.GetMutableNodeInputOrOutputMLValue(0);
  Tensor* p_tensor_arg_0 = p_ml_value ? p_ml_value->GetMutable<Tensor>() : nullptr;
  EXPECT_TRUE(p_tensor_arg_0);
  EXPECT_EQ(p_tensor_arg_0->Shape(), shape);
  EXPECT_EQ(p_tensor_arg_0->DataType(), DataTypeImpl::GetType<float>());
  EXPECT_EQ(p_tensor_arg_0->template MutableData<float>(), buffer);
}

TEST(ExecutionFrameTest, MemPatternTest) {
  auto cpu_xp = CreateCPUExecutionProvider();
  auto xp_type = cpu_xp->Type();
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[onnxruntime::kOnnxDomain] = 7;
  onnxruntime::Model model("test", true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), domain_to_version);
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
  kernel_registry_manager.RegisterKernelRegistry(cpu_xp->GetKernelRegistry());

  ExecutionProviders execution_providers;
  execution_providers.Add(xp_type, std::move(cpu_xp));

  //1. prepare input
  SessionState state{execution_providers};
  state.SetGraphViewer(std::make_unique<GraphViewer>(graph));

  MLValueNameIdxMap& mlvalue_name_idx_map{state.GetMLValueNameIdxMap()};

  mlvalue_name_idx_map.Add("X1");
  mlvalue_name_idx_map.Add("X2");
  mlvalue_name_idx_map.Add("X3");
  mlvalue_name_idx_map.Add("T1");
  mlvalue_name_idx_map.Add("T2");
  mlvalue_name_idx_map.Add("T3");

  auto cpu_allocator = execution_providers.Get(xp_type)->GetAllocator(0, OrtMemTypeDefault);

  MLValue v1, v2, v3;
  CreateMLValue<float>(cpu_allocator,
                       std::vector<int64_t>{1, 2},
                       std::vector<float>{1.0f, 1.0f}, &v1);
  CreateMLValue<float>(cpu_allocator,
                       std::vector<int64_t>{2, 2},
                       std::vector<float>(4, 1.0f), &v2);
  CreateMLValue<float>(cpu_allocator,
                       std::vector<int64_t>{2, 3},
                       std::vector<float>(6, 1.0f), &v3);

  std::unique_ptr<SequentialExecutionPlan> p_seq_exec_plan = std::make_unique<SequentialExecutionPlan>();
  status = SequentialPlanner::CreatePlan(GraphViewer(graph), {}, execution_providers, kernel_registry_manager, mlvalue_name_idx_map,
                                         p_seq_exec_plan);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  state.SetExecutionPlan(std::move(p_seq_exec_plan));

  state.CalculateNodeIndexInfo();

  vector<MLValue> outputs;
  ExecutionFrame frame(std::unordered_map<std::string, MLValue>{{"X1", v1}, {"X2", v2}, {"X3", v3}},
                       std::vector<std::string>{"T3"}, outputs, {}, state);

  status = frame.AllocateMLValueTensorSelfOwnBuffer(3,
                                                    DataTypeImpl::GetType<float>(),
                                                    cpu_allocator->Info(),
                                                    TensorShape(std::vector<int64_t>{2, 2}));
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  status = frame.AllocateMLValueTensorSelfOwnBuffer(4,
                                                    DataTypeImpl::GetType<float>(),
                                                    cpu_allocator->Info(),
                                                    TensorShape(std::vector<int64_t>{2, 3}));
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  status = frame.AllocateMLValueTensorSelfOwnBuffer(5,
                                                    DataTypeImpl::GetType<float>(),
                                                    cpu_allocator->Info(),
                                                    TensorShape(std::vector<int64_t>{2, 3}));
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  MemoryPatternGroup pattern;
  status = frame.GeneratePatterns(&pattern);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  EXPECT_EQ(pattern.patterns.size(), pattern.locations.size());
  EXPECT_EQ(pattern.patterns.size(), 1);
  auto p = pattern.GetPatterns(cpu_allocator->Info());
  EXPECT_EQ(p->PeakSize(), 2 * 64);  // each allocation is 64-byte aligned
  EXPECT_EQ(p->GetBlock(3)->offset_, 0);
  EXPECT_EQ(p->GetBlock(4)->offset_, 64);
}
}  // namespace test
}  // namespace onnxruntime
