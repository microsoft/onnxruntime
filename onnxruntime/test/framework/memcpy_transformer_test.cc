// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iterator>

#include "core/framework/execution_providers.h"
#include "core/optimizer/transformer_memcpy.h"
#include "core/graph/model.h"
#include "gtest/gtest.h"
#include "test_utils.h"
#include "test/test_environment.h"
#include "asserts.h"

using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace test {

typedef std::vector<onnxruntime::NodeArg*> ArgMap;

void ExpectSame(const onnxruntime::Node& source, const onnxruntime::Node& target, int argnum) {
  // Check that target's argnum-th input comes from the source node (without copy):
  auto* source_output = source.OutputDefs()[0];
  auto* target_input = target.InputDefs()[argnum];
  EXPECT_EQ(source_output, target_input);
}

void ExpectCopy(const onnxruntime::Node& source, const std::string& copy_op, const onnxruntime::Node& target,
                int argnum) {
  // Check that source's output is consumed by a copy_op;
  for (auto it = source.OutputNodesBegin(); it != source.OutputNodesEnd(); ++it) {
    auto& copy_node = *it;
    if (copy_node.OpType() == copy_op) {
      // Check that target's argnum-th input comes from the copy node:
      auto* copy_output = copy_node.OutputDefs()[0];
      auto* target_input = target.InputDefs()[argnum];
      EXPECT_EQ(copy_output, target_input);
      return;
    }
  }
  EXPECT_TRUE(false) << "Copy node expected but not found";
}

void ExpectCopy(const onnxruntime::NodeArg& source_arg, const std::string copy_op,
                const onnxruntime::Node& target, int argnum) {
  auto* target_input = target.InputDefs()[argnum];
  for (auto it = target.InputNodesBegin(); it != target.InputNodesEnd(); ++it) {
    auto& copy_node = *it;
    // Check if target's argnum-th input comes from this node:
    auto* copy_output = copy_node.OutputDefs()[0];
    if (copy_output == target_input) {
      EXPECT_EQ(copy_node.OpType(), copy_op);
      auto* copy_input = copy_node.InputDefs()[0];
      EXPECT_EQ(copy_input, &source_arg);
      return;
    }
  }
  EXPECT_TRUE(false) << "Copy node expected but not found";
}

void ExpectCopy(const onnxruntime::Node& source, const std::string copy_op,
                const onnxruntime::NodeArg& target_arg) {
  // Check that source's output is consumed by a copy_op;
  for (auto it = source.OutputNodesBegin(); it != source.OutputNodesEnd(); ++it) {
    auto& copy_node = *it;
    if (copy_node.OpType() == copy_op) {
      auto* copy_output = copy_node.OutputDefs()[0];
      EXPECT_EQ(copy_output, &target_arg);
      return;
    }
  }
  EXPECT_TRUE(false) << "Copy node expected but not found";
}
#ifdef USE_CUDA

TEST(TransformerTest, MemcpyTransformerTest) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 7;
  auto model = std::make_shared<onnxruntime::Model>("test", false, ModelMetaData(), PathString(),
                                                    IOnnxRuntimeOpSchemaRegistryList(),
                                                    domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
                                                    DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();

  TypeProto tensor_float_type;
  tensor_float_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  onnxruntime::NodeArg i1_def("I1", &tensor_float_type),
      i2_def("I2", &tensor_float_type),
      i3_def("I3", &tensor_float_type),
      o1_def("O1", &tensor_float_type),
      o2_def("O2", &tensor_float_type),
      o3_def("O3", &tensor_float_type),
      o4_def("O4", &tensor_float_type);

  auto& node1 = graph.AddNode("node1", "MatMul", "cpu operator1", ArgMap{&i1_def, &i2_def}, ArgMap{&o1_def});
  node1.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  auto& node2 = graph.AddNode("node2", "MatMul", "gpu operator1", ArgMap{&o1_def, &i3_def}, ArgMap{&o2_def});
  node2.SetExecutionProviderType(onnxruntime::kCudaExecutionProvider);
  auto& node3 = graph.AddNode("node3", "Clip", "cpu operator2", ArgMap{&o2_def}, ArgMap{&o3_def});
  node3.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  auto& node4 = graph.AddNode("node4", "MatMul", "gpu operator2", ArgMap{&o2_def, &o2_def}, ArgMap{&o4_def});
  node4.SetExecutionProviderType(onnxruntime::kCudaExecutionProvider);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  KernelRegistryManager kernel_registry_manager;
  ExecutionProviders execution_providers;
  execution_providers.Add(onnxruntime::kCudaExecutionProvider,
                          std::make_unique<CUDAExecutionProvider>(CUDAExecutionProviderInfo()));
  execution_providers.Add(onnxruntime::kCpuExecutionProvider,
                          std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo()));
  KernelRegistryManager test_registry_manager;
  ASSERT_STATUS_OK(test_registry_manager.RegisterKernels(execution_providers));

  MemcpyTransformer transformer({onnxruntime::kCudaExecutionProvider}, test_registry_manager);

  bool modified = false;
  status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(modified);

  // Expect: copy of O1 from cpu to gpu
  ExpectCopy(node1, "MemcpyFromHost", node2, 0);

  // Expect: copy O2 from gpu to cpu
  ExpectCopy(node2, "MemcpyToHost", node3, 0);
  ExpectSame(node2, node4, 0);
  ExpectSame(node2, node4, 1);
}

TEST(TransformerTest, MemcpyTransformerTestCudaFirst) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 7;
  auto model = std::make_shared<onnxruntime::Model>("test", false, ModelMetaData(), PathString(),
                                                    IOnnxRuntimeOpSchemaRegistryList(),
                                                    domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
                                                    DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();

  TypeProto tensor_float_type;
  tensor_float_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  onnxruntime::NodeArg i1_def("I1", &tensor_float_type),
      i2_def("I2", &tensor_float_type),
      i3_def("I3", &tensor_float_type),
      o1_def("O1", &tensor_float_type),
      o2_def("O2", &tensor_float_type),
      o3_def("O3", &tensor_float_type),
      o4_def("O4", &tensor_float_type);

  auto& node1 = graph.AddNode("node1", "MatMul", "gpu operator1", ArgMap{&i1_def, &i2_def}, ArgMap{&o1_def});
  node1.SetExecutionProviderType(onnxruntime::kCudaExecutionProvider);
  auto& node2 = graph.AddNode("node2", "MatMul", "cpu operator1", ArgMap{&o1_def, &i3_def}, ArgMap{&o2_def});
  node2.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  auto& node3 = graph.AddNode("node3", "Abs", "gpu operator2", ArgMap{&o2_def}, ArgMap{&o3_def});
  node3.SetExecutionProviderType(onnxruntime::kCudaExecutionProvider);
  auto& node4 = graph.AddNode("node4", "MatMul", "cpu operator2", ArgMap{&o2_def, &o2_def}, ArgMap{&o4_def});
  node4.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  KernelRegistryManager kernel_registry_manager;
  ExecutionProviders execution_providers;
  execution_providers.Add(onnxruntime::kCudaExecutionProvider,
                          std::make_unique<CUDAExecutionProvider>(CUDAExecutionProviderInfo()));
  execution_providers.Add(onnxruntime::kCpuExecutionProvider,
                          std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo()));
  KernelRegistryManager test_registry_manager;
  ASSERT_STATUS_OK(test_registry_manager.RegisterKernels(execution_providers));

  MemcpyTransformer transformer({onnxruntime::kCudaExecutionProvider}, test_registry_manager);

  bool modified = false;
  status = transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger());
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(modified);

  // Expect: copy of O1 from gpu to cpu
  ExpectCopy(node1, "MemcpyToHost", node2, 0);

  // Expect: copy O2 from cpu to gpu
  ExpectCopy(node2, "MemcpyFromHost", node3, 0);
  ExpectSame(node2, node4, 0);
  ExpectSame(node2, node4, 1);
}
TEST(TransformerTest, TestCopyNodeInsertionInitializerInSubgraph) {
  // In this test, we are going to create a subgraph consuming an implicit input
  // which is an initializer in the outer scope, and this implicit input to the subgraph
  // is consumed by nodes on multiple devices
  TensorProto value_tensor;
  value_tensor.add_dims(1);
  value_tensor.add_float_data(1.f);
  value_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  TypeProto tensor_float_type;
  tensor_float_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);

  TypeProto tensor_bool_type;
  tensor_bool_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);

  onnxruntime::NodeArg i1_def("I1", &tensor_bool_type),
      o1_def("O1", &tensor_float_type),
      o2_def("O2", &tensor_float_type);

  // main graph
  // this will only contain one 'If' node
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 7;
  auto model = std::make_shared<onnxruntime::Model>("test",
                                                    false,
                                                    ModelMetaData(),
                                                    PathString(),
                                                    IOnnxRuntimeOpSchemaRegistryList(),
                                                    domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
                                                    DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();

  TensorProto parent_constant(value_tensor);
  parent_constant.set_name("parent_constant");
  graph.AddInitializedTensor(parent_constant);

  // subgraph
  // this will contain 2 'Add' nodes - one on CPU and one of GPU
  // one of the inputs to the 'Add' nodes is an implicit input to the subgraph
  // which is an initializer in the main graph
  std::unordered_map<std::string, int> subgraph_domain_to_version;
  subgraph_domain_to_version[kOnnxDomain] = 7;
  auto sub_model = std::make_shared<onnxruntime::Model>("test_subgraph",
                                                        false,
                                                        ModelMetaData(),
                                                        PathString(),
                                                        IOnnxRuntimeOpSchemaRegistryList(),
                                                        subgraph_domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
                                                        DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& subgraph = sub_model->MainGraph();

  TensorProto local_constant(value_tensor);
  local_constant.set_name("local_constant");
  subgraph.AddInitializedTensor(local_constant);

  subgraph.AddOuterScopeNodeArg("parent_constant");
  subgraph.AddNode("node1", "Add", "operator1",
                   ArgMap{&subgraph.GetOrCreateNodeArg("local_constant", &tensor_float_type),
                          &graph.GetOrCreateNodeArg("parent_constant", &tensor_float_type)},
                   ArgMap{&o1_def});

  subgraph.AddNode("node2", "Add", "operator2",
                   ArgMap{&subgraph.GetOrCreateNodeArg("local_constant", &tensor_float_type),
                          &graph.GetOrCreateNodeArg("parent_constant", &tensor_float_type)},
                   ArgMap{&o2_def});

  ASSERT_STATUS_OK(subgraph.Resolve());

  // main graph continued
  // create the 'If' node
  auto& if_node = graph.AddNode("node3", "If", "cpu operator2", ArgMap{&i1_def}, ArgMap{&o1_def, &o2_def});
  if_node.AddAttribute("then_branch", subgraph.ToGraphProto());
  if_node.AddAttribute("else_branch", subgraph.ToGraphProto());

  onnxruntime::Graph* subgraph_1 = if_node.GetMutableGraphAttribute("then_branch");
  for (auto& node : subgraph_1->Nodes()) {
    if (node.Name() == "node2") {
      // only this node is on GPU
      node.SetExecutionProviderType(onnxruntime::kCudaExecutionProvider);
    } else {
      node.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
    }
  }

  onnxruntime::Graph* subgraph_2 = if_node.GetMutableGraphAttribute("else_branch");
  for (auto& node : subgraph_2->Nodes()) {
    node.SetExecutionProviderType(onnxruntime::kCpuExecutionProvider);
  }

  ASSERT_STATUS_OK(graph.Resolve());

  KernelRegistryManager kernel_registry_manager;
  ExecutionProviders execution_providers;
  execution_providers.Add(onnxruntime::kCudaExecutionProvider,
                          std::make_unique<CUDAExecutionProvider>(CUDAExecutionProviderInfo()));
  execution_providers.Add(onnxruntime::kCpuExecutionProvider,
                          std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo()));
  KernelRegistryManager test_registry_manager;
  ASSERT_STATUS_OK(test_registry_manager.RegisterKernels(execution_providers));

  MemcpyTransformer transformer({onnxruntime::kCudaExecutionProvider}, test_registry_manager);

  bool modified = false;
  ASSERT_STATUS_OK(transformer.Apply(graph, modified, DefaultLoggingManager().DefaultLogger()));
  EXPECT_TRUE(modified);
}

#endif

}  // namespace test
}  // namespace onnxruntime
