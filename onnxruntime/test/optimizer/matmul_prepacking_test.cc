// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/framework/test_utils.h"
#include "test/test_environment.h"

#include "core/graph/model.h"
#include "core/optimizer/matmul_prepacking.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/rule_based_graph_transformer.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>


#include "core/framework/execution_providers.h"
#include "core/framework/graph_partitioner.h"
#include "core/framework/kernel_registry_manager.h"


namespace onnxruntime {
namespace test {

namespace {
void ApplyPrepacking(Model& model, int max_num_threads = 2) {
  ASSERT_TRUE(model.MainGraph().Resolve().IsOK());
  ExecutionProviders providers;
  providers.Add(kCpuExecutionProvider, onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false)));
  KernelRegistryManager krm;
  krm.RegisterKernels(providers);
  GraphPartitioner partitioner(krm, providers);
  FuncManager func_mgr;
  // Assign CpuExecutionProvider to each node.
  ASSERT_TRUE(
    partitioner.Partition(model.MainGraph(), false, func_mgr).IsOK());

  GraphTransformerManager graph_transformation_mgr(1);
  auto rules_transformer = onnxruntime::make_unique<RuleBasedGraphTransformer>("MatMulPrepackingTransformer");
  ASSERT_TRUE(
    rules_transformer->Register(onnxruntime::make_unique<MatMulPrepacking>(max_num_threads)).IsOK());
  ASSERT_TRUE(
      graph_transformation_mgr.Register(std::move(rules_transformer), TransformerLevel::Level1).IsOK());
  ASSERT_TRUE(
      graph_transformation_mgr.ApplyTransformers(model.MainGraph(), TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());
}
}

TEST(MatMulPrepackingTests, SimpleTest) {
  auto model_uri = ORT_TSTR("testdata/transform/matmul_prepacking/simple_test.onnx");
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr,
        DefaultLoggingManager().DefaultLogger())
      .IsOK());
  ApplyPrepacking(*model, 2);

  Graph& graph = model->MainGraph();

  const auto& graph_inputs = graph.GetInputs();
  ASSERT_EQ(graph_inputs.size(), 1);
  ASSERT_EQ(graph_inputs[0]->Name(), "x");

  const auto& graph_outputs = graph.GetOutputs();
  ASSERT_EQ(graph_outputs.size(), 1);
  ASSERT_EQ(graph_outputs[0]->Name(), "Result");

  auto op_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_count["MatMul"], 0);
  ASSERT_EQ(op_count["MatMulPrepacked"], 1);
  ASSERT_EQ(op_count["PackForGemm"], 1);
}

TEST(MatMulPrepackingTests, GraphOutput) {
  auto model_uri = ORT_TSTR("testdata/transform/matmul_prepacking/graph_output.onnx");
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr,
        DefaultLoggingManager().DefaultLogger())
      .IsOK());
  ASSERT_TRUE(model->MainGraph().Resolve().IsOK());
  ApplyPrepacking(*model, 2);

  Graph& graph = model->MainGraph();

  const auto& graph_inputs = graph.GetInputs();
  ASSERT_EQ(graph_inputs.size(), 1);
  ASSERT_EQ(graph_inputs[0]->Name(), "x");

  const auto& graph_outputs = graph.GetOutputs();
  ASSERT_EQ(graph_outputs.size(), 1);
  ASSERT_EQ(graph_outputs[0]->Name(), "Result");

  auto op_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_count["MatMul"], 0);
  ASSERT_EQ(op_count["MatMulPrepacked"], 1);
  ASSERT_EQ(op_count["PackForGemm"], 1);
}

}
}
