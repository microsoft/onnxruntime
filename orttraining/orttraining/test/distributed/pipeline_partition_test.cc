// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/pipeline_transformer.h"
#include "core/graph/model.h"

#include "core/common/path_utils.h"
#include "test/test_environment.h"

#include "gtest/gtest.h"

#include "partition_utils.h"

namespace onnxruntime {
namespace test {

using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace onnxruntime::training;

using CutList = std::vector<TrainingSession::TrainingConfiguration::CutInfo>;

TEST(PipelinePartition, DropoutGraph2stages) {
  int num_stages = 2;
  int pipeline_stage_id = 1;
  std::map<std::string, int> input_map = {
      {"A", 0}, {"B", 0}, {"C", 0}, {"D", 0}, {"E", 0}, {"F", 0}, {"O1", 1}, {"O2", 1}, {"O3", 1}, {"O4", 1}, {"O5", 1}};

  const auto& log_manager = DefaultLoggingManager();
  const auto& default_logger = log_manager.DefaultLogger();
  const auto model_path = ORT_TSTR("testdata/transform/dropout.onnx");

  std::shared_ptr<Model> pModel;
  auto status = Model::Load(model_path, pModel, nullptr, default_logger);
  EXPECT_TRUE(status.IsOK()) << "Failed to load model. Error: "
                             << status.ErrorMessage();
  auto& graph = pModel->MainGraph();

  std::map<const Node*, int> op_to_stage = {};
  status = GetDeviceAssignmentMap(graph, input_map, op_to_stage, num_stages);
  EXPECT_TRUE(status.IsOK()) << "Failed to get stage map. Error: "
                             << status.ErrorMessage();
  // We are not running the tests with mpi ranks, so we create dummy rank_ids,
  // instead of querying the distributed run context.
  std::vector<int32_t> rank_ids(num_stages);
  for (int i = 0; i < num_stages; ++i) {
    rank_ids.at(i) = i;
  }
  status = ApplyPipelinePartitionToMainGraph(graph, op_to_stage,
                                             pipeline_stage_id, num_stages,
                                             rank_ids);
  EXPECT_TRUE(status.IsOK()) << "Failed to apply partition. Error: "
                             << status.ErrorMessage();

  // Stage 1 should have 6 nodes: 6 Identities + 1 receive.
  EXPECT_EQ(graph.NumberOfNodes(), 6);
}

void LoadAndPartitionWithCuts(const PathString& model_path,
                              int num_stages,
                              int pipeline_stage_id,
                              CutList& cuts,
                              bool use_stage_map,
                              std::shared_ptr<Model>& pModel) {
  const auto& log_manager = DefaultLoggingManager();
  const auto& default_logger = log_manager.DefaultLogger();
  auto status = Model::Load(model_path, pModel, nullptr, default_logger);
  EXPECT_TRUE(status.IsOK()) << "Failed to load model. Error: "
                             << status.ErrorMessage();
  auto& graph = pModel->MainGraph();

  if (use_stage_map) {
    // We are not running the tests with mpi ranks, so we create dummy rank_ids,
    // instead of querying the distributed run context.
    std::vector<int32_t> rank_ids(num_stages);
    for (int i = 0; i < num_stages; ++i) {
      rank_ids.at(i) = i;
    }
    std::map<const Node*, int> op_to_stage = {};
    status = GetDeviceAssignmentMap(graph, cuts, op_to_stage, num_stages);

    EXPECT_TRUE(status.IsOK()) << "Failed to get stage map. Error: "
                               << status.ErrorMessage();
    status = ApplyPipelinePartitionToMainGraph(graph, op_to_stage,
                                               pipeline_stage_id, num_stages,
                                               rank_ids);
    EXPECT_TRUE(status.IsOK()) << "Failed to apply partition. Error: "
                               << status.ErrorMessage();
  } else {
    status = CutBasedApplyPipelinePartitionToMainGraph(graph, cuts,
                                                       pipeline_stage_id, num_stages);
    EXPECT_TRUE(status.IsOK()) << "Failed to apply partition. Error: "
                               << status.ErrorMessage();
  }
}

TEST(PipelinePartition, AttentionPastState3Stages) {
  const auto filename = ORT_TSTR("testdata/attention_past_state.onnx");
  int num_stages = 3;
  int stage_id = 1;
  TrainingSession::TrainingConfiguration::CutInfo cut0 = {
      TrainingSession::TrainingConfiguration::CutEdge("94")};

  TrainingSession::TrainingConfiguration::CutInfo cut1 = {
      TrainingSession::TrainingConfiguration::CutEdge("176", {"214"}),
      TrainingSession::TrainingConfiguration::CutEdge("183", {"212"}),
      TrainingSession::TrainingConfiguration::CutEdge("210")};
  CutList cuts = {cut0, cut1};

  std::shared_ptr<Model> cb_model;
  LoadAndPartitionWithCuts(filename, num_stages, stage_id, cuts, true, cb_model);
  Graph& graph = cb_model->MainGraph();

  // The following producers should be in this partition.
  EXPECT_TRUE(graph.GetMutableProducerNode("135") != nullptr);
  EXPECT_TRUE(graph.GetMutableProducerNode("122") != nullptr);
  EXPECT_TRUE(graph.GetMutableProducerNode("100") != nullptr);
  EXPECT_TRUE(graph.GetMutableProducerNode("169") != nullptr);
  EXPECT_TRUE(graph.GetMutableProducerNode("210") != nullptr);
  EXPECT_TRUE(graph.GetMutableProducerNode("174") != nullptr);
  // The following producers should not be in this partition.
  EXPECT_FALSE(graph.GetMutableProducerNode("85") != nullptr);   // Stage 0
  EXPECT_FALSE(graph.GetMutableProducerNode("221") != nullptr);  // Stage 2
  EXPECT_FALSE(graph.GetMutableProducerNode("214") != nullptr);  // Stage 2
}

TEST(PipelinePartition, AttentionPastState2Stages) {
  const auto filename = ORT_TSTR("testdata/attention_past_state.onnx");
  int num_stages = 2;
  int stage_id = 1;
  TrainingSession::TrainingConfiguration::CutInfo cut0 = {
      TrainingSession::TrainingConfiguration::CutEdge("214")};

  CutList cuts = {cut0};

  std::shared_ptr<Model> cb_model;
  LoadAndPartitionWithCuts(filename, num_stages, stage_id, cuts, true, cb_model);
  Graph& graph = cb_model->MainGraph();

  std::vector<std::string> in_partition = {
      "215", "222", "228", "232", "226", "233", "227", "219", "216", "218", "221",
      "234", "237", "236", "235", "238", "output"};

  // +1 for the additional Send node.
  EXPECT_EQ(graph.NumberOfNodes(), in_partition.size() + 1);
  for (auto& tensor : in_partition) {
    EXPECT_TRUE(graph.GetMutableProducerNode(tensor) != nullptr);
  }
}

void compareGraphs(Graph& graph1, Graph& graph2) {
  GraphViewer gv1(graph1);
  const auto& g1_nodes = gv1.GetNodesInTopologicalOrder();

  GraphViewer gv2(graph2);
  const auto& g2_nodes = gv2.GetNodesInTopologicalOrder();

  EXPECT_EQ(g1_nodes.size(), g2_nodes.size());

  for (size_t i = 0, t = g1_nodes.size(); i < t; ++i) {
    const Node* n1 = gv1.GetNode(g1_nodes.at(i));
    const Node* n2 = gv2.GetNode(g2_nodes.at(i));
    EXPECT_EQ(n1->OpType(), n2->OpType());
    EXPECT_EQ(n1->InputDefs().size(), n2->InputDefs().size());
    EXPECT_EQ(n1->OutputDefs().size(), n2->OutputDefs().size());
  }
}

void comparePartitionTest(const PathString& filename, int num_stages,
                          int pipeline_stage_id, CutList& cuts) {
  std::shared_ptr<Model> sm_model;
  LoadAndPartitionWithCuts(filename, num_stages, pipeline_stage_id, cuts, true, sm_model);
  Graph& sm_graph = sm_model->MainGraph();

  std::shared_ptr<Model> cb_model;
  LoadAndPartitionWithCuts(filename, num_stages, pipeline_stage_id, cuts, false, cb_model);
  Graph& cb_graph = cb_model->MainGraph();

  compareGraphs(sm_graph, cb_graph);
}

TEST(ComparePartitions, AttentionPastState3Stages) {
  const auto filename = ORT_TSTR("testdata/attention_past_state.onnx");
  int num_stages = 3;
  TrainingSession::TrainingConfiguration::CutInfo cut0 = {
      TrainingSession::TrainingConfiguration::CutEdge("94")};
  TrainingSession::TrainingConfiguration::CutInfo cut1 = {
      TrainingSession::TrainingConfiguration::CutEdge("214")};
  CutList cuts = {cut0, cut1};
  for (int stage = 0; stage < num_stages; ++stage) {
    comparePartitionTest(filename, num_stages, stage, cuts);
  }
}

TEST(ComparePartitions, AttentionPastState3StagesMultiEdgeCut) {
  const auto filename = ORT_TSTR("testdata/attention_past_state.onnx");
  int num_stages = 3;
  TrainingSession::TrainingConfiguration::CutInfo cut0 = {
      TrainingSession::TrainingConfiguration::CutEdge("94")};
  TrainingSession::TrainingConfiguration::CutInfo cut1 = {
      TrainingSession::TrainingConfiguration::CutEdge("176", {"214"}),
      TrainingSession::TrainingConfiguration::CutEdge("183", {"212"}),
      TrainingSession::TrainingConfiguration::CutEdge("210")};
  CutList cuts = {cut0, cut1};
  for (int stage = 0; stage < num_stages; ++stage) {
    comparePartitionTest(filename, num_stages, stage, cuts);
  }
}

TEST(ComparePartitions, BertToy) {
  const auto filename = ORT_TSTR("testdata/bert_toy_optimized.onnx");
  int num_stages = 3;
  TrainingSession::TrainingConfiguration::CutInfo cut0 = {
      TrainingSession::TrainingConfiguration::CutEdge("326"),
      TrainingSession::TrainingConfiguration::CutEdge("103", {"413", "529"})};
  TrainingSession::TrainingConfiguration::CutInfo cut1 = {
      TrainingSession::TrainingConfiguration::CutEdge("558"),
      TrainingSession::TrainingConfiguration::CutEdge("103", {"645"})};
  CutList cuts = {cut0, cut1};
  for (int stage = 0; stage < num_stages; ++stage) {
    comparePartitionTest(filename, num_stages, stage, cuts);
  }
}

}  // namespace test
}  // namespace onnxruntime
