// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/pipeline_transformer.h"
#include "core/graph/model.h"

#include "core/common/path_utils.h"
#include "test/test_environment.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace onnxruntime::training;


TEST(PipelinePartition, DropoutGraph2stages) {
  int nstages = 2;
  int pipeline_stage_id = 1;
  std::map<std::string, int> input_map = {
    {"A", 0}, {"B", 0}, {"C", 0}, {"D", 0}, {"E", 0}, {"F", 0},
    {"O1", 1}, {"O2", 1}, {"O3", 1}, {"O4", 1}, {"O5", 1}
  };

  const auto& log_manager = DefaultLoggingManager();
  const auto& default_logger = log_manager.DefaultLogger();
  const auto model_path = ORT_TSTR("testdata/transform/dropout.onnx");

  std::shared_ptr<Model> pModel;
  auto status = Model::Load(model_path, pModel, nullptr, default_logger);
  EXPECT_TRUE(status.IsOK()) << "Failed to load model. Error: "
                             << status.ErrorMessage();
  auto& graph = pModel->MainGraph();

  std::map<Node*, int> op_to_stage = {};
  status = GetDeviceAssignmentMap(graph, input_map, op_to_stage);

  EXPECT_TRUE(status.IsOK()) << "Failed to get stage map. Error: "
                             << status.ErrorMessage();
  status = ApplyPipelinePartitionToMainGraph(graph, op_to_stage,
                                             pipeline_stage_id, nstages);
  EXPECT_TRUE(status.IsOK()) << "Failed to apply partition. Error: "
                             << status.ErrorMessage();

  // Stage 1 should have 6 nodes: 6 Identities + 1 receive.
  EXPECT_EQ(graph.NumberOfNodes(), 6);
}

TEST(PipelinePartition, AttentionPastState2Stages) {
  int nstages = 3;
  int pipeline_stage_id = 1;
  TrainingSession::TrainingConfiguration::CutInfo cut0 = {
    TrainingSession::TrainingConfiguration::CutEdge("94")
  };
  TrainingSession::TrainingConfiguration::CutInfo cut1 = {
    TrainingSession::TrainingConfiguration::CutEdge("176"),
    TrainingSession::TrainingConfiguration::CutEdge("183"),
    TrainingSession::TrainingConfiguration::CutEdge("210")
  };
  std::vector<TrainingSession::TrainingConfiguration::CutInfo> cuts = {
    cut0, cut1
  };

  const auto& log_manager = DefaultLoggingManager();
  const auto& default_logger = log_manager.DefaultLogger();
  const auto model_path = ORT_TSTR("testdata/attention_past_state.onnx");

  std::shared_ptr<Model> pModel;
  auto status = Model::Load(model_path, pModel, nullptr, default_logger);
  EXPECT_TRUE(status.IsOK()) << "Failed to load model. Error: "
                             << status.ErrorMessage();
  auto& graph = pModel->MainGraph();

  std::map<Node*, int> op_to_stage = {};
  status = GetDeviceAssignmentMap(graph, cuts, op_to_stage);

  EXPECT_TRUE(status.IsOK()) << "Failed to get stage map. Error: "
                             << status.ErrorMessage();
  status = ApplyPipelinePartitionToMainGraph(graph, op_to_stage,
                                             pipeline_stage_id, nstages);
  EXPECT_TRUE(status.IsOK()) << "Failed to apply partition. Error: "
                             << status.ErrorMessage();

  // The following producers should be in this partition.
  EXPECT_TRUE(graph.GetMutableProducerNode("135") != nullptr);
  EXPECT_TRUE(graph.GetMutableProducerNode("122") != nullptr);
  EXPECT_TRUE(graph.GetMutableProducerNode("100") != nullptr);
  EXPECT_TRUE(graph.GetMutableProducerNode("169") != nullptr);
  EXPECT_TRUE(graph.GetMutableProducerNode("174") != nullptr);
  // The following producers should not be in this partition.
  EXPECT_FALSE(graph.GetMutableProducerNode("85") != nullptr);
  EXPECT_FALSE(graph.GetMutableProducerNode("221") != nullptr);
  EXPECT_FALSE(graph.GetMutableProducerNode("210") != nullptr);
  EXPECT_FALSE(graph.GetMutableProducerNode("214") != nullptr);
}

}
}