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
  EXPECT_TRUE(status.IsOK()) << "Failed to load model.";
  auto& graph = pModel->MainGraph();

  std::map<Node*, int> op_to_stage = {};
  status = GetDeviceAssignmentMap(graph, input_map, op_to_stage);

  EXPECT_TRUE(status.IsOK()) << "Failed to get stage map.";
  status = ApplyPipelinePartitionToMainGraph(graph, op_to_stage,
                                             pipeline_stage_id, nstages);
  std::cout << "**" << status.ErrorMessage() << std::endl;
  EXPECT_TRUE(status.IsOK()) << "Failed to apply partition.";

  Model::Save(*(pModel.get()), "tes.onnx");
}

}
}