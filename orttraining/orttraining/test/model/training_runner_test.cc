// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/models/runner/training_runner.h"

#include "gtest/gtest.h"

#include "core/common/path_string.h"
#include "core/framework/path_lib.h"
#include "orttraining/models/runner/data_loader.h"
#include "orttraining/models/runner/training_util.h"

namespace onnxruntime {
namespace training {
namespace test {

const PathString k_original_model_path =
    ConcatPathComponent<PathChar>(ORT_TSTR("testdata"), ORT_TSTR("test_training_model.onnx"));
const PathString k_backward_model_path =
    ConcatPathComponent<PathChar>(ORT_TSTR("testdata"), ORT_TSTR("temp_backward_model.onnx"));

const PathString k_output_directory = ORT_TSTR("training_runner_test_output");

TEST(TrainingRunnerTest, Basic) {
  TrainingRunner::Parameters params{};
  params.model_path = k_original_model_path;
  params.model_with_training_graph_path = k_backward_model_path;
  params.output_dir = k_output_directory;
  params.is_perf_test = false;
  params.batch_size = 1;
  params.eval_batch_size = 1;
  params.num_train_steps = 1;
  params.display_loss_steps = 10;
  params.fetch_names = {"predictions"};
  params.loss_func_info = LossFunctionInfo(OpDef("MeanSquaredError"), "loss", {"predictions", "labels"});

  TrainingRunner runner{params};

  auto status = runner.Initialize();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  std::vector<std::string> tensor_names{
      "X", "labels"};
  std::vector<TensorShape> tensor_shapes{
      {1, 784}, {1, 10}};
  std::vector<ONNX_NAMESPACE::TensorProto_DataType> tensor_types{
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT, ONNX_NAMESPACE::TensorProto_DataType_FLOAT};

  auto data_set = std::make_shared<RandomDataSet>(1, tensor_names, tensor_shapes, tensor_types);
  auto data_loader = std::make_shared<SingleDataLoader>(data_set, tensor_names);

  status = runner.Run(data_loader.get(), data_loader.get());
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  status = runner.EndTraining(data_loader.get());
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
}

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
