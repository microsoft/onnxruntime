// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gradient_op_test_utils.h"
#include "core/session/inference_session.h"
#include "core/training/training_session.h"
#include "core/training/gradient_graph_builder.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {

// TODO: Refactor this so that we dont build gradient graph in every run
void GradientOpTester::Run(
    int output_index_to_use_as_loss,
    int data_index_of_output,
    ExpectResult expect_result,
    const std::string& expected_failure_string,
    const std::unordered_set<std::string>&,
    const RunOptions* run_options,
    std::vector<std::unique_ptr<IExecutionProvider>>*) {
  try {
#ifndef NDEBUG
    run_called_ = true;
#endif
    fetches_.clear();
    auto p_model = BuildGraph();
    auto& graph = p_model->MainGraph();

    Status status = Status::OK();
    if (expect_result == ExpectResult::kExpectFailure) {
      // capture possible exceptions from shape inference for invalid testcase
      try {
        status = graph.Resolve();
      } catch (const std::exception& ex) {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, ex.what());
      }
    } else {
      status = graph.Resolve();
    }

    if (!status.IsOK()) {
      if (expect_result == ExpectResult::kExpectFailure) {
        EXPECT_TRUE(!status.IsOK());
        EXPECT_THAT(status.ErrorMessage(), testing::HasSubstr(expected_failure_string));
      } else {
        LOGS_DEFAULT(ERROR) << "Resolve failed with status: " << status.ErrorMessage();
        EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
      }
    }

    // TODO: We will need finer control over both inputs and ouptuts
    // Not all inputs/outputs reqiures/have a gradient, e.g.index in gather
    VectorString weights_to_train;
    for (int i = 0; i < input_data_.size(); i++) {
      if (input_infos_[i].has_gradient) {
        weights_to_train.push_back(input_data_[i].def_.Name());
      }
    }

    VectorString dy_values;
    for (int i = 0; i < output_data_.size(); i++) {
      if (output_infos_[i].has_gradient) {
        dy_values.push_back(output_data_[i].def_.Name());
      }
    }

    training::GradientGraphBuilder grad_graph_builder(&graph,
                                                      dy_values,
                                                      weights_to_train,
                                                      "");
    status = grad_graph_builder.Build();
    if (!status.IsOK()) {
      LOGS_DEFAULT(ERROR) << "Gradient graph build failed: " << status.ErrorMessage();
      return;
    }

    // Hookup the inputs and outputs
    std::unordered_map<std::string, MLValue> feeds;
    std::vector<std::string> output_names;
    FillFeedsAndOutputNames(feeds, output_names, output_index_to_use_as_loss, data_index_of_output);

    // Add new inputs to the graph.
    std::vector<const NodeArg*> new_input_args = graph.GetInputs();  // Make a copy of existing input args.
    for (const auto& feed : feeds) {
      const auto* input_arg = graph.GetNodeArg(feed.first);
      EXPECT_TRUE(input_arg != nullptr);
      if (std::find(new_input_args.begin(), new_input_args.end(), input_arg) == new_input_args.end()) {
        new_input_args.emplace_back(input_arg);
      }
    }
    graph.SetInputOrder(new_input_args);  // By setting this, Graph::SetGraphInputsOutputs could infer the input as expected.
    graph.SetGraphResolveNeeded();
    graph.SetGraphProtoSyncNeeded();

    EXPECT_TRUE(graph.Resolve().IsOK());

    // Run the model
    SessionOptions so;
    so.session_logid = op_;
    so.session_log_verbosity_level = 1;

    onnxruntime::training::TrainingSession session_object{so};
    fetches_ = ExecuteModel<onnxruntime::training::TrainingSession>(*p_model, session_object, expect_result, expected_failure_string, run_options,
                                                                    feeds, output_names, onnxruntime::kCpuExecutionProvider);

  } catch (const std::exception& ex) {
    std::cerr << ex.what();
    // rethrow as some tests for error handling expect this
    throw;
  }
}

void GradientOpTester::FillFeedsAndOutputNames(std::unordered_map<std::string, MLValue>& feeds,
                                               std::vector<std::string>& output_names,
                                               int output_index_to_use_as_loss,
                                               int data_index_of_output) {
  OpTester::FillFeedsAndOutputNames(feeds, output_names);
  output_names.clear();  //ignore output names

  // add gradients as output instead
  for (int i = 0; i < input_data_.size(); ++i) {
    if (!input_infos_[i].has_gradient) {
      continue;
    }
    output_names.push_back(input_data_[i].def_.Name() + "_grad");
  }

  // Append gradient names and values to feeds
  std::vector<Data> gradient_data;
  for (int i = 0; i < output_data_.size(); i++) {
    if (!output_infos_[i].has_gradient) {
      continue;
    }
    auto shape = output_data_[i].data_.Get<Tensor>().Shape();
    std::vector<float> values(shape.Size(), 0.0);
    if (output_index_to_use_as_loss == i) {
      values[data_index_of_output] = 1.0;  //set only one value to one to construct jacobian matrix
    }
    AddData<float>(gradient_data, (output_data_[i].def_.Name() + "_grad").c_str(), shape.GetDims(), values.data(), values.size(), true);
  }

  for (auto i = 0; i < gradient_data.size(); ++i) {
    feeds[gradient_data[i].def_.Name()] = gradient_data[i].data_;
  }
}

}  // namespace test
}  // namespace onnxruntime
