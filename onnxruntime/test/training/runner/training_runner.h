// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <utility>
#include <vector>
#include "core/framework/ml_value.h"
#include "core/training/training_session.h"

namespace onnxruntime {
namespace training {

struct DataPerRun {
  std::vector<std::string> names_;
  std::vector<MLValue> values_;
  size_t label_index_;  // which one is the label in the above vectors.
};

class TrainingRunner {
 public:
  typedef std::vector<std::unique_ptr<DataPerRun>> TrainingData;
  typedef std::vector<std::unique_ptr<DataPerRun>> TestData;

  struct Parameters {
    std::string model_path_;
    std::string model_with_loss_func_path_;
    std::string model_with_training_graph_path_;
    std::string model_trained_path_;
    std::string model_trained_with_loss_func_path_;
    LossFunctionInfo loss_func_info_;
    std::vector<std::string> weights_to_train_;

    size_t batch_size_;
    size_t num_of_epoch_;
    float learning_rate_;

    // When doing evaluation, some number of test samples will be selected to run evaluation.
    size_t num_of_samples_for_evaluation_;

    // error_function_ is called when evaluating the error for a single sample.
    std::function<void(const MLValue& /*predict*/, const MLValue& /*label*/)> error_function_;

    // post_evaluation_callback_ is called when a batch of evaluation is done.
    std::function<void(size_t /*num_of_test_sample_run*/)> post_evaluation_callback_;
  };

  TrainingRunner(TrainingData& trainingData, TestData& testData, const Parameters& params);

  common::Status Initialize();

  common::Status Run();

 private:
  Status TrainingLoop();
  Status EndTraining();
  Status Evaluate(InferenceSession& session, bool use_full_set = false);

  TrainingData& training_data_;
  TrainingData& test_data_;
  Parameters params_;
  TrainingSession session_;
};

}  // namespace training
}  // namespace onnxruntime
