// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <utility>
#include <vector>
#include "core/framework/ml_value.h"
#include "core/training/training_session.h"
#include "test/training/runner/training_util.h"
#include "core/graph/training/in_graph_training_optimizer.h"
#include "test/training/runner/data_loader.h"

namespace onnxruntime {
namespace training {

class TrainingRunner {
 public:
  struct AdamOptimizerParams {
    float alpha_;
    float beta_;
    float lambda_;
    float epsilon_;
  };

  struct Parameters {
    Parameters() {}

    std::string model_name;
    std::string model_path_;
    std::string model_with_loss_func_path_;          // To save the model after adding loss func.
    std::string model_with_training_graph_path_;     // To save the model after adding loss func and backward graph.
    std::string model_actual_running_graph_path_;    // To save the model with the actual running graph after transformations.
    std::string model_trained_path_;                 // To save the model after training.
    std::string model_trained_with_loss_func_path_;  // To save the model with loss func after training.

    PATH_STRING_TYPE train_data_dir;
    PATH_STRING_TYPE test_data_dir;
    PATH_STRING_TYPE log_dir;  // Path to write Tensorboard events to.

    bool is_perf_test;
    int num_of_perf_samples;

    LossFunctionInfo loss_func_info_;

    // The in-graph optimizer info.
    // It is the name of the optimizer OP.
    // If specified, every gradient output will be connected to a new optimizer node
    // who has the updated weights as new graph outputs.
    // For now all to-be-trained weights use the same optimizer type.
    std::string in_graph_optimizer_name_ = "SGDOptimizer";

    // For some model, loss function's input "prediction" is not the model output.
    // So model_prediction_name must be specified.
    std::string model_prediction_name_;

    // The weights to train, exclusive with weights_not_to_train_.
    std::unordered_set<std::string> weights_to_train_;

    // The weights not to train. If not empty, all the initializers not in the vector will be trained.
    // exclusive with weights_not_to_train_.
    std::unordered_set<std::string> weights_not_to_train_;

    TrainingSession::ImmutableWeights immutable_weigths_;

    MapStringToString input_name_map_;

    bool shuffle_data_;
    size_t batch_size_;
    size_t eval_batch_size;
    size_t num_of_epoch_;
    size_t evaluation_period;

    // Optimizer Parameter
    float learning_rate_;
    AdamOptimizerParams adam_opt_params_;

    // error_function_ is called when evaluating the error for a single sample.
    std::function<void(const std::vector<std::string>& feed_names,
                       const std::vector<OrtValue>& feeds,
                       const std::vector<std::string>& fetch_names,
                       const std::vector<OrtValue>& fetches)>
        error_function_;

    // post_evaluation_callback_ is called when a batch of evaluation is done.
    std::function<void(size_t /*eval_batch_size*/, size_t /*step*/)>
        post_evaluation_callback_;

    // Use CUDA providers or not.
    // TODO: support a list of providers.
    bool use_cuda_ = false;

    // Whether we collect execution profile trace during this run.
    bool use_profiler = false;

    int world_rank_ = 0;
    int world_size_ = 1;

    bool skip_evaluation_ = false;
    bool dump_fetches = false;

    VectorString fetch_names;
  };

  TrainingRunner(std::shared_ptr<DataSet> training_data,
                 std::shared_ptr<DataSet> test_data,
                 const Parameters& params);

  TrainingRunner(std::shared_ptr<DataLoader> training_data_loader,
                 std::shared_ptr<DataLoader> test_data_loader,
                 const Parameters& params)
      : TrainingRunner(training_data_loader->MutableDataSet(),
                       test_data_loader->MutableDataSet(),
                       params) {
    training_data_loader_ = training_data_loader;
    test_data_loader_ = test_data_loader;
  }

  common::Status Initialize();

  common::Status Run();

 private:
  Status TrainingLoop();
  Status EndTraining();
  Status Evaluate(InferenceSession& session);
  Status LoadAndEvaluate(const std::string& model_path);
  Status SetupOptimizerParams(const std::unordered_set<std::string>& weights_to_train,
                              std::unordered_map<std::string, OptimizerInfo>& infos);

  std::shared_ptr<DataLoader> training_data_loader_ = nullptr;
  std::shared_ptr<DataLoader> test_data_loader_ = nullptr;
  std::shared_ptr<DataSet> training_data_;
  std::shared_ptr<DataSet> test_data_;
  size_t step_;

  Parameters params_;
  TrainingSession session_;
};

}  // namespace training
}  // namespace onnxruntime
