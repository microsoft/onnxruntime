// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <memory>
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/environment.h"
#include "core/training/training_optimizer.h"
#include "core/training/weight_updater.h"
#include "test/training/runner/training_runner.h"
#include "test/training/runner/training_util.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_execution_provider.h"
#endif

using namespace std;

namespace onnxruntime {
namespace training {

const static string SGD_OP_NAME = "SGDOptimizer";
const static string SGD_LEARNING_RATE_STRING = "learning_rate";

TrainingRunner::TrainingRunner(DataSet& trainingData, DataSet& testData, const Parameters& params)
    : training_data_(trainingData), test_data_(testData), params_(params), session_(SessionOptions()) {
  ORT_ENFORCE(!params_.model_path_.empty());
  ORT_ENFORCE((!params_.weights_to_train_.empty() && params_.weights_not_to_train_.empty()) ||
              (params_.weights_to_train_.empty() && !params_.weights_not_to_train_.empty()));
  ORT_ENFORCE(!params_.model_trained_path_.empty() || !params_.model_trained_with_loss_func_path_.empty());
  ORT_ENFORCE(!params_.model_prediction_name_.empty());
  ORT_ENFORCE(params_.in_graph_optimizer_name_.empty() ||
              params_.in_graph_optimizer_name_ == SGD_OP_NAME);
}

Status TrainingRunner::Initialize() {
  ORT_RETURN_IF_ERROR(session_.Load(params_.model_path_));

  // Add loss func
  ORT_RETURN_IF_ERROR(session_.AddLossFuncion(params_.loss_func_info_));
  if (!params_.model_with_loss_func_path_.empty()) {
    ORT_RETURN_IF_ERROR(session_.Save(params_.model_with_loss_func_path_,
                                      TrainingSession::SaveOption::NO_RELOAD));
  }

  // Get the weights-to-train list if user specify it.
  // Otherweise, generate the list by removing not-to-train ones from all initializers.
  auto weights_to_train = params_.weights_to_train_;
  if (weights_to_train.empty()) {
    auto all_weights = session_.GetModelInitializers();
    for (const auto& not_to_train : params_.weights_not_to_train_) {
      all_weights.erase(not_to_train);
    }
    weights_to_train.reserve(all_weights.size());
    for (const auto& w : all_weights) {
      weights_to_train.push_back(w);
    }
  }

  // If in-graph optimizer is used, prepare the weight<->optimizer mapping.
  // Here all weights use the same SGDOptimizer.
  bool use_in_graph_optimizer = !params_.in_graph_optimizer_name_.empty();
  vector<in_graph_optimizer::OptimizerInfo> opt_info;
  opt_info.reserve(weights_to_train.size());
  if (use_in_graph_optimizer) {
    for (int i = 0; i < weights_to_train.size(); ++i) {
      opt_info.push_back({params_.in_graph_optimizer_name_, {SGD_LEARNING_RATE_STRING}});
    }
  }

  // Add gradient graph
  ORT_RETURN_IF_ERROR(session_.BuildGradientGraph(weights_to_train, params_.loss_func_info_.loss_name_, opt_info));
  if (!params_.model_with_training_graph_path_.empty()) {
    ORT_RETURN_IF_ERROR(session_.Save(params_.model_with_training_graph_path_,
                                      TrainingSession::SaveOption::NO_RELOAD));
  }

#ifdef USE_CUDA
  if (params_.use_cuda_) {
    CUDAExecutionProviderInfo xp_info;
    ORT_RETURN_IF_ERROR(session_.RegisterExecutionProvider(std::make_unique<CUDAExecutionProvider>(xp_info)));
  }
#endif

  return session_.Initialize();
}

Status TrainingRunner::Run() {
  if (!params_.model_actual_running_graph_path_.empty()) {
    ORT_RETURN_IF_ERROR(session_.Save(params_.model_actual_running_graph_path_, TrainingSession::SaveOption::NO_RELOAD));
  }

  printf("Before training \n");
  ORT_RETURN_IF_ERROR(Evaluate(session_, false));

  ORT_RETURN_IF_ERROR(TrainingLoop());

  return EndTraining();
}

Status TrainingRunner::TrainingLoop() {
  // The optimizer out of the graph, will be used if params_.in_graph_optimizer_name_ is not set
  WeightUpdater<out_graph_optimizer::GradientDescent> weight_updater(session_,
                                                                     {params_.learning_rate_,
                                                                      TrainingUtil::GetCpuAllocator()});

  // Prepare output names
  auto output_names_include_gradients = session_.GetModelOutputNames();
  vector<string> training_output_names(output_names_include_gradients.begin(), output_names_include_gradients.end());

  // If use in-graph optimizer, the learning_rate is a new feed
  bool use_in_graph_optimizer = !params_.in_graph_optimizer_name_.empty();
  vector<string> feed_names = training_data_.TensorNames();
  MLValue learning_rate_ml_value;
  if (use_in_graph_optimizer) {
    feed_names.insert(feed_names.begin(), SGD_LEARNING_RATE_STRING);
    TrainingUtil::CreateMLValue(TrainingUtil::GetCpuAllocator(),
                                {1},
                                vector<float>(1, params_.learning_rate_),
                                &learning_rate_ml_value);
  }

  for (size_t epoch = 0; epoch < params_.num_of_epoch_; ++epoch) {
    // Shuffle the data for each epoch
    training_data_.RandomShuffle();

    // loop through the data
    for (size_t batch = 0; batch < training_data_.TotalBatch(params_.batch_size_); ++batch) {
      std::vector<MLValue> feeds = training_data_.GetKthBatch(params_.batch_size_, batch);
      if (use_in_graph_optimizer) {
        feeds.insert(feeds.begin(), learning_rate_ml_value);
      }

      vector<MLValue> gradient_fetches;
      ORT_RETURN_IF_ERROR(session_.Run(RunOptions(),
                                       feed_names,
                                       feeds,
                                       training_output_names,
                                       &gradient_fetches));

      NameMLValMap grad;
      for (int i = 0; i < training_output_names.size(); i++) {
        if (training_output_names[i] == params_.loss_func_info_.loss_name_ ||
            training_output_names[i] == params_.model_prediction_name_) {
          continue;
        }
        grad.insert(make_pair(training_output_names[i], gradient_fetches[i]));
      }

      // Print some info when reaching the end of the batch.
      printf("batch: %d/%d, epoch: %d/%d \n",
             static_cast<int>(batch),
             static_cast<int>(training_data_.TotalBatch(params_.batch_size_)),
             static_cast<int>(epoch + 1),
             static_cast<int>(params_.num_of_epoch_));
      printf("Training data range: [%d - %d)\n",
             static_cast<int>(batch * params_.batch_size_),
             static_cast<int>((batch + 1) * params_.batch_size_ - 1));

      if (params_.in_graph_optimizer_name_.empty()) {
        weight_updater.Update(grad, params_.batch_size_);
      }
      ORT_RETURN_IF_ERROR(Evaluate(session_));
    }
  }

  return Status::OK();
}

Status TrainingRunner::EndTraining() {
  // Test the in-memory model before saving.
  printf("\nEvaluateing the final model on the test set.\n");
  ORT_RETURN_IF_ERROR(Evaluate(session_, true));

  printf("\nSaving the trained model.\n");
  if (!params_.model_trained_path_.empty()) {
    ORT_RETURN_IF_ERROR(session_.Save(params_.model_trained_path_,
                                      TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS));
  }
  if (!params_.model_trained_with_loss_func_path_.empty()) {
    ORT_RETURN_IF_ERROR(session_.Save(params_.model_trained_with_loss_func_path_,
                                      TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC));
  }

  // Load and test the trained model.
  SessionOptions test_so;
  InferenceSession test_session{test_so};

  ORT_RETURN_IF_ERROR(test_session.Load(params_.model_trained_with_loss_func_path_));
  ORT_RETURN_IF_ERROR(test_session.Initialize());
  printf("\nTesting the saved model: %s\n", params_.model_trained_with_loss_func_path_.c_str());
  return Evaluate(test_session, true /*use full test set*/);
}

Status TrainingRunner::Evaluate(InferenceSession& session, bool use_full_set) {
  // A static batch index representing current test batch
  static size_t current_batch = 0;

  if (current_batch == 0 && !use_full_set) {
    test_data_.RandomShuffle();
    printf("Randomly shuffle test data.\n");
  }

  size_t evaluation_batch_size = use_full_set ? test_data_.NumSamples() : params_.num_of_samples_for_evaluation_;

  printf("Test data range: [%d - %d)\n",
         static_cast<int>(current_batch * evaluation_batch_size),
         static_cast<int>((current_batch + 1) * evaluation_batch_size - 1));

  // If use in-graph optimizer, the learning_rate is a new feed
  bool use_in_graph_optimizer = !params_.in_graph_optimizer_name_.empty();

  vector<string> feed_names = test_data_.TensorNames();
  std::vector<MLValue> feeds = test_data_.GetKthBatch(evaluation_batch_size, current_batch);
  if (use_in_graph_optimizer) {
    feed_names.insert(feed_names.begin(), SGD_LEARNING_RATE_STRING);

    MLValue learning_rate_ml_value;
    TrainingUtil::CreateMLValue(TrainingUtil::GetCpuAllocator(),
                                {1},
                                vector<float>(1, params_.learning_rate_),
                                &learning_rate_ml_value);
    feeds.insert(feeds.begin(), learning_rate_ml_value);
  }
  vector<MLValue> fetches;
  ORT_RETURN_IF_ERROR(session.Run(RunOptions(),
                                  feed_names,
                                  feeds,
                                  {params_.model_prediction_name_, params_.loss_func_info_.loss_name_},
                                  &fetches));

  // Call error function with predict, label and loss.
  if (params_.error_function_) {
    params_.error_function_(fetches[0] /*predict*/, feeds.back() /*label*/, fetches[1] /*loss*/);
  }

  // Call afer a test batch.
  if (params_.post_evaluation_callback_) {
    params_.post_evaluation_callback_(evaluation_batch_size);
  }

  current_batch++;
  return Status::OK();
}
}  // namespace training
}  // namespace onnxruntime
