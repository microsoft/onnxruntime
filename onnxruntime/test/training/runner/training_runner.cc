// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <memory>
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/environment.h"
#include "test/training/runner/training_runner.h"
#include "test/training/runner/training_util.h"
#include "core/training/training_optimizer.h"
#include "core/training/weight_updater.h"

using namespace std;

namespace onnxruntime {
namespace training {

TrainingRunner::TrainingRunner(TrainingData& trainingData, TestData& testData, const Parameters& params)
    : training_data_(trainingData), test_data_(testData), params_(params), session_(SessionOptions()) {
  ORT_ENFORCE(!params_.model_path_.empty());
  ORT_ENFORCE(!params_.weights_to_train_.empty());
  ORT_ENFORCE(!params_.model_trained_path_.empty() || !params_.model_trained_with_loss_func_path_.empty());
}

Status TrainingRunner::Initialize() {
  ORT_RETURN_IF_ERROR(session_.Load(params_.model_path_));

  // Add loss func
  ORT_RETURN_IF_ERROR(session_.AddLossFuncion(params_.loss_func_info_));
  if (!params_.model_with_loss_func_path_.empty()) {
    ORT_RETURN_IF_ERROR(session_.Save(params_.model_with_loss_func_path_,
                                      TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC));
  }

  // Add gradient graph
  ORT_RETURN_IF_ERROR(session_.BuildGradientGraph(params_.weights_to_train_, params_.loss_func_info_.loss_name_));
  if (!params_.model_with_training_graph_path_.empty()) {
    ORT_RETURN_IF_ERROR(session_.Save(params_.model_with_training_graph_path_,
                                      TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS));
  }

  return session_.Initialize();
}

Status TrainingRunner::Run() {
  printf("Before training \n");
  ORT_RETURN_IF_ERROR(Evaluate(session_, true));

  ORT_RETURN_IF_ERROR(TrainingLoop());

  return EndTraining();
}

// A wrapper to get the k-th batch's [start, end) in a vector.
template <typename T>
class BatchWrapper {
 public:
  BatchWrapper(const vector<T>& vec, size_t batch_size) : vec_(vec),
                                                          batch_size_(min(batch_size, vec.size())) {
  }

  // Get total num of batch
  size_t TotalBatch() const {
    return vec_.size() / batch_size_ + ((vec_.size() % batch_size_ > 0) ? 1 : 0);
  }

  // Get the k-th batch's [start, end)
  pair<typename vector<T>::const_iterator, typename vector<T>::const_iterator> KthBatchRange(size_t k_th) const {
    auto startIt = vec_.cbegin();
    advance(startIt, min(vec_.size(), batch_size_ * k_th));

    auto endIt = vec_.cbegin();
    advance(endIt, min(vec_.size(), batch_size_ * k_th + batch_size_));
    return make_pair(startIt, endIt);
  }

 private:
  const vector<T>& vec_;
  size_t batch_size_;
};

Status TrainingRunner::TrainingLoop() {
  WeightUpdater<GradientDescent> weight_updater(session_, {params_.learning_rate_, TrainingUtil::GetCpuAllocator()});

  // Prepare output names
  auto output_names_include_gradients = session_.GetModelOutputNames();
  vector<string> training_output_names(output_names_include_gradients.begin(), output_names_include_gradients.end());

  BatchWrapper<TrainingRunner::TrainingData::value_type> batches(training_data_, params_.batch_size_);
  for (size_t epoch = 0; epoch < params_.num_of_epoch_; ++epoch) {
    // Shuffle the data for each epoch
    random_shuffle(training_data_.begin(), training_data_.end());

    // loop through the data
    for (size_t batch = 0; batch < batches.TotalBatch(); ++batch) {
      // Accumulated gradients.
      vector<NameMLValMap> grads_batch;
      auto batchRange = batches.KthBatchRange(batch);

      for (auto it = batchRange.first; it < batchRange.second; ++it) {
        vector<MLValue> gradient_fetches;
        ORT_RETURN_IF_ERROR(session_.Run(RunOptions(),
                                         (*it)->names_,
                                         (*it)->values_,
                                         training_output_names,
                                         &gradient_fetches));

        // Accumulated grads from multi run.
        NameMLValMap grad;
        for (int i = 0; i < training_output_names.size(); i++) {
          if (training_output_names[i] == params_.loss_func_info_.loss_name_ ||
              training_output_names[i] == params_.loss_func_info_.prediction_name_) {
            continue;
          }
          grad.insert(make_pair(training_output_names[i], gradient_fetches[i]));
        }
        grads_batch.emplace_back(grad);
      }

      // Print some info when reaching the end of the batch.
      printf("batch: %d/%d, epoch: %d/%d \n",
             static_cast<int>(batch),
             static_cast<int>(batches.TotalBatch()),
             static_cast<int>(epoch + 1),
             static_cast<int>(params_.num_of_epoch_));
      printf("Training data range: [%d - %d)\n",
             static_cast<int>(distance(training_data_.cbegin(), batchRange.first)),
             static_cast<int>(distance(training_data_.cbegin(), batchRange.second)));

      weight_updater.Update(grads_batch);
      ORT_RETURN_IF_ERROR(Evaluate(session_));
    }
  }

  return Status::OK();
}

Status TrainingRunner::EndTraining() {
  if (!params_.model_trained_path_.empty()) {
    ORT_RETURN_IF_ERROR(session_.Save(params_.model_trained_path_,
                                      TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC));
  }
  if (!params_.model_trained_with_loss_func_path_.empty()) {
    ORT_RETURN_IF_ERROR(session_.Save(params_.model_trained_with_loss_func_path_,
                                      TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS));
  }

  //Load and test the trained model.
  SessionOptions test_so;
  InferenceSession test_session{test_so};

  ORT_RETURN_IF_ERROR(test_session.Load(params_.model_trained_path_));
  ORT_RETURN_IF_ERROR(test_session.Initialize());
  printf("\nTesting the saved model:\n");
  return Evaluate(test_session, true /*use full test set*/);
}

Status TrainingRunner::Evaluate(InferenceSession& session, bool use_full_set) {
  // A static batch index representing current test batch
  static size_t current_batch = 0;
  if (current_batch == 0 && !use_full_set) {
    random_shuffle(test_data_.begin(), test_data_.end());
    printf("Randomly shuffle test data.\n");
  }

  BatchWrapper<TrainingRunner::TestData::value_type> batches(test_data_, params_.num_of_samples_for_evaluation_);
  auto batchRange = batches.KthBatchRange(current_batch);
  if (use_full_set) {
    // If full set is used, reset the range.
    batchRange = {test_data_.begin(), test_data_.end()};
  } else {
    // Increase current_batch for next Evaluate() call.
    current_batch = (current_batch + 1) % batches.TotalBatch();
  }

  printf("Test data range: [%d - %d)\n",
         static_cast<int>(distance(test_data_.cbegin(), batchRange.first)),
         static_cast<int>(distance(test_data_.cbegin(), batchRange.second)));

  for (auto it = batchRange.first; it < batchRange.second; ++it) {
    vector<MLValue> fetches;
    ORT_RETURN_IF_ERROR(session.Run(RunOptions(),
                                    (*it)->names_,
                                    (*it)->values_,
                                    {params_.loss_func_info_.prediction_name_, params_.loss_func_info_.loss_name_},
                                    &fetches));
    // Call error function with predict and label.
    if (params_.error_function_) {
      params_.error_function_(fetches[0], (*it)->values_[(*it)->label_index_]);
    }
  }

  // Call afer a test batch.
  if (params_.post_evaluation_callback_) {
    size_t total_run = distance(batchRange.first, batchRange.second);
    params_.post_evaluation_callback_(total_run);
  }

  return Status::OK();
}
}  // namespace training
}  // namespace onnxruntime
