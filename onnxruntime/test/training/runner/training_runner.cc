// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <memory>
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/session/environment.h"
#include "test/training/runner/training_runner.h"
#include "test/training/runner/training_util.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_execution_provider.h"
#endif

using namespace std;

namespace onnxruntime {
namespace training {

static SessionOptions SESSION_OPTION = {
    true,                              //enable_sequential_execution
    false,                             //enable_profiling
    true,                              //enable_mem_pattern
    true,                              //enable_cpu_mem_arena
    ORT_TSTR("onnxruntime_profile_"),  //profile_file_prefix
    "",                                //session_logid
    -1,                                //session_log_severity_level
    0,                                 //session_log_verbosity_level
    5,                                 //max_num_graph_transformation_steps
    TransformerLevel::Level1,          //graph_optimization_level
    0,                                 //session_thread_pool_size
};

TrainingRunner::TrainingRunner(std::shared_ptr<IDataLoader> training_data_loader,
                               std::shared_ptr<IDataLoader> test_data_loader,
                               const Parameters& params)
    : training_data_loader_(training_data_loader),
      test_data_loader_(test_data_loader),
      step_(0),
      params_(params),
      session_(SESSION_OPTION) {
  ORT_ENFORCE(!params_.model_path.empty());
  if (!params.weights_to_train.empty())
    ORT_ENFORCE(params.weights_not_to_train.empty());
  ORT_ENFORCE(!params_.model_trained_path.empty() || !params_.model_trained_with_loss_func_path.empty());
  ORT_ENFORCE(!params_.training_optimizer_name.empty());
}

Status TrainingRunner::Initialize() {
  ORT_RETURN_IF_ERROR(session_.Load(params_.model_path));

  // Add loss func
  ORT_RETURN_IF_ERROR(session_.BuildLossFunction(params_.loss_func_info));
  if (params_.mpi_context.world_rank == 0 && !params_.model_with_loss_func_path.empty()) {
    session_.Save(params_.model_with_loss_func_path, TrainingSession::SaveOption::NO_RELOAD);
  }

  // Get the weights-to-train list if user specify it.
  // Otherwise, generate the list by removing not-to-train ones from all initializers.
  auto weights_to_train = params_.weights_to_train;
  if (weights_to_train.empty()) {
    weights_to_train = session_.GetTrainableModelInitializers(params_.immutable_weights);
    for (const auto& not_to_train : params_.weights_not_to_train) {
      weights_to_train.erase(not_to_train);
    }
  }

  for (const std::string& weight : weights_to_train) {
    std::cout << "Training weight " << weight << std::endl;
  }

  // Add gradient graph
  ORT_RETURN_IF_ERROR(session_.BuildGradientGraph(weights_to_train, params_.loss_func_info.loss_name));

  // Add optimizer
  std::unordered_map<std::string, OptimizerInfo> opt_info;
  ORT_RETURN_IF_ERROR(SetupOptimizerParams(weights_to_train, opt_info));
  ORT_RETURN_IF_ERROR(session_.BuildOptimizer(opt_info));

  if (params_.use_mixed_precision) {
    ORT_RETURN_IF_ERROR(session_.EnableMixedPrecision(weights_to_train));
  }

  // Expose all fetches as graph outputs
  ORT_RETURN_IF_ERROR(session_.OverrideGraphOutputs(params_.fetch_names));

  if (params_.mpi_context.world_rank == 0 && !params_.model_with_training_graph_path.empty()) {
    session_.Save(params_.model_with_training_graph_path, TrainingSession::SaveOption::NO_RELOAD);
  }

  if (params_.use_gist) {
    ORT_RETURN_IF_ERROR(session_.AddGistEncoding());
    if (!params_.model_gist_encode.empty()) {
      session_.Save(params_.model_gist_encode, TrainingSession::SaveOption::NO_RELOAD);
    }
  }

#ifdef USE_CUDA
  if (params_.use_cuda) {
    CUDAExecutionProviderInfo xp_info{params_.mpi_context.local_rank};
    ORT_RETURN_IF_ERROR(session_.RegisterExecutionProvider(std::make_unique<CUDAExecutionProvider>(xp_info)));
  }
#endif
  ORT_RETURN_IF_ERROR(session_.UpdateTrainableWeightsInfoInGraph());

  if (params_.use_profiler && !SESSION_OPTION.enable_profiling) {
    // Profiling has not already been enabled, so override from command line options.

    if (params_.max_profile_records > 0) {
      profiling::Profiler::max_num_events_ = params_.max_profile_records;
    }

    session_.StartProfiling(SESSION_OPTION.profile_file_prefix);
  }

  return session_.Initialize();
}

Status TrainingRunner::Run() {
  if (params_.mpi_context.world_rank == 0 && !params_.model_actual_running_graph_path.empty()) {
    session_.Save(params_.model_actual_running_graph_path, TrainingSession::SaveOption::NO_RELOAD);
  }

  ORT_RETURN_IF_ERROR(TrainingLoop());

  ORT_RETURN_IF_ERROR(EndTraining());
  return Status::OK();
}

Status TrainingRunner::TrainingLoop() {
  // Prepare fetches
  const VectorString& fetch_names = params_.fetch_names;
  const VectorString& feed_names = training_data_loader_->DataSetTensorNames();

  double total_time{0};
  //Set the first N batchs as warm-up iterations
  size_t warm_up_iters = 10;

  size_t num_shards_to_visit = params_.num_of_epoch;
  if (training_data_loader_) {
    num_shards_to_visit *= training_data_loader_->NumShards();
  }

  size_t total_batch_num = 0;
  for (size_t shard_it = 0; shard_it < num_shards_to_visit; ++shard_it) {
    auto training_data = training_data_loader_->CurrentDataSet();

    // Shuffle the data for each epoch
    if (params_.shuffle_data) {
      printf("Randomly shuffle training data.\n");
      training_data->RandomShuffle();
    }

    // loop through the data
    size_t batch_num_cur_shard = training_data->TotalBatch(params_.batch_size);
    total_batch_num += batch_num_cur_shard;
    for (size_t batch = 0; batch < batch_num_cur_shard; ++batch) {
      std::vector<MLValue> feeds = training_data->GetKthBatch(params_.batch_size, batch);
      vector<MLValue> fetches;

      std::chrono::duration<double> duration_seconds;
      auto start = std::chrono::high_resolution_clock::now();
      auto end = start;
      ORT_RETURN_IF_ERROR(session_.Run(RunOptions(),
                                       feed_names,
                                       feeds,
                                       fetch_names,
                                       &fetches));
      step_++;

      //Start counting after warm-up iterations
      if (batch >= warm_up_iters || shard_it > 0) {
        end = std::chrono::high_resolution_clock::now();
        duration_seconds = end - start;
        total_time += duration_seconds.count();
      }

      // Print some info when reaching the end of the batch.
      printf("batch: %d/%d, shard_iteration: %d/%d \n",
             static_cast<int>(batch),
             static_cast<int>(batch_num_cur_shard),
             static_cast<int>(shard_it + 1),
             static_cast<int>(num_shards_to_visit));
      printf("Training data range: [%d - %d)\n",
             static_cast<int>(batch * params_.batch_size),
             static_cast<int>((batch + 1) * params_.batch_size - 1));

      if (step_ % params_.evaluation_period == 0) {
        ORT_RETURN_IF_ERROR(Evaluate(session_));
      }
    }

    // Move to next shard of data, except for the last iteration.
    if (training_data_loader_ != nullptr && shard_it != num_shards_to_visit - 1) {
      training_data_loader_->MoveToNextDataSet();
    }
  }

  auto total_batchs = total_batch_num - warm_up_iters;
  std::cout << "Total running time:" << total_time << " seconds" << std::endl
            << "Average running time per batch:" << total_time / total_batchs * 1000 << " ms" << std::endl
            << "Throughput: " << params_.batch_size * total_batchs / total_time << " Examples / second" << std::endl;

  return Status::OK();
}

Status TrainingRunner::EndTraining() {
  if (params_.use_profiler) {
    // Write profiler data to disk.
    // We do this first in case there are any problems saving the trained model.
    std::string profile_file = session_.EndProfiling();
    std::cout << "Profiler data written to file " << profile_file;
  }

  if (params_.mpi_context.world_rank != 0) {
    printf("Skipping end-training on Device #%d, as it's not the root.", params_.mpi_context.world_rank);
    return Status::OK();
  }

  // Test the in-memory model before saving.
  printf("\nEvaluating the final model on the test set.\n");
  ORT_RETURN_IF_ERROR(Evaluate(session_));

  printf("\nSaving the trained model.\n");
  if (!params_.model_trained_path.empty()) {
    session_.Save(params_.model_trained_path, TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS);
  }
  if (!params_.model_trained_with_loss_func_path.empty()) {
    session_.Save(params_.model_trained_with_loss_func_path,
                  TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC);
  }

  // Load and test the trained model.
  printf("\nTesting the saved model: %s\n", params_.model_trained_with_loss_func_path.c_str());
  return LoadAndEvaluate(params_.model_trained_with_loss_func_path);
}

Status TrainingRunner::Evaluate(InferenceSession& session) {
  if (params_.skip_evaluation) {
    printf("Skipping evaluation...\n");
    return Status::OK();
  }

  if (params_.mpi_context.world_rank != 0) {
    printf("Skipping evaluation on Device #%d, as it's not the root.\n", params_.mpi_context.world_rank);
    return Status::OK();
  }

  // A static batch index representing current test batch
  static size_t current_batch = 0;
  const vector<string> feed_names = test_data_loader_->DataSetTensorNames();
  auto test_data = test_data_loader_->CurrentDataSet();
  if (params_.shuffle_data && current_batch == 0) {
    printf("Randomly shuffle test data.\n");
    test_data->RandomShuffle();
  }

  const size_t evaluation_batch_size = params_.eval_batch_size;

  printf("Test data range: [%d - %d)\n",
         static_cast<int>(current_batch * evaluation_batch_size),
         static_cast<int>((current_batch + 1) * evaluation_batch_size - 1));

  const size_t num_batches = size_t(ceil((float)evaluation_batch_size / (float)params_.batch_size));
  if (evaluation_batch_size % params_.batch_size != 0) {
    printf(
        "evaluation_batch_size %zu is not an integer multiple of batch_size %zu. "
        "Using evaluation_batch_size %zu",
        evaluation_batch_size,
        params_.batch_size,
        num_batches * params_.batch_size);
  }

  RunOptions run_options;
  run_options.only_execute_path_to_fetches = true;
  for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    std::vector<MLValue> feeds = test_data->GetKthBatch(params_.batch_size, current_batch);
    vector<MLValue> fetches;
    ORT_RETURN_IF_ERROR(session.Run(run_options,
                                    feed_names,
                                    feeds,
                                    params_.fetch_names,
                                    &fetches));

    // Call error function
    if (params_.error_function) {
      params_.error_function(feed_names, feeds, params_.fetch_names, fetches);
    }

    // Set to next batch
    if (++current_batch >= test_data->TotalBatch(params_.batch_size)) {
      if (test_data_loader_ != nullptr) {
        // Move to next shard
        test_data = test_data_loader_->MoveToNextDataSet();
      }
      current_batch = 0;
    }
  }

  // Call after a test batch.
  if (params_.post_evaluation_callback) {
    params_.post_evaluation_callback(evaluation_batch_size, step_);
  }

  return Status::OK();
}

Status TrainingRunner::LoadAndEvaluate(const std::string& model_path) {
  InferenceSession s{SessionOptions()};
#ifdef USE_CUDA
  CUDAExecutionProviderInfo xp_info{params_.mpi_context.world_rank};
  ORT_RETURN_IF_ERROR(s.RegisterExecutionProvider(std::make_unique<CUDAExecutionProvider>(xp_info)));
#endif
  ORT_RETURN_IF_ERROR(s.Load(model_path));
  ORT_RETURN_IF_ERROR(s.Initialize());
  return Evaluate(s);
}

Status TrainingRunner::SetupOptimizerParams(const std::unordered_set<std::string>& weights_to_train,
                                            std::unordered_map<std::string, OptimizerInfo>& opt_infos) {
  // Prepare the weight<->optimizer mapping.
  // All weights use the same type of optimizer
  OptimizerInfo opt_info{
      params_.training_optimizer_name,
      params_.learning_rate,
      params_.mpi_context.world_rank,
      params_.mpi_context.world_size,
      params_.optimizer_attributes};

  opt_infos.reserve(weights_to_train.size());
  for (const auto& weight_name : weights_to_train) {
    opt_infos[weight_name] = opt_info;
  }

  return Status::OK();
}
}  // namespace training
}  // namespace onnxruntime
