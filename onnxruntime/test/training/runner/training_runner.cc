// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/training/runner/training_runner.h"

#include <algorithm>
#include <memory>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/session/environment.h"
#include "core/training/optimizer_graph_builder.h"
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
    ORT_TSTR(""),                      //optimized_model_filepath
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

TrainingRunner::TrainingRunner(Parameters params)
    : step_(0),
      round_(0),
      weight_update_step_count_(0),
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
  ORT_RETURN_IF_ERROR(session_.ApplyTransformationsToMainGraph());

  std::string loss_scale_input_name{};
  if (params_.use_mixed_precision) {
    ORT_RETURN_IF_ERROR(session_.BuildLossScalingFactorInput(params_.loss_scale, loss_scale_input_name));
    params_.scalar_names.push_back(loss_scale_input_name);

    if (params_.loss_scale == 0.0f) {
      // use dynamic loss_scale
      loss_scaler_ = std::make_unique<LossScaler>(loss_scale_input_name, true, static_cast<float>(1 << 20));
    } else {
      // use static loss_scale
      loss_scaler_ = std::make_unique<LossScaler>(loss_scale_input_name, false, params_.loss_scale);
    }
  }

  // Add loss func
  std::string actual_loss_name{};
  ORT_RETURN_IF_ERROR(session_.BuildLossFunction(
      params_.loss_func_info, loss_scale_input_name, actual_loss_name));
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
  ORT_RETURN_IF_ERROR(session_.BuildGradientGraph(weights_to_train, actual_loss_name));

  std::unordered_map<std::string, NodeArg*> fp16_weights_map;
  if (params_.use_mixed_precision) {
    ORT_RETURN_IF_ERROR(session_.EnableMixedPrecision(weights_to_train, params_.use_fp16_initializer, fp16_weights_map));
  }

  // Add optimizer
  OptimizerGraphConfig opt_graph_config{};
  std::unordered_map<std::string, OptimizerNodeConfig> opt_configs;
  std::unordered_map<std::string, std::string> opt_graph_outputs;
  ORT_RETURN_IF_ERROR(SetupOptimizerParams(
      weights_to_train, fp16_weights_map, loss_scale_input_name,
      opt_graph_config, opt_configs));
  ORT_RETURN_IF_ERROR(session_.BuildOptimizer(opt_graph_config, opt_configs, opt_graph_outputs));
  opt_graph_outputs_ = opt_graph_outputs;

  // Add tensorboard
  if (params_.EnableTensorboard()) {
    if (opt_graph_outputs_.count(kGradientAllIsFiniteOutputKey) > 0){
      params_.scalar_names.push_back(opt_graph_outputs_[kGradientAllIsFiniteOutputKey]);
    }
    ORT_RETURN_IF_ERROR(session_.AddTensorboard(
      params_.summary_name, params_.scalar_names, params_.histogram_names,
      params_.norm_names, params_.dump_convergence_metrics));
  }

  // Expose all fetches as graph outputs
  VectorString fetch_names = params_.fetch_names;
  for (const auto& it : opt_graph_outputs) {
    fetch_names.push_back(it.second);
  }
  ORT_RETURN_IF_ERROR(session_.OverrideGraphOutputs(fetch_names));

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

Status TrainingRunner::Run(std::shared_ptr<IDataLoader> training_data_loader, std::shared_ptr<IDataLoader> test_data_loader) {
  if (params_.mpi_context.world_rank == 0 && !params_.model_actual_running_graph_path.empty()) {
    session_.Save(params_.model_actual_running_graph_path, TrainingSession::SaveOption::NO_RELOAD);
  }

  // Add one for each call of Run(..).
  ORT_RETURN_IF_ERROR(TrainingLoop(training_data_loader, test_data_loader));
  round_++;

  return Status::OK();
}

Status TrainingRunner::TrainingLoop(std::shared_ptr<IDataLoader> training_data_loader, std::shared_ptr<IDataLoader> test_data_loader) {
  step_ = 0;
  VectorString feed_names = training_data_loader->DataSetTensorNames();
  if (loss_scaler_) {
    feed_names.push_back(loss_scaler_->GetLossScaleInputName());
  }
  feed_names.push_back(params_.lr_params.feed_name);

  OrtValue loss_scale_val;
  OrtValue lr_ort_val;

  VectorString fetch_names = params_.fetch_names;
  if (params_.use_mixed_precision) {
    auto it = opt_graph_outputs_.find(kGradientAllIsFiniteOutputKey);
    ORT_RETURN_IF(it == opt_graph_outputs_.end(), "Gradient AllIsFinite output is missing in the optimizer output");
    fetch_names.push_back(it->second);
  }

  VectorString fetch_grad_accumulator_output;
  if (params_.gradient_accumulation_steps > 1) {
    auto it = opt_graph_outputs_.find(kGradientAccumulationOutputKey);
    ORT_RETURN_IF(it == opt_graph_outputs_.end(), "Gradient accumulation output is missing in the optimizer output");
    fetch_grad_accumulator_output.push_back(it->second);
  }

  if (params_.is_perf_test && params_.perf_warm_up_iters > 0) {
    auto training_data = training_data_loader->CurrentDataSet();
    auto num_batches = training_data->TotalBatch(params_.batch_size);
    ORT_RETURN_IF(params_.perf_warm_up_iters > num_batches,
                  "perf_warm_up_iters is bigger than number of available batches.");

    printf("Warming up for perf test.\n");
    for (size_t batch = 0; batch < params_.perf_warm_up_iters; ++batch) {
      std::vector<MLValue> feeds = training_data->GetKthBatch(params_.batch_size, batch);
      if (loss_scaler_) {
        float loss_scale = loss_scaler_->GetLossScale();
        TrainingUtil::CreateCpuMLValue({1}, std::vector<float>{loss_scale}, &loss_scale_val);
        feeds.push_back(loss_scale_val);
      }
      TrainingUtil::CreateCpuMLValue({1}, std::vector<float>{params_.lr_params.initial_lr}, &lr_ort_val);
      feeds.push_back(lr_ort_val);

      vector<MLValue> fetches;
      ORT_RETURN_IF_ERROR(session_.Run(RunOptions(),
                                       feed_names,
                                       feeds,
                                       fetch_names,
                                       &fetches));
    }
  }

  const size_t num_shards_to_visit = training_data_loader->NumShards();
  const auto lr_scheduler = LearningRateScheduler::Create(params_.lr_params, params_.num_train_steps);

  double total_time{0};
  size_t epoch = 0;
  size_t gradient_accumulation_step_count = 0;

  while (step_ < params_.num_train_steps) {
    for (size_t shard_it = 0; shard_it < num_shards_to_visit; ++shard_it) {
      auto training_data = training_data_loader->CurrentDataSet();
      if (training_data == nullptr) {
        printf("Skip shard %d which is failed to load.\n", static_cast<int>(shard_it));
        training_data_loader->MoveToNextDataSet();
        continue;
      }

      // Shuffle the data for each epoch
      if (params_.shuffle_data) {
        printf("Randomly shuffle training data.\n");
        training_data->RandomShuffle();
      }

      // loop through the data
      size_t batch_num_cur_shard = training_data->TotalBatch(params_.batch_size);
      for (size_t batch = 0; batch < batch_num_cur_shard && step_ < params_.num_train_steps; ++batch) {
        std::vector<MLValue> feeds = training_data->GetKthBatch(params_.batch_size, batch);
        if (loss_scaler_) {
          float loss_scale = loss_scaler_->GetLossScale();
          TrainingUtil::CreateCpuMLValue({1}, std::vector<float>{loss_scale}, &loss_scale_val);
          feeds.push_back(loss_scale_val);
        }

        {
          float learning_rate = lr_scheduler->GetLearningRate(step_ + 1);
          TrainingUtil::CreateCpuMLValue({1}, std::vector<float>{learning_rate}, &lr_ort_val);
          feeds.push_back(lr_ort_val);
        }

        vector<MLValue> fetches;

        auto start = std::chrono::high_resolution_clock::now();

        if ((step_ + 1) % params_.gradient_accumulation_steps == 0) {
          ORT_RETURN_IF_ERROR(session_.Run(RunOptions(),
                                           feed_names,
                                           feeds,
                                           fetch_names,
                                           &fetches));

          if (loss_scaler_) {
            auto it = std::find(fetch_names.begin(), fetch_names.end(), opt_graph_outputs_[kGradientAllIsFiniteOutputKey]);
            if (it != fetch_names.end()) {
              const size_t index = static_cast<size_t>(std::distance(fetch_names.begin(), it));
              const Tensor& all_is_finite_t = fetches[index].Get<Tensor>();
              const bool is_all_finite = *(all_is_finite_t.template Data<bool>());
              loss_scaler_->UpdateLossScale(is_all_finite);
            }
          }

          if (!params_.is_perf_test && weight_update_step_count_ % params_.display_loss_steps == 0) {
            if (params_.error_function) {
              params_.error_function(feed_names, feeds, fetch_names, fetches, weight_update_step_count_);
            }
            if (params_.post_evaluation_callback) {
              params_.post_evaluation_callback(params_.batch_size, weight_update_step_count_, "train");
            }
          }

          weight_update_step_count_++;
        } else {
          RunOptions run_options;
          run_options.only_execute_path_to_fetches = true;
          ORT_RETURN_IF_ERROR(session_.Run(run_options,
                                           feed_names,
                                           feeds,
                                           fetch_grad_accumulator_output,
                                           &fetches));
          gradient_accumulation_step_count++;
        }
        step_++;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_seconds = end - start;
        total_time += duration_seconds.count();

        // Print some info when reaching the end of the batch.
        printf("Round %d, Step: %d, epoch: %d, batch: %d/%d, shard_iteration: %d/%d, time: %.2f ms, throughput: %.2f ex/sec \n",
               static_cast<int>(round_),
               static_cast<int>(step_),
               static_cast<int>(epoch),
               static_cast<int>(batch),
               static_cast<int>(batch_num_cur_shard),
               static_cast<int>(shard_it + 1),
               static_cast<int>(num_shards_to_visit),
               duration_seconds.count() * 1000,
               params_.batch_size * step_ / total_time);
        printf("Training data range: [%d - %d)\n",
               static_cast<int>(batch * params_.batch_size),
               static_cast<int>((batch + 1) * params_.batch_size - 1));

        if (params_.do_eval && step_ % params_.evaluation_period == 0) {
          ORT_RETURN_IF_ERROR(Evaluate(session_, test_data_loader));
        }
      }  // end of one file/shard

      if (step_ < params_.num_train_steps) {
        training_data_loader->MoveToNextDataSet();
      }
    }  // end of one epoch

    epoch++;
  }

  std::cout << "Round: " << round_ << "\n"
            << "Batch size: " << params_.batch_size << "\n"
            << "Number of Batches: " << step_ << "\n"
            << "Gradient Accumulation Steps: " << gradient_accumulation_step_count << "\n"
            << "Weight Update Steps: " << weight_update_step_count_ << "\n"
            << "Total Running Time: " << total_time << " Seconds \n"
            << "Average Running Time Per Batch: " << total_time / step_ * 1000 << " ms\n"
            << "Throughput: " << params_.batch_size * step_ / total_time << " Examples / Second\n";
  return Status::OK();
}

Status TrainingRunner::EndTraining(std::shared_ptr<IDataLoader> data_loader) {
  if (params_.use_profiler) {
    // Write profiler data to disk.
    // We do this first in case there are any problems saving the trained model.
    std::string profile_file = session_.EndProfiling();
    std::cout << "Profiler data written to file " << profile_file << "\n";
  }

  if (params_.mpi_context.world_rank != 0) {
    printf("Skipping end-training on Device #%d, as it's not the root.\n", params_.mpi_context.world_rank);
    return Status::OK();
  }

  // Test the in-memory model before saving.
  printf("\nEvaluating the final model on the test set.\n");
  ORT_RETURN_IF_ERROR(Evaluate(session_, data_loader));

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
  return LoadAndEvaluate(params_.model_trained_with_loss_func_path, data_loader);
}

Status TrainingRunner::Evaluate(InferenceSession& session, std::shared_ptr<IDataLoader> data_loader) {
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
  vector<string> feed_names = data_loader->DataSetTensorNames();
  if (loss_scaler_) {
    feed_names.push_back(loss_scaler_->GetLossScaleInputName());
  }
  feed_names.push_back(params_.lr_params.feed_name);
  // TODO add loss scaling factor and learning rate to feeds
  auto test_data = data_loader->CurrentDataSet();
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
        "WARNING: evaluation_batch_size %zu is not an integer multiple of batch_size %zu. "
        "Using evaluation_batch_size %zu\n",
        evaluation_batch_size,
        params_.batch_size,
        num_batches * params_.batch_size);
  }

  OrtValue loss_scale_val;
  TrainingUtil::CreateCpuMLValue({1}, std::vector<float>{params_.loss_scale}, &loss_scale_val);

  RunOptions run_options;
  run_options.only_execute_path_to_fetches = true;
  for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
    std::vector<MLValue> feeds = test_data->GetKthBatch(params_.batch_size, current_batch);
    if (loss_scaler_) {
      feeds.push_back(loss_scale_val);
    }
    OrtValue lr_ort_val;
    TrainingUtil::CreateCpuMLValue({1}, std::vector<float>{params_.lr_params.initial_lr}, &lr_ort_val);
    feeds.push_back(lr_ort_val);
    vector<MLValue> fetches;
    ORT_RETURN_IF_ERROR(session.Run(run_options,
                                    feed_names,
                                    feeds,
                                    params_.fetch_names,
                                    &fetches));

    // Call error function
    if (params_.error_function) {
      params_.error_function(feed_names, feeds, params_.fetch_names, fetches, step_);
    }

    // Set to next batch
    if (++current_batch >= test_data->TotalBatch(params_.batch_size)) {
      // Move to next shard
      test_data = data_loader->MoveToNextDataSet();
      current_batch = 0;
    }
  }

  // Call after a test batch.
  if (params_.post_evaluation_callback) {
    params_.post_evaluation_callback(evaluation_batch_size, step_, "test");
  }

  return Status::OK();
}

Status TrainingRunner::LoadAndEvaluate(const std::string& model_path, std::shared_ptr<IDataLoader> data_loader) {
  InferenceSession s{SessionOptions()};
#ifdef USE_CUDA
  CUDAExecutionProviderInfo xp_info{params_.mpi_context.world_rank};
  ORT_RETURN_IF_ERROR(s.RegisterExecutionProvider(std::make_unique<CUDAExecutionProvider>(xp_info)));
#endif
  ORT_RETURN_IF_ERROR(s.Load(model_path));
  ORT_RETURN_IF_ERROR(s.Initialize());
  return Evaluate(s, data_loader);
}

Status TrainingRunner::SetupOptimizerParams(const std::unordered_set<std::string>& weights_to_train,
                                            const std::unordered_map<std::string, NodeArg*>& fp16_weights_map,
                                            const std::string& loss_scale_input_name,
                                            OptimizerGraphConfig& opt_graph_config_result,
                                            std::unordered_map<std::string, OptimizerNodeConfig>& opt_configs) {
  opt_configs.reserve(weights_to_train.size());
  for (const auto& weight_name : weights_to_train) {
    // Prepare the weight<->optimizer mapping.
    // All weights use the same type of optimizer
    OptimizerNodeConfig opt_config{
        params_.training_optimizer_name,
        nullptr,
        params_.lr_params.feed_name,
        params_.optimizer_attributes(weight_name),
        params_.use_fp16_moments};

    const auto it = fp16_weights_map.find(weight_name);
    if (it != fp16_weights_map.cend()) {
      opt_config.fp16_weight_arg = it->second;
    }

    opt_configs[weight_name] = opt_config;
  }

  // set up optimizer graph config
  OptimizerGraphConfig opt_graph_config{};
  opt_graph_config.use_mixed_precision = params_.use_mixed_precision;
  opt_graph_config.always_do_update = params_.is_perf_test;
  opt_graph_config.loss_scale_input_name = loss_scale_input_name;
  opt_graph_config.world_rank = params_.mpi_context.world_rank;
  opt_graph_config.world_size = params_.mpi_context.world_size;
  opt_graph_config.gradient_accumulation_steps = params_.gradient_accumulation_steps;
  opt_graph_config.allreduce_in_fp16 = params_.allreduce_in_fp16;

  opt_graph_config_result = std::move(opt_graph_config);

  return Status::OK();
}

Status TrainingRunner::UpdateParams(Parameters params) {
  params_.lr_params.initial_lr = params.lr_params.initial_lr;
  params_.lr_params.warmup_ratio = params.lr_params.warmup_ratio;
  params_.num_train_steps = params.num_train_steps;
  params_.batch_size = params.batch_size;
  params_.gradient_accumulation_steps = params.gradient_accumulation_steps;
  loss_scaler_->Reset();
  return Status::OK();
}
}  // namespace training
}  // namespace onnxruntime
