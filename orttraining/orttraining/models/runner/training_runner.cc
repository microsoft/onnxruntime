// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/models/runner/training_runner.h"

#include <algorithm>
#include <memory>
#include <sstream>

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/path_lib.h"
#include "core/platform/env.h"
#include "core/session/environment.h"
#include "orttraining/core/framework/optimizer_graph_builder.h"
#include "orttraining/models/runner/training_util.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_execution_provider.h"
#endif

using namespace std;

namespace onnxruntime {
namespace training {

static std::vector<FreeDimensionOverride> overrides = {};
static SessionOptions SESSION_OPTION = {
    ExecutionMode::ORT_SEQUENTIAL,     //execution_mode
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
    0,                                 //intra_op_num_threads
    0,                                 //inter_op_num_threads
    overrides                          //free_dimension_overrides
};

TrainingRunner::TrainingRunner(Parameters params)
    : step_(0),
      round_(0),
      weight_update_step_count_(0),
      training_data_set_index_(0),
      params_(params),
      session_(SESSION_OPTION),
      pinned_allocator_(nullptr) {
  ORT_ENFORCE(!params_.model_path.empty());
  if (!params.weights_to_train.empty())
    ORT_ENFORCE(params.weights_not_to_train.empty());
  ORT_ENFORCE(!params_.training_optimizer_name.empty());
  if (params.partition_optimizer)
    ORT_ENFORCE(params.use_nccl, "Optimizer partitioning is only supported with NCCL distributed training.");
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
      loss_scaler_ = onnxruntime::make_unique<LossScaler>(loss_scale_input_name, true, static_cast<float>(1 << 16));
    } else {
      // use static loss_scale
      loss_scaler_ = onnxruntime::make_unique<LossScaler>(loss_scale_input_name, false, params_.loss_scale);
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
    if (opt_graph_outputs_.count(kGradientAllIsFiniteOutputKey) > 0) {
      params_.scalar_names.push_back(opt_graph_outputs_[kGradientAllIsFiniteOutputKey]);
    }
    if (opt_graph_outputs_.count(kGlobalGradientNormOutputKey) > 0) {
      params_.scalar_names.push_back(opt_graph_outputs_[kGlobalGradientNormOutputKey]);
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
    if (!params_.model_gist_encode_path.empty()) {
      session_.Save(params_.model_gist_encode_path, TrainingSession::SaveOption::NO_RELOAD);
    }
  }

#ifdef USE_CUDA
  if (params_.use_cuda) {
    CUDAExecutionProviderInfo xp_info{params_.mpi_context.local_rank};
    auto cuda_xp = onnxruntime::make_unique<CUDAExecutionProvider>(xp_info);
    pinned_allocator_ = cuda_xp->GetAllocator(0, OrtMemTypeCPUOutput);
    if (params_.cuda_mem_limit_in_gb > 0)
      ORT_RETURN_IF_ERROR(session_.RegisterExecutionProvider(onnxruntime::make_unique<CUDAExecutionProvider>(xp_info, false, (size_t)(params_.cuda_mem_limit_in_gb * 1024 * 1024 * 1024))));
    else
      ORT_RETURN_IF_ERROR(session_.RegisterExecutionProvider(onnxruntime::make_unique<CUDAExecutionProvider>(xp_info)));
  }
#endif
  ORT_RETURN_IF_ERROR(session_.UpdateTrainableWeightsInfoInGraph());

  // Checkpointing initialization
  if (!params_.checkpoints_dir.empty()) {
    ORT_RETURN_IF_ERROR(Env::Default().CreateFolder(params_.checkpoints_dir));

    checkpoint_registry_ = onnxruntime::make_unique<CheckpointRegistry>(
        params_.checkpoints_dir, params_.max_num_checkpoints);

    // Load checkpoint, if any
    PathString checkpoint_to_load_path = params_.checkpoint_to_load_path;
    if (!checkpoint_to_load_path.empty() ||
        checkpoint_registry_->TryGetLatestCheckpoint(checkpoint_to_load_path)) {
      std::unordered_map<std::string, std::string> checkpoint_properties{};
      ORT_RETURN_IF_ERROR(session_.LoadCheckpointAndUpdateInitializedTensors(
          checkpoint_to_load_path, checkpoint_properties));
      ORT_RETURN_IF_ERROR(LoadCheckpointProperties(checkpoint_properties));
    }
  }

  // Create output directory if needed.
  if (!params_.output_dir.empty()) {
    ORT_RETURN_IF_ERROR(Env::Default().CreateFolder(params_.output_dir));
  }

  if (params_.use_profiler && !SESSION_OPTION.enable_profiling) {
    // Profiling has not already been enabled, so override from command line options.

    if (params_.max_profile_records > 0) {
      profiling::Profiler::max_num_events_ = params_.max_profile_records;
    }

    session_.StartProfiling(SESSION_OPTION.profile_file_prefix);
  }

  return session_.Initialize();
}

Status TrainingRunner::Run(IDataLoader* training_data_loader, IDataLoader* test_data_loader) {
  if (params_.mpi_context.world_rank == 0 && !params_.model_actual_running_graph_path.empty()) {
    session_.Save(params_.model_actual_running_graph_path, TrainingSession::SaveOption::NO_RELOAD);
  }

  // maybe in the future we can support an evaluation-only run
  if (!training_data_loader) {
    LOGS_DEFAULT(WARNING) << "training data loader not provided, nothing to do";
    return Status::OK();
  }

  ORT_RETURN_IF_ERROR(TrainingLoop(*training_data_loader, test_data_loader));

  // after successful Run(), update counters
  round_++;
  step_ = 0;

  return Status::OK();
}

Status TrainingRunner::TrainingLoop(IDataLoader& training_data_loader, IDataLoader* test_data_loader) {
  const bool enable_checkpoint_saving =
      params_.mpi_context.world_rank == 0 &&
      checkpoint_registry_ && params_.checkpoint_period > 0;

  VectorString feed_names = training_data_loader.DataSetTensorNames();
  if (loss_scaler_) {
    feed_names.push_back(loss_scaler_->GetLossScaleInputName());
  }
  feed_names.push_back(params_.lr_params.feed_name);

  OrtValue loss_scale_val;
  OrtValue lr_ort_val;

  VectorString fetch_names = params_.fetch_names;
  if (params_.use_mixed_precision) {
    auto it = opt_graph_outputs_.find(kGradientAllIsFiniteOutputKey);
    ORT_RETURN_IF(it == opt_graph_outputs_.end(), "Gradient norm's IsFinite output is missing in the optimizer output");
    fetch_names.push_back(it->second);
  }

  VectorString fetch_grad_accumulator_output;
  if (params_.gradient_accumulation_steps > 1) {
    auto it = opt_graph_outputs_.find(kGradientAccumulationOutputKey);
    ORT_RETURN_IF(it == opt_graph_outputs_.end(), "Gradient accumulation output is missing in the optimizer output");
    fetch_grad_accumulator_output.push_back(it->second);
  }

  if (test_data_loader) {
    ORT_RETURN_IF_ERROR(test_data_loader->InitializeDataSetIndex(0));
  }
  ORT_RETURN_IF_ERROR(training_data_loader.InitializeDataSetIndex(training_data_set_index_));

  const size_t num_shards_to_visit = training_data_loader.NumShards();
  const auto lr_scheduler = LearningRateScheduler::Create(params_.lr_params, params_.num_train_steps);

  double total_time{0};
  size_t epoch = 0;  // Note: epoch is not set properly when loaded from a checkpoint, but it's only for display.
  size_t gradient_accumulation_step_count = 0;
  const auto step_start = step_;
  const auto weight_update_step_count_start = weight_update_step_count_;

  while (step_ < params_.num_train_steps) {
    for (size_t shard_it = 0; shard_it < num_shards_to_visit; ++shard_it) {
      auto training_data = training_data_loader.CurrentDataSet();
      training_data_set_index_ = training_data_loader.CurrentDataSetIndex();
      if (training_data == nullptr) {
        printf("Skipping shard at index %d, which failed to load.\n",
               static_cast<int>(training_data_loader.CurrentDataSetIndex()));
        training_data_loader.MoveToNextDataSet();
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
        std::vector<MLValue> feeds = training_data->GetKthBatch(params_.batch_size, batch, pinned_allocator_);
        if (loss_scaler_) {
          float loss_scale = loss_scaler_->GetLossScale();
          TrainingUtil::CreateCpuMLValue({1}, std::vector<float>{loss_scale}, &loss_scale_val, pinned_allocator_);
          feeds.push_back(loss_scale_val);
        }

        {
          float learning_rate = lr_scheduler->GetLearningRate(step_ + 1);
          TrainingUtil::CreateCpuMLValue({1}, std::vector<float>{learning_rate}, &lr_ort_val, pinned_allocator_);
          feeds.push_back(lr_ort_val);
        }

        vector<MLValue> fetches;

        const bool is_weight_update_step = (step_ + 1) % params_.gradient_accumulation_steps == 0;
        auto start = std::chrono::high_resolution_clock::now();

        if (is_weight_update_step) {
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
               params_.batch_size * (step_ - step_start) / total_time);
        printf("Training data range: [%d - %d)\n",
               static_cast<int>(batch * params_.batch_size),
               static_cast<int>((batch + 1) * params_.batch_size - 1));

        if (test_data_loader &&
            params_.do_eval && step_ % params_.evaluation_period == 0) {
          ORT_RETURN_IF_ERROR(Evaluate(session_, *test_data_loader));
        }

        if (enable_checkpoint_saving && is_weight_update_step &&
            weight_update_step_count_ % params_.checkpoint_period == 0) {
          PathString new_checkpoint_path, old_checkpoint_path;
          bool should_remove_old_checkpoint;

          ORT_RETURN_IF_ERROR(checkpoint_registry_->AddCheckpoint(
              weight_update_step_count_, new_checkpoint_path,
              should_remove_old_checkpoint, old_checkpoint_path));

          if (should_remove_old_checkpoint) {
            const auto status = Env::Default().DeleteFolder(old_checkpoint_path);
            LOGS_DEFAULT_IF(!status.IsOK(), WARNING)
                << "Failed to delete old checkpoint. "
                << "Path: " << ToMBString(old_checkpoint_path)
                << ", error: " << status.ErrorMessage();
          }

          std::unordered_map<std::string, std::string> checkpoint_properties{};
          ORT_RETURN_IF_ERROR(SaveCheckpointProperties(checkpoint_properties));
          ORT_RETURN_IF_ERROR(session_.SaveCheckpoint(new_checkpoint_path, checkpoint_properties));
        }
      }  // end of one file/shard

      if (step_ < params_.num_train_steps) {
        training_data_loader.MoveToNextDataSet();
      }
    }  // end of one epoch

    epoch++;
  }

  std::cout << "Round: " << round_ << "\n"
            << "Batch size: " << params_.batch_size << "\n"
            << "Number of Batches: " << (step_ - step_start) << "\n"
            << "Gradient Accumulation Steps: " << gradient_accumulation_step_count << "\n"
            << "Weight Update Steps: " << (weight_update_step_count_ - weight_update_step_count_start) << "\n"
            << "Total Running Time: " << total_time << " Seconds \n"
            << "Average Running Time Per Batch: " << total_time / (step_ - step_start) * 1000 << " ms\n"
            << "Throughput: " << params_.batch_size * (step_ - step_start) / total_time << " Examples / Second\n";
  return Status::OK();
}

Status TrainingRunner::EndTraining(IDataLoader* data_loader, bool do_load_and_evaluate) {
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

  if (data_loader) {
    // Test the in-memory model before saving.
    printf("\nEvaluating the final model on the test set.\n");
    ORT_RETURN_IF_ERROR(Evaluate(session_, *data_loader));
  }

  if (params_.output_dir.empty()) {
    printf("No output directory specified, skipping save of trained model.\n");
    return Status::OK();
  }

  printf("\nSaving the trained model.\n");
  const PathString model_base_name = GetLastComponent(params_.model_path);

  const PathString trained_model_path =
      params_.output_dir + GetPathSep<PathChar>() + model_base_name + ORT_TSTR("_trained.onnx");
  ORT_RETURN_IF_ERROR(session_.Save(
      trained_model_path, TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS));

  const PathString trained_model_with_loss_func_path =
      params_.output_dir + GetPathSep<PathChar>() + model_base_name + ORT_TSTR("_with_cost_trained.onnx");
  ORT_RETURN_IF_ERROR(session_.Save(
      trained_model_with_loss_func_path, TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC));

  // Load and test the trained model.
  if (data_loader && do_load_and_evaluate) {
    printf("\nTesting the saved model: %s\n", ToMBString(trained_model_with_loss_func_path).c_str());
    ORT_RETURN_IF_ERROR(LoadAndEvaluate(trained_model_with_loss_func_path, *data_loader));
  }

  return Status::OK();
}

Status TrainingRunner::Evaluate(InferenceSession& session, IDataLoader& data_loader) {
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
  vector<string> feed_names = data_loader.DataSetTensorNames();
  if (loss_scaler_) {
    feed_names.push_back(loss_scaler_->GetLossScaleInputName());
  }
  feed_names.push_back(params_.lr_params.feed_name);
  auto test_data = data_loader.CurrentDataSet();
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
      test_data = data_loader.MoveToNextDataSet();
      current_batch = 0;
    }
  }

  // Call after a test batch.
  if (params_.post_evaluation_callback) {
    params_.post_evaluation_callback(evaluation_batch_size, step_, "test");
  }

  return Status::OK();
}

Status TrainingRunner::LoadAndEvaluate(const PathString& model_path, IDataLoader& data_loader) {
  InferenceSession s{SessionOptions()};
#ifdef USE_CUDA
  CUDAExecutionProviderInfo xp_info{params_.mpi_context.world_rank};
  ORT_RETURN_IF_ERROR(s.RegisterExecutionProvider(onnxruntime::make_unique<CUDAExecutionProvider>(xp_info)));
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
        loss_scale_input_name,
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
  opt_graph_config.use_nccl = params_.use_nccl;
  opt_graph_config.partition_optimizer = params_.partition_optimizer;

  opt_graph_config_result = std::move(opt_graph_config);

  return Status::OK();
}

namespace {
namespace property_names {
constexpr const char* k_step = "step";
constexpr const char* k_round = "round";
constexpr const char* k_weight_update_step = "weight_update_step";
constexpr const char* k_training_data_set_index = "training_data_set_index";
constexpr const char* k_loss_scaler_state = "loss_scaler_state";
}  // namespace property_names

template <typename T>
Status FromString(const std::string& s, T& t) {
  std::istringstream i{s};
  ORT_RETURN_IF_NOT(i >> t && i.eof());
  return Status::OK();
}
}  // namespace

Status TrainingRunner::SaveCheckpointProperties(
    std::unordered_map<std::string, std::string>& properties) const {
  auto save_property = [&properties](const char* name, auto val) {
    properties[name] = std::to_string(val);
  };

  save_property(property_names::k_step, step_);
  save_property(property_names::k_round, round_);
  save_property(property_names::k_weight_update_step, weight_update_step_count_);
  save_property(property_names::k_training_data_set_index, training_data_set_index_);

  if (loss_scaler_) {
    properties[property_names::k_loss_scaler_state] = loss_scaler_->SaveToString();
  }

  return Status::OK();
}

Status TrainingRunner::LoadCheckpointProperties(
    const std::unordered_map<std::string, std::string>& properties) {
  auto load_property = [&properties](const char* name, auto& val) {
    auto prop_it = properties.find(name);
    ORT_RETURN_IF_NOT(prop_it != properties.end());
    ORT_RETURN_IF_ERROR(FromString(prop_it->second, val));
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(load_property(property_names::k_step, step_));
  ORT_RETURN_IF_ERROR(load_property(property_names::k_round, round_));
  ORT_RETURN_IF_ERROR(load_property(
      property_names::k_weight_update_step, weight_update_step_count_));
  ORT_RETURN_IF_ERROR(load_property(
      property_names::k_training_data_set_index, training_data_set_index_));

  if (loss_scaler_) {
    auto prop_it = properties.find(property_names::k_loss_scaler_state);
    ORT_RETURN_IF_NOT(prop_it != properties.end());
    ORT_RETURN_IF_ERROR(loss_scaler_->LoadFromString(prop_it->second));
  }

  return Status::OK();
}

Status TrainingRunner::UpdateParams(Parameters params) {
  params_.lr_params.initial_lr = params.lr_params.initial_lr;
  params_.lr_params.warmup_ratio = params.lr_params.warmup_ratio;
  params_.num_train_steps = params.num_train_steps;
  params_.batch_size = params.batch_size;
  params_.gradient_accumulation_steps = params.gradient_accumulation_steps;
  return Status::OK();
}

Status TrainingRunner::ResetLossScaler() {
  if (loss_scaler_) {
    loss_scaler_->Reset();
  }
  return Status::OK();
}
}  // namespace training
}  // namespace onnxruntime
