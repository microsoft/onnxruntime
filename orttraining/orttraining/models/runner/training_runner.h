// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <utility>
#include <vector>

#include "core/common/path_string.h"
#include "core/framework/ml_value.h"
#include "core/providers/providers.h"
#include "orttraining/core/framework/checkpoint_registry.h"
#include "orttraining/core/framework/mpi_setup.h"
#include "orttraining/core/graph/optimizer_config.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/models/runner/data_loader.h"

namespace onnxruntime {
namespace training {

struct WorkerState {
  RunOptions run_options;
  std::vector<std::string> feed_names;
  std::vector<MLValue> feeds;
  std::vector<std::string> fetch_names;
  std::vector<MLValue> fetches;
  MLValue fw_waited_value;
  MLValue fw_recorded_value;
  MLValue bw_waited_value;
  MLValue bw_recorded_value;
};

class TrainingRunner {
 public:
  struct Parameters {
    std::string model_name;
    PathString model_path;
    PathString model_with_loss_func_path;        // To save the model after adding loss func.
    PathString model_with_training_graph_path;   // To save the model after adding loss func and backward graph.
    PathString model_actual_running_graph_path;  // To save the model with the actual running graph after transformations.
    PathString model_gist_encode_path;           // To save the model with gist encoding.

    PathString train_data_dir;
    PathString test_data_dir;
    PathString output_dir;  // Output of training, e.g., trained model files.

    LossFunctionInfo loss_func_info;

    // The training optimizer name
    // Every weight's gradient will be connected to an optimizer node
    // For now all to-be-trained weights use the same optimizer type.
    std::string training_optimizer_name = "SGDOptimizer";
    std::function<std::unordered_map<std::string, float>(const std::string& weight)> optimizer_attributes =
        [](const std::string&) { return std::unordered_map<std::string, float>(); };
    std::function<std::unordered_map<std::string, int64_t>(const std::string& weight)> optimizer_int_attributes =
        [](const std::string&) { return std::unordered_map<std::string, int64_t>(); };
    LearningRateParameters lr_params;
    int gradient_accumulation_steps = 1;

    // The weights to train, exclusive with weights_not_to_train_.
    std::unordered_set<std::string> weights_to_train;

    // The weights not to train. If not empty, all the initializers not in the vector will be trained.
    // exclusive with weights_to_train_.
    std::unordered_set<std::string> weights_not_to_train;

    TrainingSession::ImmutableWeights immutable_weights;

    MapStringToString input_name_map;

    bool is_perf_test;
    bool shuffle_data;
    size_t batch_size;
    size_t eval_batch_size;
    size_t num_train_steps;
    size_t evaluation_period;
    bool do_eval = false;
    size_t display_loss_steps;

    // error_function_ is called when evaluating the error for a single sample.
    std::function<void(const std::vector<std::string>& feed_names,
                       const std::vector<OrtValue>& feeds,
                       const std::vector<std::string>& fetch_names,
                       const std::vector<OrtValue>& fetches,
                       size_t step)>
        error_function;

    // post_evaluation_callback_ is called when a batch of evaluation is done.
    std::function<void(size_t /*eval_batch_size*/,
                       size_t /*step*/,
                       const std::string& /*tag*/)>
        post_evaluation_callback;

    // Allocator to use for allocating inputs from the dataset (optional).
    AllocatorPtr input_allocator;
    // List of execution providers to register.
    std::unordered_map<std::string, std::shared_ptr<IExecutionProviderFactory>> providers;
    // Whether to use NCCL for distributed training.
    bool use_nccl = false;
    // Whether to partition the optimizer state across nodes for distributed training.
    bool partition_optimizer = false;
    // Use Adasum for allreduce.
    bool use_adasum = false;
    // Use Gist on CPU.
    bool use_gist = false;
    // Whether we collect execution profile trace during this run.
    bool use_profiler = false;
    MPIContext mpi_context;
    bool skip_evaluation = false;
    bool dump_fetches = false;
    bool dump_convergence_metrics = false;

    VectorString fetch_names;

    bool use_mixed_precision = false;
    float loss_scale = 1.0f;
    bool use_fp16_moments = false;
    bool use_fp16_initializer = true;
    bool allreduce_in_fp16 = false;

    // Tensorboard configuration.
    PathString log_dir;  // Path to write Tensorboard events to.
    std::string summary_name = "summary";
    VectorString scalar_names;
    VectorString histogram_names;
    VectorString norm_names;

    //Default value is -1.0f. When cuda_mem_limit_in_gb < 0, ORT can use all cuda memory available.
    float cuda_mem_limit_in_gb = -1.0f;

    bool EnableTensorboard() const {
      return !is_perf_test && !log_dir.empty() && mpi_context.world_rank == 0;
    }

    bool UseCuda() const {
      return providers.find(kCudaExecutionProvider) != providers.end();
    }

    AdasumReductionType GetAdasumReductionType() const {
      // TODO support more algos when they become available.
      if (!use_adasum) {
        return AdasumReductionType::None;
      } else if (!UseCuda()) {
        return AdasumReductionType::CpuReduction;
      } else {
        return AdasumReductionType::GpuHierarchical;
      }
    }

    // checkpoint configuration

    // directory used for saving/loading checkpoint files
    // empty means no checkpoints are saved
    PathString checkpoints_dir;
    // path to checkpoint to load
    // if empty and checkpoints_dir contains any checkpoints, load the latest checkpoint there
    // otherwise, no checkpoint is loaded
    PathString checkpoint_to_load_path;
    // interval in weight-update steps at which to save checkpoints
    // 0 means no checkpoints are saved
    size_t checkpoint_period = 0;
    // upper limit on number of checkpoint files to keep
    size_t max_num_checkpoints = 1;

    int data_parallel_size = 1;
    int horizontal_parallel_size = 1;
    // Enable gradient clipping.
    bool enable_grad_norm_clip=true;
  };

  TrainingRunner(Parameters params, const Environment& env);
  TrainingRunner(Parameters params, const Environment& env, SessionOptions session_options);

  common::Status Initialize();

  common::Status Run(IDataLoader* training_data_loader, IDataLoader* test_data_loader);

  common::Status EndTraining(IDataLoader* data_loader);

  common::Status UpdateParams(Parameters params);

  common::Status ResetLossScaler();

  size_t GetRound() const { return round_; }

  void join_all_workers() {
    for (size_t i = 0; i < workers_.size(); ++i) {
      if (workers_[i].joinable())
        workers_[i].join();
    }
  }

  void join_worker(size_t worker_id) {
    if (workers_[worker_id].joinable()) {
      workers_[worker_id].join();
    }
  }

  int64_t get_forward_waited_event_id(size_t step_id) {
    return step_id;
  }

  int64_t get_forward_recorded_event_id(size_t step_id) {
    return step_id;
  }

  int64_t get_backward_waited_event_id(size_t step_id) {
    return step_id;
  }
  
  int64_t get_backward_recorded_event_id(size_t step_id) {
    return step_id;
  }

 private:
  Status TrainingLoop(IDataLoader& training_data_loader, IDataLoader* test_data_loader);
  Status Evaluate(InferenceSession& session, IDataLoader& data_loader);

  Status SaveCheckpoint(const PathString& checkpoint_path);
  Status LoadCheckpoint(const PathString& checkpoint_path);
  Status SaveCheckpointProperties(std::unordered_map<std::string, std::string>& properties) const;
  Status LoadCheckpointProperties(const std::unordered_map<std::string, std::string>& properties);

  size_t step_;
  size_t round_;
  size_t weight_update_step_count_;
  size_t training_data_set_index_;
  OptimizerOutputKeyMap<std::string> opt_graph_outputs_;

  std::unique_ptr<LossScaler> loss_scaler_ = nullptr;

  Parameters params_;
  const SessionOptions session_options_;
  TrainingSession session_;
  AllocatorPtr input_allocator_;

  std::unique_ptr<CheckpointRegistry> checkpoint_registry_;

  size_t num_pipeline_stages_;
  std::vector<std::thread> workers_;
  std::vector<WorkerState> worker_states_;
  std::string waited_forward_event_name_;
  std::string recorded_forward_event_name_;
  std::string waited_backward_event_name_;
  std::string recorded_backward_event_name_;
  bool do_pipedream_;
};

}  // namespace training
}  // namespace onnxruntime
