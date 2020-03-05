// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <utility>
#include <vector>

#include "core/common/path_string.h"
#include "core/framework/ml_value.h"
#include "orttraining/core/framework/checkpoint_registry.h"
#include "orttraining/core/framework/mpi_setup.h"
#include "orttraining/core/graph/optimizer_config.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/models/runner/data_loader.h"

namespace onnxruntime {
namespace training {
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

    // Use CUDA providers or not.
    // TODO: support a list of providers.
    bool use_cuda = false;
    // Whether to use NCCL for distributed training.
    bool use_nccl = false;
    // Whether to partition the optimizer state across nodes for distributed training.
    bool partition_optimizer = false;
    // Use Adasum for allreduce
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

    AdasumReductionType GetAdasumReductionType() const {
      // TODO support more algos when they become available.
      if (!use_adasum) {
        return AdasumReductionType::None;
      } else if (!use_cuda) {
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
  };

  TrainingRunner(Parameters params);

  common::Status Initialize();

  common::Status Run(IDataLoader* training_data_loader, IDataLoader* test_data_loader);

  common::Status EndTraining(IDataLoader* data_loader, bool do_load_and_evaluate);

  common::Status UpdateParams(Parameters params);

  common::Status ResetLossScaler();

  size_t GetRound() const { return round_; }

 private:
  Status TrainingLoop(IDataLoader& training_data_loader, IDataLoader* test_data_loader);
  Status Evaluate(InferenceSession& session, IDataLoader& data_loader);
  Status LoadAndEvaluate(const PathString& model_path, IDataLoader& data_loader);

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
  TrainingSession session_;
  AllocatorPtr pinned_allocator_;

  std::unique_ptr<CheckpointRegistry> checkpoint_registry_;
};

}  // namespace training
}  // namespace onnxruntime
