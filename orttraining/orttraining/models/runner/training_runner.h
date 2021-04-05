// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <utility>
#include <vector>

#include "core/common/path_string.h"
#include "core/framework/ml_value.h"
#include "core/providers/providers.h"
#include "orttraining/core/framework/checkpoint_registry.h"
#include "orttraining/core/framework/communication/mpi/mpi_context.h"
#include "orttraining/core/framework/pipeline.h"
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
    PathString pipeline_partitioned_model_path;  // To save the model after pipeline partition. Note: in the pipeline case,
                                                 // different ranks may resident in the same node. This could lead to a
                                                 // potential write conflict. It is user's responsibility to make sure
                                                 // different rank is passed in with different pipeline_partitioned_model_path value.

    PathString train_data_dir;
    PathString test_data_dir;
    PathString output_dir;       // Output of training, e.g., trained model files.
    PathString perf_output_dir;  // training perf metrics
    std::string model_type;      // bert/gpt2/...

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
    ZeROConfig deepspeed_zero{};
    // Use Adasum for allreduce.
    bool enable_adasum = false;
    // Use Gist on CPU.
    bool use_gist = false;
    // Whether we collect execution profile trace during this run.
    bool use_profiler = false;
    bool skip_evaluation = false;
    bool dump_fetches = false;
    bool dump_convergence_metrics = false;

    VectorString fetch_names;

    bool use_mixed_precision = false;
    bool use_bfloat16 = false;
    float loss_scale = 1.0f;
    bool use_mixed_precision_moments = false;
    bool use_mixed_precision_initializer = true;
    bool allreduce_in_mixed_precision_type = false;
    bool layernorm_stash_as_fp32 = true;

    // Tensorboard configuration.
    PathString log_dir;  // Path to write Tensorboard events to.
    std::string summary_name = "summary";
    VectorString scalar_names;
    VectorString histogram_names;
    VectorString norm_names;

    //Default value is -1.0f. When gpu_mem_limit_in_gb < 0, ORT can use all cuda memory available.
    float gpu_mem_limit_in_gb = -1.0f;

    bool EnableTensorboard() const {
      return !is_perf_test && !log_dir.empty() && MPIContext::GetInstance().GetWorldRank() == 0;
    }

    bool UseCuda() const {
      return providers.find(kCudaExecutionProvider) != providers.end() ||
             providers.find(kRocmExecutionProvider) != providers.end();
    }

    AdasumReductionType GetAdasumReductionType() const {
      // TODO support more algos when they become available.
      if (!enable_adasum) {
        return AdasumReductionType::None;
      } else if (!UseCuda()) {
        return AdasumReductionType::CpuReduction;
      } else {
        return AdasumReductionType::GpuHierarchicalReduction;
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
    // pipeline_parallel_size > 1 means pipeline is enabled.
    // pipeline_parallel_size == 1 means pipeline is disabled.
    int pipeline_parallel_size = 1;
    // pipeline partition information to do online-partition. If the graph is
    // pre-partitioned, no need to fill this value.
    std::vector<TrainingSession::TrainingConfiguration::CutInfo> pipeline_partition_cut_list;
    // Alternative for partition. We map each operator's string identifier to
    // a stage identifier. We identify operators using the name of any of
    // their outputs. All operators in the graph must be in the domain of this
    // map.
    // For example, op_id_to_stage["MatMul0"] being 5 means the operator node
    // called "MatMul0" locates on the 6th stage. Note that stage ID is 0-based
    // index.
    std::map<std::string, int> op_id_to_stage;

    // model_paths[i] is the name of the pipeline stage for i-th process.
    // The i-th file is run by the i-th MPI rank.
    // If model_paths is not empty, model partition transformation may not be internally invoked.
    VectorString pipeline_stage_paths;
    // Enable gradient clipping.
    bool enable_grad_norm_clip = true;

    // Enable GELU approximation
    bool enable_gelu_approximation = false;
    // Enable checkpointing of attention dropout to save memory
    bool attn_dropout_recompute = false;
    // Enable checkpointing of Gelu activation output to save memory
    bool gelu_recompute = false;
    // Enable checkpointing of transformer layer output to save memory
    bool transformer_layer_recompute = false;
    // Number of layers to apply recompute
    int number_recompute_layers = 0;
    // Use invertible layernorm grad
    bool use_invertible_layernorm_grad = false;
  };

  TrainingRunner(Parameters params, const Environment& env);
  TrainingRunner(Parameters params, const Environment& env, SessionOptions session_options);

  common::Status Initialize();

  common::Status Run(IDataLoader* training_data_loader, IDataLoader* test_data_loader,
                     const MapStringToString& mapped_dimensions = {});

  common::Status EndTraining(IDataLoader* data_loader);

  common::Status UpdateParams(Parameters params);

  common::Status ResetLossScaler();

  size_t GetRound() const { return round_; }
  TrainingSession& GetSession() { return session_; }

 private:
  enum SessionMode : int { ModelUpdateStep,
                           GradientAccumulateStep,
                           EvaluateStep };
  Status PrepareFeedNamesAndFeeds(const SessionMode mode,
                                  IDataLoader& training_data_loader,
                                  DataSet& training_data,
                                  LearningRateScheduler* lr_scheduler,
                                  const size_t batch_index,
                                  std::vector<std::string>& feed_names,
                                  std::vector<MLValue>& feeds);
  Status PrepareFetchNamesAndFetches(const SessionMode mode,
                                     std::vector<std::string>& fetch_names,
                                     std::vector<MLValue>& fetches);
  void RunWithUpdate(VectorString& feed_names,
                     VectorString& fetch_names,
                     std::vector<MLValue>& feeds,
                     std::vector<MLValue>& fetches);
  void RunWithoutUpdate(VectorString& feed_names,
                        VectorString& fetch_names,
                        std::vector<MLValue>& feeds,
                        size_t& gradient_accumulation_step_count);
  void CheckWorkerException(const std::exception_ptr& p);
  Status TrainingLoop(IDataLoader& training_data_loader, IDataLoader* test_data_loader,
    const MapStringToString& mapped_dimensions);
  Status Evaluate(TrainingSession& session, IDataLoader& data_loader);

  Status SaveCheckpoint(const PathString& checkpoint_path);
  Status LoadCheckpoint(const PathString& checkpoint_path);
  Status SaveCheckpointProperties(std::unordered_map<std::string, std::string>& properties) const;
  Status LoadCheckpointProperties(const std::unordered_map<std::string, std::string>& properties);

  Status SavePerfMetrics(const size_t number_of_batches, const size_t gradient_accumulation_steps,
                         const size_t weight_update_steps, const double total_time,
                         const double avg_time_per_batch, const double throughput, const double stabilized_throughput,
                         const double e2e_throughput, const MapStringToString& mapped_dimensions,
                         const short average_cpu_usage, const size_t peak_workingset_size);

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

  // Pipeline fields are valid only if params_.pipeline_parallel_size > 1.
  // Information for running pipeline.
  pipeline::PipelineContext pipeline_context_;
  // Pipeline schedule for deciding when to run batch, forward, or backward.
  pipeline::PipelineScheduler pipeline_schedule_;
  // Workers to run pipeline stage.
  pipeline::PipelineWorkerPool pipeline_worker_pool_;
};

}  // namespace training
}  // namespace onnxruntime
