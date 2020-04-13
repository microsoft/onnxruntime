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

#include <thread>
#include <string>
#include <fstream>
#include "orttraining/training_ops/cpu/controlflow/event_pool.h"  // TODO: move with PipelineBatchPlanner

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

struct PipelineBatchInfo {
  // Event pairs for each pipeline slot to WaitEvent when start, and RecordEvent when end
  std::vector<std::pair<int64_t, int64_t>> events;
  // indices of retired batches, so their data could be reused
  // a batch can only be retired after finished backward in stage 0
  // this can be used to join worker threads or reuse buffers
  // for example, in a node with N GPUs and B batches to run in pipeline (with one stage for each GPU)
  // there will be (N * B) threads created, and by being able to retire,
  // only at most (N * (2 * N - 1)) concurrent threads are needed
  // for small number of B, there's no retired threads so total count would be the same.
  // for big number of B, this would be helpful
  std::vector<int64_t> retired_batches;
};

class PipelineTimeline {
 public:
  struct Slot {
    enum class Type {
      Unused,
      Forward,
      Backward
    };
    Type type;
    size_t batch_id;

    Slot() : type(Type::Unused) {}
  };

  PipelineTimeline() = default;

  void Initialize(size_t num_stages, size_t num_slots) {
    slots_.resize(num_stages);
    for (size_t s = 0; s < num_stages; ++s) {
      slots_[s].resize(num_slots);
    }
  }

  bool IsOccupied(size_t s, size_t t) const {
    return slots_[s][t].type != Slot::Type::Unused;
  }

  const Slot& Get(size_t s, size_t t) const {
    return slots_[s][t];
  }

  size_t GetNumSlots() const {
    return slots_[0].size();
  }

  void Occupy(size_t s, size_t t, size_t batch_id, Slot::Type st) {
    Slot& slot = slots_[s][t];
    ORT_ENFORCE(slot.type == Slot::Type::Unused);
    slot.type = st;
    slot.batch_id = batch_id;
  }

 private:
  std::vector<std::vector<Slot>> slots_;
};

// pipeline planner for batches
class PipelineBatchPlanner {
 private:
  int64_t max_id_;
  PipelineTimeline timeline_;

 public:
  PipelineBatchPlanner()
      : max_id_(::onnxruntime::contrib::OrtEventPool::GetPoolSize() - 1) {
  }

  // Generate timeline for one-forward-one-backward scheduling,
  // which schedules execution in batch order to minimize latency for onging batches
  // each stage requires 2 pair of wait/record events for FW/BW
  void GenerateOneFWOneBWTimeline(size_t num_stages, size_t num_batches) {
    // The first batch has 2 * (num_stages - 1) gaps between FW and BW
    // then 2 slots for FW/BW in each batch
    size_t num_slots = 2 * (num_stages - 1) + num_batches * 2;
    timeline_.Initialize(num_stages, num_slots);

    // fw time slot to start the search for empty ones in each stage
    std::vector<size_t> t_fw(num_stages, 0);
    // bw time slot to start the search for empty ones in each stage
    std::vector<size_t> t_bw(num_stages, 0);

    // generate timeline in batch order to minimize latency for ongoing batches
    for (size_t batch_id = 0; batch_id < num_batches; ++batch_id) {
      // plan for FW
      for (size_t s = 0; s < num_stages; ++s) {
        while (timeline_.IsOccupied(s, t_fw[s])) {
          ++t_fw[s];
        }
        // after find a slot, update t[s+1..] if needed
        for (size_t ss = s + 1; ss < num_stages; ++ss) {
          t_fw[ss] = std::max(t_fw[ss], t_fw[s] + (ss - s));
        }
        // occupy slot in timeline
        timeline_.Occupy(s, t_fw[s]++, batch_id, PipelineTimeline::Slot::Type::Forward);
      }
      // plan for BW
      for (int s = gsl::narrow<int>(num_stages - 1); s >= 0; --s) {
        t_bw[s] = std::max(t_fw[s], t_bw[s]);
        while (timeline_.IsOccupied(s, t_bw[s])) {
          t_bw[s]++;
        }
        // after find a slot, update t_bw[s-1..]
        for (int ss = s - 1; ss >= 0; --ss) {
          t_bw[ss] = std::max(t_bw[ss], t_bw[s] + (s - ss));
        }
        // occupy slot in timeline
        timeline_.Occupy(s, t_bw[s], batch_id, PipelineTimeline::Slot::Type::Backward);
      }
    }
  }

  // create pipeline plans according to generated timeline
  // with start_event_id = s, the output of each stage is [-1, s], [s, s+1], [s+1, s+2]... for each occupied slot
  // and will be assigned to each batch's PipelineBatchInfo
  // returns the first unused event_id after creating
  int64_t CreatePlan(int64_t start_event_id, const size_t stage, std::vector<PipelineBatchInfo>& plan) {
    // fill in plan
    int64_t prev_event_id = -1;
    int64_t event_id = start_event_id;
    std::vector<int64_t> retired_batches;
    for (size_t t = 0; t < timeline_.GetNumSlots(); ++t) {
      if (!timeline_.IsOccupied(stage, t))
        continue;

      const auto& slot = timeline_.Get(stage, t);
      ORT_ENFORCE(event_id < max_id_);
      if (stage == 0) {
        if (slot.type == PipelineTimeline::Slot::Type::Forward) {
          // set retired batches when starting a new batch (s == 0 && !bw)
          plan[slot.batch_id].retired_batches = retired_batches;
          retired_batches.clear();
        } else if (slot.type == PipelineTimeline::Slot::Type::Backward) {
          // add to retired batches after backward of stage 0
          retired_batches.push_back(gsl::narrow<int64_t>(slot.batch_id));
        }
      }
      // add a pair of wait/record event ids to given batch_id
      plan[slot.batch_id].events.emplace_back(prev_event_id, event_id);
      prev_event_id = event_id;
      ++event_id;
    }
    return event_id;
  }
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
    const size_t pipeline_batch_id = step_id % num_gradient_accumulation_steps_;
    if (pipeline_batch_id < num_pipeline_batches_) {
      // A gradient accumulation step.
      if (pipeline_stage_id_ < num_pipeline_stages_) {
        // Non-last stage event. 
        return plan_[pipeline_batch_id].events[0].first;
      } else {
        // Last stage event. 
        return plan_[pipeline_batch_id].events[0].first;
      }
    } else {
      // A update step.
      return -1;
    }
  }

  int64_t get_forward_recorded_event_id(size_t step_id) {
    const size_t pipeline_batch_id = step_id % num_gradient_accumulation_steps_;
    if (pipeline_batch_id < num_pipeline_batches_) {
      // A gradient accumulation step.
      if (pipeline_stage_id_ < num_pipeline_stages_) {
        // Non-last stage event. 
        return plan_[pipeline_batch_id].events[0].second;
      } else {
        // Last stage event. 
        return -1;
      }
    } else {
      // A update step.
      return -1;
    }
  }

  int64_t get_backward_waited_event_id(size_t step_id) {
    const size_t pipeline_batch_id = step_id % num_gradient_accumulation_steps_;
    if (pipeline_batch_id < num_pipeline_batches_ - 1) {
      // Non-last stage event. 
      return plan_[pipeline_batch_id].events[1].first;
    } else if(pipeline_batch_id == num_pipeline_batches_ - 1) {
      // Last stage event.
      return plan_[pipeline_batch_id].events[1].second;
    } else {
      // No event to wait.
      return -1;
    }
  }
  
  int64_t get_backward_recorded_event_id(size_t step_id) {
    const size_t pipeline_batch_id = step_id % num_gradient_accumulation_steps_;
    if (pipeline_batch_id < num_pipeline_batches_ - 1) {
      // Non-last stage event. 
      return plan_[pipeline_batch_id].events[1].second;
    } else if(pipeline_batch_id == num_pipeline_batches_ - 1) {
      // Last stage event.
      return -1;
    } else {
      // No event to wait.
      return -1;
    }
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
  TrainingSession session1_;
  AllocatorPtr input_allocator_;

  std::unique_ptr<CheckpointRegistry> checkpoint_registry_;

  std::vector<std::thread> workers_;
  std::vector<WorkerState> worker_states_;
  std::string waited_forward_event_name_;
  std::string recorded_forward_event_name_;
  std::string waited_backward_event_name_;
  std::string recorded_backward_event_name_;
  bool do_pipedream_;

  size_t num_pipeline_stages_;
  size_t pipeline_stage_id_;
  size_t num_pipeline_batches_;
  size_t num_gradient_accumulation_steps_;
  std::vector<PipelineBatchInfo> plan_;
  PipelineBatchPlanner planner_;;
};


}  // namespace training
}  // namespace onnxruntime
