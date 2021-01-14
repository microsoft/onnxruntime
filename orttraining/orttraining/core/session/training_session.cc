﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/session/training_session.h"

#include "core/framework/data_transfer_utils.h"
#include "core/graph/model.h"
#include "core/session/IOBinding.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "orttraining/core/graph/loss_function_builder.h"
#include "orttraining/core/graph/optimizer_builder.h"
#include "orttraining/core/framework/checkpointing.h"
#include "orttraining/core/framework/gradient_graph_builder.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/core/framework/communication/mpi/mpi_context.h"
#include "orttraining/core/graph/optimizer_graph_builder_registry.h"
#include "orttraining/core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "orttraining/core/graph/mixed_precision_transformer.h"
#include "orttraining/core/graph/tensorboard_transformer.h"
#include "orttraining/core/graph/pipeline_transformer.h"
#include "orttraining/core/graph/gradient_builder_base.h"
#include "orttraining/core/optimizer/megatron_transformer.h"

//Gist Encoding
#include "orttraining/core/optimizer/gist_encode_decode.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_allocator.h"
#endif

namespace onnxruntime {
namespace training {

namespace {
Status SetupOptimizerParams(
    const std::unordered_set<std::string>& weight_names_to_train,
    const std::unordered_map<std::string, NodeArg*>& fp32_weight_names_to_mixed_precision_node_args,
    const optional<std::string>& loss_scale_input_name,
    const TrainingSession::TrainingConfiguration& config,
    OptimizerGraphConfig& opt_graph_config_result,
    std::unordered_map<std::string, OptimizerNodeConfig>& opt_node_configs_result,
    std::unordered_map<std::string, std::string>& weight_name_map_after_graph_transform) {
  ORT_RETURN_IF_NOT(config.optimizer_config.has_value());
  const auto& optimizer_config = config.optimizer_config.value();

  // This is the mapping from the new weight name to the original weight name
  // It is required to look up the optimizer config for the original weight
  // passed in the training session config
  std::unordered_map<std::string, std::string> reversed_weight_names_map;
  for (auto& p : weight_name_map_after_graph_transform) {
    reversed_weight_names_map.insert({p.second, p.first});
  }

  std::unordered_map<std::string, OptimizerNodeConfig> opt_node_configs{};
  for (const auto& weight_name : weight_names_to_train) {
    OptimizerNodeConfig opt_node_config{};
    opt_node_config.name = optimizer_config.name;
    opt_node_config.lr_feed_name = optimizer_config.learning_rate_input_name;
    std::string original_weight_name = weight_name;
    if (reversed_weight_names_map.find(original_weight_name) != reversed_weight_names_map.end()) {
      original_weight_name = reversed_weight_names_map.at(original_weight_name);
    }
    try {
      opt_node_config.attributes = optimizer_config.weight_attributes_generator(original_weight_name);
    } catch (const std::exception& ex) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, ex.what());
    }

    try {
      opt_node_config.int_attributes = optimizer_config.weight_int_attributes_generator(original_weight_name);
    } catch (const std::exception& ex) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, ex.what());
    }

    // TODO make OptimizerNodeConfig::loss_scale_input_name optional<string>
    opt_node_config.loss_scale_input_name =
        loss_scale_input_name.has_value() ? loss_scale_input_name.value() : "";
    opt_node_config.use_mixed_precision_moments = optimizer_config.use_mixed_precision_moments;

    const auto mixed_precision_weight_name_it = fp32_weight_names_to_mixed_precision_node_args.find(weight_name);
    if (mixed_precision_weight_name_it != fp32_weight_names_to_mixed_precision_node_args.end()) {
      opt_node_config.mixed_precision_weight_arg = mixed_precision_weight_name_it->second;
    }

    // check if initial optimizer states have been provided for weight
    if (config.init_optimizer_states) {
      const auto optim_state_it = config.init_optimizer_states->find(weight_name);
      if (optim_state_it != config.init_optimizer_states->end()) {
        opt_node_config.initial_states = optim_state_it->second;
      }
    }

    opt_node_configs.emplace(weight_name, std::move(opt_node_config));
  }

  OptimizerGraphConfig opt_graph_config{};
  opt_graph_config.use_mixed_precision = config.mixed_precision_config.has_value();
  if (opt_graph_config.use_mixed_precision) {
    opt_graph_config.mixed_precision_type = config.mixed_precision_config.value().mixed_precision_type;
  }

  // TODO make OptimizerGraphConfig::loss_scale_input_name optional<string>
  opt_graph_config.loss_scale_input_name =
      loss_scale_input_name.has_value() ? loss_scale_input_name.value() : "";
  ;
  opt_graph_config.local_size = DistributedRunContext::RunConfig().local_size;
  opt_graph_config.local_rank = DistributedRunContext::RunConfig().local_rank;
  opt_graph_config.data_parallel_group_rank = DistributedRunContext::RankInGroup(WorkerGroupType::DataParallel);
  opt_graph_config.data_parallel_group_size = DistributedRunContext::GroupSize(WorkerGroupType::DataParallel);
  opt_graph_config.gradient_accumulation_steps = config.gradient_accumulation_steps;
  opt_graph_config.allreduce_in_mixed_precision_type = optimizer_config.do_all_reduce_in_mixed_precision_type;
  opt_graph_config.use_nccl = optimizer_config.use_nccl;
  opt_graph_config.adasum_reduction_type = optimizer_config.adasum_reduction_type;
  opt_graph_config.enable_grad_norm_clip = optimizer_config.enable_grad_norm_clip;
  opt_graph_config.deepspeed_zero = optimizer_config.deepspeed_zero;

  // check if shared initial optimizer states have been provided
  if (config.init_optimizer_states) {
    const auto optim_state_it = config.init_optimizer_states->find(onnxruntime::training::SHARED_OPTIMIZER_STATES_KEY);
    if (optim_state_it != config.init_optimizer_states->end()) {
      opt_graph_config.shared_optimizer_states = std::move(optim_state_it->second);
    }
  }
  opt_node_configs_result = std::move(opt_node_configs);
  opt_graph_config_result = std::move(opt_graph_config);

  return Status::OK();
}

bool IsRootNode(const TrainingSession::TrainingConfiguration& config) {
  return config.distributed_config.world_rank == 0;
}
}  // namespace

void TrainingSession::FilterUnusedWeights(const std::unordered_set<std::string>& weight_names_to_train,
                                          std::unordered_set<std::string>& filtered_weight_names_to_train) {
  filtered_weight_names_to_train.clear();
  for (const auto& name : weight_names_to_train) {
    auto nodes = model_->MainGraph().GetConsumerNodes(name);
    if (!nodes.empty())
      filtered_weight_names_to_train.insert(name);
    else
      LOGS(*session_logger_, WARNING)
          << "Couldn't find any consumer node for weight " << name << ", exclude it from training.";
  }
}

const std::string TrainingSession::training_mode_string_ = "training_mode";

Status TrainingSession::ConfigureForTraining(
    const TrainingConfiguration& config, TrainingConfigurationResult& config_result_out) {
  ORT_RETURN_IF(
      IsInitialized(),
      "TrainingSession::ConfigureForTraining() must be called before TrainingSession::Initialize().");

  if (is_configured_) return Status::OK();

  std::unordered_set<std::string> filtered_config_weight_names_to_train;
  FilterUnusedWeights(config.weight_names_to_train, filtered_config_weight_names_to_train);

  TrainingConfigurationResult config_result{};

  ORT_ENFORCE(config.distributed_config.pipeline_parallel_size > 0,
              "This parameter should be 1 if there is no pipeline parallelism. "
              "Otherwise, it's the number of pipeline stages.");

  DistributedRunContext::CreateInstance({config.distributed_config.world_rank,
                                         config.distributed_config.world_size,
                                         config.distributed_config.local_rank,
                                         config.distributed_config.local_size,
                                         config.distributed_config.data_parallel_size,
                                         config.distributed_config.horizontal_parallel_size,
                                         config.distributed_config.pipeline_parallel_size});

#ifdef USE_MPI
  const std::vector<MPIGroup>& mpi_groups = MPIContext::GetInstance().GetAllMPIGroups();
  for (int i = 0; i < WorkerGroupType::WorkerGroupTypeCount; i++) {
    if (!mpi_groups[i].is_group_initialized && MPIContext::GetInstance().GetWorldSize() > 1) {
      MPIContext::GetInstance().AddMPIGroup(static_cast<WorkerGroupType>(i),
                                            DistributedRunContext::GetInstance().GetWorkerGroup(static_cast<WorkerGroupType>(i)));
    }
  }
#endif

  const int32_t pipeline_stage_id = config.pipeline_config.has_value() ? DistributedRunContext::RankInGroup(WorkerGroupType::ModelParallel) : -1;

  if (config.pipeline_config.has_value() && config.pipeline_config.value().do_partition) {
    // Apply online pipeline partition to graph obj. This needs to be done first before any graph
    // transportation which may alter node_arg and invalidate cut_list info from the original graph.
    ORT_ENFORCE(pipeline_stage_id >= 0, "invalid pipelie stage id (", pipeline_stage_id, ") before doing online partition.");
    int n_stages = config.distributed_config.pipeline_parallel_size;
    std::map<const Node*, int> op_to_stage;
    const auto& cut_list = config.pipeline_config.value().cut_list;
    if (cut_list.size() > 0) {
      ORT_RETURN_IF_ERROR(
          GetDeviceAssignmentMap(model_->MainGraph(), cut_list, op_to_stage, n_stages));
    } else {
      const auto& id_to_stage = config.pipeline_config.value().op_id_to_stage;
      ORT_RETURN_IF_ERROR(
          GetDeviceAssignmentMap(model_->MainGraph(), id_to_stage, op_to_stage, n_stages));
    }

    auto ranks = DistributedRunContext::GetRanks(WorkerGroupType::ModelParallel);
    ORT_RETURN_IF_ERROR(
        ApplyPipelinePartitionToMainGraph(model_->MainGraph(), op_to_stage,
                                          pipeline_stage_id, n_stages, ranks));

    if (config.pipeline_config.value().partitioned_model_path.has_value()) {
      // Save the partitioned file out.
      // To avoid writing conflict, only the ranks in first pipeline group write the partition file out.
      if (DistributedRunContext::GroupId(WorkerGroupType::ModelParallel) == 0) {
        ORT_IGNORE_RETURN_VALUE(Save(
            config.pipeline_config.value().partitioned_model_path.value(), SaveOption::NO_RELOAD));
      }
    }
  }

  is_mixed_precision_enabled_ = config.mixed_precision_config.has_value();

  std::string loss_name{};
  // Enable loss scale if mixed precision is enabled AND at pipeline last stage if pipeline is used.
  // We are currently making the assumption that no data parallelism is used together with model parallelism.
  // So we can check the last stage by checking the world_rank and world_size. Once DP and MP combination is
  // enabled, we need to devise another way to check MP stages.
  bool enable_loss_scale = is_mixed_precision_enabled_ &&
                           config.mixed_precision_config.value().mixed_precision_type == MixedPrecisionDataType::FP16 &&
                           (!config.pipeline_config.has_value() ||
                            (pipeline_stage_id + 1 == config.distributed_config.pipeline_parallel_size));
  optional<std::string> loss_scale_input_name =
      enable_loss_scale ? optional<std::string>{""} : optional<std::string>{};
  if (config.pipeline_config.has_value()) {
    // if use pipeline, first check if model contains send op. If it does, set the
    // send node's output as the start tensor to build gradient graph
    GetPipelineSendOutput(model_->MainGraph(), loss_name);
  }

  if (loss_name.empty()) {
    const optional<LossFunctionInfo> loss_function_info =
        config.loss_function_config.has_value()
            ? config.loss_function_config.value().loss_function_info
            : optional<LossFunctionInfo>{};
    ORT_RETURN_IF_ERROR(ConfigureLossFunction(
        config.loss_name, loss_function_info,
        loss_scale_input_name.has_value() ? &loss_scale_input_name.value() : nullptr, loss_name));
  }

  ORT_ENFORCE(
      !loss_scale_input_name.has_value() || !loss_scale_input_name.value().empty(),
      "loss_scale_input_name should not be set to an empty string.");

  if (enable_loss_scale) {
    TrainingConfigurationResult::MixedPrecisionConfigurationResult mp_result{};
    mp_result.loss_scale_input_name = loss_scale_input_name.value();
    config_result.mixed_precision_config_result = mp_result;
  }

  if (IsRootNode(config) && config.model_with_loss_function_path.has_value()) {
    ORT_IGNORE_RETURN_VALUE(Save(
        config.model_with_loss_function_path.value(), SaveOption::NO_RELOAD));
  }

  // We need to get trainable weights to prevent constant folding from them. This works well if trainable weights are passed from config.
  // For case we use GetTrainableModelInitializers to get trainable weights such as C++ frontend, it may get more initializers
  // than trainable weights here as it's before transformers. So the constant folding may miss some nodes we actually can fold.
  std::unordered_set<std::string> trainable_initializers =
      !filtered_config_weight_names_to_train.empty()
          ? filtered_config_weight_names_to_train
          : GetTrainableModelInitializers(config.immutable_weights, loss_name);
  if (config.weight_names_to_not_train.size() > 0) {
    LOGS(*session_logger_, INFO) << "Excluding following weights from trainable list as specified in configuration:";
    for (const auto& weight_name_to_not_train : config.weight_names_to_not_train) {
      trainable_initializers.erase(weight_name_to_not_train);
      LOGS(*session_logger_, INFO) << weight_name_to_not_train;
    }
  }

  ORT_RETURN_IF_ERROR(ApplyTransformationsToMainGraph(trainable_initializers, config.graph_transformer_config));

  ORT_RETURN_IF_ERROR(ApplyModelParallelTransformationsToMainGraph(trainable_initializers, config_result));

  weight_partition_info_ = config_result.weight_partition_info;
  
  // Save the model after graph transformations
  if (IsRootNode(config) && config.model_after_graph_transforms_path.has_value()) {
    ORT_IGNORE_RETURN_VALUE(Save(
        config.model_after_graph_transforms_path.value(), SaveOption::NO_RELOAD));
  }

  // derive actual set of weights to train
  std::unordered_set<std::string> weight_names_to_train =
      !filtered_config_weight_names_to_train.empty()
          ? trainable_initializers
          : GetTrainableModelInitializers(config.immutable_weights, loss_name);

  for (const auto& weight_name_to_not_train : config.weight_names_to_not_train) {
    if (config_result.weight_name_map_after_graph_transform.find(weight_name_to_not_train) !=
        config_result.weight_name_map_after_graph_transform.end()) {
      weight_names_to_train.erase(config_result.weight_name_map_after_graph_transform.at(weight_name_to_not_train));
    } else {
      weight_names_to_train.erase(weight_name_to_not_train);
    }
  }

  {
    std::ostringstream weight_names_stream{};
    for (const auto& weight_name : weight_names_to_train) {
      weight_names_stream << "  " << weight_name << "\n";
    }
    LOGS(*session_logger_, INFO) << "Training weights:\n"
                                 << weight_names_stream.str();
  }

  // Transform for mixed precision on forward graph.
  std::unordered_map<std::string, NodeArg*> fp32_weight_name_to_mixed_precision_node_arg{};
  if (is_mixed_precision_enabled_) {
    const auto& mixed_precision_config = config.mixed_precision_config.value();
    ORT_RETURN_IF_ERROR(EnableMixedPrecision(weight_names_to_train,
                                             mixed_precision_config,
                                             fp32_weight_name_to_mixed_precision_node_arg));
  }

  ORT_RETURN_IF_ERROR(BuildGradientGraph(
      weight_names_to_train, loss_name, config.gradient_graph_config, *session_logger_));
    
  if (IsRootNode(config) && config.model_with_gradient_graph_path.has_value()) {
    ORT_IGNORE_RETURN_VALUE(Save(
        config.model_with_gradient_graph_path.value(), SaveOption::NO_RELOAD));
  }

  if (config.pipeline_config.has_value()) {
    TrainingConfigurationResult::PipelineConfigurationResult pipeline_result{};
    ORT_RETURN_IF_ERROR(InsertPipelineOps(weight_names_to_train,
                                          pipeline_result.pipeline_tensor_names));
    // Records which which tensors can be fed into the graph.
    // It may be different than the original graph because of extra event tensors.
    for (auto& node_arg : model_->MainGraph().GetInputsIncludingInitializers()) {
      pipeline_result.feed_names.push_back(node_arg->Name());
    }
    // The following loop is for not to fetch tensors not in this pipeline stage.
    for (size_t i = 0; i < config.pipeline_config.value().fetch_names.size(); ++i) {
      auto name = config.pipeline_config.value().fetch_names[i];
      const auto* node_arg = model_->MainGraph().GetNodeArg(name);
      if (!node_arg) {
        // This pipelie stage doesn't contain this name.
        // Let's not to fetch it.
        continue;
      }
      pipeline_result.fetch_names.push_back(name);
    }
    pipeline_result.pipeline_stage_id =
        config.distributed_config.world_rank /
        (config.distributed_config.data_parallel_size * config.distributed_config.horizontal_parallel_size);
    config_result.pipeline_config_result = pipeline_result;
  }

  // All non-float tensors are not trainable. Remove those weights.
  // TODO: this is a temp workaround for removing rank tensor before adding optimizer.
  // Re-visit after we port logic for model splitting and hence know the rank tensor name.
  for (auto it = weights_to_train_.begin(); it != weights_to_train_.end();) {
    const auto* node_arg = model_->MainGraph().GetNodeArg(*it);
    ORT_RETURN_IF_NOT(node_arg, "Failed to get NodeArg with name ", *it);
    if (node_arg->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
        node_arg->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 &&
        node_arg->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16) {
      it = weights_to_train_.erase(it);
    } else {
      ++it;
    }
  }

  // add optimizer or gradient accumulation
  if (config.optimizer_config.has_value()) {
    OptimizerGraphConfig opt_graph_config{};
    std::unordered_map<std::string, OptimizerNodeConfig> opt_node_configs{};
    ORT_RETURN_IF_ERROR(SetupOptimizerParams(
        weights_to_train_, fp32_weight_name_to_mixed_precision_node_arg,
        loss_scale_input_name, config, opt_graph_config, opt_node_configs, config_result.weight_name_map_after_graph_transform));
    TrainingConfigurationResult::OptimizerConfigurationResult optimizer_config_result{};
    ORT_RETURN_IF_ERROR(BuildOptimizer(
        opt_graph_config, opt_node_configs,
        optimizer_config_result.output_key_to_graph_output_name));
    config_result.opt_config_result = optimizer_config_result;
  } else {
    if (config.gradient_accumulation_steps > 1) {
      ORT_RETURN_IF_ERROR(BuildAccumulationNode(weights_to_train_));
    }
  }

  // Set eval feed names for nodes that differ between training and inferencing.
  ORT_RETURN_IF_ERROR(SetEvalFeedNames());

  // add Tensorboard
  if (config.tensorboard_config.has_value()) {
    const auto& tensorboard_config = config.tensorboard_config.value();

    std::vector<std::string> tensorboard_scalar_names(tensorboard_config.scalar_node_names);

    if (loss_scale_input_name.has_value()) {
      tensorboard_scalar_names.emplace_back(loss_scale_input_name.value());
    }

    // add some tensors from optimizer graph outputs
    if (config_result.opt_config_result.has_value()) {
      const auto& opt_output_key_to_graph_output_name =
          config_result.opt_config_result.value().output_key_to_graph_output_name;

      auto add_opt_graph_output_by_key =
          [&tensorboard_scalar_names, &opt_output_key_to_graph_output_name](OptimizerOutputKey key) {
            const auto it = opt_output_key_to_graph_output_name.find(key);
            if (it != opt_output_key_to_graph_output_name.end()) {
              tensorboard_scalar_names.emplace_back(it->second);
            }
          };

      add_opt_graph_output_by_key(OptimizerOutputKey::GradientAllIsFinite);
      add_opt_graph_output_by_key(OptimizerOutputKey::GlobalGradientNorm);
    }

    ORT_RETURN_IF_ERROR(AddTensorboard(
        tensorboard_config.summary_name, tensorboard_scalar_names,
        tensorboard_config.histogram_node_names, tensorboard_config.norm_node_names,
        tensorboard_config.dump_convergence_metrics));
  }

  // add GIST encoding
  if (config.gist_config.has_value()) {
    ORT_RETURN_IF_ERROR(AddGistEncoding());
  }

  // If the current node is in rank0 or if the current session is running pipeline (in which case different rank would
  // store different model partition), and if model_with_training_graph_path is specified, save the model.
  // Note: in the pipeline case, different ranks may resident in the same node. This could lead to a potential write
  // conflict. It is user's responsibility to make sure different rank is passed in with different. Also, to avoid
  // writing conflict, only the ranks in first pipeline group write the partition file out.
  // model_with_training_graph_path value.
  if ((IsRootNode(config) || (config.pipeline_config.has_value() &&
                              DistributedRunContext::GroupId(WorkerGroupType::ModelParallel) == 0)) &&
      config.model_with_training_graph_path.has_value()) {
    ORT_IGNORE_RETURN_VALUE(Save(
        config.model_with_training_graph_path.value(), SaveOption::NO_RELOAD));
  }

  // After pipeline partition, we need to return the inputs allowed in this partition.
  if (config.pipeline_config.has_value()) {
    const auto& allowed_inputs = model_->MainGraph().GetInputsIncludingInitializers();
    const auto& allowed_outputs = model_->MainGraph().GetInputsIncludingInitializers();
    for (size_t i = 0; i < allowed_inputs.size(); ++i) {
      const auto name = allowed_inputs[i]->Name();
      config_result.pipeline_config_result.value().feed_names.push_back(name);
    }
    for (size_t i = 0; i < allowed_outputs.size(); ++i) {
      const auto name = allowed_outputs[i]->Name();
      config_result.pipeline_config_result.value().fetch_names.push_back(name);
    }
  }

  config_result_out = std::move(config_result);
  is_configured_ = true;

  return Status::OK();
}

static Status AddLossScaling(
    const std::string& loss_name,
    Graph& graph, std::string* loss_scale_input_name, std::string& scaled_loss_name) {
  if (!loss_scale_input_name) {
    scaled_loss_name = loss_name;
    return Status::OK();
  }

  // add node to scale loss_name by loss_scale_input_name
  GraphAugmenter::GraphDefs defs{};
  *loss_scale_input_name = graph.GenerateNodeArgName("loss_scale");
  const auto* loss_scale_input_type =
      defs.CreateTypeProto({1}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  scaled_loss_name = graph.GenerateNodeArgName("scaled_loss");
  defs.AddNodeDef(NodeDef{
      "Mul",
      {ArgDef{loss_name}, ArgDef{*loss_scale_input_name, loss_scale_input_type}},
      {ArgDef{scaled_loss_name}},
      NodeAttributes(),
      scaled_loss_name});
  defs.AddGraphInputs({*loss_scale_input_name});

  ORT_RETURN_IF_ERROR(GraphAugmenter::AugmentGraph(graph, defs));

  return Status::OK();
}

static Status ConfigureLossFunctionInternal(
    const optional<std::string>& external_loss_name,
    ILossFunction* loss_graph_builder,
    const optional<LossFunctionInfo>& loss_func_info,
    Graph& graph,
    std::string* loss_scale_input_name,
    std::string& actual_loss_name) {
  // build loss function or use external one
  ORT_RETURN_IF_NOT(
      (loss_func_info.has_value() && loss_graph_builder) ^ external_loss_name.has_value(),
      "Either loss function information should be provided or an external "
      "loss name should be given.");

  std::string unscaled_loss_name;
  if (external_loss_name.has_value()) {
    unscaled_loss_name = external_loss_name.value();
  } else {
    auto loss_function_graph_defs = (*loss_graph_builder)(graph, loss_func_info.value());
    ORT_RETURN_IF_ERROR(GraphAugmenter::AugmentGraph(graph, loss_function_graph_defs));
    unscaled_loss_name = loss_func_info.value().loss_name;
  }

  ORT_RETURN_IF_ERROR(AddLossScaling(
      unscaled_loss_name, graph, loss_scale_input_name, actual_loss_name));

  return Status::OK();
}

static Status BuildGradientGraphInternal(Graph& graph,
                                         const std::string& loss_function_output_name,
                                         const std::unordered_set<std::string>& node_arg_names_to_train,
                                         const std::unordered_map<std::string, std::string>* p_weight_to_mixed_precision_map,
                                         const GradientGraphConfiguration& gradient_graph_config,
                                         const logging::Logger& logger) {
  std::unordered_set<std::string> names_to_train;
  if (p_weight_to_mixed_precision_map != nullptr) {
    names_to_train = std::unordered_set<std::string>{};
    std::transform((*p_weight_to_mixed_precision_map).begin(), (*p_weight_to_mixed_precision_map).end(),
                   std::inserter(names_to_train, names_to_train.begin()),
                   [](auto pair) { return pair.second; });
  } else {
    names_to_train = node_arg_names_to_train;
  }
  // Compute the gradient graph def.
  // If mixed precision is enabled and use mixed precision initializers,
  // p_weight_to_mixed_precision_map will not be empty and contains arg names of mixed precision initializers,
  // in this case, the original weigth names need to be kept when resolve graph in GradientGraphBuilder::Build.
  GradientGraphBuilder grad_graph_builder(&graph,
                                          {loss_function_output_name},
                                          names_to_train,
                                          loss_function_output_name,
                                          gradient_graph_config,
                                          logger);
  return grad_graph_builder.Build(p_weight_to_mixed_precision_map != nullptr ? &node_arg_names_to_train : nullptr);
}

static Status BuildOptimizerInternal(Graph& graph,
                                     const OptimizerGraphConfig& opt_graph_config,
                                     const std::unordered_map<std::string, OptimizerNodeConfig>& opt_configs,
                                     std::unordered_set<std::string>& opt_state_initializer_names,
                                     OptimizerOutputKeyMap<std::string>& opt_graph_outputs,
                                     std::unordered_map<std::string, std::string>& updated_weight_names_map,
                                     std::unordered_map<std::string, TrainingSession::PartitionInfo>& weight_partition_info,
                                     std::unordered_map<std::string, std::unordered_map<std::string, std::string>>& weight_to_opt_mapping) {
  OptimizerBuilderRegistry& optimizer_registry = OptimizerBuilderRegistry::GetInstance();
  OptimizerGraphBuilderRegistry& optimizer_graph_registry = OptimizerGraphBuilderRegistry::GetInstance();
  std::string graph_builder_name = optimizer_graph_registry.GetNameFromConfig(opt_graph_config);
  auto optimizer_graph_builder = optimizer_graph_registry.MakeUnique(
      graph_builder_name, optimizer_registry, opt_graph_config, opt_configs, updated_weight_names_map, weight_partition_info);
  ORT_RETURN_IF_ERROR(optimizer_graph_builder->Build(
      graph, weight_to_opt_mapping, opt_graph_outputs));
  // set opt_state_initializer_names from weight_to_opt_mapping
  for (const auto& weight_set : weight_to_opt_mapping) {
    for (const auto& optimizer_name_item : weight_set.second) {
      opt_state_initializer_names.emplace(optimizer_name_item.second);
    }
  }
  return Status::OK();
}

static Status AddGradientAccumulationNodes(Graph& graph,
                                           const NodeArgNameGeneratorFn& nodearg_name_generator,
                                           const std::vector<std::string> gradient_names) {
  GraphAugmenter::GraphDefs graph_defs;
  std::vector<ArgDef> gradient_argdefs;
  ORT_RETURN_IF_ERROR(GetArgDefsFromGraph(graph, gradient_names, gradient_argdefs));
  std::vector<ArgDef> gradient_accumulation_buffers;
  gradient_accumulation_buffers.resize(gradient_argdefs.size());
  std::vector<std::string> grad_acc_outputs;
  for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
    grad_acc_outputs.push_back(
        BuildGradientAccumulationNode(
            nodearg_name_generator, gradient_argdefs[i], gradient_accumulation_buffers[i], graph_defs, false)
            .name);
  }
  return GraphAugmenter::AugmentGraph(graph, graph_defs);
}

Status TrainingSession::ApplyTransformationsToMainGraph(const std::unordered_set<std::string>& weights_to_train,
                                                        const TrainingConfiguration::GraphTransformerConfiguration& config) {
  GraphTransformerManager graph_transformation_mgr{2};
  // TODO: ideally we can just reuse the CPU EP registered with the session, but in the training session case
  // the EPs are registered after ConfigureForTraining and before Initialize is called. Hence we don't have access
  // to the registered CPU EP at this stage. Hence creating the EP here again. This is still much better than
  // creating an EP instance for every single node in ConstantFolding.
  // Create execution frame for executing constant nodes.
  std::unique_ptr<CPUExecutionProvider> cpu_execution_provider =
      onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  AddPreTrainingTransformers(*cpu_execution_provider, graph_transformation_mgr, weights_to_train, config);

  // apply transformers
  Graph& graph = model_->MainGraph();
  for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    ORT_RETURN_IF_ERROR(graph_transformation_mgr.ApplyTransformers(
        graph, static_cast<TransformerLevel>(i), *session_logger_));
  }
  return common::Status::OK();
}

// Registers all the pre transformers with transformer manager
void TrainingSession::AddPreTrainingTransformers(const IExecutionProvider& execution_provider,
                                                 GraphTransformerManager& transformer_manager,
                                                 const std::unordered_set<std::string>& weights_to_train,
                                                 const TrainingConfiguration::GraphTransformerConfiguration& config,
                                                 TransformerLevel graph_optimization_level,
                                                 const std::vector<std::string>& custom_list) {
  auto add_transformers = [&](TransformerLevel level) {
    // Generate and register transformers for level

    auto transformers_to_register = transformer_utils::GeneratePreTrainingTransformers(
        level, weights_to_train, config, execution_provider, custom_list);
    for (auto& entry : transformers_to_register) {
      transformer_manager.Register(std::move(entry), level);
    }
  };

  ORT_ENFORCE(graph_optimization_level <= TransformerLevel::MaxLevel,
              "Exceeded max transformer level. Current level is set to " +
                  std::to_string(static_cast<uint32_t>(graph_optimization_level)));

  for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    TransformerLevel level = static_cast<TransformerLevel>(i);
    if ((graph_optimization_level >= level) || !custom_list.empty()) {
      add_transformers(level);
    }
  }
}

// Registers all the predefined transformers with transformer manager
void TrainingSession::AddPredefinedTransformers(GraphTransformerManager& transformer_manager,
                                                TransformerLevel graph_optimization_level,
                                                const std::vector<std::string>& custom_list) {
  auto add_transformers = [&](TransformerLevel level) {
    // Generate and register transformers for level
    auto transformers_to_register = transformer_utils::GenerateTransformers(
        level, weights_to_train_, GetSessionOptions().free_dimension_overrides, custom_list);
    for (auto& entry : transformers_to_register) {
      transformer_manager.Register(std::move(entry), level);
    }
  };

  ORT_ENFORCE(graph_optimization_level <= TransformerLevel::MaxLevel,
              "Exceeded max transformer level. Current level is set to " +
                  std::to_string(static_cast<uint32_t>(graph_optimization_level)));

  for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    TransformerLevel level = static_cast<TransformerLevel>(i);
    if ((graph_optimization_level >= level) || !custom_list.empty()) {
      add_transformers(level);
    }
  }
}

Status TrainingSession::ApplyModelParallelTransformationsToMainGraph(std::unordered_set<std::string>& weights_to_train,
                                                                     TrainingConfigurationResult& config_result_out) {
  const auto horizontal_parallel_size = training::DistributedRunContext::GroupSize(training::WorkerGroupType::HorizontalParallel);
  if (horizontal_parallel_size == 1) {
    return common::Status::OK();
  }

  GraphTransformerManager graph_transformation_mgr{1};
  std::vector<std::unique_ptr<GraphTransformer>> transformers_to_register;
  std::unordered_set<std::string> compatible_eps = {};
  LOGS_DEFAULT(WARNING) << horizontal_parallel_size << "-way horizontal model parallel is enabled";
  transformers_to_register.emplace_back(onnxruntime::make_unique<MegatronTransformer>(
      training::DistributedRunContext::RankInGroup(training::WorkerGroupType::HorizontalParallel),
      horizontal_parallel_size, config_result_out.weight_name_map_after_graph_transform, weights_to_train,
      config_result_out.weight_partition_info, compatible_eps));

  // Generate and register transformers for level
  for (auto& entry : transformers_to_register) {
    graph_transformation_mgr.Register(std::move(entry), TransformerLevel::Level1);
  }

  // apply transformers
  Graph& graph = model_->MainGraph();
  ORT_RETURN_IF_ERROR(graph_transformation_mgr.ApplyTransformers(
      graph, TransformerLevel::Level1, *session_logger_));
  return common::Status::OK();
}

Status TrainingSession::AddGistEncoding() {
  try {
    Graph& graph = model_->MainGraph();

    auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleGistTransformer1");
    rule_transformer_L1->Register(onnxruntime::make_unique<GistEncodeDecode>());
    onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
    graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

    ORT_RETURN_IF_ERROR(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *session_logger_));
  } catch (const OnnxRuntimeException& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to add Gist Encoding:", exp.what());
  }
  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::AddTensorboard(const std::string& summary_name,
                                       const std::vector<std::string>& scalar_nodes,
                                       const std::vector<std::string>& histogram_nodes,
                                       const std::vector<std::string>& norm_nodes,
                                       const bool dump_convergence_metrics) {
  ORT_RETURN_IF_ERROR(
      TransformGraphForTensorboard(
          model_->MainGraph(), summary_name, scalar_nodes, histogram_nodes, norm_nodes, dump_convergence_metrics));
  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::InsertPipelineOps(
    const std::unordered_set<std::string>& initializer_names_to_preserve,
    pipeline::PipelineTensorNames& pipeline_tensor_names) {
  ORT_RETURN_IF_ERROR(TransformGraphForPipeline(
      model_->MainGraph(),
      initializer_names_to_preserve,
      pipeline_tensor_names));
  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::ConfigureLossFunction(
    const optional<std::string>& external_loss_name,
    const optional<LossFunctionInfo>& loss_function_info,
    std::string* loss_scale_input_name,
    std::string& actual_loss_name) {
  external_loss_name_ = external_loss_name;
  loss_function_info_ = loss_function_info;

  if (loss_function_info_.has_value()) {
    const auto& loss_function_info_value = loss_function_info_.value();
    ORT_RETURN_IF(
        loss_function_info_value.op_def.type.empty() || loss_function_info_value.loss_name.empty(),
        "loss_function_info is invalid.");

    loss_graph_builder_ = LossFunctionBuilder::Build(loss_function_info_value.op_def.type);

    ORT_RETURN_IF_NOT(loss_graph_builder_);
  }

  try {
    ORT_RETURN_IF_ERROR(ConfigureLossFunctionInternal(
        external_loss_name_, loss_graph_builder_.get(), loss_function_info_,
        model_->MainGraph(), loss_scale_input_name, actual_loss_name));
  } catch (const OnnxRuntimeException& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to add loss function:", exp.what());
  }
  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::EnableMixedPrecision(
    const std::unordered_set<std::string>& weights_to_train,
    const TrainingConfiguration::MixedPrecisionConfiguration& mixed_precision_config,
    std::unordered_map<std::string, NodeArg*>& fp32_weight_name_to_mixed_precision_node_arg) {
  ORT_RETURN_IF_ERROR(TransformGraphForMixedPrecision(
      model_->MainGraph(),
      weights_to_train,
      mixed_precision_config.use_mixed_precision_initializers,
      mixed_precision_config.TensorProtoDataType(),
      fp32_weight_name_to_mixed_precision_node_arg,
      mixed_precision_config.layernorm_stash_as_fp32));

  std::unordered_map<std::string, std::string> weight_to_mixed_precision_map{};
  std::transform(
      weights_to_train.cbegin(), weights_to_train.cend(),
      std::inserter(weight_to_mixed_precision_map, weight_to_mixed_precision_map.begin()),
      [&fp32_weight_name_to_mixed_precision_node_arg](const std::string& name) {
        return fp32_weight_name_to_mixed_precision_node_arg.find(name) != fp32_weight_name_to_mixed_precision_node_arg.end() ? std::make_pair(name, fp32_weight_name_to_mixed_precision_node_arg[name]->Name()) : std::make_pair(name, name);
      });
  weight_to_mixed_precision_map_ = std::move(weight_to_mixed_precision_map);

  return Status::OK();
}

Status TrainingSession::BuildGradientGraph(const std::unordered_set<std::string>& weights_to_train,
                                           const std::string& loss_function_output_name,
                                           const GradientGraphConfiguration& gradient_graph_config,
                                           const logging::Logger& logger) {
  // Fill weights_to_train_ according to weights_to_train
  weights_to_train_ = weights_to_train;
  gradient_graph_config_ = gradient_graph_config;

  ORT_RETURN_IF_ERROR(BuildGradientGraphInternal(model_->MainGraph(),
                                                 loss_function_output_name,
                                                 weights_to_train_,
                                                 weight_to_mixed_precision_map_.empty() ? nullptr : &weight_to_mixed_precision_map_,
                                                 gradient_graph_config_,
                                                 logger));

  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::BuildAccumulationNode(const std::unordered_set<std::string>& weights_to_train) {
  std::vector<std::string> gradient_names;
  gradient_names.reserve(weights_to_train.size());
  std::transform(
      weights_to_train.begin(), weights_to_train.end(), std::back_inserter(gradient_names),
      GradientBuilderBase::GradientName);
  auto nodearg_name_generator = [](const std::string& base_name) {
    return base_name;
  };
  ORT_RETURN_IF_ERROR(AddGradientAccumulationNodes(model_->MainGraph(), nodearg_name_generator, gradient_names));
  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::BuildOptimizer(
    const OptimizerGraphConfig& opt_graph_config,
    const std::unordered_map<std::string, OptimizerNodeConfig>& opt_configs,
    OptimizerOutputKeyMap<std::string>& opt_graph_outputs) {
  ORT_RETURN_IF_NOT(
      opt_configs.size() == weights_to_train_.size(),
      "Number of optimizer configurations does not match number of weights to train.")

  for (const auto& weight_name : weights_to_train_) {
    ORT_RETURN_IF_NOT(
        opt_configs.find(weight_name) != opt_configs.end(),
        "Optimizer configuration was not found for weight to train: ", weight_name);
  }

  opt_graph_config_ = opt_graph_config;
  opt_configs_ = opt_configs;

  ORT_RETURN_IF_ERROR(BuildOptimizerInternal(model_->MainGraph(),
                                             opt_graph_config_,
                                             opt_configs_,
                                             opt_state_initializer_names_,
                                             opt_graph_outputs,
                                             updated_weight_names_map_,
                                             weight_partition_info_,
                                             weight_to_opt_mapping_));

  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::OverrideGraphOutputs(const std::vector<std::string>& outputs) {
  ORT_RETURN_IF_ERROR(GraphAugmenter::OverrideGraphOutputs(model_->MainGraph(), outputs));
  return DoPostLoadProcessing(*model_);
}

NameMLValMap TrainingSession::GetWeights() const {
  return GetSessionState().GetInitializedTensors(weights_to_train_);
}

static Status UpdateWeightsBeforeSaving(
    Graph& graph, const NameMLValMap& weights, const DataTransferManager& data_transfer_manager) {
  // Store MLValue (either in CPU or CUDA) into TensorProto
  // TODO: support more types than float

  static const OrtMemoryInfo cpu_alloc_info{onnxruntime::CPU, OrtDeviceAllocator};
  for (const auto& name_and_ml_value : weights) {
    const auto& src_tensor = name_and_ml_value.second.Get<Tensor>();

    const ONNX_NAMESPACE::TensorProto* old_tensor_proto = nullptr;
    if (!graph.GetInitializedTensor(name_and_ml_value.first, old_tensor_proto)) {
      continue;
    }
    ONNX_NAMESPACE::TensorProto new_tensor_proto = *old_tensor_proto;
    if (new_tensor_proto.has_raw_data()) {
      auto* const raw_data = new_tensor_proto.mutable_raw_data();
      auto dst_span = gsl::make_span(&(*raw_data)[0], raw_data->size());
      ORT_RETURN_IF_ERROR(CopyTensorDataToByteSpan(
          data_transfer_manager, src_tensor, cpu_alloc_info, dst_span));
    } else {
      ORT_ENFORCE(new_tensor_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT);
      auto* const float_data = new_tensor_proto.mutable_float_data();
      auto dst_span = gsl::make_span(float_data->mutable_data(), float_data->size());
      ORT_RETURN_IF_ERROR(CopyTensorDataToSpan(
          data_transfer_manager, src_tensor, cpu_alloc_info, dst_span));
    }

    // Replace the TensorProto in the model.
    ORT_RETURN_IF_ERROR(graph.ReplaceInitializedTensor(new_tensor_proto));
  }
  return Status::OK();
}

Status TrainingSession::Save(const PathString& model_uri, TrainingSession::SaveOption opt) {
  // Delete the old file before saving.
  std::remove(ToMBString(model_uri).c_str());  // TODO would be good to have something like RemoveFile(PathString)

  if (opt == TrainingSession::SaveOption::NO_RELOAD) {
    return Model::Save(*model_, model_uri);
  }

  // Have to load the original model again.
  // Because after Initialize(), the model has been optimized and the saved graph doesn't look like what we expect.
  std::shared_ptr<Model> new_model;
  ORT_RETURN_IF_ERROR(Model::Load(model_location_, new_model, nullptr, *session_logger_));
  ORT_RETURN_IF_ERROR(UpdateWeightsBeforeSaving(
      new_model->MainGraph(), GetWeights(), GetSessionState().GetDataTransferMgr()));

  std::string actual_loss_name{};
  optional<std::string> loss_scale_input_name =
      is_mixed_precision_enabled_ ? optional<std::string>{""} : optional<std::string>{};

  if (opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC /* with weights and loss func*/ ||
      opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS /*with everything*/) {
    ORT_RETURN_IF_ERROR(ConfigureLossFunctionInternal(
        external_loss_name_, loss_graph_builder_.get(), loss_function_info_,
        new_model->MainGraph(),
        loss_scale_input_name.has_value() ? &loss_scale_input_name.value() : nullptr, actual_loss_name));
  }

  if (opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS) {
    ORT_RETURN_IF_ERROR(BuildGradientGraphInternal(new_model->MainGraph(),
                                                   actual_loss_name,
                                                   weights_to_train_,
                                                   weight_to_mixed_precision_map_.empty() ? nullptr : &weight_to_mixed_precision_map_,
                                                   gradient_graph_config_,
                                                   *session_logger_));

    OptimizerOutputKeyMap<std::string> opt_graph_outputs;
    std::unordered_set<std::string> opt_state_initializer_names;
    std::unordered_map<std::string, std::string> updated_weight_names_map;
    std::unordered_map<std::string, TrainingSession::PartitionInfo> weight_partition_info;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> weight_to_opt_mapping;
    ORT_RETURN_IF_ERROR(BuildOptimizerInternal(new_model->MainGraph(),
                                               opt_graph_config_,
                                               opt_configs_,
                                               opt_state_initializer_names,
                                               opt_graph_outputs,
                                               updated_weight_names_map,
                                               weight_partition_info,
                                               weight_to_opt_mapping));
  }

  auto status = Model::Save(*new_model, model_uri);

  if (!status.IsOK()) {
    LOGS(*session_logger_, WARNING)
        << "Error when saving model " << ToMBString(model_uri) << " : " << status.ErrorMessage();
  }

  return status;
}

common::Status TrainingSession::GetStateTensors(NameMLValMap& state_tensors) {
  bool allow_missing = (opt_graph_config_.deepspeed_zero.stage != 0);
  return GetSessionState().GetInitializedTensors(GetStateTensorNames(), allow_missing, state_tensors);
}

common::Status TrainingSession::GetOptimizerState(std::unordered_map<std::string, NameMLValMap>& opt_state_tensors) {
  const bool allow_missing = (opt_graph_config_.deepspeed_zero.stage != 0);
  // weight_to_opt_mapping_ is in the format of {weight_name: {prefix: full_optimizer_name, ..}, ..}
  for (const auto& weight_map : weight_to_opt_mapping_) {
    std::unordered_set<std::string> opt_names;
    for (const auto& opt_pair : weight_map.second) {
      opt_names.emplace(opt_pair.second);
    }
    NameMLValMap curr_opt_tensors;
    const auto& weight_name = weight_map.first;
    GetSessionState().GetInitializedTensors(opt_names, allow_missing, curr_opt_tensors);
    opt_state_tensors[weight_name] = {};
    // Keep only prefix in returned value
    for (const auto& opt_pair : weight_map.second) {
      const auto& opt_prefix = opt_pair.first;
      const auto& opt_name = opt_pair.second;
      opt_state_tensors[weight_name][opt_prefix] = curr_opt_tensors[opt_name];
    }
  }
  // Change key from sharded_name to weight_name using partition_info
  for (const auto& weight : weight_partition_info_) {
    const auto& it = opt_state_tensors.find(weight.second.view_name);
    ORT_ENFORCE(it != opt_state_tensors.end(), "Cannot find weight: " + weight.second.view_name + " in weight_partition_info_");
    opt_state_tensors[weight.first] = it->second;
    opt_state_tensors.erase(it);
  }
  return Status::OK();
}

common::Status TrainingSession::GetModelState(std::unordered_map<std::string, NameMLValMap>& model_state_tensors, bool include_mixed_precision_weights) {
  const bool allow_missing = (opt_graph_config_.deepspeed_zero.stage != 0);
  std::unordered_set<std::string> fp_tensor_names{};
  fp_tensor_names.insert(
      weights_to_train_.begin(), weights_to_train_.end());
  // Add zero sharded weights, only needed for fp32 weights in mixed precision run
  for (const auto& weight_sharded_pair : updated_weight_names_map_) {
    fp_tensor_names.erase(weight_sharded_pair.first);  // remove the original name
    fp_tensor_names.insert(weight_sharded_pair.second);
  }
  NameMLValMap fp_weights;
  GetSessionState().GetInitializedTensors(fp_tensor_names, allow_missing, fp_weights);
  // Change key from sharded_name to weight_name
  for (const auto& weight_sharded_pair : updated_weight_names_map_) {
    const auto& it = fp_weights.find(weight_sharded_pair.second);
    ORT_ENFORCE(it != fp_weights.end(), "Cannot find weight: " + weight_sharded_pair.second + " in updated_weight_names_map_");
    fp_weights[weight_sharded_pair.first] = it->second;
    fp_weights.erase(it);
  }
  model_state_tensors["full_precision"] = fp_weights;
  if (include_mixed_precision_weights) {
    std::unordered_set<std::string> mp_tensor_names{};
    std::unordered_set<std::string> mixed_precision_weight_initializer_names{};
    std::transform(weight_to_mixed_precision_map_.begin(), weight_to_mixed_precision_map_.end(),
                   std::inserter(mixed_precision_weight_initializer_names, mixed_precision_weight_initializer_names.begin()),
                   [](auto pair) { return pair.second; });
    mp_tensor_names.insert(
        mixed_precision_weight_initializer_names.begin(), mixed_precision_weight_initializer_names.end());
    NameMLValMap mp_weights;
    GetSessionState().GetInitializedTensors(mp_tensor_names, allow_missing, mp_weights);
    // Change key from fp16_name to weight_name
    for (const auto& weight_fp16_pair : weight_to_mixed_precision_map_) {
      const auto& it = mp_weights.find(weight_fp16_pair.second);
      ORT_ENFORCE(it != mp_weights.end(), "Cannot find weight: " + weight_fp16_pair.second + " in weight_to_mixed_precision_map_");
      mp_weights[weight_fp16_pair.first] = it->second;
      mp_weights.erase(it);
    }
    model_state_tensors["mixed_precision"] = mp_weights;
  }
  return Status::OK();
}

common::Status TrainingSession::GetPartitionInfoMap(std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int>>>& part_info_map) {
  for (const auto& weight : weight_partition_info_) {
    const auto& weight_name = weight.first;
    std::transform(weight_partition_info_[weight_name].original_dim.begin(), weight_partition_info_[weight_name].original_dim.end(),
                   std::inserter(part_info_map[weight_name]["original_dim"], part_info_map[weight_name]["original_dim"].end()),
                   [](const int64_t& dim) { return (int)dim; });
    part_info_map[weight_name]["megatron_row_partition"] = std::vector<int>{weight_partition_info_[weight_name].megatron_row_partition};
  }
  return Status::OK();
}

const DataTransferManager& TrainingSession::GetDataTransferManager() const {
  return GetSessionState().GetDataTransferMgr();
}

bool TrainingSession::IsGraphOutputFp32Node(const std::string& output_name) const {
  auto output_producer_node = model_->MainGraph().GetProducerNode(output_name);
  ORT_ENFORCE(output_producer_node != nullptr, "Output: " + output_name + " is not produced by any node.");

  for (auto output : output_producer_node->OutputDefs()) {
    if (output->Name() == output_name && output->TypeAsProto() != nullptr && output->TypeAsProto()->has_tensor_type() &&
        output->TypeAsProto()->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      return true;
    }
  }

  return false;
}

common::Status TrainingSession::Run(const RunOptions& run_options, IOBinding& io_binding) {
  // Override initializers in eval mode.
  if (!run_options.training_mode) {
    std::vector<std::pair<std::string, OrtValue>> new_feeds;
    if (!dropout_eval_feeds_.empty()) {
      // override all dropout ratios to 0
      for (auto& drop_ratio : dropout_eval_feeds_) {
        OrtValue feed_value;
        // We allocate on CPU first, copy will be taken care of downstream.
        const auto* cpu_ep = GetSessionState().GetExecutionProviders().Get(onnxruntime::kCpuExecutionProvider);
        const auto cpu_allocator = cpu_ep->GetAllocator(0, OrtMemTypeDefault);
        feed_value = onnxruntime::MakeScalarMLValue<float>(cpu_allocator, 0.f, true /*is_1d*/);
        // Bind new feed to graph input.
        new_feeds.emplace_back(drop_ratio, feed_value);
      }
    } else {
      auto& input_names = io_binding.GetInputNames();
      if (GetSessionState().GetInputNodeInfoMap().find(training_mode_string_) !=
              GetSessionState().GetInputNodeInfoMap().end() &&
          std::find(input_names.begin(), input_names.end(), training_mode_string_) == input_names.end()) {
        // Set training_mode input to false
        OrtValue training_mode_feed_value;
        // We allocate on CPU first, copy will be taken care of downstream.
        const auto* cpu_ep = GetSessionState().GetExecutionProviders().Get(onnxruntime::kCpuExecutionProvider);
        const auto cpu_allocator = cpu_ep->GetAllocator(0, OrtMemTypeDefault);
        training_mode_feed_value = onnxruntime::MakeScalarMLValue<bool>(cpu_allocator, false, true /*is_1d*/);
        new_feeds.emplace_back(training_mode_string_, training_mode_feed_value);
      }
    }
    for (auto& new_feed : new_feeds) {
      // Bind new feed to graph input.
      ORT_RETURN_IF_ERROR(io_binding.BindInput(new_feed.first, new_feed.second));
    }
  }

  // Call Run in inferenceSession
  return InferenceSession::Run(run_options, io_binding);
}

static const std::unordered_set<std::string> Nodes_Need_Eval_Feeds = {
    // TODO remove this once ONNX TrainableDropout is completely deprecated.
    "TrainableDropout",
    "Dropout",
};
Status TrainingSession::SetEvalFeedNames() {
  Graph& graph = model_->MainGraph();

  GraphAugmenter::GraphDefs defs{};
  std::set<std::string> def_graph_input_names;
  std::set<std::string> def_graph_initializer_names;

  for (auto& node : graph.Nodes()) {
    auto it = Nodes_Need_Eval_Feeds.find(node.OpType());
    if (it != Nodes_Need_Eval_Feeds.cend()) {
      // The opset is < 12, add each ratio input to graph inputs for overriding.
      // Needs to be removed when TrainableDropout is deprecated.
      if (it->compare("TrainableDropout") == 0) {
        auto& ratio_name = node.InputDefs()[1]->Name();
        dropout_eval_feeds_.insert(ratio_name);
        ORT_ENFORCE(model_->MainGraph().GetProducerNode(ratio_name) == nullptr,
                    "Input: " + ratio_name + " should not have any producer node.");
        if (def_graph_input_names.find(ratio_name) == def_graph_input_names.end()) {
          defs.AddGraphInputs({ratio_name});
          def_graph_input_names.insert(ratio_name);
        }
      }
      // Found an opset-12 dropout node, replace initializer name.
      else if (node.InputArgCount().size() > 2) {
        auto& mode_input = node.MutableInputDefs()[2];
        const ONNX_NAMESPACE::TensorProto* mode_initializer = nullptr;
        if (!graph.GetInitializedTensor(training_mode_string_, mode_initializer)) {
          // training_mode initializer has not been added before, add it here.
          // Ideally we want only 1 training_mode initializer to control all relevant nodes.
          const ONNX_NAMESPACE::TensorProto* original_mode_initializer = nullptr;
          ORT_ENFORCE(graph.GetInitializedTensor(mode_input->Name(), original_mode_initializer) == true,
                      "Dropout's input: " + mode_input->Name() + " must be an initializer.");
          ONNX_NAMESPACE::TensorProto new_mode_initializer(*original_mode_initializer);
          new_mode_initializer.set_name(training_mode_string_);
          if (def_graph_initializer_names.find(training_mode_string_) == def_graph_initializer_names.end()) {
            defs.AddInitializers({new_mode_initializer});
            def_graph_initializer_names.insert(training_mode_string_);
          }
        }
        mode_input = &model_->MainGraph().GetOrCreateNodeArg(training_mode_string_, mode_input->TypeAsProto());
        // Set training_mode as graph input if any node that needs eval feed is found,
        // it's okay to add it multiple times since it will be de-dup'ed downstream.
        if (def_graph_input_names.find(training_mode_string_) == def_graph_input_names.end()) {
          defs.AddGraphInputs({training_mode_string_});
          def_graph_input_names.insert(training_mode_string_);
        }
      }
    }
  }

  ORT_RETURN_IF_ERROR(GraphAugmenter::AugmentGraph(graph, defs));
  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::SetStateTensors(const NameMLValMap& state_tensors, bool strict) {
  ORT_RETURN_IF_NOT(IsInitialized(), "Can't update initializers before session has been initialized.");

  std::unordered_set<std::string> ckpt_initializer_names;
  std::transform(state_tensors.begin(), state_tensors.end(),
                 std::inserter(ckpt_initializer_names, ckpt_initializer_names.end()),
                 [](const auto& pair) { return pair.first; });

  NameMLValMap initializers;
  ORT_RETURN_IF_ERROR(GetSessionState().GetInitializedTensors(ckpt_initializer_names, !strict, initializers));

  const std::unordered_set<std::string> valid_state_tensor_names = GetStateTensorNames();

  for (auto& state : state_tensors) {
    const bool is_valid_state_tensor =
        valid_state_tensor_names.find(state.first) != valid_state_tensor_names.end();
    const auto initializer_it = initializers.find(state.first);
    const bool is_tensor_present = initializer_it != initializers.end();

    if (strict) {
      ORT_RETURN_IF_NOT(
          is_valid_state_tensor,
          "Checkpoint tensor: ", state.first, " is not a known state tensor.");
      ORT_RETURN_IF_NOT(
          is_tensor_present,
          "Checkpoint tensor: ", state.first, " is not present in the model.");
    }

    if (is_valid_state_tensor && is_tensor_present) {
      ORT_RETURN_IF_NOT(
          initializer_it->second.IsTensor() && state.second.IsTensor(),
          "Non-tensor type as initializer is not expected.")

      auto* initializer_tensor = initializer_it->second.GetMutable<Tensor>();
      auto& ckpt_tensor = state.second.Get<Tensor>();
      ORT_RETURN_IF_ERROR(GetSessionState().GetDataTransferMgr().CopyTensor(ckpt_tensor, *initializer_tensor));
    }
  }

  return Status::OK();
}

std::unordered_set<std::string> TrainingSession::GetStateTensorNames() const {
  std::unordered_set<std::string> checkpointed_tensor_names{};
  checkpointed_tensor_names.insert(
      weights_to_train_.begin(), weights_to_train_.end());
  for (const auto& p : updated_weight_names_map_) {
    checkpointed_tensor_names.insert(p.second);
  }
  checkpointed_tensor_names.insert(
      opt_state_initializer_names_.begin(), opt_state_initializer_names_.end());
  std::unordered_set<std::string> mixed_precision_weight_initializer_names{};
  std::transform(weight_to_mixed_precision_map_.begin(), weight_to_mixed_precision_map_.end(),
                 std::inserter(mixed_precision_weight_initializer_names, mixed_precision_weight_initializer_names.begin()),
                 [](auto pair) { return pair.second; });
  checkpointed_tensor_names.insert(
      mixed_precision_weight_initializer_names.begin(), mixed_precision_weight_initializer_names.end());
  return checkpointed_tensor_names;
}

bool TrainingSession::IsUntrainable(const Node* node, const std::string& initializer_name,
                                    const logging::Logger* logger) {
  auto it = STOP_GRADIENT_EDGES.find(node->OpType());
  if (it != STOP_GRADIENT_EDGES.end()) {
    for (auto input_idx : it->second) {
      if (input_idx < node->InputDefs().size() &&
          node->InputDefs()[input_idx]->Name() == initializer_name) {
        if (logger) {
          VLOGS(*logger, 1) << "Excluding " << node->Name() << "'s input " << input_idx
                            << " initializer: " << initializer_name;
        }
        return true;
      }
    }
  }
  return false;
}

bool TrainingSession::IsImmutableWeight(const ImmutableWeights& immutable_weights,
                                        const Node* node, const TensorProto* tensor,
                                        const logging::Logger* logger) {
  auto it = immutable_weights.find(node->OpType());
  if (it == immutable_weights.end()) {
    return false;
  }

  for (auto pair : it->second) {
    size_t& input_idx = pair.first;
    float& value = pair.second;

    if (input_idx < node->InputDefs().size() &&
        node->InputDefs()[input_idx]->Name() == tensor->name()) {
      if (tensor->data_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT && tensor->dims_size() == 0) {
        float tensor_value;
        if (tensor->has_raw_data()) {
          memcpy(&tensor_value, tensor->raw_data().data(), sizeof(float));
        } else {
          tensor_value = *(tensor->float_data().data());
        }
        if (tensor_value == value) {
          if (logger) {
            VLOGS(*logger, 1) << "Excluding " << node->Name() << "'s input " << input_idx
                              << " initializer: " << tensor->name() << " with value " << tensor_value;
          }
          return true;
        }
      }
    }
  }

  return false;
}

std::unordered_set<std::string> TrainingSession::GetTrainableModelInitializers(
    const ImmutableWeights& immutable_weights, const std::string& loss_name) const {
  const Graph& graph = model_->MainGraph();
  const auto& initialized_tensors = graph.GetAllInitializedTensors();
  std::unordered_set<std::string> trainable_initializers;

  auto add_trainable_initializers = [&](const Node* node) {
    for (auto input : node->InputDefs()) {
      std::string initializer_name = input->Name();
      if (initialized_tensors.count(initializer_name) == 0)
        continue;

      if (IsUntrainable(node, initializer_name, session_logger_) ||
          IsImmutableWeight(immutable_weights, node, initialized_tensors.at(initializer_name), session_logger_))
        continue;

      trainable_initializers.insert(initializer_name);
    }
  };

  auto stop_at_untrainable = [&](const Node* from, const Node* to) {
    auto is_trainable_from_to_link = [&](Node::EdgeEnd e) {
      if (&e.GetNode() != to)
        return false;

      std::string input_name = from->InputDefs()[e.GetDstArgIndex()]->Name();
      return !IsUntrainable(from, input_name, session_logger_);
    };

    bool proceed = std::any_of(from->InputEdgesBegin(), from->InputEdgesEnd(), is_trainable_from_to_link);
    if (!proceed && session_logger_) {
      VLOGS(*session_logger_, 1)
          << "Stopping training parameters discovery traversal from " << from->Name() << " to " << to->Name();
    }

    return !proceed;
  };

  // perform reverse dfs from output node to discover trainable parameters
  graph.ReverseDFSFrom({graph.GetProducerNode(loss_name)}, add_trainable_initializers, {}, {}, stop_at_untrainable);
  return trainable_initializers;
}

}  // namespace training
}  // namespace onnxruntime
