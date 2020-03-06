// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"

#include "orttraining/core/session/training_session.h"

#include "core/graph/model.h"
#include "orttraining/core/graph/loss_function_builder.h"
#include "orttraining/core/graph/optimizer_builder.h"
#include "orttraining/core/framework/checkpointing.h"
#include "orttraining/core/framework/data_transfer_utils.h"
#include "orttraining/core/framework/gradient_graph_builder.h"
#include "orttraining/core/framework/distributed_run_context.h"
#include "orttraining/core/graph/optimizer_graph_builder_registry.h"
#include "orttraining/core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "orttraining/core/graph/mixed_precision_transformer.h"
#include "orttraining/core/graph/tensorboard_transformer.h"
#include "orttraining/core/graph/gradient_builder_base.h"

//Gist Encoding
#include "orttraining/core/optimizer/gist_encode_decode.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_allocator.h"
#endif

#ifdef USE_HOROVOD
#include "orttraining/core/graph/horovod_adapters.h"
#endif

namespace onnxruntime {
namespace training {

namespace {
Status SetupOptimizerParams(
    const std::unordered_set<std::string>& weight_names_to_train,
    const std::unordered_map<std::string, NodeArg*>& fp32_weight_names_to_fp16_node_args,
    const std::string& loss_scale_input_name,
    const TrainingSession::TrainingConfiguration& config,
    OptimizerGraphConfig& opt_graph_config_result,
    std::unordered_map<std::string, OptimizerNodeConfig>& opt_node_configs_result) {
  ORT_RETURN_IF_NOT(config.optimizer_config.has_value());
  const auto& optimizer_config = config.optimizer_config.value();

  std::unordered_map<std::string, OptimizerNodeConfig> opt_node_configs{};
  for (const auto& weight_name : weight_names_to_train) {
    OptimizerNodeConfig opt_node_config{};
    opt_node_config.name = optimizer_config.name;
    opt_node_config.lr_feed_name = optimizer_config.learning_rate_input_name;
    try {
      opt_node_config.attributes = optimizer_config.weight_attributes_generator(weight_name);
    } catch (const std::exception& ex) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, ex.what());
    }
    opt_node_config.loss_scale_input_name = loss_scale_input_name;
    opt_node_config.use_fp16_moments = optimizer_config.use_fp16_moments;

    const auto fp16_weight_name_it = fp32_weight_names_to_fp16_node_args.find(weight_name);
    if (fp16_weight_name_it != fp32_weight_names_to_fp16_node_args.end()) {
      opt_node_config.fp16_weight_arg = fp16_weight_name_it->second;
    }
    opt_node_configs.emplace(weight_name, std::move(opt_node_config));
  }

  OptimizerGraphConfig opt_graph_config{};
  opt_graph_config.use_mixed_precision = config.mixed_precision_config.has_value();
  opt_graph_config.loss_scale_input_name = loss_scale_input_name;
  opt_graph_config.local_size = DistributedRunContext::RunConfig().local_size;
  opt_graph_config.local_rank = DistributedRunContext::RunConfig().local_rank;
  opt_graph_config.data_parallel_group_rank = DistributedRunContext::RankInGroup(WorkerGroupType::DataParallel);
  opt_graph_config.data_parallel_group_size = DistributedRunContext::GroupSize(WorkerGroupType::DataParallel);
  opt_graph_config.gradient_accumulation_steps = config.gradient_accumulation_steps;
  opt_graph_config.allreduce_in_fp16 = optimizer_config.do_all_reduce_in_fp16;
  opt_graph_config.use_nccl = optimizer_config.use_nccl;
  opt_graph_config.adasum_reduction_type = optimizer_config.adasum_reduction_type;
#if USE_HOROVOD
  opt_graph_config.horovod_reduce_op =
      opt_graph_config.adasum_reduction_type == AdasumReductionType::None
          ? static_cast<int64_t>(hvd::ReduceOp::SUM)
          : static_cast<int64_t>(hvd::ReduceOp::ADASUM);
#endif
  opt_graph_config.partition_optimizer = optimizer_config.partition_optimizer;
  opt_node_configs_result = std::move(opt_node_configs);
  opt_graph_config_result = std::move(opt_graph_config);

  return Status::OK();
}

bool IsRootNode(const TrainingSession::TrainingConfiguration& config) {
  return config.distributed_config.world_rank == 0;
}
}  // namespace

Status TrainingSession::ConfigureForTraining(
    const TrainingConfiguration& config, TrainingConfigurationResult& config_result_out) {
  ORT_RETURN_IF(
      IsInitialized(),
      "TrainingSession::ConfigureForTraining() must be called before TrainingSession::Initialize().");

  if (is_configured_) return Status::OK();

  TrainingConfigurationResult config_result{};
  std::vector<std::string> tensorboard_scalar_names{};

  DistributedRunContext::CreateInstance({config.distributed_config.world_rank,
                                         config.distributed_config.world_size,
                                         config.distributed_config.local_rank,
                                         config.distributed_config.local_size,
                                         config.distributed_config.data_parallel_size,
                                         config.distributed_config.horizontal_parallel_size});

  ORT_RETURN_IF_ERROR(ApplyTransformationsToMainGraph());

  // add loss scale
  std::string loss_scale_input_name{};
  if (config.mixed_precision_config.has_value()) {
    const auto& mixed_precision_config = config.mixed_precision_config.value();

    TrainingConfigurationResult::MixedPrecisionConfigurationResult mixed_precision_config_result{};

    if (mixed_precision_config.add_loss_scaling) {
      ORT_RETURN_IF_ERROR(BuildLossScalingFactorInput(
          mixed_precision_config.initial_loss_scale_value,
          loss_scale_input_name));

      tensorboard_scalar_names.emplace_back(loss_scale_input_name);
      mixed_precision_config_result.loss_scale_input_name = loss_scale_input_name;
    }

    config_result.mixed_precision_config_result = mixed_precision_config_result;
  }

  // configure loss function or use external one
  ORT_RETURN_IF_NOT(
      config.loss_function_config.has_value() ^ config.loss_name.has_value(),
      "Exactly one of loss_function_config or loss_name should be given.");

  std::string loss_name{};
  if (config.loss_function_config.has_value()) {
    const auto& loss_function_config = config.loss_function_config.value();
    ORT_RETURN_IF_ERROR(BuildLossFunction(
        loss_function_config.loss_function_info, loss_scale_input_name, loss_name));

    if (IsRootNode(config) && config.model_with_loss_function_path.has_value()) {
      ORT_IGNORE_RETURN_VALUE(Save(
          config.model_with_loss_function_path.value(), SaveOption::NO_RELOAD));
    }
  } else {
    loss_name = config.loss_name.value();
  }

  // derive actual set of weights to train
  std::unordered_set<std::string> weight_names_to_train =
      !config.weight_names_to_train.empty()
          ? config.weight_names_to_train
          : GetTrainableModelInitializers(config.immutable_weights);
  for (const auto& weight_name_to_not_train : config.weight_names_to_not_train) {
    weight_names_to_train.erase(weight_name_to_not_train);
  }

  {
    std::ostringstream weight_names_stream{};
    for (const auto& weight_name : weight_names_to_train) {
      weight_names_stream << "  " << weight_name << "\n";
    }
    LOGS(*session_logger_, INFO) << "Training weights:\n"
                                 << weight_names_stream.str();
  }

  // add gradient graph
  ORT_RETURN_IF_ERROR(BuildGradientGraph(
      weight_names_to_train, loss_name, config.set_gradients_as_graph_outputs));

  // transform for mixed precision
  std::unordered_map<std::string, NodeArg*> fp32_weight_name_to_fp16_node_arg{};
  if (config.mixed_precision_config.has_value()) {
    const auto& mixed_precision_config = config.mixed_precision_config.value();
    ORT_RETURN_IF_ERROR(EnableMixedPrecision(
        weight_names_to_train, mixed_precision_config.use_fp16_initializers, fp32_weight_name_to_fp16_node_arg));
  }

  // add optimizer or gradient accumulation
  if (config.optimizer_config.has_value()) {
    OptimizerGraphConfig opt_graph_config{};
    std::unordered_map<std::string, OptimizerNodeConfig> opt_node_configs{};
    ORT_RETURN_IF_ERROR(SetupOptimizerParams(
        weight_names_to_train, fp32_weight_name_to_fp16_node_arg,
        loss_scale_input_name, config, opt_graph_config, opt_node_configs));

    TrainingConfigurationResult::OptimizerConfigurationResult optimizer_config_result{};
    ORT_RETURN_IF_ERROR(BuildOptimizer(
        opt_graph_config, opt_node_configs,
        optimizer_config_result.output_key_to_graph_output_name));

    config_result.opt_config_result = optimizer_config_result;
  } else {
    if (config.gradient_accumulation_steps > 1) {
      ORT_RETURN_IF_ERROR(BuildAccumulationNode(weight_names_to_train));
    }
  }

  // add Tensorboard
  if (config.tensorboard_config.has_value()) {
    const auto& tensorboard_config = config.tensorboard_config.value();

    tensorboard_scalar_names.insert(
        tensorboard_scalar_names.end(),
        tensorboard_config.scalar_node_names.begin(), tensorboard_config.scalar_node_names.end());

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

  if (IsRootNode(config) && config.model_with_training_graph_path.has_value()) {
    ORT_IGNORE_RETURN_VALUE(Save(
        config.model_with_training_graph_path.value(), SaveOption::NO_RELOAD));
  }

  config_result_out = std::move(config_result);
  is_configured_ = true;

  return Status::OK();
}

static Status AddLossFunctionInternal(Graph& graph,
                                      ILossFunction& loss_graph_builder,
                                      const LossFunctionInfo& loss_func_info,
                                      const std::string& loss_scale_input_name,
                                      std::string& actual_loss_name) {
  auto loss_function_graph_defs = loss_graph_builder(graph, loss_func_info);

  if (!loss_scale_input_name.empty()) {
    // add node to scale by loss_scale_input_name
    TypeProto* loss_type_proto = loss_function_graph_defs.CreateTypeProto({1}, ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    actual_loss_name = graph.GenerateNodeArgName("scaled_loss");
    loss_function_graph_defs.AddNodeDefs(
        {NodeDef{
            "Mul",
            {ArgDef{loss_func_info.loss_name}, ArgDef{loss_scale_input_name, loss_type_proto}},
            {ArgDef{actual_loss_name, loss_type_proto}},
            NodeAttributes(),
            actual_loss_name}});
  } else {
    actual_loss_name = loss_func_info.loss_name;
  }

  return GraphAugmenter::AugmentGraph(graph, loss_function_graph_defs);
}

static Status BuildGradientGraphInternal(Graph& graph,
                                         const std::string& loss_function_output_name,
                                         const std::unordered_set<std::string>& node_arg_names_to_train,
                                         const bool set_gradient_as_graph_output = false) {
  // Compute the gradient graph def.
  GradientGraphBuilder grad_graph_builder(&graph,
                                          {loss_function_output_name},
                                          node_arg_names_to_train,
                                          loss_function_output_name,
                                          set_gradient_as_graph_output);
  return grad_graph_builder.Build();
}

static Status BuildOptimizerInternal(Graph& graph,
                                     const OptimizerGraphConfig& opt_graph_config,
                                     const std::unordered_map<std::string, OptimizerNodeConfig>& opt_configs,
                                     std::unordered_set<std::string>& opt_state_initializer_names,
                                     OptimizerOutputKeyMap<std::string>& opt_graph_outputs) {
  OptimizerBuilderRegistry& optimizer_registry = OptimizerBuilderRegistry::GetInstance();
  OptimizerGraphBuilderRegistry& optimizer_graph_registry = OptimizerGraphBuilderRegistry::GetInstance();
  std::string graph_builder_name = optimizer_graph_registry.GetNameFromConfig(opt_graph_config);
  auto optimizer_graph_builder = optimizer_graph_registry.MakeUnique(
      graph_builder_name, optimizer_registry, opt_graph_config, opt_configs);
  ORT_RETURN_IF_ERROR(optimizer_graph_builder->Build(
      graph, opt_state_initializer_names, opt_graph_outputs));

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
    grad_acc_outputs.push_back(BuildGradientAccumulationNode(
                                   nodearg_name_generator, gradient_argdefs[i], gradient_accumulation_buffers[i], graph_defs, false)
                                   .name);
  }
  return GraphAugmenter::AugmentGraph(graph, graph_defs);
}

Status TrainingSession::ApplyTransformationsToMainGraph() {
  GraphTransformerManager graph_transformation_mgr{1};
  AddPreTrainingTransformers(graph_transformation_mgr);

  // apply transformers
  Graph& graph = model_->MainGraph();
  for (int i = static_cast<int>(TransformerLevel::Level1); i <= static_cast<int>(TransformerLevel::MaxLevel); i++) {
    ORT_RETURN_IF_ERROR(graph_transformation_mgr.ApplyTransformers(graph, static_cast<TransformerLevel>(i), *session_logger_));
  }
  return common::Status::OK();
}

// Registers all the pre transformers with transformer manager
void TrainingSession::AddPreTrainingTransformers(GraphTransformerManager& transformer_manager,
                                                 TransformerLevel graph_optimization_level,
                                                 const std::vector<std::string>& custom_list) {
  auto add_transformers = [&](TransformerLevel level) {
    // Generate and register transformers for level
    auto transformers_to_register = transformer_utils::GeneratePreTrainingTransformers(level, custom_list);
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
    auto transformers_to_register = transformer_utils::GenerateTransformers(level, session_options_.free_dimension_overrides, custom_list);
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

Status TrainingSession::BuildLossScalingFactorInput(const float loss_scale, std::string& loss_scale_input_name) {
  const std::string input_name = model_->MainGraph().GenerateNodeArgName("loss_scale");
  GraphAugmenter::GraphDefs defs{};
  defs.AddInitializers({CreateTensorProto<float>(input_name, loss_scale, {1})});
  ORT_RETURN_IF_ERROR(GraphAugmenter::AugmentGraph(model_->MainGraph(), defs));
  ORT_RETURN_IF_ERROR(DoPostLoadProcessing(*model_));
  loss_scale_input_name = input_name;
  return Status::OK();
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

Status TrainingSession::BuildLossFunction(const LossFunctionInfo& loss_func_info,
                                          const std::string& loss_scale_input_name,
                                          std::string& actual_loss_name) {
  try {
    ORT_RETURN_IF(loss_func_info.op_def.type.empty() || loss_func_info.loss_name.empty(),
                  "BuildLossFunction's loss_function_info is invalid.");

    loss_func_info_ = loss_func_info;
    loss_graph_builder_ = LossFunctionBuilder::Build(loss_func_info_.op_def.type);
    loss_scale_input_name_ = loss_scale_input_name;

    ORT_RETURN_IF_NOT(loss_graph_builder_);
    ORT_RETURN_IF_ERROR(AddLossFunctionInternal(
        model_->MainGraph(), *loss_graph_builder_, loss_func_info_,
        loss_scale_input_name_, actual_loss_name));
  } catch (const OnnxRuntimeException& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to add loss function:", exp.what());
  }
  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::EnableMixedPrecision(const std::unordered_set<std::string>& weights_to_train,
                                             bool use_fp16_initializer,
                                             std::unordered_map<std::string, NodeArg*>& fp32_weight_name_to_fp16_node_arg) {
  ORT_RETURN_IF_ERROR(TransformGraphForMixedPrecision(model_->MainGraph(), weights_to_train, use_fp16_initializer, fp32_weight_name_to_fp16_node_arg));

  std::unordered_set<std::string> fp16_weight_initializer_names{};
  std::transform(
      fp32_weight_name_to_fp16_node_arg.cbegin(), fp32_weight_name_to_fp16_node_arg.cend(),
      std::inserter(fp16_weight_initializer_names, fp16_weight_initializer_names.begin()),
      [](const std::pair<std::string, NodeArg*>& p) {
        return p.second->Name();
      });
  fp16_weight_initializer_names_ = std::move(fp16_weight_initializer_names);

  return Status::OK();
}

Status TrainingSession::BuildGradientGraph(const std::unordered_set<std::string>& weights_to_train,
                                           const std::string& loss_function_output_name,
                                           const bool set_gradient_as_graph_output) {
  // Fill weights_to_train_ according to weights_to_train
  weights_to_train_ = weights_to_train;

  ORT_RETURN_IF_ERROR(BuildGradientGraphInternal(model_->MainGraph(),
                                                 loss_function_output_name,
                                                 weights_to_train_,
                                                 set_gradient_as_graph_output));

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
                                             opt_graph_outputs));

  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::OverrideGraphOutputs(const std::vector<std::string>& outputs) {
  ORT_RETURN_IF_ERROR(GraphAugmenter::OverrideGraphOutputs(model_->MainGraph(), outputs));
  return DoPostLoadProcessing(*model_);
}

NameMLValMap TrainingSession::GetWeights() const {
  return session_state_->GetInitializedTensors(weights_to_train_);
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
      new_model->MainGraph(), GetWeights(), session_state_->GetDataTransferMgr()));

  std::string actual_loss_name{};
  if (opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC /* with weights and loss func*/ ||
      opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS /*with everything*/) {
    ORT_RETURN_IF_NOT(loss_graph_builder_);
    ORT_RETURN_IF_ERROR(AddLossFunctionInternal(
        new_model->MainGraph(),
        *loss_graph_builder_, loss_func_info_,
        loss_scale_input_name_, actual_loss_name));
  }

  if (opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS) {
    ORT_RETURN_IF_ERROR(BuildGradientGraphInternal(new_model->MainGraph(),
                                                   actual_loss_name,
                                                   weights_to_train_,
                                                   false));

    OptimizerOutputKeyMap<std::string> opt_graph_outputs;
    std::unordered_set<std::string> opt_state_initializer_names;
    ORT_RETURN_IF_ERROR(BuildOptimizerInternal(new_model->MainGraph(),
                                               opt_graph_config_,
                                               opt_configs_,
                                               opt_state_initializer_names,
                                               opt_graph_outputs));
  }

  auto status = Model::Save(*new_model, model_uri);

  if (!status.IsOK()) {
    LOGS(*session_logger_, WARNING) << "Error when saving model " << ToMBString(model_uri) << " : " << status.ErrorMessage();
  }

  return status;
}

common::Status TrainingSession::GetStateTensors(NameMLValMap& state_tensors) {
  return session_state_->GetInitializedTensors(GetStateTensorNames(), false, state_tensors);
}

const DataTransferManager& TrainingSession::GetDataTransferManager() const {
  return session_state_->GetDataTransferMgr();
}

Status TrainingSession::SetStateTensors(const NameMLValMap& state_tensors, bool strict) {
  ORT_RETURN_IF_NOT(IsInitialized(), "Can't update initializers before session has been initialized.");

  std::unordered_set<std::string> ckpt_initializer_names;
  std::transform(state_tensors.begin(), state_tensors.end(),
                 std::inserter(ckpt_initializer_names, ckpt_initializer_names.end()),
                 [](auto pair) { return pair.first; });

  NameMLValMap initializers;
  ORT_RETURN_IF_ERROR(session_state_->GetInitializedTensors(ckpt_initializer_names, !strict, initializers));

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
      ORT_RETURN_IF_ERROR(session_state_->GetDataTransferMgr().CopyTensor(ckpt_tensor, *initializer_tensor));
    }
  }

  return Status::OK();
}

std::unordered_set<std::string> TrainingSession::GetStateTensorNames() const {
  std::unordered_set<std::string> checkpointed_tensor_names{};
  checkpointed_tensor_names.insert(
      weights_to_train_.begin(), weights_to_train_.end());
  checkpointed_tensor_names.insert(
      opt_state_initializer_names_.begin(), opt_state_initializer_names_.end());
  checkpointed_tensor_names.insert(
      fp16_weight_initializer_names_.begin(), fp16_weight_initializer_names_.end());
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
    const ImmutableWeights& immutable_weights) const {
  const Graph& graph = model_->MainGraph();
  const auto& initialized_tensors = graph.GetAllInitializedTensors();
  std::unordered_set<std::string> model_initializers;
  std::transform(initialized_tensors.begin(),
                 initialized_tensors.end(),
                 std::inserter(model_initializers, model_initializers.end()),
                 [](const auto& pair) { return pair.first; });

  std::unordered_set<std::string> trainable_initializers(model_initializers);
  for (const std::string& initializer_name : model_initializers) {
    const auto& nodes = graph.GetConsumerNodes(initializer_name);
    for (const Node* node : nodes) {
      if (IsUntrainable(node, initializer_name, session_logger_) ||
          IsImmutableWeight(immutable_weights, node, initialized_tensors.at(initializer_name), session_logger_)) {
        trainable_initializers.erase(initializer_name);
      }
    }
  }

  return trainable_initializers;
}

}  // namespace training
}  // namespace onnxruntime
