// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/training/training_session.h"

#include "core/graph/model.h"
#include "core/graph/training/loss_function_builder.h"
#include "core/graph/training/training_optimizer.h"
#include "core/training/checkpointing.h"
#include "core/training/data_transfer_utils.h"
#include "core/training/gradient_graph_builder.h"
#include "core/training/optimizer_graph_builder.h"
#include "core/optimizer/gelu_fusion.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/insert_output_rewriter.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/graph/training/mixed_precision_transformer.h"
#include "core/graph/training/tensorboard_transformer.h"
#include "core/graph/training/gradient_builder_base.h"

//Gist Encoding
#include "core/optimizer/gist_encode_decode.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_allocator.h"
#endif

using namespace std;

namespace onnxruntime {
namespace training {

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
            {ArgDef{actual_loss_name, loss_type_proto}}}});
  } else {
    actual_loss_name = loss_func_info.loss_name;
  }

  return GraphAugmenter::AugmentGraph(graph, loss_function_graph_defs);
}

static Status BuildGradientGraphInternal(Graph& graph,
                                         const string& loss_function_output_name,
                                         const unordered_set<string>& node_arg_names_to_train,
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
                                     const unordered_map<string, OptimizerNodeConfig>& opt_configs,
                                     std::unordered_set<std::string>& opt_state_initializer_names,
                                     std::unordered_map<std::string, std::string>& opt_graph_outputs) {
  OptimizerGraphBuilder optimizer_graph_builder{
      OptimizerBuilderRegistry::GetInstance(), opt_graph_config, opt_configs};

  ORT_RETURN_IF_ERROR(optimizer_graph_builder.Build(
      graph, opt_state_initializer_names, opt_graph_outputs));

  return Status::OK();
}

static Status AddGradientAccumulationNodes(Graph& graph,
                                           const NodeArgNameGeneratorFn& nodearg_name_generator,
                                           const std::vector<std::string> gradient_names,
                                           bool add_accumulate_as_graph_output) {
  GraphAugmenter::GraphDefs graph_defs{};

  std::vector<ArgDef> gradient_argdefs{};
  ORT_RETURN_IF_ERROR(GetArgDefsFromGraph(graph, gradient_names, gradient_argdefs));
  std::vector<ArgDef> gradient_accumulation_buffers;
  gradient_accumulation_buffers.resize(gradient_argdefs.size());
  std::vector<std::string> grad_acc_outputs;
  for (size_t i = 0; i < gradient_argdefs.size(); ++i) {
    grad_acc_outputs.push_back(BuildGradientAccumulationNode(
                                   nodearg_name_generator, gradient_argdefs[i], gradient_accumulation_buffers[i], graph_defs, false)
                                   .name);
  }
  if (add_accumulate_as_graph_output)
    graph_defs.AddGraphOutputs(grad_acc_outputs);
  return GraphAugmenter::AugmentGraph(graph, graph_defs);
}

Status TrainingSession::ApplyTransformationsToMainGraph() {
  try {
    Graph& graph = model_->MainGraph();

    GraphTransformerManager graph_transformation_mgr{1};

    // MUST be empty here, because this is called before partition, so the node's execution type is not decided yet.
    // If we give values here, the check in transformer will fail.
    std::unordered_set<std::string> compatible_eps = {};
    auto gelu_transformer = std::make_unique<GeluFusion>(compatible_eps);
    graph_transformation_mgr.Register(std::move(gelu_transformer), TransformerLevel::Level2);

    auto status = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2);
    return status;
  } catch (const OnnxRuntimeException& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to apply default optimization passes: ", exp.what());
  }
}

Status TrainingSession::AddGistEncoding() {
  try {
    Graph& graph = model_->MainGraph();

    auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleGistTransformer1");
    rule_transformer_L1->Register(std::make_unique<GistEncodeDecode>());
    onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
    graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

    ORT_RETURN_IF_ERROR(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1));
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

Status TrainingSession::BuildGradientGraph(const unordered_set<string>& weights_to_train,
                                           const string& loss_function_output_name,
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
  std::vector<std::string> gradient_names{};
  gradient_names.reserve(weights_to_train.size());
  std::transform(
      weights_to_train.begin(), weights_to_train.end(), std::back_inserter(gradient_names),
      GradientBuilderBase::GradientName);
  auto nodearg_name_generator = [](const std::string& base_name) {
    return base_name;
  };
  ORT_RETURN_IF_ERROR(AddGradientAccumulationNodes(model_->MainGraph(), nodearg_name_generator, gradient_names, false));
  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::BuildOptimizer(
    const OptimizerGraphConfig& opt_graph_config,
    const unordered_map<string, OptimizerNodeConfig>& opt_configs,
    std::unordered_map<std::string, std::string>& opt_graph_outputs) {
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
  return session_state_.GetInitializedTensors(weights_to_train_);
}

Status TrainingSession::UpdateWeightsInSessionState(const NameMLValMap& new_weights) {
  session_state_.UpdateInitializedTensors(new_weights);
  VLOGS(*session_logger_, 1) << "Done updating weights";
  return Status::OK();
}

static Status UpdateWeightsBeforeSaving(
    Graph& graph, const NameMLValMap& weights, const DataTransferManager& data_transfer_manager) {
  // Store MLValue (either in CPU or CUDA) into TensorProto
  // TODO: support more types than float

  static const OrtAllocatorInfo cpu_alloc_info{onnxruntime::CPU, OrtDeviceAllocator};
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
  shared_ptr<Model> new_model;
  ORT_RETURN_IF_ERROR(Model::Load(model_location_, new_model));
  ORT_RETURN_IF_ERROR(UpdateWeightsBeforeSaving(
      new_model->MainGraph(), GetWeights(), session_state_.GetDataTransferMgr()));

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

    std::unordered_map<std::string, std::string> opt_graph_outputs;
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
  std::unordered_set<std::string> checkpointed_tensor_names{};
  checkpointed_tensor_names.insert(
      weights_to_train_.begin(), weights_to_train_.end());
  checkpointed_tensor_names.insert(
      opt_state_initializer_names_.begin(), opt_state_initializer_names_.end());
  checkpointed_tensor_names.insert(
      fp16_weight_initializer_names_.begin(), fp16_weight_initializer_names_.end());

  return session_state_.GetInitializedTensors(checkpointed_tensor_names, false, state_tensors);
}

const DataTransferManager& TrainingSession::GetDataTransferManager() {
  return session_state_.GetDataTransferMgr();
}

Status TrainingSession::SaveCheckpoint(
    const PathString& checkpoint_path,
    const std::unordered_map<std::string, std::string>& properties) {
  const bool is_profiler_enabled = session_state_.Profiler().IsEnabled();
  const TimePoint start_time = is_profiler_enabled ? session_state_.Profiler().StartTime() : TimePoint{};

  NameMLValMap checkpointed_initialized_tensors{};
  ORT_RETURN_IF_ERROR(GetStateTensors(checkpointed_initialized_tensors));

  ORT_RETURN_IF_ERROR(SaveModelCheckpoint(
      checkpoint_path, session_state_.GetDataTransferMgr(),
      checkpointed_initialized_tensors, properties));

  if (is_profiler_enabled) {
    session_state_.Profiler().EndTimeAndRecordEvent(
        profiling::EventCategory::SESSION_EVENT, "checkpoint_save", start_time);
  }

  return Status::OK();
}

Status TrainingSession::LoadCheckpointAndUpdateInitializedTensors(
    const PathString& checkpoint_path,
    std::unordered_map<std::string, std::string>& properties) {
  const bool is_profiler_enabled = session_state_.Profiler().IsEnabled();
  const TimePoint start_time = is_profiler_enabled ? session_state_.Profiler().StartTime() : TimePoint{};

  std::vector<ONNX_NAMESPACE::TensorProto> loaded_tensor_protos{};
  ORT_RETURN_IF_ERROR(LoadModelCheckpoint(
      checkpoint_path, model_location_, loaded_tensor_protos, properties));

  // overwrite graph initializers
  Graph& graph = model_->MainGraph();
  for (const auto& tensor_proto : loaded_tensor_protos) {
    ORT_RETURN_IF_ERROR(graph.ReplaceInitializedTensor(tensor_proto));
  }

  if (is_profiler_enabled) {
    session_state_.Profiler().EndTimeAndRecordEvent(
        profiling::EventCategory::SESSION_EVENT, "checkpoint_load", start_time);
  }

  return Status::OK();
}

common::Status TrainingSession::UpdateInitializedTensors(const NameMLValMap& state_tensors, bool strict) {
  if (!IsInitialized())
    return Status(ONNXRUNTIME, FAIL, "Can't update initializers before session initialized.");
  NameMLValMap initializers;
  std::unordered_set<std::string> ckpt_initializer_names;
  std::transform(state_tensors.begin(), state_tensors.end(),
                 std::inserter(ckpt_initializer_names, ckpt_initializer_names.end()),
                 [](auto pair) { return pair.first; });
  ORT_RETURN_IF_ERROR(session_state_.GetInitializedTensors(ckpt_initializer_names, !strict, initializers));
  for (auto& state : state_tensors) {
    auto it = initializers.find(state.first);
    if (it != initializers.end()) {
      if (!it->second.IsTensor() || !state.second.IsTensor())
        return Status(ONNXRUNTIME, FAIL, "Non-tensor type as initializer is not expected.");
      auto* initializer_tensor = it->second.GetMutable<Tensor>();
      auto& ckpt_tensor = state.second.Get<Tensor>();
      ORT_RETURN_IF_ERROR(session_state_.GetDataTransferMgr().CopyTensor(ckpt_tensor, *initializer_tensor));
    } else if (strict) {
      return Status(ONNXRUNTIME, FAIL, "Checkpoint tensor: " + state.first + " is not found in training session");
    }
  }
  return Status::OK();
}

std::unordered_set<std::string> TrainingSession::GetModelInputNames() const {
  return model_input_names_;
}

std::unordered_set<std::string> TrainingSession::GetModelOutputNames() const {
  return model_output_names_;
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
      if (tensor->data_type() == TensorProto_DataType_FLOAT && tensor->dims_size() == 0) {
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
  for (const string& initializer_name : model_initializers) {
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

Status TrainingSession::UpdateTrainableWeightsInfoInGraph() {
  Graph& graph = model_->MainGraph();
  const auto& graph_inputs = graph.GetInputsIncludingInitializers();
  std::unordered_set<const NodeArg*> inputs_to_add{};
  std::transform(
      weights_to_train_.begin(), weights_to_train_.end(), std::inserter(inputs_to_add, inputs_to_add.end()),
      [&graph](const std::string& node_name) {
        return graph.GetNodeArg(node_name);
      });
  for (const NodeArg* graph_input : graph_inputs) {
    inputs_to_add.erase(graph_input);
  }
  std::vector<const NodeArg*> new_graph_inputs(graph_inputs);
  new_graph_inputs.insert(new_graph_inputs.end(), inputs_to_add.begin(), inputs_to_add.end());
  graph.SetInputs(new_graph_inputs);
  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
