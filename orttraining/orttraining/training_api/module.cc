// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_api/module.h"

#include "core/common/safeint.h"
#include "core/common/string_utils.h"
#include "core/framework/execution_provider.h"
#include "core/framework/mldata_type_utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/session/inference_session.h"
#include "core/session/environment.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/graph/graph_utils.h"

#include "orttraining/training_api/checkpoint.h"

using namespace onnxruntime;

namespace onnxruntime {
namespace training {
namespace api {

namespace {

// TODO: consolidate with frontend tooling
const std::string ACCUMULATE_GRAD_CONTROL_INPUT_NAME{"lazy_reset_grad"};

#if !defined(ORT_MINIMAL_BUILD)
std::unordered_set<const Node*> GetReverseReachableNodes(Graph& inference_graph,
                                                         InlinedVector<const NodeArg*>& output_node_args) {
  // Perform a graph traversal from the graph outputs to collect all reachable nodes from the outputs
  InlinedVector<NodeIndex> nodes;
  nodes.reserve((output_node_args.size()));
  std::unordered_set<const Node*> visited_nodes;
  for (auto node_arg : output_node_args) {
    auto* node = inference_graph.GetProducerNode(node_arg->Name());
    if (!node || std::find(nodes.begin(), nodes.end(), node->Index()) != nodes.end()) {
      continue;
    }

    nodes.push_back(node->Index());
  }

  inference_graph.ReverseDFSFrom(nodes, [&visited_nodes](const Node* node) { visited_nodes.insert(node); }, {});

  return visited_nodes;
}

Status RemoveUnusedNodes(Graph& inference_graph, InlinedVector<const NodeArg*>& output_node_args) {
  const auto reachable_nodes = GetReverseReachableNodes(inference_graph, output_node_args);

  // Get all graph nodes and remove those that are not in the reachable nodes.
  GraphViewer graph_viewer(inference_graph);
  const auto node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (size_t idx = node_indices.size(); idx > 0; --idx) {
    const NodeIndex node_index = idx - 1;
    auto* node = inference_graph.GetNode(node_index);
    if (!reachable_nodes.count(node)) {
      graph_utils::RemoveNodeOutputEdges(inference_graph, *node);
      inference_graph.RemoveNode(node_index);
    }
  }

  return Status::OK();
}

Status TransformModelOutputsForInference(Graph& inference_graph,
                                         gsl::span<const std::string> inference_graph_outputs) {
  // Model is updated to remove any outputs that are not defined in inference_graph_outputs. Nodes
  // producing these unused model outputs are also subsequently removed.

  ORT_RETURN_IF(inference_graph_outputs.empty(),
                "Expected a non empty vector of graph output names. Got an empty vector.");

  InlinedVector<const NodeArg*> inference_graph_output_node_args;
  inference_graph_output_node_args.reserve(inference_graph_outputs.size());
  for (const auto& output_name : inference_graph_outputs) {
    const NodeArg* output_node_arg = inference_graph.GetNodeArg(std::string(output_name));
    ORT_RETURN_IF_NOT(output_node_arg, "Expected graph output for inference graph " + std::string(output_name) +
                                           " could not be found. Please regenerate the eval graph.");
    inference_graph_output_node_args.push_back(output_node_arg);
  }

  // Set the inference graph outputs, and remove any unused nodes.
  inference_graph.SetOutputs(inference_graph_output_node_args);
  ORT_RETURN_IF_ERROR(RemoveUnusedNodes(inference_graph, inference_graph_output_node_args));

  ORT_RETURN_IF_ERROR(inference_graph.Resolve());

  return Status::OK();
}

Status TransformModelInputsForInference(Graph& inference_graph,
                                        const std::unordered_map<
                                            std::string, std::shared_ptr<Parameter>>& named_parameters,
                                        const DataTransferManager& data_transfer_manager) {
  std::vector<const NodeArg*> user_graph_inputs;
  for (auto& graph_input_node_arg : inference_graph.GetInputs()) {
    auto named_parameter_it = named_parameters.find(graph_input_node_arg->Name());
    if (named_parameter_it == named_parameters.end()) {
      if (inference_graph.GetConsumerNodes(graph_input_node_arg->Name()).empty()) {
        continue;
      }
      user_graph_inputs.emplace_back(graph_input_node_arg);
    } else {
      ORT_ENFORCE(!inference_graph.IsInitializedTensor(named_parameter_it->first),
                  "The eval graph is invalid. Expected model parameter ",
                  named_parameter_it->first, " to be a graph input, not a graph initializer.");
      inference_graph.AddInitializedTensor(utils::CopyTensorToTensorProto(
          named_parameter_it->second->Data().Get<onnxruntime::Tensor>(),
          named_parameter_it->first, data_transfer_manager));
    }
  }

  inference_graph.SetInputs(user_graph_inputs);
  ORT_RETURN_IF_ERROR(inference_graph.Resolve());

  return Status::OK();
}
#endif

InlinedHashMap<std::string, const NodeArg*> BuildParameterToInputNodeArgMap(const ModuleCheckpointState& state,
                                                                            const InputDefList* model_inputs) {
  ORT_ENFORCE(model_inputs != nullptr, "Model inputs are not defined.");
  InlinedHashMap<std::string, const NodeArg*> parameter_to_input_node_arg_map;
  parameter_to_input_node_arg_map.reserve(state.named_parameters.size());
  for (const auto& input_def : *model_inputs) {
    const std::string& input_name = input_def->Name();
    const auto param_it = state.named_parameters.find(input_name);
    if (param_it == state.named_parameters.end()) {
      continue;
    }
    parameter_to_input_node_arg_map[input_name] = input_def;
  }
  return parameter_to_input_node_arg_map;
}

InlinedHashMap<std::string, size_t> BuildParameterToGradInputIndexMap(gsl::span<const std::string> grad_names) {
  InlinedHashMap<std::string, size_t> param_name_to_grad_input_index_map;
  param_name_to_grad_input_index_map.reserve(grad_names.size());
  for (size_t i = 0; i < grad_names.size(); ++i) {
    std::string param_name;
    utils::GetParamNameFromGradient(grad_names[i], param_name);
    param_name_to_grad_input_index_map.insert({param_name, i});
  }
  return param_name_to_grad_input_index_map;
}

Status LoadParameter(const std::string& param_name, const Tensor& src_weight_tensor,
                     const SessionState& session_state, const bool force_load,
                     const InlinedHashMap<std::string, size_t>& param_to_grad_index,
                     gsl::span<const std::string> grad_names, Parameter& param) {
  InlinedVector<SessionState::NodeInfo> node_info_vec;
  ORT_THROW_IF_ERROR(session_state.GetInputNodeInfo(param_name, node_info_vec));
  const auto& node_info = node_info_vec.front();
  const auto target_device = *node_info.device;
  for (auto it = node_info_vec.begin(); it != node_info_vec.end(); ++it) {
    ORT_ENFORCE(target_device == *(it->device), "Inconsistent device requirements found for input: ", param_name);
  }

  if (force_load || src_weight_tensor.Location().device.Type() != target_device.Type()) {
    auto weight_allocator = session_state.GetAllocator(target_device);
    ORT_ENFORCE(weight_allocator != nullptr);

    // Create a new tensor on the target_device and switch the source_ortvalue to point to this new tensor
    auto dst_weight_tensor = std::make_unique<Tensor>(src_weight_tensor.DataType(), src_weight_tensor.Shape(),
                                                      weight_allocator);
    ORT_THROW_IF_ERROR(session_state.GetDataTransferMgr().CopyTensor(src_weight_tensor, *dst_weight_tensor.get()));
    auto ml_tensor_type = DataTypeImpl::GetType<Tensor>();
    param.Data().Init(dst_weight_tensor.release(), ml_tensor_type, ml_tensor_type->GetDeleteFunc());
  }

  if (param.RequiresGrad()) {
    // Create gradient accumulation buffer.
    auto grad_it = param_to_grad_index.find(param_name);
    ORT_ENFORCE(grad_it != param_to_grad_index.end(), "Gradient buffer input not provided for param: ",
                param_name);

    const size_t grad_input_index = grad_it->second;
    auto& param_grad_name = grad_names[grad_input_index];

    OrtValue param_grad;
    ORT_THROW_IF_ERROR(utils::CreateZeroValuedOrtValueLike(session_state, param.Data(), param_grad));
    ORT_THROW_IF_ERROR(param.SetGrad(param_grad_name, param_grad));
  }

  return Status::OK();
}

}  // namespace

Status Parameter::CopyTo(const DataTransferManager* data_transfer_manager, OrtValue& data) const {
  ORT_ENFORCE(data.IsAllocated(), "Given parameter data is not allocated. Cannot copy the checkpoint parameter to it.");
  ORT_ENFORCE(data.IsTensor(), "Parameter data should be of tensor type.");
  ORT_ENFORCE(data.Get<Tensor>().Shape() == data_.Get<Tensor>().Shape(),
              "Parameter data shape mismatch. Expected: ", data_.Get<Tensor>().Shape().ToString(),
              ", Got: ", data.Get<Tensor>().Shape().ToString());
#ifdef ENABLE_STRIDED_TENSORS
  auto data_strides = data.Get<Tensor>().Strides();
  auto param_strides = data_.Get<Tensor>().Strides();
  ORT_ENFORCE(data_strides.size() == param_strides.size(),
              "Parameter data stride mismatch. Expected strides of size: ", param_strides.size(),
              ", Got: ", data_strides.size());
  ORT_ENFORCE(std::equal(data_strides.begin(), data_strides.end(), param_strides.begin()),
              "Parameter data stride value mismatch.");
#endif
  ORT_ENFORCE(data.Get<Tensor>().DataType() == data_.Get<Tensor>().DataType(),
              "Parameter data type mismatch. Expected: ", data_.Get<Tensor>().DataType(),
              ", Got: ", data.Get<Tensor>().DataType());
  ORT_ENFORCE(data_transfer_manager != nullptr,
              "Data transfer manager must be provided to copy data to the parameter. "
              "Please create the TrainingSession before trying to update the parameter.");

  ORT_THROW_IF_ERROR(data_transfer_manager->CopyTensor(data_.Get<Tensor>(), *data.GetMutable<Tensor>()));

  return Status::OK();
}

Status Parameter::CopyFrom(const DataTransferManager* data_transfer_manager, const OrtValue& data) {
  ORT_ENFORCE(data_.IsAllocated(),
              "The checkpoint parameter is not allocated. Cannot copy the given parameter data to it.");
  ORT_ENFORCE(data.IsTensor(), "Parameter data should be of tensor type.");
  ORT_ENFORCE(data.Get<Tensor>().Shape() == data_.Get<Tensor>().Shape(),
              "Parameter data shape mismatch. Expected: ", data_.Get<Tensor>().Shape().ToString(),
              ", Got: ", data.Get<Tensor>().Shape().ToString());
#ifdef ENABLE_STRIDED_TENSORS
  auto data_strides = data.Get<Tensor>().Strides();
  auto param_strides = data_.Get<Tensor>().Strides();
  ORT_ENFORCE(data_strides.size() == param_strides.size(),
              "Parameter data stride mismatch. Expected strides of size: ", param_strides.size(),
              ", Got: ", data_strides.size());
  ORT_ENFORCE(std::equal(data_strides.begin(), data_strides.end(), param_strides.begin()),
              "Parameter data stride value mismatch.");
#endif
  ORT_ENFORCE(data.Get<Tensor>().DataType() == data_.Get<Tensor>().DataType(),
              "Parameter data type mismatch. Expected: ", data_.Get<Tensor>().DataType(),
              ", Got: ", data.Get<Tensor>().DataType());
  ORT_ENFORCE(data_transfer_manager != nullptr,
              "Data transfer manager must be provided to copy data to the parameter. "
              "Please create the TrainingSession before trying to update the parameter.");

  ORT_THROW_IF_ERROR(data_transfer_manager->CopyTensor(data.Get<Tensor>(), *data_.GetMutable<Tensor>()));

  return Status::OK();
}

Status Parameter::SetGrad(const std::string& gradient_name, const OrtValue& param_grad) {
  // assert param is allocated
  ORT_ENFORCE(data_.IsAllocated(), "Parameter data should be allocated before allocating gradient.");
  ORT_ENFORCE(requires_grad_, "Gradient should only be allocated for trainable parameters.");

  gradient_name_ = gradient_name;
  gradient_ = param_grad;
  return Status::OK();
}

Status Parameter::ResetGrad() {
  if (!requires_grad_) {
    return Status::OK();
  }
  Tensor* p_tensor = gradient_.GetMutable<Tensor>();
  const auto& device = p_tensor->Location().device;
  if (device.Type() == OrtDevice::CPU) {
    memset(p_tensor->MutableDataRaw(), 0, p_tensor->SizeInBytes());
  }
#if defined(USE_CUDA) || defined(USE_ROCM)
  else if (device.Type() == OrtDevice::GPU) {
    ORT_NOT_IMPLEMENTED("Not implemented.");
  }
#endif
  else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unknown device type ", device.Type(), " for param:", name_);
  }
  return Status::OK();
}

Module::Module(const ModelIdentifiers& model_identifiers,
               CheckpointState* state,
               const onnxruntime::SessionOptions& session_options,
               const Environment& env,
               const std::vector<std::shared_ptr<IExecutionProvider>>& providers,
               [[maybe_unused]] gsl::span<OrtCustomOpDomain* const> op_domains)
    : state_{state} {
  // Enforce weight prepacking is disabled
  // If the user explicitly enabled weight prepacking then return an error.
  // Default value is enabled. Therefore, explicitly disable it if the value is not set by the user.
  std::string disable_prepacking = "";
  if (session_options.config_options.TryGetConfigEntry(kOrtSessionOptionsConfigDisablePrepacking, disable_prepacking)) {
    ORT_ENFORCE(disable_prepacking == "1", "Prepacking is not supported for training scenarios.");
  } else {
    const_cast<SessionOptions&>(session_options)
        .config_options.configurations[kOrtSessionOptionsConfigDisablePrepacking] = "1";
  }

  train_sess_ = std::make_unique<onnxruntime::InferenceSession>(session_options, env);
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  if (!op_domains.empty()) {
    ORT_THROW_IF_ERROR(train_sess_->AddCustomOpDomains(op_domains));
  }
#endif

  // Load the training model
  ORT_THROW_IF_ERROR(std::holds_alternative<std::string>(model_identifiers.train_model)
                         ? train_sess_->Load(std::get<std::string>(model_identifiers.train_model))
                         : train_sess_->Load(std::get<gsl::span<const uint8_t>>(model_identifiers.train_model).data(),
                                             static_cast<int>(std::get<gsl::span<const uint8_t>>(model_identifiers.train_model).size())));

  for (const auto& provider : providers) {
    ORT_THROW_IF_ERROR(train_sess_->RegisterExecutionProvider(provider));
  }
  ORT_THROW_IF_ERROR(train_sess_->Initialize());

  // Make sure that the checkpoint state can copy tensors
  state_->module_checkpoint_state.train_session_data_transfer_mgr = &train_sess_->GetDataTransferManager();

  // Extract model input and output names
  InlinedVector<std::string> train_input_names, train_output_names;
  utils::GetGraphInputOutputNames(train_sess_, train_input_names, train_output_names);

  // Reorder the extracted input names in the following order:
  // user inputs, weights, gradients, reset_grad
  InlinedVector<std::string> user_input_names, param_input_names, grad_input_names, reset_grad_name;

  for (const auto& input_name : train_input_names) {
    auto it = state_->module_checkpoint_state.named_parameters.find(input_name);
    if (it != state_->module_checkpoint_state.named_parameters.end()) {
      param_input_names.emplace_back(input_name);
    } else if (input_name == ACCUMULATE_GRAD_CONTROL_INPUT_NAME) {
      reset_grad_name.emplace_back(input_name);
    } else if (std::string param_name; utils::GetParamNameFromGradient(input_name, param_name)) {
      grad_input_names.emplace_back(input_name);
    } else {
      user_input_names.emplace_back(input_name);
    }
  }

  gradients_.resize(grad_input_names.size());

  train_input_names_ = TrainInputNames(user_input_names, param_input_names, grad_input_names);

  for (const auto& output_name : train_output_names) {
    if (std::string param_name; !utils::GetParamNameFromGradient(output_name, param_name)) {
      train_output_names_.emplace_back(output_name);
    }
  }

  if (!state_->module_checkpoint_state.is_nominal_state) {
    // ORT_THROW_IF_ERROR(AllocateMemoryForWeights());
    // Loop each parameter, and allocate its memory based on the user-specified device.
    const auto param_to_grad_index = BuildParameterToGradInputIndexMap(train_input_names_.GradientInputNames());
    for (auto& param_name : train_input_names_.WeightsInputNames()) {
      auto params_iter = state_->module_checkpoint_state.named_parameters.find(param_name);
      ORT_ENFORCE(params_iter != state_->module_checkpoint_state.named_parameters.end());

      OrtValue& param_data = params_iter->second->Data();
      ORT_ENFORCE(param_data.IsTensor(), "Expected: Parameter data should be of tensor type. Actual: ",
                  params_iter->second->Name(), " is not a tensor.");
      ORT_THROW_IF_ERROR(LoadParameter(param_name, param_data.Get<Tensor>(), train_sess_->GetSessionState(),
                                       false /* force_load */, param_to_grad_index,
                                       train_input_names_.GradientInputNames(), *params_iter->second));
      weights_.push_back(param_data);
      if (params_iter->second->RequiresGrad()) {
        gradients_[param_to_grad_index.at(param_name)] = params_iter->second->Gradient();
      }
    }
  }

  if (model_identifiers.IsEvalModelAvailable()) {
    eval_sess_ = std::make_unique<onnxruntime::InferenceSession>(session_options, env);
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
    if (!op_domains.empty()) {
      ORT_THROW_IF_ERROR(eval_sess_->AddCustomOpDomains(op_domains));
    }
#endif
    if (std::holds_alternative<std::optional<std::string>>(model_identifiers.eval_model)) {
      ORT_THROW_IF_ERROR(eval_sess_->Load(std::get<std::optional<std::string>>(model_identifiers.eval_model).value()));
    } else {
      auto model_data = std::get<gsl::span<const uint8_t>>(model_identifiers.eval_model);
      ORT_THROW_IF_ERROR(eval_sess_->Load(model_data.data(), static_cast<int>(model_data.size())));
    }
  } else {
    return;
  }

  for (const auto& provider : providers) {
    ORT_THROW_IF_ERROR(eval_sess_->RegisterExecutionProvider(provider));
  }
  ORT_THROW_IF_ERROR(eval_sess_->Initialize());
  utils::GetGraphInputOutputNames(eval_sess_, eval_input_names_, eval_output_names_);

  // Eval model validation
  // We are making certain assumptions: Like the order in which parameters occur will be same between train and eval
  // graphs, and all the weights present in both graphs match.
  // TODO(askhade): Add the checks instead of making assumptions??
  InlinedVector<std::string> eval_user_input_names, eval_param_input_names;
  for (const auto& input_name : eval_input_names_) {
    if (state_->module_checkpoint_state.named_parameters.find(input_name) !=
        state_->module_checkpoint_state.named_parameters.end()) {
      // it is a parameter
      eval_param_input_names.emplace_back(input_name);
      continue;
    } else {
      // It is user input. We handle user inputs separately in the eval
      // because the eval graph might have different user inputs.
      // Eg if loss is not a part of the eval graph, it won't have
      // certain inputs like targets
      eval_user_input_names.emplace_back(input_name);
    }
  }
  eval_input_names_ = eval_user_input_names;
  eval_user_input_count_ = eval_user_input_names.size();
  eval_input_names_.insert(eval_input_names_.end(), eval_param_input_names.begin(), eval_param_input_names.end());

  // Keep a copy of the eval model path to be able to later export the model for inferencing.
  // The inference model will be reconstructed from the eval model.
  // TODO(askhade): Find a fix to export model for inference when the eval model is loaded from a buffer.
  if (std::holds_alternative<std::optional<std::string>>(model_identifiers.eval_model)) {
    eval_model_path_ = std::get<std::optional<std::string>>(model_identifiers.eval_model);
  }
}

Module::~Module() {
  state_->module_checkpoint_state.train_session_data_transfer_mgr = nullptr;
}

size_t Module::GetTrainingModelOutputCount() const noexcept {
  return train_output_names_.size();
}

size_t Module::GetEvalModelOutputCount() const noexcept {
  return eval_output_names_.size();
}

std::string Module::GetTrainingModelOutputName(size_t index) const {
  ORT_ENFORCE(index < train_output_names_.size(), "Train output name index out of range. Expected in range [0-", train_output_names_.size(), "). Actual: ", index);
  return train_output_names_.at(index);
}

std::string Module::GetEvalModelOutputName(size_t index) const {
  ORT_ENFORCE(index < eval_output_names_.size(), "Eval output name index out of range. Expected in range [0-",
              eval_output_names_.size(), "). Actual: ", index);
  return eval_output_names_.at(index);
}

size_t Module::GetParametersSize(const bool trainable_only) const {
  SafeInt<size_t> parameters_size = 0;
  const auto model_inputs_with_error = GetTrainingModelInputs();
  ORT_THROW_IF_ERROR(model_inputs_with_error.first);
  ORT_ENFORCE(model_inputs_with_error.second, "Training model graph inputs are not defined.");
  for (const auto& input_def : *model_inputs_with_error.second) {
    const std::string& input_name = input_def->Name();
    const auto param_it = state_->module_checkpoint_state.named_parameters.find(input_name);
    if (param_it == state_->module_checkpoint_state.named_parameters.end() ||
        (trainable_only && !param_it->second->RequiresGrad())) {
      continue;
    }
    parameters_size += onnxruntime::utils::GetTensorShapeFromTensorShapeProto(*input_def->Shape()).Size();
  }
  return parameters_size;
}

std::vector<std::shared_ptr<Parameter>> Module::Parameters() const {
  ORT_ENFORCE(!state_->module_checkpoint_state.is_nominal_state,
              "Cannot fetch parameters from a nominal checkpoint state. Please load the model parameters first.");
  std::vector<std::shared_ptr<Parameter>> params;
  for (auto& it : state_->module_checkpoint_state.named_parameters) {
    params.push_back(it.second);
  }
  return params;
}

std::unordered_map<std::string, std::shared_ptr<Parameter>> Module::NamedParameters() const {
  ORT_ENFORCE(!state_->module_checkpoint_state.is_nominal_state,
              "Cannot fetch named parameters from a nominal checkpoint state. Please load the model parameters first.");
  return state_->module_checkpoint_state.named_parameters;
}

Status Module::CopyParametersToBuffer(OrtValue& parameters_buffer, const bool trainable_only) {
  ORT_RETURN_IF(state_->module_checkpoint_state.is_nominal_state,
                "Cannot copy parameters from a nominal checkpoint state. Please load the model parameters first.");
  ORT_RETURN_IF_NOT(parameters_buffer.IsAllocated(), "Parameters buffer should be pre-allocated.");
  ORT_RETURN_IF_NOT(parameters_buffer.IsTensor(), "Parameters buffer should be of tensor type.");
  auto* init_tensor = parameters_buffer.GetMutable<Tensor>();
  ORT_ENFORCE(nullptr != init_tensor);
  auto expected_buffer_size = static_cast<int64_t>(GetParametersSize(trainable_only));
  ORT_RETURN_IF(init_tensor->Shape().Size() != expected_buffer_size,
                "Parameters buffer size incorrect. Expected:", expected_buffer_size,
                ", Actual:", init_tensor->Shape().Size());

  const DataTransferManager& sess_data_transfer_manager = train_sess_->GetDataTransferManager();

  size_t offset = 0;
  for (const auto& param_name : train_input_names_.WeightsInputNames()) {
    auto& param = state_->module_checkpoint_state.named_parameters.at(param_name);
    if (trainable_only && !param->RequiresGrad()) {
      continue;
    }
    OrtValue& weight = param->Data();
    auto* weight_tensor = weight.GetMutable<Tensor>();

    const TensorShape& shape = weight_tensor->Shape();
    auto element_type = init_tensor->DataType();
    ORT_RETURN_IF(weight_tensor->DataType() != element_type, "Data types must match.");

    const OrtMemoryInfo& info = init_tensor->Location();
    std::unique_ptr<Tensor> p_tensor;

    if (onnxruntime::utils::IsPrimitiveDataType<float>(element_type)) {
      float* data_buffer = init_tensor->MutableData<float>();
      p_tensor = std::make_unique<Tensor>(element_type,
                                          shape,
                                          data_buffer + offset,
                                          info);
    } else {
      ORT_THROW("Unsupported type: ", element_type, " encountered while copying parameters to buffer. ",
                "Only float is supported.");
    }
    ORT_RETURN_IF_ERROR(sess_data_transfer_manager.CopyTensor(*weight_tensor, *p_tensor.get()));
    offset += shape.Size();
  }
  return Status::OK();
}

Status Module::CopyBufferToParameters(OrtValue& parameters_buffer, const bool trainable_only) {
  // In case of a nominal checkpoint state, all parameters need to be loaded into the model.
  // i.e. trainable_only must be false.
  ORT_RETURN_IF(trainable_only && state_->module_checkpoint_state.is_nominal_state,
                "For nominal checkpoint state, all parameters need to be loaded into the model "
                "(trainable_only = false).");
  ORT_RETURN_IF_NOT(parameters_buffer.IsAllocated(), "Parameters buffer should be pre-allocated.");
  ORT_RETURN_IF_NOT(parameters_buffer.IsTensor(), "Parameters buffer should be of tensor type.");
  auto* buffer_tensor = parameters_buffer.GetMutable<Tensor>();
  ORT_RETURN_IF(nullptr == buffer_tensor, "Expected valid parameter buffer. Actual: nullptr.");
  auto expected_buffer_size = static_cast<int64_t>(GetParametersSize(trainable_only));
  ORT_RETURN_IF(buffer_tensor->Shape().Size() != expected_buffer_size,
                "Parameters buffer size incorrect. Expected:", expected_buffer_size,
                ", Actual:", buffer_tensor->Shape().Size());

  auto& train_sess_state = train_sess_->GetSessionState();
  const DataTransferManager& sess_data_transfer_manager = train_sess_->GetDataTransferManager();
  const auto model_inputs_with_error = GetTrainingModelInputs();
  ORT_RETURN_IF_ERROR(model_inputs_with_error.first);
  ORT_RETURN_IF_NOT(model_inputs_with_error.second, "Training model graph inputs are not defined.");
  const auto param_to_node_arg = BuildParameterToInputNodeArgMap(state_->module_checkpoint_state,
                                                                 model_inputs_with_error.second);
  const auto param_to_grad_index = BuildParameterToGradInputIndexMap(train_input_names_.GradientInputNames());

  if (state_->module_checkpoint_state.is_nominal_state) {
    // weights_ vector is not initialized for a nominal state. This function is expected to
    // initialize the weights_.
    ORT_ENFORCE(weights_.empty(), "Weights vector should be empty for a nominal state.");
  }

  size_t offset = 0;
  for (const auto& param_name : train_input_names_.WeightsInputNames()) {
    auto& param = state_->module_checkpoint_state.named_parameters.at(param_name);
    if (trainable_only && !param->RequiresGrad()) {
      continue;
    }
    OrtValue& weight = param->Data();

    auto param_it = param_to_node_arg.find(param_name);
    const TensorShape shape = onnxruntime::utils::GetTensorShapeFromTensorShapeProto(
        *(param_it->second->Shape()));
    const auto element_type = static_cast<const TensorTypeBase*>(
                                  onnxruntime::utils::GetMLDataType(*param_it->second))
                                  ->GetElementType();

    const OrtMemoryInfo& info = buffer_tensor->Location();
    std::unique_ptr<Tensor> src_tensor;

    if (onnxruntime::utils::IsPrimitiveDataType<float>(element_type)) {
      float* data_buffer = buffer_tensor->MutableData<float>();
      src_tensor = std::make_unique<Tensor>(element_type,
                                            shape,
                                            data_buffer + offset,
                                            info);
    } else {
      ORT_THROW("Unsupported type: ", element_type, " encountered while copying buffer to parameters. ",
                "Only float is supported.");
    }

    if (state_->module_checkpoint_state.is_nominal_state) {
      // If state is a nominal state, then we first need to allocate the memory for
      // parameters and their gradients in the checkpoint state before copying the data.
      ORT_RETURN_IF_ERROR(LoadParameter(param_name, *src_tensor, train_sess_state, true,
                                        param_to_grad_index, train_input_names_.GradientInputNames(),
                                        *param));
      weights_.push_back(param->Data());
      if (param->RequiresGrad()) {
        // It is expected that the gradients_ vector is already initialized with the correct size
        // in the Module constructor (even though the OrtValues contained in the vector are empty).
        gradients_[param_to_grad_index.at(param_name)] = param->Gradient();
      }
    } else {
      // If state is not a nominal state, then we can directly copy the data to the existing
      // parameters in the checkpoint state.
      auto* weight_tensor = weight.GetMutable<Tensor>();
      ORT_ENFORCE(weight_tensor->DataType() == element_type, "Data types must match.");
      ORT_THROW_IF_ERROR(sess_data_transfer_manager.CopyTensor(*src_tensor.get(), *weight_tensor));
    }

    offset += shape.Size();
  }

  if (state_->module_checkpoint_state.is_nominal_state) {
    // Once the parameters are loaded, the state is no longer a nominal state.
    state_->module_checkpoint_state.is_nominal_state = false;
  }

  return Status::OK();
}

Status Module::LazyResetGrad() {
  accumulate_gradient_ = false;
  return Status::OK();
}

Status Module::TrainStep(const std::vector<OrtValue>& inputs, std::vector<OrtValue>& outputs) {
  ORT_RETURN_IF(state_->module_checkpoint_state.is_nominal_state,
                "Cannot perform TrainStep with a nominal state. Please load the model parameters first.");
  std::vector<std::shared_ptr<Parameter>> params;
  std::vector<OrtValue> feeds{inputs};
  feeds.insert(feeds.end(), weights_.begin(), weights_.end());
  feeds.insert(feeds.end(), gradients_.begin(), gradients_.end());
  // TODO: consider maintaining this as ortvalue instead of bool
  OrtValue reset_grad_input;
  utils::WrapInOrtValue<bool>(!accumulate_gradient_, &reset_grad_input);
  feeds.push_back(reset_grad_input);

  ORT_THROW_IF_ERROR(train_sess_->Run(RunOptions(), train_input_names_.AllInputNames(), feeds, train_output_names_, &outputs));

  // Reset the flag after every step. In case the ResetGrad was called before running
  // the current step, it will have done the effective resetting during the
  // InPlaceAccumulator execution.
  accumulate_gradient_ = true;

  return Status::OK();
}

Status Module::EvalStep(const std::vector<OrtValue>& inputs, std::vector<OrtValue>& outputs) {
  ORT_RETURN_IF(state_->module_checkpoint_state.is_nominal_state,
                "Cannot perform EvalStep with a nominal state. Please load the model parameters first.");
  ORT_ENFORCE(nullptr != eval_sess_, "Evaluation session not initialized.");
  std::vector<OrtValue> feeds{inputs};
  feeds.insert(feeds.end(), weights_.begin(), weights_.end());
  auto status = eval_sess_->Run(RunOptions(), eval_input_names_, feeds, eval_output_names_, &outputs);
  ORT_THROW_IF_ERROR(status);
  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD)
// TODO (baijumeswani): ExportModelForInferencing should work irrespective of whether
//                      the build is minimal or not. This will require to read the ort_format eval model,
//                      transform it to an inference model and save it in ort_format.
Status Module::ExportModelForInferencing(const std::string& inference_model_path,
                                         gsl::span<const std::string> graph_output_names) const {
  ORT_RETURN_IF(state_->module_checkpoint_state.is_nominal_state,
                "Cannot export the model with a nominal state. Please load the model parameters first.");
  ORT_RETURN_IF(!eval_sess_ || !eval_model_path_.has_value(),
                "Eval model was not provided. Cannot export a model for inferencing.");

  ONNX_NAMESPACE::ModelProto eval_model;
  ORT_THROW_IF_ERROR(Model::Load(ToPathString(eval_model_path_.value()), eval_model));

  // Clone the eval mode into an inference onnxruntime::Model.
  std::shared_ptr<Model> inference_model;
  ORT_RETURN_IF_ERROR(Model::Load(eval_model, inference_model, nullptr, logging::LoggingManager::DefaultLogger()));

  // The cloned model's outputs are transformed such that the model has outputs as defined by graph_output_names
  // Any nodes not contributing to the inference outputs will be pruned.
  ORT_THROW_IF_ERROR(TransformModelOutputsForInference(inference_model->MainGraph(), graph_output_names));

  // The cloned model's inputs are transformed such that the model has only user defined inputs. All parameters
  // are moved to be constant initializers for the model.
  ORT_RETURN_IF_ERROR(TransformModelInputsForInference(inference_model->MainGraph(), state_->module_checkpoint_state.named_parameters,
                                                       eval_sess_->GetDataTransferManager()));

  // Save the model at the desired location.
  ORT_THROW_IF_ERROR(Model::Save(*inference_model, inference_model_path));
  return Status::OK();
}
#endif

size_t Module::GetTrainingModelInputCount() const noexcept {
  return train_input_names_.UserInputNames().size();
}

size_t Module::GetEvalModelInputCount() const noexcept {
  return eval_user_input_count_;
}

std::string Module::GetTrainingModelInputName(size_t index) const {
  ORT_ENFORCE(index < train_input_names_.UserInputNames().size(),
              "Train input name index out of range. Expected in range [0-", train_input_names_.UserInputNames().size(), "). Actual: ",
              index);
  return train_input_names_.UserInputNames()[index];
}

std::string Module::GetEvalModelInputName(size_t index) const {
  ORT_ENFORCE(index < eval_user_input_count_,
              "Eval input name index out of range. Expected in range [0-", eval_user_input_count_, "). Actual: ",
              index);
  return eval_input_names_.at(index);
}

std::pair<common::Status, const InputDefList*> Module::GetTrainingModelInputs() const noexcept {
  return train_sess_->GetModelInputs();
}

std::pair<common::Status, const InputDefList*> Module::GetEvalModelInputs() const noexcept {
  return eval_sess_->GetModelInputs();
}

Module::TrainInputNames::TrainInputNames(gsl::span<const std::string> user_input_names,
                                         gsl::span<const std::string> weights_input_names,
                                         gsl::span<const std::string> gradient_input_names) {
  train_input_names_.reserve(user_input_names.size() +
                             weights_input_names.size() +
                             gradient_input_names.size() +
                             1U);  // +1 for the reset gradient flag input
  train_input_index_offsets_.reserve(3);

  train_input_names_.insert(train_input_names_.end(),
                            user_input_names.begin(), user_input_names.end());
  train_input_index_offsets_.push_back(train_input_names_.size());
  train_input_names_.insert(train_input_names_.end(),
                            weights_input_names.begin(), weights_input_names.end());
  train_input_index_offsets_.push_back(train_input_names_.size());
  train_input_names_.insert(train_input_names_.end(),
                            gradient_input_names.begin(), gradient_input_names.end());
  train_input_index_offsets_.push_back(train_input_names_.size());
  train_input_names_.push_back(ACCUMULATE_GRAD_CONTROL_INPUT_NAME);
}

gsl::span<const std::string> Module::TrainInputNames::AllInputNames() const { return train_input_names_; }

gsl::span<const std::string> Module::TrainInputNames::UserInputNames() const {
  return gsl::span<const std::string>{train_input_names_.begin(), train_input_index_offsets_[0]};
}

gsl::span<const std::string> Module::TrainInputNames::WeightsInputNames() const {
  return gsl::span<const std::string>{train_input_names_.begin() + train_input_index_offsets_[0],
                                      train_input_index_offsets_[1] - train_input_index_offsets_[0]};
}

gsl::span<const std::string> Module::TrainInputNames::GradientInputNames() const {
  return gsl::span<const std::string>{train_input_names_.begin() + train_input_index_offsets_[1],
                                      train_input_index_offsets_[2] - train_input_index_offsets_[1]};
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
