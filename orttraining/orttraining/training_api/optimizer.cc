// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_api/optimizer.h"
#include "core/flatbuffers/flatbuffers_utils.h"
#include "core/framework/execution_provider.h"
#include "core/framework/TensorSeq.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/session/environment.h"

#include "orttraining/training_api/checkpoint.h"
#include "orttraining/training_api/utils.h"

namespace onnxruntime {
namespace training {
namespace api {

namespace {

constexpr char GROUP_ZERO_NAME[] = "group0";
static constexpr std::array CommonOptimizerInputs{"learning_rate", "step", "params", "gradients"};

Status GraphInputsAreExpected(gsl::span<std::string> actual_graph_inputs,
                              gsl::span<std::string> expected_graph_inputs) {
  const auto stringify = [](const auto& container) {
    if (container.empty()) {
      return std::string("[]");
    }
    std::string container_str("[");
    for (const auto& val : container) {
      container_str += std::string(val) + ", ";
    }
    container_str.pop_back();
    container_str.back() = ']';

    return container_str;
  };

  const auto construct_unexpected_input_status = [&stringify](const auto& actual_inputs, const auto& expected_inputs) {
    std::ostringstream error_stream;
    error_stream << "Invalid graph inputs."
                 << "\n\tExpected: " << stringify(expected_inputs)
                 << "\n\tActual: " << stringify(actual_inputs);
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, error_stream.str());
  };

  if (actual_graph_inputs.size() != expected_graph_inputs.size()) {
    return construct_unexpected_input_status(actual_graph_inputs, expected_graph_inputs);
  }

  for (size_t input_idx = 0; input_idx < expected_graph_inputs.size(); ++input_idx) {
    if (actual_graph_inputs[input_idx] != expected_graph_inputs[input_idx]) {
      return construct_unexpected_input_status(actual_graph_inputs, expected_graph_inputs);
    }
  }

  return Status::OK();
}

}  // namespace

std::unique_ptr<OptimizerAlgorithmBase> OptimizerAlorithmFactory::CreateInstance(
    std::shared_ptr<Model> model, int32_t& group_count) {
  std::map<std::pair<std::string, std::string>, int32_t> opt_type_to_freq_map;
#if !defined(ORT_MINIMAL_BUILD)
  if (model != nullptr) {
    Graph& graph = model->MainGraph();
    for (auto& node : graph.Nodes()) {
      if (node.Domain() == kMSDomain && (node.OpType() == "AdamWOptimizer" || node.OpType() == "SGDOptimizerV2")) {
        auto domain_type_pair = std::make_pair(node.Domain(), node.OpType());
        if (opt_type_to_freq_map.find(domain_type_pair) == opt_type_to_freq_map.end()) {
          opt_type_to_freq_map[domain_type_pair] = 0;
        }

        opt_type_to_freq_map[domain_type_pair] += 1;
      }
    }
  } else {
#else
  ORT_UNUSED_PARAMETER(model);
#endif
    // TODO(baijumeswani): Figure out the best way to extract the optimizer type
    // from the model (either onnx model or ort format model) or from the checkpoint.
    // For now, assume that the optimizer type is AdamWOptimizer when using ort format models.
    opt_type_to_freq_map[std::make_pair(kMSDomain, "AdamWOptimizer")] = 1;
#if !defined(ORT_MINIMAL_BUILD)
  }
#endif

  ORT_ENFORCE(opt_type_to_freq_map.size() == 1U, "Only support one type of optimizer algorithm, but got: " +
                                                     std::to_string(opt_type_to_freq_map.size()));
  auto opt_it = opt_type_to_freq_map.begin();
  auto& op_type = opt_it->first.second;
  group_count = opt_it->second;
  ORT_ENFORCE(group_count == 1, "Group count can only be 1, but got: " + std::to_string(group_count));

  // TODO: to support multiple groups, need to create a mapping between each group to its parameter list.
  if (op_type == "AdamWOptimizer") {
    return std::make_unique<AdamWOptimizerAlgorithm>();
  } else if (op_type == "SGDOptimizerV2") {
    return std::make_unique<SGDOptimizerV2Algorithm>();
  } else {
    ORT_NOT_IMPLEMENTED("Not implemented for optimizer algo: " + opt_it->first.second);
  }
}

std::unique_ptr<OptimizerAlgorithmBase> OptimizerAlorithmFactory::CreateInstance(
    const PathString& optim_path, int32_t& group_count) {
  std::shared_ptr<Model> model = nullptr;
#if !defined(ORT_MINIMAL_BUILD)
  if (!fbs::utils::IsOrtFormatModel(optim_path)) {
    ORT_ENFORCE(Model::Load(optim_path, model, nullptr,
                            logging::LoggingManager::DefaultLogger())
                    .IsOK());
  }
#else
  ORT_UNUSED_PARAMETER(optim_path);
#endif
  return CreateInstance(model, group_count);
}

std::unique_ptr<OptimizerAlgorithmBase> OptimizerAlorithmFactory::CreateInstance(
    const uint8_t* optim_model_data, size_t optim_model_data_len, int32_t& group_count) {
  std::shared_ptr<Model> model = nullptr;
#if !defined(ORT_MINIMAL_BUILD)
  if (!fbs::utils::IsOrtFormatModelBytes(optim_model_data, static_cast<int>(optim_model_data_len))) {
    ONNX_NAMESPACE::ModelProto model_proto;
    ORT_ENFORCE(model_proto.ParseFromArray(optim_model_data, static_cast<int>(optim_model_data_len)) == true,
                "Failed to load model because protobuf parsing failed.");

    ORT_ENFORCE(Model::Load(std::move(model_proto), model, nullptr,
                            logging::LoggingManager::DefaultLogger(), ModelOptions(true, true))
                    .IsOK());
  }
#else
  ORT_UNUSED_PARAMETER(optim_model_data);
  ORT_UNUSED_PARAMETER(optim_model_data_len);
#endif

  return CreateInstance(model, group_count);
}

Status Optimizer::GenerateMomentumNamedStates(OptimizerCheckpointState& optimizer_checkpoint_states) {
  auto group_optimizer_state_it =
      optimizer_checkpoint_states.group_named_optimizer_states.find(GROUP_ZERO_NAME);
  ORT_ENFORCE(group_optimizer_state_it != optimizer_checkpoint_states.group_named_optimizer_states.end(),
              "Group 0 not found in the optimizer checkpoint states.");

  optimizer_state_ = group_optimizer_state_it->second;

  auto& param_named_optimizer_states = optimizer_state_->param_named_optimizer_states;
  auto& optim_sess_state = optim_sess_->GetSessionState();
  for (auto& pair : state_->module_checkpoint_state.named_parameters) {
    if (pair.second->RequiresGrad()) {
      param_named_optimizer_states.insert({pair.first, ParameterOptimizerState()});
      ParameterOptimizerState& cur_param_optimizer_states = param_named_optimizer_states[pair.first];
      for (auto& state_name : optimizer_algo_ptr_->momentum_keys) {
        OrtValue param_state;
        ORT_ENFORCE(utils::CreateZeroValuedOrtValueLike(optim_sess_state, pair.second->Data(), param_state).IsOK(),
                    "Error generating moment state for ", pair.first);
        cur_param_optimizer_states.insert({state_name, std::move(param_state)});
      }
    }
  }

  return Status::OK();
}

// Constructs the ortvalue inputs to be fed to the graph at each step
Status Optimizer::ConstructInputs() {
  inputs_.clear();

  auto& param_named_optimizer_states = optimizer_state_->param_named_optimizer_states;

  InlinedVector<Tensor> params, grads;
  InlinedVector<InlinedVector<Tensor>> list_of_momentums;
  list_of_momentums.resize(optimizer_algo_ptr_->momentum_keys.size());

  // Collect all the non-user-defined inputs from the named_parameters_.
  for (auto& [parameter_name, parameter] : state_->module_checkpoint_state.named_parameters) {
    if (parameter->RequiresGrad()) {
      // Collect parameters and prepare for tensorseq creation
      auto* param_tensor = parameter->Data().GetMutable<Tensor>();
      params.emplace_back(
          Tensor(param_tensor->DataType(), param_tensor->Shape(),
                 param_tensor->MutableDataRaw(), param_tensor->Location()));

      // Collect gradients and prepare for tensorseq creation
      auto* grad_tensor = parameter->Gradient().GetMutable<Tensor>();
      grads.emplace_back(
          Tensor(grad_tensor->DataType(), grad_tensor->Shape(),
                 grad_tensor->MutableDataRaw(), grad_tensor->Location()));

      // Collect moments and prepare for tensorseq creation
      for (size_t m_index = 0; m_index < optimizer_algo_ptr_->momentum_keys.size(); ++m_index) {
        auto* moment_tensor =
            param_named_optimizer_states.at(parameter_name)
                .at(optimizer_algo_ptr_->momentum_keys[m_index])
                .GetMutable<Tensor>();
        list_of_momentums[m_index].emplace_back(
            Tensor(moment_tensor->DataType(), moment_tensor->Shape(),
                   moment_tensor->MutableDataRaw(), moment_tensor->Location()));
      }
    }
  }

  const auto tensorseq_inserter = [](auto& tensors, auto* inputs) {
    ORT_ENFORCE(!tensors.empty(), "Tensors vector cannot be empty while building a tensor sequence.");

    auto tensor_seq = std::make_unique<TensorSeq>(tensors.front().DataType());
    tensor_seq->Reserve(tensors.size());
    for (auto& tensor : tensors) {
      tensor_seq->Add(std::move(tensor));
    }
    inputs->emplace_back(
        OrtValue(tensor_seq.release(), DataTypeImpl::GetType<TensorSeq>(),
                 DataTypeImpl::GetType<TensorSeq>()->GetDeleteFunc()));
  };

  // Add the params/grads as tensorseq ortvalues to inputs
  tensorseq_inserter(params, &inputs_);
  tensorseq_inserter(grads, &inputs_);
  // Add all other momentums as tensorseq ortvalues to inputs.
  for (auto& m : list_of_momentums) {
    tensorseq_inserter(m, &inputs_);
  }

  return Status::OK();
}  // namespace api

Optimizer::Optimizer(const ModelIdentifiers& model_identifiers,
                     CheckpointState* state,
                     const onnxruntime::SessionOptions& session_options,
                     const Environment& env,
                     const std::vector<std::shared_ptr<IExecutionProvider>>& providers,
                     gsl::span<OrtCustomOpDomain* const> op_domains)
    : optim_sess_(std::make_unique<InferenceSession>(session_options, env)), state_(state) {
  Initialize(model_identifiers, providers, op_domains);

  ORT_ENFORCE(state != nullptr, "Checkpoint state cannot be null.");
  auto g_it = state_->optimizer_checkpoint_state.group_named_optimizer_states.find(GROUP_ZERO_NAME);
  bool find_group_zero = g_it != state_->optimizer_checkpoint_state.group_named_optimizer_states.end();
  if (!find_group_zero || g_it->second->param_named_optimizer_states.empty()) {
    if (!find_group_zero)
      state_->optimizer_checkpoint_state.group_named_optimizer_states.insert(
          {GROUP_ZERO_NAME, std::make_shared<GroupOptimizerState>()});
    ORT_THROW_IF_ERROR(GenerateMomentumNamedStates(state_->optimizer_checkpoint_state));
    ORT_THROW_IF_ERROR(ConstructInputs());
  } else {
    ORT_THROW_IF_ERROR(LoadStateDict(state_->optimizer_checkpoint_state));
  }
}

void Optimizer::Initialize(const ModelIdentifiers& model_identifiers,
                           const std::vector<std::shared_ptr<IExecutionProvider>>& providers,
                           [[maybe_unused]] gsl::span<OrtCustomOpDomain* const> op_domains) {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_MINIMAL_BUILD_CUSTOM_OPS)
  if (!op_domains.empty()) {
    ORT_THROW_IF_ERROR(optim_sess_->AddCustomOpDomains(op_domains));
  }
#endif

  for (const auto& execution_provider : providers) {
    ORT_THROW_IF_ERROR(optim_sess_->RegisterExecutionProvider(execution_provider));
  }

  ORT_ENFORCE(model_identifiers.IsOptimizerModelAvailable(), "Optimizer model is not available.");

  if (std::holds_alternative<std::optional<std::string>>(model_identifiers.optim_model)) {
    auto optimizer_model = std::get<std::optional<std::string>>(model_identifiers.optim_model);
    // The above call to IsOptimizerModelAvailable() ensures that optimizer_model is not nullopt
    ORT_THROW_IF_ERROR(optim_sess_->Load(optimizer_model.value()));
    optimizer_algo_ptr_ = OptimizerAlorithmFactory::CreateInstance(ToWideString(optimizer_model.value()), group_count_);
  } else {
    auto optimizer_model = std::get<gsl::span<const uint8_t>>(model_identifiers.optim_model);
    ORT_THROW_IF_ERROR(optim_sess_->Load(optimizer_model.data(),
                                         static_cast<int>(optimizer_model.size())));
    optimizer_algo_ptr_ = OptimizerAlorithmFactory::CreateInstance(optimizer_model.data(),
                                                                   optimizer_model.size(),
                                                                   group_count_);
  }

  ORT_THROW_IF_ERROR(optim_sess_->Initialize());

  // Make sure that the checkpoint state can copy tensors
  state_->optimizer_checkpoint_state.optimizer_session_data_transfer_mgr = &optim_sess_->GetDataTransferManager();

  utils::GetGraphInputOutputNames(optim_sess_, input_names_, output_names_);

  InlinedVector<std::string> all_input_names;
  all_input_names.reserve(CommonOptimizerInputs.size() + optimizer_algo_ptr_->optimizer_states_inputs.size());
  all_input_names.insert(all_input_names.end(), CommonOptimizerInputs.begin(),
                         CommonOptimizerInputs.end());
  all_input_names.insert(all_input_names.end(), optimizer_algo_ptr_->optimizer_states_inputs.begin(),
                         optimizer_algo_ptr_->optimizer_states_inputs.end());
  ORT_THROW_IF_ERROR(GraphInputsAreExpected(input_names_, all_input_names));
}

Status Optimizer::Step() {
  OrtValue learning_rate_input, step_input;
  utils::WrapInOrtValue<float>(optimizer_state_->learning_rate, &learning_rate_input);
  // Use step count + 1 before running optimizer step.
  // This is necessary since bias correction uses the step
  // as a power. Using the power of 0 is wrong.
  utils::WrapInOrtValue<int64_t>(optimizer_state_->step + 1, &step_input);
  std::vector<OrtValue> feeds({learning_rate_input, step_input});
  feeds.insert(feeds.end(), inputs_.begin(), inputs_.end());

  std::vector<OrtValue> outputs;
  auto status = optim_sess_->Run(RunOptions(), input_names_, feeds, output_names_, &outputs);
  ORT_THROW_IF_ERROR(status);

  // Extract step output and update
  if (utils::GetScalarFromOrtValue<bool>(outputs[0]) == true) {
    optimizer_state_->step++;
  }

  return Status::OK();
}

Status Optimizer::LoadStateDict(OptimizerCheckpointState& optimizer_checkpoint_states) {
  auto group_optimizer_state_it =
      optimizer_checkpoint_states.group_named_optimizer_states.find(GROUP_ZERO_NAME);
  ORT_ENFORCE(group_optimizer_state_it != optimizer_checkpoint_states.group_named_optimizer_states.end(),
              "Group 0 not found in the optimizer checkpoint states.");

  optimizer_state_ = group_optimizer_state_it->second;
  constexpr bool strict_match = true;

  ORT_RETURN_IF_NOT(optim_sess_, "optimizer session not initialized");
  auto& optim_sess_state = optim_sess_->GetSessionState();
  auto& param_named_optimizer_states = optimizer_state_->param_named_optimizer_states;

  for (auto& params_iter : state_->module_checkpoint_state.named_parameters) {
    if (params_iter.second->RequiresGrad()) {
      bool src_exist = param_named_optimizer_states.find(params_iter.first) !=
                       param_named_optimizer_states.cend();

      ORT_ENFORCE(src_exist || !strict_match, "Parameter ", params_iter.first,
                  " not found in the source optimizer checkpoint states.");

      InlinedHashMap<std::string, OrtValue>& momentum_named_states =
          param_named_optimizer_states.at(params_iter.first);

      OrtValue& param_data = params_iter.second->Data();
      ORT_ENFORCE(param_data.IsTensor());
      const Tensor& param_data_tensor = param_data.Get<Tensor>();
      const auto& param_data_device = param_data_tensor.Location().device;
      auto target_allocator = optim_sess_state.GetAllocator(param_data_device);
      ORT_ENFORCE(target_allocator != nullptr);

      for (auto& momentum_state_pair : momentum_named_states) {
        OrtValue& param_momentum = momentum_state_pair.second;
        ORT_ENFORCE(param_momentum.IsTensor());
        const Tensor& param_momentum_tensor = param_momentum.Get<Tensor>();

        // If the source device type is already the same as the target device skip copy.
        if (param_momentum_tensor.Location().device.Type() != param_data_device.Type()) {
          // Create a new tensor on the target_device and switch the source_ortvalue to point to this new tensor
          auto target_tensor = std::make_unique<Tensor>(param_momentum_tensor.DataType(),
                                                        param_momentum_tensor.Shape(),
                                                        target_allocator);
          ORT_THROW_IF_ERROR(optim_sess_state.GetDataTransferMgr().CopyTensor(param_momentum_tensor,
                                                                              *target_tensor.get()));
          auto ml_tensor_type = DataTypeImpl::GetType<Tensor>();
          param_momentum.Init(target_tensor.release(), ml_tensor_type, ml_tensor_type->GetDeleteFunc());
        }
      }
    }
  }

  ORT_THROW_IF_ERROR(ConstructInputs());

  return Status::OK();
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
