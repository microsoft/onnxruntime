// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/execution_provider.h"
#include "core/session/inference_session.h"
#include "core/session/environment.h"

#include "orttraining/training_api/include/module.h"
#include "orttraining/training_api/include/utils.h"

using namespace onnxruntime;

namespace onnxruntime {
namespace training {
namespace api {

namespace {

// TODO: consolidate with frontend tooling
const std::string ACCUMULATE_GRAD_CONTROL_INPUT_NAME{"lazy_reset_grad"};

}  // namespace

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

Module::Module(const std::string& train_model_path_or_bytes,
               const std::unordered_map<std::string, std::shared_ptr<Parameter>>& named_parameters,
               const onnxruntime::SessionOptions& session_options,
               const Environment& env,
               const std::vector<std::shared_ptr<IExecutionProvider>>& providers,
               const std::optional<std::string>& eval_model_path_or_bytes)
    : named_parameters_{named_parameters} {
  train_sess_ = std::make_unique<onnxruntime::InferenceSession>(session_options, env);
  ORT_THROW_IF_ERROR(train_sess_->Load(train_model_path_or_bytes));
  for (const auto& provider : providers) {
    ORT_THROW_IF_ERROR(train_sess_->RegisterExecutionProvider(provider));
  }
  ORT_THROW_IF_ERROR(train_sess_->Initialize());

  // Extract model input and output names
  std::vector<std::string> train_input_names, train_output_names;
  utils::GetGraphInputOutputNames(train_sess_, train_input_names, train_output_names);

  // Reorder the extracted input names in the following order:
  // user inputs, weights, gradients, reset_grad
  std::vector<std::string> user_input_names, param_input_names, grad_input_names, reset_grad_name;
  std::string name;

  std::unordered_map<std::string, size_t> param_name_to_grad_input_index_map;
  for (const auto& input_name : train_input_names) {
    auto it = named_parameters_.find(input_name);
    if (it != named_parameters_.end()) {
      param_input_names.emplace_back(input_name);
    } else if (input_name == ACCUMULATE_GRAD_CONTROL_INPUT_NAME) {
      reset_grad_name.emplace_back(input_name);
    } else if (utils::GetParamNameFromGradient(input_name, name)) {
      param_name_to_grad_input_index_map.insert({name, grad_input_names.size()});
      grad_input_names.emplace_back(input_name);
    } else {
      user_input_names.emplace_back(input_name);
    }
  }

  gradients_.resize(grad_input_names.size());

  train_input_names_ = user_input_names;
  train_input_names_.insert(train_input_names_.end(), param_input_names.begin(), param_input_names.end());
  train_input_names_.insert(train_input_names_.end(), grad_input_names.begin(), grad_input_names.end());
  train_input_names_.insert(train_input_names_.end(), reset_grad_name.begin(), reset_grad_name.end());

  for (const auto& output_name : train_output_names) {
    if (!utils::GetParamNameFromGradient(output_name, name)) {
      train_output_names_.emplace_back(output_name);
    }
  }

  // Loop each parameter, allocate it's memory based on user specified device.
  auto& train_sess_state = train_sess_->GetSessionState();
  for (auto& param_name : param_input_names) {
    auto params_iter = named_parameters_.find(param_name);
    ORT_ENFORCE(params_iter != named_parameters_.end());

    // Retrieve the target device for "param_name"
    InlinedVector<SessionState::NodeInfo> node_info_vec;
    ORT_THROW_IF_ERROR(train_sess_state.GetInputNodeInfo(param_name, node_info_vec));
    const auto& node_info = node_info_vec.front();
    const auto target_device = *node_info.device;
    for (auto it = node_info_vec.begin(); it != node_info_vec.end(); ++it) {
      ORT_ENFORCE(target_device == *(it->device), "Inconsistent device requirements found for input: ", param_name);
    }

    // TODO(pengwa): consider whether we should alloc contiguous buffer for parameters or gradients.
    // Copy ortvalue buffer from CPU to target_device for this "param_name" (based on graph partitioning)
    // Only copies data if target device is not the same as the current device the buffer is placed on

    OrtValue& param_data = params_iter->second->Data();
    ORT_ENFORCE(param_data.IsTensor());
    const Tensor& param_data_tensor = param_data.Get<Tensor>();
    // If the source device type is already same as target device skip copy
    if (param_data_tensor.Location().device.Type() != target_device.Type()) {
      // TODO: move this outside of the for loop?
      auto target_allocator = train_sess_state.GetAllocator(target_device);
      ORT_ENFORCE(target_allocator != nullptr);

      // Create a new tensor on the target_device and switch the source_ortvalue to point to this new tensor
      auto target_tensor = std::make_unique<Tensor>(param_data_tensor.DataType(), param_data_tensor.Shape(),
                                                    target_allocator);
      ORT_THROW_IF_ERROR(train_sess_state.GetDataTransferMgr().CopyTensor(param_data_tensor, *target_tensor.get()));
      auto ml_tensor_type = DataTypeImpl::GetType<Tensor>();
      param_data.Init(target_tensor.release(), ml_tensor_type, ml_tensor_type->GetDeleteFunc());
    }

    weights_.push_back(param_data);

    // Create gradient buffer when parameter requires gradient.
    if (params_iter->second->RequiresGrad()) {
      // Create gradient accumulation buffer.
      auto it = param_name_to_grad_input_index_map.find(param_name);
      ORT_ENFORCE(it != param_name_to_grad_input_index_map.end(), "Gradient buffer input not providered for param: ",
                  param_name);

      const size_t grad_input_index = it->second;
      auto& param_grad_name = grad_input_names[grad_input_index];
      // TODO: don't pre-allocate the gradient buffer.
      // Gradient usually stays on the same device of its parameter.
      OrtValue param_grad;
      ORT_THROW_IF_ERROR(utils::OrtValueLike(train_sess_state, param_data, param_grad));
      ORT_THROW_IF_ERROR(params_iter->second->SetGrad(param_grad_name, param_grad));
      gradients_[grad_input_index] = params_iter->second->Gradient();
    }
  }

  if (eval_model_path_or_bytes.has_value()) {
    eval_sess_ = std::make_unique<onnxruntime::InferenceSession>(session_options, env);
    ORT_THROW_IF_ERROR(eval_sess_->Load(eval_model_path_or_bytes.value()));
    ORT_THROW_IF_ERROR(eval_sess_->Initialize());
    utils::GetGraphInputOutputNames(eval_sess_, eval_input_names_, eval_output_names_);

    // Eval model validation
    // We are making certain assumptions: Like the order in which parameters occur will be same between train and eval
    // graphs, and all the weights present in both graphs match.
    // TODO: Add the checks instead of making assumptions??
    std::vector<std::string> eval_user_input_names, eval_param_input_names;
    for (const auto& input_name : eval_input_names_) {
      if (named_parameters_.find(input_name) != named_parameters_.end()) {
        // it is a parameter
        eval_param_input_names.emplace_back(input_name);
        continue;
      } else {
        // It is a user input. We handle user inputs separately in eval
        // because eval graph might have different user inputs.
        // Eg if loss is not a part of eval graph, it won't have
        // certain inputs like targets
        eval_user_input_names.emplace_back(input_name);
      }
    }
    eval_input_names_ = eval_user_input_names;
    eval_input_names_.insert(eval_input_names_.end(), eval_param_input_names.begin(), eval_param_input_names.end());
  }
}

size_t Module::GetTrainModeOutputCount() const noexcept {
  return train_output_names_.size();
}

size_t Module::GetEvalModeOutputCount() const noexcept {
  return eval_output_names_.size();
}

std::vector<std::shared_ptr<Parameter>> Module::Parameters() const {
  std::vector<std::shared_ptr<Parameter>> params;
  for (auto& it : named_parameters_) {
    params.push_back(it.second);
  }
  return params;
}

Status Module::ResetGrad() {
  accumulate_gradient_ = false;
  return Status::OK();
}

Status Module::TrainStep(const std::vector<OrtValue>& inputs, std::vector<OrtValue>& outputs) {
  std::vector<OrtValue> feeds{inputs};
  feeds.insert(feeds.end(), weights_.begin(), weights_.end());
  feeds.insert(feeds.end(), gradients_.begin(), gradients_.end());
  // TODO: consider maintaining this as ortvalue instead of bool
  OrtValue reset_grad_input;
  utils::WrapInOrtValue<bool>(!accumulate_gradient_, &reset_grad_input);
  feeds.push_back(reset_grad_input);

  auto status = train_sess_->Run(RunOptions(), train_input_names_, feeds, train_output_names_, &outputs);
  ORT_THROW_IF_ERROR(status);

  // Reset the flag after every step. In case the ResetGrad was called before running
  // the current step, it will have done the effective resetting during the
  // InPlaceAccumulator execution.
  accumulate_gradient_ = true;

  return Status::OK();
}

Status Module::EvalStep(const std::vector<OrtValue>& inputs, std::vector<OrtValue>& outputs) {
  ORT_ENFORCE(nullptr != eval_sess_, "Evaluation session not initialized.");
  std::vector<OrtValue> feeds{inputs};
  feeds.insert(feeds.end(), weights_.begin(), weights_.end());
  auto status = eval_sess_->Run(RunOptions(), eval_input_names_, feeds, eval_output_names_, &outputs);
  ORT_THROW_IF_ERROR(status);
  return Status::OK();
}

Status Module::GetStateDict(ModuleCheckpointState& module_checkpoint_state) {
  module_checkpoint_state.named_parameters = NamedParameters();

  // Pass the training session data transfer manager for data copying when saving.
  // An alternative is, we can do copy at this stage.
  ORT_RETURN_IF_NOT(train_sess_, "training session not initialized");
  const DataTransferManager& sess_data_transfer_manager = train_sess_->GetDataTransferManager();
  module_checkpoint_state.train_session_data_transfer_mgr = &sess_data_transfer_manager;
  return Status::OK();
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
