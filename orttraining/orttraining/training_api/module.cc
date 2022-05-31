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

// TODO: consolidate with frontend tooling
const std::string ACCUMULATE_GRAD_CONTROL_INPUT_NAME{"lazy_reset_grad"};

Status Parameter::AllocateGrad(const std::string& gradient_name, const OrtValue& param_grad) {
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

Module::Module(std::unordered_map<std::string, std::shared_ptr<Parameter>>& named_parameters,
               InferenceSession* train_session) {
  train_sess_ = train_session;
  utils::GetGraphInputOutputNames(train_sess_, train_input_names_, train_output_names_);

  auto& train_sess_state = train_sess_->GetSessionState();
  std::vector<std::string> param_input_names, grad_input_names, user_input_names, reset_grad_name;
  std::string param_name;

  std::unordered_map<std::string, size_t> param_name_to_grad_input_index_map;
  for (const auto& input_name : train_input_names_) {
    auto it = named_parameters.find(input_name);
    if (it != named_parameters.end()) {
      param_input_names.emplace_back(input_name);
    } else if (input_name == ACCUMULATE_GRAD_CONTROL_INPUT_NAME) {
      reset_grad_name.emplace_back(input_name);
    } else if (utils::GetParamNameFromGradient(input_name, param_name)) {
      param_name_to_grad_input_index_map.insert({param_name, grad_input_names.size()});
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

  // Loop each parameter, allocate it's memory based on user specified device.
  for (auto& param_name : param_input_names) {
    ORT_ENFORCE(named_parameters.find(param_name) != named_parameters.end());
    OrtValue& source_ortvalue = named_parameters[param_name]->Data();
    ORT_ENFORCE(source_ortvalue.IsTensor());
    const Tensor& source_tensor = source_ortvalue.Get<Tensor>();

    std::vector<SessionState::NodeInfo> node_info_vec;
    ORT_THROW_IF_ERROR(train_sess_state.GetInputNodeInfo(param_name, node_info_vec));
    const auto& node_info = node_info_vec.front();
    const auto target_device = *node_info.device;
    for (auto it = node_info_vec.begin(); it != node_info_vec.end(); ++it) {
      ORT_ENFORCE(target_device == *(it->device));
    }

    // Create parameter value copy with corresponding device user sets the session on.
    // We did not re-use the data even CPU tensor is needed.
    // TODO(pengwa): consider whether we should alloc contiguous buffer for parameters or gradients.
    OrtValue target_ortvalue;
    auto allocator = train_sess_state.GetAllocator(target_device);
    ORT_ENFORCE(allocator != nullptr);

    Tensor::InitOrtValue(source_tensor.DataType(),
                         source_tensor.Shape(),
                         allocator, target_ortvalue);
    Tensor* target_tensor_ptr = target_ortvalue.GetMutable<Tensor>();
    ORT_THROW_IF_ERROR(train_sess_state.GetDataTransferMgr().CopyTensor(source_tensor, *target_tensor_ptr));

    auto param_share_ptr = std::make_shared<Parameter>(param_name, target_ortvalue, named_parameters[param_name]->RequiresGrad());
    named_parameters_.insert({param_name, param_share_ptr});
    weights_.push_back(param_share_ptr->Data());

    // Create gradient buffer when paramter requires gradient.
    if (param_share_ptr->RequiresGrad()) {
      // Create gradient accumulation buffer.
      auto it = param_name_to_grad_input_index_map.find(param_name);
      ORT_ENFORCE(it != param_name_to_grad_input_index_map.end(), "Gradient buffer input not providered for param: ", param_name);

      auto& param_grad_buffer_name = grad_input_names[param_name_to_grad_input_index_map[param_name]];
      // TODO: don't pre-allocate the gradient buffer.
      // Gradient usually stays on the same device of its parameter.
      OrtValue param_grad_buffer_ortvalue;
      ORT_THROW_IF_ERROR(utils::OrtValueLike(train_sess_state, target_ortvalue, param_grad_buffer_ortvalue));
      ORT_THROW_IF_ERROR(param_share_ptr->AllocateGrad(param_grad_buffer_name, param_grad_buffer_ortvalue));
      gradients_[param_name_to_grad_input_index_map[param_name]] = param_share_ptr->Gradient();
    }
  }
}

Module::Module(std::unordered_map<std::string, std::shared_ptr<Parameter>>& named_parameters,
               InferenceSession* train_session,
               InferenceSession* eval_session) : Module(named_parameters, train_session) {
  if (eval_session) {
    eval_sess_ = eval_session;
    utils::GetGraphInputOutputNames(eval_sess_, eval_input_names_, eval_output_names_);

    // Eval model validation
    // We are making certain assumptions: Like the order in which parameters
    // occur will be same between train and eval graphs,
    // and all the weights present in both graphs match.
    std::vector<std::string> eval_user_input_names, param_input_names;
    for (const auto& input_name : eval_input_names_) {
      if (named_parameters_.find(input_name) != named_parameters_.end()) {
        // it is a parameter
        param_input_names.emplace_back(input_name);
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
    eval_input_names_.insert(eval_input_names_.end(), param_input_names.begin(), param_input_names.end());
  }
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
  OrtValue do_update_input;
  utils::WarpInOrtValue<bool>(accumulate_gradient_, &do_update_input);
  feeds.push_back(do_update_input);

  // TODO: need to filter out the grads from the output ortvalues
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
