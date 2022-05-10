// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/inference_session.h"
#include "core/session/environment.h"

#include "orttraining/training_api/include/optimizer.h"

namespace onnxruntime {
namespace training {
namespace api {

namespace {

Status CreateOrtValueFromOrtValue(
    const OrtValue& src_ort_value,
    OrtValue& dest_ort_value,
    onnxruntime::InferenceSession* sess) {
  const Tensor& tensor = src_ort_value.Get<Tensor>();
  AllocatorPtr allocator = sess->GetAllocator(tensor.Location());
  const TensorShape& tensor_shape = tensor.Shape();
  MLDataType element_type = tensor.DataType();

  auto p_tensor = std::make_unique<Tensor>(element_type, tensor_shape, allocator);
  dest_ort_value.Init(p_tensor.release(), DataTypeImpl::GetType<Tensor>(), DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  return Status::OK();
}

}  // namespace

Optimizer::Optimizer(const std::string& optim_path_or_bytes,
                     const std::unordered_map<std::string, std::shared_ptr<Parameter>>& parameters) {
  std::unordered_map<std::string, ParameterOptimizerState>&
      param_named_optimizer_states = optimizer_state_.param_named_optimizer_states;

  const SessionOptions session_options;
  std::unique_ptr<Environment> env;
  ORT_ENFORCE(Environment::Create(nullptr, env) == Status::OK(), "Enviroment creation fails.");
  optim_sess_ = std::move(std::make_unique<InferenceSession>(session_options, *env));

  ORT_ENFORCE(optim_sess_->Load(optim_path_or_bytes).IsOK());
  ORT_ENFORCE(optim_sess_->Initialize().IsOK());

  // TODO: don't hard code the state names, should get the state names according to the optimizer types.
  std::vector<std::string> state_names{"momentum0", "momentum1"};
  for (auto& pair : parameters) {
    if (pair.second->RequiresGrad()) {
      param_named_optimizer_states.insert({pair.first, ParameterOptimizerState()});
      ParameterOptimizerState& cur_param_optimizer_states = param_named_optimizer_states[pair.first];
      for (auto& state_name : state_names) {
        OrtValue param_state;
        // TODO: should reset the state to zero (for both CPU or CUDA Tensors.)
        ORT_ENFORCE(CreateOrtValueFromOrtValue(pair.second->Data(), param_state, optim_sess_.get()).IsOK());
        cur_param_optimizer_states.momentum_named_states.insert({state_name, std::make_shared<OrtValue>(param_state)});
      }
    }
  }
}

Status Optimizer::GetStateDict(OptimizerCheckpointState& optimizer_checkpoint_state) {
  auto& grouped_optimizer_states = optimizer_checkpoint_state.group_named_optimizer_states;

  // Currently all parameters are in a single group, so we hardcode group0 here.
  // To support multiple groups, Optimizer constructor need accept informations for groupping.
  const std::string group_zero_name = "group0";
  grouped_optimizer_states.insert({group_zero_name, std::make_shared<GroupOptimizerState>(optimizer_state_)});

  // Pass the optimizer session data transfer manager for data copying when saving.
  // An alternative is, we can do copy at this stage.
  ORT_RETURN_IF_NOT(optim_sess_, "optimizer session not initialized");
  const DataTransferManager& sess_data_transfer_manager = optim_sess_->GetDataTransferManager();
  optimizer_checkpoint_state.optimizer_session_data_transfer_mgr = &sess_data_transfer_manager;
  return Status::OK();
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
