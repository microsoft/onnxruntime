// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ENABLE_TRAINING) && defined(ENABLE_TRAINING_ON_DEVICE)
#include "orttraining/training_api/interfaces.h"
#include "core/graph/model.h"

namespace onnxruntime {
namespace training {
namespace api_test {

static std::unique_ptr<Environment> env;
const std::vector<std::string> GRAD_SUFFIX{"_grad.accumulation.buffer", "_grad"};
const std::string MOMENT_1{".exp_avg"};
const std::string MOMENT_2{".exp_avg_sq"};

void GetGraphInputOutputNames(const Graph& graph,
                              std::vector<std::string>& input_names,
                              std::vector<std::string>& output_names) {
  auto inputs = graph.GetInputs();
  auto outputs = graph.GetOutputs();

  auto get_names = [&](const std::vector<const NodeArg*>& node_args, std::vector<std::string>& names) {
    for (const auto* arg : node_args) {
      names.push_back(arg->Name());
    }
  };

  get_names(inputs, input_names);
  get_names(outputs, output_names);
}

bool GetParamNameFromSuffix(const std::string& name, const std::string& suffix, std::string& param_name) {
  bool endswith = std::equal(suffix.rbegin(), suffix.rend(), name.rbegin());
  if (endswith) {
    param_name = name.substr(0, name.length() - suffix.length());
    return true;
  } else {
    param_name = "";
    return false;
  }
}

bool GetParamNameFromGradient(const std::string& grad_name, std::string& param_name) {
  for (auto& suffix : GRAD_SUFFIX) {
    if (GetParamNameFromSuffix(grad_name, suffix, param_name)) {
      return true;
    }
  }
  return false;
}

Status OrtValueLike(const SessionState& sess_state, const OrtValue& input_val, OrtValue& output_val) {
  const auto& param_tensor = input_val.template Get<Tensor>();
  const TensorShape& shape = param_tensor.Shape();
  AllocatorPtr allocator = sess_state.GetAllocator(param_tensor.Location());
  // AllocatorPtr allocator = GetAllocator(param_tensor.Location());

  auto element_type = param_tensor.DataType();
  auto p_tensor = std::make_unique<Tensor>(element_type, shape, allocator);

  // TODO: handle CUDA memset
  memset(p_tensor->MutableDataRaw(), 0, p_tensor->SizeInBytes());
  output_val.Init(p_tensor.release(),
                  DataTypeImpl::GetType<Tensor>(),
                  DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  return Status::OK();
}

Status Parameter::AllocateGrad(const std::string& gradient_name, const SessionState& sess_state) {
  // assert param is allocated
  ORT_ENFORCE(data_.IsAllocated());
  ORT_ENFORCE(requires_grad_);
  gradient_name_ = gradient_name;
  ORT_ENFORCE(OrtValueLike(sess_state, data_, gradient_).IsOK());
  return Status::OK();
}

Status Parameter::ResetGrad() {
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
               std::map<std::string, std::shared_ptr<Parameter>>& parameters,
               const std::optional<std::string>& eval_model_path_or_bytes) {
  parameters_ = std::move(parameters);

  auto so = onnxruntime::SessionOptions();

  ORT_THROW_IF_ERROR(Environment::Create(nullptr, env));

  train_sess_ = std::make_unique<onnxruntime::InferenceSession>(so, *env);
  ORT_THROW_IF_ERROR(train_sess_->Load(train_model_path_or_bytes));
  ORT_THROW_IF_ERROR(train_sess_->Initialize());
  if (eval_model_path_or_bytes.has_value()) {
    eval_sess_ = std::make_unique<onnxruntime::InferenceSession>(so, *env);
    ORT_THROW_IF_ERROR(eval_sess_->Load(eval_model_path_or_bytes.value()));
    ORT_THROW_IF_ERROR(eval_sess_->Initialize());
  }
  auto& train_sess_state = train_sess_->GetSessionState();

  std::shared_ptr<onnxruntime::Model> model;
  ORT_THROW_IF_ERROR(onnxruntime::Model::Load(train_model_path_or_bytes, model, nullptr, env->GetLoggingManager()->DefaultLogger()));
  GetGraphInputOutputNames(model->MainGraph(), input_names_, output_names_);

  std::vector<std::string> param_input_names, grad_input_names, user_input_names;
  std::string param_name;
  for (auto input_name : input_names_) {
    auto it = parameters_.find(input_name);
    if (it != parameters_.end()) {
      param_input_names.emplace_back(input_name);
      weights_.push_back(it->second->data());
    } else if (GetParamNameFromGradient(input_name, param_name)) {
      grad_input_names.emplace_back(input_name);
      // create gradient buffer
      // assert param_name is valid.
      auto it = parameters_.find(param_name);
      if (it != parameters_.end()) {
        ORT_THROW_IF_ERROR(it->second->AllocateGrad(input_name, train_sess_state));
        gradients_.push_back(it->second->gradient());
      } else {
        // raise error here.
      }
    } else {
      user_input_names.emplace_back(input_name);
    }
  }
  input_names_ = user_input_names;
  input_names_.insert(input_names_.end(), param_input_names.begin(), param_input_names.end());
  input_names_.insert(input_names_.end(), grad_input_names.begin(), grad_input_names.end());
}

Status Module::ResetGrad() {
  for (auto& it : parameters_) {
    ORT_ENFORCE(it.second->ResetGrad().IsOK());
  }
  return Status::OK();
}

Status Module::TrainStep(const std::vector<OrtValue>& inputs, std::vector<OrtValue>& outputs) {
  std::vector<OrtValue> feeds{inputs};
  feeds.insert(feeds.end(), weights_.begin(), weights_.end());
  feeds.insert(feeds.end(), gradients_.begin(), gradients_.end());

  // TODO: need to filter out the grads from the output ortvalues
  auto status = train_sess_->Run(RunOptions(), input_names_, feeds, output_names_, &outputs);
  ORT_THROW_IF_ERROR(status);
  return status;
}

Optimizer::Optimizer(const std::string& optim_path_or_bytes,
                     std::map<std::string, std::shared_ptr<Parameter>>& parameters) {
  parameters_ = std::move(parameters);

  auto so = onnxruntime::SessionOptions();

  ORT_THROW_IF_ERROR(Environment::Create(nullptr, env));

  optim_sess_ = std::make_unique<onnxruntime::InferenceSession>(so, *env);
  ORT_THROW_IF_ERROR(optim_sess_->Load(optim_path_or_bytes));
  ORT_THROW_IF_ERROR(optim_sess_->Initialize());

  std::shared_ptr<onnxruntime::Model> model;
  ORT_THROW_IF_ERROR(onnxruntime::Model::Load(optim_path_or_bytes, model, nullptr, env->GetLoggingManager()->DefaultLogger()));
  GetGraphInputOutputNames(model->MainGraph(), input_names_, output_names_);
  ORT_ENFORCE(input_names_[0] == "learning_rate");  // TODO: make this better
  ORT_ENFORCE(input_names_[1] == "step");           // TODO: make this better

  std::vector<std::string> param_names, grad_names, moment1_names, moment2_names, empty_names;
  std::string param_name;
  for (size_t i = 2; i < input_names_.size(); i++) {
    std::string& name = input_names_[i];
    auto it = parameters_.find(name);
    if (it != parameters_.end()) {  // is param
      param_names.push_back(name);
      weights_.push_back(it->second->data());
    } else if (GetParamNameFromGradient(name, param_name)) {
      grad_names.emplace_back(name);
      // create gradient buffer
      // assert param_name is valid.
      auto it = parameters_.find(param_name);
      if (it != parameters_.end()) {
        gradients_.push_back(it->second->gradient());
      } else {
        // raise error here.
      }
    } else if (GetParamNameFromSuffix(name, MOMENT_1, param_name)) {
      moment1_names.push_back(name);
    } else if (GetParamNameFromSuffix(name, MOMENT_2, param_name)) {
      moment1_names.push_back(name);
    } else {
      empty_names.push_back(name);
    }
  }

  auto& optim_sess_state = optim_sess_->GetSessionState();
  std::unordered_map<std::string, ParameterOptimizerState>&
      param_named_optimizer_states = optimizer_state_.optimizer_states_;

  // TODO: don't hard code the state names, should get the state names according to the optimizer types.
  std::vector<std::string> state_names{"momentum0", "momentum1"};
  for (auto& pair : parameters) {
    if (pair.second->RequiresGrad()) {
      param_named_optimizer_states.insert({pair.first, ParameterOptimizerState()});
      ParameterOptimizerState& cur_param_optimizer_states = param_named_optimizer_states[pair.first];
      for (auto& state_name : state_names) {
        OrtValue param_state;
        // TODO: should reset the state to zero (for both CPU or CUDA Tensors.)
        ORT_ENFORCE(OrtValueLike(optim_sess_state, pair.second->data(), param_state).IsOK());
        cur_param_optimizer_states.states_.insert({state_name, std::make_shared<OrtValue>(param_state)});
      }
    }
  }
}

}  // namespace api_test
}  // namespace training
}  // namespace onnxruntime

#endif
