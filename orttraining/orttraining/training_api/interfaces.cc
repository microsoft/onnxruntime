// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ENABLE_TRAINING) && defined(ENABLE_TRAINING_ON_DEVICE)
#include "orttraining/training_api/interfaces.h"
#include "core/graph/model.h"

namespace onnxruntime {
namespace training {
namespace api_test {

static std::unique_ptr<Environment> env;
const std::string GRAD_SUFFIX{"_grad.accumulation.buffer"};

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

void GetParamNameFromGradient(std::string& grad_name, std::string& param_name) {
  bool endswith = std::equal(GRAD_SUFFIX.rbegin(), GRAD_SUFFIX.rend(), grad_name.rbegin());
  if (endswith) {
    param_name = grad_name.substr(0, grad_name.length() - GRAD_SUFFIX.length());
  } else {
    param_name = "";
  }
}

Status Parameter::AllocateGrad(const std::string& gradient_name, const SessionState& sess_state) {
  // assert param is allocated
  ORT_ENFORCE(data_.IsAllocated());
  ORT_ENFORCE(requires_grad_);
  gradient_name_ = gradient_name;
  const auto& param_tensor = data_.template Get<Tensor>();
  const TensorShape& shape = param_tensor.Shape();
  AllocatorPtr allocator = sess_state.GetAllocator(param_tensor.Location());
  // AllocatorPtr allocator = GetAllocator(param_tensor.Location());

  auto element_type = param_tensor.DataType();
  auto p_tensor = std::make_unique<Tensor>(element_type, shape, allocator);

  // TODO: handle CUDA memset
  memset(p_tensor->MutableDataRaw(), 0, p_tensor->SizeInBytes());
  gradient_.Init(p_tensor.release(),
                 DataTypeImpl::GetType<Tensor>(),
                 DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  return Status::OK();
}

Status Parameter::ResetGrad() {
  Tensor* p_tensor = gradient_.GetMutable<Tensor>();
  const OrtMemoryInfo& device = p_tensor->Location().device;
  ORT_RETURN_IF_ERROR()
  if (device == )
    memset(p_tensor.MutableDataRaw(), 0, p_tensor.SizeInBytes());
  
  return Status::OK();
}

Module::Module(const std::string& train_model_path_or_bytes,
               std::map<std::string, std::shared_ptr<Parameter>>& parameters,
               const std::optional<std::string>& eval_model_path_or_bytes) {
  parameters_ = std::move(parameters);

  auto so = onnxruntime::SessionOptions();
  std::string default_logger_id{"Default"};

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
  for (auto input_name : input_names_) {
    auto it = parameters_.find(input_name);
    if (it != parameters_.end()) {
      param_input_names.emplace_back(input_name);
      weights_.push_back(it->second->data());
    } else if (std::equal(GRAD_SUFFIX.rbegin(), GRAD_SUFFIX.rend(), input_name.rbegin())) {
      grad_input_names.emplace_back(input_name);
      // create gradient buffer
      std::string param_name;
      GetParamNameFromGradient(input_name, param_name);
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

Status Module::TrainStep(const std::vector<OrtValue>& inputs, std::vector<OrtValue>& outputs) {
  std::vector<OrtValue> feeds{inputs};
  feeds.insert(feeds.end(), weights_.begin(), weights_.end());
  feeds.insert(feeds.end(), gradients_.begin(), gradients_.end());

  // TODO: need to filter out the grads from the output ortvalues
  auto status = train_sess_->Run(RunOptions(), input_names_, feeds, output_names_, &outputs);
  ORT_THROW_IF_ERROR(status);
  return status;
}

}  // namespace api_test
}  // namespace training
}  // namespace onnxruntime

#endif
