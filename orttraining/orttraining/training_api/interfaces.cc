// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ENABLE_TRAINING) && defined(ENABLE_TRAINING_ON_DEVICE)
#include "orttraining/training_api/interfaces.h"
#include "core/graph/model.h"

namespace onnxruntime {
namespace training {
namespace api_test {

static std::unique_ptr<Environment> env;

void GetGraphInputOutputNames(const Graph& graph,
                              std::vector<std::string> input_names,
                              std::vector<std::string> output_names) {
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

Module::Module(const std::string& train_model_path_or_bytes,
               std::map<std::string, std::shared_ptr<Parameter>>& parameters,
               const std::optional<std::string>& eval_model_path_or_bytes) {
  parameters_ = std::move(parameters);

  for (auto it = parameters_.begin(); it != parameters_.end(); it++) {
    ORT_ENFORCE(it->first == it->second->name());
    weights_.push_back(it->second->data());
    gradients_.push_back(it->second->gradient());
  }

  auto so = onnxruntime::SessionOptions();
  std::string default_logger_id{"Default"};

  ORT_THROW_IF_ERROR(Environment::Create(std::make_unique<logging::LoggingManager>(std::make_unique<logging::CLogSink>(),
                                                                                   logging::Severity::kWARNING,
                                                                                   false,
                                                                                   logging::LoggingManager::InstanceType::Default,
                                                                                   &default_logger_id),
                                         env));

  train_sess_ = std::make_unique<onnxruntime::InferenceSession>(so, *env, train_model_path_or_bytes);
  if (eval_model_path_or_bytes.has_value()) {
    eval_sess_ = std::make_unique<onnxruntime::InferenceSession>(so, *env, eval_model_path_or_bytes.value());
  }
  
  std::shared_ptr<onnxruntime::Model> model;
  ORT_THROW_IF_ERROR(onnxruntime::Model::Load(train_model_path_or_bytes, model, nullptr, env->GetLoggingManager()->DefaultLogger()));
  GetGraphInputOutputNames(model->MainGraph(), input_names_, output_names_);
}

Status Module::TrainStep(const std::vector<OrtValue>& inputs, std::vector<OrtValue>& outputs) {
  ORT_NOT_IMPLEMENTED("Not implemented.");
  std::vector<OrtValue> feeds{inputs};
  feeds.insert(feeds.end(), weights_.begin(), weights_.end());

  std::vector<OrtValue> fetches{outputs};
  fetches.insert(fetches.end(), gradients_.begin(), gradients_.end());

  auto status = train_sess_->Run(RunOptions(), input_names_, feeds, output_names_, &fetches);

  return status;
}

}  // namespace api_test
}  // namespace training
}  // namespace onnxruntime

#endif
