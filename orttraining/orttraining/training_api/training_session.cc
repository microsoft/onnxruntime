// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#ifdef ENABLE_ON_DEVICE_TRAINING
#include "orttraining/training_api/include/training_session.h"

namespace onnxruntime {
namespace training {
namespace api {

using namespace onnxruntime;

TrainingSession::TrainingSession(const SessionOptions& session_options,
                                 const Environment& session_env)
                                 : session_options_{session_options},
                                 environment_(session_env) {}

#ifdef _WIN32

#endif

Status TrainingSession::Initialize(std::unordered_map<std::string, std::shared_ptr<Parameter>>& parameters,
                            const std::string& train_model_uri, const std::optional<std::string>& eval_model_uri,
                            const std::optional<std::string>& optim_model_uri) {
    module_.reset(new Module(train_model_uri, parameters, eval_model_uri));

    if(optim_model_uri.has_value()) {
        optimizer_.reset(new Optimizer(optim_model_uri.value(), parameters));
    }

    return Status::OK();
}

Status TrainingSession::InitializeTrainingSession(std::unordered_map<std::string, std::shared_ptr<Parameter>>& parameters,
                                                  const std::string& train_model_uri,
                                                  const std::optional<std::string>& eval_model_uri) {
    module_.reset(new Module(train_model_uri, parameters, eval_model_uri));
    return Status::OK();
}

Status TrainingSession::InitializeOptimizerSession(std::unordered_map<std::string, std::shared_ptr<Parameter>>& parameters,
                                                   const std::string& optim_model_uri) {
    optimizer_.reset(new Optimizer(optim_model_uri, parameters));
    return Status::OK();
}


Status TrainingSession::TrainStep(const RunOptions& ,
                                  const std::vector<OrtValue>& inputs,
                                  std::vector<OrtValue>& fetches) {
    return module_->TrainStep(inputs, fetches);
}

Status TrainingSession::EvalStep(const RunOptions& ,
                                 const std::vector<OrtValue>& inputs,
                                 std::vector<OrtValue>& fetches) {
    return module_->EvalStep(inputs, fetches);
}

Status TrainingSession::ResetGrad() {
    return module_->ResetGrad();
}

Status TrainingSession::OptimizerStep(const RunOptions& ) {
    return optimizer_->Step();
}

}
}
}
//#endif
