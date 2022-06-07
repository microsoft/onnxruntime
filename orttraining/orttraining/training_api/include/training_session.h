// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "module.h"
#include "optimizer.h"
#include "checkpoint.h"

namespace onnxruntime {
namespace training {
namespace api {
using namespace common;
//#ifdef ENABLE_ON_DEVICE_TRAINING


// Wrapper on top of module and optimizer classes and is the only class exposed via capis
class TrainingSession {
 public:
   TrainingSession(const SessionOptions& session_options, const Environment& session_env);

#ifdef _WIN32

#endif

   Status Initialize(std::unordered_map<std::string, std::shared_ptr<Parameter>>& parameters,
                     const std::string& train_model_uri, const std::optional<std::string>& eval_model_uri,
                     const std::optional<std::string>& optim_model_uri);

   Status InitializeTrainingSession(std::unordered_map<std::string, std::shared_ptr<Parameter>>& parameters,
                                    const std::string& train_model_uri,
                                    const std::optional<std::string>& eval_model_uri);

   Status InitializeOptimizerSession(std::unordered_map<std::string, std::shared_ptr<Parameter>>& parameters,
                                     const std::string& optim_model_uri);


   Status TrainStep(const RunOptions& run_options, const std::vector<OrtValue>& inputs, std::vector<OrtValue>& fetches);

   Status EvalStep(const RunOptions& run_options, const std::vector<OrtValue>& inputs, std::vector<OrtValue>& fetches);

   Status ResetGrad();

   Status OptimizerStep(const RunOptions& run_options);

 private:
   ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TrainingSession);

   std::unique_ptr<Module> module_;
   std::unique_ptr<Optimizer> optimizer_;
   SessionOptions session_options_;
   const Environment& environment_;
};

//#endif
}
}
}
