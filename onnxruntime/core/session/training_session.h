// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/session/inference_session_impl.h"
#include "core/framework/ml_value.h"

namespace onnxruntime {  // forward declarations
struct SessionOptions;

namespace logging {
class LoggingManager;
}

namespace training {

// Although being used as pimpl, TrainingSessionImpl has to be outside of TrainingSession in order to inherit from InferenceSession::Impl.
// Because:
// 1. it needs to be declared friend of InferenceSession, otherwise it cannot access InferenceSession::Impl;
// 2. InferenceSession doesn't want to see the definition of TrainingSession.
class TrainingSessionImpl;

class TrainingSession {
 public:
  explicit TrainingSession(const SessionOptions& session_options,
                           logging::LoggingManager* logging_manager = nullptr);

  ~TrainingSession();

  common::Status Load(const std::string& model_uri);

  // TODO: common::Status RegisterCostFuncion ...

  common::Status BuildGradientGraph(const std::vector<std::string>& weights_to_train, const std::string& loss_function_output_name);

  common::Status Initialize();

  // TODO: merge BuildGradientGraph() into Initialize()
  // We don't do this right now for debugging purpose, because we want to save and check the bw graph before Initialize() which
  // could make the graph messy due to optimization.
  common::Status Initialize(const std::vector<std::string>& weights_to_train, const std::string& loss_function_output_name) {
    throw std::runtime_error("Initialize(const std::vector<std::string>&, const std::string&) Not implemented");
  }

  // Compute gradients.
  common::Status Run(const NameMLValMap& feeds,
                     const std::vector<std::string>& output_names,
                     std::vector<MLValue>* p_fetches);

  common::Status Save(const std::string& model_uri, bool include_gradient_graph = false);

  // TODO: remove or refine below temp interfaces.
  NameMLValMap GetWeights() const;
  common::Status UpdateWeights(const NameMLValMap& new_weights);
  std::unordered_set<std::string> GetModelInputNames() const;
  std::unordered_set<std::string> GetModelOutputNames() const;

 private:
  std::unique_ptr<TrainingSessionImpl> impl_;
};
}  // namespace training
}  // namespace onnxruntime
