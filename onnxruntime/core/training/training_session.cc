// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "training_session.h"
#include "core/training/gradients.h"

using namespace std;

namespace onnxruntime {
namespace training {

class TrainingSessionImpl : public InferenceSession::Impl {
 public:
  TrainingSessionImpl(const SessionOptions& session_options,
                      logging::LoggingManager* logging_manager = nullptr)
      : InferenceSession::Impl(session_options, logging_manager) {}

  ~TrainingSessionImpl() {}

  Status Load(const string& model_uri) {
    original_model_uri_ = model_uri;
    return InferenceSession::Impl::Load(model_uri);
  }

  Status BuildGradientGraph(const vector<string>& weights_to_train, const std::string& loss_function_output_name) {
    // Fill weights_to_train_ according to weights_to_train
    weights_to_train_ = weights_to_train;

    GradientGraphBuilder grad_graph_builder(&model_->MainGraph(), {loss_function_output_name}, weights_to_train, loss_function_output_name);
    ORT_RETURN_IF_ERROR(grad_graph_builder.Build());

    model_->MainGraph().SetGraphResolveNeeded();
    model_->MainGraph().SetGraphProtoSyncNeeded();
    ORT_RETURN_IF_ERROR(model_->MainGraph().Resolve());
    return DoPostLoadProcessing(*model_);
  }

  Status Initialize() {
    return InferenceSession::Impl::Initialize();
  }

  Status Run(const NameMLValMap& feeds,
             const vector<string>& output_names,
             vector<MLValue>* p_fetches) {
    return InferenceSession::Impl::Run(feeds, output_names, p_fetches);
  }

  NameMLValMap GetWeights() const {
    return session_state_.GetInitializedTensors(weights_to_train_);
  }

  Status UpdateWeights(const NameMLValMap& new_weights) {
    session_state_.UpdateInitializedTensors(new_weights);
    VLOGS(*session_logger_, 1) << "Done updating weights";
    return Status::OK();
  }

  Status Save(const string& model_uri, bool include_gradient_graph) {
    if (include_gradient_graph) {
      std::lock_guard<onnxruntime::OrtMutex> l(session_mutex_);
      if (is_inited_) {
        return Status(common::ONNXRUNTIME, common::FAIL, "Could not save a model with gradient graph after calling Initialize()");
      }
      return Model::Save(*model_, model_uri);
    } else {
      // Have to load the original model again.
      // Because after Initialize(), the model has been optimized and could not be saved correctly.
      shared_ptr<Model> model;
      Model::Load(original_model_uri_, model);
      ORT_RETURN_IF_ERROR(model->UpdateWeights(GetWeights()));
      return Model::Save(*model, model_uri);
    }
  }

  std::unordered_set<std::string> GetModelInputNames() const {
    return model_input_names_;
  }

  std::unordered_set<std::string> GetModelOutputNames() const {
    return model_output_names_;
  }

 private:
  vector<string> weights_to_train_;
  string original_model_uri_;
};

TrainingSession::TrainingSession(const SessionOptions& session_options,
                                 logging::LoggingManager* logging_manager)
    : impl_(make_unique<TrainingSessionImpl>(session_options, logging_manager)) {
}

TrainingSession::~TrainingSession() {
}

Status TrainingSession::Load(const string& model_uri) {
  return impl_->Load(model_uri);
}

Status TrainingSession::BuildGradientGraph(const vector<string>& weights_to_train, const std::string& loss_function_output_name) {
  return impl_->BuildGradientGraph(weights_to_train, loss_function_output_name);
}

Status TrainingSession::Initialize() {
  return impl_->Initialize();
}

// Compute gradients.
Status TrainingSession::Run(const NameMLValMap& feeds,
                            const vector<string>& output_names,
                            vector<MLValue>* p_fetches) {
  return impl_->Run(feeds, output_names, p_fetches);
}

NameMLValMap TrainingSession::GetWeights() const {
  return impl_->GetWeights();
}

Status TrainingSession::UpdateWeights(const NameMLValMap& new_weights) {
  return impl_->UpdateWeights(new_weights);
}

Status TrainingSession::Save(const string& model_uri, bool include_gradient_graph) {
  return impl_->Save(model_uri, include_gradient_graph);
}

std::unordered_set<std::string> TrainingSession::GetModelInputNames() const {
  return impl_->GetModelInputNames();
}

std::unordered_set<std::string> TrainingSession::GetModelOutputNames() const {
  return impl_->GetModelOutputNames();
}
}  // namespace training
}  // namespace onnxruntime
