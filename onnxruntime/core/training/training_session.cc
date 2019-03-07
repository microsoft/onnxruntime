// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/training/gradient_graph_builder.h"
#include "core/training/loss_function_builder.h"
#include "core/training/training_session.h"

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

  Status RegisterCustomLossFunction(const std::string& loss_func_name) {
    auto& loss_func_reg = LossFunctionRegistry::GetInstance();
    try {
      loss_func_reg.RegisterCustomLossFunction(loss_func_name);
      return Status::OK();
    } catch (const OnnxRuntimeException& exp) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to register custom loss function.", exp.what());
    }
  }

  Status AddLossFuncion(const LossFunctionInfo& loss_func_info) {
    loss_func_info_ = loss_func_info;

    try {
      ORT_RETURN_IF_ERROR(AddLossFuncion(model_->MainGraph(), loss_func_info_));
    } catch (const OnnxRuntimeException& exp) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to add loss function:", exp.what());
    }
    return DoPostLoadProcessing(*model_);
  }

  Status BuildGradientGraph(const vector<string>& weights_to_train, const std::string& loss_function_output_name) {
    // Fill weights_to_train_ according to weights_to_train
    weights_to_train_ = weights_to_train;

    ORT_RETURN_IF_ERROR(BuildGradientGraph(model_->MainGraph(),
                                           loss_function_output_name,
                                           weights_to_train_));

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

  Status Save(const string& model_uri, TrainingSession::SaveOption opt) {
    // Have to load the original model again.
    // Because after Initialize(), the model has been optimized and the saved graph doesn't look like what we expect.
    shared_ptr<Model> new_model;
    ORT_RETURN_IF_ERROR(Model::Load(original_model_uri_, new_model));
    ORT_RETURN_IF_ERROR(new_model->UpdateWeights(GetWeights()));

    if (opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC /* with weights and loss func*/ ||
        opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS /*with everything*/) {
      ORT_RETURN_IF_ERROR(AddLossFuncion(new_model->MainGraph(), loss_func_info_));
    }

    if (opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS) {
      ORT_RETURN_IF_ERROR(BuildGradientGraph(new_model->MainGraph(),
                                             loss_func_info_.loss_name_,
                                             weights_to_train_));
    }

    return Model::Save(*new_model, model_uri);
  }

  std::unordered_set<std::string> GetModelInputNames() const {
    return model_input_names_;
  }

  std::unordered_set<std::string> GetModelOutputNames() const {
    return model_output_names_;
  }

  std::unordered_set<std::string> GetModelInitializers() const {
    const auto& initialized_tensors = model_->MainGraph().GetAllInitializedTensors();
    std::unordered_set<std::string> model_initializers_;
    std::transform(initialized_tensors.begin(),
                   initialized_tensors.end(),
                   std::inserter(model_initializers_, model_initializers_.end()),
                   [](const auto& pair) { return pair.first; });

    return model_initializers_;
  }

 private:
  static Status AddLossFuncion(Graph& graph,
                               const LossFunctionInfo& loss_func_info) {
    return GraphAugmenter::AugmentGraph(graph, LossFunctionBuilder().Build(graph, loss_func_info));
  }

  static Status BuildGradientGraph(Graph& graph,
                                   const std::string& loss_function_output_name,
                                   const std::vector<std::string>& node_arg_names_to_train) {
    // Compute the gradient graph def.
    GradientGraphBuilder grad_graph_builder(&graph,
                                            {loss_function_output_name},
                                            node_arg_names_to_train,
                                            loss_function_output_name);
    GraphAugmenter::GraphDefs gradient_graph_def;
    ORT_RETURN_IF_ERROR(grad_graph_builder.Build(gradient_graph_def));

    return GraphAugmenter::AugmentGraph(graph, gradient_graph_def);
  }

  vector<string> weights_to_train_;
  string original_model_uri_;

  LossFunctionInfo loss_func_info_;
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

Status TrainingSession::RegisterCustomLossFunction(const std::string& loss_func_name) {
  return impl_->RegisterCustomLossFunction(loss_func_name);
}

Status TrainingSession::AddLossFuncion(const LossFunctionInfo& loss_func_info) {
  return impl_->AddLossFuncion(loss_func_info);
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

Status TrainingSession::Save(const string& model_uri, SaveOption opt) {
  return impl_->Save(model_uri, opt);
}

std::unordered_set<std::string> TrainingSession::GetModelInputNames() const {
  return impl_->GetModelInputNames();
}

std::unordered_set<std::string> TrainingSession::GetModelOutputNames() const {
  return impl_->GetModelOutputNames();
}

std::unordered_set<std::string> TrainingSession::GetModelInitializers() const {
  return impl_->GetModelInitializers();
}

}  // namespace training
}  // namespace onnxruntime
