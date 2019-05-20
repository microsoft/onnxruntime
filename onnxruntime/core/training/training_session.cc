// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/model.h"
#include "core/graph/training/loss_function_builder.h"
#include "core/graph/training/in_graph_training_optimizer.h"
#include "core/training/training_optimizer.h"
#include "core/training/weight_updater.h"
#include "core/training/gradient_graph_builder.h"
#include "core/training/training_session.h"

using namespace std;

namespace onnxruntime {
namespace training {

static Status AddLossFuncionInternal(Graph& graph,
                                     const LossFunctionInfo& loss_func_info) {
  return GraphAugmenter::AugmentGraph(graph, LossFunctionBuilder().Build(graph, loss_func_info));
}

static Status BuildGradientGraphInternal(Graph& graph,
                                         const string& loss_function_output_name,
                                         const vector<string>& node_arg_names_to_train,
                                         const vector<in_graph_optimizer::OptimizerInfo>& opt_info) {
  // Compute the gradient graph def.
  GradientGraphBuilder grad_graph_builder(&graph,
                                          {loss_function_output_name},
                                          node_arg_names_to_train,
                                          loss_function_output_name,
                                          opt_info);
  return grad_graph_builder.Build();
}

Status TrainingSession::AddLossFuncion(const LossFunctionInfo& loss_func_info) {
  loss_func_info_ = loss_func_info;

  try {
    ORT_RETURN_IF_ERROR(AddLossFuncionInternal(model_->MainGraph(), loss_func_info_));
  } catch (const OnnxRuntimeException& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to add loss function:", exp.what());
  }
  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::BuildGradientGraph(const vector<string>& weights_to_train,
                                           const string& loss_function_output_name,
                                           const vector<in_graph_optimizer::OptimizerInfo>& opt_info) {
  // Fill weights_to_train_ according to weights_to_train
  weights_to_train_ = weights_to_train;
  opt_info_ = opt_info;

  ORT_RETURN_IF_ERROR(BuildGradientGraphInternal(model_->MainGraph(),
                                                 loss_function_output_name,
                                                 weights_to_train_,
                                                 opt_info_));

  return DoPostLoadProcessing(*model_);
}

NameMLValMap TrainingSession::GetWeights() const {
  return session_state_.GetInitializedTensors(weights_to_train_);
}

Status TrainingSession::UpdateWeights(const NameMLValMap& new_weights) {
  session_state_.UpdateInitializedTensors(new_weights);
  VLOGS(*session_logger_, 1) << "Done updating weights";
  return Status::OK();
}

Status TrainingSession::Save(const string& model_uri, TrainingSession::SaveOption opt) {
  // Delete the old file before saving.
  std::remove(model_uri.c_str());

  if (opt == TrainingSession::SaveOption::NO_RELOAD) {
    return Model::Save(*model_, model_uri);
  }

  // Have to load the original model again.
  // Because after Initialize(), the model has been optimized and the saved graph doesn't look like what we expect.
  shared_ptr<Model> new_model;
  ORT_RETURN_IF_ERROR(Model::Load(model_location_, new_model));
  ORT_RETURN_IF_ERROR(new_model->UpdateWeights(GetWeights()));

  if (opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC /* with weights and loss func*/ ||
      opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS /*with everything*/) {
    ORT_RETURN_IF_ERROR(AddLossFuncionInternal(new_model->MainGraph(), loss_func_info_));
  }

  if (opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS) {
    ORT_RETURN_IF_ERROR(BuildGradientGraphInternal(new_model->MainGraph(),
                                                   loss_func_info_.loss_name_,
                                                   weights_to_train_,
                                                   opt_info_));
  }

  return Model::Save(*new_model, model_uri);
}

std::unordered_set<std::string> TrainingSession::GetModelInputNames() const {
  return model_input_names_;
}

std::unordered_set<std::string> TrainingSession::GetModelOutputNames() const {
  return model_output_names_;
}

std::unordered_set<std::string> TrainingSession::GetModelInitializers() const {
  const auto& initialized_tensors = model_->MainGraph().GetAllInitializedTensors();
  std::unordered_set<std::string> model_initializers;
  std::transform(initialized_tensors.begin(),
                 initialized_tensors.end(),
                 std::inserter(model_initializers, model_initializers.end()),
                 [](const auto& pair) { return pair.first; });

  return model_initializers;
}
}  // namespace training
}  // namespace onnxruntime
