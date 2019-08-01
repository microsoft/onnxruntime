// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/model.h"
#include "core/graph/training/loss_function_builder.h"
#include "core/graph/training/in_graph_training_optimizer.h"
#include "core/training/gradient_graph_builder.h"
#include "core/training/training_session.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_allocator.h"
#endif

using namespace std;

namespace onnxruntime {
namespace training {

static Status AddLossFuncionInternal(Graph& graph,
                                     std::shared_ptr<ILossFunction>& loss_graph_builder,
                                     const LossFunctionInfo& loss_func_info) {
  return GraphAugmenter::AugmentGraph(graph, loss_graph_builder->operator()(graph, loss_func_info));
}

static Status BuildGradientGraphInternal(Graph& graph,
                                         const string& loss_function_output_name,
                                         const unordered_set<string>& node_arg_names_to_train,
                                         const unordered_map<string, OptimizerInfo>& opt_info) {
  // Compute the gradient graph def.
  GradientGraphBuilder grad_graph_builder(&graph,
                                          {loss_function_output_name},
                                          node_arg_names_to_train,
                                          loss_function_output_name,
                                          opt_info);
  return grad_graph_builder.Build();
}

Status TrainingSession::BuildLossFunction(const LossFunctionInfo& loss_func_info) {
  if (loss_func_info.op_def.type.empty() || loss_func_info.loss_name.empty()) {
    ORT_THROW("BuildLossFuncion's loss_function_info is invalid.");
  }

  loss_func_info_ = loss_func_info;
  loss_graph_builder_ = LossFunctionBuilder::Build(loss_func_info_.op_def.type);

  try {
    ORT_RETURN_IF_ERROR(AddLossFuncionInternal(model_->MainGraph(), loss_graph_builder_, loss_func_info_));
  } catch (const OnnxRuntimeException& exp) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to add loss function:", exp.what());
  }
  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::BuildGradientGraph(const unordered_set<string>& weights_to_train,
                                           const string& loss_function_output_name,
                                           const unordered_map<string, OptimizerInfo>& opt_info) {
  // Fill weights_to_train_ according to weights_to_train
  weights_to_train_ = weights_to_train;
  opt_info_ = opt_info;

  ORT_RETURN_IF_ERROR(BuildGradientGraphInternal(model_->MainGraph(),
                                                 loss_function_output_name,
                                                 weights_to_train_,
                                                 opt_info_));

  return DoPostLoadProcessing(*model_);
}

Status TrainingSession::ExposeAsGraphOutput(const std::vector<std::string>& node_args) {
  GraphAugmenter::GraphDefs graph_defs;
  graph_defs.AddGraphOutputs(node_args);

  ORT_RETURN_IF_ERROR(GraphAugmenter::AugmentGraph(model_->MainGraph(), graph_defs));

  return DoPostLoadProcessing(*model_);
}

NameMLValMap TrainingSession::GetWeights() const {
  return session_state_.GetInitializedTensors(weights_to_train_);
}

Status TrainingSession::UpdateWeightsInSessionState(const NameMLValMap& new_weights) {
  session_state_.UpdateInitializedTensors(new_weights);
  VLOGS(*session_logger_, 1) << "Done updating weights";
  return Status::OK();
}

static Status UpdateWeightsBeforeSaving(Graph& graph, const NameMLValMap& weights) {
  // Store MLValue (either in CPU or CUDA) into TensorProto
  // TODO: support more types than float

  for (const auto& name_and_ml_value : weights) {
    // Set src_data pointer
    const auto& src_tensor = name_and_ml_value.second.Get<Tensor>();
    const void* src_data = src_tensor.DataRaw(src_tensor.DataType());

    // Set dst_data pointer
    const ONNX_NAMESPACE::TensorProto* old_tensor_proto = nullptr;
    if (!graph.GetInitializedTensor(name_and_ml_value.first, old_tensor_proto)) {
      continue;
    }
    ONNX_NAMESPACE::TensorProto new_tensor_proto = *old_tensor_proto;
    void* dst_data = nullptr;
    if (new_tensor_proto.has_raw_data()) {
      dst_data = const_cast<char*>(new_tensor_proto.mutable_raw_data()->data());
    } else {
      ORT_ENFORCE(new_tensor_proto.data_type() == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT);
      dst_data = new_tensor_proto.mutable_float_data()->mutable_data();
    }

    // Copy from src_data to dst_data.
    auto data_size = src_tensor.Size();
    if (strcmp(src_tensor.Location().name, CPU) == 0) {
      memcpy(dst_data, src_data, data_size);
    }
#ifdef USE_CUDA
    else if (strcmp(src_tensor.Location().name, CUDA) == 0) {
      ORT_RETURN_IF_NOT(cudaSuccess == cudaMemcpy(dst_data, src_data, data_size, cudaMemcpyDeviceToHost),
                        "cudaMemcpy returns error");
    }
#endif
    else {
      ORT_THROW("Device is not supported:", src_tensor.Location().name);
    }

    // Replace the TensorProto in the model.
    graph.RemoveInitializedTensor(old_tensor_proto->name());
    graph.AddInitializedTensor(new_tensor_proto);
  }
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
  ORT_RETURN_IF_ERROR(UpdateWeightsBeforeSaving(new_model->MainGraph(), GetWeights()));

  if (opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC /* with weights and loss func*/ ||
      opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS /*with everything*/) {
    ORT_RETURN_IF_ERROR(AddLossFuncionInternal(new_model->MainGraph(), loss_graph_builder_, loss_func_info_));
  }

  if (opt == TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS) {
    ORT_RETURN_IF_ERROR(BuildGradientGraphInternal(new_model->MainGraph(),
                                                   loss_func_info_.loss_name,
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

bool TrainingSession::IsUntrainable(const Node* node, const std::string& initializer_name,
                                    const logging::Logger* logger) {
  auto it = STOP_GRADIENT_EDGES.find(node->OpType());
  if (it != STOP_GRADIENT_EDGES.end()) {
    for (auto input_idx : it->second) {
      if (input_idx < node->InputDefs().size() &&
          node->InputDefs()[input_idx]->Name() == initializer_name) {
        if (logger) {
          VLOGS(*logger, 1) << "Excluding " << node->Name() << "'s input " << input_idx
                            << " initializer: " << initializer_name;
        }
        return true;
      }
    }
  }
  return false;
}

bool TrainingSession::IsImmutableWeight(const ImmutableWeights& immutable_weights,
                                        const Node* node, const TensorProto* tensor,
                                        const logging::Logger* logger) {
  auto it = immutable_weights.find(node->OpType());
  if (it == immutable_weights.end()) {
    return false;
  }

  for (auto pair : it->second) {
    size_t& input_idx = pair.first;
    float& value = pair.second;

    if (input_idx < node->InputDefs().size() &&
        node->InputDefs()[input_idx]->Name() == tensor->name()) {
      if (tensor->data_type() == TensorProto_DataType_FLOAT && tensor->dims_size() == 0) {
        float tensor_value;
        if (tensor->has_raw_data()) {
          memcpy(&tensor_value, tensor->raw_data().data(), sizeof(float));
        } else {
          tensor_value = *(tensor->float_data().data());
        }
        if (tensor_value == value) {
          if (logger) {
            VLOGS(*logger, 1) << "Excluding " << node->Name() << "'s input " << input_idx
                              << " initializer: " << tensor->name() << " with value " << tensor_value;
          }
          return true;
        }
      }
    }
  }

  return false;
}

std::unordered_set<std::string> TrainingSession::GetTrainableModelInitializers(
    const ImmutableWeights& immutable_weights) const {
  const Graph& graph = model_->MainGraph();
  const auto& initialized_tensors = graph.GetAllInitializedTensors();
  std::unordered_set<std::string> model_initializers;
  std::transform(initialized_tensors.begin(),
                 initialized_tensors.end(),
                 std::inserter(model_initializers, model_initializers.end()),
                 [](const auto& pair) { return pair.first; });

  std::unordered_set<std::string> trainable_initializers(model_initializers);
  for (const string& initializer_name : model_initializers) {
    const auto& nodes = graph.GetConsumerNodes(initializer_name);
    for (const Node* node : nodes) {
      if (IsUntrainable(node, initializer_name, session_logger_) ||
          IsImmutableWeight(immutable_weights, node, initialized_tensors.at(initializer_name), session_logger_)) {
        trainable_initializers.erase(initializer_name);
      }
    }
  }

  return trainable_initializers;
}

common::Status TrainingSession::UpdateTrainableWeightsInfoInGraph() {
  Graph& graph = model_->MainGraph();
  const auto& graph_inputs = graph.GetInputsIncludingInitializers();
  std::unordered_set<const NodeArg*> inputs_to_add{};
  std::transform(
      weights_to_train_.begin(), weights_to_train_.end(), std::inserter(inputs_to_add, inputs_to_add.end()),
      [&graph](const std::string& node_name) {
        return graph.GetNodeArg(node_name);
      });
  for (const NodeArg* graph_input : graph_inputs) {
    inputs_to_add.erase(graph_input);
  }
  std::vector<const NodeArg*> new_graph_inputs(graph_inputs);
  new_graph_inputs.insert(new_graph_inputs.end(), inputs_to_add.begin(), inputs_to_add.end());
  graph.SetInputs(new_graph_inputs);
  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
