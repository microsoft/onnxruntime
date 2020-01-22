// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>
#include "core/graph/training/loss_func/loss_func_common.h"
#include "core/graph/training/loss_function_registry.h"
#include "core/platform/path_string.h"
#include "core/session/inference_session.h"
#include "core/training/optimizer_config.h"

namespace onnxruntime {
namespace training {

class TrainingSession : public InferenceSession {
 public:
  explicit TrainingSession(const SessionOptions& session_options,
                           logging::LoggingManager* logging_manager = nullptr)
      : InferenceSession(session_options, logging_manager) {}

  /** Add a graph input suitable for use as a scaling factor for loss scaling.
  It will be a scalar float tensor.
  @param loss_scale_input_name The name of the added graph input.
  @return Status of the graph input addition.
  */
  common::Status BuildLossScalingFactorInput(const float loss_scale, std::string& loss_scale_input_name);

  /** Add a system provided or an op as loss function to the model.
  After the call, the model have one more input named as label_name and one more output named as loss_func_output_name.
  @param loss_func_info The loss function info.
  @param loss_scale_input_name If non-empty, specifies that loss scaling should be applied using the named
         loss scaling factor input. Otherwise, loss scaling will not be applied.
  @param actual_loss_name The actual name of the loss function output from which to start the backward graph.
  @returns Status indicating success or providing an error message.
  @remarks When using a custom/standard op as loss function, 2 ops must have been registered:
             1. an op for loss function, schema:
                 Inputs:
                     OUT
                     LABEL
                 Outputs:
                     LOSS
             2. an op to calculate gradients, schema:
                 Inputs:
                     GRADIENT_OF_OUTPUT
                     OUT
                     LABEL
                 Outputs:
                     GRADIENT_OF_OUT
                     GRADIENT_OF_LABEL
           And also in gradient_builder.cc, the gradient builder must have been registered.
  */
  common::Status BuildLossFunction(const LossFunctionInfo& loss_func_info,
                                   const std::string& loss_scale_input_name,
                                   std::string& actual_loss_name);

  common::Status AddGistEncoding();

  /** Add tensorboard summary nodes to the graph.
  @param summary_name name for the merged summary node.
  @param scalar_nodes tensor names to add scalar summary nodes for.
  @param histogram_nodes tensor names to add histogram summary nodes for.
  @param norm_nodes tensor names to add norm summary nodes for.
  @param dump_convergence_metrics add convergence metrics such as gradient norm to the summary or not.
  */
  common::Status AddTensorboard(const std::string& summary_name,
                                const std::vector<std::string>& scalar_nodes,
                                const std::vector<std::string>& histogram_nodes,
                                const std::vector<std::string>& norm_nodes,
                                const bool dump_convergence_metrics);

  common::Status ApplyTransformationsToMainGraph();

  /** Perform auto-diff to add backward graph into the model.
  @param weights_to_train a set of weights to be training.
  @param loss_function_output_name the name of the loss function's output.
  @param set_gradient_as_graph_output if it is true, set gradient of trainable weight as graph output
  */
  common::Status BuildGradientGraph(const std::unordered_set<std::string>& weights_to_train,
                                    const std::string& loss_function_output_name,
                                    const bool set_gradient_as_graph_output = false);

  common::Status BuildAccumulationNode(const std::unordered_set<std::string>& weights_to_train);

  /** Add optimizer into the model. Each trainable weight will have an optimizer
  @param opt_graph_config The configuration that applies to all optimizers.
  @param opt_configs specify the optimizers used by each weight in weights_to_train, 1-1 mapping to weights_to_train.
  @param opt_graph_outputs The outputs of optimizer graph
  */
  common::Status BuildOptimizer(
      const OptimizerGraphConfig& opt_graph_config,
      const std::unordered_map<std::string, OptimizerNodeConfig>& opt_configs,
      std::unordered_map<std::string, std::string>& opt_graph_outputs);

  /** Enable mixed precision training
  @param weights_to_train a set of weights to be training.
  @param use_fp16_initializer specify whether fp16 initialier is created.
  @param fp32_weight_name_to_fp16_node_arg the map between weights and FP16 weights.
  */
  common::Status EnableMixedPrecision(const std::unordered_set<std::string>& weights_to_train,
                                      bool use_fp16_initializer,
                                      std::unordered_map<std::string, NodeArg*>& fp32_weight_name_to_fp16_node_arg);

  common::Status OverrideGraphOutputs(const std::vector<std::string>& outputs);

  /** Save a model, 3 options:
  1. save with updated weights
  2. save with updated weights and loss function
  3. save with updated weights, loss function and gradients
  */
  enum class SaveOption {
    NO_RELOAD,
    WITH_UPDATED_WEIGHTS,
    WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC,
    WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS
  };

  /** Save the new model.
  @param model_uri the path for the new model.
  @param opt see SaveOption.
  */
  common::Status Save(const PathString& model_uri, SaveOption opt);

  /** Saves a model checkpoint.
  @param checkpoint_path The path to the model checkpoint.
  @param properties The checkpoint properties to save.
  */
  common::Status SaveCheckpoint(
      const PathString& checkpoint_path,
      const std::unordered_map<std::string, std::string>& properties);

  /** Loads a model checkpoint.
  @param checkpoint_path The path to the model checkpoint.
  @param properties[out] The loaded checkpoint properties.
  */
  common::Status LoadCheckpointAndUpdateInitializedTensors(
      const PathString& checkpoint_path,
      std::unordered_map<std::string, std::string>& properties);

  /** Update the session initializers with passed-in state tensors
  @param state_tensors, a map of previous state, usually load from checkpoint
  @param strict, if it is true, it require current session has exact the same set of initializers as the checkpoint.
         if false, it allow some initializers are missing in checkpoints, or some initializers in checkpoint are missing in current session
  */
  common::Status UpdateInitializedTensors(const NameMLValMap& state_tensors, bool strict = false);

  // TODO: remove or refine below temp interfaces.
  NameMLValMap GetWeights() const;

  common::Status UpdateTrainableWeightsInfoInGraph();

  common::Status GetStateTensors(NameMLValMap& state_tensors);

  const DataTransferManager& GetDataTransferManager();

  // (To be deprecated)
  // Update the weights when updater is not part of the training graph
  common::Status UpdateWeightsInSessionState(const NameMLValMap& new_weights);
  std::unordered_set<std::string> GetModelInputNames() const;
  std::unordered_set<std::string> GetModelOutputNames() const;

  typedef std::unordered_map<std::string /*OpType*/,
                             std::vector<std::pair<size_t /*InputIndex*/, float /*value*/>>>
      ImmutableWeights;

  std::unordered_set<std::string> GetTrainableModelInitializers(const ImmutableWeights& immutable_weights) const;

  static bool IsImmutableWeight(const ImmutableWeights& immutable_weights,
                                const Node* node,
                                const TensorProto* weight_tensor,
                                const logging::Logger* logger = nullptr);

  static bool IsUntrainable(const Node* node,
                            const std::string& initializer_name,
                            const logging::Logger* logger = nullptr);

 private:
  std::unordered_set<std::string> weights_to_train_;
  // names of additional initializers to be included in checkpoints
  std::unordered_set<std::string> opt_state_initializer_names_;
  std::unordered_set<std::string> fp16_weight_initializer_names_;

  std::unique_ptr<ILossFunction> loss_graph_builder_;
  LossFunctionInfo loss_func_info_;
  std::string loss_scale_input_name_;

  OptimizerGraphConfig opt_graph_config_;
  std::unordered_map<std::string, OptimizerNodeConfig> opt_configs_;
};
}  // namespace training
}  // namespace onnxruntime
