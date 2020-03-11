// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>
#include "core/common/optional.h"
#include "core/common/path_string.h"
#include "core/session/inference_session.h"
#include "orttraining/core/graph/loss_func/loss_func_common.h"
#include "orttraining/core/graph/loss_function_registry.h"
#include "orttraining/core/graph/optimizer_graph_output_key.h"
#include "orttraining/core/graph/optimizer_config.h"

namespace onnxruntime {
namespace training {

class TrainingSession : public InferenceSession {
 public:
  typedef std::unordered_map<std::string /*OpType*/,
                             std::vector<std::pair<size_t /*InputIndex*/, float /*value*/>>>
      ImmutableWeights;

  explicit TrainingSession(const SessionOptions& session_options,
                           logging::LoggingManager* logging_manager = nullptr)
      : InferenceSession(session_options, logging_manager) {}

  /**
   * The training configuration options.
   */
  struct TrainingConfiguration {
    // The path at which to save the intermediate model with the added loss function.
    optional<PathString> model_with_loss_function_path{};
    // The path at which to save the intermediate model with the whole training graph.
    optional<PathString> model_with_training_graph_path{};

    // The names of the weights to train.
    // If empty, a default set is used.
    std::unordered_set<std::string> weight_names_to_train{};
    // The names of the weights to not train.
    // These are removed from the set of names of weights to train.
    std::unordered_set<std::string> weight_names_to_not_train{};

    // The immutable weights specification.
    ImmutableWeights immutable_weights;

    // Whether to set the gradients as graph outputs.
    bool set_gradients_as_graph_outputs{false};

    // The number of gradient accumulation steps.
    int gradient_accumulation_steps{1};

    struct DistributedConfiguration {
      // The rank of the node.
      int world_rank{0};
      // The local rank.
      int local_rank{0};
      // The total number of ranks.
      int world_size{1};
      // The number of local ranks on a node.
      int local_size{1};
      // The number of ranks for data parallel group
      int data_parallel_size{1};
      // The number of ranks for horizontal model parallel group
      int horizontal_parallel_size{1};
    };
    // The distributed training configuration.
    DistributedConfiguration distributed_config{};

    struct MixedPrecisionConfiguration {
      // Whether to add loss scaling.
      bool add_loss_scaling{};
      // The initial loss scaling factor.
      float initial_loss_scale_value{};
      // Whether to use FP16 initializers.
      bool use_fp16_initializers{};
    };
    // The mixed precision configuration.
    // If not provided, mixed precision is disabled.
    optional<MixedPrecisionConfiguration> mixed_precision_config{};

    struct LossFunctionConfiguration {
      // The loss function configuration options.
      LossFunctionInfo loss_function_info{};
    };
    // The loss function configuration.
    // If not provided, no loss function is added and an external one is expected.
    // Exactly one of loss_function_config or loss_name should be given.
    optional<LossFunctionConfiguration> loss_function_config{};
    // The name of the external loss function's output.
    // Exactly one of loss_function_config or loss_name should be given.
    optional<std::string> loss_name{};

    struct GistConfiguration {};
    // The GIST configuration.
    // If not provided, GIST is disabled.
    optional<GistConfiguration> gist_config{};

    struct TensorboardConfiguration {
      // The summary name.
      std::string summary_name{};
      // The names of the scalar nodes.
      std::vector<std::string> scalar_node_names{};
      // The names of the histogram nodes.
      std::vector<std::string> histogram_node_names{};
      // The names of the norm nodes.
      std::vector<std::string> norm_node_names{};
      // Whether to dump the convergence metrics.
      bool dump_convergence_metrics{};
    };
    // The TensorBoard configuration.
    // If not provided, TensorBoard output is disabled.
    optional<TensorboardConfiguration> tensorboard_config{};

    struct OptimizerConfiguration {
      // The optimizer name.
      std::string name{};
      // The learning rate input name.
      std::string learning_rate_input_name{};
      // The per-weight attribute map generator.
      // It should accept a weight name and return the appropriate attribute map.
      std::function<std::unordered_map<std::string, float>(const std::string&)> weight_attributes_generator{};
      // Whether to use FP16 moments.
      bool use_fp16_moments{};
      // Whether to use FP16 for the all reduce.
      bool do_all_reduce_in_fp16{};
      // Whether to use NCCL.
      bool use_nccl{};
      // Whether to partition the optimizer state.
      bool partition_optimizer{};
      // Selects the reduction algorithm for Adasum.
      AdasumReductionType adasum_reduction_type{AdasumReductionType::None};
    };
    // The optimizer configuration.
    // If not provided, no optimizer is added.
    optional<OptimizerConfiguration> optimizer_config{};
  };

  /**
   * The training configuration output.
   */
  struct TrainingConfigurationResult {
    struct MixedPrecisionConfigurationResult {
      // The name of the loss scaling factor input if loss scaling was added.
      optional<std::string> loss_scale_input_name;
    };
    // The mixed precision configuration output.
    // This is only set if mixed precision is enabled.
    optional<MixedPrecisionConfigurationResult> mixed_precision_config_result;

    struct OptimizerConfigurationResult {
      // The mapping of optimizer output key to graph output name.
      OptimizerOutputKeyMap<std::string> output_key_to_graph_output_name;
    };
    // The optimizer configuration output.
    // This is only set if an optimizer is added.
    optional<OptimizerConfigurationResult> opt_config_result;
  };

  /**
   * Configures the session for training.
   * Note: This is known to NOT be thread-safe.
   * @param config The training configuration.
   * @param[out] config_result The configuration output.
   * @return The status of the configuration.
   */
  common::Status ConfigureForTraining(
      const TrainingConfiguration& config, TrainingConfigurationResult& config_result);

  /**
   * Overrides the graph outputs with the specified output names.
   * @param outputs The new output names.
   * @return The status of the operation.
   */
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

  /** Update the session initializers with passed-in state tensors
   * @param state_tensors A map of state tensors to set, usually loaded from a checkpoint.
   * @param strict Whether entries in state_tensors which are unknown or not present in the model are treated as an error or ignored.
   */
  common::Status SetStateTensors(const NameMLValMap& state_tensors, bool strict = false);

  /**
   * Gets the state tensors.
   * @param[out] The state tensors.
   * @return The status of the operation.
   */
  common::Status GetStateTensors(NameMLValMap& state_tensors);

  /** Gets the DataTransferManager instance. */
  const DataTransferManager& GetDataTransferManager() const;

  /** Gets the model location. */
  const PathString& GetModelLocation() const { return model_location_; }

 private:
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

  /** configure initial transformers for training */
  void AddPreTrainingTransformers(GraphTransformerManager& transformer_manager,
                                  TransformerLevel graph_optimization_level = TransformerLevel::MaxLevel,
                                  const std::vector<std::string>& custom_list = {});

  /** override the parent method in inference session for training specific transformers */
  void AddPredefinedTransformers(GraphTransformerManager& transformer_manager,
                                 TransformerLevel graph_optimization_level,
                                 const std::vector<std::string>& custom_list) override;

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
      OptimizerOutputKeyMap<std::string>& opt_graph_outputs);

  /** Enable mixed precision training
  @param weights_to_train a set of weights to be training.
  @param use_fp16_initializer specify whether fp16 initialier is created.
  @param fp32_weight_name_to_fp16_node_arg the map between weights and FP16 weights.
  */
  common::Status EnableMixedPrecision(const std::unordered_set<std::string>& weights_to_train,
                                      bool use_fp16_initializer,
                                      std::unordered_map<std::string, NodeArg*>& fp32_weight_name_to_fp16_node_arg);

  std::unordered_set<std::string> GetTrainableModelInitializers(const ImmutableWeights& immutable_weights) const;

  std::unordered_set<std::string> GetStateTensorNames() const;

  NameMLValMap GetWeights() const;

  static bool IsImmutableWeight(const ImmutableWeights& immutable_weights,
                                const Node* node,
                                const TensorProto* weight_tensor,
                                const logging::Logger* logger = nullptr);

  static bool IsUntrainable(const Node* node,
                            const std::string& initializer_name,
                            const logging::Logger* logger = nullptr);

  bool is_configured_{false};

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
