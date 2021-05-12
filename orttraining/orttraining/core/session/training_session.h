// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>
#include "core/common/optional.h"
#include "core/common/path_string.h"
#include "core/session/inference_session.h"
#include "orttraining/core/framework/pipeline.h"
#include "orttraining/core/graph/loss_func/loss_func_common.h"
#include "orttraining/core/graph/loss_function_registry.h"
#include "orttraining/core/graph/optimizer_graph_output_key.h"
#include "orttraining/core/graph/optimizer_config.h"
#include "orttraining/core/graph/gradient_config.h"
#include "orttraining/core/optimizer/graph_transformer_config.h"

namespace onnxruntime {
namespace training {

constexpr char SHARED_OPTIMIZER_STATES_KEY[] = "shared_optimizer_state";

class TrainingSession : public InferenceSession {
 public:
  typedef std::unordered_map<std::string /*OpType*/,
                             std::vector<std::pair<size_t /*InputIndex*/, float /*value*/>>>
      ImmutableWeights;

  typedef std::unordered_map<std::string /* Model weight name*/,
                             NameMLValMap /* 'Moment_1': OrtValue, 'Moment_2': OrtValue etc...*/>
      OptimizerState;

  /**
   * Partition information of each paritioned weight
   */
  struct PartitionInfo {
    // value of the original shape of the weight
    std::vector<int64_t> original_dim;
    // indicates whether weight was megatron partitioned or not.
    // -1: not partitioned; 0: column partitioned; 1: row partitioned
    int megatron_row_partition = -1;
    // name of the partition used to look up partitioned weight and optimizer state values
    std::string partition_name;
    // whether the weight itself was paritioned or not(eg:just the optimizer state for fp32 Zero-1)
    bool weight_partitioned = false;
  };

  TrainingSession(const SessionOptions& session_options, const Environment& env)
      : InferenceSession(session_options, env) {}
  virtual ~TrainingSession(){};

  /**
   * The training configuration options.
   */
  struct TrainingConfiguration {
    // The path at which to save the intermediate model with the added loss function.
    optional<PathString> model_with_loss_function_path{};
    // The path at which to save the model after applying the graph transformations.
    optional<PathString> model_after_graph_transforms_path{};
    // The path at which to save the model with gradient graph added.
    optional<PathString> model_with_gradient_graph_path{};
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

    // Gradient graph configuration
    GradientGraphConfiguration gradient_graph_config{};

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
      // The number of ranks for data parallel group.
      int data_parallel_size{1};
      // The number of ranks for horizontal model parallel group.
      int horizontal_parallel_size{1};
      // The number of pipeline stages.
      int pipeline_parallel_size{1};
      // The number of micro-batches run by pipeline parallel after calling one session.Run(...).
      int num_pipeline_micro_batches{1};
      // We assume one process only run a portion of the graph when pipeline parallel is enabled.
      // This field is the graph partition's ID this process run.
      int pipeline_stage_id{0};
      // This field contains ONNX model's names for input and output tensors to be sliced.
      std::vector<std::string> sliced_tensor_names;
      // Shapes of inputs and outputs for micro-batch.
      std::unordered_map<std::string, std::vector<int>> sliced_schema;
      // The axies to slice tensors along to create tensors in micro-batch.
      // If we have a tensor named "x", slicing x along axis sliced_axes["x"] generates
      // "x" in micro-batch.
      std::unordered_map<std::string, int> sliced_axes;
    };
    // The distributed training configuration.
    DistributedConfiguration distributed_config{};

    struct MixedPrecisionConfiguration {
      // Whether to use mixed precision initializers.
      bool use_mixed_precision_initializers{};
      MixedPrecisionDataType mixed_precision_type{MixedPrecisionDataType::FP16};

      bool layernorm_stash_as_fp32{true};

      ONNX_NAMESPACE::TensorProto_DataType TensorProtoDataType() const {
        switch (mixed_precision_type) {
          case MixedPrecisionDataType::FP16:
            return ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
          case MixedPrecisionDataType::BF16:
            return ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
          default:
            return ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
        }
      }
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

    struct GistConfiguration {
      // The operator type to which GIST is applied. Valid Values - 1 (Softmax), 2 (Transpose), 3 (Reshape),
      // 4 (Add), 5 (Dropout), 6 (LayerNormalization), 7 (MatMul), 8 (Relu), 9 (All the above)
      int op_type{};
      // The compression type used for GIST. Valid values - GistBinarize, GistPack1, GistPack8, GistPack16, GistPackMsfp15
      std::string compr_type{};
    };
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
      std::function<std::unordered_map<std::string, int64_t>(const std::string&)> weight_int_attributes_generator{};

      // Whether to use mixed precision moments.
      bool use_mixed_precision_moments{};
      // Whether to use mixed precision type for the all reduce.
      bool do_all_reduce_in_mixed_precision_type{};
      // Whether to use NCCL.
      bool use_nccl{};
      // Whether to partition the optimizer state.
      ZeROConfig deepspeed_zero{};
      // Selects the reduction algorithm for Adasum.
      AdasumReductionType adasum_reduction_type{AdasumReductionType::None};
      // Whether to enable gradient clipping.
      bool enable_grad_norm_clip{true};
    };
    // The optimizer configuration.
    // If not provided, no optimizer is added.
    optional<OptimizerConfiguration> optimizer_config{};

    // optional initial states for optimizer
    // These states are partitioned wherever the weights are partitioned for eg in Zero, Megatron
    // This is loaded into the optimizer initializers when the optimizer graph is created
    optional<OptimizerState> init_optimizer_states{};

    // struct to describe a specific edge. An edge is not the same as a node_arg. Edge represents a connection between two operators.
    // For example, an operator A's output tensor T is connecting to another operator B's input, then this constructs
    // an edge from A to B. If A's output tensor T has multiple consumers, i.e. it's fed into multiple operators' inputs,
    // there would be multiple edges, each from A, to a consumer operator.
    // CutEdge information is used in pipeline online partition tool to identify which edge to cut to make the
    // corresponding partition.
    struct CutEdge {
      std::string node_arg_name;
      optional<std::vector<std::string>> consumer_nodes;

      // If the edge is unique, i.e. only have one consumer node, or all the edges
      // with the same node_arg_name needs to be cut, specify the node_arg_name
      // suffices.
      CutEdge(std::string edge) : node_arg_name(edge){};
      // If the edges with same node_arg_name belongs to different cut, i.e. some of its
      // consumer node belongs to one partition, and some belongs to another, specify
      // the consumer node names which you want to perform the cut on.
      CutEdge(std::string edge, std::vector<std::string> nodes) : node_arg_name(edge), consumer_nodes(nodes){};
    };
    // CutInfo is a group of CutEdges that describes a specific cut that composed of splitting those edges.
    typedef std::vector<CutEdge> CutInfo;

    struct PipelineConfiguration {
      // If model partition happens outside ORT, this flag should be false.
      // Otherwise, use true to trigger ORT's pipeline partition.
      bool do_partition;
      // Tensors to fetch as specified by the user.
      // Each pipeline stage should pick up some strings from this field.
      std::vector<std::string> fetch_names;
      // cut_list contains the list of CutInfo to make the graph partitions.
      // cut_list[i] contains the CutInfo to make the partition between stage i and stage i+1
      std::vector<CutInfo> cut_list;
      // Alternative for partition. We map each operator's string identifier to
      // a stage identifier. We identify operators using the name of any of
      // their outputs. All operators in the graph must be in the domain of this
      // map.
      std::map<std::string, int> op_id_to_stage;

      // The base path at which to save the intermediate partitioned input model (forward pass only).
      optional<PathString> partitioned_model_path{};
    };

    // If pipeline is enabled, this field's has_value() returns true.
    // Otherwise, it returns false.
    optional<PipelineConfiguration> pipeline_config{};

    TrainingGraphTransformerConfiguration graph_transformer_config{};
  };

  /**
   * The training configuration output.
   */
  struct TrainingConfigurationResult {
    struct MixedPrecisionConfigurationResult {
      // The name of the loss scaling factor input.
      std::string loss_scale_input_name;
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

    // The names of pipeline events in model's input list.
    // If an event is not used, its name should be empty.
    struct PipelineConfigurationResult {
      // Index of obtained pipeline stage. The first stage is indexed by 0.
      int pipeline_stage_id;
      // The names of pipeline events in model's input list.
      // This field also includes the first output name of each event operator.
      pipeline::PipelineTensorNames pipeline_tensor_names;
      // Tensors to feed at this pipeline stage.
      std::vector<std::string> feed_names;
      // Tensors to fetch at this pipeline stage.
      // It's a subset of PipelineConfiguration.fetch_names.
      std::vector<std::string> fetch_names;
    };

    // The pipeline configuration output.
    // This is only set if an pipeline is enabled.
    optional<PipelineConfigurationResult> pipeline_config_result;

    // Mapped initialized names after weight partitioning for example MegatronTransformer
    std::unordered_map<std::string, std::string> weight_name_map_after_graph_transform{};

    std::unordered_map<std::string, PartitionInfo> weight_partition_info;
  };

  /**
   * Configures the session for training.
   * Note: This is known to NOT be thread-safe.
   * @param config The training configuration.
   * @param[out] config_result The configuration output.
   * @return The status of the configuration.
   */
  virtual common::Status ConfigureForTraining(
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

  common::Status GetOptimizerState(std::unordered_map<std::string, NameMLValMap>& opt_state_tensors);

  common::Status GetModelState(std::unordered_map<std::string, NameMLValMap>& model_state_tensors, bool include_mixed_precision_weights = false);

  common::Status GetPartitionInfoMap(std::unordered_map<std::string, std::unordered_map<std::string, std::vector<int>>>& part_info_map);

  /** Gets the DataTransferManager instance. */
  const DataTransferManager& GetDataTransferManager() const;

  /** Gets the model location. */
  const PathString& GetModelLocation() const { return model_location_; }

  /**
   * Checks to be see if given graph output is produced by an fp32-only node.
   * @param The name of the output.
   * @return Whether output is from fp32-only node or not.
   */
  bool IsGraphOutputFp32Node(const std::string& output_name) const;

  /**
   * Gets the list of Dropout ratio inputs that will be used as feeds in eval mode,
   * since each ratio input has its own name.
   * @return The list of feed names.
   */
  std::unordered_set<std::string> GetDropoutEvalFeeds() const { return dropout_eval_feeds_; }

  /** Override Run function in InferenceSession to inject some training-specific logics **/
  using InferenceSession::Run;  // For overload resolution.
  common::Status Run(const RunOptions& run_options, IOBinding& io_binding) override;

 protected:
  /** Configures the loss function.
  The loss function can either be provided externally or built from the provided loss function information.
  Exactly one of external_loss_name or loss_function_info should be given.
  Optionally, a loss scaling factor can be applied to the loss function output.
  @param external_loss_name The name of the externally provided loss function output. Specifies that an external loss
         function should be used.
  @param loss_func_info The loss function information. Specifies that the loss function should be built.
  @param loss_scale_input_name[in,out] If provided, indicates that loss scaling should be applied and will be set to
         the name of the loss scale input.
  @param actual_loss_name[out] The actual name of the loss function output from which to start the backward graph.
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
  common::Status ConfigureLossFunction(
      const optional<std::string>& external_loss_name,
      const optional<LossFunctionInfo>& loss_func_info,
      std::string* loss_scale_input_name,
      std::string& actual_loss_name);

  common::Status AddGistEncoding(const int op_type, const std::string compr_type);

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

  virtual common::Status PartitionGraphForPipeline(
      const int32_t pipeline_stage_id,
      const optional<TrainingConfiguration::PipelineConfiguration>& pipeline_config,
      const optional<TrainingConfiguration::DistributedConfiguration>& distributed_config,
      const std::unordered_set<std::string>& weight_names_to_train,
      std::unordered_set<std::string>& filtered_config_weight_names_to_train);

  // Insert operators for running pipeline and return event tensor names.
  // For an intermediate pipeline stage, its original computation is
  //
  //  Recv --> Forward --> Send -->
  //  Recv --> Backward --> Send
  //
  // After this function, the resulted computation is
  //
  //  WaitEvent --> Recv --> RecordEvent --> WaitEvent --> Forward --> RecordEvent --> WaitEvent --> Send --> RecordEvent -->
  //  WaitEvent --> Recv --> RecordEvent --> WaitEvent --> Backward --> RecordEvent --> WaitEvent --> Send --> RecordEvent
  //
  // As you can see, some event operators are inserted. For each event operator, its dependent
  // event tensor name is written to an input references, for example, "forward_waited_event_name".
  //
  // This function assumes that
  //  1. Only one Recv and only one Send present in forward pass.
  //  2. Only one Recv and only one Send present in backward pass.
  //  3. Backward operators' descriptions are all "Backward pass". This assumption is used to
  //     identify backward nodes.
  //  4. No event operator is inserted by other graph transform.
  virtual common::Status SetEventSynchronization(
      const int32_t pipeline_stage_id,
      const optional<TrainingConfiguration::PipelineConfiguration>& pipeline_config,
      const optional<TrainingConfiguration::DistributedConfiguration>& distributed_config,
      const std::unordered_set<std::string>& weight_names_to_train,
      optional<TrainingConfigurationResult::PipelineConfigurationResult>& pipeline_config_result);

  common::Status ApplyTransformationsToMainGraph(const std::unordered_set<std::string>& weights_to_train,
                                                 const TrainingGraphTransformerConfiguration& config);

  common::Status ApplyModelParallelTransformationsToMainGraph(std::unordered_set<std::string>& weights_to_train,
                                                              TrainingConfigurationResult& config_result_out);

  /** configure initial transformers for training */
  void AddPreTrainingTransformers(const IExecutionProvider& execution_provider,  // for constant folding
                                  GraphTransformerManager& transformer_manager,
                                  const std::unordered_set<std::string>& weights_to_train,
                                  const TrainingGraphTransformerConfiguration& config,
                                  TransformerLevel graph_optimization_level = TransformerLevel::MaxLevel);

  /** override the parent method in inference session for training specific transformers */
  void AddPredefinedTransformers(GraphTransformerManager& transformer_manager,
                                 TransformerLevel graph_optimization_level) override;

  /** Perform auto-diff to add backward graph into the model.
  @param weights_to_train a set of weights to be training.
  @param loss_function_output_name the name of the loss function's output.
  */
  common::Status BuildGradientGraph(const std::unordered_set<std::string>& weights_to_train,
                                    const std::string& loss_function_output_name,
                                    const GradientGraphConfiguration& gradient_graph_config,
                                    const logging::Logger& logger);

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

  common::Status BuildLoss(
      const optional<std::string>& external_loss_name,
      std::string& loss_name,
      const optional<TrainingConfiguration::LossFunctionConfiguration>& loss_function_config,
      optional<std::string>& loss_scale_input_name);

  virtual common::Status BuildLossAndLossScaling(
      const int32_t pipeline_stage_id,
      const optional<std::string>& external_loss_name,
      const optional<TrainingConfiguration::MixedPrecisionConfiguration>& mixed_precision_config,
      const optional<TrainingConfiguration::DistributedConfiguration>& distributed_config,
      const optional<TrainingConfiguration::LossFunctionConfiguration>& loss_function_config,
      std::string& loss_name,
      optional<std::string>& loss_scale_input_name,
      optional<TrainingConfigurationResult::MixedPrecisionConfigurationResult>& mixed_precision_config_result);

  /** Enable mixed precision training
  @param weights_to_train a set of weights to be training.
  @param mixed_precision_config The mixed precision configuration.
  @param fp32_weight_name_to_mixed_precision_node_arg the map between weights and mixed precision weights.
  */
  common::Status EnableMixedPrecision(const std::unordered_set<std::string>& weights_to_train,
                                      const TrainingConfiguration::MixedPrecisionConfiguration& mixed_precision_config,
                                      std::unordered_map<std::string, NodeArg*>& fp32_weight_name_to_mixed_precision_node_arg);

  /** Discover all trainable initializers by reverse DFS starting from a given tensor (for example, the loss value)
  @param immutable_weights do not include initializers matching an (op_type, input_index, value) entry from this table
  @param backprop_source_name reverse DFS back propagation source name (i.e. loss name or pipeline send output name)
  */
  std::unordered_set<std::string> GetTrainableModelInitializers(const ImmutableWeights& immutable_weights,
                                                                const std::string& backprop_source_name) const;

  std::unordered_set<std::string> GetStateTensorNames() const;

  common::Status SetEvalFeedNames();

  NameMLValMap GetWeights() const;

  void FilterUnusedWeights(const std::unordered_set<std::string>& weight_names_to_train,
                           std::unordered_set<std::string>& filtered_weight_names_to_train);

  static bool IsImmutableWeight(const ImmutableWeights& immutable_weights,
                                const Node* node,
                                const TensorProto* weight_tensor,
                                const logging::Logger* logger = nullptr);

  static bool IsUntrainable(const Node* node,
                            const std::string& initializer_name,
                            const logging::Logger* logger = nullptr);

  bool is_configured_{false};

  std::unordered_set<std::string> weights_to_train_;
  OptimizerState init_optimizer_states_;
  // names of additional initializers to be included in checkpoints
  std::unordered_map<std::string, std::string> updated_weight_names_map_;
  std::unordered_set<std::string> opt_state_initializer_names_;
  std::unordered_map<std::string, std::string> weight_to_mixed_precision_map_;
  std::unordered_map<std::string, std::unordered_map<std::string, std::string>> weight_to_opt_mapping_;
  std::unordered_map<std::string, TrainingSession::PartitionInfo> weight_partition_info_;

  bool is_mixed_precision_enabled_;
  optional<std::string> external_loss_name_;
  std::unique_ptr<ILossFunction> loss_graph_builder_;
  optional<LossFunctionInfo> loss_function_info_;

  std::unordered_set<std::string> dropout_eval_feeds_;
  OptimizerGraphConfig opt_graph_config_;
  std::unordered_map<std::string, OptimizerNodeConfig> opt_configs_;

  GradientGraphConfiguration gradient_graph_config_;
  static const std::string training_mode_string_;
};

class PipelineTrainingSession final : public TrainingSession {
 public:
  PipelineTrainingSession(const SessionOptions& session_options, const Environment& env)
      : TrainingSession(session_options, env) {}
  common::Status ConfigureForTraining(const TrainingConfiguration& config, TrainingConfigurationResult& config_result_out) override;
  common::Status Run(const RunOptions& run_options, IOBinding& io_binding) override;
  ~PipelineTrainingSession();

 protected:
  common::Status PartitionGraphForPipeline(
      const int32_t pipeline_stage_id,
      const optional<TrainingConfiguration::PipelineConfiguration>& pipeline_config,
      const optional<TrainingConfiguration::DistributedConfiguration>& distributed_config,
      const std::unordered_set<std::string>& weight_names_to_train,
      std::unordered_set<std::string>& filtered_config_weight_names_to_train) override;

  common::Status SetEventSynchronization(
      const int32_t pipeline_stage_id,
      const optional<TrainingConfiguration::PipelineConfiguration>& pipeline_config,
      const optional<TrainingConfiguration::DistributedConfiguration>& distributed_config,
      const std::unordered_set<std::string>& weight_names_to_train,
      optional<TrainingConfigurationResult::PipelineConfigurationResult>& pipeline_config_result) override;

  common::Status BuildLossAndLossScaling(
      const int32_t pipeline_stage_id,
      const optional<std::string>& external_loss_name,
      const optional<TrainingConfiguration::MixedPrecisionConfiguration>& mixed_precision_config,
      const optional<TrainingConfiguration::DistributedConfiguration>& distributed_config,
      const optional<TrainingConfiguration::LossFunctionConfiguration>& loss_function_config,
      std::string& loss_name,
      optional<std::string>& loss_scale_input_name,
      optional<TrainingConfigurationResult::MixedPrecisionConfigurationResult>& mixed_precision_config_result) override;

  // Set some PipelineContext fields based on configuration result
  // returned by TrainingSession::ConfigureForTraining.
  common::Status SetPipelineContext(const TrainingConfigurationResult& config_result);

  common::Status SetExtraDataDependency();

  void CreatePipelineEvents(
      const bool traning_mode,
      const int batch_id,
      const int stage_id,
      IOBinding& io_binding);

  void CreateMicroBatchVariables(
      IOBinding& io_binding, IOBinding& sub_io_binding,
      const size_t slice_id, const size_t num_slices);

#if defined(USE_CUDA) && defined(ORT_USE_NCCL) && defined(USE_NCCL_P2P)
  void LaunchNcclService(const int pipeline_stage_id);
#endif

  common::Status RunWithPipeline(const RunOptions& run_options, IOBinding& io_binding);

  // Pipeline fields are valid only if params_.pipeline_parallel_size > 1.
  // Information for running pipeline.
  pipeline::PipelineContext pipeline_context_;
  // Pipeline schedule for deciding when to run batch, forward, or backward.
  pipeline::PipelineScheduler pipeline_schedule_;
  // Workers to run pipeline stage.
  pipeline::PipelineWorkerPool pipeline_worker_pool_;
};
}  // namespace training
}  // namespace onnxruntime
