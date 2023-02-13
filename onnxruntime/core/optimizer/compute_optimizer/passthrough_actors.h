// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_CORE
#pragma once

// Uncomment for debugging
// #define NEED_LOG_DEBUG_INFO 1

#ifdef NEED_LOG_DEBUG_INFO
#define LOG_DEBUG_INFO(logger, message) LOGS(logger, WARNING) << message
#else
#define LOG_DEBUG_INFO(logger, message) \
  ORT_UNUSED_PARAMETER(logger);         \
  do {                                  \
  } while (0)
#endif

namespace onnxruntime::optimizer::compute_optimizer {

/**
 * @brief Struct to hold the information of the slicing operations.
 *
 * Initially, an instance of this class for entry node is created, as the slice op propagates to entry node's inputs,
 * more instances of this class are created. The propagation stops when the all inputs are not supported to be sliced.
 */
struct SliceInfo {
  static constexpr int kSliceDataInputIndex = 0;
  static constexpr int kSliceOutputIndex = 0;

  SliceInfo(Node* slice_node,
            bool is_slice_scalar,
            const std::string& slice_axis_attr_name,
            int slice_axis,
            bool is_entry_node_ptr = false)
      : node_ptr(slice_node), is_scalar_slice(is_slice_scalar) {
    axis_attr_name = slice_axis_attr_name;

    const NodeArg* input = node_ptr->InputDefs()[kSliceDataInputIndex];
    const NodeArg* output = node_ptr->OutputDefs()[kSliceOutputIndex];
    input_rank = input->Shape()->dim_size();
    non_negative_axis = slice_axis < 0 ? input_rank + slice_axis : slice_axis;

    if (!is_scalar_slice) {
      output_dim_on_axis = output->Shape()->dim(non_negative_axis);
    }

    if (is_entry_node_ptr) {
      entry_slice_arg_name = node_ptr->OutputDefs()[kSliceOutputIndex]->Name();
    }
  }

  int GetDataInputIndex() const {
    return kSliceDataInputIndex;
  }

  int GetOutputIndex() const {
    return kSliceOutputIndex;
  }

  Node* node_ptr;        // The Gather/GatherND node that triggers the optimization search.
  bool is_scalar_slice;  // whether the slice is a scalar, if it is, after Gather, rank will be reduced by 1.
  std::string axis_attr_name;
  int non_negative_axis;  // The axis to slice on
  std::string entry_slice_arg_name;

  int input_rank;  // rank of the Gather data input tensor

  // The dimension of the output tensor on the slicing axis
  // Be noted: if it is a scalar slicing, this dim will not be set, which means, afterward when use it to update
  // shapes, that dim at axis will be removed.
  ONNX_NAMESPACE::TensorShapeProto_Dimension output_dim_on_axis;
};

/**
 * @brief Base class for all pass through actors.
 *
 * Each actors defines rules to determine whether a node can be passed through, and how to do the pass through.
 * PreCheck is the interface to check whether a node can be passed through.
 * The pass through is done transparently, without any interface required to implemented.
 * PostProcess is the interface to do some adaptor work after the pass through.
 */
class OperatorPassThroughActorBase {
 public:
  OperatorPassThroughActorBase() = default;
  virtual ~OperatorPassThroughActorBase() = default;

  /**
   * @brief Check whether a node can be passed through.
   *  At this point, graph modification is not started, once we see any clues that this node cannot be passed through,
   *  We should return false immediately.
   *
   * @param graph The graph that the node belongs to.
   * @param current_node The node to be checked.
   * @param info The slicing info of the Gather/GatherND node.
   * @param allowed_input_indices The input indices explicitly specified of the current_node that are allowed to do pass
   *  through.
   * @param propagate_input_config: Used as a return value - a map of input index to new slice axis.
   *  The key is an integer, which is the index of the input of the current_node.
   *  The value is an integer, which is the new axis index after the pass through on the corresponding input.
   *  For example:
   *    > if the current_node is a Add node, and the slice axe is 1, then the corresponding input should
   *      also have axis 1 when we move the slice to the input.
   *    > if the current_node is a Transpose (perm=[1, 0, 2]) node, and the slice
   *      axis is 1, then the new axis for the input should be 0.
   * @param input_has_dim_1_for_axis: Used as a return value - a bool indicates whether any of current_node' input
   *  has dim 1 on the slice axis.
   */
  virtual bool PreCheck(const Graph& graph, const Node& current_node, const SliceInfo& info,
                        const std::vector<int>& allowed_input_indices,
                        const logging::Logger& logger,
                        std::unordered_map<int, int>& propagate_input_config,
                        bool& input_has_dim_1_for_axis) = 0;

  /**
   * @brief After slice op pass through all inputs, do some post process work.
   *
   * Be noted: at this point, slice op is already removed, so we cannot access SliceInfo any more, instead,
   * we pass important infos including slice_axis, input_rank, is_scalar_slice, etc as parameters of this function.
   *
   * @param graph The graph that the node belongs to.
   * @param current_node The node that has been passed through.
   * @param current_node_output_index The output index of the current_node connecting to slice op.
   * @param slice_axis slice axis of the slice op.
   * @param is_slice_scalar whether the slice is a scalar.
   * @param input_has_dim_1_for_axis whether any of current_node's inputs has dim 1 on the slice axis.
   * @param output_dim_on_axis dimension of the slice op's output tensor on the slice axis.
   * @param entry_node_name name of entry node that trigger the pass through search, for naming only.
   * @param new_gather_infos new gather infos that are generated during the pass through for current_node's inputs.
   * @param logger
   * @return
   */
  virtual bool PostProcess(Graph& graph, Node& current_node, int current_node_output_index,
                           int slice_axis, bool is_slice_scalar, bool input_has_dim_1_for_axis,
                           const ONNX_NAMESPACE::TensorShapeProto_Dimension& output_dim_on_axis,
                           const std::string& entry_node_name,
                           const std::unordered_map<int, SliceInfo>& new_gather_infos,
                           const logging::Logger& logger) = 0;
};

class DefaultOperatorPassThroughActorBase : public OperatorPassThroughActorBase {
 public:
  DefaultOperatorPassThroughActorBase() = default;
  ~DefaultOperatorPassThroughActorBase() = default;

  bool PreCheck(const Graph&, const Node&, const SliceInfo&, const std::vector<int>&, const logging::Logger&,
                std::unordered_map<int, int>&, bool&) override {
    return true;
  };

  bool PostProcess(Graph& graph, Node& current_node, int current_node_output_index,
                   int slice_axis, bool is_slice_scalar, bool input_has_dim_1_for_axis,
                   const ONNX_NAMESPACE::TensorShapeProto_Dimension& output_dim_on_axis,
                   const std::string& entry_node_name,
                   const std::unordered_map<int, SliceInfo>& new_gather_infos,
                   const logging::Logger& logger) override;
};

class SimplePassThroughActor : public DefaultOperatorPassThroughActorBase {
 public:
  SimplePassThroughActor() = default;
  ~SimplePassThroughActor() = default;

  bool PreCheck(const Graph& graph, const Node& current_node, const SliceInfo& info,
                const std::vector<int>& allowed_input_indices,
                const logging::Logger& logger,
                std::unordered_map<int, int>& propagate_input_config,
                bool& input_has_dim_1_for_axis) override;
};

class ReductionOpPassThroughActor : public SimplePassThroughActor {
 public:
  ReductionOpPassThroughActor() = default;
  ~ReductionOpPassThroughActor() = default;

  bool PreCheck(const Graph& graph, const Node& current_node, const SliceInfo& info,
                const std::vector<int>& allowed_input_indices,
                const logging::Logger& logger,
                std::unordered_map<int, int>& propagate_input_config,
                bool& input_has_dim_1_for_axis) override;
};

class ReshapePassThroughActor : public DefaultOperatorPassThroughActorBase {
 public:
  ReshapePassThroughActor() = default;
  ~ReshapePassThroughActor() = default;

  bool PreCheck(const Graph& graph, const Node& current_node, const SliceInfo& info,
                const std::vector<int>& allowed_input_indices,
                const logging::Logger& logger,
                std::unordered_map<int, int>& propagate_input_config,
                bool& input_has_dim_1_for_axis) override;

  // Once slice node is passed through, we need to update the shape accordingly.
  bool PostProcess(Graph& graph, Node& current_node, int current_node_output_index,
                   int slice_axis, bool is_slice_scalar, bool input_has_dim_1_for_axis,
                   const ONNX_NAMESPACE::TensorShapeProto_Dimension& output_dim_on_axis,
                   const std::string& entry_node_name,
                   const std::unordered_map<int, SliceInfo>& new_gather_infos,
                   const logging::Logger& logger) override;
};

class TransposePassThroughActor : public DefaultOperatorPassThroughActorBase {
 public:
  TransposePassThroughActor() = default;
  ~TransposePassThroughActor() = default;

  bool PreCheck(const Graph& graph, const Node& current_node, const SliceInfo& info,
                const std::vector<int>& allowed_input_indices,
                const logging::Logger& logger,
                std::unordered_map<int, int>& propagate_input_config,
                bool& input_has_dim_1_for_axis) override;

  // If scalar slice happens, we need adapt the input, otherwise the perm cannot be matched.
  bool PostProcess(Graph& graph, Node& current_node, int current_node_output_index,
                   int slice_axis, bool is_slice_scalar, bool input_has_dim_1_for_axis,
                   const ONNX_NAMESPACE::TensorShapeProto_Dimension& output_dim_on_axis,
                   const std::string& entry_node_name,
                   const std::unordered_map<int, SliceInfo>& new_gather_infos,
                   const logging::Logger& logger) override;
};

class MatMulPassThroughActor : public DefaultOperatorPassThroughActorBase {
 public:
  MatMulPassThroughActor() = default;
  ~MatMulPassThroughActor() = default;

  // Check which inputs can be propagated according to the slice axis.
  bool PreCheck(const Graph& graph, const Node& current_node, const SliceInfo& info,
                const std::vector<int>& allowed_input_indices,
                const logging::Logger& logger,
                std::unordered_map<int, int>& propagate_input_config,
                bool& input_has_dim_1_for_axis) override;

  // If scalar slice happens in the second last dimension, we need to adapt the input.
  bool PostProcess(Graph& graph, Node& current_node, int current_node_output_index,
                   int slice_axis, bool is_slice_scalar, bool input_has_dim_1_for_axis,
                   const ONNX_NAMESPACE::TensorShapeProto_Dimension& output_dim_on_axis,
                   const std::string& entry_node_name,
                   const std::unordered_map<int, SliceInfo>& new_gather_infos,
                   const logging::Logger& logger) override;
};

/**
 * @brief Update the dim value using given new dim value at specified axis.
 *
 * @param arg_to_update The NodeArg to be updated.
 * @param reverse_axis A negative axis MUST be given here. This is to make sure if arg_to_update has less rank
 *   than expected value, the update will be ignored.
 * @param output_dim_on_axis New dim value to be updated.
 * @return true if the update is done.
 */
bool UpdateSliceOutputShape(NodeArg& arg_to_update, int reverse_axis,
                            const ONNX_NAMESPACE::TensorShapeProto_Dimension& new_dim_value);

/**
 * @brief Insert a new node to the graph,
 *  1. taking dest_node.input[dest_input_index] as the input of the new node.
 *  2. remove connection of dest_node and it's dest_input_index-th input producer node.
 *  3. connect the new node and dest_node.
 *
 * Original graph:
 *         Node A
 *       /        \
 *  A-output-0    A-output-1
 *                  \         B-input-1
 *                   \       /
 *                     Node B
 *                       |
 *
 * dest_node = Node B
 * dest_input_index = 0
 * op_type = C
 *
 * After inserting the new node:
 *         Node A
 *       /        \
 *  A-output-0    A-output-1
 *                  \
 *                 Node C
 *                  /  \
 *         c-output-0  C-output-1     B-input-1
 *                         \       /
 *                          Node B
 *                             |
 * @param graph  Graph to insert the new node.
 * @param dest_node The node to insert the new node before.
 * @param dest_in_index The input index of the dest_node to insert the new node before.
 * @param new_node_output_index The output index of the new node to connect to the dest_node.
 * @param name The name of the new node.
 * @param op_type The op_type of the new node.
 * @param description The description of the new node.
 * @param input_args The input args of the new node. At least one of the input args should be the
 *   dest_node's dest_in_index-th input arg.
 * @param attributes The attributes of the new node.
 * @param domain The domain of the new node.
 * @param logger The logger.
 * @return
 */
Node* InsertIntermediateNodeOnDestInput(Graph& graph,
                                        Node& dest_node, int dest_in_index,
                                        int new_node_input_index,
                                        int new_node_output_index,
                                        const std::string& name, const std::string& op_type,
                                        const std::string& description,
                                        const InlinedVector<NodeArg*>& input_args,
                                        const InlinedVector<NodeArg*>& output_args,
                                        const onnxruntime::NodeAttributes& attributes,
                                        const std::string& domain,
                                        const logging::Logger& logger);

/**
 * @brief Insert adaptor nodes for the inputs and output, to make sure they remain the same rank, when scalar slicing
 *  is done.
 *
 * Be noted: at this point, slice node already been removed.
 *
 * @param graph Graph to insert the adaptor nodes.
 * @param current_node For whom to insert the adaptor nodes.
 * @param slice_axis The axis of the slice node.
 * @param entry_node_name Then name of the entry slice node, used for naming only.
 * @param new_gather_infos Populated slicing infos for current_node's inputs.
 * @param target_node_output_index output_index of current_node's output, connecting to the slice node.
 * @param logger Logger.
 */
void AdaptInputAndOutputForScalarSlice(Graph& graph, Node& current_node, int current_node_output_index,
                                       int slice_axis, const std::string& entry_node_name,
                                       const std::unordered_map<int, SliceInfo>& new_gather_infos,
                                       const logging::Logger& logger);

}  // namespace onnxruntime::optimizer::compute_optimizer

#endif
