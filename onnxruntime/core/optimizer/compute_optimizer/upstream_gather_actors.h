// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The optimization here ideally applies to both training and inferencing,
// while so far we mainly validate training during cooking the optimization.
#ifdef ENABLE_TRAINING
#pragma once

#include "core/optimizer/compute_optimizer/shared_utils.h"

namespace onnxruntime::optimizer::compute_optimizer {

/**
 * @brief Struct to hold the information of the slicing operations.
 *
 * Initially, an instance of this class for the entry node is created, as the slice op propagates to the entry node's
 * inputs, more instances of this class are created. The propagation stops when all inputs are not supported to
 * be sliced.
 */
struct SliceInfo : public UpstreamOperatorInfoBase {
 private:
  static constexpr int kSliceDataInputIndex_ = 0;
  static constexpr int kSliceOutputIndex_ = 0;

 public:
  SliceInfo(const Graph& graph, Node* slice_node,
            bool is_slice_scalar,
            std::variant<std::string, int> axis_name_or_index,
            int slice_axis,
            int rank_of_axis,
            bool is_entry_node_ptr = false)
      : UpstreamOperatorInfoBase(slice_node, is_entry_node_ptr), is_scalar_slice(is_slice_scalar) {
    axis_attr_name_or_input_index = axis_name_or_index;
    rank_of_axis_value = rank_of_axis;

    if (std::holds_alternative<int>(axis_name_or_index)) {
      int axis_input_index = std::get<int>(axis_name_or_index);
      ORT_ENFORCE(axis_input_index >= 0, "Axis input index is invalid");
    }

    ORT_ENFORCE(rank_of_axis_value == 0 || rank_of_axis_value == 1, "Rank of axis value is invalid: " +
                                                                        std::to_string(rank_of_axis_value));

    const NodeArg* input = node_ptr->InputDefs()[kSliceDataInputIndex_];
    const NodeArg* output = node_ptr->OutputDefs()[kSliceOutputIndex_];
    input_rank = input->Shape()->dim_size();
    non_negative_axis = slice_axis < 0 ? input_rank + slice_axis : slice_axis;

    if (!is_scalar_slice) {
      output_dim_on_axis = output->Shape()->dim(non_negative_axis);
    }

    if (is_entry_node_ptr) {
      entry_slice_arg_name = output->Name();
    }

    const Node* producer = graph.GetProducerNode(input->Name());
    if (producer) {
      // Allow the data input to be graph input or initializer, but this won't be passed through further.
      slice_data_producer_output_index_ = optimizer_utils::IndexOfNodeOutput(*producer, *input);
    }
  }

  int GetDataInputIndex() const {
    return kSliceDataInputIndex_;
  }

  int GetOutputIndex() const {
    return kSliceOutputIndex_;
  }

  int GetDataProducerOutputIndex() const {
    ORT_ENFORCE(slice_data_producer_output_index_ >= 0, "Data producer output index is not set");
    return slice_data_producer_output_index_;
  }

  bool is_scalar_slice;  // whether the slice is a scalar, if it is after Gather, the rank will be reduced by 1.

  // The index of the input that contains the axis value. If it is a string, then axis will be treated as an attribute.
  std::variant<std::string, int> axis_attr_name_or_input_index;

  int non_negative_axis;  // The axis to slice on

  // The rank of value for axis attribute. For example, for Gather, its axis attribute is a scalar, so the rank is 0.
  // For Slice, its axes attribute is a 1D tensor, so the rank is 1.
  int rank_of_axis_value;

  std::string entry_slice_arg_name;

  int input_rank;  // rank of the Gather data input tensor

  // The dimension of the output tensor on the slicing axis
  // Be noted: if it is a scalar slicing, this dim will not be set, which means, afterward when using it to update
  // shapes, that dim at the axis will be removed.
  ONNX_NAMESPACE::TensorShapeProto_Dimension output_dim_on_axis;

 private:
  int slice_data_producer_output_index_{-1};
};

/**
 * @brief Base class for all pass-through actors.
 *
 * Each actor defines rules to determine whether a node can be passed through, and post-process after pass through.
 * PreCheck is the interface to check whether a node can be passed through.
 * The pass-through is done transparently, without any interface required to implement.
 * PostProcess is the interface to do some adaptor work after the pass-through.
 */
class UpStreamGatherOperatorActorBase : public UpStreamOperatorActorBase {
 public:
  UpStreamGatherOperatorActorBase() = default;
  virtual ~UpStreamGatherOperatorActorBase() = default;

  /**
   * @brief Check whether a node can be passed through.
   *  At this point, graph modification is not started, once we see any clues that this node cannot be passed through,
   *  We should return false immediately.
   *
   * @param current_node The node to be checked.
   * @param info The slicing info of the Gather/GatherND node.
   * @param propagate_input_indices: Used as a return value - a map of input index to new slice axis.
   *  The key is an integer, which is the index of the input of the current_node.
   *  The value is an integer, which is the new axis index after the pass-through on the corresponding input.
   *  For example:
   *    > if the current_node is an Add node, and the slicing axe is 1, then the corresponding input should
   *      also have axis 1 when we move the slice to the input.
   *    > if the current_node is a Transpose (perm=[1, 0, 2]) node, and the slice
   *      axis is 1, then the new axis for the input should be 0.
   * @param all_input_cmp_rets: used as a return value - a map of shape compare result for ALL input of current_node.
   *   This will be used later to check whether we need adapt for some inputs.
   * @param shape_update_func: used as a return value - a functor used to update shapes for current_node's all outputs.
   */
  virtual bool PreCheck(const Graph& graph, const Node& current_node, const SliceInfo& info,
                        const logging::Logger& logger,
                        std::unordered_map<int, int>& propagate_input_indices,
                        std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                        std::function<void(Node& node)>& shape_update_func) = 0;

  /**
   * @brief After slice op pass through all inputs, do some post-process work.
   *
   * @param graph The graph that the node belongs to.
   * @param current_node The node that has been passed through.
   * @param info_without_node The slicing info of the Gather/GatherND node. BUT AT THIS POINT, the node
   *   pointer is invalid, usage for it is not allowed.
   * @param propagate_input_indices: a map of input index to new slice axis, generated by PreCheck().
   * @param all_input_cmp_rets: a map of shape compare result for ALL input of current_node, generated by PreCheck()
   * @param new_gather_infos new gather infos that are generated during the pass through for current_node's inputs.
   * @return
   */
  virtual bool PostProcess(Graph& graph, Node& current_node, const SliceInfo& info_without_node,
                           const logging::Logger& logger,
                           const std::unordered_map<int, int>& propagate_input_indices,
                           const std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                           const std::unordered_map<int, SliceInfo>& new_gather_infos) = 0;
};

template <bool AreAllOutputShapesEqual>
class SimplePointwiseGatherActor : public UpStreamGatherOperatorActorBase {
 public:
  SimplePointwiseGatherActor() = default;
  ~SimplePointwiseGatherActor() = default;

  bool PreCheck(const Graph& graph, const Node& current_node, const SliceInfo& info,
                const logging::Logger& logger,
                std::unordered_map<int, int>& propagate_input_indices,
                std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                std::function<void(Node& node)>& shape_update_func) override;

  bool PostProcess(Graph& graph, Node& current_node, const SliceInfo& info_without_node,
                   const logging::Logger& logger,
                   const std::unordered_map<int, int>& propagate_input_indices,
                   const std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                   const std::unordered_map<int, SliceInfo>& new_gather_infos) override;
};

class LayerNormalizationGatherActor : public UpStreamGatherOperatorActorBase {
 public:
  LayerNormalizationGatherActor() = default;
  ~LayerNormalizationGatherActor() = default;

  bool PreCheck(const Graph& /* graph */, const Node& current_node, const SliceInfo& info,
                const logging::Logger& logger,
                std::unordered_map<int, int>& propagate_input_indices,
                std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                std::function<void(Node& node)>& shape_update_func) override;

  bool PostProcess(Graph& /* graph */, Node& /* current_node */, const SliceInfo& /* info_without_node */,
                   const logging::Logger& /* logger */,
                   const std::unordered_map<int, int>& /* propagate_input_indices */,
                   const std::unordered_map<int, std::vector<DimCompare>>& /* all_input_cmp_rets */,
                   const std::unordered_map<int, SliceInfo>& /* new_gather_infos */) override { return true; }
};

class SoftmaxGatherActor : public SimplePointwiseGatherActor<true> {
 public:
  SoftmaxGatherActor() = default;
  ~SoftmaxGatherActor() = default;

  bool PreCheck(const Graph& graph, const Node& current_node, const SliceInfo& info,
                const logging::Logger& logger,
                std::unordered_map<int, int>& propagate_input_indices,
                std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                std::function<void(Node& node)>& shape_update_func) override;
};

class ReshapeGatherActor : public UpStreamGatherOperatorActorBase {
 public:
  ReshapeGatherActor() = default;
  ~ReshapeGatherActor() = default;

  bool PreCheck(const Graph& graph, const Node& current_node, const SliceInfo& info,
                const logging::Logger& logger,
                std::unordered_map<int, int>& propagate_input_indices,
                std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                std::function<void(Node& node)>& shape_update_func) override;

  // Once slice node is passed through, we need to update the `shape` constant accordingly.
  bool PostProcess(Graph& graph, Node& current_node, const SliceInfo& info_without_node,
                   const logging::Logger& logger,
                   const std::unordered_map<int, int>& propagate_input_indices,
                   const std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                   const std::unordered_map<int, SliceInfo>& new_gather_infos) override;
};

class TransposeGatherActor : public UpStreamGatherOperatorActorBase {
 public:
  TransposeGatherActor() = default;
  ~TransposeGatherActor() = default;

  bool PreCheck(const Graph& graph, const Node& current_node, const SliceInfo& info,
                const logging::Logger& logger,
                std::unordered_map<int, int>& propagate_input_indices,
                std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                std::function<void(Node& node)>& shape_update_func) override;

  // If scalar slice happens, we need adapt the input, otherwise the perm cannot be matched.
  bool PostProcess(Graph& graph, Node& current_node, const SliceInfo& info_without_node,
                   const logging::Logger& logger,
                   const std::unordered_map<int, int>& propagate_input_indices,
                   const std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                   const std::unordered_map<int, SliceInfo>& new_gather_infos) override;
};

/**
 * Inherit from SimplePointwiseGatherActor<false> to leverage its PreCheck when slicing on batch dims.
 */
class MatMulGatherActor : public SimplePointwiseGatherActor<false> {
 public:
  MatMulGatherActor() = default;
  ~MatMulGatherActor() = default;

  // Check which inputs can be propagated according to the slice axis.
  bool PreCheck(const Graph& graph, const Node& current_node, const SliceInfo& info,
                const logging::Logger& logger,
                std::unordered_map<int, int>& propagate_input_indices,
                std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                std::function<void(Node& node)>& shape_update_func) override;

  // If scalar slice happens in the second last dimension, we need to adapt the input.
  bool PostProcess(Graph& graph, Node& current_node, const SliceInfo& info_without_node,
                   const logging::Logger& logger,
                   const std::unordered_map<int, int>& propagate_input_indices,
                   const std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                   const std::unordered_map<int, SliceInfo>& new_gather_infos) override;
};

/**
 * @brief Update the dim value using given new dim value at specified axis.
 *
 * @param arg_to_update The NodeArg to be updated.
 * @param axis A non-negative axis MUST be given here.
 * @param output_dim_on_axis The new dimension value. If not provided, the dimension will be removed.
 * @return true if the update is done.
 */
bool UpdateSliceOutputShape(NodeArg& arg_to_update, int axis,
                            const ONNX_NAMESPACE::TensorShapeProto_Dimension& new_dim_value);

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
