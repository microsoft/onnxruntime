// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The optimization here ideally is applicable to both training and inferencing,
// while so far we mainly validate on training during cooking the optimization.
#ifdef ENABLE_TRAINING_CORE
#pragma once

#include "core/optimizer/compute_optimizer/shared_utils.h"

namespace onnxruntime::optimizer::compute_optimizer {

/**
 * @brief Struct to hold the information of the reshape operations.
 *
 * Initially, an instance of this class for entry node is created, as the Reshape op propagates to entry node's inputs,
 * more instances of this class are created. The propagation stops when the all inputs are not supported to be reshaped.
 */
struct ReshapeInfo : public UpstreamOperatorInfoBase {
  static constexpr int kDataInputIndex = 0;
  static constexpr int kReshapeOutputIndex = 0;

  ReshapeInfo(Node* reshape_node,
              bool is_entry_node_ptr = false)
      : UpstreamOperatorInfoBase(reshape_node) {
    const NodeArg* input = reshape_node->InputDefs()[kDataInputIndex];
    ORT_ENFORCE(input->Shape()->dim_size() == 3, "Only support data of 3D");
    const NodeArg* output = node_ptr->OutputDefs()[kReshapeOutputIndex];
    last_dim = output->Shape()->dim(0);

    if (is_entry_node_ptr) {
      entry_reshape_arg_name = node_ptr->OutputDefs()[kReshapeOutputIndex]->Name();
    }
  }

  int GetDataInputIndex() const {
    return kDataInputIndex;
  }

  int GetOutputIndex() const {
    return kReshapeOutputIndex;
  }

  std::string entry_reshape_arg_name;

  // The dimension of the output tensor after merging the two leading dims.
  ONNX_NAMESPACE::TensorShapeProto_Dimension last_dim;
};

enum class DimCompareRet {
  Equal = 0,
  BroadCast = 1,  // e.g. dim value is 1.
  NotExist = 2,
  NotEqual = 3,
  DimCompareRetMax = 4,
};

/**
 * @brief Base class for all pass through actors.
 *
 * Each actors defines rules to determine whether a node can be passed through, and post
 * process after pass through.
 * PreCheck is the interface to check whether a node can be passed through.
 * The pass through is done transparently, without any interface required to implemented.
 * PostProcess is the interface to do some adaptor work after the pass through.
 */
class UpStreamReshapeOperatorActorBase : public UpStreamOperatorActorBase {
 public:
  UpStreamReshapeOperatorActorBase() = default;
  virtual ~UpStreamReshapeOperatorActorBase() = default;

  /**
   * @brief Check whether a node can be passed through.
   *  At this point, graph modification is not started, once we see any clues that this node
   *  cannot be passed through, we should return false immediately.
   *
   * @param graph The graph that the node belongs to.
   * @param current_node The node to be checked.
   * @param info The reshape info of the Reshape node.
   * @param allowed_input_indices The input indices explicitly specified of the current_node
   * that are allowed to do pass through.
   * @param propagate_input_config: Used as a return value - a map of input index to the input's dim compare result.
   *  The key is an integer, which is the index of the input of the current_node.
   *  The value is a vector of DimCompareRet.
   * @param shape_update_func: Used as a return value - a function to update the shape of the current_node.
   */
  virtual bool PreCheck(const Graph& /*graph*/, const Node& current_node, const ReshapeInfo& info,
                        const std::vector<int>& allowed_input_indices,
                        const logging::Logger& logger,
                        std::unordered_map<int, std::vector<DimCompareRet>>& propagate_input_config,
                        std::function<void(Node& node)>& shape_update_func) = 0;

  /**
   * @brief After reshape op pass through all inputs, do some post process work.
   *
   * Be noted: at this point, reshape op is already removed, so we cannot access ReshapeInfo directly, instead,
   * we pass important infos as parameters of this function.
   *
   * So far, we don't have requirements override PostProcess function.
   *
   */
  bool PostProcess(Graph& /*graph*/, Node& /*current_node*/, int /*current_node_output_index*/,
                   const ONNX_NAMESPACE::TensorShapeProto_Dimension& /*last_dim*/,
                   const std::string& /*entry_node_name*/,
                   const std::unordered_map<int, ReshapeInfo>& /*new_reshape_infos*/,
                   const logging::Logger& /*logger*/) {
    return true;
  }
};

// The inputs are broad-cast-able. The outputs should have same shape (fully broadcasted shape)
// If an operator cannot meet this requirements, we need add specialized actor for it.
template <bool AreAllOutputShapesEqual>
class SimplePointwiseReshapeActor : public UpStreamReshapeOperatorActorBase {
 public:
  SimplePointwiseReshapeActor() = default;
  ~SimplePointwiseReshapeActor() = default;

  bool PreCheck(const Graph& /*graph*/, const Node& current_node, const ReshapeInfo& info,
                const std::vector<int>& allowed_input_indices,
                const logging::Logger& logger,
                std::unordered_map<int, std::vector<DimCompareRet>>& propagate_input_config,
                std::function<void(Node& node)>& shape_update_func) override;
};

class MatMulReshapeActor : public UpStreamReshapeOperatorActorBase {
 public:
  MatMulReshapeActor() = default;
  ~MatMulReshapeActor() = default;

  bool PreCheck(const Graph& /*graph*/, const Node& current_node, const ReshapeInfo& info,
                const std::vector<int>& allowed_input_indices,
                const logging::Logger& logger,
                std::unordered_map<int, std::vector<DimCompareRet>>& propagate_input_config,
                std::function<void(Node& node)>& shape_update_func) override;
};

class LayerNormalizationReshapeActor : public UpStreamReshapeOperatorActorBase {
 public:
  LayerNormalizationReshapeActor() = default;
  ~LayerNormalizationReshapeActor() = default;

  bool PreCheck(const Graph& /*graph*/, const Node& current_node, const ReshapeInfo& info,
                const std::vector<int>& allowed_input_indices,
                const logging::Logger& logger,
                std::unordered_map<int, std::vector<DimCompareRet>>& propagate_input_config,
                std::function<void(Node& node)>& shape_update_func) override;
};

/**
 * @brief From given TensorShape, update specified dimension with given value.
 * If no new_dim is provided, the dimension will be removed.
 *
 * @param shape TensorShape used as base shape to modify.
 * @param new_dim The new dimension value.
 * @return TensorShapeProto A copy of "shape" after modification.
 */
ONNX_NAMESPACE::TensorShapeProto CreateNewShapeWithMergedTwoLeadingDims(
    const ONNX_NAMESPACE::TensorShapeProto* shape,
    const ONNX_NAMESPACE::TensorShapeProto_Dimension& new_dim);

}  // namespace onnxruntime::optimizer::compute_optimizer

#endif
