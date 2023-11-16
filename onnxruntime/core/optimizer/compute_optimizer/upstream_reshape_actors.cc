// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING

#include <onnx/defs/attr_proto_util.h>
#include "core/optimizer/utils.h"
#include "core/optimizer/compute_optimizer/upstream_reshape_actors.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;

namespace onnxruntime::optimizer::compute_optimizer {

/**
 * @brief From the given TensorShape, update the specified dimension with the given value.
 * If no new_dim is provided, the dimension will be removed.
 *
 * @param shape TensorShape used as base shape to modify.
 * @param new_dim The new dimension value.
 * @return TensorShapeProto A copy of "shape" after modification.
 */
TensorShapeProto CreateNewShapeWithMergedTwoLeadingDims(const TensorShapeProto* shape,
                                                        const TensorShapeProto_Dimension& new_dim) {
  ORT_ENFORCE(shape->dim_size() >= 2, "shape should have at least 2 dimensions");
  TensorShapeProto output_shape;
  for (int i = 1; i < shape->dim_size(); ++i) {
    auto& dim = shape->dim(i);
    if (i == 1) {
      if (new_dim.has_dim_value()) {
        output_shape.add_dim()->set_dim_value(new_dim.dim_value());
      } else if (new_dim.has_dim_param()) {
        output_shape.add_dim()->set_dim_param(new_dim.dim_param());
      } else {
        ORT_THROW("Invalid new_dim found in CreateNewShapeWithMergedTwoLeadingDims");
      }
      continue;
    }

    if (dim.has_dim_value()) {
      output_shape.add_dim()->set_dim_value(dim.dim_value());
    } else if (dim.has_dim_param()) {
      output_shape.add_dim()->set_dim_param(dim.dim_param());
    } else {
      ORT_THROW("Invalid dim found in CreateNewShapeWithMergedTwoLeadingDims");
    }
  }

  return output_shape;
}

template <bool AreAllOutputShapesEqual>
bool SimplePointwiseReshapeActor<AreAllOutputShapesEqual>::PreCheck(
    const Node& current_node, const ReshapeInfo& info,
    const logging::Logger& logger,
    std::vector<int>& propagate_input_indices,
    std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
    std::function<void(Node& node)>& shape_update_func) {
  LOG_DEBUG_INFO(logger, "Enter SimplePointwiseReshapeActor::PreCheck for node " + current_node.Name());

  int current_node_output_index = info.GetDataProducerOutputIndex();
  const NodeArg* data_input_arg = current_node.OutputDefs()[current_node_output_index];

  propagate_input_indices.clear();
  all_input_cmp_rets.clear();
  propagate_input_indices.reserve(current_node.InputDefs().size());
  all_input_cmp_rets.reserve(current_node.InputDefs().size());

  // Here we extend all input shapes to have the same length as the output shape (e.g. fully broadcasted shape).
  // For each input, we check
  //   1). besides the first two dims, all other dims are equal, and
  //   2). the first two dims are compatible with the output shape.
  //
  // The dims are compatible as long as it meets one of the following conditions:
  // 1. The first two dims do not exist (e.g. input shape is [1024] and output shape is [2, 16, 1024])
  // 2. The first two dims exist, and are equal (e.g. input shape is [2, 16, 1024] and output shape is [2, 16, 1024])
  // 3. [Not Supported] The first two dims exist, and the first dim is 1 (e.g. input shape is [1, 16, 1024] and output
  //    shape is [2, 16, 1024])
  // 4. [Not Supported] The first two dims exist, and the second dim is 1 (e.g. input shape is [2, 1, 1024] and output
  //    shape is [2, 16, 1024])
  // 5. [Not Supported] The first dim does not exist, and the second dim exist (can be 1 or not).
  for (int input_idx = 0; input_idx < static_cast<int>(current_node.InputDefs().size()); ++input_idx) {
    auto input_shape = current_node.InputDefs()[input_idx]->Shape();

    auto [success, ret] = CompareInputShapeWithOutputShape(data_input_arg->Shape(),
                                                           input_shape);

    if (!success) {
      LOG_DEBUG_INFO(logger, "Fail SimplePointwiseReshapeActor::PreCheck for node " + current_node.Name() +
                                 ": reshape's data input rank < passthrough node's input rank");
      return false;
    }

    if (ret.size() < 2) {
      LOG_DEBUG_INFO(logger, "Fail SimplePointwiseReshapeActor::PreCheck for node " + current_node.Name() +
                                 ": full broadcasted shape has dim < 2.");
      return false;
    }

    all_input_cmp_rets[input_idx] = ret;

    if (ret[0] == DimCompare::NotExist && ret[1] == DimCompare::NotExist) {
      // Don't need to propagate Reshape since the two leading dims to flatten do not exist.
      LOG_DEBUG_INFO(logger, "In SimplePointwiseReshapeActor::PreCheck for node " + current_node.Name() +
                                 ": skip propagating for the input because the merged dims does not exist.");
      continue;
    } else if (ret[0] == DimCompare::Equal && ret[1] == DimCompare::Equal) {
      propagate_input_indices.push_back(input_idx);
    } else {
      LOG_DEBUG_INFO(logger, "Fail SimplePointwiseReshapeActor::PreCheck for node " + current_node.Name() +
                                 ": not supported cases for two leading dims check.");
      return false;
    }

    // All other dims should be equal
    for (int dim_index = 2; dim_index < static_cast<int>(ret.size()); ++dim_index) {
      if (ret[dim_index] != DimCompare::Equal) {
        LOG_DEBUG_INFO(logger, "Fail SimplePointwiseReshapeActor::PreCheck for node " + current_node.Name() +
                                   ": unflatten dims should all be equal to full broadcast shape.");
        return false;
      }
    }
  }

  if (AreAllOutputShapesEqual) {
    // Make sure once Reshape is moved before the target node, all its outputs can be correctly reshaped.
    std::unordered_map<int, int> output_indices;
    for (size_t output_idx = 0; output_idx < current_node.OutputDefs().size(); ++output_idx) {
      if (static_cast<int>(output_idx) == current_node_output_index) {
        continue;
      }

      auto [success, out_cmp_ret] = CompareInputShapeWithOutputShape(data_input_arg->Shape(),
                                                                     current_node.OutputDefs()[output_idx]->Shape());

      if (!success) {
        LOG_DEBUG_INFO(logger, "Fail SimplePointwiseReshapeActor::PreCheck for node " + current_node.Name() +
                                   ": reshape's data input rank < passthrough node's output rank");
        return false;
      }

      // All dims should be equal
      for (int dim_index = 0; dim_index < static_cast<int>(out_cmp_ret.size()); ++dim_index) {
        if (out_cmp_ret[dim_index] != DimCompare::Equal) {
          LOG_DEBUG_INFO(logger, "Fail SimplePointwiseReshapeActor::PreCheck for node " + current_node.Name() +
                                     ": output shapes not equal.");
          return false;
        }
      }
    }
    auto last_dim = info.last_dim;
    shape_update_func = [last_dim](Node& node) -> void {
      for (size_t output_idx = 0; output_idx < node.MutableOutputDefs().size(); ++output_idx) {
        node.MutableOutputDefs()[output_idx]->SetShape(
            CreateNewShapeWithMergedTwoLeadingDims(node.MutableOutputDefs()[output_idx]->Shape(), last_dim));
      }
    };
  } else {
    // For cases AreAllOutputShapesEqual is False, a customized shape update function should be provided.
    ORT_ENFORCE(shape_update_func,
                "AreAllOutputShapesEqual is false, a custom shape update function should be provided.");
  }

  return propagate_input_indices.size() > 0;
}

bool MatMulReshapeActor::PreCheck(
    const Node& current_node, const ReshapeInfo& info,
    const logging::Logger& logger,
    std::vector<int>& propagate_input_indices,
    std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
    std::function<void(Node& node)>& shape_update_func) {
  LOG_DEBUG_INFO(logger, "Enter MatMulReshapeActor::PreCheck for node " + current_node.Name());

  int current_node_output_index = info.GetDataProducerOutputIndex();
  const NodeArg* data_input_arg = current_node.OutputDefs()[current_node_output_index];

  propagate_input_indices.clear();
  all_input_cmp_rets.clear();

  propagate_input_indices.reserve(current_node.InputDefs().size());

  // Check limited input shape combinations, supported cases:
  // 1. The first input have rank = 3, second input has rank = 2.
  //     Example: [2, 16, 1024] * [1024, 1024], having output shape [2, 16, 1024], merging the first two dims.

  auto lhs_input_shape = current_node.InputDefs()[0]->Shape();
  auto rhs_input_shape = current_node.InputDefs()[1]->Shape();

  auto lhs_input_rank = lhs_input_shape->dim_size();
  auto rhs_input_rank = rhs_input_shape->dim_size();
  if (lhs_input_rank != 3 || rhs_input_rank != 2) {
    LOG_DEBUG_INFO(logger, "Fail MatMulReshapeActor::PreCheck for node " + current_node.Name() +
                               ": lhs_input_rank is " + std::to_string(lhs_input_rank) +
                               ", rhs_input_rank is " + std::to_string(rhs_input_rank));
    return false;
  }

  auto [success, ret] = CompareInputShapeWithOutputShape(data_input_arg->Shape(),
                                                         lhs_input_shape);

  if (!success) {
    LOG_DEBUG_INFO(logger, "Fail MatMulReshapeActor::PreCheck for node " + current_node.Name() +
                               ": reshape's data input rank < passthrough node's input rank");
    return false;
  }

  all_input_cmp_rets[0] = ret;

  propagate_input_indices.push_back(0);
  auto last_dim = info.last_dim;
  shape_update_func = [last_dim](Node& node) -> void {
    ORT_ENFORCE(static_cast<size_t>(1) == node.MutableOutputDefs().size());
    node.MutableOutputDefs()[0]->SetShape(
        CreateNewShapeWithMergedTwoLeadingDims(node.MutableOutputDefs()[0]->Shape(), last_dim));
  };

  return propagate_input_indices.size() > 0;
}

bool LayerNormalizationReshapeActor::PreCheck(
    const Node& current_node, const ReshapeInfo& info,
    const logging::Logger& logger,
    std::vector<int>& propagate_input_indices,
    std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
    std::function<void(Node& node)>& shape_update_func) {
  LOG_DEBUG_INFO(logger, "Enter LayerNormalizationReshapeActor::PreCheck for node " + current_node.Name());

  auto axis = static_cast<int64_t>(current_node.GetAttributes().at("axis").i());
  axis = axis < 0 ? axis + current_node.InputDefs()[0]->Shape()->dim_size() : axis;

  // Make sure the layer norm's reduction happens after the axis we want to slice.
  // 1. If axis is either the first or second dim, we can't merge the first two dims.
  //    Example: [2, 16, 1024], axis = 0 or 1, we can't merge the first two dims.
  // 2. If axis is the third dim, we can merge the first two dims.
  //    Example: [2, 16, 1024], axis = 2, we can merge the first two dims, because reduction is still done for each
  //    set of 1024 elements.
  if (axis < 2) {
    LOG_DEBUG_INFO(logger, "Fail LayerNormalizationReshapeActor::PreCheck for node " + current_node.Name() +
                               ": axis is " + std::to_string(axis) + ", which blocks merging leading two dims.");
    return false;
  }

  int current_node_output_index = info.GetDataProducerOutputIndex();
  const NodeArg* data_input_arg = current_node.OutputDefs()[current_node_output_index];

  propagate_input_indices.clear();
  all_input_cmp_rets.clear();

  propagate_input_indices.reserve(current_node.InputDefs().size());
  all_input_cmp_rets.reserve(current_node.InputDefs().size());

  // Only handle the case where the first input is 3D.
  auto data_input_shape = current_node.InputDefs()[0]->Shape();
  auto data_input_rank = data_input_shape->dim_size();
  if (data_input_rank != 3) {
    LOG_DEBUG_INFO(logger, "Fail LayerNormalizationReshapeActor::PreCheck for node " + current_node.Name() +
                               ": data_input_rank is " + std::to_string(data_input_rank));
    return false;
  }

  auto [success, ret] = CompareInputShapeWithOutputShape(data_input_arg->Shape(),
                                                         data_input_shape);

  if (!success) {
    LOG_DEBUG_INFO(logger, "Fail LayerNormalizationReshapeActor::PreCheck for node " + current_node.Name() +
                               ": reshape's data input rank < passthrough node's input rank");
    return false;
  }

  all_input_cmp_rets[0] = ret;

  // Only propagate the first input.
  propagate_input_indices.push_back(0);
  auto last_dim = info.last_dim;
  shape_update_func = [last_dim](Node& node) -> void {
    for (size_t i = 0; i < node.MutableOutputDefs().size(); ++i) {
      node.MutableOutputDefs()[i]->SetShape(
          CreateNewShapeWithMergedTwoLeadingDims(node.MutableOutputDefs()[i]->Shape(), last_dim));
    }
  };

  return propagate_input_indices.size() > 0;
}

bool LayerNormalizationReshapeActor::PostProcess(
    Graph& /* graph */, Node& current_node, const ReshapeInfo& /* info_without_node */,
    const logging::Logger& /* logger */,
    std::vector<int>& /* propagate_input_indices */,
    const std::unordered_map<int, std::vector<DimCompare>>& /* all_input_cmp_rets */,
    const std::unordered_map<int, ReshapeInfo>& /* new_reshape_infos */) {
  auto axis = static_cast<int64_t>(current_node.GetAttributes().at("axis").i());
  // When Reshape(from 3D to 2D, with the first two dimensions be merged) upstream a LayerNormalization,
  // The axis attribute of LayerNormalization should be decreased by 1 if it is greater than 1.
  if (axis > 1) {
    auto new_axis = axis - 1;
    auto& attributes = current_node.GetMutableAttributes();
    attributes["axis"] = ONNX_NAMESPACE::MakeAttribute("axis", static_cast<int64_t>(new_axis));
  }
  return true;
}

template class SimplePointwiseReshapeActor<true>;
template class SimplePointwiseReshapeActor<false>;

}  // namespace onnxruntime::optimizer::compute_optimizer

#endif
