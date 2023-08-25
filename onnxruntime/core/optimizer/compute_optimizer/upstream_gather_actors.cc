// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING

#include <onnx/defs/attr_proto_util.h>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/compute_optimizer/upstream_gather_actors.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime::optimizer::compute_optimizer {

// Put some utils in anonymous namespace
namespace {

/**
 * @brief From given TensorShape, update specified dimension with given value.
 * If no new_dim is provided, the dimension will be removed.
 *
 * @param shape TensorShape used as base shape to modify.
 * @param axis The dimension to be replaced/removed.
 * @param new_dim The new dimension value. If not provided, the dimension will be removed.
 * @return TensorShapeProto A copy of "shape" after modification.
 */
TensorShapeProto CreateNewShapeWithUpdatedDim(const TensorShapeProto* shape, const int axis,
                                              const TensorShapeProto_Dimension& new_dim) {
  ORT_ENFORCE(axis >= 0 && axis < shape->dim_size());
  TensorShapeProto output_shape;
  for (int i = 0; i < shape->dim_size(); ++i) {
    auto& dim = shape->dim(i);
    if (i == axis) {
      if (new_dim.has_dim_value()) {
        output_shape.add_dim()->set_dim_value(new_dim.dim_value());
      } else if (new_dim.has_dim_param()) {
        output_shape.add_dim()->set_dim_param(new_dim.dim_param());
      } else {
        // do nothing, unassigned dim will be removed.
      }

      continue;
    }

    if (dim.has_dim_value()) {
      output_shape.add_dim()->set_dim_value(dim.dim_value());
    } else if (dim.has_dim_param()) {
      output_shape.add_dim()->set_dim_param(dim.dim_param());
    } else {
      ORT_THROW("Invalid dim found in CreateNewShapeWithUpdatedDim");
    }
  }

  return output_shape;
}

TensorShapeProto CreateTensorShapeInsertDimAtAxis(const TensorShapeProto* src_shape, int axis, int64_t dim_value) {
  ORT_ENFORCE(axis <= src_shape->dim_size(), "axis is out of range.", axis, " vs ", src_shape->dim_size());
  TensorShapeProto updated_shape;
  int j = 0;
  for (j = 0; j < axis; ++j) {
    auto dim = src_shape->dim(j);
    if (dim.has_dim_value()) {
      updated_shape.add_dim()->set_dim_value(dim.dim_value());
    } else if (dim.has_dim_param()) {
      updated_shape.add_dim()->set_dim_param(dim.dim_param());
    } else {
      ORT_THROW("Invalid dim found in CreateTensorShapeInsertDimAtAxis");
    }
  }
  updated_shape.add_dim()->set_dim_value(dim_value);
  for (; j < src_shape->dim_size(); ++j) {
    auto dim = src_shape->dim(j);
    if (dim.has_dim_value()) {
      updated_shape.add_dim()->set_dim_value(dim.dim_value());
    } else if (dim.has_dim_param()) {
      updated_shape.add_dim()->set_dim_param(dim.dim_param());
    } else {
      ORT_THROW("Invalid dim found in CreateTensorShapeInsertDimAtAxis");
    }
  }
  return updated_shape;
}

}  // namespace

bool UpdateSliceOutputShape(NodeArg& arg_to_update, int axis_to_update,
                            const TensorShapeProto_Dimension& output_dim_on_axis) {
  const TensorShapeProto* shape = arg_to_update.Shape();
  int rank = shape->dim_size();
  ORT_ENFORCE(axis_to_update >= 0 && axis_to_update < rank, " axis should be non-negative, representing the index from left to right.");

  TensorShapeProto new_output_shape = CreateNewShapeWithUpdatedDim(shape, axis_to_update, output_dim_on_axis);
  arg_to_update.SetShape(new_output_shape);
  return true;
}

void AdaptInputAndOutputForScalarSlice(Graph& graph, Node& current_node, int current_node_output_index,
                                       int slice_axis, const std::string& entry_node_name,
                                       const std::unordered_map<int, SliceInfo>& new_gather_infos,
                                       const logging::Logger& logger) {
  LOG_DEBUG_INFO(logger, "AdaptInputAndOutputForScalarSlice for Node " + current_node.Name() + "(" +
                             current_node.OpType() + ")");

  // For each handled inputs, insert Unsqueeze node to get the removed dim back at slice_axis.
  for (auto pair : new_gather_infos) {
    int input_index = pair.first;
    Node* new_node = nullptr;
    // Be noted, the Unsqueeze should happens on the axis of new slice node.
    if (GetONNXOpSetVersion(graph) < 13) {
      onnxruntime::NodeAttributes attributes;
      attributes["axes"] = ONNX_NAMESPACE::MakeAttribute("axes", std::vector<int64_t>{pair.second.non_negative_axis});

      new_node =
          InsertIntermediateNodeOnDestInput(
              graph,
              current_node, input_index,
              0 /* new node input index to connect to current_node's input node*/,
              0 /* new node output index to connect to current_node*/,
              graph.GenerateNodeName(entry_node_name + "_adapt_input"),
              "Unsqueeze",
              "Unsqueeze node",
              {current_node.MutableInputDefs()[input_index]},
              {&graph.GetOrCreateNodeArg(
                  graph.GenerateNodeArgName("unsqueeze_adaptor"),
                  current_node.MutableInputDefs()[input_index]->TypeAsProto())},
              attributes, kOnnxDomain,
              logger);
    } else {
      new_node =
          InsertIntermediateNodeOnDestInput(
              graph,
              current_node, input_index,
              0 /* new node input index to connect to current_node's input node*/,
              0 /* new node output index to connect to current_node*/,
              graph.GenerateNodeName(entry_node_name + "_adapt_input"),
              "Unsqueeze",
              "Unsqueeze node",
              {current_node.MutableInputDefs()[input_index],
               CreateInitializerFromVector(graph, {1}, {pair.second.non_negative_axis},
                                           graph.GenerateNodeArgName("axes"))},
              {&graph.GetOrCreateNodeArg(
                  graph.GenerateNodeArgName("unsqueeze_adaptor"),
                  current_node.MutableInputDefs()[input_index]->TypeAsProto())},
              {}, kOnnxDomain,
              logger);
    }
    new_node->SetExecutionProviderType(current_node.GetExecutionProviderType());
    // Set correct shape for Unsqueeze node
    const TensorShapeProto* unsqueeze_input_shape = new_node->MutableInputDefs()[0]->Shape();
    new_node->MutableOutputDefs()[0]->SetShape(
        CreateTensorShapeInsertDimAtAxis(unsqueeze_input_shape, pair.second.non_negative_axis, 1));
  }

  // Find the consumer node of MatMul, and the input index of that node connect to MatMul.
  std::vector<const Node*> consumers =
      graph.GetConsumerNodes(current_node.MutableOutputDefs()[current_node_output_index]->Name());
  ORT_ENFORCE(consumers.size() >= 1, "MatMul should have at least one consumer at this point. " +
                                         std::to_string(consumers.size()) + " consumers found.");
  Node& consumer = *graph.GetNode(consumers[0]->Index());
  int index = -1;
  for (size_t i = 0; i < consumer.InputDefs().size(); ++i) {
    auto input_arg = consumer.InputDefs()[i];
    if (input_arg->Name().compare(current_node.MutableOutputDefs()[current_node_output_index]->Name()) == 0) {
      index = static_cast<int>(i);
      break;
    }
  }

  // Create Squeeze node connecting MatMul output to consumer node.
  Node* matmul_out_adaptor_node = nullptr;
  if (GetONNXOpSetVersion(graph) < 13) {
    onnxruntime::NodeAttributes attributes;
    attributes["axes"] = ONNX_NAMESPACE::MakeAttribute("axes", std::vector<int64_t>{slice_axis});
    matmul_out_adaptor_node =
        InsertIntermediateNodeOnDestInput(
            graph, consumer, index,
            0,
            0 /* new node output index*/,
            graph.GenerateNodeName(current_node.OpType() + "_output"),
            "Squeeze",
            "Squeeze node",
            {consumer.MutableInputDefs()[index]},
            {&graph.GetOrCreateNodeArg(
                graph.GenerateNodeArgName("squeeze_adaptor"),
                consumer.MutableInputDefs()[index]->TypeAsProto())},
            attributes, kOnnxDomain, logger);
  } else {
    matmul_out_adaptor_node =
        InsertIntermediateNodeOnDestInput(
            graph, consumer, index,
            0,
            0 /* new node output index*/,
            graph.GenerateNodeName(current_node.OpType() + "_output"),
            "Squeeze",
            "Squeeze node",
            {consumer.MutableInputDefs()[index],
             CreateInitializerFromVector(graph, {1}, {slice_axis}, graph.GenerateNodeArgName("axes"))},
            {&graph.GetOrCreateNodeArg(
                graph.GenerateNodeArgName("squeeze_adaptor"),
                consumer.MutableInputDefs()[index]->TypeAsProto())},
            {}, kOnnxDomain, logger);
  }

  matmul_out_adaptor_node->SetExecutionProviderType(current_node.GetExecutionProviderType());

  // Don't need set shape for Squeeze because original MatMul output is used as its output type.
  // Set correct shape for MatMul node
  const TensorShapeProto* matmul_out_shape = matmul_out_adaptor_node->MutableOutputDefs()[0]->Shape();
  current_node.MutableOutputDefs()[0]->SetShape(CreateTensorShapeInsertDimAtAxis(matmul_out_shape, slice_axis, 1));
}

template <bool AreAllOutputShapesEqual>
bool SimplePointwiseGatherActor<AreAllOutputShapesEqual>::PreCheck(const Graph& /* graph */,
                                                                   const Node& current_node, const SliceInfo& info,
                                                                   const logging::Logger& logger,
                                                                   std::unordered_map<int, int>& propagate_input_indices,
                                                                   std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                                                                   std::function<void(Node& node)>& shape_update_func) {
  LOG_DEBUG_INFO(logger, "Enter SimplePointwiseGatherActor::PreCheck for node " + current_node.Name());

  const NodeArg* gather_data_input_arg = current_node.OutputDefs()[info.GetDataProducerOutputIndex()];
  const auto& data_input_shape = gather_data_input_arg->Shape();

  propagate_input_indices.clear();
  all_input_cmp_rets.clear();

  /* For each input of current_node, use this function to check if the input can be passed through.
   * If the input has dim on the slicing axis and
   * 1). either the dimension (if exists) including and before the slicing axis is same as the target node's
   *  output shape.
   * 2). or the dimension including and before slicing axis is either 1 , or equal with target dim , or not exist.
   * Otherwise, we will skip the optimization.
   *
   * Example 1: [Can be passed through]
   *    input_0 [M, N, K]    input_1 [K]
   *                \        /
   *                Add [M, N, K] (current_node)
   *                     |
   *            Gather0(axis=1, indices=[1])
   *                     |
   *              output [M, 1, K]
   * In this case, we can propagate Gather to input_0 branch, input_1 is skipped because it did not have dim on
   * slicing axis.
   *
   * Example 2: [Can be passed through]
   *    input_0 [M, N, K]    input_1 [N, K]
   *                \        /
   *                Add [M, N, K] (current_node)
   *                     |
   *            Gather0(axis=1, indices=[1])
   *                     |
   *              output [M, 1, K]
   * In this case, we can propagate Gather to input_0 and input-1 branch, because including and before slicing axis 1,
   * all dims are equal.
   *
   * Example 3: [Can be passed through]
   *    input_0 [M, N, K]    input_1 [1, K]
   *                \        /
   *                Add [M, N, K] (current_node)
   *                     |
   *            Gather0(axis=1, indices=[1])
   *                     |
   *              output [M, 1, K]
   * In this case, we can propagate Gather to input_0 branch, input_1 branch is skipped because it has dim 1 on slicing
   * axis.
   *
   * Example 4: [Can be passed through]
   *    input_0 [M, N, K]    input_1 [1, N, K]
   *                \        /
   *                Add [M, N, K] (current_node)
   *                     |
   *            Gather0(axis=1, indices=[1])
   *                     |
   *              output [M, 1, K]
   * In this case, we can propagate Gather to input_0 and input_1 branches.
   *
   * Example 5: [Can be passed through]
   *    input_0 [M, N, K]    input_1 [M, 1, K]
   *                \        /
   *                Add [M, N, K] (current_node)
   *                     |
   *            Gather0(axis=1, indices=[1])
   *                     |
   *              output [M, 1, K]
   * In this case, we can propagate Gather to input_0 branch, input_1 branch is skipped because it has dim 1 on slicing.
   *
   * Example 6: [CANNOT be passed through]
   *    input_0 [M, N, K]    input_1 [L, N, K]
   *                \        /
   *                Add [M, N, K] (current_node)
   *                     |
   *            Gather0(axis=1, indices=[1])
   *                     |
   *              output [M, 1, K]
   *
   */
  for (size_t input_idx = 0; input_idx < current_node.InputDefs().size(); ++input_idx) {
    auto input_shape = current_node.InputDefs()[input_idx]->Shape();

    auto [success, ret] = CompareInputShapeWithOutputShape(data_input_shape, input_shape);
    if (!success) {
      LOG_DEBUG_INFO(logger, "Fail SimplePointwiseGatherActor::PreCheck for node " + current_node.Name() +
                                 ": gather's data input rank < passthrough node's input rank");
      return false;
    }

    // Make sure the fully broadcasted shape has rank >= slice axis.
    if (ret.size() < static_cast<size_t>(info.non_negative_axis)) {
      LOG_DEBUG_INFO(logger, "Fail SimplePointwiseGatherActor::PreCheck for node " + current_node.Name() +
                                 ": full broadcasted shape has rank < slice axis." + std::to_string(ret.size()) +
                                 " < " + std::to_string(info.non_negative_axis));
      return false;
    }

    all_input_cmp_rets[static_cast<int>(input_idx)] = ret;

    bool ld_dims_exactly_same = true;
    bool ld_dims_broadcasted_equal = true;
    // Check the leading dimensions
    for (size_t dim_idx = 0; dim_idx <= static_cast<size_t>(info.non_negative_axis); ++dim_idx) {
      const auto& lhs_dim_ret = ret[dim_idx];
      if (lhs_dim_ret != DimCompare::Equal) {
        ld_dims_exactly_same = false;
      }

      if (lhs_dim_ret != DimCompare::Equal && lhs_dim_ret != DimCompare::BroadCast &&
          lhs_dim_ret != DimCompare::NotExist) {
        ld_dims_broadcasted_equal = false;
      }
    }

    if (!ld_dims_exactly_same && !ld_dims_broadcasted_equal) {
      LOG_DEBUG_INFO(logger, "Fail SimplePointwiseGatherActor::PreCheck for node " + current_node.Name() +
                                 ": leading dimensions are not exactly same or broadcasted equal.");
      return false;
    }

    if (ret[info.non_negative_axis] == DimCompare::BroadCast ||
        ret[info.non_negative_axis] == DimCompare::NotExist) {
      // Don't need to propagate.
      continue;
    }

    ORT_ENFORCE(ret[info.non_negative_axis] == DimCompare::Equal);
    propagate_input_indices[static_cast<int>(input_idx)] = info.non_negative_axis;
  }

  if (AreAllOutputShapesEqual) {
    // Make sure once Gather is moved before the target node, all its outputs can be correctly sliced.
    std::unordered_map<int, int> output_indices;
    for (size_t output_idx = 1; output_idx < current_node.OutputDefs().size(); ++output_idx) {
      if (static_cast<int>(output_idx) == info.GetDataProducerOutputIndex()) {
        continue;
      }

      auto [success, ret] = CompareInputShapeWithOutputShape(data_input_shape,
                                                             current_node.OutputDefs()[output_idx]->Shape());
      if (!success) {
        LOG_DEBUG_INFO(logger, "Fail SimplePointwiseGatherActor::PreCheck for node " + current_node.Name() +
                                   ": gather's data input rank < passthrough node's output rank");
        return false;
      }

      // Check all dimension match.
      for (size_t dim_idx = 0; dim_idx < ret.size(); ++dim_idx) {
        const auto& lhs_dim_ret = ret[dim_idx];
        if (lhs_dim_ret != DimCompare::Equal) {
          return false;
        }
      }
    }

    shape_update_func = [&info](Node& node) -> void {
      for (size_t output_idx = 0; output_idx < node.MutableOutputDefs().size(); ++output_idx) {
        UpdateSliceOutputShape(*node.MutableOutputDefs()[output_idx], info.non_negative_axis,
                               info.output_dim_on_axis);
      }
    };

  } else {
    // For cases AreAllOutputShapesEqual is False, a customized shape update function should be provided.
    ORT_ENFORCE(shape_update_func,
                "SimplePointwiseGatherActor<false>::PreCheck requires a custom shape update function provided.");
  }

  return propagate_input_indices.size() > 0;
}

template <bool AreAllOutputShapesEqual>
bool SimplePointwiseGatherActor<AreAllOutputShapesEqual>::PostProcess(
    Graph& graph, Node& current_node, const SliceInfo& info_without_node,
    const logging::Logger& logger,
    const std::unordered_map<int, int>& /* propagate_input_indices */,
    const std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
    const std::unordered_map<int, SliceInfo>& new_gather_infos) {
  LOG_DEBUG_INFO(logger, "Enter SimplePointwiseGatherActor::PostProcess for Node " + current_node.Name() +
                             "(" + current_node.OpType() + ")");

  const int slice_axis = info_without_node.non_negative_axis;

  // TODO(pengwa): we should only handle the inputs that have dim value = 1 using Squeeze, instead of
  // handling all inputs/outputs for all new Gather inserted.
  bool found_dim_value_1_in_inputs = false;
  for (const auto& pair : all_input_cmp_rets) {
    if (pair.second[slice_axis] == DimCompare::BroadCast) {
      found_dim_value_1_in_inputs = true;
      break;
    }
  }

  if (info_without_node.is_scalar_slice && found_dim_value_1_in_inputs) {
    AdaptInputAndOutputForScalarSlice(graph, current_node, info_without_node.GetDataProducerOutputIndex(),
                                      slice_axis, info_without_node.entry_node_name, new_gather_infos, logger);
  }

  return true;
}

bool LayerNormalizationGatherActor::PreCheck(const Graph& /* graph */,
                                             const Node& current_node,
                                             const SliceInfo& info,
                                             const logging::Logger& logger,
                                             std::unordered_map<int, int>& propagate_input_indices,
                                             std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                                             std::function<void(Node& node)>& shape_update_func) {
  auto axis = static_cast<int64_t>(current_node.GetAttributes().at("axis").i());
  axis = axis < 0 ? axis + current_node.InputDefs()[0]->Shape()->dim_size() : axis;

  // Make sure LayerNormalization's reduction happens after the axis we want to slice.
  if (axis <= info.non_negative_axis) {
    return false;
  }

  const NodeArg* gather_data_input_arg = current_node.OutputDefs()[info.GetDataProducerOutputIndex()];
  const auto& gather_data_input_shape = gather_data_input_arg->Shape();

  auto [success, ret] = CompareInputShapeWithOutputShape(gather_data_input_shape,
                                                         current_node.InputDefs()[0]->Shape());
  if (!success) {
    // This should not happen!!!
    LOG_DEBUG_INFO(logger, "Fail LayerNormalizationGatherActor::PreCheck for node " + current_node.Name() +
                               ": reshape's data input rank < passthrough node's input rank");
    return false;
  }

  // Only propagate the first input.
  propagate_input_indices[0] = info.non_negative_axis;
  all_input_cmp_rets[0] = std::move(ret);

  shape_update_func = [&info](Node& node) -> void {
    // Be noted: If LayerNorm's data input is [dim1, dim2, dim3], reduce axis is 1,
    // then its 2nd and 3rd outputs have shape [dim1, dim2, 1]. The dim is still kept even
    // for reduced axes, so the slicing axis is same with the 1st output.
    for (size_t output_idx = 0; output_idx < node.MutableOutputDefs().size(); ++output_idx) {
      UpdateSliceOutputShape(*node.MutableOutputDefs()[output_idx], info.non_negative_axis,
                             info.output_dim_on_axis);
    }
  };

  return true;
}

bool LayerNormalizationGatherActor::PostProcess(Graph& /*graph*/, Node& current_node,
                                                const SliceInfo& info_without_node,
                                                const logging::Logger& /*logger*/,
                                                const std::unordered_map<int, int>& /*propagate_input_indices*/,
                                                const std::unordered_map<int, std::vector<DimCompare>>&
                                                /*all_input_cmp_rets*/,
                                                const std::unordered_map<int, SliceInfo>& /*new_gather_infos*/) {
  // Update LayerNormalization's axis attribute if it is scalar slice.
  if (info_without_node.is_scalar_slice) {
    auto axis = static_cast<int64_t>(current_node.GetAttributes().at("axis").i());
    auto original_ln_input_rank = info_without_node.input_rank;
    axis = axis < 0 ? axis + original_ln_input_rank : axis;
    auto new_axis = axis - 1;

    auto& attributes = current_node.GetMutableAttributes();
    attributes["axis"] = ONNX_NAMESPACE::MakeAttribute("axis", static_cast<int64_t>(new_axis));
  }

  return true;
}

bool SoftmaxGatherActor::PreCheck(const Graph& graph, const Node& current_node, const SliceInfo& info,
                                  const logging::Logger& logger,
                                  std::unordered_map<int, int>& propagate_input_indices,
                                  std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                                  std::function<void(Node& node)>& shape_update_func) {
  auto axis = static_cast<int64_t>(current_node.GetAttributes().at("axis").i());
  axis = axis < 0 ? axis + current_node.InputDefs()[0]->Shape()->dim_size() : axis;

  // Make sure Softmax's reduction happens after the axis we want to slice.
  if (axis <= info.non_negative_axis) {
    return false;
  }

  return SimplePointwiseGatherActor<true>::PreCheck(graph, current_node, info, logger,
                                                    propagate_input_indices, all_input_cmp_rets, shape_update_func);
}

bool SoftmaxGatherActor::PostProcess(Graph& graph, Node& current_node, const SliceInfo& info_without_node,
                                     const logging::Logger& logger,
                                     const std::unordered_map<int, int>& propagate_input_indices,
                                     const std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                                     const std::unordered_map<int, SliceInfo>& new_gather_infos) {
  SimplePointwiseGatherActor<true>::PostProcess(graph, current_node, info_without_node, logger,
                                                propagate_input_indices, all_input_cmp_rets, new_gather_infos);

  // Update Softmax's axis attribute if it is scalar slice.
  if (info_without_node.is_scalar_slice) {
    auto axis = static_cast<int64_t>(current_node.GetAttributes().at("axis").i());
    auto original_ln_input_rank = info_without_node.input_rank;
    axis = axis < 0 ? axis + original_ln_input_rank : axis;
    auto new_axis = axis - 1;

    auto& attributes = current_node.GetMutableAttributes();
    attributes["axis"] = ONNX_NAMESPACE::MakeAttribute("axis", static_cast<int64_t>(new_axis));
  }

  return true;
}

bool ReshapeGatherActor::PreCheck(const Graph& graph, const Node& current_node, const SliceInfo& info,
                                  const logging::Logger& logger,
                                  std::unordered_map<int, int>& propagate_input_indices,
                                  std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                                  std::function<void(Node& node)>& shape_update_func) {
  auto data_input_shape = current_node.InputDefs()[0]->Shape();
  auto shape_input_shape = current_node.InputDefs()[1]->Shape();
  auto output_shape = current_node.OutputDefs()[0]->Shape();
  if (data_input_shape == nullptr || shape_input_shape == nullptr || shape_input_shape->dim_size() != 1 ||
      output_shape == nullptr) {
    LOG_DEBUG_INFO(logger, "Reshape input/output node arg shape is not valid.");
    return false;
  }

  if (!graph_utils::IsConstantInitializer(graph, current_node.InputDefs()[1]->Name())) {
    LOG_DEBUG_INFO(logger, "Skip handle the Reshape, because the new shape is not constant.");
    return false;
  }

  propagate_input_indices.clear();
  all_input_cmp_rets.clear();

  InlinedVector<int64_t> new_shape_const_values;
  optimizer_utils::AppendTensorFromInitializer(graph, *current_node.InputDefs()[1], new_shape_const_values, true);
  // Only below two cases are supported for easier updating shape data after propagate slice ops.
  // 1). If the shape data on slicing axis is zero (e.g. remain the same after slicing), we support it.
  // 2). Or if the sliced dim value is a constant, we also support it, and can update the shape data directly.
  // For other cases, it is feasible to support but we don't support for now.
  if (new_shape_const_values[info.non_negative_axis] == 0 || info.output_dim_on_axis.has_dim_value()) {
    auto in_dims = data_input_shape->dim();
    auto out_dims = output_shape->dim();
    int in_rank = in_dims.size();
    int out_rank = out_dims.size();

    int reshape_input_axis = -1;
    // Match from left to right.
    for (int i = 0; i < std::min(in_rank, out_rank); ++i) {
      bool dim_value_eq = in_dims[i].has_dim_value() && out_dims[i].has_dim_value() &&
                          in_dims[i].dim_value() == out_dims[i].dim_value();
      bool dim_param_eq = in_dims[i].has_dim_param() && out_dims[i].has_dim_param() &&
                          in_dims[i].dim_param() == out_dims[i].dim_param();
      if (dim_value_eq || dim_param_eq) {
        if (i == info.non_negative_axis) {
          reshape_input_axis = i;
          break;
        }
        continue;
      }
    }

    if (reshape_input_axis == -1) {
      // Match from right to left.
      for (int i = 0; i < std::min(in_rank, out_rank); ++i) {
        int in_index = in_rank - 1 - i;
        int out_index = out_rank - 1 - i;
        bool dim_value_eq = in_dims[in_index].has_dim_value() && out_dims[out_index].has_dim_value() &&
                            in_dims[in_index].dim_value() == out_dims[out_index].dim_value();
        bool dim_param_eq = in_dims[in_index].has_dim_param() && out_dims[out_index].has_dim_param() &&
                            in_dims[in_index].dim_param() == out_dims[out_index].dim_param();
        if (dim_value_eq || dim_param_eq) {
          if (out_index == info.non_negative_axis) {
            reshape_input_axis = in_index;
            break;
          }
          continue;
        }
      }
    }

    if (reshape_input_axis == -1) {
      LOG_DEBUG_INFO(logger, "Cannot find Reshape's input axis for Gather.");
      return false;
    }

    propagate_input_indices[0] = reshape_input_axis;
    all_input_cmp_rets[0] = {};

    shape_update_func = [&info](Node& node) -> void {
      for (size_t output_idx = 0; output_idx < node.MutableOutputDefs().size(); ++output_idx) {
        UpdateSliceOutputShape(*node.MutableOutputDefs()[output_idx], info.non_negative_axis,
                               info.output_dim_on_axis);
      }
    };

    return true;
  }

  LOG_DEBUG_INFO(logger, "Skip handle the Reshape, new_shape_const_values[info.non_negative_axis]:" +
                             std::to_string(new_shape_const_values[info.non_negative_axis]) +
                             ", info.output_dim_on_axis.has_dim_value(): " +
                             std::to_string(info.output_dim_on_axis.has_dim_value()) + ".");

  return false;
}

bool ReshapeGatherActor::PostProcess(
    Graph& graph, Node& current_node, const SliceInfo& info_without_node,
    const logging::Logger& logger,
    const std::unordered_map<int, int>& /* propagate_input_indices */,
    const std::unordered_map<int, std::vector<DimCompare>>& /* all_input_cmp_rets */,
    const std::unordered_map<int, SliceInfo>& /* new_gather_infos */
) {
  LOG_DEBUG_INFO(logger, "ReshapeGatherActor::PostProcess for Node " + current_node.Name() +
                             "(" + current_node.OpType() + ")");
  InlinedVector<int64_t> new_shape_const_values;
  optimizer_utils::AppendTensorFromInitializer(graph, *current_node.InputDefs()[1], new_shape_const_values, true);
  const int slice_axis = info_without_node.non_negative_axis;

  // If the shape constant on slice_axis is 0, then it keeps the original dim of input.
  // If it is a scalar slice, then we just remove that dim. Otherwise, we don't need to update the dim value.
  if (new_shape_const_values[slice_axis] == 0) {
    if (info_without_node.is_scalar_slice) {
      LOG_DEBUG_INFO(logger, "Removing axis " + std::to_string(slice_axis) + " from shape tensor.");
      NodeArg* arg_to_be_replaced = current_node.MutableInputDefs()[1];
      InlinedVector<int64_t> new_values;
      for (int i = 0; i < static_cast<int>(new_shape_const_values.size()); ++i) {
        if (i != slice_axis) {
          new_values.push_back(new_shape_const_values[i]);
        }
      }
      auto new_shape_arg = CreateInitializerFromVector(graph, {static_cast<int64_t>(new_values.size())}, new_values,
                                                       graph.GenerateNodeArgName(arg_to_be_replaced->Name()));
      graph_utils::ReplaceNodeInput(current_node, 1, *new_shape_arg);
    } else {
      LOG_DEBUG_INFO(logger, "Reshape's shape has 0 specified for axis: " + std::to_string(slice_axis) +
                                 ", not need an update.");
    }
    return true;
  }

  // If the selected shape is a dim value, we can update the shape tensor directory.
  if (info_without_node.output_dim_on_axis.has_dim_value()) {
    new_shape_const_values[slice_axis] = info_without_node.output_dim_on_axis.dim_value();
    auto new_shape_arg =
        CreateInitializerFromVector(graph, {static_cast<int64_t>(new_shape_const_values.size())},
                                    new_shape_const_values,
                                    graph.GenerateNodeArgName(current_node.MutableInputDefs()[1]->Name()));
    graph_utils::ReplaceNodeInput(current_node, 1, *new_shape_arg);
    return true;
  }

  ORT_THROW("Fail to update shape data in ReshapeGatherActor::PostProcess, but this should not be called.");
}

bool TransposeGatherActor::PreCheck(const Graph& /* graph */, const Node& current_node, const SliceInfo& info,
                                    const logging::Logger& logger,
                                    std::unordered_map<int, int>& propagate_input_indices,
                                    std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                                    std::function<void(Node& node)>& shape_update_func) {
  InlinedVector<int64_t> perm;
  if (!graph_utils::GetRepeatedNodeAttributeValues(current_node, "perm", perm)) {
    LOG_DEBUG_INFO(logger, "perm attribute is not set for node " + current_node.Name());
    return false;
  }
  propagate_input_indices.clear();
  all_input_cmp_rets.clear();

  propagate_input_indices[0] = static_cast<int>(perm[info.non_negative_axis]);
  all_input_cmp_rets[0] = {};

  shape_update_func = [&info](Node& node) -> void {
    for (size_t output_idx = 0; output_idx < node.MutableOutputDefs().size(); ++output_idx) {
      UpdateSliceOutputShape(*node.MutableOutputDefs()[output_idx], info.non_negative_axis,
                             info.output_dim_on_axis);
    }
  };

  return true;
}

bool TransposeGatherActor::PostProcess(
    Graph& graph, Node& current_node, const SliceInfo& info_without_node,
    const logging::Logger& logger,
    const std::unordered_map<int, int>& /* propagate_input_indices */,
    const std::unordered_map<int, std::vector<DimCompare>>& /* all_input_cmp_rets */,
    const std::unordered_map<int, SliceInfo>& new_gather_infos) {
  LOG_DEBUG_INFO(logger, "Enter TransposeGatherActor::PostProcess for Node " + current_node.Name() + "(" +
                             current_node.OpType() + ")");

  // We need to keep the original dimension to align with an original perm.
  if (info_without_node.is_scalar_slice) {
    AdaptInputAndOutputForScalarSlice(graph, current_node, info_without_node.GetDataProducerOutputIndex(),
                                      info_without_node.non_negative_axis,
                                      info_without_node.entry_node_name, new_gather_infos, logger);
  }
  return true;
}

bool MatMulGatherActor::PreCheck(const Graph& graph, const Node& current_node, const SliceInfo& info,
                                 const logging::Logger& logger,
                                 std::unordered_map<int, int>& propagate_input_indices,
                                 std::unordered_map<int, std::vector<DimCompare>>& all_input_cmp_rets,
                                 std::function<void(Node& node)>& shape_update_func) {
  LOG_DEBUG_INFO(logger, "Enter MatMulGatherActor::PreCheck for node " + current_node.Name());
  auto lhs_rank = current_node.InputDefs()[0]->Shape()->dim_size();
  auto rhs_rank = current_node.InputDefs()[1]->Shape()->dim_size();

  if (!(lhs_rank >= 2 && rhs_rank >= 2)) {
    LOG_DEBUG_INFO(logger, "MatMul input rank lower than 2, skip.");
    return false;
  }

  shape_update_func = [&info](Node& node) -> void {
    for (size_t output_idx = 0; output_idx < node.MutableOutputDefs().size(); ++output_idx) {
      UpdateSliceOutputShape(*node.MutableOutputDefs()[output_idx], info.non_negative_axis,
                             info.output_dim_on_axis);
    }
  };

  propagate_input_indices.clear();
  all_input_cmp_rets.clear();

  if (info.non_negative_axis == info.input_rank - 1) {
    propagate_input_indices[1] = rhs_rank - 1;
    all_input_cmp_rets[1] = {};
    return true;
  } else if (info.non_negative_axis == info.input_rank - 2) {
    propagate_input_indices[0] = lhs_rank - 2;
    all_input_cmp_rets[0] = {};
    return true;
  }

  // Gather on batch dimensions, the logic is very similar to SimplePointwiseGatherActor's PreCheck.
  return SimplePointwiseGatherActor<false>::PreCheck(graph, current_node, info, logger,
                                                     propagate_input_indices, all_input_cmp_rets,
                                                     shape_update_func);
}

bool MatMulGatherActor::PostProcess(
    Graph& graph, Node& current_node, const SliceInfo& info_without_node,
    const logging::Logger& logger,
    const std::unordered_map<int, int>& /* propagate_input_indices */,
    const std::unordered_map<int, std::vector<DimCompare>>& /* all_input_cmp_rets */,
    const std::unordered_map<int, SliceInfo>& new_gather_infos) {
  LOG_DEBUG_INFO(logger, "Enter MatMulGatherActor::PostProcess for Node " + current_node.Name() + "(" +
                             current_node.OpType() + ")");

  // We need to keep the original dimension to avoid the matmul inputs cannot be compatible to compute.
  if (info_without_node.is_scalar_slice) {
    AdaptInputAndOutputForScalarSlice(graph, current_node, info_without_node.GetDataProducerOutputIndex(),
                                      info_without_node.non_negative_axis,
                                      info_without_node.entry_node_name, new_gather_infos, logger);
  }
  return true;
}

template class SimplePointwiseGatherActor<true>;
template class SimplePointwiseGatherActor<false>;

}  // namespace onnxruntime::optimizer::compute_optimizer

#endif
