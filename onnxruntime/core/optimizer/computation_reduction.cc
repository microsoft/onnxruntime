// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/defs/attr_proto_util.h>

#include "core/common/safeint.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/computation_reduction.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

static std::pair<bool, bool> CompatibleDimCheck(const TensorShapeProto* full_broadcasted_shape,
                                                const TensorShapeProto* shape_to_compare,
                                                const int minimum_rank_to_handle) {
  bool equal = true;
  bool equal_if_allow_dim_1 = true;
  int output_rank = shape_to_compare->dim_size();
  int input_rank = full_broadcasted_shape->dim_size();
  ORT_ENFORCE(output_rank >= minimum_rank_to_handle && input_rank >= minimum_rank_to_handle,
              "Rank of input and output should be larger than ", minimum_rank_to_handle,
              " input_rank: ", input_rank, " output_rank: ", output_rank);
  int min_rank = std::min(output_rank, input_rank);
  for (int i = static_cast<int>(minimum_rank_to_handle); i <= min_rank; ++i) {
    auto& output_dim = shape_to_compare->dim(output_rank - i);
    auto& input_dim = full_broadcasted_shape->dim(input_rank - i);
    if (output_dim.has_dim_value() && input_dim.has_dim_value()) {
      if (output_dim.dim_value() != input_dim.dim_value()) {
        equal = false;
        if (output_dim.dim_value() != 1) {
          equal_if_allow_dim_1 = false;
        }
      }
    } else if (output_dim.has_dim_param() && input_dim.has_dim_param()) {
      if (output_dim.dim_param() != input_dim.dim_param()) {
        equal = false;
        equal_if_allow_dim_1 = false;
      }
    } else {
      equal = false;
      if (!(output_dim.has_dim_value() && output_dim.dim_value() == 1)) {
        equal_if_allow_dim_1 = false;
      }
    }
  }

  return std::make_pair(equal, equal_if_allow_dim_1);
}

static TensorShapeProto ReplaceSymbolicDimValue(const TensorShapeProto* shape, const int replacement_axis,
                                                const std::string& replacement_dim_value, bool keep_dim) {
  ORT_ENFORCE(replacement_axis >= 0 && replacement_axis < shape->dim_size());
  TensorShapeProto output_shape;
  for (int i = 0; i < shape->dim_size(); ++i) {
    auto& dim = shape->dim(i);
    if (i == replacement_axis) {
      if (keep_dim) {
        output_shape.add_dim()->set_dim_param(replacement_dim_value);
      }
      continue;
    }

    if (dim.has_dim_value()) {
      output_shape.add_dim()->set_dim_value(dim.dim_value());
    } else if (dim.has_dim_param()) {
      output_shape.add_dim()->set_dim_param(dim.dim_param());
    } else {
      ORT_THROW("Invalid dim found in ReplaceSymbolicDimValue");
    }
  }

  return output_shape;
}

static TensorShapeProto ReplaceConcreteDimValue(const TensorShapeProto* shape, const int replacement_axis,
                                                const int& replacement_dim_value, bool keep_dim) {
  ORT_ENFORCE(replacement_axis >= 0 && replacement_axis < shape->dim_size());
  TensorShapeProto output_shape;
  for (int i = 0; i < shape->dim_size(); ++i) {
    auto& dim = shape->dim(i);
    if (i == replacement_axis) {
      if (keep_dim) {
        output_shape.add_dim()->set_dim_value(replacement_dim_value);
      }
      continue;
    }

    if (dim.has_dim_value()) {
      output_shape.add_dim()->set_dim_value(dim.dim_value());
    } else if (dim.has_dim_param()) {
      output_shape.add_dim()->set_dim_param(dim.dim_param());
    } else {
      ORT_THROW("Invalid dim found in ReplaceSymbolicDimValue");
    }
  }

  return output_shape;
}

static TensorShapeProto GetUpdatedShapeProto(int axis,
                                             int ref_gather_data_input_rank,
                                             ONNX_NAMESPACE::TensorShapeProto_Dimension& output_dim_on_axis,
                                             bool is_slice_scalar,
                                             const TensorShapeProto* base_shape_to_update) {
  TensorShapeProto new_output_shape_for_gathernd;

  int negative_axis = axis - ref_gather_data_input_rank;
  int axis_to_update = base_shape_to_update->dim_size() + negative_axis;
  auto orig_gather_output_dim = output_dim_on_axis;
  auto dim_to_update = base_shape_to_update->dim(axis_to_update);

  if (orig_gather_output_dim.has_dim_value()) {
    ORT_ENFORCE(dim_to_update.has_dim_value() && dim_to_update.dim_value() >= orig_gather_output_dim.dim_value(),
                "dim_to_update.dim_value()" + std::to_string(dim_to_update.dim_value()) + "orig_gather_output_dim.dim_value()" + std::to_string(orig_gather_output_dim.dim_value()));
    new_output_shape_for_gathernd =
        ReplaceConcreteDimValue(base_shape_to_update, axis_to_update, orig_gather_output_dim.dim_value(), true);
  } else if (orig_gather_output_dim.has_dim_param()) {
    ORT_ENFORCE(dim_to_update.has_dim_param());
    // BUT how doe we know the new dim is always smaller than orig_gather_output_dim? If not, then this will hurt the perf.
    new_output_shape_for_gathernd =
        ReplaceSymbolicDimValue(base_shape_to_update, axis_to_update, orig_gather_output_dim.dim_param(), true);
  } else {
    ORT_ENFORCE(is_slice_scalar, "Gather output dim is empty only when it is a scalar slice.");
    if (dim_to_update.has_dim_value()) {
      new_output_shape_for_gathernd =
          ReplaceConcreteDimValue(base_shape_to_update, axis_to_update, -1, false);
    } else if (dim_to_update.has_dim_param()) {
      new_output_shape_for_gathernd =
          ReplaceSymbolicDimValue(base_shape_to_update, axis_to_update, "", false);
    } else {
      ORT_THROW("Invalid dim found in GetUpdatedShapeProto");
    }
  }

  return new_output_shape_for_gathernd;
}

Node* InsertNodeBetweenProducerAndConsumer(Graph& graph,
                                           Node* dest_node, int dest_in_index,
                                           const std::string& name,
                                           const std::string& op_type,
                                           const std::string& description,
                                           InlinedVector<NodeArg*>& other_input_args,
                                           const onnxruntime::NodeAttributes& attributes,
                                           const std::string& domain,
                                           const std::string& entry_node_output_arg_name,
                                           const logging::Logger& logger) {
  LOGS(logger, WARNING) << "Inserting " << op_type << " node on " << dest_node->Name()
                        << " 's " << dest_in_index << "th input " << dest_node->InputDefs()[dest_in_index]->Name();
  ORT_ENFORCE(dest_in_index < static_cast<int>(dest_node->InputDefs().size()));

  // Prepare Input and Outputs for the duplicated Gather/GatherND node.
  NodeArg* src_node_arg = dest_node->MutableInputDefs()[dest_in_index];
  int input_count = other_input_args.size();  //1 + other_input_args.size();
  InlinedVector<NodeArg*> new_input_args;
  new_input_args.reserve(input_count);
  // new_input_args.push_back(src_node_arg);
  new_input_args.insert(new_input_args.end(), other_input_args.begin(), other_input_args.end());

  InlinedVector<NodeArg*> new_output_args;
  new_output_args.push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(entry_node_output_arg_name), src_node_arg->TypeAsProto()));

  // Create the duplicated Gather/GatherND node.
  Node& new_gather_node = graph.AddNode(name,
                                        op_type,
                                        description,
                                        new_input_args,
                                        new_output_args,
                                        &attributes,
                                        domain);

  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(new_gather_node),
              "Failed to set op schema for " + new_gather_node.Name());

  // Connect dest_node's input node to dumplicated node.
  // Update new node producer and consumer map.
  for (size_t j = 0; j < new_gather_node.MutableOutputDefs().size(); ++j) {
    graph.UpdateProducerNode(new_gather_node.MutableOutputDefs()[j]->Name(), new_gather_node.Index());
  }
  graph.AddConsumerNode(src_node_arg->Name(), &new_gather_node);
  const Node* src_node = graph.GetProducerNode(src_node_arg->Name());
  if (src_node) {
    int src_out_index = optimizer_utils::IndexOfNodeOutput(*src_node, *src_node_arg);
    graph.AddEdge(src_node->Index(), new_gather_node.Index(), src_out_index, 0);
  }

  // Remove edge between dest_node and src_node.
  // Be noted, this will remove dest_node's input edges to src_node
  // (and also the src_node's output edges to dest_node).
  std::vector<graph_utils::GraphEdge> input_edge_to_remove;
  input_edge_to_remove.reserve(1);
  for (auto it = dest_node->InputEdgesBegin(), end = dest_node->InputEdgesEnd(); it != end; ++it) {
    LOGS(logger, WARNING) << "dest_node " << dest_node->Name() << " input edge: " << it->GetNode().Name()
                          << " output index: " << it->GetSrcArgIndex() << " input index: " << it->GetDstArgIndex();
    if (it->GetDstArgIndex() == dest_in_index) {
      input_edge_to_remove.push_back(graph_utils::GraphEdge::CreateGraphEdge(*dest_node, *it, true));
      break;
    }
  }

  // If the input is graph input or initializer, no edge will be removed.
  if (input_edge_to_remove.size() > 0) {
    graph_utils::GraphEdge::RemoveGraphEdges(graph, input_edge_to_remove);

    // Remove target node from target input arg's consumer list.
    const std::string& src_node_arg_name = src_node_arg->Name();
    int input_use_count_by_dest_node = 0;
    for (size_t i = 0; i < dest_node->InputDefs().size(); ++i) {
      if (dest_node->InputDefs()[i]->Name().compare(src_node_arg_name) == 0) {
        ++input_use_count_by_dest_node;
      }
    }

    if (input_use_count_by_dest_node == 1) {
      graph.RemoveConsumerNode(src_node_arg_name, dest_node);
    }
  }

  // Connect duplicated gather node to target node's input.
  dest_node->MutableInputDefs()[dest_in_index] = new_gather_node.MutableOutputDefs()[0];
  // Add new edge connecting the duplicated gather with the target node directly.
  // This also updates the destination node's input node args
  graph.AddEdge(new_gather_node.Index(), dest_node->Index(), 0, dest_in_index);
  graph.AddConsumerNode(new_gather_node.MutableOutputDefs()[0]->Name(), dest_node);

  LOGS(logger, WARNING) << "Inserted " << op_type << " node on " << dest_node->Name()
                        << " 's " << dest_in_index << "th input " << dest_node->InputDefs()[dest_in_index]->Name();
  return &new_gather_node;
}

void AdaptInputAndOutputForScalarSlice(Graph& graph, Node& target_node, const GatherInfo& info,
                                       const std::unordered_map<int, GatherInfo>& new_gather_infos,
                                       int target_node_output_index,
                                       const logging::Logger& logger) {
  LOGS(logger, WARNING) << "AdaptInputAndOutputForScalarSlice for Node " << target_node.Name() << "(" << target_node.OpType() << ")";

  if (!info.is_slice_scalar) {
    return;
  }

  for (auto pair : new_gather_infos) {
    int input_index = pair.first;
    // TODO: support Unsqueeze < 13
    ONNX_NAMESPACE::TensorProto axes_const_tensor;
    axes_const_tensor.set_name(graph.GenerateNodeArgName("axes"));
    axes_const_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    axes_const_tensor.add_dims(1);
    std::vector<int64_t> values{pair.second.axis};
    axes_const_tensor.set_raw_data(values.data(), sizeof(int64_t));
    NodeArg* axes_arg = &graph_utils::AddInitializer(graph, axes_const_tensor);
    InlinedVector<NodeArg*> new_input_args{target_node.MutableInputDefs()[input_index], axes_arg};
    Node* new_node = InsertNodeBetweenProducerAndConsumer(
        graph,
        &target_node, input_index,
        graph.GenerateNodeName(target_node.OpType() + "_input"),
        "Unsqueeze",
        "Unsqueeze node",
        new_input_args,
        {}, kOnnxDomain,
        graph.GenerateNodeName("input_adapter"),
        logger);

    new_node->SetExecutionProviderType(target_node.GetExecutionProviderType());

    // Set correct shape for Unsquee node
    const TensorShapeProto* unsqueeze_input_shape = new_node->MutableInputDefs()[0]->Shape();
    TensorShapeProto updated_shape;
    int j = 0;
    for (j = 0; j < pair.second.axis; ++j) {
      auto dim = unsqueeze_input_shape->dim(j);
      if (dim.has_dim_value()) {
        updated_shape.add_dim()->set_dim_value(dim.dim_value());
      } else if (dim.has_dim_param()) {
        updated_shape.add_dim()->set_dim_param(dim.dim_param());
      } else {
        ORT_THROW("Invalid dim found in ReplaceSymbolicDimValue");
      }
    }
    updated_shape.add_dim()->set_dim_value(1);
    for (; j < unsqueeze_input_shape->dim_size(); ++j) {
      auto dim = unsqueeze_input_shape->dim(j);
      if (dim.has_dim_value()) {
        updated_shape.add_dim()->set_dim_value(dim.dim_value());
      } else if (dim.has_dim_param()) {
        updated_shape.add_dim()->set_dim_param(dim.dim_param());
      } else {
        ORT_THROW("Invalid dim found in ReplaceSymbolicDimValue");
      }
    }
    new_node->MutableOutputDefs()[0]->SetShape(updated_shape);
  }

  std::vector<const Node*> consumers = graph.GetConsumerNodes(target_node.MutableOutputDefs()[target_node_output_index]->Name());
  ORT_ENFORCE(consumers.size() >= 1, "MatMul should have only one consumer at this point. " + std::to_string(consumers.size()) + " consumers found.");
  Node& consumer = *graph.GetNode(consumers[0]->Index());
  int index = -1;
  for (size_t i = 0; i < consumer.InputDefs().size(); ++i) {
    auto input_arg = consumer.InputDefs()[i];
    if (input_arg->Name().compare(target_node.MutableOutputDefs()[target_node_output_index]->Name()) == 0) {
      index = i;
      break;
    }
  }

  // TODO: support Unsqueeze < 13
  ONNX_NAMESPACE::TensorProto axes_const_tensor;
  axes_const_tensor.set_name(graph.GenerateNodeArgName("axes"));
  axes_const_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  axes_const_tensor.add_dims(1);
  std::vector<int64_t> values{info.axis};
  axes_const_tensor.set_raw_data(values.data(), sizeof(int64_t));
  NodeArg* axes_arg = &graph_utils::AddInitializer(graph, axes_const_tensor);
  InlinedVector<NodeArg*> new_input_args{consumer.MutableInputDefs()[index], axes_arg};
  Node* matmul_out_adaptor_node = InsertNodeBetweenProducerAndConsumer(
      graph,
      &consumer, index,
      graph.GenerateNodeName(target_node.OpType() + "_output"),
      "Squeeze",
      "Squeeze node",
      new_input_args,
      {}, kOnnxDomain,
      graph.GenerateNodeName("output_adapter"),
      logger);

  matmul_out_adaptor_node->SetExecutionProviderType(target_node.GetExecutionProviderType());

  // graph_utils::ReplaceDownstreamNodeInput(graph, target_node, target_node_output_index /*output_idx*/, *matmul_out_adaptor_node, 0 /*replacement_output_idx*/);

  // Don't need set shape for Squeeze because original MatMul output is used as its output type.

  // Set correct shape for MatMul node
  {
    const TensorShapeProto* matmul_out_shape = matmul_out_adaptor_node->MutableOutputDefs()[0]->Shape();
    TensorShapeProto updated_shape;
    int j = 0;
    for (j = 0; j < info.axis; ++j) {
      auto dim = matmul_out_shape->dim(j);
      if (dim.has_dim_value()) {
        updated_shape.add_dim()->set_dim_value(dim.dim_value());
      } else if (dim.has_dim_param()) {
        updated_shape.add_dim()->set_dim_param(dim.dim_param());
      } else {
        ORT_THROW("Invalid dim found in ReplaceSymbolicDimValue");
      }
    }
    updated_shape.add_dim()->set_dim_value(1);
    for (; j < matmul_out_shape->dim_size(); ++j) {
      auto dim = matmul_out_shape->dim(j);
      if (dim.has_dim_value()) {
        updated_shape.add_dim()->set_dim_value(dim.dim_value());
      } else if (dim.has_dim_param()) {
        updated_shape.add_dim()->set_dim_param(dim.dim_param());
      } else {
        ORT_THROW("Invalid dim found in ReplaceSymbolicDimValue");
      }
    }
    target_node.MutableOutputDefs()[0]->SetShape(updated_shape);
  }
}

bool SimplePassThroughPreCheck(const Graph& /*graph*/, const Node& target_node, const GatherInfo& info,
                               std::unordered_map<int, int>& target_node_input_indices,
                               std::vector<int>& input_dices, bool& input_has_dim_1_for_axis,
                               const logging::Logger& logger) {
  auto gathernd_node = info.gather_node;
  int target_node_output_index = optimizer_utils::IndexOfNodeOutput(target_node, *gathernd_node->InputDefs()[0]);
  int minimum_rank_to_handle = info.input_rank - info.axis;
  const NodeArg* gather_data_input_arg = target_node.OutputDefs()[target_node_output_index];
  LOGS(logger, WARNING) << "Enter SimplePassThroughPreCheck for node " << target_node.Name();
  // For each input of target_node, check whether it meets a requirements,
  // 1). either its rank is lower than minimum_rank_to_handle.
  // 2). or the dimension (if exists) before minimum_rank_to_handle is same as target node's output shape.
  auto check_shapes = [&info, minimum_rank_to_handle,
                       gather_data_input_arg](const NodeArg* input_arg_to_compare,
                                              bool& fatal_error_found, bool& dim_1_for_axis_found) -> std::optional<int> {
    dim_1_for_axis_found = false;
    fatal_error_found = false;
    if (input_arg_to_compare->Shape()->dim_size() < minimum_rank_to_handle) {
      // Skip if target node's input rank is less than minimum rank to handle.
      // Encentially this means the input did not affect the Gather axis.
      // todo skip the adaptor for this case.
      return std::nullopt;
    }

    auto ret_pair = CompatibleDimCheck(gather_data_input_arg->Shape(), input_arg_to_compare->Shape(),
                                       minimum_rank_to_handle);
    if (ret_pair.first) {
      return info.axis;
    }

    if (ret_pair.second) {
      dim_1_for_axis_found = true;
      return std::nullopt;
    }

    fatal_error_found = true;
    return std::nullopt;
  };

  target_node_input_indices.clear();
  input_has_dim_1_for_axis = false;
  for (size_t i = 0; i < target_node.InputDefs().size(); ++i) {
    if (input_dices.size() > 0 && std::find(input_dices.begin(), input_dices.end(), i) == input_dices.end()) {
      continue;
    }
    bool fatal_error_found = false;
    bool dim_1_for_axis_found = false;
    auto ret = check_shapes(target_node.InputDefs()[i], fatal_error_found, dim_1_for_axis_found);
    if (fatal_error_found) {
      LOGS(logger, WARNING) << "Skip for node " << target_node.Name() << " due to input check failure at index " << i;
      return false;
    } else if (ret.has_value()) {
      target_node_input_indices[static_cast<int>(i)] = ret.value();
    } else {
      input_has_dim_1_for_axis = input_has_dim_1_for_axis || dim_1_for_axis_found;
      LOGS(logger, WARNING) << "Skip for node " << target_node.Name() << " at index " << i << ", where dim is 1, no need to Gather from.";
    }
  }

  // Make sure once Gather is moved before target node, all its outputs can be correctly be sliced.
  std::unordered_map<int, int> output_dices;
  for (size_t i = 0; i < target_node.OutputDefs().size(); ++i) {
    if (static_cast<int>(i) == target_node_output_index) {
      continue;
    }

    bool fatal_error_found = false;
    bool dim_1_for_axis_found = false;
    auto ret = check_shapes(target_node.OutputDefs()[i], fatal_error_found, dim_1_for_axis_found);
    if (fatal_error_found) {
      LOGS(logger, WARNING) << "Skip for node " << target_node.Name() << " due to output check failure at index " << i;
      return false;
    } else if (ret.has_value()) {
      output_dices[static_cast<int>(i)] = ret.value();
    } else {
      ORT_THROW("Should not have dim 1 in output in Gather axis.");
    }
  }
  bool output_check_success = output_dices.size() == target_node.OutputDefs().size() - 1;

  return output_check_success;
}

bool LayerNormPreCheck(const Graph& graph, const Node& target_node, const GatherInfo& info,
                       std::unordered_map<int, int>& target_node_input_indices,
                       std::vector<int>& input_dices, bool& input_has_dim_1_for_axis,
                       const logging::Logger& logger) {
  auto axis = static_cast<int64_t>(target_node.GetAttributes().at("axis").i());
  axis = axis < 0 ? axis + target_node.InputDefs()[0]->Shape()->dim_size() : axis;

  // Make sure layer norm's reduction happens after the axes we want to slice.
  if (axis <= info.axis) {
    return false;
  }

  return SimplePassThroughPreCheck(graph, target_node, info, target_node_input_indices, input_dices, input_has_dim_1_for_axis, logger);
}

bool ReshapePreCheck(const Graph& graph, const Node& target_node, const GatherInfo& info,
                     std::unordered_map<int, int>& target_node_input_indices,
                     std::vector<int>& /*input_dices*/, bool& /*input_has_dim_1_for_axis*/,
                     const logging::Logger& logger) {
  auto data_input_shape = target_node.InputDefs()[0]->Shape();
  auto shape_input_shape = target_node.InputDefs()[1]->Shape();
  auto output_shape = target_node.OutputDefs()[0]->Shape();
  if (data_input_shape == nullptr || shape_input_shape == nullptr || shape_input_shape->dim_size() != 1 || output_shape == nullptr) {
    LOGS(logger, WARNING) << "Reshape input/output node arg shape is not valid.";
    return false;
  }

  if (!graph_utils::IsConstantInitializer(graph, target_node.InputDefs()[1]->Name())) {
    LOGS(logger, WARNING) << "Skip handle the Reshape, because the new shape is not constant.";
    return false;
  }

  auto in_dims = data_input_shape->dim();
  auto out_dims = output_shape->dim();
  int in_rank = in_dims.size();
  int out_rank = out_dims.size();

  int reshape_input_axis = -1;
  // Match from left to right.
  for (int i = 0; i < std::min(in_rank, out_rank); ++i) {
    bool dim_value_eq = in_dims[i].has_dim_value() && out_dims[i].has_dim_value() && in_dims[i].dim_value() == out_dims[i].dim_value();
    bool dim_param_eq = in_dims[i].has_dim_param() && out_dims[i].has_dim_param() && in_dims[i].dim_param() == out_dims[i].dim_param();
    if (dim_value_eq || dim_param_eq) {
      if (i == info.axis) {
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
      bool dim_value_eq = in_dims[in_index].has_dim_value() && out_dims[out_index].has_dim_value() && in_dims[in_index].dim_value() == out_dims[out_index].dim_value();
      bool dim_param_eq = in_dims[in_index].has_dim_param() && out_dims[out_index].has_dim_param() && in_dims[in_index].dim_param() == out_dims[out_index].dim_param();
      if (dim_value_eq || dim_param_eq) {
        if (out_index == info.axis) {
          reshape_input_axis = in_index;
          break;
        }
        continue;
      }
    }
  }

  if (reshape_input_axis == -1) {
    LOGS(logger, WARNING) << "Cannot find Reshape's input axis for Gather.";
    return false;
  }

  target_node_input_indices[0] = reshape_input_axis;

  return true;
}

bool ReshapePostProcess(Graph& graph, Node& target_node, const GatherInfo& info,
                        const std::unordered_map<int, GatherInfo>& new_gather_infos,
                        int /*target_node_input_index*/, bool /*input_has_dim_1_for_axis*/,
                        const logging::Logger& logger) {
  LOGS(logger, WARNING) << "ReshapePostProcess for Node " << target_node.Name() << "(" << target_node.OpType() << ")";
  InlinedVector<int64_t> new_shape_const_values;
  optimizer_utils::AppendTensorFromInitializer(graph, *target_node.InputDefs()[1], new_shape_const_values, true);

  auto create_new_initializer_from_vector = [&graph](NodeArg* arg_to_be_replaced,
                                                     const InlinedVector<int64_t>& new_values) -> NodeArg* {
    // Create new TensorProto.
    ONNX_NAMESPACE::TensorProto constant_tensor_proto;
    constant_tensor_proto.set_name(graph.GenerateNodeArgName(arg_to_be_replaced->Name()));
    constant_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    auto length = new_values.size();
    constant_tensor_proto.add_dims(length);
    constant_tensor_proto.set_raw_data(new_values.data(), length * sizeof(int64_t));

    // Add initializer into Graph.
    NodeArg* new_shape_arg = &graph_utils::AddInitializer(graph, constant_tensor_proto);
    // Update the output arg shape.
    ONNX_NAMESPACE::TensorShapeProto new_shape;
    new_shape.add_dim()->set_dim_value(length);
    new_shape_arg->SetShape(new_shape);

    return new_shape_arg;
  };

  if (new_shape_const_values[info.axis] == 0) {
    if (info.is_slice_scalar) {
      LOGS(logger, WARNING) << "Removing axis " << info.axis << " from shape tensor.";
      NodeArg* arg_to_be_replaced = target_node.MutableInputDefs()[1];
      InlinedVector<int64_t> new_values;
      for (int i = 0; i < static_cast<int>(new_shape_const_values.size()); ++i) {
        if (i != info.axis) {
          new_values.push_back(new_shape_const_values[i]);
        }
      }
      auto new_shape_arg = create_new_initializer_from_vector(arg_to_be_replaced, new_values);
      graph_utils::ReplaceNodeInput(target_node, 1, *new_shape_arg);
    } else {
      LOGS(logger, WARNING) << "Reshape's shape has 0 specified for aixs: " << info.axis << ", not need update.";
    }
    return true;
  }

  // TODO: add tests for this branch.
  // If it selected shape is dim value, we can update the shape tensor directory.
  if (info.output_dim_on_axis.has_dim_value()) {
    new_shape_const_values[info.axis] = info.output_dim_on_axis.dim_value();
    auto new_shape_arg = create_new_initializer_from_vector(target_node.MutableInputDefs()[1], new_shape_const_values);
    graph_utils::ReplaceNodeInput(target_node, 1, *new_shape_arg);
    return true;
  }

  // TODO: add tests for this branch.
  // If it selected shape is dim param, it requires multiple Slice, one Shape and one Concat to get the updated shape tensor.
  auto slice_shape_func = [&graph](NodeArg* shape, int64_t start, int64_t end) -> Node* {
    InlinedVector<NodeArg*> new_input_args;
    ONNX_NAMESPACE::TensorProto starts_const_tensor;
    starts_const_tensor.set_name(graph.GenerateNodeArgName("starts"));
    starts_const_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    starts_const_tensor.add_dims(1);
    std::vector<int64_t> dim_values{start};
    starts_const_tensor.set_raw_data(dim_values.data(), sizeof(int64_t));
    NodeArg* starts_arg = &graph_utils::AddInitializer(graph, starts_const_tensor);

    ONNX_NAMESPACE::TensorProto ends_const_tensor;
    ends_const_tensor.set_name(graph.GenerateNodeArgName("ends"));
    ends_const_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    ends_const_tensor.add_dims(1);
    std::vector<int64_t> ends_dim_values{end};
    ends_const_tensor.set_raw_data(ends_dim_values.data(), sizeof(int64_t));
    NodeArg* ends_arg = &graph_utils::AddInitializer(graph, ends_const_tensor);

    ONNX_NAMESPACE::TensorProto axes_const_tensor;
    axes_const_tensor.set_name(graph.GenerateNodeArgName("axes"));
    axes_const_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    axes_const_tensor.add_dims(1);
    std::vector<int64_t> axes_dim_values{0};
    axes_const_tensor.set_raw_data(axes_dim_values.data(), sizeof(int64_t));
    NodeArg* axes_arg = &graph_utils::AddInitializer(graph, axes_const_tensor);

    new_input_args.push_back(shape);
    new_input_args.push_back(starts_arg);
    new_input_args.push_back(ends_arg);
    new_input_args.push_back(axes_arg);

    InlinedVector<NodeArg*> new_output_args;
    new_output_args.push_back(&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("slice_out_compute_opt"),
                                                        shape->TypeAsProto()));

    // Create the duplicated Gather/GatherND node.
    Node& first_slice_node = graph.AddNode(graph.GenerateNodeName("slice_to_compute_opt"),
                                           "Slice",
                                           "Slice op for compute optimization",
                                           new_input_args,
                                           new_output_args);

    auto shape_node = graph.GetProducerNode(shape->Name());
    if (shape_node) {
      first_slice_node.SetExecutionProviderType(shape_node->GetExecutionProviderType());
    }
    return &first_slice_node;
  };

  Node* first_slice = nullptr;
  if (info.axis > 0) {
    first_slice = slice_shape_func(target_node.MutableInputDefs()[1], 0, info.axis);
    ONNX_NAMESPACE::TensorShapeProto result_shape;
    result_shape.add_dim()->set_dim_value(info.axis);
    first_slice->MutableOutputDefs()[0]->SetShape(result_shape);
  }

  Node* second_slice = nullptr;
  if (info.axis + 1 <= info.input_rank - 1) {
    second_slice = slice_shape_func(target_node.MutableInputDefs()[1], info.axis + 1, info.input_rank);
    ONNX_NAMESPACE::TensorShapeProto result_shape;
    result_shape.add_dim()->set_dim_value(info.input_rank - info.axis - 1);
    second_slice->MutableOutputDefs()[0]->SetShape(result_shape);
  }

  Node* third_slice = nullptr;
  if (!info.is_slice_scalar) {
    int new_gather_input_rank = new_gather_infos.at(0).input_rank;
    NodeArg* shape_out = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("shape_out"),
                                                   target_node.MutableInputDefs()[1]->TypeAsProto());
    Node& new_dim_retrieve_node = graph.AddNode(graph.GenerateNodeName("shape_retrieve"),
                                                "Shape",
                                                "Shape op for compute optimization",
                                                {target_node.MutableInputDefs()[0]},
                                                {shape_out});
    // new_dim_retrieve_node.SetExecutionProviderType(target_node.MutableInputDefs()[1]->GetExecutionProviderType());
    ONNX_NAMESPACE::TensorShapeProto result_shape;
    result_shape.add_dim()->set_dim_value(new_gather_input_rank);
    new_dim_retrieve_node.MutableOutputDefs()[0]->SetShape(result_shape);
    third_slice = slice_shape_func(shape_out, new_gather_infos.at(0).axis, new_gather_infos.at(0).axis + 1);
  }

  // Concatenate the two slices.
  InlinedVector<NodeArg*> new_input_args;
  if (first_slice) {
    new_input_args.push_back(first_slice->MutableOutputDefs()[0]);
  }

  if (third_slice) {
    new_input_args.push_back(third_slice->MutableOutputDefs()[0]);
  }

  if (second_slice) {
    new_input_args.push_back(second_slice->MutableOutputDefs()[0]);
  }

  int data_input_index = 1;
  InsertNodeBetweenProducerAndConsumer(
      graph,
      &target_node, data_input_index,
      graph.GenerateNodeName("Concat"),
      "Concat",
      "Adapter for Reshape data input",
      new_input_args,
      {}, kOnnxDomain,
      target_node.MutableInputDefs()[1]->Name(),
      logger);

  return true;
}

bool TransposePassThroughPreCheck(const Graph& /*graph*/, const Node& target_node, const GatherInfo& info,
                                  std::unordered_map<int, int>& target_node_input_indices,
                                  std::vector<int>& /*input_dices*/, bool& /*input_has_dim_1_for_axis*/,
                                  const logging::Logger& logger) {
  InlinedVector<int64_t> perm;
  if (!graph_utils::GetRepeatedNodeAttributeValues(target_node, "perm", perm)) {
    LOGS(logger, WARNING) << "perm attribute is not set for node " << target_node.Name();
    return false;
  }

  target_node_input_indices[0] = perm[info.axis];
  return true;
}

bool MatMulPostProcess(Graph& graph, Node& target_node, const GatherInfo& info,
                       const std::unordered_map<int, GatherInfo>& new_gather_infos,
                       int target_node_output_index, bool /*input_has_dim_1_for_axis*/,
                       const logging::Logger& logger) {
  LOGS(logger, WARNING) << "MatMulPostProcess for Node " << target_node.Name() << "(" << target_node.OpType() << ")";

  if (!info.is_slice_scalar) {
    return true;
  }

  AdaptInputAndOutputForScalarSlice(graph, target_node, info, new_gather_infos, target_node_output_index, logger);
  return true;
}

bool TransposePostProcess(Graph& graph, Node& target_node, const GatherInfo& info,
                          const std::unordered_map<int, GatherInfo>& new_gather_infos,
                          int target_node_output_index, bool /*input_has_dim_1_for_axis*/,
                          const logging::Logger& logger) {
  LOGS(logger, WARNING) << "TransposePostProcess for Node " << target_node.Name() << "(" << target_node.OpType() << ")";

  if (!info.is_slice_scalar) {
    return true;
  }
  AdaptInputAndOutputForScalarSlice(graph, target_node, info, new_gather_infos, target_node_output_index, logger);
  return true;
}

bool SimplePassThroughPostProcess(Graph& graph, Node& target_node, const GatherInfo& info,
                                  const std::unordered_map<int, GatherInfo>& new_gather_infos,
                                  int target_node_output_index, bool input_has_dim_1_for_axis,
                                  const logging::Logger& logger) {
  LOGS(logger, WARNING) << "SimplePassThroughPostProcess for Node " << target_node.Name() << "(" << target_node.OpType() << ")";

  if (!info.is_slice_scalar) {
    return true;
  }

  if (input_has_dim_1_for_axis)
    AdaptInputAndOutputForScalarSlice(graph, target_node, info, new_gather_infos, target_node_output_index, logger);
  return true;
}

void ReorderHandle::RegisterOperators() {
  AllowedPassThroughOps.insert({
      // Things to consider when more operators are added here:
      // 1. Whether the operator is safe to pass through in term of compute equivalence.
      //    If optype is not enough to gurantee the equivalence, we need to add a pre-check function (as LayerNormalization
      //    did).
      // 2. Whether the outputs has the same dim changes if Gather node moves before that operator.
      // 3. Should all inputs be allowed when track back further (bottom-up);
      //    if not, add the input index restriction as MatMul did.
      {"Add", OpPassThroughConfig({}, SimplePassThroughPreCheck, SimplePassThroughPostProcess)},
      {"BiasGelu", OpPassThroughConfig({}, SimplePassThroughPreCheck, nullptr)},
      {"BitmaskBiasDropout", OpPassThroughConfig({}, SimplePassThroughPreCheck, nullptr)},
      {"Cast", OpPassThroughConfig({}, SimplePassThroughPreCheck, nullptr)},
      {"Div", OpPassThroughConfig({}, SimplePassThroughPreCheck, nullptr)},
      {"Dropout", OpPassThroughConfig({}, SimplePassThroughPreCheck, nullptr)},
      {"Gelu", OpPassThroughConfig({}, SimplePassThroughPreCheck, nullptr)},
      {"LayerNormalization", OpPassThroughConfig({0}, LayerNormPreCheck, nullptr)},
      {"MatMul", OpPassThroughConfig({0}, SimplePassThroughPreCheck, MatMulPostProcess)},
      {"Reshape", OpPassThroughConfig({0}, ReshapePreCheck, ReshapePostProcess)},
      {"Softmax", OpPassThroughConfig({0}, LayerNormPreCheck, nullptr)},
      {"Transpose", OpPassThroughConfig({}, TransposePassThroughPreCheck, TransposePostProcess)},
  });
}

GatherInfo ReorderHandle::DuplicatedGatherNodeForOneInput(Graph& graph, Node* gather_node,
                                                          Node* target_node,
                                                          int target_node_input_index,
                                                          GatherInfo& info,
                                                          const logging::Logger& logger,
                                                          int new_axis) {
  bool keep_dim = !info.is_slice_scalar;
  LOGS(logger, WARNING) << "DuplicatedGatherNode for Node " << target_node->Name() << "("
                        << target_node->OpType() << ") with input index "
                        << target_node_input_index << ", keep_dim = " << keep_dim;

  // todo: need check whether the slices need to be adapted or not once the new Gather input rank is changed.
  // Gather should be fine for that case. GatherND indices may need to be changed.
  int input_count = gather_node->InputDefs().size();
  InlinedVector<NodeArg*> other_input_args_for_new_gather;
  other_input_args_for_new_gather.reserve(input_count);
  other_input_args_for_new_gather.push_back(target_node->MutableInputDefs()[target_node_input_index]);
  for (int i = 1; i < input_count; ++i) {
    other_input_args_for_new_gather.push_back(gather_node->MutableInputDefs()[i]);
  }

  onnxruntime::NodeAttributes attributes = gather_node->GetAttributes();
  if (info.axis != new_axis) {
    attributes["axis"] = ONNX_NAMESPACE::MakeAttribute("axis", static_cast<int64_t>(new_axis));
  }

  Node* new_gather_node = InsertNodeBetweenProducerAndConsumer(
      graph,
      target_node, target_node_input_index,
      graph.GenerateNodeName(gather_node->Name()),
      gather_node->OpType(),
      "Duplicated Gather node",
      other_input_args_for_new_gather,
      attributes, gather_node->Domain(),
      gather_node->MutableOutputDefs()[0]->Name(),
      logger);

  new_gather_node->SetExecutionProviderType(gather_node->GetExecutionProviderType());

  // Set correct shape for dup node
  ORT_ENFORCE(gather_node->MutableOutputDefs().size() == 1, "GatherND/Gather only support one output.");
  auto new_gather_data_node_arg = new_gather_node->MutableInputDefs()[0];
  auto new_gather_output_node_arg = new_gather_node->MutableOutputDefs()[0];
  TensorShapeProto new_output_shape_for_gathernd = GetUpdatedShapeProto(
      new_axis,
      new_gather_data_node_arg->Shape()->dim_size(),
      info.output_dim_on_axis,
      info.is_slice_scalar,
      new_gather_data_node_arg->Shape());
  new_gather_output_node_arg->SetShape(new_output_shape_for_gathernd);
  return GatherInfo(new_axis, info.is_slice_scalar, new_gather_node);
}

Status ReorderHandle::RemoveGatherNodeAndUpdateTargetNode(Graph& graph, Node& gathernd_node, Node& target_node,
                                                          const logging::Logger& logger, GatherInfo& info) {
  bool keep_dim = !(info.is_slice_scalar);
  LOGS(logger, WARNING) << "RemoveGatherNodeAndUpdateTargetNode target_node " << target_node.Name() << "("
                        << target_node.OpType() << ") gather_node " << gathernd_node.Name() << "("
                        << gathernd_node.OpType() << "), keep_dim = " << keep_dim;

  int output_index = optimizer_utils::IndexOfNodeOutput(target_node, *gathernd_node.MutableInputDefs()[0]);
  auto target_node_out_arg = target_node.MutableOutputDefs()[output_index];
  auto gathernd_out_arg = gathernd_node.MutableOutputDefs()[0];
  const auto& graph_outputs = graph.GetOutputs();
  bool need_update_graph_output = false;
  if (std::find(graph_outputs.begin(), graph_outputs.end(), gathernd_out_arg) != graph_outputs.end()) {
    need_update_graph_output = true;
  }

  // Loop all outputs of target node, update the shape accordingly.
  // 1. For elementwise ops like (LayerNorm/Dropout/Add), we should all outputs.
  // 2. If we want to explicitly ignore updating some outputs, we should define that by op type.
  for (size_t i = 0; i < target_node.MutableOutputDefs().size(); ++i) {
    auto target_node_out_arg = target_node.MutableOutputDefs()[i];
    TensorShapeProto new_output_shape_for_gathernd = GetUpdatedShapeProto(info.axis,
                                                                          info.input_rank,
                                                                          info.output_dim_on_axis,
                                                                          info.is_slice_scalar,
                                                                          target_node_out_arg->Shape());
    target_node_out_arg->SetShape(new_output_shape_for_gathernd);
  }
  LOGS(logger, WARNING) << "RemoveGatherNodeAndUpdateTargetNode Replace all usage of output " << gathernd_out_arg->Name() << ":0"
                        << " with " << target_node_out_arg->Name() << ":" << output_index;

  for (auto it = gathernd_node.OutputEdgesBegin(), end = gathernd_node.OutputEdgesEnd(); it != end; ++it) {
    if (static_cast<size_t>(it->GetSrcArgIndex()) == 0) {
      LOGS(logger, WARNING) << "RemoveGatherNodeAndUpdateTargetNode Gather's output edge " << it->GetNode().Name() << "("
                            << it->GetNode().OpType() << ") input index " << it->GetDstArgIndex();
    }
  }

  graph_utils::ReplaceDownstreamNodeInput(graph, gathernd_node, 0 /*output_idx*/, target_node, output_index /*replacement_output_idx*/);
  auto gather_origin_consumer_nodes = graph.GetConsumerNodes(gathernd_out_arg->Name());
  std::vector<Node*> gathernd_consumer_nodes;
  gathernd_consumer_nodes.reserve(gather_origin_consumer_nodes.size());
  for (auto& consumer_node : gather_origin_consumer_nodes) {
    gathernd_consumer_nodes.push_back(graph.GetNode(consumer_node->Index()));
    LOGS(logger, WARNING) << "RemoveGatherNodeAndUpdateTargetNode Gather's consumer node " << consumer_node->Name() << "("
                          << consumer_node->OpType() << ")";
  }
  graph.UpdateConsumerNodes(target_node.OutputDefs()[output_index]->Name(), gathernd_consumer_nodes);
  // graph.RemoveConsumerNode(target_node.OutputDefs()[output_index]->Name(), &gathernd_node);

  graph.UpdateConsumerNodes(gathernd_out_arg->Name(), {});
  graph.RemoveNode(gathernd_node.Index());

  if (need_update_graph_output) {
    InlinedVector<const NodeArg*> graph_new_outputs;
    graph_new_outputs.reserve(graph_outputs.size());
    for (auto out_arg : graph_outputs) {
      if (out_arg->Name().compare(gathernd_out_arg->Name()) == 0) {
        graph_new_outputs.push_back(target_node_out_arg);
      } else {
        graph_new_outputs.push_back(out_arg);
      }
    }
    graph.SetOutputs(graph_new_outputs);
    graph.SetGraphResolveNeeded();
    graph.SetGraphProtoSyncNeeded();
  }

  return Status::OK();
}

std::optional<GatherInfo> ComputationReductionTransformer::IsSupportedGatherND(Graph& /*graph*/, Node& node,
                                                                               const logging::Logger& logger) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "GatherND", {1, 12, 13}, kOnnxDomain) ||
      !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
    return std::nullopt;
  }

  auto data_shape = node.MutableInputDefs()[0]->Shape();
  auto indices_shape = node.MutableInputDefs()[1]->Shape();
  auto gather_out_shape = node.MutableOutputDefs()[0]->Shape();
  if (data_shape == nullptr || indices_shape == nullptr || gather_out_shape == nullptr) {
    LOGS(logger, WARNING) << "Skip GatherND node " << node.Name() << " due to undefined shape.";
    return std::nullopt;
  }

  const auto data_rank = data_shape->dim_size();
  const auto indices_rank = indices_shape->dim_size();

  // batch_dims is an integer indicating the number of batch dimensions,
  // i.e the leading b number of dimensions of data tensor and indices are representing the batches,
  // and the gather starts from the b+1 dimension.
  auto batch_dims = static_cast<int64_t>(node.GetAttributes().at("batch_dims").i());
  ORT_ENFORCE(batch_dims >= 0 && batch_dims < indices_rank && batch_dims < data_rank,
              "batch_dims must be in the range [0, min(indices_rank, data_rank)):" + std::to_string(batch_dims) +
                  " indices_rank:" + std::to_string(indices_rank) + " data_rank:" + std::to_string(data_rank));

  // Since GatherND is assumed to have batch_dims=1, if the input data's shape is [batch, sequence, ..., ... ],
  // limiting indices_rank=3 will make sure produced output is in shape [batch, sliced_sequence, ..., ...]
  // and the rank did not change.
  // TODO: release the constraint here.
  if (data_rank != 3 || indices_rank != 3 || batch_dims != 1) {
    return std::nullopt;
  }

  auto& indices_last_dim = indices_shape->dim(indices_rank - 1);
  if (!(indices_last_dim.has_dim_value() && indices_last_dim.dim_value() == 1)) {
    return std::nullopt;
  }

  return GatherInfo(batch_dims, false, &node);
}

std::optional<GatherInfo> ComputationReductionTransformer::IsSupportedGather(Graph& /*graph*/, Node& node, const logging::Logger& logger) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gather", {1, 11, 13}, kOnnxDomain) ||
      !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
    return std::nullopt;
  }

  auto data_shape = node.MutableInputDefs()[0]->Shape();
  auto indices_shape = node.MutableInputDefs()[1]->Shape();
  auto gather_out_shape = node.MutableOutputDefs()[0]->Shape();
  if (data_shape == nullptr || indices_shape == nullptr || gather_out_shape == nullptr) {
    LOGS(logger, WARNING) << "Skip Gather node " << node.Name() << " due to undefined shape.";
    return std::nullopt;
  }

  const auto data_rank = data_shape->dim_size();
  if (data_rank != 3) {
    return std::nullopt;
  }

  auto axis = static_cast<int>(node.GetAttributes().at("axis").i());
  axis = axis < 0 ? axis + data_rank : axis;
  size_t dim_size = static_cast<size_t>(indices_shape->dim_size());
  bool is_single_value_1d_tensor = dim_size != 0 && (dim_size == 1 && utils::HasDimValue(indices_shape->dim(0)) &&
                                                     indices_shape->dim(0).dim_value() == 1);
  if (dim_size != 0 && !is_single_value_1d_tensor) {
    return std::nullopt;
  }

  return GatherInfo(axis, dim_size == 0, &node);
}

Status ComputationReductionTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                                  const logging::Logger& logger) const {
  LOGS(logger, WARNING) << "Enter ComputationReductionTransformer";
  bool reordered = false;
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  size_t reordered_node_count = 0;

  for (auto index : order) {
    auto* node_ptr = graph.GetNode(index);
    if (!node_ptr)
      // node was removed.
      continue;

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    std::optional<GatherInfo> gather_info;
    // Same ideas might apply for GatherElements, Slice, Split, etc..
    gather_info = IsSupportedGatherND(graph, node, logger);
    if (!gather_info.has_value()) {
      gather_info = IsSupportedGather(graph, node, logger);
    }

    if (!gather_info.has_value()) {
      continue;
    }

    std::string node_name = node.Name();
    std::string node_type = node.OpType();

    std::deque<GatherInfo> gather_queue;
    gather_queue.push_back(gather_info.value());

    std::string log_prefix = "Entry node " + node_name + " (" + node_type + ") ";
    LOGS(logger, WARNING) << log_prefix << " starts re-ordering check";

    ReorderHandle handle(node_name);

    // DON'T operate on `node` once this loop starts, as it may be removed from the graph.
    int iteration = 0;
    while (!gather_queue.empty()) {
      GatherInfo info = gather_queue.front();
      Node* gather_node = info.gather_node;
      gather_queue.pop_front();
      const Node* gathernd_data_producer = graph.GetProducerNode(gather_node->MutableInputDefs()[0]->Name());
      if (gathernd_data_producer == nullptr) {
        break;
      }
      Node* input_node = const_cast<Node*>(gathernd_data_producer);
      if (graph.GetConsumerNodes(input_node->MutableOutputDefs()[0]->Name()).size() > 1) {
        LOGS(logger, WARNING) << log_prefix << " stops at node " << input_node->Name() << " since multiple consumer found";
        continue;
      }

      auto ret = handle(graph, *gather_node, *input_node, info, logger, gather_queue);
      if (ret) {
        LOGS(logger, WARNING) << log_prefix << " moves up across node " << input_node->Name();
        modified = true;
        reordered = true;
        iteration += 1;
        // if (iteration == 2) {
        //   break;
        // }
      } else {
        LOGS(logger, WARNING) << log_prefix << " stops when handling " << input_node->Name();
      }
    }

    if (reordered) {
      ++reordered_node_count;
    }
  }

  if (modified) {
    graph.SetGraphResolveNeeded();
  }
  LOGS(logger, WARNING) << "Exit ComputationReductionTransformer with summary - reordred_node_count:" << reordered_node_count << " nodes.";
  return Status::OK();
}

}  // namespace onnxruntime
