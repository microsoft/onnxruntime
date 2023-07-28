// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING

#include <onnx/defs/attr_proto_util.h>
#include "core/framework/random_seed.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/compute_optimizer/shared_utils.h"

namespace onnxruntime::optimizer::compute_optimizer {

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
                                        const logging::Logger& logger) {
  LOG_DEBUG_INFO(logger, "Inserting " + op_type + " node on " + dest_node.Name() + " 's " +
                             std::to_string(dest_in_index) + "th input " +
                             dest_node.InputDefs()[dest_in_index]->Name() + ", and connect inserted node's " +
                             std::to_string(new_node_output_index) + "th output to " + dest_node.Name() + " 's " +
                             std::to_string(dest_in_index) + "th input.");

  ORT_ENFORCE(dest_in_index < static_cast<int>(dest_node.InputDefs().size()));
  ORT_ENFORCE(new_node_input_index < static_cast<int>(input_args.size()), "new_node_input_index is out of range.");
  ORT_ENFORCE(new_node_output_index < static_cast<int>(output_args.size()), "new_node_output_index is out of range.");
  ORT_ENFORCE(dest_node.MutableInputDefs()[dest_in_index] == input_args[new_node_input_index],
              "input_args[new_node_input_index] is not the same as dest_node.MutableInputDefs()[dest_in_index].",
              dest_node.MutableInputDefs()[dest_in_index]->Name(), " vs ", input_args[new_node_input_index]->Name());

  // Prepare Input and Outputs for the duplicated Gather/GatherND node.
  NodeArg* src_node_arg = dest_node.MutableInputDefs()[dest_in_index];

  // Create the duplicated Gather/GatherND node.
  Node& new_node = graph.AddNode(name, op_type, description, input_args, output_args, &attributes, domain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(new_node), "Failed to set op schema for " + new_node.Name());

  // Connect dest_node's input node to duplicated node.
  // Update new node producer and consumer map.
  for (size_t j = 0; j < new_node.MutableOutputDefs().size(); ++j) {
    graph.UpdateProducerNode(new_node.MutableOutputDefs()[j]->Name(), new_node.Index());
  }

  for (size_t j = 0; j < new_node.MutableInputDefs().size(); ++j) {
    graph.AddConsumerNode(new_node.MutableInputDefs()[j]->Name(), &new_node);
  }

  const Node* src_node = graph.GetProducerNode(src_node_arg->Name());
  if (src_node) {
    int src_out_index = optimizer_utils::IndexOfNodeOutput(*src_node, *src_node_arg);
    graph.AddEdge(src_node->Index(), new_node.Index(), src_out_index, new_node_input_index);
  }

  // Remove edge between dest_node and src_node.
  // Be noted, this will remove dest_node's input edges to src_node
  // (and also the src_node's output edges to dest_node).
  std::vector<graph_utils::GraphEdge> input_edge_to_remove;
  input_edge_to_remove.reserve(1);
  for (auto it = dest_node.InputEdgesBegin(), end = dest_node.InputEdgesEnd(); it != end; ++it) {
    LOG_DEBUG_INFO(logger, "dest_node " + dest_node.Name() + " input edge: " + it->GetNode().Name() +
                               " output index: " + std::to_string(it->GetSrcArgIndex()) + " input index: " +
                               std::to_string(it->GetDstArgIndex()));
    if (it->GetDstArgIndex() == dest_in_index) {
      input_edge_to_remove.push_back(graph_utils::GraphEdge::CreateGraphEdge(dest_node, *it, true));
      break;
    }
  }

  // If the input is graph input or initializer, no edge will be removed.
  if (input_edge_to_remove.size() > 0) {
    graph_utils::GraphEdge::RemoveGraphEdges(graph, input_edge_to_remove);

    // Remove target node from target input arg's consumer list.
    const std::string& src_node_arg_name = src_node_arg->Name();
    int input_use_count_by_dest_node = 0;
    for (size_t i = 0; i < dest_node.InputDefs().size(); ++i) {
      if (dest_node.InputDefs()[i]->Name().compare(src_node_arg_name) == 0) {
        ++input_use_count_by_dest_node;
      }
    }

    if (input_use_count_by_dest_node == 1) {
      graph.RemoveConsumerNode(src_node_arg_name, &dest_node);
    }
  }

  // Connect duplicated gather node to target node's input.
  dest_node.MutableInputDefs()[dest_in_index] = new_node.MutableOutputDefs()[new_node_output_index];
  // Add new edge connecting the duplicated gather with the target node directly.
  // This also updates the destination node's input node args
  graph.AddEdge(new_node.Index(), dest_node.Index(), new_node_output_index, dest_in_index);
  graph.AddConsumerNode(new_node.MutableOutputDefs()[new_node_output_index]->Name(), &dest_node);
  LOG_DEBUG_INFO(logger, "Inserted " + op_type + " node on " + dest_node.Name() + " 's " +
                             std::to_string(dest_in_index) + "th input " +
                             dest_node.InputDefs()[dest_in_index]->Name());
  return &new_node;
}

std::pair<bool, std::vector<DimCompare>> CompareInputShapeWithOutputShape(
    const ONNX_NAMESPACE::TensorShapeProto* full_broadcasted_shape,
    const ONNX_NAMESPACE::TensorShapeProto* target_shape) {
  int full_rank = full_broadcasted_shape->dim_size();
  int target_rank = target_shape->dim_size();

  if (target_rank > full_rank) {
    return std::make_pair<bool, std::vector<DimCompare>>(false, {});
  }

  std::vector<DimCompare> rets(full_rank);
  // For broadcasted shape, we need to compare from the right to left.
  // Be noted: if the dim of target_shape does not exist, we still continue the loop unless we handle
  // all the dims of full_broadcasted_shape.
  for (int i = -1; i >= -full_rank; --i) {
    int idx = full_rank + i;
    if (i < -target_rank) {
      rets[idx] = DimCompare::NotExist;
      continue;
    }

    auto& dim = full_broadcasted_shape->dim(idx);
    auto& target_dim = target_shape->dim(target_rank + i);
    if (dim.has_dim_value() && target_dim.has_dim_value()) {
      if (dim.dim_value() != target_dim.dim_value()) {
        if (target_dim.dim_value() == 1) {
          rets[idx] = DimCompare::BroadCast;
        } else {
          rets[idx] = DimCompare::NotEqual;
        }
      } else {
        rets[idx] = DimCompare::Equal;
      }
    } else if (dim.has_dim_param() && target_dim.has_dim_param()) {
      if (dim.dim_param() != target_dim.dim_param()) {
        rets[idx] = DimCompare::NotEqual;
      } else {
        rets[idx] = DimCompare::Equal;
      }
    } else {
      if (target_dim.has_dim_value() && target_dim.dim_value() == 1) {
        rets[idx] = DimCompare::BroadCast;
      } else {
        rets[idx] = DimCompare::NotEqual;
      }
    }
  }

  return std::make_pair<bool, std::vector<DimCompare>>(true, std::move(rets));
}

int GetONNXOpSetVersion(const Graph& graph) {
  int onnx_opset = -1;
  auto onnx_domain_it = graph.DomainToVersionMap().find(kOnnxDomain);
  if (onnx_domain_it != graph.DomainToVersionMap().end()) {
    onnx_opset = onnx_domain_it->second;
  } else {
    auto onnx_domain_alias_it = graph.DomainToVersionMap().find(kOnnxDomainAlias);
    if (onnx_domain_alias_it != graph.DomainToVersionMap().end())
      onnx_opset = onnx_domain_alias_it->second;
    else
      ORT_THROW("ONNX domain not found in this model");
  }
  return onnx_opset;
}

NodeArg* CreateInitializerFromVector(Graph& graph,
                                     const InlinedVector<int64_t>& dims,
                                     const InlinedVector<int64_t>& values,
                                     const std::string& name) {
  ONNX_NAMESPACE::TensorProto const_tensor;
  const_tensor.set_name(name);
  const_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);

  int64_t total_count = 1;
  for (const int64_t dim : dims) {
    const_tensor.add_dims(dim);
    total_count *= dim;
  }

  ORT_ENFORCE(total_count == static_cast<int64_t>(values.size()),
              "The total count of dims does not match the size of values. ",
              "total_count: ", total_count, " values.size(): ", values.size());

  const_tensor.set_raw_data(values.data(), values.size() * sizeof(int64_t));
  return &graph_utils::AddInitializer(graph, const_tensor);
}

NodeArg* InsertNodesForValidIndices(Graph& graph,
                                    NodeArg* input_to_filter,
                                    NodeArg* invalid_value,
                                    const std::string& execution_provider_type) {
  InlinedVector<NodeArg*> sub_input_args{input_to_filter, invalid_value};

  InlinedVector<NodeArg*> sub_output_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("sub_result"),
                                                                    input_to_filter->TypeAsProto())};

  Node& sub_node = graph.AddNode(graph.GenerateNodeName("sub_invalid_value"), "Sub", "sub invalid value", sub_input_args,
                                 sub_output_args, nullptr, kOnnxDomain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(sub_node), "Failed to set op schema for " + sub_node.Name());
  sub_node.SetExecutionProviderType(execution_provider_type);

  auto non_zero_out_arg = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("filter_valid_result"),
                                                    input_to_filter->TypeAsProto());

  Node& non_zero_node = graph.AddNode(graph.GenerateNodeName("filter_valid_value"), "NonZero",
                                      "filtering valid value",
                                      {sub_node.MutableOutputDefs()[0]},
                                      {non_zero_out_arg}, nullptr, kOnnxDomain);

  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(non_zero_node),
              "Failed to set op schema for " + non_zero_node.Name());

  const std::string dim_name = MakeString("valid_indices_count_", utils::GetRandomSeed());

  // 1D input NonZero generates output of shape (1,dim_name).
  ONNX_NAMESPACE::TensorShapeProto non_zero_output_shape;
  non_zero_output_shape.add_dim()->set_dim_value(1);
  non_zero_output_shape.add_dim()->set_dim_param(dim_name);
  non_zero_out_arg->SetShape(non_zero_output_shape);
  non_zero_node.SetExecutionProviderType(execution_provider_type);

  InlinedVector<NodeArg*> squeeze_input_args;
  squeeze_input_args.push_back(non_zero_out_arg);

  bool opset_lower_than_13 = onnxruntime::optimizer::compute_optimizer::GetONNXOpSetVersion(graph) < 13;
  onnxruntime::NodeAttributes attributes;
  if (opset_lower_than_13) {
    attributes["axes"] = ONNX_NAMESPACE::MakeAttribute("axes", std::vector<int64_t>{0});
  } else {
    squeeze_input_args.push_back(onnxruntime::optimizer::compute_optimizer::CreateInitializerFromVector(
        graph, {1}, {0}, graph.GenerateNodeArgName("axes")));
  }

  auto squeeze_out_arg = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("squeeze_adaptor"),
                                                   non_zero_out_arg->TypeAsProto());
  Node& squeeze_node = graph.AddNode(graph.GenerateNodeName("squeeze_adaptor"), "Squeeze", "nonzero_squeezer",
                                     squeeze_input_args, {squeeze_out_arg}, &attributes, kOnnxDomain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(squeeze_node),
              "Failed to set op schema for " + squeeze_node.Name());

  // After Squeeze, the shape becomes (dim_name).
  ONNX_NAMESPACE::TensorShapeProto squeeze_output_shape;
  squeeze_output_shape.add_dim()->set_dim_param(dim_name);
  squeeze_out_arg->SetShape(squeeze_output_shape);
  squeeze_node.SetExecutionProviderType(execution_provider_type);

  return squeeze_out_arg;
}

}  // namespace onnxruntime::optimizer::compute_optimizer

#endif
