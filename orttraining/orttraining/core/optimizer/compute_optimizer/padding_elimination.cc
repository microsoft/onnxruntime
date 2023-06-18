// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING

#include <onnx/defs/attr_proto_util.h>

#include "core/framework/random_seed.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "orttraining/core/optimizer/compute_optimizer/padding_elimination.h"

using namespace onnxruntime::optimizer::compute_optimizer;

namespace onnxruntime {

namespace {

void PushAllOutputNode(Graph& graph, std::queue<Node*>& q, Node* node, std::unordered_set<Node*>& visited) {
  for (auto iter = node->OutputNodesBegin(); iter != node->OutputNodesEnd(); ++iter) {
    Node* output_node = graph.GetNode(iter->Index());
    if (visited.find(output_node) == visited.end()) {
      q.push(output_node);
    }
  }
}

bool IsATenEmbedding(const Node* node) {
  if (graph_utils::IsSupportedOptypeVersionAndDomain(*node, "ATen", {1}, kPytorchAtenDomain)) {
    for (auto kv : node->GetAttributes()) {
      if (kv.first == "operator" && kv.second.s() == "embedding") {
        return true;
      }
    }
  }
  return false;
}

// Get dims value of shape of input with indices_arg
// Implemented by add a Shape + GatherElements after input
NodeArg* GetDimsValue(Graph& graph, NodeArg* input, NodeArg* indices_arg, Node& node) {
  InlinedVector<NodeArg*> shape_output_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("shape_result"),
                                                                      nullptr)};
  Node& shape_node = graph.AddNode(graph.GenerateNodeName("shape"), "Shape", "", {input},
                                   shape_output_args, nullptr, kOnnxDomain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(shape_node), "Failed to get shape for " + shape_node.Name());
  shape_node.SetExecutionProviderType(node.GetExecutionProviderType());

  InlinedVector<NodeArg*> gather_input_args;
  gather_input_args.push_back(shape_output_args[0]);
  gather_input_args.push_back(indices_arg);

  InlinedVector<NodeArg*> gather_out_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("gather_result"),
                                                                    nullptr)};

  Node& gather_node = graph.AddNode(graph.GenerateNodeName("gather_first_dim"), "GatherElements", "", gather_input_args,
                                    gather_out_args, nullptr, kOnnxDomain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(gather_node), "Failed to get shape for " + gather_node.Name());
  gather_node.SetExecutionProviderType(node.GetExecutionProviderType());

  return gather_out_args[0];
}

// Insert Shape + ScatterElements to get an updated shape of input with index of indices_arg updated to
// the value of update_value.
// Such as, if the indices_arg is a initializer of [0] and the original shape of input is [valid_token_count, a, b, c],
// this function will return a shape of [update_value, a, b, c]
NodeArg* UpdateShape(Graph& graph, NodeArg* input, NodeArg* update_value, NodeArg* indices_arg, Node& node) {
  InlinedVector<NodeArg*> shape_output_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("shape_result"),
                                                                      nullptr)};
  Node& shape_node = graph.AddNode(graph.GenerateNodeName("shape"), "Shape", "", {input},
                                   shape_output_args, nullptr, kOnnxDomain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(shape_node), "Failed to get shape for " + shape_node.Name());
  shape_node.SetExecutionProviderType(node.GetExecutionProviderType());

  InlinedVector<NodeArg*> scatter_input_args;
  scatter_input_args.push_back(shape_output_args[0]);
  scatter_input_args.push_back(indices_arg);
  scatter_input_args.push_back(update_value);

  InlinedVector<NodeArg*> scatter_out_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("scatter_result"),
                                                                     nullptr)};

  Node& scatter_node = graph.AddNode(graph.GenerateNodeName("update_dim"), "ScatterElements", "", scatter_input_args,
                                     scatter_out_args, nullptr, kOnnxDomain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(scatter_node), "Failed to update shape for " + scatter_node.Name());
  scatter_node.SetExecutionProviderType(node.GetExecutionProviderType());

  return scatter_out_args[0];
}

// Insert Reshape + ShrunkenGather to flatten the in_index-th input of node.
// The gather_index_arg is the indices of the elements that are not padding.
NodeArg* InsertNodesForInput(Graph& graph,
                             Node& node,
                             uint32_t in_index,
                             NodeArg* gather_index_arg,
                             const logging::Logger& logger) {
  InlinedVector<NodeArg*> reshape_input_args;
  reshape_input_args.reserve(2);
  reshape_input_args.push_back(node.MutableInputDefs()[in_index]);
  std::vector<int64_t> new_shape;
  new_shape.push_back(-1);  // only support flatten 0 and 1 dims
  auto input_shape = node.InputDefs()[in_index]->Shape();
  ORT_ENFORCE(input_shape->dim_size() >= 2);
  ONNX_NAMESPACE::TensorShapeProto flattened_shape;
  if (input_shape->dim(0).has_dim_value() && input_shape->dim(1).has_dim_value()) {
    flattened_shape.add_dim()->set_dim_value(input_shape->dim(0).dim_value() * input_shape->dim(1).dim_value());
  } else {
    std::string token_dim_name = MakeString("total_token_count_", utils::GetRandomSeed());
    flattened_shape.add_dim()->set_dim_param(token_dim_name);
  }
  for (int k = 2; k < input_shape->dim_size(); k++) {
    ORT_ENFORCE(input_shape->dim(k).has_dim_value());
    new_shape.push_back(input_shape->dim(k).dim_value());
    flattened_shape.add_dim()->set_dim_value(input_shape->dim(k).dim_value());
  }
  ONNX_NAMESPACE::TensorProto new_shape_const_tensor;
  new_shape_const_tensor.set_name(graph.GenerateNodeArgName("new_shape"));
  new_shape_const_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  new_shape_const_tensor.add_dims(new_shape.size());
  new_shape_const_tensor.set_raw_data(new_shape.data(), new_shape.size() * sizeof(int64_t));
  NodeArg* new_shape_arg = &graph_utils::AddInitializer(graph, new_shape_const_tensor);
  reshape_input_args.push_back(new_shape_arg);

  InlinedVector<NodeArg*> reshape_output_args;
  reshape_output_args.push_back(
      &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("inputs_reshape_result"), node.MutableInputDefs()[in_index]->TypeAsProto()));

  Node* new_reshape_node = InsertIntermediateNodeOnDestInput(
      graph, node,
      in_index,
      0,
      0,
      graph.GenerateNodeName("Reshape"),
      "Reshape",
      "Reshape node to filter invalid tokens.",
      reshape_input_args,
      reshape_output_args,
      {},
      "",
      logger);

  new_reshape_node->SetExecutionProviderType(node.GetExecutionProviderType());
  auto reshape_out_arg = new_reshape_node->MutableOutputDefs()[0];

  reshape_out_arg->SetShape(flattened_shape);

  InlinedVector<NodeArg*> gather_input_args;
  gather_input_args.reserve(2);
  gather_input_args.push_back(reshape_output_args[0]);
  gather_input_args.push_back(gather_index_arg);

  InlinedVector<NodeArg*> gather_output_args;
  gather_output_args.push_back(
      &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("padding_filter_result"),
                                reshape_out_arg->TypeAsProto()));

  Node* new_gather_node = InsertIntermediateNodeOnDestInput(
      graph, node,
      in_index,
      0,
      0,
      graph.GenerateNodeName("PaddingFilter"),
      "ShrunkenGather",
      "ShrunkenGather node to filter invalid tokens.",
      gather_input_args,
      gather_output_args,
      {},
      kMSDomain,
      logger);

  new_gather_node->SetExecutionProviderType(node.GetExecutionProviderType());
  auto gather_out_arg = new_gather_node->MutableOutputDefs()[0];
  return gather_out_arg;
}

// Insert GatherGrad + Reshape to unflatten the shape of the in_index-th input of node.
// The gathergrad_index_arg is the indices of the elements that are not padding.
// The new_shape_arg is the shape of [batch_size * seqlen, ...]
// gathergrad_index_arg and new_shape_arg are the arguments needed by GatherGrad.
NodeArg* InsertNodesForOutput(Graph& graph,
                              Node& node,
                              uint32_t in_index,
                              NodeArg* gathergrad_index_arg,
                              NodeArg* new_shape_arg,
                              NodeArg* first_two_dims_arg,
                              const logging::Logger& logger) {
  std::vector<int64_t> other_indices;
  auto input_shape = node.InputDefs()[in_index]->Shape();
  for (int k = 2; k < input_shape->dim_size(); k++) {
    // When executing, Shape of node here has been flattened, so the indices should be k-1.
    other_indices.push_back(int64_t(k) - 1);
  }

  // Construct the unflattened_shape_arg of [batch_size, seqlen, ...]
  NodeArg* unflattened_shape_arg = nullptr;
  if (other_indices.empty()) {
    unflattened_shape_arg = first_two_dims_arg;
  } else {
    // If the shape size of the in_index-th input of node is larger than 2 dims, we need to concat the first two dims
    // of [batch_size, seqlen] and the other dims together.
    ONNX_NAMESPACE::TensorProto other_indices_const_tensor;
    other_indices_const_tensor.set_name(graph.GenerateNodeArgName("other_shape"));
    other_indices_const_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    other_indices_const_tensor.add_dims(other_indices.size());
    other_indices_const_tensor.set_raw_data(other_indices.data(), other_indices.size() * sizeof(int64_t));
    NodeArg* other_indices_arg = &graph_utils::AddInitializer(graph, other_indices_const_tensor);
    NodeArg* other_dims_arg = GetDimsValue(graph, node.MutableInputDefs()[in_index], other_indices_arg, node);

    InlinedVector<NodeArg*> concat_input_args;
    concat_input_args.push_back(first_two_dims_arg);
    concat_input_args.push_back(other_dims_arg);

    InlinedVector<NodeArg*> concat_output_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("concat_shape_result"),
                                                                         nullptr)};

    onnxruntime::NodeAttributes attributes;
    attributes["axis"] = ONNX_NAMESPACE::MakeAttribute("axis", int64_t(0));

    Node& concat_node = graph.AddNode(graph.GenerateNodeName("concat_shape"), "Concat", "", concat_input_args,
                                      concat_output_args, &attributes, kOnnxDomain);
    ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(concat_node), "Failed to concat shape for " + concat_node.Name());
    concat_node.SetExecutionProviderType(node.GetExecutionProviderType());
    unflattened_shape_arg = concat_output_args[0];
  }

  InlinedVector<NodeArg*> gathergrad_input_args;
  gathergrad_input_args.reserve(3);
  gathergrad_input_args.push_back(new_shape_arg);
  gathergrad_input_args.push_back(gathergrad_index_arg);
  gathergrad_input_args.push_back(node.MutableInputDefs()[in_index]);

  InlinedVector<NodeArg*> gathergrad_output_args;
  gathergrad_output_args.push_back(
      &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("padding_recover_result"),
                                nullptr));

  Node* new_gathergrad_node = InsertIntermediateNodeOnDestInput(
      graph, node,
      in_index,
      2,
      0,
      graph.GenerateNodeName("PaddingRecover"),
      "GatherGrad",
      "GatherGrad node to recover invalid tokens.",
      gathergrad_input_args,
      gathergrad_output_args,
      {},
      kMSDomain,
      logger);

  new_gathergrad_node->SetExecutionProviderType(node.GetExecutionProviderType());
  auto gathergrad_out_arg = new_gathergrad_node->MutableOutputDefs()[0];

  InlinedVector<NodeArg*> reshape_input_args;
  reshape_input_args.push_back(gathergrad_out_arg);
  reshape_input_args.push_back(unflattened_shape_arg);
  InlinedVector<NodeArg*> reshape_output_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("reshape_result"),
                                                                        nullptr)};
  Node* new_reshape_node = InsertIntermediateNodeOnDestInput(
      graph, node,
      in_index,
      0,
      0,
      graph.GenerateNodeName("RecoverShape"),
      "Reshape",
      "Reshape node to recover invalid tokens.",
      reshape_input_args,
      reshape_output_args,
      {},
      kOnnxDomain,
      logger);
  new_reshape_node->SetExecutionProviderType(node.GetExecutionProviderType());
  return new_reshape_node->MutableOutputDefs()[0];
}

// Iterate the subgraph beginning from the start_node, and put all node args into 'subgraph'
// Also put all candidate input nodes and cantidate output nodes of the subgraph into candidate_inputs and
// candidate_outputs respectively.
void IterateSubgraphFromNode(Graph& graph,
                             Node* start_node,
                             std::unordered_set<NodeArg*>& subgraph,
                             std::unordered_set<Node*>& candidate_inputs,
                             std::unordered_set<Node*>& candidate_outputs,
                             const logging::Logger& logger) {
  std::queue<Node*> to_visit;
  std::unordered_set<Node*> visited;
  PushAllOutputNode(graph, to_visit, start_node, visited);
  while (!to_visit.empty()) {
    Node* cur = to_visit.front();
    to_visit.pop();
    visited.insert(cur);
    if (graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "Add", {7, 13, 14}) ||
        graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "BiasGelu", {1}, kMSDomain)) {
      ORT_ENFORCE(subgraph.find(cur->MutableInputDefs()[0]) != subgraph.end() ||
                  subgraph.find(cur->MutableInputDefs()[1]) != subgraph.end());
      NodeArg* arg_in_subgraph = nullptr;
      NodeArg* arg_not_in_subgraph = nullptr;
      if (subgraph.find(cur->MutableInputDefs()[0]) != subgraph.end()) {
        arg_in_subgraph = cur->MutableInputDefs()[0];
        arg_not_in_subgraph = cur->MutableInputDefs()[1];
      } else if (subgraph.find(cur->MutableInputDefs()[1]) != subgraph.end()) {
        arg_in_subgraph = cur->MutableInputDefs()[1];
        arg_not_in_subgraph = cur->MutableInputDefs()[0];
      }

      // arg_in_subgraph is contained in subgraph, so its shape must be [batch_size, seq_len, ...]
      // Now only support cases of the two shapes are absolutely same or the other shape dim size is smaller by 2.
      // For example, [batch_size, seqlen, hidden_size] and [batch_size, seqlen, hidden_size].
      //              [batch_size, seqlen, hidden_size] and [hidden_size].
      // TODO: support other case such as:
      //              [batch_size, seqlen, hidden_size] and [batch_size, 1, hidden_size]
      if (arg_in_subgraph->Shape() && arg_not_in_subgraph->Shape() &&
          (arg_not_in_subgraph->Shape()->dim_size() <= arg_in_subgraph->Shape()->dim_size() - 2 ||
           (arg_in_subgraph->Shape()->dim_size() == arg_not_in_subgraph->Shape()->dim_size() &&
            arg_in_subgraph->Shape()->dim(0) == arg_not_in_subgraph->Shape()->dim(0) &&
            arg_in_subgraph->Shape()->dim(1) == arg_not_in_subgraph->Shape()->dim(1)))) {
        subgraph.insert(cur->MutableOutputDefs()[0]);
        PushAllOutputNode(graph, to_visit, cur, visited);
        // There are two possibilities here:
        // 1. The size of arg_not_in_subgraph->Shape is smaller than arg_in_subgraph->Shape by 2,
        //    do not need to add flatten pattern to arg_not_in_subgraph.
        // 2. The size of arg_not_in_subgraph->Shape is same with arg_in_subgraph->Shape and the first
        //    two dims value are exactly same, then there are also two possibilities:
        //     <1>. The arg_not_in_subgraph is propagated from embedding_node (contained in subgraph),
        //          do not need to process it.
        //     <2>. The arg_not_in_subgraph is not propagated from embedding_node (not contained in subgraph),
        //          need to add flatten pattern to arg_not_in_subgraph.
        //     Here we just add cur node to candidate_inputs and process it (add flatten pattern to its input) after
        //     the graph iteration, according to whether it's contained in subgraph.
        if (arg_in_subgraph->Shape()->dim_size() == arg_not_in_subgraph->Shape()->dim_size()) {
          candidate_inputs.insert(cur);
        }
      } else {
        LOG_DEBUG_INFO(logger, "PaddingElimination::Input shapes of node:" + cur->Name() + "are not compatible.");
        candidate_outputs.insert(cur);
        continue;
      }
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "LayerNormalization", {1, 17}, kOnnxDomain)) {
      if (subgraph.find(cur->MutableInputDefs()[0]) == subgraph.end()) {
        LOG_DEBUG_INFO(logger, "PaddingElimination::First input of Normalization: " + cur->Name() +
                                   " is not in subgraph.");
        candidate_outputs.insert(cur);
        continue;
      }
      auto axis = static_cast<int64_t>(cur->GetAttributes().at("axis").i());
      axis = axis < 0 ? axis + cur->InputDefs()[0]->Shape()->dim_size() : axis;
      if (axis < 2) {
        LOG_DEBUG_INFO(logger, "PaddingElimination::axis of Normalization: " + cur->Name() + " is " +
                                   std::to_string(axis) + ", which blocks merging leading two dims.");
        candidate_outputs.insert(cur);
      } else {
        subgraph.insert(cur->MutableOutputDefs()[0]);
        PushAllOutputNode(graph, to_visit, cur, visited);
      }
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "Dropout", {12, 13})) {
      ORT_ENFORCE(subgraph.find(cur->MutableInputDefs()[0]) != subgraph.end());
      subgraph.insert(cur->MutableOutputDefs()[0]);
      subgraph.insert(cur->MutableOutputDefs()[1]);
      PushAllOutputNode(graph, to_visit, cur, visited);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "Cast", {9, 13})) {
      ORT_ENFORCE(subgraph.find(cur->MutableInputDefs()[0]) != subgraph.end());
      subgraph.insert(cur->MutableOutputDefs()[0]);
      PushAllOutputNode(graph, to_visit, cur, visited);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "MatMul", {1, 9, 13})) {
      if (subgraph.find(cur->MutableInputDefs()[0]) != subgraph.end()) {
        // If shape of [batch_size, seqlen, ...] is propagated from the first argument of MatMul.
        // The dim size of the first argument must larger than 2 to propagete the first two dims to the output.
        // Or else the first two dims of the output will not be [batch_size, seqlen] and this MatMul will be added
        // to candidate_outputs as the output of the subgraph.
        if (cur->InputDefs()[0]->Shape()->dim_size() > 2) {
          subgraph.insert(cur->MutableOutputDefs()[0]);
          PushAllOutputNode(graph, to_visit, cur, visited);
        } else {
          LOG_DEBUG_INFO(logger,
                         "PaddingElimination::dim size of left input of matmul smaller than 3 and \
                            this matmul would be output of subgraph.");
          candidate_outputs.insert(cur);
          continue;
        }
      } else if (subgraph.find(cur->MutableInputDefs()[1]) != subgraph.end()) {
        LOG_DEBUG_INFO(logger, "PaddingElimination::right edge of matmul would not included.");
        candidate_outputs.insert(cur);
        continue;
      } else {
        ORT_THROW("PaddingElimination::found matmul node without input in subgraph.");
      }
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "PythonOp", {1}, kMSDomain)) {
      if (subgraph.find(cur->MutableInputDefs()[0]) == subgraph.end()) {
        candidate_outputs.insert(cur);
        continue;
      }
      auto func_name = static_cast<std::string>(cur->GetAttributes().at("name").s());
      if (func_name == "_InspectActivation" || func_name == "_IncrementStep") {
        subgraph.insert(cur->MutableOutputDefs()[1]);
        PushAllOutputNode(graph, to_visit, cur, visited);
      } else {
        candidate_outputs.insert(cur);
      }
    } else {
      candidate_outputs.insert(cur);
    }
  }
}
}  // namespace

Status PaddingElimination::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  LOG_DEBUG_INFO(logger, "Enter PaddingElimination");

  if (sparse_embedding_input_names_.size() == 0) {
    LOG_DEBUG_INFO(logger, "Exit PaddingElimination, no sparse embedding input names.");
    return Status::OK();
  }

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  Node* embedding_node = nullptr;
  NodeArg* input_ids_arg = nullptr;
  // Make sure each node_arg in subgraph has first two consecutive dims to be flattened.
  // All node_args in subgraph is propagated from the embedding node
  std::unordered_set<NodeArg*> subgraph;
  // input args of nodes in candidate_inputs should be in subgraph or to be added Reshape + Gather
  // record node that its input args may be input of the subgraph into candidate_inputs
  std::unordered_set<Node*> candidate_inputs;
  // input args of nodes in candidate_outputs, if in subgraph, should be added GatherGrad + Reshape
  // record node that its input args may be output of the subgraph into candidate_outputs
  std::unordered_set<Node*> candidate_outputs;
  int64_t handled_input_count = 0;
  int64_t handled_output_count = 0;

  // Find the valid embedding node
  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (IsATenEmbedding(&node) &&
        graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) &&
        node.InputDefs().size() >= 3 &&
        node.InputDefs()[2]->Exists() &&
        graph_utils::IsConstantInitializer(graph, node.InputDefs()[2]->Name()) &&
        node.InputDefs()[1]->Exists() &&
        graph_utils::IsGraphInput(graph, node.InputDefs()[1]) &&
        node.InputDefs()[1]->Shape() &&
        node.InputDefs()[1]->Shape()->dim_size() >= 2) {
      if (std::find(sparse_embedding_input_names_.begin(), sparse_embedding_input_names_.end(),
                    node.InputDefs()[1]->Name()) == sparse_embedding_input_names_.end()) {
        LOG_DEBUG_INFO(logger, "Skip node " + node.Name() + "(" + node.OpType() +
                                   ") due to embedding input is not in the sparse embedding input list.");
        continue;
      }
      const ONNX_NAMESPACE::TensorProto* padding_initializer = graph_utils::GetConstantInitializer(graph, node.InputDefs()[2]->Name());
      if (padding_initializer != nullptr &&
          padding_initializer->dims_size() == 0 &&
          ((padding_initializer->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32) ||
           (padding_initializer->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64))) {
        int64_t padding_idx = *reinterpret_cast<const int64_t*>(padding_initializer->raw_data().data());
        if (padding_idx < 0) {
          continue;
        }
        embedding_node = &node;
        input_ids_arg = embedding_node->MutableInputDefs()[1];
        for (auto output_defs : embedding_node->MutableOutputDefs()) {
          subgraph.insert(output_defs);
        }
        break;
      }
    }
  }

  if (!embedding_node) {
    LOG_DEBUG_INFO(logger, "Exit PaddingElimination optimization for not finding any valid embedding node.");
    return Status::OK();
  }

  IterateSubgraphFromNode(graph, embedding_node, subgraph, candidate_inputs, candidate_outputs, logger);

  // Add Reshape + Sub + NonZero + Squeeze to get the not padding index to be gathered
  InlinedVector<NodeArg*> reshape_input_args;
  reshape_input_args.push_back(input_ids_arg);
  std::vector<int64_t> new_input_ids_shape;
  new_input_ids_shape.push_back(-1);  // Flatten the two leading dims
  auto input_ids_shape = input_ids_arg->Shape();
  for (int k = 2; k < input_ids_shape->dim_size(); k++) {
    ORT_ENFORCE(input_ids_shape->dim(k).has_dim_value());
    new_input_ids_shape.push_back(input_ids_shape->dim(k).dim_value());
  }
  ONNX_NAMESPACE::TensorProto new_shape_const_tensor;
  new_shape_const_tensor.set_name(graph.GenerateNodeArgName("flattened_shape"));
  new_shape_const_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  new_shape_const_tensor.add_dims(new_input_ids_shape.size());
  new_shape_const_tensor.set_raw_data(new_input_ids_shape.data(), new_input_ids_shape.size() * sizeof(int64_t));
  NodeArg* new_input_ids_shape_arg = &graph_utils::AddInitializer(graph, new_shape_const_tensor);
  reshape_input_args.push_back(new_input_ids_shape_arg);

  InlinedVector<NodeArg*> reshape_output_args;
  reshape_output_args.push_back(
      &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("flattened_input_ids"), nullptr));

  Node& reshape_node = graph.AddNode(graph.GenerateNodeName("inputs_reshape"),
                                     "Reshape",
                                     "input flatten first two dims",
                                     reshape_input_args,
                                     reshape_output_args,
                                     nullptr,
                                     kOnnxDomain);
  ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(reshape_node), "Failed to set op schema for " + reshape_node.Name());
  reshape_node.SetExecutionProviderType(embedding_node->GetExecutionProviderType());

  NodeArg* squeeze_out_arg = InsertNodesForValidIndices(
      graph, reshape_output_args[0], embedding_node->MutableInputDefs()[2], embedding_node->GetExecutionProviderType());

  // Add flatten pattern to each input node of the subgraph
  // to flattern the shape of [batch_size, seqlen, ...] to [valid_token_count, ...]
  InsertNodesForInput(graph, *embedding_node, 1, squeeze_out_arg, logger);
  handled_input_count++;
  modified = true;
  for (auto& node : candidate_inputs) {
    for (uint32_t i = 0; i < node->InputDefs().size(); ++i) {
      if (subgraph.find(node->MutableInputDefs()[i]) == subgraph.end()) {
        InsertNodesForInput(graph, *node, i, squeeze_out_arg, logger);
        handled_input_count++;
      }
    }
  }

  std::vector<int64_t> first_indices;
  first_indices.push_back(0);
  ONNX_NAMESPACE::TensorProto first_indice_const_tensor;
  first_indice_const_tensor.set_name(graph.GenerateNodeArgName("indices"));
  first_indice_const_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  first_indice_const_tensor.add_dims(first_indices.size());
  first_indice_const_tensor.set_raw_data(first_indices.data(), first_indices.size() * sizeof(int64_t));
  NodeArg* first_index_arg = &graph_utils::AddInitializer(graph, first_indice_const_tensor);

  // Get the first dim value of flattened input_ids which is batch_size * seq_len
  NodeArg* first_dim = GetDimsValue(graph, reshape_output_args[0], first_index_arg, *embedding_node);

  std::vector<int64_t> first_two_indices{0, 1};
  ONNX_NAMESPACE::TensorProto first_two_indices_const_tensor;
  first_two_indices_const_tensor.set_name(graph.GenerateNodeArgName("first_two_indices"));
  first_two_indices_const_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  first_two_indices_const_tensor.add_dims(first_two_indices.size());
  first_two_indices_const_tensor.set_raw_data(first_two_indices.data(), first_two_indices.size() * sizeof(int64_t));
  NodeArg* first_two_indices_arg = &graph_utils::AddInitializer(graph, first_two_indices_const_tensor);
  // Get the first two dims value of input_ids which is [batch_size, seq_len]
  NodeArg* first_two_dims_arg = GetDimsValue(graph, input_ids_arg, first_two_indices_arg, *embedding_node);

  // Add pattern to each output node of the subgraph
  // to unflatten the shape of [valid_token_count, ...] to [batch_size, seq_len, ...]
  for (const auto& node : candidate_outputs) {
    for (uint32_t i = 0; i < node->InputDefs().size(); ++i) {
      if (subgraph.find(node->MutableInputDefs()[i]) != subgraph.end()) {
        // Get a shape of the i-th input of the node with first index updated to value of first_dim
        // which is batch_size * seq_len. This shape arg will be used as the shape input of GatherGrad
        NodeArg* shape_arg_for_gather_grad = UpdateShape(
            graph, node->MutableInputDefs()[i], first_dim, first_index_arg, *node);
        InsertNodesForOutput(graph, *node, i, squeeze_out_arg, shape_arg_for_gather_grad, first_two_dims_arg, logger);
        handled_output_count++;
      }
    }
  }

  std::string token_dim_name = MakeString("valid_token_count_", utils::GetRandomSeed());
  // Update shape for each edge of the subgraph
  for (auto edge : subgraph) {
    ONNX_NAMESPACE::TensorShapeProto flattened_shape;
    flattened_shape.add_dim()->set_dim_param(token_dim_name);
    auto input_shape = edge->Shape();
    for (int k = 2; k < input_shape->dim_size(); k++) {
      ORT_ENFORCE(input_shape->dim(k).has_dim_value());
      flattened_shape.add_dim()->set_dim_value(input_shape->dim(k).dim_value());
      edge->SetShape(flattened_shape);
    }
  }
  LOGS(logger, INFO) << "PaddingElimination::Total handled input node count:  " << handled_input_count
                     << " output node count: " << handled_output_count;
  return Status::OK();
}

}  // namespace onnxruntime

#endif
