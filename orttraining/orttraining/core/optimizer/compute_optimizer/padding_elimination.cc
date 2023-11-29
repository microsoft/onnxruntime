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

// TODO(pengwa): remove this once customized PythonOp shape inference is supported.
constexpr const char* kInspectActivationFuncName = "onnxruntime.training.utils.hooks._subscriber_manager._InspectActivation";
constexpr const char* kIncrementStepFuncName = "onnxruntime.training.utils.hooks._subscriber_manager._IncrementStep";

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

// Insert Expand to the in_index-th input of node.
// The node should have two inputs and the shape of the other input (node.InputDefs()[1-in_index]) should be
// [batch_size, seq_len, ...]. This function insert an Expand to expand shape of the in_index-th input of node with
// a shape arg of [batch_size, seq_len, 1, 1, ...] which size is equal with node.InputDefs()[1-in_index]->Shape().size.
NodeArg* InsertExpandForNodeInput(Graph& graph,
                                  Node& node,
                                  uint32_t in_index,
                                  NodeArg* first_two_dims_arg,
                                  const logging::Logger& logger) {
  auto full_sized_input_shape = node.InputDefs()[1 - in_index]->Shape();
  ORT_ENFORCE(full_sized_input_shape->dim_size() >= 2);
  NodeArg* expand_shape_arg = nullptr;
  if (full_sized_input_shape->dim_size() == 2) {
    expand_shape_arg = first_two_dims_arg;
  } else {
    InlinedVector<int64_t> other_indices(static_cast<int64_t>(full_sized_input_shape->dim_size()) - 2, 1);
    InlinedVector<NodeArg*> concat_input_args;
    concat_input_args.push_back(first_two_dims_arg);
    concat_input_args.push_back(
        CreateInitializerFromVector(graph,
                                    {static_cast<int64_t>(other_indices.size())},
                                    other_indices,
                                    graph.GenerateNodeArgName("other_shape")));

    InlinedVector<NodeArg*> concat_output_args{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("concat_shape_result"),
                                                                         nullptr)};

    onnxruntime::NodeAttributes attributes;
    attributes["axis"] = ONNX_NAMESPACE::MakeAttribute("axis", int64_t(0));

    Node& concat_node = graph.AddNode(graph.GenerateNodeName("concat_shape"), "Concat", "", concat_input_args,
                                      concat_output_args, &attributes, kOnnxDomain);
    ORT_ENFORCE(graph.SetOpSchemaFromRegistryForNode(concat_node), "Failed to concat shape for " + concat_node.Name());
    concat_node.SetExecutionProviderType(node.GetExecutionProviderType());
    expand_shape_arg = concat_output_args[0];
  }

  InlinedVector<NodeArg*> expand_input_args;
  expand_input_args.reserve(2);
  expand_input_args.push_back(node.MutableInputDefs()[in_index]);
  expand_input_args.push_back(expand_shape_arg);

  InlinedVector<NodeArg*> expand_output_args;
  expand_output_args.push_back(
      &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("inputs_expand_result"),
                                node.MutableInputDefs()[1 - in_index]->TypeAsProto()));

  Node* new_expand_node = InsertIntermediateNodeOnDestInput(
      graph, node,
      in_index,
      0,
      0,
      graph.GenerateNodeName("ExpandPaddingShape"),
      "Expand",
      "Expand shape of one input arg to align the other arg.",
      expand_input_args,
      expand_output_args,
      {},
      "",
      logger);
  new_expand_node->SetExecutionProviderType(node.GetExecutionProviderType());
  return new_expand_node->MutableOutputDefs()[0];
}

// Insert FlattenAndUnpad to flatten and unpad the in_index-th input of node.
// The gather_index_arg is the indices of the elements that are not padding.
NodeArg* InsertFlattenPatternForInput(Graph& graph,
                                      Node& node,
                                      uint32_t in_index,
                                      NodeArg* gather_index_arg,
                                      const logging::Logger& logger) {
  InlinedVector<NodeArg*> unpad_input_args;
  unpad_input_args.reserve(2);
  unpad_input_args.push_back(node.MutableInputDefs()[in_index]);
  unpad_input_args.push_back(gather_index_arg);

  InlinedVector<NodeArg*> unpad_output_args;
  unpad_output_args.push_back(
      &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("padding_filter_result"),
                                nullptr));
  unpad_output_args.push_back(
      &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("d1_d2_shape"),
                                nullptr));

  Node* unpad_node = InsertIntermediateNodeOnDestInput(
      graph, node,
      in_index,
      0,
      0,
      graph.GenerateNodeName("PaddingFilter"),
      "FlattenAndUnpad",
      "FlattenAndUnpad node to filter invalid tokens.",
      unpad_input_args,
      unpad_output_args,
      {},
      kMSDomain,
      logger);

  unpad_node->SetExecutionProviderType(node.GetExecutionProviderType());
  auto unpad_out_arg = unpad_node->MutableOutputDefs()[0];
  return unpad_out_arg;
}

// Insert PadAndUnflatten to unflatten the shape of the in_index-th input of node.
// The gathergrad_index_arg is the indices of the elements that are not padding.
// The new_shape_arg is the shape of [batch_size * seqlen, ...]
// gathergrad_index_arg and new_shape_arg are the arguments needed by GatherGrad.
NodeArg* InsertNodesForOutput(Graph& graph,
                              Node& node,
                              uint32_t in_index,
                              NodeArg* gathergrad_index_arg,
                              NodeArg* first_two_dims_arg,
                              const logging::Logger& logger) {
  InlinedVector<NodeArg*> pad_node_input_args;
  pad_node_input_args.reserve(3);
  pad_node_input_args.push_back(node.MutableInputDefs()[in_index]);
  pad_node_input_args.push_back(gathergrad_index_arg);
  pad_node_input_args.push_back(first_two_dims_arg);

  InlinedVector<NodeArg*> pad_node_output_args;
  pad_node_output_args.push_back(
      &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("padded_result"),
                                nullptr));
  Node* new_gathergrad_node = InsertIntermediateNodeOnDestInput(
      graph, node,
      in_index,
      0 /* new_node_input_index*/,
      0 /* new_node_output_index*/,
      graph.GenerateNodeName("PaddingRecover"),
      "PadAndUnflatten",
      "PadAndUnflatten node to recover invalid tokens.",
      pad_node_input_args,
      pad_node_output_args,
      {},
      kMSDomain,
      logger);

  new_gathergrad_node->SetExecutionProviderType(node.GetExecutionProviderType());
  return new_gathergrad_node->MutableOutputDefs()[0];
}

// Iterate the subgraph beginning from the start_node, and put all node args into 'subgraph'
// Also put all candidate input nodes and candidate output nodes of the subgraph into candidate_inputs and
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
        graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "BiasGelu", {1}, kMSDomain) ||
        graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "Sub", {7, 13, 14}) ||
        graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "Mul", {7, 13, 14})) {
      ORT_ENFORCE(subgraph.find(cur->MutableInputDefs()[0]) != subgraph.end() ||
                  subgraph.find(cur->MutableInputDefs()[1]) != subgraph.end());
      if (cur->InputDefs()[0]->Shape() && cur->InputDefs()[1]->Shape()) {
        if ((subgraph.find(cur->MutableInputDefs()[0]) == subgraph.end() &&
             cur->InputDefs()[0]->Shape()->dim_size() > cur->InputDefs()[1]->Shape()->dim_size()) ||
            (subgraph.find(cur->MutableInputDefs()[1]) == subgraph.end() &&
             cur->InputDefs()[1]->Shape()->dim_size() > cur->InputDefs()[0]->Shape()->dim_size())) {
          // If the shape of one of the inputs is not in the subgraph, and it has more dimensions,
          // this case is not supported now.
          LOG_DEBUG_INFO(logger, "PaddingElimination::Input shapes of node:" + cur->Name() + " are not compatible." +
                                     " arg not in subgraph has more dimensions.");
          candidate_outputs.insert(cur);
          continue;
        }
        subgraph.insert(cur->MutableOutputDefs()[0]);
        PushAllOutputNode(graph, to_visit, cur, visited);
        candidate_inputs.insert(cur);
      } else {
        LOG_DEBUG_INFO(logger, "PaddingElimination::Input of node:" + cur->Name() + " have no shape.");
        candidate_outputs.insert(cur);
        continue;
      }
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "LayerNormalization", {1, 17}, kOnnxDomain) ||
               graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "SimplifiedLayerNormalization", {1}, kOnnxDomain)) {
      if (subgraph.find(cur->MutableInputDefs()[0]) == subgraph.end()) {
        LOG_DEBUG_INFO(logger, "PaddingElimination::First input of Normalization: " + cur->Name() +
                                   " is not in subgraph.");
        candidate_outputs.insert(cur);
        continue;
      }
      if (!cur->InputDefs()[0]->Shape()) {
        LOG_DEBUG_INFO(logger, "PaddingElimination::First input of Normalization: " + cur->Name() +
                                   " has no shape.");
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
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "Cast", {9, 13}) ||
               graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "Gelu", {1}, kMSDomain)) {
      ORT_ENFORCE(subgraph.find(cur->MutableInputDefs()[0]) != subgraph.end());
      subgraph.insert(cur->MutableOutputDefs()[0]);
      PushAllOutputNode(graph, to_visit, cur, visited);
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "MatMul", {1, 9, 13})) {
      if (subgraph.find(cur->MutableInputDefs()[0]) != subgraph.end()) {
        // If shape of [batch_size, seqlen, ...] is propagated from the first argument of MatMul.
        // The dim size of the first argument must be larger than 2 to propagate the first two dims to the output.
        // Or else the first two dims of the output will not be [batch_size, seqlen] and this MatMul will be added
        // to candidate_outputs as the output of the subgraph.
        if (cur->InputDefs()[0]->Shape()->dim_size() > 2) {
          subgraph.insert(cur->MutableOutputDefs()[0]);
          PushAllOutputNode(graph, to_visit, cur, visited);
        } else {
          LOG_DEBUG_INFO(logger,
                         "PaddingElimination::dim size of left input of MatMul smaller than 3 and \
                            this MatMul would be the output of the subgraph.");
          candidate_outputs.insert(cur);
          continue;
        }
      } else if (subgraph.find(cur->MutableInputDefs()[1]) != subgraph.end()) {
        LOG_DEBUG_INFO(logger, "PaddingElimination::right edge of MatMul would not included.");
        candidate_outputs.insert(cur);
        continue;
      } else {
        ORT_THROW("PaddingElimination::found MatMul node without input in subgraph.");
      }
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "PythonOp", {1}, kMSDomain)) {
      if (subgraph.find(cur->MutableInputDefs()[0]) == subgraph.end()) {
        candidate_outputs.insert(cur);
        continue;
      }
      auto func_name = static_cast<std::string>(cur->GetAttributes().at("name").s());
      if (func_name == kInspectActivationFuncName || func_name == kIncrementStepFuncName) {
        subgraph.insert(cur->MutableOutputDefs()[1]);
        PushAllOutputNode(graph, to_visit, cur, visited);
      } else {
        candidate_outputs.insert(cur);
      }
    } else if (graph_utils::IsSupportedOptypeVersionAndDomain(*cur, "ReduceMean", {1, 11, 13, 18})) {
      if (cur->InputDefs()[0]->Shape()) {
        auto axes = cur->GetAttributes().at("axes").ints();
        bool axes_check = (axes.size() > 0);
        for (int64_t axis : axes) {
          axis = axis < 0 ? axis + cur->InputDefs()[0]->Shape()->dim_size() : axis;
          if (axis < 2) {
            LOG_DEBUG_INFO(logger, "PaddingElimination::axis of ReduceMean: " + cur->Name() + " is " +
                                       std::to_string(axis) + ", which blocks merging leading two dims.");
            axes_check = false;
            break;
          }
        }
        if (axes_check) {
          LOG_DEBUG_INFO(logger, "PaddingElimination::ReduceMean: " + cur->Name() + " is added to subgraph.");
          subgraph.insert(cur->MutableOutputDefs()[0]);
          PushAllOutputNode(graph, to_visit, cur, visited);
        } else {
          candidate_outputs.insert(cur);
        }
      } else {
        LOG_DEBUG_INFO(logger, "PaddingElimination::shape of input of ReduceMean: " + cur->Name() + " is unknown.");
        candidate_outputs.insert(cur);
        continue;
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
  // input args of nodes in candidate_inputs should be in subgraph or to be added Reshape + Gather.
  // Put nodes that its input args may be input of the subgraph into candidate_inputs
  std::unordered_set<Node*> candidate_inputs;
  // input args of nodes in candidate_outputs, if in subgraph, should be added GatherGrad + Reshape
  // record node that its input args may be output of the subgraph into candidate_outputs
  std::unordered_set<Node*> candidate_outputs;
  int64_t handled_input_count = 0;
  int64_t handled_output_count = 0;
  int64_t expanded_input_count = 0;

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
      const ONNX_NAMESPACE::TensorProto* padding_initializer =
          graph_utils::GetConstantInitializer(graph, node.InputDefs()[2]->Name());
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

  if (!input_ids_arg->Shape()) {
    LOG_DEBUG_INFO(logger, "Exit PaddingElimination optimization for not finding shape of input_ids.");
    return Status::OK();
  }
  auto input_ids_shape = input_ids_arg->Shape();
  // For now, we only support all the dims of input_ids_shape besides the first two has dim_value.
  for (int k = 2; k < input_ids_shape->dim_size(); k++) {
    if (!input_ids_shape->dim(k).has_dim_value()) {
      LOG_DEBUG_INFO(logger, "Exit PaddingElimination optimization for shape dims of input_ids has no value.");
      return Status::OK();
    }
  }

  IterateSubgraphFromNode(graph, embedding_node, subgraph, candidate_inputs, candidate_outputs, logger);

  // Add Reshape + Sub + NonZero + Squeeze to get the not padding index to be gathered
  InlinedVector<NodeArg*> reshape_input_args;
  reshape_input_args.reserve(2);
  reshape_input_args.push_back(input_ids_arg);
  InlinedVector<int64_t> new_input_ids_shape;
  new_input_ids_shape.reserve(static_cast<int64_t>(input_ids_shape->dim_size()) - 1);
  new_input_ids_shape.push_back(-1);  // Flatten the two leading dims
  for (int k = 2; k < input_ids_shape->dim_size(); k++) {
    new_input_ids_shape.push_back(input_ids_shape->dim(k).dim_value());
  }
  reshape_input_args.push_back(
      CreateInitializerFromVector(graph,
                                  {static_cast<int64_t>(new_input_ids_shape.size())},
                                  new_input_ids_shape,
                                  graph.GenerateNodeArgName("flattened_shape")));

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

  // Get the first two dims value of input_ids which is [batch_size, seq_len]
  NodeArg* first_two_dims_arg = GetDimsValue(graph,
                                             input_ids_arg,
                                             CreateInitializerFromVector(graph, {2}, {0, 1},
                                                                         graph.GenerateNodeArgName("first_two_indices")),
                                             *embedding_node);

  // Add flatten pattern to each input node of the subgraph
  // to flattern the shape of [batch_size, seqlen, ...] to [valid_token_count, ...]
  InsertFlattenPatternForInput(graph, *embedding_node, 1, squeeze_out_arg, logger);
  handled_input_count++;
  modified = true;
  for (auto& node : candidate_inputs) {
    for (uint32_t i = 0; i < node->InputDefs().size(); ++i) {
      if (subgraph.find(node->MutableInputDefs()[i]) == subgraph.end()) {
        // The type of node is one of Elementwise ops.
        // The input size must be 2 and there must be more than one input in the subgraph.
        ORT_ENFORCE(node->InputDefs().size() == 2);
        // Because candidate_inputs are nodes iterated from embedding node, each of them must have at least one arg in
        // the subgraph and the i-th input of this node is not in the subgraph, so the other input must be in the subgraph
        // and has shape of [batch_size, seq_len, ...]
        NodeArg* arg_in_subgraph = node->MutableInputDefs()[1 - i];
        NodeArg* arg_not_in_subgraph = node->MutableInputDefs()[i];
        // There are three possibilities for the shape of arg_not_in_subgraph:
        //  1. The size of arg_not_in_subgraph->Shape is smaller than arg_in_subgraph->Shape by 2,
        //     which means the shape of arg_not_in_subgraph has no [batch_size, seq_len] in beginning,
        //     and do not need to add flatten pattern to it.
        //  2. The arg_not_in_subgraph->Shape.size == arg_in_subgraph->Shape.size or arg_in_subgraph->Shape.size - 1,
        //     and the first two dims do not equal [batch_size, seq_len].
        //     In this case we just expand the arg_not_in_subgraph->Shape to [batch_size, seq_len, ...],
        //     then the case becomes same with 3.
        //  3. The size of arg_not_in_subgraph->Shape is equal with size of arg_in_subgraph->Shape,
        //     and the first two dims of arg_not_in_subgraph->Shape is [batch_size, seq_len].
        //     Because the shape of arg_in_subgraph will be flattened to [valid_tokens, ... ] automatically after
        //     the shape of input_ids is flattened, so we need to insert flatten pattern for arg_not_in_subgraph->Shape.
        if (arg_not_in_subgraph->Shape()->dim_size() <= arg_in_subgraph->Shape()->dim_size() - 2) {
          continue;
        } else if (arg_in_subgraph->Shape()->dim_size() != arg_not_in_subgraph->Shape()->dim_size() ||
                   arg_in_subgraph->Shape()->dim(0) != arg_not_in_subgraph->Shape()->dim(0) ||
                   arg_in_subgraph->Shape()->dim(1) != arg_not_in_subgraph->Shape()->dim(1)) {
          InsertExpandForNodeInput(graph, *node, i, first_two_dims_arg, logger);
          expanded_input_count++;
        }
        InsertFlattenPatternForInput(graph, *node, i, squeeze_out_arg, logger);
        handled_input_count++;
      }
    }
  }

  // Add pattern to each output node of the subgraph
  // to unflatten the shape of [valid_token_count, ...] to [batch_size, seq_len, ...]
  for (const auto& node : candidate_outputs) {
    for (uint32_t i = 0; i < node->InputDefs().size(); ++i) {
      if (subgraph.find(node->MutableInputDefs()[i]) != subgraph.end()) {
        InsertNodesForOutput(graph, *node, i, squeeze_out_arg, first_two_dims_arg, logger);
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
  if (handled_input_count > 0 || handled_output_count > 0) {
    LOGS(logger, INFO) << "PaddingElimination::Total handled input node count:  " << handled_input_count
                       << " output node count: " << handled_output_count
                       << " expanded input count: " << expanded_input_count;
  }
  return Status::OK();
}

}  // namespace onnxruntime

#endif
