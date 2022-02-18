// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph.h"

#include <cassert>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stack>
#include <queue>

#include "gsl/gsl"
#include "core/common/logging/logging.h"
#include "core/flatbuffers/flatbuffers_utils.h"
#include "core/flatbuffers/schema/ort.fbs.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/graph/graph_flatbuffers_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/graph/model.h"
#include "core/graph/model_load_utils.h"
#include "core/graph/op.h"
#include "core/graph/runtime_optimization_record_container.h"
#include "core/graph/graph_utils.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/graph/function.h"
#include "core/graph/function_impl.h"
#include "core/graph/schema_registry.h"
#include "onnx/checker.h"
using namespace ONNX_NAMESPACE::checker;
#endif

using namespace ONNX_NAMESPACE;
using namespace ONNX_NAMESPACE::Utils;
using namespace ::onnxruntime::common;

namespace onnxruntime {

#if !defined(ORT_MINIMAL_BUILD)
#define NO_CHANGE_ON_SYNC_FLAG(...)                  \
  do {                                               \
    const bool sync_needed = GraphProtoSyncNeeded(); \
    { __VA_ARGS__; }                                 \
    GraphProtoSyncNeeded(sync_needed);               \
  } while (0)

static bool UsingLatestOnnxOpset(const DomainToVersionMap& opset_versions) {
  bool is_latest_opset = false;
  auto onnx_opset = opset_versions.find(kOnnxDomain);

  if (onnx_opset != opset_versions.cend()) {
    static int latest_onnx_version = model_load_utils::IsAllowReleasedONNXOpsetsOnlySet()
                                         ? ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange().LastReleaseVersionMap().at(ONNX_NAMESPACE::ONNX_DOMAIN)
                                         : ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange().Map().at(ONNX_NAMESPACE::ONNX_DOMAIN).second;
    if (onnx_opset->second == latest_onnx_version) {
      is_latest_opset = true;
    }
  }

  return is_latest_opset;
}

static bool GraphLoadedFromModelFile(const GraphProto* graph_proto) {
  return graph_proto && (graph_proto->node_size() != 0 ||
                         graph_proto->output_size() != 0);
}


#endif  // !defined(ORT_MINIMAL_BUILD)


// Constructor: Given a <GraphProto> loaded from model file, construct
// a <Graph> object and Resolve() it.
// Status Graph::LoadGraph(const GraphProto& graph_proto,
//                        const std::unordered_map<std::string, int>& domain_to_version,
//                        Version ir_version,
//                        std::unique_ptr<Graph>& new_graph) {
//  // create instance. need to call private ctor so can't use make_unique
//  GSL_SUPPRESS(r .11)
//  new_graph.reset(new Graph(nullptr, &graph_proto, domain_to_version, ir_version));
//
//  // as we just loaded from file we want to fully initialize/Resolve, but not let that change
//  // the proto sync flag
//  ResolveOptions options;
//  options.no_proto_sync_required = true;
//  auto status = new_graph->Resolve(options);
//  return status;
//}
using google::protobuf::RepeatedPtrField;

#if !defined(ORT_MINIMAL_BUILD)

Graph::Graph(const Model& owning_model,
             GraphProto* graph_proto,
             const std::unordered_map<std::string, int>& domain_to_version,
             Version ir_version,
             IOnnxRuntimeOpSchemaCollectionPtr schema_registry,
             const std::vector<const ONNX_NAMESPACE::FunctionProto*>& model_functions,
             const logging::Logger& logger)
    : Graph(owning_model, graph_proto, domain_to_version, ir_version, schema_registry, nullptr, nullptr, model_functions, logger) {}

Graph::Graph(const Model& owning_model,
             GraphProto* graph_proto, const std::unordered_map<std::string, int>& domain_to_version, Version ir_version,
             IOnnxRuntimeOpSchemaCollectionPtr schema_registry, Graph* parent_graph, const Node* parent_node,
             //TODO!!!
             //Fix the function
             const std::vector<const ONNX_NAMESPACE::FunctionProto*>& /*model_functions*/,
             const logging::Logger& logger)
    : owning_model_(owning_model),
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
      runtime_optimizations_ptr_(std::make_unique<RuntimeOptimizationRecordContainer>()),
      runtime_optimizations_(*runtime_optimizations_ptr_),
#endif
      schema_registry_(schema_registry),
      graph_resolve_needed_(true),
      domain_to_version_(domain_to_version),
      ir_version_(ir_version),
      using_latest_onnx_opset_(UsingLatestOnnxOpset(domain_to_version)),
      parent_graph_(parent_graph),
      parent_node_(parent_node),
      logger_(logger),
      is_loaded_from_model_file_(GraphLoadedFromModelFile(graph_proto)),
      graph_context_(graph_proto, ModelPath(), this, ir_version, IsSubgraph(), logger_){
  ORT_ENFORCE(graph_proto != nullptr, "graph_proto cannot be null");
  ArgNameToTypeMap name_to_type_map;
  //TODO!!!
  //Re-implement function
  /*for (auto func : model_functions) {
    model_local_functions_[function_utils::GetFunctionIdentifier(func->domain(), func->name())] = func;
  }*/

  // set the input/output for main function
  if (is_loaded_from_model_file_) {
    InitializeStateFromModelFileGraphProto(graph_proto);
  }
}

Graph::Graph(Graph& parent_graph, const Node& parent_node, ONNX_NAMESPACE::GraphProto& subgraph_proto)
    : Graph(parent_graph.owning_model_,
            &subgraph_proto,
            parent_graph.DomainToVersionMap(), parent_graph.IrVersion(), parent_graph.schema_registry_,
            &parent_graph,
            &parent_node, {},
            parent_graph.logger_) {
}

// init the main funciton's input/output
void Graph::InitializeStateFromModelFileGraphProto(ONNX_NAMESPACE::GraphProto* graph_proto) {
  auto& main_func_inputs = graph_context_.GetMainFunction().GetInputs();
  auto& main_func_outputs = graph_context_.GetMainFunction().GetOutputs();
  auto& main_func_value_info = graph_context_.GetMainFunction().GetValueInfo();
  ORT_ENFORCE(
      graph_inputs_excluding_initializers_.empty() && main_func_inputs.empty() &&
          main_func_value_info.empty() && main_func_outputs.empty(),
      "Graph state to be loaded into must be empty.");

  // Name to NodeArg mapping of all graph initializers.
  std::unordered_map<std::string, const NodeArg*> graph_initializers;
  std::vector<const NodeArg*> v_graph_initializers;

  // Name to NodeArg mapping of all graph inputs.
  std::unordered_map<std::string, const NodeArg*> graph_inputs;
  std::vector<const NodeArg*> v_graph_inputs;

  // Name to NodeArg mapping of all graph node outputs.
  std::unordered_map<std::string, const NodeArg*> nodes_outputs;
  std::vector<const NodeArg*> v_graph_outputs;

  for (auto& initializer : graph_proto->initializer()) {
    auto& initializer_name = initializer.name();
    auto initializer_arg = GetNodeArg(initializer_name);
    graph_initializers.insert({initializer_name, initializer_arg});
  }

  // Set graph inputs.
  // <graph_inputs_including_initializers_> contains inputs exactly specified in proto.
  // <graph_inputs_excluding_initializers_> contains inputs without default value (specified as initializer).
  for (auto& graph_input : graph_proto->input()) {
    auto& name = graph_input.name();
    const auto* node_arg = GetNodeArg(name);
    ORT_ENFORCE(node_arg, "Graph ctor should have created NodeArg for initializer. Missing:", name);
    graph_inputs.insert({name, node_arg});
    v_graph_inputs.push_back(node_arg);
    if (graph_initializers.end() == graph_initializers.find(name)) {
      graph_inputs_excluding_initializers_.push_back(node_arg);
    }
  }
  graph_context_.GetMutableMainFunction()->SetInputs(v_graph_inputs);

  for (const auto& node : Nodes()) {
    for (const auto* output_def : node.OutputDefs()) {
      nodes_outputs.insert({output_def->Name(), output_def});
    }
  }

  // Set graph outputs.
  // Graph outputs specified in the model must be nodes' outputs, initializers or graph inputs.
  for (auto& graph_output : graph_proto->output()) {
    auto& graph_output_name = graph_output.name();
    auto iter = nodes_outputs.find(graph_output_name);
    if (nodes_outputs.end() == iter) {
      // Graph output is not found as any node's output.
      auto iter2 = graph_initializers.find(graph_output_name);
      if (graph_initializers.end() == iter2) {
        // Graph output is not found as any initializer.
        auto iter3 = graph_inputs.find(graph_output_name);
        if (graph_inputs.end() == iter3) {
          // Graph output is not found as any graph input.
          ORT_THROW(
              "This is an invalid model. Graph output (", graph_output_name,
              ") does not exist in the graph.");
        }
        v_graph_outputs.push_back(iter3->second);
        continue;
      }
      v_graph_outputs.push_back(iter2->second);
      continue;
    }
    v_graph_outputs.push_back(iter->second);
  }
  graph_context_.GetMutableMainFunction()->SetOutputs(v_graph_outputs);

  // Set graph value_info_.
  for (const auto& graph_value_info : graph_proto->value_info()) {
    const auto& name = graph_value_info.name();
    const auto* node_arg = GetNodeArg(name);
    if (node_arg != nullptr) {
      graph_context_.GetMutableMainFunction()->AddValueInfo(node_arg);
    }
  }

  ComputeOverridableInitializers();
}

Status Graph::VerifyNoDuplicateName() {
  auto& inputs_and_initializers = resolve_context_.inputs_and_initializers;
  auto& output_args = resolve_context_.output_args;
  auto& node_name_to_index = resolve_context_.node_name_to_index;

  output_args.clear();
  node_name_to_index.clear();
  // inputs_and_initializers: this is passed in as a parameter, since functions don't have initializers
  // but graphs have them.

  for (auto& node : Nodes()) {
    // Verify node name should be unique.
    auto& node_name = node.Name();

    if (!node_name.empty() && node_name_to_index.end() != node_name_to_index.find(node_name)) {
      // The node has name and its name was used by another node.
      Status status(ONNXRUNTIME, FAIL,
                    "This is an invalid model. Error: two nodes with same node name (" + node_name + ").");
      return status;
    }

    node_name_to_index[node_name] = node.Index();

    // Verify node outputs' name should be unique.
    int output_index = -1;
    for (const auto* output_def : node.OutputDefs()) {
      ++output_index;
      if (output_def->Exists()) {
        auto& output_arg_name = output_def->Name();
        if (inputs_and_initializers.count(output_arg_name)) {
          Status status(ONNXRUNTIME, FAIL,
                        "This is an invalid model. Error: Duplicate definition of name (" + output_arg_name + ").");
          return status;
        }
        auto result = output_args.insert({output_arg_name, {&node, output_index}});
        if (!result.second) {
          // Two outputs with same name, so that insertion fails.
          Status status(ONNXRUNTIME, FAIL,
                        "This is an invalid model. Error: Duplicate definition of name (" + output_arg_name + ").");
          return status;
        }
      }
    }
  }
  return Status::OK();
}

// Recurse into any subgraphs to update the list of NodeArg values in outer scope.
// This information is needed to resolve any dependencies on outer scope values.
common::Status Graph::SetOuterScopeNodeArgs(const std::unordered_set<std::string>& outer_scope_node_args) {
  resolve_context_.outer_scope_node_args = outer_scope_node_args;

  if (!resolve_context_.nodes_with_subgraphs.empty()) {
    // Build the list of NodeArg's that are valid for a subgraph of this GraphBase instance:
    //   - outer scope for this graph
    //   - any inputs/initializers from this graph
    //   - any outputs from nodes in this graph
    //
    // We provide outputs from all nodes in this graph at this stage.
    // BuildConnections will link the node with the subgraph to any outer scope Node/NodeArgs it consumes.
    // PerformTopologicalSortAndCheckIsAcyclic will validate these links.
    std::unordered_set<std::string> node_args_in_scope_for_subgraph = outer_scope_node_args;

    node_args_in_scope_for_subgraph.insert(resolve_context_.inputs_and_initializers.cbegin(),
                                           resolve_context_.inputs_and_initializers.cend());

    std::transform(resolve_context_.output_args.cbegin(), resolve_context_.output_args.cend(),
                   std::inserter(node_args_in_scope_for_subgraph, node_args_in_scope_for_subgraph.end()),
                   [](const std::pair<std::string, std::pair<Node*, int>>& entry) { return entry.first; });

    for (auto* node : resolve_context_.nodes_with_subgraphs) {
      for (auto& subgraph : node->MutableSubgraphs()) {
        auto status = subgraph->SetOuterScopeNodeArgs(node_args_in_scope_for_subgraph);
        ORT_RETURN_IF_ERROR(status);
      }
    }
  }

  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
void Graph::AddEdge(NodeIndex src_node_index, NodeIndex dst_node_index, int src_arg_slot, int dst_arg_slot) {
  graph_context_.GetMutableMainFunction()->AddEdge(src_node_index, dst_node_index, src_arg_slot, dst_arg_slot);
}

void Graph::RemoveEdge(NodeIndex src_node_index, NodeIndex dst_node_index, int src_arg_slot, int dst_arg_slot) {
  graph_context_.GetMutableMainFunction()->AddEdge(src_node_index, dst_node_index, src_arg_slot, dst_arg_slot);
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)
GSL_SUPPRESS(es .84)  // ignoring return value from unordered_map::insert causes noisy complaint
Status Graph::BuildConnections(std::unordered_set<std::string>& outer_scope_node_args_consumed) {
  const std::unordered_set<std::string>& outer_scope_node_args = resolve_context_.outer_scope_node_args;

  std::unordered_set<std::string> node_args_consumed_by_subgraphs;

  // recurse into subgraphs first so we can update any nodes in this graph that are used by those subgraphs
  if (!resolve_context_.nodes_with_subgraphs.empty()) {
    for (auto* node : resolve_context_.nodes_with_subgraphs) {
      for (auto& subgraph : node->MutableSubgraphs()) {
        std::unordered_set<std::string> node_args_consumed;
        ORT_RETURN_IF_ERROR(subgraph->BuildConnections(node_args_consumed));

        for (auto& node_arg_name : node_args_consumed) {
          auto node_arg = GetNodeArg(node_arg_name);

          if (node_arg == nullptr) {
            // it's a node arg from outside this graph's scope, so add that to the list we return
            // so that we can add the dependency at the next level up. this happens if you have multiple
            // levels of subgraphs between the graph with the original NodeArg and the subgraph with implicit usage.
            ORT_IGNORE_RETURN_VALUE(outer_scope_node_args_consumed.insert(node_arg_name));

            if (!parent_graph_) {
              return ORT_MAKE_STATUS(
                  ONNXRUNTIME, INVALID_GRAPH,
                  "This is an invalid model. At top level graph without matching NodeArg that subgraph consumes. Name=",
                  node_arg_name,
                  " Graph may not conform to the ONNX spec and contain initializers that are not graph inputs.");
            }

            node_arg = parent_graph_->GetNodeArgIncludingParentGraphs(node_arg_name);

            // make sure the node arg is found in the parent graph/s
            if (!node_arg) {
              return ORT_MAKE_STATUS(
                  ONNXRUNTIME, INVALID_GRAPH,
                  "This is an invalid model. Failed to find NodeArg in all parent graphs. Name=", node_arg_name,
                  " Graph may not conform to the ONNX spec and contain initializers that are not graph inputs.");
            }
          } else {
            // this value may be produced by this graph, or it could still be coming from a parent graph if it
            // is also directly consumed at this level as we create a NodeArg for all Node inputs in this graph.
            // due to that we need to check the outputs from this level to determine if it is an outer scope value.
            // we don't have that info yet so store and check before returning from BuildConnections
            ORT_IGNORE_RETURN_VALUE(node_args_consumed_by_subgraphs.insert(node_arg_name));
          }

          // add it to the Node's list of implicit inputs
          auto& implicit_inputs = node->MutableDefinitions().implicit_input_defs;
          int input_slot_index = static_cast<int>(node->GetDefinitions().input_defs.size());
          auto iter = std::find(implicit_inputs.cbegin(), implicit_inputs.cend(), node_arg);
          if (implicit_inputs.cend() == iter) {
            implicit_inputs.push_back(node_arg);
            input_slot_index += static_cast<int>(implicit_inputs.size() - 1);
          } else {
            input_slot_index += static_cast<int>(iter - implicit_inputs.cbegin());
          }

          auto entry = resolve_context_.output_args.find(node_arg_name);
          if (entry != resolve_context_.output_args.end()) {
            // Create relationship between this node (node), and the node providing the output (output_node).
            Node& output_node = *entry->second.first;
            AddEdge(output_node.Index(), node->Index(), entry->second.second, input_slot_index);

            // If this Graph was built manually, remove the implicit input from the graph outputs
            // if it is present there and not explicitly listed in the ordered graph outputs
            // (as that implies we should leave it as an output).
            // If the Graph was loaded from a GraphProto, honor the explicit graph outputs and leave as is.
            if (!is_loaded_from_model_file_) {
              graph_context_.GetMutableMainFunction()->RemoveFromOutputs(node_arg);
            }
          }
        }
      }
    }
  }

  // now build connections within this Graph instance
  for (auto& node : Nodes()) {
    const auto input_args = node.InputDefs();

    if (!input_args.empty()) {
      // This node needs inputs.

      int input_slot_index = -1;
      for (const auto* input_arg : input_args) {
        ++input_slot_index;
        if (!input_arg->Exists()) {
          // This input could be optional and it does not exist in this case.
          continue;
        }

        const auto& input_arg_name = input_arg->Name();
        auto output_arg_iter = resolve_context_.output_args.find(input_arg_name);
        if (resolve_context_.output_args.end() != output_arg_iter) {
          // The input to this node is an output from a previous node in this graph.
          // Create relationship between this node (node), and the node providing the output (output_node).
          Node& output_node = *output_arg_iter->second.first;
          AddEdge(output_node.Index(), node.Index(), output_arg_iter->second.second, input_slot_index);
        } else {
          // the value is either an input, an initializer, or coming from outer scope. we only need to take action
          // if coming from outer scope, so first check if this is a subgraph (otherwise there is no outer scope).
          if (parent_graph_ != nullptr) {
            // make sure it's not an input or initializer first as those override any outer scope values
            if (resolve_context_.inputs_and_initializers.find(input_arg_name) ==
                resolve_context_.inputs_and_initializers.cend()) {
              // If it is present in the outer scope it will be 'fed' by the execution frame
              // providing access to the OrtValue from the outer scope. Pass the name back up so nodes can
              // be linked correctly at that level.
              if (outer_scope_node_args.find(input_arg_name) != outer_scope_node_args.cend()) {
                ORT_IGNORE_RETURN_VALUE(outer_scope_node_args_consumed.insert(input_arg_name));
              }
            }
          } else {
            // Check all the inputs are found.
            //
            // Ignore a Fused node as it could have moved things like initializers to a different device
            // (they're internally available to the fused node but removed from the Graph instance).
            // Fusion happens after the model was loaded in full so we know the inputs were valid originally.
            bool check = node.NodeType() != Node::Type::Fused;
#if defined(ENABLE_TRAINING)
            // Only check initial model load for training as graph modifications there also render inputs 'invalid'.
            check = check && num_resolves_ == 0;
#endif
            auto& outer_scope_arg_names = graph_context_.GetMainFunction().GetOuterScopeNodeArgNames();
            if (check &&
                resolve_context_.inputs_and_initializers.find(input_arg_name) ==
                    resolve_context_.inputs_and_initializers.cend() &&
                // if we're manually creating a Graph for use as a subgraph the outer scope names are manually set
                outer_scope_arg_names.find(input_arg_name) == outer_scope_arg_names.cend()) {
              return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid model. Node input '", input_arg_name,
                                     "' is not a graph input, initializer, or output of a previous node.");
            }
          }
        }
      }
    } else if (node.OutputDefs().empty()) {
      // This is a useless node.
      // It has no input/output.
      RemoveNode(node.Index());
    }
  }

  ORT_RETURN_IF_ERROR(PopulateNodeArgToProducerConsumerLookupsFromNodes());

  // finally check any node args consumed by subgraphs to see if they're available locally.
  // if not we add them to the list of outer scope values consumed.
  for (const auto& name : node_args_consumed_by_subgraphs) {
    if (!graph_context_.GetMainFunction().IsProducedInCurrentGraph(name) &&
        resolve_context_.inputs_and_initializers.find(name) == resolve_context_.inputs_and_initializers.cend()) {
      ORT_IGNORE_RETURN_VALUE(outer_scope_node_args_consumed.insert(name));
    }
  }

  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

NodeArg* Graph::GetNodeArgIncludingParentGraphs(const std::string& node_arg_name) {
  NodeArg* node_arg = GetNodeArg(node_arg_name);

  if (!node_arg && parent_graph_) {
    node_arg = parent_graph_->GetNodeArgIncludingParentGraphs(node_arg_name);
  }

  return node_arg;
}

void Graph::ReverseDFSFrom(const std::vector<NodeIndex>& from,
                           const std::function<void(const Node*)>& enter,
                           const std::function<void(const Node*)>& leave,
                           const std::function<bool(const Node*, const Node*)>& comp) const {
  std::vector<const Node*> node_vec;
  node_vec.reserve(from.size());
  for (auto i : from) {
    node_vec.push_back(GetNode(i));
  }

  ReverseDFSFrom(node_vec, enter, leave, comp, {});
}

void Graph::ReverseDFSFrom(const std::vector<const Node*>& from,
                           const std::function<void(const Node*)>& enter,
                           const std::function<void(const Node*)>& leave,
                           const std::function<bool(const Node*, const Node*)>& comp) const {
  ReverseDFSFrom(from, enter, leave, comp, {});
}

void Graph::ReverseDFSFrom(const std::vector<const Node*>& from,
                           const std::function<void(const Node*)>& enter,
                           const std::function<void(const Node*)>& leave,
                           const std::function<bool(const Node*, const Node*)>& comp,
                           const std::function<bool(const Node* from, const Node* to)>& stop) const {
  graph_context_.GetMainFunction().ReverseDFSFrom(from, enter, leave, comp, stop);
}

#if !defined(ORT_MINIMAL_BUILD)
void Graph::KahnsTopologicalSort(const std::function<void(const Node*)>& enter,
                                 const std::function<bool(const Node*, const Node*)>& comp) const {
  graph_context_.GetMainFunction().KahnsTopologicalSort(enter, comp);
}

GSL_SUPPRESS(es .84)  // noisy warning about ignoring return value from insert(...)
Status Graph::PerformTopologicalSortAndCheckIsAcyclic() {
  nodes_in_topological_order_.clear();
  std::unordered_set<NodeIndex> downstream_nodes;  // nodes downstream of the node we're currently checking
  std::unordered_set<NodeIndex> nodes_seen;        // nodes we have seen but may not have been added to nodes_added yet
  std::unordered_set<NodeIndex> nodes_added;       // nodes added to topo order
  std::stack<NodeIndex> stack;

  // push the root nodes into nodes_in_topological_order in the order they were defined in the model
  // to ensure that is consistent.
  auto& nodes_in_original_order = Nodes();
  std::for_each(nodes_in_original_order.cbegin(), nodes_in_original_order.cend(),
                [&](const Node& node) {
                  auto index = node.Index();

                  // find the top level nodes in the graph.
                  // need to also consider nodes that only have Constants as inputs as top level nodes,
                  // as the constant will get replaced by an initializer.
                  auto input_edges = node.GetRelationships().input_edges;
                  auto has_inputs = std::any_of(input_edges.cbegin(), input_edges.cend(),
                                                [](const Node::EdgeEnd& edge) {
                                                  return edge.GetNode().OpType() != kConstant;
                                                });

                  if (!has_inputs) {
                    // add to the topological list, and ensure we skip these nodes when walking the graph
                    nodes_in_topological_order_.push_back(index);
                    nodes_added.insert(index);
                    nodes_seen.insert(index);
                  }
                });

  // find all the leaf nodes (nodes with no output edges as there's no edge to a graph output)
  for (auto iter = Nodes().begin(); iter != Nodes().end(); ++iter) {
    if (iter->relationships_.output_edges.empty()) {
      stack.push(iter->Index());
    }
  }

  // work our way up from the leaf nodes
  while (!stack.empty()) {
    const NodeIndex current = stack.top();
    stack.pop();

    if (nodes_added.find(current) != nodes_added.end()) {
      continue;
    }

    if (nodes_seen.find(current) != nodes_seen.end()) {
      // we popped the stack and are back to a node that was seen previously,
      // so we know all the upstream nodes from it have been added.
      nodes_in_topological_order_.push_back(current);
      nodes_added.insert(current);
      downstream_nodes.erase(current);
      continue;
    }

    const Node* node = GetNode(current);
    if (!node) {
      continue;
    }

    // node hasn't been seen before, so mark it as seen and re-add it along with its inputs
    // also mark it as downstream of anything new that is added to the stack to detect acyclic graphs
    nodes_seen.insert(current);
    downstream_nodes.insert(current);

    stack.push(current);

    for (auto iter = node->InputNodesBegin(), end = node->InputNodesEnd(); iter != end; ++iter) {
      const NodeIndex idx = iter->Index();
      // the input to this node is also downstream of this node
      if (downstream_nodes.find(idx) != downstream_nodes.end()) {
        Status status(ONNXRUNTIME, FAIL, "This is an invalid model. Error: the graph is not acyclic.");
        return status;
      }

      // avoid re-processing nodes
      if (nodes_seen.find(idx) == nodes_seen.end()) {
        stack.push(idx);
      }
    }
  }

  auto num_of_nodes = graph_context_.GetMainFunction().NumberOfNodes();
  if (num_of_nodes >= 0 && static_cast<size_t>(num_of_nodes) == nodes_in_topological_order_.size()) {
    return Status::OK();
  }

  return Status(ONNXRUNTIME, FAIL, "This is an invalid model. Error: the graph is not acyclic.");
}

bool FullyDefinedType(const TypeProto& type_proto) {
  switch (type_proto.value_case()) {
    case TypeProto::kTensorType: {
      auto& tensor_type = type_proto.tensor_type();
      return utils::HasElemType(tensor_type);
    }
#if !defined(DISABLE_SPARSE_TENSORS)
    case TypeProto::kSparseTensorType: {
      auto& tensor_type = type_proto.sparse_tensor_type();
      return utils::HasElemType(tensor_type);
    }
#endif
    case TypeProto::kSequenceType: {
      auto& seq_type = type_proto.sequence_type();
      return utils::HasElemType(seq_type) && FullyDefinedType(seq_type.elem_type());
    }
#if !defined(DISABLE_OPTIONAL_TYPE)
    case TypeProto::kOptionalType: {
      auto& optional_type = type_proto.optional_type();
      return utils::HasElemType(optional_type) && FullyDefinedType(optional_type.elem_type());
    }
#endif
    case TypeProto::kMapType: {
      auto& map_type = type_proto.map_type();
      return utils::HasKeyType(map_type) &&
             utils::HasValueType(map_type) &&
             FullyDefinedType(map_type.value_type());
    }
    case TypeProto::kOpaqueType:
      return true;
    case TypeProto::VALUE_NOT_SET:
    default:
      return false;
  }
}

// function to handle type/shape inferencing of a subgraph.
// parameters are the Graph instance for the subgraph, the input types from the control flow node that contains
// the subgraph, and the vector to write the output from the inferencing.
using SubgraphInferencingFunc =
    std::function<Status(const Node&, Graph&, const std::vector<const TypeProto*>&, std::vector<const TypeProto*>&, const Graph::ResolveOptions&)>;

class GraphInferencerImpl : public ONNX_NAMESPACE::GraphInferencer {
 public:
  GraphInferencerImpl(const Node& node, Graph& graph, SubgraphInferencingFunc& inferencing_func, const Graph::ResolveOptions& options)
      : node_(node), graph_(graph), inferencing_func_(inferencing_func), options_(options) {
  }

  // Perform inferencing on the graph contained in GraphInferencer.
  // Returns the graph output types post-inferencing.
  // We ignore input_data currently as the inferencing happens prior to receiving user input.
  std::vector<const TypeProto*> doInferencing(const std::vector<const TypeProto*>& input_types,
                                              const std::vector<const TensorProto*>& /*input_data*/) override {
    std::vector<const TypeProto*> output_types;

    auto status = inferencing_func_(node_, graph_, input_types, output_types, options_);

    if (status != Status::OK()) {
      fail_type_inference("Graph attribute inferencing failed: ", status.ErrorMessage());
    }

    return output_types;
  }

 private:
  const Node& node_;
  Graph& graph_;
  SubgraphInferencingFunc& inferencing_func_;
  const Graph::ResolveOptions& options_;
};

// An implementation of the InferenceContext interface required by operator-specific
// shape inference for onnxruntime graphs.
class InferenceContextImpl : public ONNX_NAMESPACE::InferenceContext {
  using AttributeGraphMap = std::unordered_map<std::string, Graph*>;

 public:
  InferenceContextImpl(Node& node,
                       SubgraphInferencingFunc subgraph_inferencing_func,
                       const Graph& graph,
                       const Graph::ResolveOptions& options) noexcept
      : node_(node),
        subgraph_inferencing_func_(subgraph_inferencing_func),
        graph_(graph),
        options_(options) {
    node_output_types_.resize(node.OutputDefs().size());
  }

  void RunInferencing() {
    auto schema = node_.Op();
    if (nullptr != schema) {
      schema->GetTypeAndShapeInferenceFunction()(*this);
    }
  }

  std::vector<TypeProto> InferredOutputTypes() const { return node_output_types_; }

  const AttributeProto* getAttribute(const std::string& name) const override {
    auto& attribute_value_map = node_.GetAttributes();
    auto iter = attribute_value_map.find(name);
    if (iter == attribute_value_map.end()) {
      return nullptr;
    }
    return &iter->second;
  }

  size_t getNumInputs() const noexcept override {
    return node_.InputDefs().size();
  }

  const TypeProto* getInputType(size_t index) const override {
    const TypeProto* type = nullptr;
    auto p_node_arg = node_.InputDefs().at(index);
    if ((nullptr != p_node_arg) && p_node_arg->Exists()) {
      type = p_node_arg->TypeAsProto();
    }

    return type;
  }

  size_t getNumOutputs() const noexcept override {
    return node_output_types_.size();
  }

  TypeProto* getOutputType(size_t index) override {
    return &node_output_types_[index];
  }

  const TensorProto* getInputData(size_t index) const override {
    auto def = node_.InputDefs()[index];
    if (!def)
      return nullptr;

    // only return data if it's for a constant initializer. checks for outer scope initializers
    // if this is a subgraph and the name isn't found locally.
    const TensorProto* initializer = graph_.GetConstantInitializer(def->Name(), true);
    return initializer;
  }

  // ORT does not implement partial data propagation yet so just return nullptr.
  const TensorShapeProto* getSymbolicInput(size_t) const override {
    return nullptr;
  }

  GraphInferencer* getGraphAttributeInferencer(const std::string& attribute_name) override {
    GraphInferencer* graph_inferencer = nullptr;

    auto* subgraph = node_.GetMutableGraphAttribute(attribute_name);

    if (subgraph) {
      auto inferencer = std::make_unique<GraphInferencerImpl>(node_, *subgraph, subgraph_inferencing_func_, options_);
      graph_inferencer = inferencer.get();
      graph_inferencers_.push_back(std::move(inferencer));
    } else {
      fail_type_inference("No Graph instance was found for attribute ",
                          attribute_name, " in node ", node_.Name());
    }

    return graph_inferencer;
  }

  // XXX: When we changed and kept sparse constant initializers in sparse form,
  // we would adjust this method
  const SparseTensorProto* getInputSparseData(size_t) const override {
    return nullptr;
  }

 private:
  Node& node_;
  // node_output_types_ will be populated by the operator-specific shape inference.
  std::vector<TypeProto> node_output_types_;
  SubgraphInferencingFunc subgraph_inferencing_func_;
  std::vector<std::unique_ptr<GraphInferencerImpl>> graph_inferencers_;
  const Graph& graph_;
  const Graph::ResolveOptions& options_;
};

Status Graph::InferAndVerifySubgraphTypes(const Node& node, Graph& subgraph,
                                          const std::vector<const TypeProto*>& input_types,
                                          std::vector<const TypeProto*>& output_types,
                                          const Graph::ResolveOptions& options) {
  auto status = Status::OK();

  output_types.clear();

  // the spec says all inputs should be provided for the subgraph so default to that first
  auto* subgraph_inputs = &subgraph.GetInputsIncludingInitializers();
  auto num_subgraph_inputs = subgraph_inputs->size();

  if (num_subgraph_inputs != input_types.size()) {
    // we also allow for just the required inputs to be provided to be user friendly due to ONNX requiring
    // initializers to have matching inputs (making them optional inputs that most likely the user doesn't want to
    // override).
    auto& required_subgraph_inputs = subgraph.GetInputs();
    auto num_required_subgraph_inputs = required_subgraph_inputs.size();

    if (num_required_subgraph_inputs != input_types.size()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Size mismatch validating subgraph inputs. Got ", input_types.size(),
                             " inputs but subgraph has ", num_subgraph_inputs,
                             " inputs and requires ", num_required_subgraph_inputs,
                             " inputs. Either provide all subgraph inputs, or just the required inputs.");
    }

    subgraph_inputs = &required_subgraph_inputs;
    num_subgraph_inputs = num_required_subgraph_inputs;
  }

  // apply type/shape info to the subgraph's inputs
  for (size_t i = 0; i < num_subgraph_inputs; ++i) {
    const auto* input_type = input_types[i];
    if (input_type == nullptr) {
      // optional input
      continue;
    }

    const auto& subgraph_input = *subgraph_inputs->at(i);

    NodeArg* mutable_nodearg = subgraph.GetNodeArg(subgraph_input.Name());
    status = mutable_nodearg->UpdateTypeAndShape(*input_type, true, options.override_types, subgraph.logger_);
    if (!status.IsOK()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Node:", node.Name(), " ", status.ErrorMessage());
    }
  }

  // Apply any current input type/shape information to the Nodes in the subgraph that are implicitly
  // consuming NodeArg's from this scope or higher.
  // The NodeArg's that implicit_input_defs point to would have any type/shape inferencing applied to them
  // by now. As the subgraph is referring to the outer scope NodeArg, we simply replace any information in
  // the subgraph with the details from the outer scope NodeArg.
  const auto& implicit_input_defs = node.GetDefinitions().implicit_input_defs;
  for (const auto* implicit_node_arg : implicit_input_defs) {
    auto subgraph_nodearg = subgraph.GetNodeArg(implicit_node_arg->Name());

    // the implicit input defs may be for a nested subgraph, so it won't necessarily match here.
    // if that is the case, we will update the type/shape information when we descend into the
    // nested subgraph later.
    if (!subgraph_nodearg)
      continue;

    status = subgraph_nodearg->UpdateTypeAndShape(*implicit_node_arg, true, options.override_types, subgraph.logger_);
    if (!status.IsOK()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Node:", node.Name(), " ", status.ErrorMessage());
    }

    // all values above us should have a type by now due to ONNX requirements.
    if (subgraph_nodearg->Type() == nullptr)
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Subgraph input missing type.");
  }

  // now that we have handled the input types, do the type/shape inferencing for the subgraph
  // to flow the type/shape info through it
  status = subgraph.PerformTypeAndShapeInferencing(options);
  ORT_RETURN_IF_ERROR(status);

  auto& subgraph_outputs = subgraph.GetOutputs();
  for (const auto* output : subgraph_outputs) {
    output_types.push_back(output->TypeAsProto());
  }

  return Status::OK();
}

Status Graph::UpdateShapeInference(Node& node) {
  // We only use this during constant folding, and we don't constant fold control flow nodes.
  ORT_ENFORCE(node.GetAttributeNameToMutableSubgraphMap().empty(),
              "UpdateTypeShapeInference is not intended to be used with control flow nodes containing subgraphs");

  // Whilst the type inferencing will run again we don't allow type overrides due to using the default
  // ResolveOptions settings, so essentially this can only change the shape information.
  return InferAndVerifyTypeMatch(node, *node.Op(), {});
}

// Implementation of type-inference and type-checking for a single node
GSL_SUPPRESS(f .23)  // spurious warning about inferred_type never being checked for null
Status Graph::InferAndVerifyTypeMatch(Node& node, const OpSchema& op, const ResolveOptions& options) {
  auto& node_name = node.Name();

  // if we're building a graph we permit outer scope node args to have no type
  // as the 'real' Resolve at runtime will have type inferencing
  auto is_outer_scope_nodearg = [this](const std::string& name) {
    auto& outer_scope_node_arg_names = graph_context_.GetMainFunction().GetOuterScopeNodeArgNames();
    return outer_scope_node_arg_names.find(name) != outer_scope_node_arg_names.cend();
  };

  // <k> index used to navigate node->InputDefs().
  int k = 0;
  std::unordered_map<std::string, DataType> type_parameter_to_type_map;

  for (size_t i = 0; i < node.InputArgCount().size(); ++i) {
    // Number of inputs corresponding to the i-th argument.
    const int arg_count = node.InputArgCount()[i];
    // The i-th formal parameter definition.
    auto op_formal_parameter = op.inputs()[i];

    // Check all <arg_count> actual parameters (corresponding to the k-th input)
    // match the formal parameter definition (i-th argument).
    for (int j = 0; j < arg_count; ++j, ++k) {
      const auto* input_def = node.GetDefinitions().input_defs[k];
      if (!input_def->Exists())
        continue;

      if (input_def->Type() == nullptr) {
        // if we are building a subgraph that uses outer scope values,
        // allow an empty type as it will be copied from the outer scope graph at runtime
        if (is_outer_scope_nodearg(input_def->Name()))
          continue;

        // Logic error: This should not happen if we properly checked that every use has
        // a corresponding def, for which type-inference already produced a valid type
        Status status(ONNXRUNTIME, FAIL,
                      "This is an invalid model. "
                      "Node (" +
                          node_name + ") input arg (" +
                          input_def->Name() + ") does not have type information set by parent node.");
        return status;
      }

      // Verify that the actual parameter's type is one of permitted types of the formal parameter
      DataType input_type = input_def->Type();
      auto& permitted_types = op_formal_parameter.GetTypes();
      if (0 == permitted_types.count(input_type)) {
        std::string null_pointer("(null)");
        if (input_type == nullptr) input_type = &null_pointer;
        // Type error in input model/graph.

        Status status(ONNXRUNTIME, INVALID_GRAPH,
                      "This is an invalid model. "
                      "Type Error: Type '" +
                          *input_type + "' of input parameter (" + input_def->Name() +
                          ") of operator (" + op.Name() + ") in node (" + node_name + ") is invalid.");
        return status;
      }

      // When multiple parameters have the same type-variable, they are all required
      // to have the same type. E.g., when adding tensors A and B, it is an error if
      // input A is of type "tensor(int32)" and B is of type "tensor(float)".
      // For variadic arguments, this verification rule is normally applicable:
      // e.g., Concat/Max/Mean/Min/Sum all require all input tensors to be of same type.
      // However, some ops, like the control-flow constructs (Scan, If, Loop) have variadic
      // inputs and outputs of different types. The check is not applicable to such ops.
      if (op_formal_parameter.GetIsHomogeneous()) {
        auto param_to_type_iter = type_parameter_to_type_map.find(op_formal_parameter.GetTypeStr());
        if (type_parameter_to_type_map.end() == param_to_type_iter) {
          // Bind the corresponding type-parameter's value to the actual type:
          type_parameter_to_type_map[op_formal_parameter.GetTypeStr()] = input_type;
        } else if (param_to_type_iter->second != input_type) {
          // Type error in input model/graph:
          // The type-parameter T is bound to different values for different inputs.
          Status status(ONNXRUNTIME, FAIL,
                        "Type Error: Type parameter (" + op_formal_parameter.GetTypeStr() +
                            ") of Optype (" + op.Name() + ") bound to different types (" + *(param_to_type_iter->second) +
                            " and " + *(input_def->Type()) +
                            " in node (" + node_name + ").");
          return status;
        }
      }
    }
  }

  // Apply ONNX's type/shape inference to this node.
  // This will call InferAndVerifySubgraphTypes if the ONNX level type/shape inferencing for the Node attempts
  // to do subgraph type/shape inferencing (Scan/If/Loop nodes).
  // InferAndVerifySubgraphTypes will call PerformTypeAndShapeInferencing for the subgraph, which will recursively
  // handle type/shape inferencing for it.
  // Once that completes, the outputs from the node containing the subgraph will be updated, and the final values
  // returned here.
  SubgraphInferencingFunc func(Graph::InferAndVerifySubgraphTypes);
  InferenceContextImpl context(node, func, *this, options);

  {
    auto status = Status::OK();
    ORT_TRY {
      context.RunInferencing();
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Node (", node.Name(), ") Op (", node.OpType(), ") ", ex.what());
      });
    }
    ORT_RETURN_IF_ERROR(status);
  }

  const auto& onnx_inferred_types(context.InferredOutputTypes());

  // Infer and verify node output arg type information.
  int i = -1;
  for (auto& output_def : node.MutableDefinitions().output_defs) {
    ++i;
    if (!output_def->Exists()) continue;

    // if the number of actual parameters exceeds the number of formal parameters,
    // then the op has variadic outputs and the trailing extra actual parameters
    // correspond to the last formal parameter. (The ONNX schema verification check
    // would have checked that the corresponding formal parameter is variadic.)

    const int num_formal_params = gsl::narrow_cast<int>(op.outputs().size());
    auto operand_index = std::min(i, num_formal_params - 1);
    auto op_formal_parameter = op.outputs().at(operand_index);

    const TypeProto& onnx_inferred_type = onnx_inferred_types[i];
    DataType existing_type = output_def->Type();
    DataType inferred_type = nullptr;

    // Infer output arg type if it is constrained to be of the same type as some input:
    // For example, the output of "Abs" is of the same type as its input.
    bool homogeneous = op_formal_parameter.GetIsHomogeneous();
    auto input_types_iter = type_parameter_to_type_map.find(op_formal_parameter.GetTypeStr());
    if (homogeneous && (type_parameter_to_type_map.end() != input_types_iter)) {
      inferred_type = input_types_iter->second;
    } else if (1 == op_formal_parameter.GetTypes().size()) {
      // Infer output arg type if operator definition specifies unique output type:
      inferred_type = *(op_formal_parameter.GetTypes().begin());
    } else if (FullyDefinedType(onnx_inferred_type)) {
      // Use output type inferred by ONNX inference
      inferred_type = DataTypeUtils::ToType(onnx_inferred_type);
    } else if (existing_type != nullptr) {
      inferred_type = existing_type;
    } else {
      // This should not happen: indicates incompleteness in ONNX inference.
      Status status(ONNXRUNTIME, FAIL,
                    "Node (" + node_name + ") output arg (" + output_def->Name() + ") type inference failed");
      return status;
    }

    if ((existing_type != inferred_type) && (existing_type != nullptr)) {
      // A type exists for this output but does not match the inferred type.

      if (options.override_types) {
        // Replace existing type by inferred type: for use after graph-transformations
        // that change types of variables such as mixed-precision transformation.
        // Note: This reuses the original shape, with inferred type. Transformations
        // that can affect the shape are not yet supported.

        // The "SetType" call will override the shape information to empty.
        // If the original tensor has shape information, need to set it back.
        if (output_def->Shape()) {
          auto old_shape = *output_def->Shape();
          output_def->SetType(inferred_type);
          output_def->SetShape(old_shape);
        } else {
          output_def->SetType(inferred_type);
        }
      } else
        return Status(ONNXRUNTIME, FAIL,
                      "Type Error: Type (" + *existing_type + ") of output arg (" +
                          output_def->Name() + ") of node (" + node_name +
                          ") does not match expected type (" + *inferred_type + ").");
    }

    if (existing_type == nullptr)
      output_def->SetType(inferred_type);

    // Update output-shape if it was inferred:
    // HasShape()/GetShape() work for tensor types
    // if the behavior changes the below may need adjustment
    if (utils::HasShape(onnx_inferred_type)) {
      if (output_def->Shape() == nullptr) {
        output_def->SetShape(utils::GetShape(onnx_inferred_type));
      } else {
        // we need to merge the shapes as a subgraph may have placeholder dimensions to represent the rank
        // that have no values.
        TypeProto merge_target;
        if (utils::HasTensorType(onnx_inferred_type)) {
          *merge_target.mutable_tensor_type()->mutable_shape() = *output_def->Shape();
        }
#if !defined(DISABLE_OPTIONAL_TYPE)
        else if (utils::HasOptionalTensorType(onnx_inferred_type)) {
          *utils::GetMutableOptionalTypeProto(merge_target)
               ->mutable_tensor_type()
               ->mutable_shape() = *output_def->Shape();
        }
#endif

#if !defined(DISABLE_SPARSE_TENSORS)
        else if (utils::HasSparseTensorType(onnx_inferred_type)) {
          *merge_target.mutable_sparse_tensor_type()->mutable_shape() = *output_def->Shape();
        }
#endif
        auto status = graph_utils::MergeShapeInfo(output_def->Name(), onnx_inferred_type, merge_target, using_latest_onnx_opset_, logger_);
        if (!status.IsOK()) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Node:", node_name, " ", status.ErrorMessage());
        }
        // we may have cleared the shape if there was a mismatch so handle that
        if (utils::HasShape(merge_target))
          output_def->SetShape(utils::GetShape(merge_target));
        else
          output_def->ClearShape();
      }
    }
  }

  return Status::OK();
}

// Apply type-inference and type-checking to all inputs and initializers:
common::Status Graph::TypeCheckInputsAndInitializers() {
  // Check that the type of every input is specified:
  for (auto* graph_input : GetInputs()) {
    if (nullptr == graph_input->Type()) {
      Status status(ONNXRUNTIME, FAIL,
                    "This is an invalid model. "
                    "Model input (" +
                        graph_input->Name() + ") does not have type information.");
      return status;
    }
  }

  // Infer/check type and shape for all initializers from their values
  for (auto& initializer_pair : graph_context_.GetAllInitializedTensors()) {
    const std::string& name = initializer_pair.first;
    auto* node_arg = GetNodeArg(name);
    // If node_arg is null, we ignore this as a potentially unused initializer here
    if (nullptr != node_arg) {
      const TensorProto* tensor_proto = initializer_pair.second;
      TypeProto tensor_type;
      tensor_type.mutable_tensor_type()->set_elem_type(tensor_proto->data_type());
      auto initializer_type = DataTypeUtils::ToType(tensor_type);
      auto nodearg_type = node_arg->Type();
      if (nullptr == nodearg_type)
        node_arg->SetType(initializer_type);
      else if (initializer_type != nodearg_type) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Type Error: Data in initializer '", name, "' has element type ", *initializer_type,
                               " but usage of initializer in graph expects ", *nodearg_type);
      }

      // Set shape accordingly.
      TensorShapeProto inferred_shape;
      for (auto dim : tensor_proto->dims()) {
        inferred_shape.add_dim()->set_dim_value(dim);
      }

      const TensorShapeProto* p_existing_shape = node_arg->Shape();
      if (nullptr == p_existing_shape) {
        // use the inferred shape if this is a constant initializer (cannot be overridden).
        // if not it has a matching graph input, and we prefer the shape info (or lack of info) from the graph input
        if (GetConstantInitializer(name, false) != nullptr) {
          node_arg->SetShape(inferred_shape);
        }
      } else {
        if (p_existing_shape->dim_size() != tensor_proto->dims_size()) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Type Error: Shape of initializer ", name, " does not match. ",
                                 *p_existing_shape, " != ", *tensor_proto);
        }

        for (int i = 0; i < p_existing_shape->dim_size(); ++i) {
          auto& d = p_existing_shape->dim(i);
          if (utils::HasDimValue(d) && (d.dim_value() != tensor_proto->dims(i))) {
            return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                   "Type Error: Shape of initializer ", name, " does not match. ",
                                   *p_existing_shape, " != ", *tensor_proto);
          }
        }
      }
    }
  }

  return Status::OK();
}

Status Graph::VerifyNodeAndOpMatch(const ResolveOptions& options) {
  CheckerContext ctx;
  ctx.set_ir_version(gsl::narrow_cast<int>(IrVersion()));
  ctx.set_opset_imports(DomainToVersionMap());
  ctx.set_schema_registry(schema_registry_.get());
  // Set the parent directory of model path to load external tensors if exist
  ctx.set_model_dir(ToUTF8String(ModelPath().ParentPath().ToPathString()));

  LexicalScopeContext lsc;
  lsc.output_names.insert(resolve_context_.inputs_and_initializers.cbegin(),
                          resolve_context_.inputs_and_initializers.cend());

  // technically we could add values from Node.GetDefinitions().implicit_input_defs on a per-node basis inside
  // the below loop so that we only check against the specific outer dependencies of the node.
  // doing that requires lots of copies of LexicalScopeContext.output_names to clear out the per-Node values
  // after each loop. instead add all the outer scope values upfront so we can just accumulate new inner scope values
  // during each loop iteration.
  lsc.output_names.insert(resolve_context_.outer_scope_node_args.cbegin(),
                          resolve_context_.outer_scope_node_args.cend());

  // we may have some locally defined outer scope args if we're in the middle of constructing a subgraph
  // and need to call Resolve
  auto& outer_scope_node_args = graph_context_.GetMainFunction().GetOuterScopeNodeArgNames();
  lsc.output_names.insert(outer_scope_node_args.cbegin(), outer_scope_node_args.cend());

  for (auto node_index : nodes_in_topological_order_) {
    // Node verification.
    auto& node = *GetNode(node_index);

    NodeProto node_proto;
    node.ToProto(node_proto);
    const auto& node_name = node.Name();

    if (!node.Op()) {
      {
        auto status = Status::OK();
        ORT_TRY {
          checker::check_node(node_proto, ctx, lsc);
        }
        ORT_CATCH(const std::exception& ex) {
          ORT_HANDLE_EXCEPTION([&]() {
            status = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_GRAPH,
                                     "This is an invalid model. In Node, ", node, ", Error ", ex.what());
          });
        }
        ORT_RETURN_IF_ERROR(status);
      }

      SetOpSchemaFromRegistryForNode(node);

      //TODO!!!!
      //Re-implement funciton
      /*if (!node.op_ || (node.op_ && (node.op_->HasFunction() || node.op_->HasContextDependentFunction()))) {
        InitFunctionBodyForNode(node);
      }*/

      if (!node.op_) {
        return Status(ONNXRUNTIME, FAIL, "Fatal error: " + node.OpType() + " is not a registered function/op");
      }

      // For ops without schema (like model local functions set the since version after constructing the schema.
      // schema construction will happen during function body initialization.
      if (node.since_version_ == -1) {
        node.since_version_ = node.op_->since_version();
      }
    } else {
      // This is only applicable for model local functions.
      // In case of nested model local functions, graph resolve is called during resolve for parent
      // function body graph otherwise type inference for nest function cannot happen.
      //TODO!!!!
      //Re-implement funciton
      /*if (options.traverse_function_body && node.GetFunctionBody() != nullptr) {
        ORT_RETURN_IF_ERROR(node.GetMutableFunctionBody()->MutableBody().Resolve(options));
      }*/
    }

    ORT_RETURN_IF_ERROR(node.UpdateInputArgCount());

    // currently an Op is required by ValidateVersion, so we use gsl::not_null to validate that.
    // This may change in the future to allow a null Op
    const gsl::not_null<const OpSchema*> p_op{node.Op()};

    // Attribute verification and fill node attribute with
    // default value defined in operator definition if needed.
    // Fill node attribute with default value specified in operator definition if any.
    const auto& node_attributes = node.GetAttributes();
    for (const auto& attr_def : p_op->attributes()) {
      auto node_attr_iter = node_attributes.find(attr_def.first);
      if (node_attributes.end() == node_attr_iter) {
        // The attribute was not specified in the node.
        if (!attr_def.second.required) {
          if (utils::HasName(attr_def.second.default_value)) {
            // Set default value to the node attributes.
            node.AddAttribute(attr_def.first, attr_def.second.default_value);
          }
          // TODO: Handle optional attribute but no default value specified in op definition.
        } else {
          Status status(ONNXRUNTIME, FAIL,
                        "This is an invalid model. "
                        "Node (" +
                            node_name + ") attribute (" + attr_def.first +
                            ") is required but not specified.");
          return status;
        }
      }
    }

    NO_CHANGE_ON_SYNC_FLAG(ORT_RETURN_IF_ERROR(InferAndVerifyTypeMatch(node, *p_op, options)));

    // Accumulate output names of the iterated Node
    for (auto& output_name : node_proto.output()) {
      lsc.output_names.insert(output_name);
    }
  }

  // verify subgraphs
  for (auto node_index : nodes_in_topological_order_) {
    auto& node = *GetNode(node_index);
    for (auto& entry : node.GetAttributeNameToMutableSubgraphMap()) {
      Graph* subgraph = entry.second;
      ORT_RETURN_IF_ERROR(subgraph->VerifyNodeAndOpMatch(options));
    }
  }

  return Status::OK();
}
//TODO!!!
//Re-implement this
//const std::unordered_map<std::string, const ONNX_NAMESPACE::FunctionProto*>& Graph::GetModelLocalFunctions() const {
//  if (parent_graph_ == nullptr) {
//    return model_local_functions_;
//  }
//  return parent_graph_->GetModelLocalFunctions();
//}

//TODO!!!
//Re-implement this
//void Graph::InitFunctionBodyForNode(Node& node) {
//  ONNX_NAMESPACE::FunctionProto onnx_function_proto;
//  if (node.op_ && (node.op_->HasFunction() || node.op_->HasContextDependentFunction())) {
//    // This node has a schema defined function proto. If it is a context dependent function
//    // then build it otherwise fetch the FunctionProto from schema.
//    if (node.op_->HasContextDependentFunction()) {
//      NodeProto node_proto;
//      node.ToProto(node_proto);
//      std::vector<TypeProto> input_types;
//      for (size_t i = 0, n = node.InputDefs().size(); i < n; i++) {
//        auto p_node_arg = node.InputDefs().at(i);
//        if ((nullptr != p_node_arg) && p_node_arg->Exists()) {
//          auto& type = *(p_node_arg->TypeAsProto());
//          input_types.emplace_back(type);
//        } else
//          input_types.emplace_back();
//      }
//      ONNX_NAMESPACE::FunctionBodyBuildContextImpl function_body_ctx(node_proto, input_types);
//      if (!node.op_->BuildContextDependentFunction(function_body_ctx, onnx_function_proto))
//        return;
//    } else {
//      onnx_function_proto = *(node.op_->GetFunction());
//    }
//  } else {
//    std::string func_identifier = function_utils::GetFunctionIdentifier(node.Domain(), node.OpType());
//    const auto& model_local_functions = GetModelLocalFunctions();
//    auto iter = model_local_functions.find(func_identifier);
//    if (iter == model_local_functions.end()) {
//      return;
//    }
//
//    // This node has a model local function proto.
//    onnx_function_proto = *(iter->second);
//  }
//
//  ORT_TRY {
//    // Explicitly pass the model local functions as t
//    auto func_ptr = std::make_unique<onnxruntime::FunctionImpl>(*this, node.Index(), onnx_function_proto,
//                                                                GetModelLocalFunctions(), function_container_, logger_);
//    function_container_.emplace_back(std::move(func_ptr));
//    node.SetFunctionBody(*function_container_.back());
//  }
//  ORT_CATCH(const std::exception& e) {
//    LOGS(logger_, WARNING) << "Function body initialization failed for node '"
//                           << node.Name() << "' optype " << node.OpType()
//#ifndef ORT_NO_EXCEPTIONS
//                           << ". Error message " << e.what()
//#endif  // ORT_NO_EXCEPTIONS
//                           << ". Execution will fail if ORT does not have a specialized kernel for this op";
//    // Return without using this function op's expansion. No need to fail just yet.
//    // If ORT has a specialized kernel for this op then execution will proceed
//    return;
//  }
//}

Status Graph::VerifyInputAndInitializerNames() {
  std::unordered_set<std::string>& inputs_and_initializers = resolve_context_.inputs_and_initializers;

  for (auto* input : GetInputs()) {
    auto result = inputs_and_initializers.insert(input->Name());
    if (!result.second) {
      Status status(ONNXRUNTIME, FAIL,
                    "Error: Duplicate definition-site for (" + input->Name() + ").");
      return status;
    }
  }

  //// if no check here, then it is safe to remove it?
  //for (auto& initializer_pair : name_to_initial_tensor_) {
  //  GSL_SUPPRESS(es .84)
  //  inputs_and_initializers.insert(initializer_pair.first);
  //  // Initializers are expected to be included in inputs (according to ONNX spec).
  //  // onnxruntime relaxes this constraint. No duplicate-name check here.
  //}

  return Status::OK();
}

Status Graph::InitInputsInitializersOutputs() {
  resolve_context_.Clear();

  // clear the previous relationships, as we re-create them when resolving.
  // same applies to the implicit input defs as they are built from any subgraphs within this graph.
  for (auto& node : Nodes()) {
    node.MutableRelationships().Clear();
    node.MutableDefinitions().implicit_input_defs.clear();
  }

  // add the subgraph pointers to the resolve context.
  for (auto& node : Nodes()) {
    auto& subgraphs = node.MutableSubgraphs();
    if (!subgraphs.empty()) {
      resolve_context_.nodes_with_subgraphs.insert(&node);
    }
  }

  ORT_RETURN_IF_ERROR(SetGraphInputsOutputs());
  ORT_RETURN_IF_ERROR(VerifyInputAndInitializerNames());
  ORT_RETURN_IF_ERROR(VerifyNoDuplicateName());

  return Status::OK();
}

Status Graph::PerformTypeAndShapeInferencing(const ResolveOptions& options) {
  ORT_RETURN_IF_ERROR(TypeCheckInputsAndInitializers());

  // type/shape inferencing on the nodes is done recursively as we need subgraph outputs
  // to be applied to Node outputs for the node containing the subgraph.
  // Call path is
  // VerifyNodeAndOpMatch
  //   Iterates Nodes
  //     Runs ONNX type/shape inferencing for each Node
  //      - If it hits a node with a subgraph, InferenceContext::getGraphAttributeInferencer is called
  //        by the ONNX level type/shape inferencing, which updates the subgraph inputs using GraphInferencerImpl
  //      - GraphInferencerImpl::doInferencing calls PerformTypeShapeInferencing to execute type/shape inferencing
  //        for all nodes in the subgraph. This leads to recursively handling all subgraphs contained in the node.
  //      - once we finish processing the subgraph/s we apply resultant type/shape information to the outputs
  //        of the node that contains the subgraph.
  ORT_RETURN_IF_ERROR(VerifyNodeAndOpMatch(options));

  return Status::OK();
}

void Graph::FindAllSubgraphs(std::vector<Graph*>& subgraphs) {
  for (auto& node : Nodes()) {
    for (auto& subgraph : node.MutableSubgraphs()) {
      subgraphs.push_back(subgraph.get());
      subgraph->FindAllSubgraphs(subgraphs);
    }
  }
}

Status Graph::ForThisAndAllSubgraphs(const std::vector<Graph*>& subgraphs, std::function<Status(Graph&)> func) {
  auto status = func(*this);
  ORT_RETURN_IF_ERROR(status);

  for (auto& subgraph : subgraphs) {
    status = func(*subgraph);
    ORT_RETURN_IF_ERROR(status);
  }

  return status;
}

Status Graph::Resolve(const ResolveOptions& options) {
  if (parent_graph_) {
    // Resolve must start at the top level graph in-order to handle outer scope
    // connections correctly, so recurse up to that level to start
    return parent_graph_->Resolve(options);
  }

  // find all subgraphs including nested ones.
  std::vector<Graph*> all_subgraphs;
  FindAllSubgraphs(all_subgraphs);

  bool subgraphs_need_resolve = std::any_of(all_subgraphs.cbegin(), all_subgraphs.cend(),
                                            [](const Graph* graph) {
                                              return graph->GraphResolveNeeded();
                                            });

  if (!GraphResolveNeeded() && !subgraphs_need_resolve) {
    return Status::OK();
  }

  // init all graph/subgraphs. non-recursive.
  auto init_func = [](Graph& graph) { return graph.InitInputsInitializersOutputs(); };
  ORT_RETURN_IF_ERROR(ForThisAndAllSubgraphs(all_subgraphs, init_func));

  // recursively set the outer scope node args.
  ORT_RETURN_IF_ERROR(SetOuterScopeNodeArgs(resolve_context_.outer_scope_node_args));

  std::unordered_set<std::string> outer_scope_node_args_consumed;

  // recursively build connections between nodes in this graph and all subgraphs
  ORT_RETURN_IF_ERROR(BuildConnections(outer_scope_node_args_consumed));
  ORT_ENFORCE(outer_scope_node_args_consumed.empty(),
              "Shouldn't be possible to have NodeArgs that haven't been handled already.");

  // topological sort of this and any subgraphs is non-recursive
  auto topo_sort_func = [](Graph& graph) { return graph.PerformTopologicalSortAndCheckIsAcyclic(); };
  ORT_RETURN_IF_ERROR(ForThisAndAllSubgraphs(all_subgraphs, topo_sort_func));

  // type/shape validation and inferencing on this and any subgraphs
  // recurses into subgraphs via the ONNX checker, which descends into the GraphProto in node attributes
  // which define a subgraph.
  ORT_RETURN_IF_ERROR(PerformTypeAndShapeInferencing(options));

  // perform the final steps for this graph and all subgraphs
  auto finalize_func = [&options](Graph& graph) {
            graph.CleanUnusedInitializersAndNodeArgs(options.initializer_names_to_preserve);
            graph.GraphResolveNeeded(false);

            // if we are resolving immediately after loading from a GraphProto, we don't need to
            // do a proto sync
            if (options.no_proto_sync_required) {
                graph.GraphProtoSyncNeeded(false);
            }

            return Status::OK(); };

  ORT_RETURN_IF_ERROR(ForThisAndAllSubgraphs(all_subgraphs, finalize_func));

  ++num_resolves_;

  return Status::OK();
}

void Graph::SetName(const std::string& /*name*/) {
  //TODO!!!
  //save it as graph state?
  /*graph_proto_->set_name(name);*/
}

void Graph::SetDescription(const std::string& /*description*/) {
  //TODO!!!
  //save it as graph state?
  /*graph_proto_->set_doc_string(description);*/
}

#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
void Graph::AddInitializedTensor(const TensorProto& tensor) {
  graph_context_.AddInitializedTensor(tensor);
  SetGraphResolveNeeded();
  if (!is_loaded_from_model_file_ && GetNodeArg(tensor.name()) == nullptr) {
    // make sure there is a NodeArg for the initializer as SetGraphInputsOutputs may add it to the graph inputs.
    // the shape will be set to the correct value in TypeCheckInputsAndInitializers as we don't yet know whether there
    // will be a matching graph input for this initializer (we prefer shape info from the graph input).
    TypeProto t;
    t.mutable_tensor_type()->set_elem_type(tensor.data_type());

    ORT_IGNORE_RETURN_VALUE(GetOrCreateNodeArg(tensor.name(), &t));
  }
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

const std::string& Graph::Name() const noexcept {
  //TODO!!!
  //save it as graph state?
  return graph_context_.graph_proto_->name();
}

const std::string& Graph::Description() const noexcept {
  //TODO!!!
  //save it as graph state?
  return graph_context_.graph_proto_->doc_string();
}

const Path& Graph::ModelPath() const {
  return owning_model_.ModelPath();
}

template <typename T, typename TIter>
static void RemoveRepeatedFieldEntry(T& repeated_field, const TIter& entry_to_remove) {
  auto num_entries = repeated_field.size();
  if (num_entries > 1) {
    // swap the entry being deleted with the last one, and delete it.
    // we do this so we don't have to move all the entries past the one being deleted down one.
    auto slot = entry_to_remove - repeated_field.begin();
    auto last_entry = repeated_field.end() - 1;
    repeated_field.SwapElements(gsl::narrow<int>(slot), gsl::narrow<int>(num_entries - 1));
    repeated_field.erase(last_entry);
  } else {
    repeated_field.erase(entry_to_remove);
  }
}

bool Graph::IsInitializedTensor(const std::string& name) const {
  return graph_context_.IsInitializedTensor(name);
}

#if !defined(DISABLE_SPARSE_TENSORS)
bool Graph::IsSparseInitializer(const std::string& name) const {
  return graph_context_.IsSparseInitializer(name);
}
#endif

void Graph::RemoveInitializedTensor(const std::string& tensor_name) {
  graph_context_.RemoveInitializedTensor(tensor_name);
}

#if !defined(ORT_MINIMAL_BUILD)
Status Graph::ReplaceInitializedTensor(const ONNX_NAMESPACE::TensorProto& new_initializer) {
  return graph_context_.ReplaceInitializedTensor(new_initializer);
}
#endif  // !defined(ORT_MINIMAL_BUILD)

bool Graph::GetInitializedTensor(const std::string& tensor_name, const TensorProto*& value) const {
  auto& name_to_initial_tensor = graph_context_.GetAllInitializedTensors();
  auto iter = name_to_initial_tensor.find(tensor_name);
  if (name_to_initial_tensor.end() == iter) {
    value = nullptr;
    return false;
  }
  value = iter->second;
  return true;
}

void Graph::CleanAllInitializedTensors() noexcept {
  graph_context_.CleanAllInitializedTensors();
}

const ONNX_NAMESPACE::TensorProto* Graph::GetConstantInitializer(const std::string& initializer_name,
                                                                 bool check_outer_scope) const {
  const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
  if (GetInitializedTensor(initializer_name, initializer)) {
    if (CanOverrideInitializer()) {
      const auto& graph_inputs = GetInputsIncludingInitializers();
      bool is_constant = std::none_of(graph_inputs.cbegin(), graph_inputs.cend(),
                                      [&initializer_name](const NodeArg* input) {
                                        return input->Name() == initializer_name;
                                      });

      if (!is_constant) {
        initializer = nullptr;
      }
    }
  } else if (check_outer_scope && IsSubgraph()) {
    // make sure there's not a local value with the same name. if there is it shadows any initializer in outer scope.
    if (IsOuterScopeValue(initializer_name)) {
      initializer = parent_graph_->GetConstantInitializer(initializer_name, check_outer_scope);
    }
  }

  return initializer;
}

const ONNX_NAMESPACE::TensorProto* Graph::GetInitializer(const std::string& initializer_name,
                                                         bool check_outer_scope) const {
  const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
  if (GetInitializedTensor(initializer_name, initializer)) {
    return initializer;
  } else if (check_outer_scope && IsSubgraph()) {
    // make sure there's not a local value with the same name. if there is it shadows any initializer in outer scope.
    if (IsOuterScopeValue(initializer_name)) {
      initializer = parent_graph_->GetInitializer(initializer_name, check_outer_scope);
    }
  }

  return initializer;
}

#if !defined(ORT_MINIMAL_BUILD)
void Graph::AddValueInfo(const NodeArg* new_value_info) {
  graph_context_.GetMutableMainFunction()->AddValueInfo(new_value_info);
}

std::vector<NodeArg*> Graph::CreateNodeArgs(const google::protobuf::RepeatedPtrField<std::string>& names,
                                            const ArgNameToTypeMap& name_to_type_map) {
  return graph_context_.GetMutableMainFunction()->CreateNodeArgs(names, name_to_type_map);
}

Node& Graph::AddNode(const Node& other) {
  const auto& definitions = other.GetDefinitions();

  auto& new_node = AddNode(other.Name(), other.OpType(), other.Description(),
                           definitions.input_defs,
                           definitions.output_defs,
                           &other.GetAttributes(),
                           other.Domain());

  return new_node;
}

Node& Graph::AddNode(const NodeProto& node_proto,
                     const ArgNameToTypeMap& name_to_type_map) {
  return graph_context_.GetMutableMainFunction()->AddNode(node_proto, name_to_type_map);
}

static flatbuffers::Offset<flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>>
SaveInputsOutputsToOrtFormat(flatbuffers::FlatBufferBuilder& builder, const std::vector<const NodeArg*>& src) {
  std::vector<flatbuffers::Offset<flatbuffers::String>> vec(src.size());
  std::transform(src.cbegin(), src.cend(), vec.begin(),
                 [&builder](const NodeArg* entry) {
                   return builder.CreateSharedString(entry->Name());
                 });
  return builder.CreateVector(vec);
}

common::Status Graph::SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                      flatbuffers::Offset<fbs::Graph>& fbs_graph) const {
  auto& main_func_inputs = graph_context_.GetMainFunction().GetInputs();
  auto& main_func_outputs = graph_context_.GetMainFunction().GetOutputs();
  auto inputs = SaveInputsOutputsToOrtFormat(builder, main_func_inputs);
  auto outputs = SaveInputsOutputsToOrtFormat(builder, main_func_outputs);

  auto& sparse_tensor_names = graph_context_.GetSparseTensorNames();
  auto& name_to_initial_tensors = graph_context_.GetAllInitializedTensors();

#if !defined(DISABLE_SPARSE_TENSORS)
  std::vector<flatbuffers::Offset<fbs::SparseTensor>> sparse_initializers_data;
  sparse_initializers_data.reserve(sparse_tensor_names.size());
#endif
  const auto sparse_end = sparse_tensor_names.end();

  std::vector<flatbuffers::Offset<fbs::Tensor>> initializers_data;
#if !defined(DISABLE_SPARSE_TENSORS)
  assert(sparse_tensor_names.size() <= name_to_initial_tensors.size());
  initializers_data.reserve(name_to_initial_tensors.size() - sparse_tensor_names.size());
#else
  initializers_data.reserve(name_to_initial_tensor_.size());
#endif
  const auto& model_path = ModelPath();

  for (const auto& pair : name_to_initial_tensors) {
    if (sparse_tensor_names.find(pair.first) == sparse_end) {
      flatbuffers::Offset<fbs::Tensor> fbs_tensor;
      ORT_RETURN_IF_ERROR(
          fbs::utils::SaveInitializerOrtFormat(builder, *pair.second, model_path, fbs_tensor));
      initializers_data.push_back(fbs_tensor);
    }
#if !defined(DISABLE_SPARSE_TENSORS)
    else {
      SparseTensorProto sparse_initializer;
      ORT_RETURN_IF_ERROR(utils::DenseTensorToSparseTensorProto(*pair.second, model_path, sparse_initializer));
      flatbuffers::Offset<fbs::SparseTensor> fbs_sparse_tensor;
      ORT_RETURN_IF_ERROR(
          fbs::utils::SaveSparseInitializerOrtFormat(builder, sparse_initializer, model_path, fbs_sparse_tensor));
      sparse_initializers_data.push_back(fbs_sparse_tensor);
    }
#endif
  }
#if !defined(DISABLE_SPARSE_TENSORS)
  auto sparse_initializers = builder.CreateVector(sparse_initializers_data);
#endif
  auto initializers = builder.CreateVector(initializers_data);

  std::vector<flatbuffers::Offset<fbs::ValueInfo>> node_args_data;
  node_args_data.reserve(graph_context_.GetMainFunction().node_args_.size());
  for (const auto& pair : graph_context_.GetMainFunction().node_args_) {
    flatbuffers::Offset<fbs::ValueInfo> fbs_val_info;
    ORT_RETURN_IF_ERROR(
        fbs::utils::SaveValueInfoOrtFormat(builder, pair.second->ToProto(), fbs_val_info));
    node_args_data.push_back(fbs_val_info);
  }
  auto node_args = builder.CreateVector(node_args_data);

  std::vector<flatbuffers::Offset<fbs::Node>> nodes_vec;
  std::vector<flatbuffers::Offset<fbs::NodeEdge>> node_edges_vec;
  node_edges_vec.reserve(graph_context_.GetMainFunction().nodes_.size());
  for (const auto& node : graph_context_.GetMainFunction().nodes_) {
    if (node != nullptr) {
      flatbuffers::Offset<fbs::Node> fbs_node;
      ORT_RETURN_IF_ERROR(node->SaveToOrtFormat(builder, fbs_node));
      nodes_vec.push_back(fbs_node);
      node_edges_vec.push_back(node->SaveEdgesToOrtFormat(builder));
    }
  }
  auto nodes = builder.CreateVector(nodes_vec);
  auto node_edges = builder.CreateVector(node_edges_vec);

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
  auto runtime_optimizations = flatbuffers::Offset<fbs::RuntimeOptimizations>{};  // null value
  if (!RuntimeOptimizations().IsEmpty()) {
    flatbuffers::Offset<RuntimeOptimizationRecordContainer::FbsRuntimeOptimizationRecordContainer>
        runtime_optimization_records;
    ORT_RETURN_IF_ERROR(RuntimeOptimizations().SaveToOrtFormat(builder, runtime_optimization_records));
    runtime_optimizations = fbs::CreateRuntimeOptimizations(builder, runtime_optimization_records);
  }
#endif

  fbs::GraphBuilder gb(builder);
  gb.add_initializers(initializers);
  gb.add_node_args(node_args);
  gb.add_nodes(nodes);
  gb.add_max_node_index(gsl::narrow_cast<uint32_t>(graph_context_.GetMainFunction().nodes_.size()));
  gb.add_node_edges(node_edges);
  gb.add_inputs(inputs);
  gb.add_outputs(outputs);
#if !defined(DISABLE_SPARSE_TENSORS)
  gb.add_sparse_initializers(sparse_initializers);
#endif
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
  gb.add_runtime_optimizations(runtime_optimizations);
#endif
  fbs_graph = gb.Finish();
  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
std::string Graph::GenerateNodeArgName(const std::string& base_name) {
  std::string new_name = base_name;
  // Check if new_name has been used in as any of node_args_' names.
  // Check if new_name has been generated by this function.
  // If both are not, add new_name into name set and return the new_name
  // as the generated name. Otherwise, keep generating new names.
  while (graph_context_.GetMainFunction().node_args_.find(new_name) != graph_context_.GetMainFunction().node_args_.end() ||
         generated_node_arg_names_.find(new_name) != generated_node_arg_names_.end()) {
    std::ostringstream str;
    str << base_name << "_token_" << name_generator_++;
    new_name = str.str();
  }

  generated_node_arg_names_.insert(new_name);
  return new_name;
}

std::string Graph::GenerateNodeName(const std::string& base_name) {
  // Define name-checking function for node name.
  // Return true if the input name hasn't been used. Otherwise, return false.
  auto name_is_ok = [&](const std::string name) {
    for (auto it = graph_context_.GetMainFunction().nodes_.begin(); it != graph_context_.GetMainFunction().nodes_.end(); ++it) {
      if (*it == nullptr) {
        continue;
      }
      if (it->get()->Name() != name) {
        continue;
      }
      // Find a matched name so we cannot reuse the input name.
      return false;
    }

    if (generated_node_names_.find(name) != generated_node_names_.end()) {
      // Find a matched name so we cannot reuse the input name.
      return false;
    }

    // The input name can be reused.
    return true;
  };

  // Start with the input name.
  std::string new_name = base_name;

  while (!name_is_ok(new_name)) {
    std::ostringstream str;
    str << base_name << "_token_" << name_generator_++;
    new_name = str.str();
  }

  // Make sure this new_name is not going to be reused.
  generated_node_names_.insert(new_name);

  return new_name;
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
Node& Graph::AddNode(const std::string& name,
                     const std::string& op_type,
                     const std::string& description,
                     const std::vector<NodeArg*>& input_args,
                     const std::vector<NodeArg*>& output_args,
                     const NodeAttributes* attributes,
                     const std::string& domain) {
  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;
  inputs.resize(input_args.size());
  outputs.resize(output_args.size());
  int i = 0;
  for (auto input_arg : input_args) {
    inputs[i++] = &GetOrCreateNodeArg(input_arg->Name(), input_arg->TypeAsProto());
  }
  i = 0;
  for (auto output_arg : output_args) {
    outputs[i++] = &GetOrCreateNodeArg(output_arg->Name(), output_arg->TypeAsProto());
  }

  const gsl::not_null<Node*> node = AllocateNode();
  node->Init(name, op_type, description, inputs, outputs, attributes, domain);
  if (0 != op_type.compare(kNoOp)) {
    GraphProtoSyncNeeded(true);
  }

  return *node;
}

bool Graph::RemoveNode(NodeIndex p_index) {
  auto node = GetNode(p_index);
  if (nullptr == node) {
    return false;
  }

  // Node must be disconnected from any downstream nodes before removal
  ORT_ENFORCE(node->GetOutputEdgesCount() == 0, "Can't remove node ", node->Name(), " as it still has output edges.");

  // Remove all input edges.
  // Need to copy the edge info first so we can remove the real edges while iterating the copy of edge info.
  auto input_edges = node->GetRelationships().input_edges;

  for (auto& input_edge : input_edges) {
    RemoveEdge(input_edge.GetNode().Index(), p_index, input_edge.GetSrcArgIndex(), input_edge.GetDstArgIndex());
  }

  return ReleaseNode(p_index);
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)
bool Graph::AddControlEdge(NodeIndex src_node_index, NodeIndex dst_node_index) {
  return graph_context_.GetMutableMainFunction()->AddControlEdge(src_node_index, dst_node_index);
}

const ONNX_NAMESPACE::GraphProto& Graph::ToGraphProto() {
  auto* graph_proto = graph_context_.graph_proto_;
  if (!GraphProtoSyncNeeded()) {
    return *graph_proto;
  }

  // Nodes.
  ToGraphProtoInternal(*graph_proto);

  GraphProtoSyncNeeded(false);

  return *graph_proto;
}

ONNX_NAMESPACE::GraphProto Graph::ToGraphProto() const {
  auto* graph_proto = graph_context_.graph_proto_;
#if !defined(DISABLE_SPARSE_TENSORS)
  if (!GraphProtoSyncNeeded() && graph_context_.sparse_tensor_names_.empty()) {
    return *graph_proto;
  }
#else
  if (!GraphProtoSyncNeeded()) {
    return *graph_proto_;
  }
#endif

  GraphProto result;
  ToGraphProtoInternal(result);
  // Path of the owning model
  // This is used for constructing full path for external data
  // if it exists

#if !defined(DISABLE_SPARSE_TENSORS)
  const auto& model_path = ModelPath();
  // We want to make sure that sparse initializers do not appear
  // as dense duplicates within the initializers list.
  if (!graph_context_.sparse_tensor_names_.empty()) {
    const auto sparse_end = graph_context_.sparse_tensor_names_.end();
    auto* mutable_initializer = result.mutable_initializer();
    for (const auto& initializer : graph_context_.graph_proto_->initializer()) {
      if (sparse_end == graph_context_.sparse_tensor_names_.find(initializer.name())) {
        *mutable_initializer->Add() = initializer;
      } else {
        auto& sparse_initializer = *result.add_sparse_initializer();
        auto status = utils::DenseTensorToSparseTensorProto(initializer, model_path, sparse_initializer);
        ORT_ENFORCE(status.IsOK(), "Failed to convert dense initializer to sparse");
      }
    }
  } else {
    *result.mutable_initializer() = graph_context_.graph_proto_->initializer();
  }
#else
  *result.mutable_initializer() = graph_proto_->initializer();
#endif

  return result;
}

ONNX_NAMESPACE::GraphProto Graph::ToGraphProtoWithExternalInitializers(const std::string& external_file_name,
                                                                       size_t initializer_size_threshold) const {
  GraphProto result;
  ToGraphProtoInternal(result);

  std::ofstream external_stream(external_file_name, std::ofstream::out | std::ofstream::binary);
  ORT_ENFORCE(external_stream.is_open());
  int64_t external_offset = 0;

  // Add the initializers to the result graph.
#if !defined(DISABLE_SPARSE_TENSORS)
  const auto& model_path = ModelPath();
  const auto sparse_end = graph_context_.sparse_tensor_names_.end();
#endif

  for (const auto& initializer : graph_context_.graph_proto_->initializer()) {
#if !defined(DISABLE_SPARSE_TENSORS)
    if (sparse_end != graph_context_.sparse_tensor_names_.find(initializer.name())) {
      // Sparse tensors are added to the ONNX file.
      auto& sparse_initializer = *result.add_sparse_initializer();
      auto status = utils::DenseTensorToSparseTensorProto(initializer, model_path, sparse_initializer);
      ORT_ENFORCE(status.IsOK(), "Failed to convert dense initializer to sparse");
    } else {
#endif
      // Dense tensors larger than the threshold are added to the external file.
      TensorProto* output_proto = result.add_initializer();

      std::vector<uint8_t> raw_data;
      ORT_THROW_IF_ERROR(utils::UnpackInitializerData(initializer, Path(), raw_data));
      size_t tensor_bytes_size = raw_data.size();
      if (tensor_bytes_size < initializer_size_threshold) {
        *output_proto = initializer;
        continue;
      }

      for (size_t index = 0; index != tensor_bytes_size; ++index) {
        external_stream << raw_data[index];
      }

      output_proto->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);
      ONNX_NAMESPACE::StringStringEntryProto* location = output_proto->add_external_data();
      location->set_key("location");
      location->set_value(external_file_name);
      ONNX_NAMESPACE::StringStringEntryProto* offset = output_proto->add_external_data();
      offset->set_key("offset");
      offset->set_value(std::to_string(external_offset));
      ONNX_NAMESPACE::StringStringEntryProto* length = output_proto->add_external_data();
      length->set_key("length");
      length->set_value(std::to_string(tensor_bytes_size));

      output_proto->set_name(initializer.name());
      output_proto->set_data_type(initializer.data_type());
      for (int i = 0; i != initializer.dims_size(); ++i) {
        output_proto->add_dims(initializer.dims(i));
      }
      output_proto->set_doc_string(initializer.doc_string());

      external_offset += tensor_bytes_size;
#if !defined(DISABLE_SPARSE_TENSORS)
    }
#endif
  }

  return result;
}

void Graph::ToGraphProtoInternal(ONNX_NAMESPACE::GraphProto& graph_proto) const {
  auto* graph_proto_in_context = graph_context_.graph_proto_;
  graph_proto_in_context->clear_node();
  graph_proto_in_context->clear_input();
  graph_proto_in_context->clear_output();
  graph_proto_in_context->clear_value_info();
  graph_proto.set_name(Name());
  graph_proto.set_doc_string(Description());

  for (const auto* input_arg : GetInputsIncludingInitializers()) {
    *(graph_proto.mutable_input()->Add()) = input_arg->ToProto();
  }

  for (const auto* output_arg : GetOutputs()) {
    *(graph_proto.mutable_output()->Add()) = output_arg->ToProto();
  }

  for (const auto* value_info : graph_context_.GetMainFunction().GetValueInfo()) {
    *(graph_proto.mutable_value_info()->Add()) = value_info->ToProto();
  }

  // add the NodeArg info for outer scope NodeArgs so we capture the type information
  for (const auto& name : graph_context_.GetMainFunction().GetOuterScopeNodeArgNames()) {
    auto* node_arg = GetNodeArg(name);
    ORT_ENFORCE(node_arg, "Outer scope node arg name '" + name + "'was added but does not exist. ");
    *(graph_proto.mutable_value_info()->Add()) = node_arg->ToProto();
  }

  GraphViewer graph_viewer(*this);
  // Nodes must be sorted in Topological Order in the GraphProto per ONNX spec.
  for (auto& node_idx : graph_viewer.GetNodesInTopologicalOrder()) {
    const gsl::not_null<NodeProto*> node_proto{graph_proto.add_node()};
    const gsl::not_null<const Node*> p_node{GetNode(node_idx)};
    // we need to update any GraphProto attributes for subgraphs so that any changes made by things
    // such as the optimizers are captured. otherwise we can end up saving an invalid graph.
    p_node->ToProto(*node_proto, /* update_subgraphs */ true);
  }
}

void Graph::CleanUnusedInitializersAndNodeArgs(const std::unordered_set<std::string>* initializer_names_to_preserve) {
  // Node Args being used
  std::unordered_set<const NodeArg*> used_args;
  used_args.reserve(graph_context_.GetMainFunction().node_args_.size());

  // Node Args we want to preserved even not being used
  std::unordered_set<const NodeArg*> node_args_to_preserve;
  if (initializer_names_to_preserve) {
    node_args_to_preserve.reserve(initializer_names_to_preserve->size());
    for (const auto& initializer_name : *initializer_names_to_preserve) {
      const auto* initializer_node_arg = GetNodeArg(initializer_name);
      if (initializer_node_arg != nullptr) {
        ORT_IGNORE_RETURN_VALUE(node_args_to_preserve.insert(initializer_node_arg));
      }
    }
  }

  // anything that provides a required graph input (GetInputs), an optional graph input (GetOverridableInitializers)
  // or a graph output (GetOutputs) cannot be removed
  const auto& inputs = GetInputs();
  const auto& overridable_initializers = GetOverridableInitializers();
  const auto& outputs = GetOutputs();

  std::for_each(inputs.cbegin(), inputs.cend(), [&used_args](const NodeArg* input) {
    ORT_IGNORE_RETURN_VALUE(used_args.insert(input));
  });

  std::for_each(overridable_initializers.cbegin(), overridable_initializers.cend(),
                [&used_args](const NodeArg* input) {
                  ORT_IGNORE_RETURN_VALUE(used_args.insert(input));
                });

  std::for_each(outputs.cbegin(), outputs.cend(), [&used_args](const NodeArg* output) {
    ORT_IGNORE_RETURN_VALUE(used_args.insert(output));
  });

  for (const auto& node : Nodes()) {
    for (const auto* def : node.InputDefs()) {
      ORT_IGNORE_RETURN_VALUE(used_args.insert(def));
    }

    for (const auto* def : node.ImplicitInputDefs()) {
      ORT_IGNORE_RETURN_VALUE(used_args.insert(def));
    }
  }

  std::vector<std::string> erase_list;
  auto used_args_end = used_args.cend();
  for (const auto& pv : graph_context_.name_to_initial_tensor_) {
    const std::string& name = pv.first;
    const auto* initializer_node_arg = GetNodeArg(name);
    ORT_ENFORCE(initializer_node_arg != nullptr, "Cannot find NodeArgs for [", name, "]");
    if (used_args.find(initializer_node_arg) == used_args_end &&
        node_args_to_preserve.find(initializer_node_arg) == node_args_to_preserve.cend()) {
      // on the first call to Graph::Resolve we are removing unnecessary initializers that should be removed
      // from the model.
      // on later calls we are removing initializers that optimizations have made redundant.
      if (num_resolves_ == 0) {
        LOGS(logger_, WARNING) << "Removing initializer '"
                               << name << "'. It is not used by any node and should be removed from the model.";
      } else {
        LOGS(logger_, INFO) << "Removing initializer '" << name << "'. It is no longer used by any node.";
      }

      erase_list.push_back(name);
    }
  }

  std::for_each(erase_list.cbegin(), erase_list.cend(),
                [this](const std::string& name) {
                  RemoveInitializedTensor(name);

                  // handle edge case where the unused initializer has a matching graph input.
                  // this can only happen when initializers cannot be overridden via an optional graph input.
                  // (otherwise this initializer wouldn't be allowed to be removed due to it backing an optional
                  // graph input).
                  if (CanOverrideInitializer() == false) {
                    auto& proto_inputs = *graph_context_.graph_proto_->mutable_input();
                    auto i = std::find_if(proto_inputs.begin(), proto_inputs.end(),
                                          [&name](const ONNX_NAMESPACE::ValueInfoProto& input) {
                                            return input.name() == name;
                                          });

                    if (i != proto_inputs.end()) {
                      RemoveRepeatedFieldEntry(proto_inputs, i);
                    }

                    //TODO!!
                    //fix the erase later when move to Graph Resolver
                    auto inputs_including_initializers = graph_context_.GetMainFunction().GetInputs();
                    auto j = std::find_if(inputs_including_initializers.begin(), inputs_including_initializers.end(),
                                          [&name](const NodeArg* input) { return input->Name() == name; });

                    if (j != inputs_including_initializers.end()) {
                      inputs_including_initializers.erase(j);
                    }
                    graph_context_.GetMutableMainFunction()->SetInputs(inputs_including_initializers);
                  }
                });

  // Clear the unused NodeArgs
  // We also want to scan the output NodeArgs of each node
  // In case one output of a node is neither used as an input of another node nor an output of graph
  for (const auto& node : Nodes()) {
    for (const auto* def : node.OutputDefs()) {
      ORT_IGNORE_RETURN_VALUE(used_args.insert(def));
    }
  }

  // We also need to check the Outer Scope NodeArgs
  for (const auto& outer_scope_node_arg_name : graph_context_.GetMainFunction().GetOuterScopeNodeArgNames()) {
    const auto* outer_scope_node_arg = GetNodeArg(outer_scope_node_arg_name);
    ORT_ENFORCE(outer_scope_node_arg != nullptr, "Cannot find NodeArgs for [", outer_scope_node_arg_name, "]");
    ORT_IGNORE_RETURN_VALUE(node_args_to_preserve.insert(outer_scope_node_arg));
  }

  auto node_args_to_preserve_end = node_args_to_preserve.cend();
  for (auto it = graph_context_.GetMainFunction().node_args_.cbegin(), node_args_end = graph_context_.GetMainFunction().node_args_.cend(); it != node_args_end; /* no increment */) {
    auto current_entry = it++;
    const auto* current_node_arg = current_entry->second.get();
    const auto& node_arg_name = current_entry->first;
    // For some reason, we still have some code hold the raw pointer to the unused NodeArgs,
    // Remove only the NodeArgs with no type for now
    // TODO, investigate the issue when running using mpirun
    if (!node_arg_name.empty() && used_args.find(current_node_arg) == used_args_end &&
        node_args_to_preserve.find(current_node_arg) == node_args_to_preserve_end &&
        !current_node_arg->ToProto().has_type()) {
      LOGS(logger_, INFO) << "Removing NodeArg '" << node_arg_name << "'. It is no longer used by any node.";
      // Need to remove the NodeArg from both value_info_ and node_args_
      graph_context_.GetMutableMainFunction()->value_info_.erase(current_node_arg);
      graph_context_.GetMutableMainFunction()->node_args_.erase(current_entry);
    }
  }
}

#endif  // !defined(ORT_MINIMAL_BUILD)

void Graph::ComputeOverridableInitializers() {
  graph_overridable_initializers_.clear();
  if (CanOverrideInitializer()) {
    // graph_inputs_excluding_initializers_ and graph_inputs_including_initializers_
    // are inserted in the same order. So we walk and compute the difference.
    auto& graph_inputs_including_initializers = graph_context_.GetMainFunction().GetInputs();
    auto f_incl = graph_inputs_including_initializers.cbegin();
    const auto l_incl = graph_inputs_including_initializers.cend();
    auto f_excl = graph_inputs_excluding_initializers_.cbegin();
    const auto l_excl = graph_inputs_excluding_initializers_.cend();

    while (f_incl != l_incl) {
      // Equal means not an initializer
      if (f_excl != l_excl && *f_incl == *f_excl) {
        ++f_incl;
        ++f_excl;
        continue;
      }
      graph_overridable_initializers_.push_back(*f_incl);
      ++f_incl;
    }
  }
}

#if !defined(ORT_MINIMAL_BUILD)

GSL_SUPPRESS(es .84)  // warning about ignoring return value from insert(...)
Status Graph::SetGraphInputsOutputs() {
  // If loaded from a model file, we start from the specified inputs and
  // outputs set earlier by InitializeStateFromModelFileGraphProto().
  // Otherwise (!is_loaded_from_model_file_), we need to fix up the inputs and
  // may also need to infer inputs and outputs.
  // In either case, calls to SetInputs() or SetOutputs() may affect the actual
  // inputs and outputs.
  if (is_loaded_from_model_file_) return Status::OK();

  std::vector<const NodeArg*> func_inputs;
  std::vector<const NodeArg*> func_outputs;
  std::unordered_set<const NodeArg*> func_value_infos;

  std::unordered_map<std::string, size_t> output_name_to_node_arg_index;
  std::vector<const NodeArg*> output_node_args_in_order;

  auto& name_to_initial_tensors = graph_context_.GetAllInitializedTensors();

  // if something is coming from outer scope, consider it already added
  std::unordered_set<std::string> added_input_names{graph_context_.GetMainFunction().outer_scope_node_arg_names_};
  graph_inputs_excluding_initializers_.clear();
  if (graph_inputs_manually_set_) {
    // If we've set graph_inputs_including_initializers_ by calling SetInputs,
    // we copy its non-duplicate elements to graph_inputs_excluding_initializers_.
    // Later, we will erase initializers from graph_inputs_excluding_initializers_
    // if graph_inputs_manually_set_ is true.
    // In this way, we can ensure graph_inputs_excluding_initializers_ is the full
    // set of inputs less initializers, which could be a graph input used only
    // by a subgraph and thereby only an implicit input to a node, or a graph input
    // not used anywhere.
    // We also make sure graph_inputs_excluding_initializers_ list doesn't have any
    // duplicate names.
    std::unordered_set<std::string> existing_names;
    for (auto arg : graph_context_.GetMainFunction().GetInputs()) {
      const std::string& name = arg->Name();
      if (existing_names.count(name) == 0) {
        graph_inputs_excluding_initializers_.push_back(arg);
        existing_names.insert(name);
      }
    }
  }

  // Collect all nodes' outputs
  for (const auto& node : Nodes()) {
    for (const auto* output_def : node.OutputDefs()) {
      if (output_def->Exists()) {
        output_node_args_in_order.push_back(output_def);
        output_name_to_node_arg_index.insert({output_def->Name(), output_node_args_in_order.size() - 1});
      }
    }
  }

  // Init graph output args with copy of all node output args.
  auto graph_output_args = output_name_to_node_arg_index;
  for (const auto& node : Nodes()) {
    // Go thru all node's inputs.
    for (const auto* input_arg : node.InputDefs()) {
      if (!input_arg->Exists()) {
        // It's an optional input and does not exist in this case.
        continue;
      }

      auto output_arg_iter = output_name_to_node_arg_index.find(input_arg->Name());
      if (output_name_to_node_arg_index.end() == output_arg_iter) {
        // This input arg is not the output of another node so must come from either a graph input or an initializer.
        const std::string& name = input_arg->Name();

        if (added_input_names.end() == added_input_names.find(name)) {
          // This graph input has not been added into <graph_inputs_>.
          bool is_initializer = name_to_initial_tensors.find(name) != name_to_initial_tensors.end();

          if (!graph_inputs_manually_set_) {
            // if IR version < 4 all initializers must have a matching graph input
            // (even though the graph input is not allowed to override the initializer).
            // if IR version >= 4 initializers are not required to have a matching graph input.
            // any graph inputs that are to override initializers must be specified by calling SetInputs.
            if (!is_initializer || ir_version_ < 4) {
              func_inputs.push_back(input_arg);
            }
            if (!is_initializer) {
              // If input_arg is not of an initializer, we add it into graph_inputs_excluding_initializers_.
              graph_inputs_excluding_initializers_.push_back(input_arg);
            }
          } else {
            // graph_inputs_including_initializers_ has been manually populated by SetInputs.
            // Validation: the <input_arg> must be in graph inputs or initializers when it's manually set.
            if (!is_initializer) {
              const auto& inputs = graph_context_.GetMainFunction().GetInputs();
              bool in_inputs = std::find(inputs.begin(), inputs.end(), input_arg) != inputs.end();
              if (!in_inputs) {
                return Status(ONNXRUNTIME, FAIL,
                              name + " must be either specified in graph inputs or graph initializers.");
              }
            } else {
              // If arg_input is of an initializer, we remove it from graph_inputs_excluding_initializers_
              // whose initial content has both initializers and non-initializers.
              auto input_pos = std::find(graph_inputs_excluding_initializers_.begin(),
                                         graph_inputs_excluding_initializers_.end(),
                                         input_arg);
              if (input_pos != graph_inputs_excluding_initializers_.end()) {
                graph_inputs_excluding_initializers_.erase(input_pos);
              }
            }
          }

          added_input_names.insert(name);
        }
      } else if (graph_output_args.erase(output_arg_iter->first) >= 1) {
        // Remove the output arg name from graph outputs since it's
        // the input of this node, which we call it intermediate result
        // and store it in <m_valueinfo>.
        func_value_infos.insert(input_arg);
      }
    }
  }

  if (!graph_outputs_manually_set_) {
    // Set graph outputs in order.
    std::vector<size_t> graph_output_args_index;
    graph_output_args_index.reserve(graph_output_args.size());
    for (const auto& output_arg : graph_output_args) {
      graph_output_args_index.push_back(output_arg.second);
    }

    std::sort(graph_output_args_index.begin(), graph_output_args_index.end());
    for (auto& output_arg_index : graph_output_args_index) {
      func_outputs.push_back(output_node_args_in_order[output_arg_index]);
    }
  }

  if (!graph_inputs_manually_set_)
    graph_context_.GetMutableMainFunction()->SetInputs(func_inputs);
  if (!graph_outputs_manually_set_)
    graph_context_.GetMutableMainFunction()->SetOutputs(func_outputs);

  graph_context_.GetMutableMainFunction()->SetValueInfo(func_value_infos);

  ComputeOverridableInitializers();

  return Status::OK();
}

IOnnxRuntimeOpSchemaCollectionPtr Graph::GetSchemaRegistry() const {
  return schema_registry_;
}

bool Graph::SetOpSchemaFromRegistryForNode(Node& node) {
  if (node.op_ != nullptr) return true;

  node.op_ = [&]() -> const ONNX_NAMESPACE::OpSchema* {
    const auto domain_to_version_it = DomainToVersionMap().find(node.Domain());
    if (domain_to_version_it == DomainToVersionMap().end()) {
      return nullptr;
    }
    const auto max_inclusive_version = domain_to_version_it->second;
    return schema_registry_->GetSchema(node.OpType(), max_inclusive_version, node.Domain());
  }();

  if (node.op_) {
    node.since_version_ = node.op_->since_version();

    if (node.op_->Deprecated()) {
      node.op_ = nullptr;
    }
  }

  return node.op_ != nullptr;
}
#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
Status Graph::PopulateNodeArgToProducerConsumerLookupsFromNodes() {
  return graph_context_.GetMutableMainFunction()->PopulateNodeArgToProducerConsumerLookupsFromNodes();
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
// calling private ctor
GSL_SUPPRESS(r .11)
gsl::not_null<Node*> Graph::AllocateNode() {
  return graph_context_.GetMutableMainFunction()->AllocateNode();
}

// TODO: Does this need (and maybe AllocateNode) to be threadsafe so nodes_ and num_of_nodes_ managed more carefully?
bool Graph::ReleaseNode(NodeIndex index) {
  return graph_context_.GetMutableMainFunction()->ReleaseNode(index);
}

//TODO!!!
//Re-implemnt this
//Node& Graph::CreateFusedSubGraphNode(const IndexedSubGraph& sub_graph, const std::string& fused_node_name) {
//  const auto* func_meta_def = sub_graph.GetMetaDef();
//  ORT_ENFORCE(nullptr != func_meta_def);
//  std::vector<NodeArg*> input_args;
//  std::vector<NodeArg*> output_args;
//  std::unordered_map<std::string, int> input_indexes;
//  std::unordered_map<std::string, int> output_indexes;
//
//  int cur_idx = 0;
//  for (auto& arg_name : func_meta_def->inputs) {
//    input_args.push_back(GetNodeArg(arg_name));
//    input_indexes[arg_name] = cur_idx++;
//  }
//
//  cur_idx = 0;
//  for (auto& arg_name : func_meta_def->outputs) {
//    output_args.push_back(GetNodeArg(arg_name));
//    output_indexes[arg_name] = cur_idx++;
//  }
//
//  auto& fused_node = AddNode(fused_node_name,
//                             func_meta_def->name,
//                             func_meta_def->doc_string,
//                             input_args,
//                             output_args,
//                             &func_meta_def->attributes,
//                             func_meta_def->domain);
//
//  fused_node.SetNodeType(Node::Type::Fused);
//
//  return fused_node;
//}
//
//Node& Graph::BeginFuseSubGraph(const IndexedSubGraph& sub_graph, const std::string& fused_node_name) {
//  Node& node = CreateFusedSubGraphNode(sub_graph, fused_node_name);
//
//#if !defined(ORT_MINIMAL_BUILD)
//  // if this is a full build create the lightweight Function implementation that provides the schema so that
//  // kernel lookup works as per usual. in an extended minimal build we do the lookup via a hash so don't
//  // need to create the schema.
//  auto func = std::make_unique<ViewerFunctionImpl>(*this, sub_graph, logger_);
//  function_container_.push_back(std::move(func));
//  node.SetFunctionBody(*function_container_.back());
//#endif
//
//  return node;
//}
//
//void Graph::CancelFuseSubGraph(const Node& fused_node) {
//  auto node_idx = fused_node.Index();
//  if (!GetNode(node_idx))
//    return;
//
//  if (fused_node.NodeType() != Node::Type::Fused)
//    return;
//
//#if !defined(ORT_MINIMAL_BUILD)
//  // Remove the function body from function container
//  const auto* fused_node_func = fused_node.GetFunctionBody();
//  auto it = std::find_if(
//      function_container_.begin(), function_container_.end(),
//      [fused_node_func](const std::unique_ptr<onnxruntime::Function>& func) {
//        return func.get() == fused_node_func;
//      });
//  if (it != function_container_.end()) {
//    function_container_.erase(it);
//  }
//#endif
//
//  // Remove the fused_node
//  RemoveNode(node_idx);
//}
//
//void Graph::FinalizeFuseSubGraph(const IndexedSubGraph& sub_graph, Node& fused_node) {
//  const auto* func_meta_def = sub_graph.GetMetaDef();
//  ORT_ENFORCE(nullptr != func_meta_def);
//
//  std::unordered_map<std::string, int> input_indexes;
//  std::unordered_map<std::string, int> output_indexes;
//
//  int cur_idx = 0;
//  for (auto& arg_name : func_meta_def->inputs) {
//    input_indexes[arg_name] = cur_idx++;
//  }
//
//  cur_idx = 0;
//  for (auto& arg_name : func_meta_def->outputs) {
//    output_indexes[arg_name] = cur_idx++;
//  }
//
//  auto new_node_idx = fused_node.Index();
//
//  // Remove nodes that were fused
//  for (auto node_index : sub_graph.nodes) {
//    auto node = GetNode(node_index);
//    if (nullptr == node) {
//      continue;
//    }
//
//    // move any applicable input edges to the new node. remove all others
//    auto input_edges = node->GetRelationships().input_edges;  // copy so RemoveEdge doesn't invalidate iterator
//    for (const auto& input_edge : input_edges) {
//      const auto& producer = input_edge.GetNode();
//      auto producer_idx = producer.Index();
//      auto src_idx = input_edge.GetSrcArgIndex();
//      auto dst_idx = input_edge.GetDstArgIndex();
//
//      // if this input is an input of the fused node add an edge for that
//      if (dst_idx < (int)node->InputDefs().size()) {
//        auto it = input_indexes.find(node->InputDefs()[dst_idx]->Name());
//        if (it != input_indexes.cend()) {
//          AddEdge(producer_idx, new_node_idx, src_idx, it->second);
//        }
//      } else {
//        int dst_implicit_input_idx = dst_idx - (int)node->InputDefs().size();
//        ORT_ENFORCE(dst_implicit_input_idx < (int)node->ImplicitInputDefs().size());
//        auto it = input_indexes.find(node->ImplicitInputDefs()[dst_implicit_input_idx]->Name());
//        if (it != input_indexes.cend()) {
//          AddEdge(producer_idx, new_node_idx, src_idx, it->second);
//        }
//      }
//      RemoveEdge(producer_idx, node_index, src_idx, dst_idx);
//    }
//
//    // move any applicable output edges to the new node
//    auto output_edges = node->GetRelationships().output_edges;  // copy so RemoveEdge doesn't invalidate iterator
//    for (const auto& output_edge : output_edges) {
//      const auto& consumer = output_edge.GetNode();
//      auto consumer_idx = consumer.Index();
//      auto src_idx = output_edge.GetSrcArgIndex();
//      auto dst_idx = output_edge.GetDstArgIndex();
//
//      // if this output is an output of the fused node add an edge for that
//      auto it = output_indexes.find(node->OutputDefs()[src_idx]->Name());
//      if (it != output_indexes.cend()) {
//        AddEdge(new_node_idx, consumer_idx, it->second, dst_idx);
//      }
//
//      RemoveEdge(node_index, consumer_idx, src_idx, dst_idx);
//    }
//
//    RemoveNode(node_index);
//  }
//}

#endif  // #if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)
//TODO!!!
//Reimplement this
Node& Graph::FuseSubGraph(const IndexedSubGraph& sub_graph,
                          const std::string& fused_node_name) {
  Node& fused_node = CreateFusedSubGraphNode(sub_graph, fused_node_name);

  // create Function before we remove nodes
  function_container_.emplace_back(MakeFunction(*this, sub_graph, logger_));
  fused_node.SetFunctionBody(*function_container_.back());

  // remove nodes and update edges
  FinalizeFuseSubGraph(sub_graph, fused_node);

  return fused_node;
}
//
//Status Graph::InlineFunction(Node& node) {
//  // Remove the function node, add the nodes in function's subgraph into the
//  // main graph.
//  const Graph& subgraph = node.GetFunctionBody()->Body();
//  auto output_edges = node.GetRelationships().output_edges;
//  for (const auto& output_edge : output_edges) {
//    RemoveEdge(node.Index(), output_edge.GetNode().Index(), output_edge.GetSrcArgIndex(), output_edge.GetDstArgIndex());
//  }
//
//  // Map of function input outputs to nodes input/outputs
//  std::unordered_map<std::string, NodeArg*> remap_input_output;
//  // Set of node input output names as these names need to be preserved during inlining
//  std::unordered_set<std::string> func_input_output_names;
//
//  for (size_t i = 0; i < subgraph.GetInputsIncludingInitializers().size(); ++i) {
//    auto* input = subgraph.GetInputsIncludingInitializers()[i];
//    if (input->Name() != node.MutableInputDefs()[i]->Name()) {
//      remap_input_output[input->Name()] = node.MutableInputDefs()[i];
//    }
//    func_input_output_names.insert(input->Name());
//  }
//
//  for (size_t i = 0; i < subgraph.GetOutputs().size(); ++i) {
//    auto* output = subgraph.GetOutputs()[i];
//    if (output->Name() != node.MutableOutputDefs()[i]->Name()) {
//      remap_input_output[output->Name()] = node.MutableOutputDefs()[i];
//    }
//    func_input_output_names.insert(output->Name());
//  }
//
//  // create a uniq_identifier to append to every node name and intermediate input\outputs
//  // to make sure there are no unintended duplicates
//  std::stringstream ss;
//  ss << static_cast<const void*>(&node);
//  auto uniq_identifier = ss.str();
//
//  RemoveNode(node.Index());
//
//  const auto& model_path = ModelPath();
//  for (const auto& subgraph_node : subgraph.Nodes()) {
//    if (subgraph_node.OpType() == kConstant) {
//      // Copy constant nodes _value to name_to_initial_tensor_
//      ONNX_NAMESPACE::NodeProto subgraph_node_proto{};
//      subgraph_node.ToProto(subgraph_node_proto);
//      const gsl::not_null<TensorProto*> tensor{graph_proto_->add_initializer()};
//      ORT_RETURN_IF_ERROR(utils::ConstantNodeProtoToTensorProto(subgraph_node_proto, model_path, *tensor, subgraph_node_proto.output(0) + uniq_identifier));
//      name_to_initial_tensor_[tensor->name()] = tensor;
//    } else {
//      std::vector<NodeArg*> inputs, outputs;
//      for (auto* input : subgraph_node.InputDefs()) {
//        if (func_input_output_names.find(input->Name()) != func_input_output_names.end()) {
//          auto it = remap_input_output.find(input->Name());
//          if (it != remap_input_output.end()) {
//            // This is a function input/output and needs to be remapped to node input for correctness
//            inputs.push_back(it->second);
//          } else {
//            // This is a function input/output so preserve the existing name
//            auto& n_input = GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
//            inputs.push_back(&n_input);
//          }
//        } else {
//          // This is an intermediate input. Add a unique identifier as suffix to make sure
//          // there is no name collision with names in parent graph
//          auto& n_input = GetOrCreateNodeArg(input->Name() + uniq_identifier, input->TypeAsProto());
//          inputs.push_back(&n_input);
//        }
//      }
//      for (auto* output : subgraph_node.OutputDefs()) {
//        if (func_input_output_names.find(output->Name()) != func_input_output_names.end()) {
//          auto it = remap_input_output.find(output->Name());
//          if (it != remap_input_output.end()) {
//            outputs.push_back(it->second);
//          } else {
//            auto& n_output = GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
//            outputs.push_back(&n_output);
//          }
//        } else {
//          auto& n_output = GetOrCreateNodeArg(output->Name() + uniq_identifier, output->TypeAsProto());
//          outputs.push_back(&n_output);
//        }
//      }
//
//      auto& new_node = AddNode(subgraph_node.Name() + uniq_identifier, subgraph_node.OpType(), subgraph_node.Description(),
//                               inputs,
//                               outputs,
//                               &subgraph_node.GetAttributes(),
//                               subgraph_node.Domain());
//
//      // If this node has an initialized function body add it to the new node so that reinitialization is not required.
//      if (subgraph_node.GetFunctionBody() != nullptr) {
//        new_node.SetFunctionBody(*(const_cast<onnxruntime::Function*>(subgraph_node.GetFunctionBody())));
//      }
//    }
//  }
//
//  ORT_RETURN_IF_ERROR(this->Resolve());
//  return Status::OK();
//}

void Graph::SetInputs(const std::vector<const NodeArg*>& inputs) {
  if (is_loaded_from_model_file_) {
    // graph loaded from model file
    graph_context_.GetMutableMainFunction()->SetInputs(inputs);
    graph_inputs_excluding_initializers_.clear();
    for (const auto* input : inputs) {
      ORT_ENFORCE(input->Exists(), "Input to set must exist.");
      auto& initializers = graph_context_.GetAllInitializedTensors();
      if (initializers.find(input->Name()) == initializers.end()) {
        graph_inputs_excluding_initializers_.emplace_back(input);
      }
    }

    ComputeOverridableInitializers();
  } else {
    // creating graph from scratch
    // rely on SetGraphInputsOutputs() to fix up graph_inputs_excluding_initializers_
    graph_context_.GetMutableMainFunction()->SetInputs(inputs);
  }

  graph_inputs_manually_set_ = true;
  GraphProtoSyncNeeded(true);
  GraphResolveNeeded(true);
}

void Graph::SetOutputs(const std::vector<const NodeArg*>& outputs) {
  graph_context_.GetMutableMainFunction()->SetOutputs(outputs);
  graph_outputs_manually_set_ = true;
  GraphProtoSyncNeeded(true);
  GraphResolveNeeded(true);
}

#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
void Graph::SetNodeArgType(NodeArg& arg, const ONNX_NAMESPACE::TypeProto& type_proto) {
  graph_context_.GetMutableMainFunction()->SetNodeArgType(arg, type_proto);
  GraphResolveNeeded(true);
}
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)

Graph::~Graph() {
  // nothing to do, but we put it here so we don't need to fully define types in Graph that are held in unique_ptr
  // such as   std::unique_ptr<FunctionContainer> function_container_;
}

#if !defined(ORT_MINIMAL_BUILD)
std::ostream& operator<<(std::ostream& out, const NodeArg& node_arg) {
  out << "\"" << node_arg.Name() << "\"";
  if (node_arg.Type()) {
    out << ": " << *node_arg.Type();
  }
  return out;
}

std::ostream& operator<<(std::ostream& out, const Node& node) {
  out << "(\"" << node.Name() << "\""
      << ", "
      << node.OpType()
      << ", "
      // Use quote so default ONNX domain is shown as ""
      // rather than misleading empty string.
      << "\"" << node.Domain() << "\""
      << ", "
      << node.SinceVersion()
      << ") : (";
  for (const auto* x : node.InputDefs()) {
    if (x->Exists()) {
      out << *x << ",";
    } else {
      // Print missing (or optional) inputs
      // because operator schema uses positional
      // arguments in ONNX.
      out << "\"\""
          << ",";
    }
  }
  out << ") -> (";
  for (const auto* x : node.OutputDefs()) {
    if (x->Exists()) {
      out << *x << ",";
    } else {
      // Print missing (or optional) outputs
      // because operator schema uses positional
      // arguments in ONNX.
      out << "\"\""
          << ",";
    }
  }
  out << ") ";
  return out;
}

std::ostream& operator<<(std::ostream& out, const Graph& graph) {
  out << "Inputs:\n";
  for (const auto* x : graph.GetInputs()) {
    // Unlike we print missing input and output for operator, we don't
    // print missing input for graph because they are not helpful (we
    // don't have a fixed schema for graph to match arguments).
    if (x) {
      out << "   " << *x << "\n";
    }
  }
  out << "Nodes:\n";
  for (const auto& node : graph.Nodes()) {
    out << "   " << node << "\n";
  }
  out << "Outputs:\n";
  for (const auto* x : graph.GetOutputs()) {
    // Similar to graph input, missing graph output is not printed.
    if (x) {
      out << "   " << *x << "\n";
    }
  }
  return out;
}
#endif  // !defined(ORT_MINIMAL_BUILD)

Status Graph::LoadFromOrtFormat(const onnxruntime::fbs::Graph& fbs_graph,
                                const Model& owning_model,
                                const std::unordered_map<std::string, int>& domain_to_version,
#if !defined(ORT_MINIMAL_BUILD)
                                IOnnxRuntimeOpSchemaCollectionPtr schema_registry,
#endif
                                const logging::Logger& logger, std::unique_ptr<Graph>& graph) {
  graph = std::make_unique<Graph>(owning_model, domain_to_version,
#if !defined(ORT_MINIMAL_BUILD)
                                  schema_registry,
#endif
                                  nullptr, nullptr, logger);

  ORT_RETURN_IF_ERROR(graph->LoadFromOrtFormat(fbs_graph));

#if !defined(ORT_MINIMAL_BUILD)
  // in a full build we need to run Resolve to fully populate ResolveContext and Node::op_,
  // which will allow optimizers to run or non-ORT EPs to take nodes.
  // TODO: We could decide that an ORT model is load only even in a full build,
  // and in InferenceSession::Initialize skip partitioning and running optimizers.
  graph->SetGraphResolveNeeded();
  ORT_RETURN_IF_ERROR(graph->Resolve());
#endif

  return Status::OK();
}

Status Graph::LoadFromOrtFormat(const onnxruntime::fbs::Graph& fbs_graph,
                                Graph& parent_graph, const Node& parent_node,
                                const logging::Logger& logger, std::unique_ptr<Graph>& graph) {
  graph = std::make_unique<Graph>(parent_graph.owning_model_,
                                  parent_graph.domain_to_version_,
#if !defined(ORT_MINIMAL_BUILD)
                                  parent_graph.schema_registry_,
#endif
                                  &parent_graph, &parent_node,
                                  logger);

  return graph->LoadFromOrtFormat(fbs_graph);
}

Graph::Graph(const Model& owning_model,
             const std::unordered_map<std::string, int>& domain_to_version,
#if !defined(ORT_MINIMAL_BUILD)
             IOnnxRuntimeOpSchemaCollectionPtr schema_registry,
#endif
             Graph* parent_graph, const Node* parent_node,
             const logging::Logger& logger)
    : owning_model_(owning_model),
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
      runtime_optimizations_ptr_(std::make_unique<RuntimeOptimizationRecordContainer>()),
      runtime_optimizations_(*runtime_optimizations_ptr_),
#endif
#if !defined(ORT_MINIMAL_BUILD)
      schema_registry_(schema_registry),
#endif
      domain_to_version_(domain_to_version),
      ir_version_(owning_model.IrVersion()),
      parent_graph_(parent_graph),
      parent_node_(parent_node),
      logger_(logger),
      is_loaded_from_model_file_(true),
      graph_context_(&deserialized_proto_data_, ModelPath(), this, ir_version_, IsSubgraph(), logger_) {  // true as the Graph isn't manually constructed from scratch
}

common::Status Graph::LoadFromOrtFormat(const onnxruntime::fbs::Graph& fbs_graph) {
  // We deserialize the graph from ORT format in the following order:
  // 1. Deserialize the initializers and sparse initializers. Convert sparse to dense.
  // 2. Deserialize the NodeArgs
  //        We need all NodeArg instances to exist when deserializing Nodes to setup the Node's
  //        inputs/outputs/implicit inputs which are collections of NodeArg*.
  // 3. Deserialize the Nodes
  // 4. Deserialize the NodeEdges
  //        We need all the Node instances to exist as the EdgeEnd has a Node* for the other end of the edge
  // 5. Deserialize the Inputs/Outputs/outer_scope_node_args
  // 6. Deserialize the runtime optimizations, if enabled

  // Initializers
  auto fbs_initializers = fbs_graph.initializers();
#if !defined(DISABLE_SPARSE_TENSORS)
  auto fbs_sparse_initializers = fbs_graph.sparse_initializers();
  flatbuffers::uoffset_t map_size = (fbs_initializers != nullptr ? fbs_initializers->size() : 0U) +
                                    (fbs_sparse_initializers != nullptr ? fbs_sparse_initializers->size() : 0U);
#else
  flatbuffers::uoffset_t map_size = (fbs_initializers != nullptr ? fbs_initializers->size() : 0U);
#endif

  if (map_size > 0) {
    graph_context_.name_to_initial_tensor_.reserve(map_size);
  }

  if (fbs_initializers) {
    for (const auto* fbs_tensor : *fbs_initializers) {
      ORT_RETURN_IF(nullptr == fbs_tensor, "Initializer tensor is missing. Invalid ORT format model.");
      TensorProto* initializer = deserialized_proto_data_.add_initializer();
      ORT_RETURN_IF_ERROR(fbs::utils::LoadInitializerOrtFormat(*fbs_tensor, *initializer));
      auto p = graph_context_.name_to_initial_tensor_.emplace(initializer->name(), initializer);
      if (!p.second) {
        LOGS(logger_, WARNING) << "Duplicate initializer (dense or ConstantNode): '" << initializer->name()
                               << "' the model will use the latest encountered initializer"
                               << ". Please, fix your model.";
        p.first->second = initializer;
      }
    }
  }

#if !defined(DISABLE_SPARSE_TENSORS)
  if (fbs_sparse_initializers) {
    graph_context_.sparse_tensor_names_.reserve(fbs_sparse_initializers->size());
    const auto& model_path = ModelPath();

    for (const auto* fbs_sparse_tensor : *fbs_sparse_initializers) {
      ORT_RETURN_IF(nullptr == fbs_sparse_tensor, "Sparse Initializer tensor is missing. Invalid ORT format model.");
      SparseTensorProto sparse_initializer;
      ORT_RETURN_IF_ERROR(fbs::utils::LoadSparseInitializerOrtFormat(*fbs_sparse_tensor, sparse_initializer));
      TensorProto& initializer = *deserialized_proto_data_.add_initializer();
      ORT_RETURN_IF_ERROR(utils::SparseTensorProtoToDenseTensorProto(sparse_initializer, model_path, initializer));
      auto p = graph_context_.name_to_initial_tensor_.emplace(initializer.name(), &initializer);
      if (!p.second) {
        LOGS(logger_, WARNING) << "Duplicate initializer (dense, sparse or ConstantNode): '" << initializer.name()
                               << "' the model will use the latest encountered initializer"
                               << ". Please, fix your model.";
        p.first->second = &initializer;
      }
      graph_context_.sparse_tensor_names_.emplace(initializer.name());
    }
  }
#endif

  auto* main_func = graph_context_.GetMutableMainFunction();
  // NodeArgs
  auto fbs_node_args = fbs_graph.node_args();
  if (fbs_node_args) {
    main_func->node_args_.reserve(fbs_node_args->size());
    for (const auto* fbs_value_info : *fbs_node_args) {
      ORT_RETURN_IF(nullptr == fbs_value_info, "NodeArg is missing. Invalid ORT format model.");
      NodeArgInfo node_arg_info;
      ORT_RETURN_IF_ERROR(fbs::utils::LoadValueInfoOrtFormat(*fbs_value_info, node_arg_info));
      main_func->node_args_[fbs_value_info->name()->str()] = std::make_unique<NodeArg>(std::move(node_arg_info));
    }
  }

  // Nodes
  //
  // Since we access a node using its index, we need to have nodes_ with size max_node_index to avoid
  // out of bounds access.
  main_func->nodes_.resize(fbs_graph.max_node_index());
  auto* fbs_nodes = fbs_graph.nodes();

  // It is possible to have no nodes in the model. Most likely scenario is the subgraph of an If Node
  // where the subgraph returns a Constant node. The Constant node will be lifted to an initializer by ORT
  // (prior to serializing to ORT format), leaving a valid Graph that contains no nodes.
  if (fbs_nodes != nullptr) {
    for (const auto* fbs_node : *fbs_nodes) {
      ORT_RETURN_IF(nullptr == fbs_node, "Node is missing. Invalid ORT format model.");
      std::unique_ptr<Node> node;
      ORT_RETURN_IF_ERROR(Node::LoadFromOrtFormat(*fbs_node, *this, logger_, node));
      ORT_RETURN_IF(node->Index() >= fbs_graph.max_node_index(), "Node index is out of range");
      main_func->nodes_[node->Index()] = std::move(node);
      ++main_func->num_of_nodes_;
    }
  }

  // NodeEdges
  auto* fbs_node_edges = fbs_graph.node_edges();
  if (fbs_node_edges != nullptr) {
    for (const auto* fbs_node_edge : *fbs_node_edges) {
      ORT_RETURN_IF(nullptr == fbs_node_edge, "NodeEdge is missing. Invalid ORT format model.");
      ORT_RETURN_IF(fbs_node_edge->node_index() >= fbs_graph.max_node_index(), "Node index is out of range");
      ORT_RETURN_IF_ERROR(graph_context_.GetMutableMainFunction()->nodes_[fbs_node_edge->node_index()]->LoadEdgesFromOrtFormat(*fbs_node_edge, *this));
    }
  }

  // Inputs/Outputs/outer_scope_node_args
  auto add_node_args = [&](const flatbuffers::Vector<flatbuffers::Offset<flatbuffers::String>>* fbs_node_args,
                           std::vector<const NodeArg*>& node_args) -> Status {
    if (fbs_node_args != nullptr) {
      node_args.reserve(fbs_node_args->size());
      for (const auto* fbs_node_arg_name : *fbs_node_args) {
        ORT_RETURN_IF(nullptr == fbs_node_arg_name, "NodeArg Name is missing. Invalid ORT format model.");
        gsl::not_null<NodeArg*> node_arg = GetNodeArg(fbs_node_arg_name->str());
        node_args.push_back(node_arg);
      }
    }
    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(add_node_args(fbs_graph.inputs(), main_func->graph_inputs_));
  for (const auto* input_arg : main_func->GetInputs()) {
    if (graph_context_.name_to_initial_tensor_.count(input_arg->Name()) == 0) {
      graph_inputs_excluding_initializers_.push_back(input_arg);
    }
  }

  ComputeOverridableInitializers();

  ORT_RETURN_IF_ERROR(add_node_args(fbs_graph.outputs(), main_func->graph_outputs_));

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)
  // populate NodeArg lookups after loading Nodes and NodeArgs
  ORT_RETURN_IF_ERROR(PopulateNodeArgToProducerConsumerLookupsFromNodes());

  // runtime optimizations
  if (const auto* fbs_runtime_optimizations = fbs_graph.runtime_optimizations()) {
    if (const auto* fbs_runtime_optimization_records = fbs_runtime_optimizations->records()) {
      ORT_RETURN_IF_ERROR(MutableRuntimeOptimizations().LoadFromOrtFormat(*fbs_runtime_optimization_records));
    }
  }
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_ENABLE_RUNTIME_OPTIMIZATION_IN_MINIMAL_BUILD)

  return Status::OK();
}

}  // namespace onnxruntime
