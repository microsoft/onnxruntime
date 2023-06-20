// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <queue>

#include "core/graph/graph_utils.h"
#include "orttraining/core/framework/distributed_run_context.h"

using namespace onnxruntime::common;
using namespace onnxruntime::graph_utils;

namespace onnxruntime {
namespace test {

using namespace training;

NodeArg& CreateTypedNodeArg(Graph& graph, onnx::TensorProto_DataType type, const std::string& name) {
  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(type);
  auto actual_name = graph.GenerateNodeArgName(name);
  auto& node_arg = graph.GetOrCreateNodeArg(actual_name, &type_proto);
  return node_arg;
}

void AddNewNodeArg(Graph& graph,
                   const std::string& op_name,
                   onnx::TensorProto_DataType type,
                   std::vector<NodeArg*>& new_node_args,
                   std::vector<std::string>& new_names) {
  auto& new_node_arg = CreateTypedNodeArg(graph, type, op_name);
  new_names.push_back(new_node_arg.Name());
  new_node_args.push_back(&new_node_arg);
}

Status ResolveForTraining(Graph& graph, const std::unordered_set<std::string>& weights_to_train) {
  Graph::ResolveOptions options;
  // Reserve the training weights. In mixed precision case, without this field,
  // the original fp32 initializers could be removed due to not being used
  // at this point. But we still need to preserve them because later when optimizer is
  // is constructed, the isolated fp32 initializers will be inputs for optimizer.
  options.initializer_names_to_preserve = &weights_to_train;

  return graph.Resolve(options);
}

Status SetInputsOutputsAndResolve(Graph& graph,
                                  const std::unordered_set<std::string>& weights_to_train,
                                  const std::vector<std::string>& new_input_names,
                                  const std::vector<std::string>& new_output_names) {
  auto fill_node_args = [&](const Graph& graph,
                            const std::vector<const NodeArg*>& existed_node_args,
                            const std::vector<std::string>& new_node_arg_names,
                            std::vector<const NodeArg*>& merged_node_args) {
    merged_node_args.insert(merged_node_args.end(), existed_node_args.begin(), existed_node_args.end());
    for (auto& name : new_node_arg_names) {
      merged_node_args.push_back(graph.GetNodeArg(name));
    }
  };

  const std::vector<const NodeArg*>& graph_inputs = graph.GetInputsIncludingInitializers();
  std::vector<const NodeArg*> inputs_args_sets;
  inputs_args_sets.reserve(graph_inputs.size() + new_input_names.size());
  fill_node_args(graph, graph_inputs, new_input_names, inputs_args_sets);

  const std::vector<const NodeArg*>& graph_outputs = graph.GetOutputs();
  std::vector<const NodeArg*> outputs_args_sets;
  outputs_args_sets.reserve(graph_outputs.size() + new_output_names.size());
  fill_node_args(graph, graph_outputs, new_output_names, outputs_args_sets);

  graph.SetInputs(inputs_args_sets);
  graph.SetOutputs(outputs_args_sets);
  graph.SetGraphResolveNeeded();
  graph.SetGraphProtoSyncNeeded();

  return ResolveForTraining(graph, weights_to_train);
}

// This function is used when you want to create a scalar constant in a graph.
// It may create a NodeArg so that other Node can references its value.
// It also cerates an initializer to store its value.
template <typename T>
Status AddNewScalarNodeArgAndInitializer(Graph& graph,
                                         const std::string& op_name,
                                         onnx::TensorProto_DataType type,
                                         T data,
                                         std::vector<NodeArg*>& new_node_args,
                                         std::vector<std::string>& new_names) {
  AddNewNodeArg(graph, op_name, type, new_node_args, new_names);

  ONNX_NAMESPACE::TensorProto proto_data;
  proto_data.set_name(new_names.back());
  proto_data.set_data_type(type);

  switch (type) {
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      proto_data.add_int32_data(static_cast<int32_t>(data));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      proto_data.add_int64_data(static_cast<int64_t>(data));
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "pipeline partition unsupported 'type' value: ", type);
  }
  graph.AddInitializedTensor(proto_data);
  return Status::OK();
}

// Given a node, this function finds all its connected nodes (consumer nodes and producer nodes) and
// connected inputs and outputs in the given graph, then adds them to the containers passed in.
Status FindAllConnectedNodes(Graph& graph,
                             Node* node,
                             std::vector<Node*>& connected_nodes,
                             std::set<NodeArg*>& connected_inputs,
                             std::set<NodeArg*>& connected_outputs) {
  assert(node);
  ORT_RETURN_IF_ERROR(node->ForEachMutableWithIndex(
      node->MutableInputDefs(),
      [&](NodeArg& node_arg, size_t /*index*/) {
        if (graph.IsInputsIncludingInitializers(&node_arg) || graph.IsInitializedTensor(node_arg.Name())) {
          connected_inputs.insert(&node_arg);
        } else {
          Node* producer_node = graph.GetMutableProducerNode(node_arg.Name());
          if (producer_node == nullptr) {
            // got nullptr as producer node. This could be because the input is a constant op which will be optimized
            // away. Print out this information and continue.
            // TODO: re-visit the different cases to see if there are other situations aside from constant ops.
            LOGS_DEFAULT(WARNING) << "Cannot find producer node for node_arg: " << node_arg.Name() << ". Skipping this node.";
          } else {
            connected_nodes.push_back(producer_node);
          }
        }
        return Status::OK();
      }));

  ORT_RETURN_IF_ERROR(node->ForEachMutableWithIndex(
      node->MutableOutputDefs(),
      [&](NodeArg& node_arg, size_t /*index*/) {
        if (!graph.IsOutput(&node_arg)) {
          std::vector<Node*> consumer_nodes = graph.GetMutableConsumerNodes(node_arg.Name());
          connected_nodes.insert(std::end(connected_nodes), consumer_nodes.begin(), consumer_nodes.end());

        } else {
          connected_outputs.insert(&node_arg);
        }
        return Status::OK();
      }));
  return Status::OK();
}

// PipelineStageNodeGroup groups nodes that share the same input initializer and belong to the same stage.
// It is used to distinguish other nodes that share the same input initializer but belong to
// other pipeline partitions after split.
struct PipelineStageNodeGroup {
  const size_t stage_id;

  // Vector of nodes that have the same initializer input and belong to the same stage. Noted that
  // the consumer nodes of a particular initializer can be more than one, so we need a vector to store those
  // nodes.
  std::vector<Node*> nodes;
  PipelineStageNodeGroup(const size_t stage, std::vector<Node*>& node_group) : stage_id(stage), nodes(std::move(node_group)){};
};

// This function passes through the given initializer across stages specified in node_groups[i].stage_id.
// This applies to the case when initializer is used in multiple stages, say stage a and stage b (a<b). We will
// keep the initializer in stage a and pass it down to b through the send nodes and recv nodes.
common::Status AddPassthroughInitializer(Graph& graph,
                                         NodeArg* initializer,
                                         const std::vector<PipelineStageNodeGroup>& node_groups,
                                         const std::vector<Node*>& send_nodes,
                                         const std::vector<Node*>& recv_nodes) {
  assert(initializer);
  ORT_ENFORCE(node_groups.size() >= 2, "Initializer ", initializer->Name(), " is not shared across stages.");

  const size_t from_stage = node_groups.front().stage_id;
  const size_t to_stage = node_groups.back().stage_id;

  ORT_ENFORCE(from_stage < to_stage, "Pass through from_stage (", from_stage,
              ") is not less than the to_stage (", to_stage, ").");

  auto dtype = initializer->TypeAsProto()->tensor_type().elem_type();

  auto current_node_arg = initializer;

  size_t node_group_index = 1;
  for (auto i = from_stage; i < to_stage; ++i) {
    // processing send node in cut i
    auto& send_attributes = send_nodes[i]->GetMutableAttributes();
    auto& send_element_types = send_attributes["element_types"];
    send_element_types.add_ints(static_cast<int64_t>(dtype));
    send_nodes[i]->MutableInputDefs().push_back(current_node_arg);
    send_nodes[i]->MutableInputArgsCount().back()++;

    // Create a new node_arg for the recv, as the new node_arg from recv node should possess a different id
    // than the one in send
    assert(current_node_arg);
    current_node_arg = &CreateNodeArg(graph, *current_node_arg);

    // process recv node in cut i
    auto& recv_attributes = recv_nodes[i]->GetMutableAttributes();
    auto& recv_element_types = recv_attributes["element_types"];
    recv_element_types.add_ints(static_cast<int64_t>(dtype));
    recv_nodes[i]->MutableOutputDefs().push_back(current_node_arg);

    // update the consumer node's input if the node's group is not in the first partition
    if (node_groups[node_group_index].stage_id == (i + 1)) {
      for (auto node : node_groups[node_group_index].nodes) {
        for (auto& input_node : node->MutableInputDefs()) {
          if (input_node == initializer) {
            input_node = current_node_arg;
            break;
          }
        }
      }
      node_group_index++;
    }
  }

  ORT_ENFORCE(node_group_index == node_groups.size(),
              "Not all nodes are updated with new initializer. Updated: ", node_group_index,
              ", expected: ", node_groups.size());

  return Status::OK();
}

// Traverse the graph to find out all connected elements in the graph from start_node. The traverse treats the graph as an
// undirected graph.
Status TraverseGraphWithConnectedElement(Graph& graph,
                                         Node* start_node,
                                         std::set<Node*>& visited_nodes,
                                         std::set<NodeArg*>& visited_inputs,
                                         std::set<NodeArg*>& visited_outputs) {
  assert(start_node);
  visited_nodes.clear();
  visited_inputs.clear();
  visited_outputs.clear();

  std::queue<Node*> node_queue;
  node_queue.push(start_node);

  while (!node_queue.empty()) {
    auto node = node_queue.front();
    node_queue.pop();
    if (visited_nodes.insert(node).second) {
      std::vector<Node*> connected_nodes;
      ORT_RETURN_IF_ERROR(FindAllConnectedNodes(graph, node, connected_nodes, visited_inputs, visited_outputs));

      for (auto n : connected_nodes) {
        ORT_ENFORCE(n != nullptr, "Found nullptr in searching for connected nodes");
        node_queue.push(n);
      }
    }
  }
  return Status::OK();
}

// If an initializer is shared across partitions, instead of creating a separate all_reduce op to
// sync with those tensors in selected partitions, we save only one copy of that initializer in
// the very first partition it appears, and pass that data down to all following partitions
// where this initializer is used.
common::Status HandleSharedInitializer(Graph& graph,
                                       const std::vector<Node*>& send_nodes,
                                       const std::vector<Node*>& recv_nodes) {
  // Map a given initializer to all the partitions that its consumer nodes reside. The size of
  // the mapped vector reflects how many partitions this initializer's consumer nodes distribute.
  // If its size is greater than 1, it means this initializer is being used in more than one partition and
  // we need to proceed those cases.
  std::map<NodeArg*, std::vector<PipelineStageNodeGroup>> input_consumer_stage_map;

  for (size_t stage = 0; stage <= send_nodes.size(); ++stage) {
    std::set<Node*> visited_nodes;
    std::set<NodeArg*> visited_inputs;
    std::set<NodeArg*> visited_outputs;

    // send_nodes[i] is the Send op in i-th stage's forward pass. recv_nodes[i] is the Recv in the (i+1)-th stage's
    // forward pass. When not in last stage, traverse start from send node; otherwise start with the recv node as
    // send node doesn't exist in last partition's forward pass.
    Node* traverse_start_node = stage < send_nodes.size() ? send_nodes[stage] : recv_nodes.back();
    ORT_RETURN_IF_ERROR(TraverseGraphWithConnectedElement(graph,
                                                          traverse_start_node,
                                                          visited_nodes,
                                                          visited_inputs,
                                                          visited_outputs));

    for (const auto input : visited_inputs) {
      // If the node is an input instead of an initializer, continue
      if (!graph.IsInitializedTensor(input->Name())) {
        continue;
      }

      // group all consumer nodes that shares the same input initializer in visited_consumer_nodes
      std::vector<Node*> consumer_nodes = graph.GetMutableConsumerNodes(input->Name());
      std::vector<Node*> visited_consumer_nodes;
      for (auto consumer_node : consumer_nodes) {
        if (visited_nodes.count(consumer_node) != 0) {
          visited_consumer_nodes.push_back(consumer_node);
        }
      }

      if (input_consumer_stage_map.count(input) == 0) {
        input_consumer_stage_map[input] = std::vector<PipelineStageNodeGroup>{
            PipelineStageNodeGroup(stage, visited_consumer_nodes)};
      } else {
        input_consumer_stage_map[input].push_back({stage, visited_consumer_nodes});
      }
    }
  }

  for (const auto& entry : input_consumer_stage_map) {
    // If any initializer is shared, handle the logic of passing it from the first seen stage all
    // the way to last seen stage.
    if (entry.second.size() > 1) {
      ORT_RETURN_IF_ERROR(AddPassthroughInitializer(graph,
                                                    entry.first,   // initializer node_arg
                                                    entry.second,  // initializer consumer node groups
                                                    send_nodes,
                                                    recv_nodes));
    }
  }
  return Status::OK();
}

// split the graph into disconnected subgraph based on provided CutInfo
common::Status SplitGraph(Graph& graph,
                          std::vector<TrainingSession::TrainingConfiguration::CutInfo> split_edge_groups,
                          std::vector<Node*>& send_nodes,
                          std::vector<Node*>& recv_nodes) {
  std::vector<std::string> new_input_names;
  std::vector<std::string> new_output_names;

  // updated_node_args keeps track of the mapping between the original node_arg and its corresponding updated
  // node_arg after send and recv node is added. As multiple partitions can happen, and a single node_arg
  // can belong to different partition, updated_node_args always keeps track of the latest updated node_arg.
  // Below is one example of how this works using update_node_args:
  //    there are three edges in graph, specified as nodeA->nodeB, nodeA->nodeC, and nodeA->nodeD.
  //    those edges all share the same node_arg.
  //    but nodeA, nodeB belong to parition0, nodeC belongs to parition1, and nodeD belongs to parition2.
  //    This means we need to cut edge nodeA->nodeC for the first partition and nodeA->nodeD for the second partition.
  //
  //    During the first cut, we identify the edge nodeA->nodeC, for this edge, based on the origional node_arg,
  //    we create a new node_arg, called updated_node_arg. The inserted send node will take the original node_arg
  //    as input and the inserted recv node will take the updated_node_arg as the output.
  //    And we update updated_node_args with updated_node_args[original_node_arg] = updated_node_arg
  //
  //    Now during the second cut, we need to cut the edge nodeA->nodeD. Noted that as the cut is performed in sequential,
  //    the second cut is performed based on the graph modified after the first cut. This means, the input node_arg for
  //    nodeD shouldn't come from nodeA anymore, as nodeA now residents in partition0, which is a disconnected partition.
  //    Instead, the input node_arg of nodeD should come from the updated version: updated_node_arg from partition1.
  //    By using the updated_node_args map, we can retrieve updated_node_arg from original_node_arg, and use that as the
  //    newly inserted send's input. Also, to keep this on going for any following cut, we create an updated_node_arg_v2,
  //    and update updated_node_args with updated_node_args[original_node_arg] = updated_node_arg_v2
  std::map<NodeArg*, NodeArg*> updated_node_args;

  for (size_t index = 0; index < split_edge_groups.size(); ++index) {
    // each entry in split_edge_groups represents a partition cut. Each cut can contain the split of
    // several edges.
    auto& edgeIds = split_edge_groups[index];

    // for each cut, record the inserted input/output args.
    std::vector<NodeArg*> send_input_args;
    std::vector<NodeArg*> send_output_args;
    std::vector<NodeArg*> recv_input_args;
    std::vector<NodeArg*> recv_output_args;

    auto cut_index_str = std::to_string(index);
    // add input node_arg and initializer for send/recv
    ORT_RETURN_IF_ERROR(AddNewScalarNodeArgAndInitializer<bool>(graph,
                                                                "send_input_signal" + cut_index_str,
                                                                ONNX_NAMESPACE::TensorProto_DataType_BOOL,
                                                                true, /* initializer data */
                                                                send_input_args,
                                                                new_input_names));
    ORT_RETURN_IF_ERROR(AddNewScalarNodeArgAndInitializer<bool>(graph,
                                                                "recv_input_signal" + cut_index_str,
                                                                ONNX_NAMESPACE::TensorProto_DataType_BOOL,
                                                                true, /* initializer data */
                                                                recv_input_args,
                                                                new_input_names));

    ORT_RETURN_IF_ERROR(AddNewScalarNodeArgAndInitializer<size_t>(graph,
                                                                  "send_dst_rank" + cut_index_str,
                                                                  ONNX_NAMESPACE::TensorProto_DataType_INT64,
                                                                  static_cast<int32_t>(index + 1), /* initializer data */
                                                                  send_input_args,
                                                                  new_input_names));
    ORT_RETURN_IF_ERROR(AddNewScalarNodeArgAndInitializer<size_t>(graph,
                                                                  "recv_src_rank" + cut_index_str,
                                                                  ONNX_NAMESPACE::TensorProto_DataType_INT64,
                                                                  static_cast<int32_t>(index), /* initializer data */
                                                                  recv_input_args,
                                                                  new_input_names));

    // add output node_arg for send/recv
    AddNewNodeArg(graph, "send_output_signal" + cut_index_str, ONNX_NAMESPACE::TensorProto_DataType_BOOL,
                  send_output_args, new_output_names);
    AddNewNodeArg(graph, "receive_output_signal" + cut_index_str, ONNX_NAMESPACE::TensorProto_DataType_BOOL,
                  recv_output_args, new_output_names);

    // add attribute data for send/recv
    ONNX_NAMESPACE::AttributeProto tag;
    tag.set_name("tag");
    tag.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
    // currently hard-coded all tag to be 0. May need to change when multiple GPU stream is used.
    tag.set_i(static_cast<int64_t>(0));

    ONNX_NAMESPACE::AttributeProto element_types;
    element_types.set_name("element_types");
    element_types.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);

    // for each edge in this group, perform edge cut
    for (auto& id : edgeIds) {
      // find node whose output contains id.node_arg_name
      auto producer_node = graph.GetMutableProducerNode(id.node_arg_name);
      if (!producer_node) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Cannot find producer node of node_arg with name: ", id.node_arg_name,
                               ". Wrong cutting infomation.");
      }

      // once we find out the producer node for id.node_arg_name, find which output index that leads
      // to id.node_arg_name
      int upstream_nodes_output_index{-1};
      ORT_RETURN_IF_ERROR(producer_node->ForEachWithIndex(
          producer_node->OutputDefs(),
          [&](const NodeArg& def, size_t index) {
            if (def.Name() == id.node_arg_name) {
              upstream_nodes_output_index = static_cast<int>(index);
            }
            return Status::OK();
          }));

      if (upstream_nodes_output_index < 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Node with name: ", producer_node->Name(),
                               " doesn't have an output node_arg with name ", id.node_arg_name);
      }

      size_t idx = static_cast<size_t>(upstream_nodes_output_index);

      // original node_arg pointer from the origin graph. This serves as the key in the
      // updated_node_arg map and any reference for original node_arg name
      auto* original_node_arg = producer_node->MutableOutputDefs()[idx];

      // updated node_arg pointer from previous partition. This is the new arg_node the
      // current inserted send node will take as input node_arg.
      auto updated_node_arg = producer_node->MutableOutputDefs()[idx];
      auto exiting_updated_node_arg = updated_node_args.find(original_node_arg);
      if (exiting_updated_node_arg != updated_node_args.end()) {
        updated_node_arg = exiting_updated_node_arg->second;
      }
      assert(updated_node_arg);

      send_input_args.push_back(updated_node_arg);

      auto dtype = original_node_arg->TypeAsProto()->tensor_type().elem_type();

      element_types.add_ints(static_cast<int64_t>(dtype));

      auto& new_receive_output = CreateNodeArg(graph, *updated_node_arg);
      const auto* old_shape = updated_node_arg->Shape();
      if (old_shape != nullptr) {
        new_receive_output.SetShape(*old_shape);
      }
      recv_output_args.push_back(&new_receive_output);

      // add value info for this newly added receive_output, for shape propagation
      // when training this partition.
      graph.AddValueInfo(&new_receive_output);

      // update updated_node_args with the newly created node_arg
      updated_node_args[original_node_arg] = &new_receive_output;

      // deal with shape inference for newly added edge
      auto& output_edge_name = original_node_arg->Name();

      // deal with updating the consumer's input node_args
      std::vector<Node*> consumer_nodes;
      if (id.consumer_nodes.has_value()) {
        for (auto& consumer_node_id : id.consumer_nodes.value()) {
          consumer_nodes.push_back(graph.GetMutableProducerNode(consumer_node_id));
        }
      } else {
        consumer_nodes = graph.GetMutableConsumerNodes(output_edge_name);
      }

      for (auto consumer_node : consumer_nodes) {
        for (auto& input : consumer_node->MutableInputDefs()) {
          if (input->Name() == output_edge_name) {
            input = &new_receive_output;
            break;
          }
        }
      }
    }
    constexpr int num_attributes = 2;  // two attributes: tag and element_types
    NodeAttributes attributes;
    attributes.reserve(num_attributes);
    attributes[tag.name()] = tag;
    attributes[element_types.name()] = element_types;

    auto& send_node = graph.AddNode(graph.GenerateNodeName("Send"),
                                    "Send",
                                    "",
                                    send_input_args,
                                    send_output_args, /* output */
                                    &attributes,      /* attribute */
                                    kMSDomain);

    send_nodes.push_back(&send_node);

    auto& recv_node = graph.AddNode(graph.GenerateNodeName("Recv"),
                                    "Recv",
                                    "",
                                    recv_input_args,
                                    recv_output_args, /* output */
                                    &attributes,      /* attribute */
                                    kMSDomain);
    recv_nodes.push_back(&recv_node);
  }

  ORT_RETURN_IF_ERROR(SetInputsOutputsAndResolve(graph, {} /* weights_to_train */, new_input_names, new_output_names));
  return Status::OK();
}

// traverse the graph from start_node to get the set of nodes contains in this disconnected subgraph
common::Status GenerateSubgraph(Graph& graph, Node* start_node) {
  assert(start_node);
  std::set<Node*> visited_nodes;
  std::set<NodeArg*> visited_inputs;
  std::set<NodeArg*> visited_outputs;

  // BFS graph traverse
  ORT_RETURN_IF_ERROR(TraverseGraphWithConnectedElement(graph, start_node,
                                                        visited_nodes, visited_inputs, visited_outputs));

  std::set<NodeIndex> visited_node_index;
  for (auto n : visited_nodes) {
    visited_node_index.insert(n->Index());
  }

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // reverse iterate the nodes in tolopogical order, and delete those not visited
  for (auto it = node_topology_list.rbegin(); it != node_topology_list.rend(); it++) {
    if (visited_node_index.count(*it) == 0) {
      graph.RemoveNode(*it);
    }
  }

  // If the following line is uncommented, middle and last pipeline stages may
  // have unresolved symbolic shapes. The reason is that some symbolic shapes
  // are defined for the original inputs, if original inputs are removed, we
  // lose the hint to resolve symbolic shapes. For example, if an original
  // input's shape is [batch, sequence, 1024], that input should be provided as
  // a feed to all pipeline stages. Otherwise, we don't know the actual values
  // of "batch" and "sequence".
  //
  // graph.SetInputs({visited_inputs.begin(), visited_inputs.end()});

  // update the grah with only visited outputs
  InlinedVector<const NodeArg*> visited_flat;
  visited_flat.reserve(visited_outputs.size());
  visited_flat.assign(visited_outputs.begin(), visited_outputs.end());
  graph.SetOutputs(visited_flat);
  graph.SetGraphResolveNeeded();
  graph.SetGraphProtoSyncNeeded();

  return graph.Resolve();
}

Status CutBasedApplyPipelinePartitionToMainGraph(
    Graph& graph,
    const std::vector<TrainingSession::TrainingConfiguration::CutInfo>& cut_info,
    size_t pipeline_stage_id,
    size_t num_pipeline_stage) {
  size_t split_count = cut_info.size();

  if (num_pipeline_stage != split_count + 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Wrong pipeline partition cutting info. Total pipeline stage number is ",
                           num_pipeline_stage,
                           ", cut info length is: ",
                           split_count);
  }

  std::vector<Node*> send_nodes, recv_nodes;
  send_nodes.reserve(split_count);
  recv_nodes.reserve(split_count);

  // Split the graph by cutting edges specified in cut_info. After this function, the graph will be
  // composed of several disconnected partitions.
  ORT_RETURN_IF_ERROR(SplitGraph(graph, cut_info, send_nodes, recv_nodes));

  if (send_nodes.size() != split_count || recv_nodes.size() != split_count) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Split error: not all cut has Send and Recv inserted. Send node count: ",
                           send_nodes.size(), ", Recv node count: ", recv_nodes.size(), ", split count: ", split_count);
  }

  // Check to see if there are any initializers that is being shared between different partitions. If there
  // is, keep the initializer in the first seen partition and have it pass through by send/recv to the others.
  ORT_RETURN_IF_ERROR(HandleSharedInitializer(graph, send_nodes, recv_nodes));

  // Now remove the partitions that are not tie to the current pipeline stage and generate the sub-graph.
  if (pipeline_stage_id < split_count) {
    ORT_RETURN_IF_ERROR(GenerateSubgraph(graph, send_nodes[pipeline_stage_id]));
  } else {
    ORT_RETURN_IF_ERROR(GenerateSubgraph(graph, recv_nodes.back()));
  }

  // Post check to ensure the curent partition is correct and matches with Send/Recv nodes inserted during split.
  Node* send_node{nullptr};
  Node* recv_node{nullptr};
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Send") {
      send_node = &node;
    } else if (node.OpType() == "Recv") {
      recv_node = &node;
    }
  }

  if (pipeline_stage_id == 0) {
    // For the first stage, there should be no recv node, and the send node contained in graph should match the first
    // send_node inserted during split.
    ORT_RETURN_IF_NOT(recv_node == nullptr, "Error: first stage contains Recv node in forward pass.");
    ORT_RETURN_IF_NOT(send_node == send_nodes[0],
                      "Error: first stage doesn't contain the right Send node. Possibly CutInfo data is wrong.");
  } else if (pipeline_stage_id == split_count) {
    // For the last stage, there should be no send node, and the recv node contained in graph should match the last
    // recv_node inserted during split.
    ORT_RETURN_IF_NOT(recv_node == recv_nodes.back(),
                      "Error: last stage doesn't contain the right Recv node. Possibly CutInfo data is wrong.");
    ORT_RETURN_IF_NOT(send_node == nullptr, "Error: last stage contains Send node in forward pass.");
  } else {
    // For stages in the middle, i-th stage should contain recv node that matches the (i-1)-th inserted recv node, and the i-th
    // inserted send node.
    ORT_RETURN_IF_NOT(recv_node == recv_nodes[pipeline_stage_id - 1],
                      "Error: stage ", pipeline_stage_id, " doesn't contain the right Recv node. Possibly CutInfo data is wrong.");
    ORT_RETURN_IF_NOT(send_node == send_nodes[pipeline_stage_id],
                      "Error: stage ", pipeline_stage_id, " doesn't contain the right Send node. Possibly CutInfo data is wrong.");
  }

  return Status::OK();
}

}  // namespace test
}  // namespace onnxruntime
