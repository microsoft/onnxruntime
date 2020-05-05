// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/pipeline_transformer.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace training {

void GetPipelineSendOutput(const Graph& graph, std::string& loss_name) {
  for (auto& node : graph.Nodes()) {
    if (!node.OpType().compare("Send")) {
      // send op should always have an output, which is the OutputSignal.
      loss_name = node.OutputDefs()[0]->Name();
      return;
    }
  }
}

bool IsBackward(Node& node) {
  return (node.Description() == "Backward pass");
}

void AddInputEvent(Graph& graph, const std::string& op_name,
                   bool is_forward,
                   std::vector<NodeArg*>& input_args,
                   std::vector<std::string>& new_input_names) {
  ONNX_NAMESPACE::TypeProto event_type_proto;
  event_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);

  auto event_id_name = graph.GenerateNodeArgName(op_name + (is_forward ? "_fw" : "_bw") + "_event_id");
  auto& event_id = graph.GetOrCreateNodeArg(event_id_name, &event_type_proto);
  new_input_names.push_back(event_id_name);

  input_args.push_back(&event_id);
}

// gradient graph can contain some dangling leaf nodes. Add them all to WaitEvent
// backward node's input.
void FindLeafNodes(Graph& graph, std::vector<NodeArg*>& input_args) {
  for (auto& node : graph.Nodes()) {
    if (!IsBackward(node)) {
      // only check backward node
      continue;
    }
    bool find_consumer_nodes = false;
    std::vector<NodeArg*>& outputs = node.MutableOutputDefs();
    for (auto& output : outputs) {
      std::vector<const Node*> consumer_nodes = graph.GetConsumerNodes(output->Name());
      if (consumer_nodes.size() > 0) {
        find_consumer_nodes = true;
        break;
      }
    }
    if (!find_consumer_nodes && outputs.size() > 0) {
      input_args.push_back(outputs[0]);
    }
  }
};

NodeArg& CreateNodeArg(Graph& graph, const NodeArg* base_arg) {
  const auto& new_name = graph.GenerateNodeArgName(base_arg->Name());
  ONNX_NAMESPACE::TypeProto type_proto(*(base_arg->TypeAsProto()));
  if (graph.GetNodeArg(new_name) != nullptr) {
    ORT_THROW("Node with name ", new_name, " already exists.");
  }
  return graph.GetOrCreateNodeArg(new_name, &type_proto);
}

Status AddRecordBackward(Graph& graph,
                       Node* send_bw,
                       std::vector<std::string>& new_input_names,
                       std::vector<std::string>& new_output_names) {
  std::vector<NodeArg*> input_args;
  AddInputEvent(graph, "RecordEvent", false /* is_forward */, input_args, new_input_names);
  std::vector<NodeArg*> output_args{};

  if (send_bw) {
    // if we have send op in backward pass (at the end of the graph), we make sure the RecordEvent happens
    // after that send by adding Send's outputs to RecordEvent's input list.
    input_args.insert(std::end(input_args),
                      std::begin(send_bw->MutableOutputDefs()),
                      std::end(send_bw->MutableOutputDefs()));
  }
  FindLeafNodes(graph, input_args);

  // Optimizer will be added after applying pipeline transformer. To support partial graph evaluation,
  // the added Record backward op will have its first passthrough input as output.
  ORT_RETURN_IF_NOT(input_args.size() >= 2, "RecordEvent backward op at least have two inputs.")
  auto& new_output = CreateNodeArg(graph, input_args[1]); // the first input is signal, not passing through
  output_args.push_back(&new_output);
  new_output_names.push_back(new_output.Name());

  graph.AddNode(graph.GenerateNodeName("RecordEvent"),
                "RecordEvent",
                "Backward pass",
                input_args,
                output_args,
                nullptr,
                kMSDomain);
  return Status::OK();
}

Status AddWaitForward(Graph& graph, Node* /* recv_fw */, std::vector<std::string>& new_input_names) {
  // Append old_input to input_args and return its pass-through value. Note that
  // input_args and output_args are Wait's inputs and outputs, respectively.
  auto update_wait_input_output = [&](NodeArg* old_input,
                                      std::vector<NodeArg*>& input_args,
                                      std::vector<NodeArg*>& output_args) -> NodeArg& {
    input_args.push_back(old_input);

    const auto& new_name = graph.GenerateNodeArgName(old_input->Name());
    ONNX_NAMESPACE::TypeProto type_proto(*(old_input->TypeAsProto()));

    auto& wait_output = graph.GetOrCreateNodeArg(new_name, &type_proto);
    output_args.push_back(&wait_output);

    return wait_output;
  };

  std::vector<NodeArg*> input_args;
  std::vector<NodeArg*> output_args;
  AddInputEvent(graph, "WaitEvent", true /* is_forward */, input_args, new_input_names);
  const std::vector<const NodeArg*>& graph_inputs = graph.GetInputsIncludingInitializers();

  if (graph_inputs.size() == 0){
    ORT_THROW("Graph ", graph.Name(), " doesn't have any inputs.");
  }

  for (auto& input_arg : graph_inputs) {
    NodeArg* mutable_input = graph.GetNodeArg(input_arg->Name());
    auto& wait_output = update_wait_input_output(mutable_input, input_args, output_args);
    std::vector<Node*> nodes = graph.GetMutableConsumerNodes(input_arg->Name());
    for (auto& consumer_node : nodes) {
      for (auto& i : consumer_node->MutableInputDefs()) {
        if (i->Name() == input_arg->Name()) {
          // if the node is fed by input, re-direct it to be fed by WaitEvent's output.
          i = &wait_output;
        }
      }
    }
  }
  graph.AddNode(graph.GenerateNodeName("WaitEvent"),
                "WaitEvent",
                "",
                input_args,
                output_args,
                nullptr,
                kMSDomain);

  return Status::OK();
}

Status AddOrSkipRecordForwardWaitBackward(Graph& graph, Node* send_fw, Node* recv_bw, std::vector<std::string>& new_input_names) {
  if (!send_fw != !recv_bw){
    ORT_THROW("Graph requires either having both send forward node "
      "and recv backword node, or none of them. Currently the graph "
      "has send forward: ", send_fw, " and recv backward: ", recv_bw);
  }

  if (!send_fw && !recv_bw){
    // Last partition doesn't have send forwrad and recv backward. No insert needed.
    return Status::OK();
  }

  // if we have a send forward op followed by a recv backward op, insert WaitEvent and RecordEvent in between.
  Node* record_node = nullptr;
  Node* wait_node = nullptr;

  // Insert RecordEvent
  {
    std::vector<NodeArg*> input_args;
    std::vector<NodeArg*> output_args;
    AddInputEvent(graph, "RecordEvent", true /* is_forward */, input_args, new_input_names);

    // Add send forward op's output as record op's input and output
    for (auto& output : send_fw->MutableOutputDefs()) {
      auto& new_output = CreateNodeArg(graph, output);
      output_args.push_back(&new_output);
      input_args.push_back(output);
    }

    auto& new_node = graph.AddNode(graph.GenerateNodeName("RecordEvent"),
                                    "RecordEvent",
                                    "",
                                    input_args,
                                    output_args, /* output */
                                    {},          /* attribute */
                                    kMSDomain);
    record_node = &new_node;
  }
  // Insert WaitEvent
  {
    std::vector<NodeArg*> input_args;
    std::vector<NodeArg*> output_args;
    AddInputEvent(graph, "WaitEvent", false /* is_forward */, input_args, new_input_names);

    input_args.insert(std::end(input_args),
                      std::begin(record_node->MutableOutputDefs()),
                      std::end(record_node->MutableOutputDefs()));

    auto& input = recv_bw->MutableInputDefs()[0];
    auto& new_output = CreateNodeArg(graph, input);
    output_args.push_back(&new_output);
    input = &new_output;

    auto& new_node = graph.AddNode(graph.GenerateNodeName("WaitEvent"),
                  "WaitEvent",
                  "Backward pass",
                  input_args,
                  output_args, /* output */
                  {},          /* attribute */
                  kMSDomain);
    wait_node = &new_node;
    ORT_UNUSED_PARAMETER(wait_node);
  }

  return Status::OK();
}

NodeArg& CreateInt64NodeArg(Graph& graph, std::string name) {
  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  auto actual_name = graph.GenerateNodeArgName(name);
  auto& event = graph.GetOrCreateNodeArg(actual_name, &type_proto);
  return event;
}

// Return node_args consumed but not produced by the downstream_nodes.
std::vector<NodeArg*> FindUpstreamNodeArgs(std::vector<Node*> downstream_nodes) {
  // node_args produced by this sub-graph.
  std::unordered_set<const NodeArg*> downstream_node_args;
  // node_args from another sub-graph and consumed by this sub-graph.
  std::unordered_set<NodeArg*> upstream_node_args;

  for (auto& node: downstream_nodes) {
    for (auto& node_arg: node->OutputDefs()) {
      downstream_node_args.insert(node_arg);
    }
  }

  for (auto& node: downstream_nodes) {
    auto consumed_node_args = node->MutableInputDefs();
    for (auto& node_arg: consumed_node_args) {
      if (downstream_node_args.find(node_arg) != downstream_node_args.end()) {
        continue;
      }
      upstream_node_args.insert(node_arg);
    }
  }

  // Convert set to vector.
  std::vector<NodeArg*> node_args;
  for (auto& node_arg: upstream_node_args) {
    node_args.push_back(node_arg);
  }

  return node_args;
}

// Return mirror variables for node_args. The i-th output element mirrors node_args[i]
// but with a different name.
std::vector<NodeArg*> CreateNewNodeArgs(Graph& graph, std::vector<NodeArg*> node_args) {
  std::vector<NodeArg*> new_node_args;

  for (auto& node_arg: node_args) {
    const auto new_name = graph.GenerateNodeArgName(node_arg->Name());
    ORT_ENFORCE(graph.GetNodeArg(new_name) != nullptr,
                "NodeArg with name ",
                new_name,
                " already exists but we still want to create it." );
    ONNX_NAMESPACE::TypeProto type_proto(*node_arg->TypeAsProto());

    // new_node_arg is a mirror variable of node_arg. They have the same type.
    auto new_node_arg = &graph.GetOrCreateNodeArg(new_name, &type_proto);
    new_node_args.push_back(new_node_arg);
  }

  return new_node_args;
}
  
// Replace node_args[i] with new_node_args[i] for all inputs in nodes.
void ReplaceNodeArgs(std::vector<Node*> nodes,
                     std::vector<NodeArg*> node_args,
                     std::vector<NodeArg*> new_node_args) {
  ORT_ENFORCE(node_args.size() == new_node_args.size());
  for (size_t i = 0; i < node_args.size(); ++i) {
    // At this iteration, we replace node_args[i] with 
    ORT_ENFORCE(node_args[i]->Name() != new_node_args[i]->Name());
    ORT_ENFORCE(node_args[i]->Type() == new_node_args[i]->Type());

    for (auto& node: nodes) {
      for (auto& node_arg: node->MutableInputDefs()) {
        node_arg = new_node_args[i];
      }
    }
  }
}

// Filter out node_args not produced by nodes.
std::vector<NodeArg*> FilterStrayNodeArgs(std::vector<Node*> nodes, std::vector<NodeArg*> node_args) {
  std::unordered_set<NodeArg*> produced_node_args;
  for (auto& node: nodes) {
    for (auto& node_arg: node->MutableOutputDefs()) {
      produced_node_args.insert(node_arg);
    }
  }

  // node_args from input but excluding node_args not produced in nodes.
  std::vector<NodeArg*> filtered_node_args;

  for (auto& node_arg: node_args) {
    if (produced_node_args.find(node_arg) == produced_node_args.end()) {
      continue;
    }
    filtered_node_args.push_back(node_arg);
  }

  return filtered_node_args;
}


// Create a node with input schema [event, input1, input2, ..., inputN] and
// output schema [input1, input2, ..., inputN]
void CreateBottleneckNode(Graph& graph,
                          const std::string op_type,
                          const std::string op_name,
                          const std::string description,
                          NodeArg* event,
                          std::vector<NodeArg*> input_node_args,
                          std::vector<NodeArg*> output_node_args) {
  const auto name = graph.GenerateNodeName(op_name);
  input_node_args.insert(input_node_args.begin(), event);
  graph.AddNode(
    name,
    op_type,
    description,
    input_node_args,
    output_node_args,
    nullptr /* assume all bottleneck node have no attributes */,
    kMSDomain);
}


void InsertEventBottleneckNode(Graph& graph,
                               const std::string op_type, // WaitEvent or RecordEvent
                               const std::string op_name,
                               const std::string description,
                               std::vector<Node*> upstream_nodes,
                               std::vector<Node*> downstream_nodes,
                               NodeArg* event) {

  // Anything from upstream_nodes, inputs, and initializers
  // should go through bottleneck node to go downstream_nodes.

  // Dependencies of downstream_nodes. They are inputs of bottleneck node.
  std::vector<NodeArg*> downstream_deps = FindUpstreamNodeArgs(downstream_nodes);

  // Filter out dependencies not from upstream_nodes (a sub-graph).
  // The reason is that we only want to create a bottlneck between upstream_nodes
  // and downstrem_nodes. We don't want to create a bottleneck between downstrem_nodes
  // and the rest world.
  FilterStrayNodeArgs(upstream_nodes, downstream_deps);

  // New dependencies to be consumed in downstream_nodes. They are outputs of
  // bottleneck node.
  std::vector<NodeArg*> new_downstream_deps = CreateNewNodeArgs(graph, downstream_deps);

  // Replace downstream_deps in downstream_nodes with their corresponding new_downstream_deps. 
  ReplaceNodeArgs(downstream_nodes, downstream_deps, new_downstream_deps);

  // Create bottleneck node.
  CreateBottleneckNode(graph,
                       op_type,
                       op_name,
                       description,
                       event /* first input of event-related op */,
                       downstream_deps /* inputs pass through event-related op */,
                       new_downstream_deps);
}

// Return nodes which directly produce inputs of node.
// If node is NULL, it returns an empty vector.
std::vector<Node*> GetDirectUpstreamNodes(Graph& graph, Node* node) {
  std::vector<Node*> upstream_nodes;
  if (node) {
    // Use set to avoid duplicates.
    std::unordered_set<Node*> upstream_nodes_;

    // Find non-duplicated producer nodes for inputs of Send.
    for (auto& node_arg: node->InputDefs()) {
      auto node = graph.GetMutableProducerNode(node_arg->Name()); 
      upstream_nodes_.insert(node);
    }

    // Cast set to vector.
    for (auto& node: upstream_nodes_) {
      upstream_nodes.push_back(node);
    }
  }
  return upstream_nodes;
}

// Return nodes which directly consume outputs of node.
// If node is NULL, it returns an empty vector.
std::vector<Node*> GetDirectDownstreamNodes(Graph& graph, Node* node) {
  std::vector<Node*> downstream_nodes;
  if (node) {
    // Use set to avoid duplicates.
    std::unordered_set<Node*> downstream_nodes_;

    // Find non-duplicated consumer nodes for outputs of Send.
    for (auto& node_arg: node->OutputDefs()) {
      for (auto& node: graph.GetMutableConsumerNodes(node_arg->Name())) {
        downstream_nodes_.insert(node);
      }
    }

    // Cast set to vector.
    for (auto& node: downstream_nodes_) {
      downstream_nodes.push_back(node);
    }
  }
  return downstream_nodes;
}

NodeArg* AddForwardWaitAfterRecv(Graph& graph) {
  // Find forward Recv node.
  Node* forward_recv_node = nullptr;
  for (auto& node: graph.Nodes()) {
    if (IsBackward(node) || !node.OpType().compare("Recv")) {
      continue;
    }
    // There should be only one forward Recv.
    // By entering this block, forward_recv_node should haven't been assigned.
    ORT_ENFORCE(forward_recv_node == nullptr);

    forward_recv_node = &node;
  }

  // Find nodes connected to Recv's input node_arg. 
  auto upstream_nodes = GetDirectUpstreamNodes(graph, forward_recv_node);
  // Find nodes connected to Recv's output node_arg. 
  auto downstream_nodes = GetDirectDownstreamNodes(graph, forward_recv_node);

  // Recv is upstream because the cut happens on the output side of Recv.
  if (forward_recv_node) {
    upstream_nodes.push_back(forward_recv_node);
  }

  // Create node_arg for event ID.
  auto event_node_arg = &CreateInt64NodeArg(graph, "event_forward_wait_after_recv");

  // Insert a WaitEvent before Recv. 
  InsertEventBottleneckNode(
    graph, "WaitEvent", "forward_wait_after_recv", "",
    upstream_nodes, downstream_nodes, event_node_arg);

  return event_node_arg;
}

NodeArg* AddForwardRecordBeforeSend(Graph& graph) {
  // Find forward Send node.
  Node* forward_send_node = nullptr;
  for (auto& node: graph.Nodes()) {
    if (IsBackward(node) || !node.OpType().compare("Recv")) {
      continue;
    }
    // There should be only one forward Send.
    // By entering this block, forward_send_node should haven't been assigned.
    ORT_ENFORCE(forward_send_node == nullptr);
    forward_send_node = &node;
  }

  // Find nodes connected to Send's input node_arg. 
  auto upstream_nodes = GetDirectUpstreamNodes(graph, forward_send_node);
  // Find nodes connected to Send's output node_arg. 
  auto downstream_nodes = GetDirectDownstreamNodes(graph, forward_send_node);

  // Send is downstream because the cut happens on the input side of Send.
  if (forward_send_node) {
    downstream_nodes.push_back(forward_send_node);
  }

  // Create node_arg for event ID.
  auto event_node_arg = &CreateInt64NodeArg(graph, "event_forward_record_before_send");

  // Insert a Record before Send. 
  InsertEventBottleneckNode(
    graph, "RecordEvent", "forward_record_before_send", "",
    upstream_nodes, downstream_nodes, event_node_arg);

  return event_node_arg;
}

NodeArg* AddBackwardWaitAfterRecv(Graph& graph) {
  // Find backward Recv node.
  Node* backward_recv_node = nullptr;
  for (auto& node: graph.Nodes()) {
    if (!IsBackward(node) || !node.OpType().compare("Recv")) {
      continue;
    }
    // There should be only one backward Recv.
    // By entering this block, backward_recv_node should haven't been assigned.
    ORT_ENFORCE(backward_recv_node == nullptr);

    backward_recv_node = &node;
  }

  // Find nodes connected to Recv's input node_arg. 
  auto upstream_nodes = GetDirectUpstreamNodes(graph, backward_recv_node);
  // Find nodes connected to Recv's output node_arg. 
  auto downstream_nodes = GetDirectDownstreamNodes(graph, backward_recv_node);

  // Recv is upstream because the cut happens on the output side of Recv.
  if (backward_recv_node) {
    upstream_nodes.push_back(backward_recv_node);
  }

  // Create node_arg for event ID.
  auto event_node_arg = &CreateInt64NodeArg(graph, "event_backward_wait_after_recv");

  // Insert a WaitEvent before Recv. 
  InsertEventBottleneckNode(
    graph, "WaitEvent", "backward_wait_after_recv", "",
    upstream_nodes, downstream_nodes, event_node_arg);

  return event_node_arg;
}

NodeArg* AddBackwardRecordBeforeSend(Graph& graph) {
  // Find backward Send node.
  Node* backward_send_node = nullptr;
  for (auto& node: graph.Nodes()) {
    if (!IsBackward(node) || !node.OpType().compare("Send")) {
      continue;
    }
    // There should be only one backward Recv.
    // By entering this block, backward_recv_node should haven't been assigned.
    ORT_ENFORCE(backward_send_node == nullptr);

    backward_send_node = &node;
  }

  // Find nodes connected to Recv's input node_arg. 
  auto upstream_nodes = GetDirectUpstreamNodes(graph, backward_send_node);
  // Find nodes connected to Recv's output node_arg. 
  auto downstream_nodes = GetDirectDownstreamNodes(graph, backward_send_node);

  // Send is downstream because the cut happens on the input side of Send.
  if (backward_send_node) {
    downstream_nodes.push_back(backward_send_node);
  }

  // Create node_arg for event ID.
  auto event_node_arg = &CreateInt64NodeArg(graph, "event_backward_wait_before_send");

  // Insert a WaitEvent before Recv. 
  InsertEventBottleneckNode(
    graph, "WaitEvent", "backward_wait_before_send", "",
    upstream_nodes, downstream_nodes, event_node_arg);

  return event_node_arg;
}

Status TransformGraphForPipeline(Graph& graph) {
  // insert WaitEvent and RecordEvent to the partition
  Node* send_fw{nullptr};
  Node* send_bw{nullptr};
  Node* recv_fw{nullptr};
  Node* recv_bw{nullptr};
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Send") {
      if (IsBackward(node)) {
        send_bw = &node;
      } else {
        send_fw = &node;
      }
    } else if (node.OpType() == "Recv") {
      if (IsBackward(node)) {
        recv_bw = &node;
      } else {
        recv_fw = &node;
      }
    }
  }

  std::vector<std::string> new_input_names;
  std::vector<std::string> new_output_names;

  ORT_RETURN_IF_ERROR(AddRecordBackward(graph, send_bw, new_input_names, new_output_names));
  ORT_RETURN_IF_ERROR(AddWaitForward(graph, recv_fw, new_input_names));
  ORT_RETURN_IF_ERROR(AddOrSkipRecordForwardWaitBackward(graph, send_fw, recv_bw, new_input_names));

  auto fill_node_args = [&](const Graph& graph,
                            const std::vector<const NodeArg*>& existed_node_args,
                            std::vector<std::string>& new_node_arg_names,
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
  return graph.Resolve();
}

}  // namespace training
}  // namespace onnxruntime
