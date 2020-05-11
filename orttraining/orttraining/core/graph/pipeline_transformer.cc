// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/pipeline_transformer.h"
#include <unistd.h>

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
                       std::vector<std::string>& new_output_names,
                       std::string &event_id_tensor_name,
                       std::string &output_tensor_name) {
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

  // First input argument is the recorded event ID tensor.
  event_id_tensor_name = input_args.front()->Name();
  // Use first output as output singnal. It will be fetched outside to make sure
  // event operator is computed.
  output_tensor_name = output_args.front()->Name();
  return Status::OK();
}

Status AddWaitForward(Graph& graph,
                      Node* /* recv_fw */,
                      std::vector<std::string>& new_input_names,
                      std::string& forward_waited_event_name,
                      std::string& output_tensor_name) {
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
  forward_waited_event_name = input_args.front()->Name();
  output_tensor_name = output_args.front()->Name();
  return Status::OK();
}

Status AddOrSkipRecordForwardWaitBackward(Graph& graph,
                                          Node* send_fw,
                                          Node* recv_bw,
                                          std::vector<std::string>& new_input_names,
                                          std::string& forward_recorded_event_name,
                                          std::string& backward_waited_event_name,
                                          std::string& forward_output_name,
                                          std::string& backward_output_name) {
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

    forward_recorded_event_name = record_node->InputDefs()[0]->Name();
    forward_output_name = record_node->OutputDefs()[0]->Name();
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

    backward_waited_event_name = wait_node->InputDefs()[0]->Name();
    backward_output_name = wait_node->OutputDefs()[0]->Name();
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
    ORT_ENFORCE(graph.GetNodeArg(new_name) == nullptr,
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
void ReplaceNodeArgs(std::vector<Node*>& nodes,
                     std::vector<NodeArg*>& node_args,
                     std::vector<NodeArg*>& new_node_args) {
  ORT_ENFORCE(node_args.size() == new_node_args.size());
  for (size_t i = 0; i < node_args.size(); ++i) {
    // At this iteration, we replace node_args[i] with 
    ORT_ENFORCE(node_args[i]->Name() != new_node_args[i]->Name());
    ORT_ENFORCE(node_args[i]->Type() == new_node_args[i]->Type());

    for (auto& node: nodes) {
      for (auto& node_arg: node->MutableInputDefs()) {
        if (node_arg->Name().compare(node_args[i]->Name()) != 0) {
          continue;
        }
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
  std::cout << getpid() << ": " << op_type << ", " << name << ", " << input_node_args[0]->Name();
  for (size_t i = 0; i < output_node_args.size(); ++i) {
    std::cout << input_node_args[i + 1]->Name() << " --> " << output_node_args[i]->Name() << ", ";
  }
  std::cout << std::endl;
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
  // After this step, downstream_deps should contains node_args in downstream_nodes
  // produced by upstream_nodes.
  downstream_deps = FilterStrayNodeArgs(upstream_nodes, downstream_deps);

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
  Node* forward_wait_node = nullptr;
  for (auto& node: graph.Nodes()) {
    if (IsBackward(node)) {
      continue;
    }
    if (node.OpType().compare("Recv") == 0) {
      // There should be only one forward Recv.
      // By entering this block, forward_recv_node should haven't been assigned.
      ORT_ENFORCE(forward_recv_node == nullptr);
      forward_recv_node = &node;
    }
    if (node.OpType().compare("WaitEvent") == 0) {
      // There should be only one forward WaitEvent.
      // By entering this block, forward_wait_node should haven't been assigned.
      ORT_ENFORCE(forward_wait_node == nullptr);
      forward_wait_node = &node;
    }
  }

  Node* blocked_node = forward_recv_node ? forward_recv_node : forward_wait_node; 

  ORT_ENFORCE(blocked_node, "Either forward Recv or forward WaitEvent should be found.");

  // Find nodes connected to Recv's input node_arg. 
  auto upstream_nodes = GetDirectUpstreamNodes(graph, blocked_node);
  // Find nodes connected to Recv's output node_arg. 
  auto downstream_nodes = GetDirectDownstreamNodes(graph, blocked_node);

  upstream_nodes.push_back(blocked_node);

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
  Node* forward_record_node = nullptr;
  for (auto& node: graph.Nodes()) {
    if (IsBackward(node)) {
      continue;
    }
    if (node.OpType().compare("Recv") == 0) {
      // There should be only one forward Send.
      // By entering this block, forward_send_node should haven't been assigned.
      ORT_ENFORCE(forward_send_node == nullptr);
      forward_send_node = &node;
    }
    if (node.OpType().compare("RecordEvent") == 0) {
      // There should be only one forward RecordEvent.
      // By entering this block, forward_record_node should haven't been assigned.
      ORT_ENFORCE(forward_record_node == nullptr);
      forward_record_node = &node;
    }
  }

  Node* blocked_node = forward_send_node ? forward_send_node : forward_record_node; 

  ORT_ENFORCE(blocked_node, "Either forward Send or forward RecordEvent should be found.");

  // Find nodes connected to Send's input node_arg. 
  auto upstream_nodes = GetDirectUpstreamNodes(graph, blocked_node);
  // Find nodes connected to Send's output node_arg. 
  auto downstream_nodes = GetDirectDownstreamNodes(graph, blocked_node);

  downstream_nodes.push_back(blocked_node);

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
  Node* backward_wait_node = nullptr;
  for (auto& node: graph.Nodes()) {
    if (!IsBackward(node)) {
      continue;
    }
    std::cout << getpid() << ": backward " << node.OpType() << std::endl;
    if (node.OpType().compare("Recv") == 0) {
      // There should be only one backward Recv.
      // By entering this block, backward_recv_node should haven't been assigned.
      ORT_ENFORCE(backward_recv_node == nullptr);
      backward_recv_node = &node;
    }
    if (node.OpType().compare("WaitEvent") == 0) {
      // There should be only one backward Wait.
      // By entering this block, backward_wait_node should haven't been assigned.
      ORT_ENFORCE(backward_wait_node == nullptr);
      backward_wait_node = &node;
    }
  }

  Node* blocked_node = backward_recv_node ? backward_recv_node : backward_wait_node; 

  ORT_ENFORCE(blocked_node, "Either backward Recv or backward WaitEvent should be found.");

  // Find nodes connected to Recv's input node_arg. 
  auto upstream_nodes = GetDirectUpstreamNodes(graph, blocked_node);
  // Find nodes connected to Recv's output node_arg. 
  auto downstream_nodes = GetDirectDownstreamNodes(graph, blocked_node);

  upstream_nodes.push_back(blocked_node);

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
  Node* backward_record_node = nullptr;
  for (auto& node: graph.Nodes()) {
    if (!IsBackward(node)) {
      continue;
    }
    if (node.OpType().compare("Send") == 0) {
      // There should be only one backward Recv.
      // By entering this block, backward_recv_node should haven't been assigned.
      ORT_ENFORCE(backward_send_node == nullptr);
      backward_send_node = &node;
    }
    if (node.OpType().compare("RecordEvent") == 0) {
      // There should be only one backward Record.
      // By entering this block, backward_record_node should haven't been assigned.
      ORT_ENFORCE(backward_record_node == nullptr);
      backward_record_node = &node;
    }
  }

  Node* blocked_node = backward_send_node ? backward_send_node : backward_record_node; 

  ORT_ENFORCE(blocked_node, "Either backward Send or backward RecordEvent should be found.");

  // Find nodes connected to Recv's input node_arg. 
  auto upstream_nodes = GetDirectUpstreamNodes(graph, blocked_node);
  // Find nodes connected to Recv's output node_arg. 
  auto downstream_nodes = GetDirectDownstreamNodes(graph, blocked_node);

  // Send is downstream because the cut happens on the input side of Send.
  downstream_nodes.push_back(backward_send_node);

  // Create node_arg for event ID.
  auto event_node_arg = &CreateInt64NodeArg(graph, "event_backward_wait_before_send");

  // Insert a WaitEvent before Recv. 
  InsertEventBottleneckNode(
    graph, "WaitEvent", "backward_wait_before_send", "",
    upstream_nodes, downstream_nodes, event_node_arg);

  return event_node_arg;
}

Status AddForwardWaitAfterRecv1(
    Graph& graph,
    /* forward Recv */ Node* comm_node,
    std::vector<std::string>& new_input_names,
    std::string& event_name) {
  if (!comm_node) {
    // No WaitEvent is inserted, so its input event name is empty.
    event_name = "";
    return Status::OK();
  }

  // Get outputs of Recv nodes. They will be connected to a WaitEvent.
  // Consumers of Recv's will be connected to the newly-added WaitEvent.
  std::vector<NodeArg*> node_args = comm_node->MutableOutputDefs();

  // Declare mirror variables. node_args[i] will be replaced with new_node_args[i].
  std::vector<NodeArg*> new_node_args = CreateNewNodeArgs(graph, node_args);

  // Replace the uses of outputs with corresponding new_outputs.
  std::vector<Node*> nodes;
  for (auto& node: graph.Nodes()) {
    nodes.push_back(&node);
  }
  ReplaceNodeArgs(nodes, node_args, new_node_args); 

  // Create node_arg for event ID.
  auto event_node_arg = &CreateInt64NodeArg(graph, "event_forward_wait_after_recv");

  // Create node which produces new_node_args from event and node_arg.
  auto name = graph.GenerateNodeName("RecordEvent_ForwardAfterRecv");
  CreateBottleneckNode(graph,
                       "WaitEvent",
                       name,
                       "",
                       event_node_arg,
                       node_args,
                       new_node_args);

  event_name = event_node_arg->Name();

  new_input_names.push_back(event_name);

  return Status::OK();
}

Status AddForwardRecordBeforeSend1(
    Graph& graph,
    Node* comm_node,
    std::vector<std::string>& new_input_names,
    std::string& event_name) {
  if (!comm_node) {
    // No RecordEvent is inserted, so its input event name is empty.
    event_name = "";
    return Status::OK();
  }

  // Get outputs of Send nodes. They will be connected to a RecordEvent.
  // Consumers of Send's will be connected to the newly-added RecordEvent.
  std::vector<NodeArg*> node_args = comm_node->MutableInputDefs();

  // Declare mirror variables. node_args[i] will be replaced with new_node_args[i].
  std::vector<NodeArg*> new_node_args = CreateNewNodeArgs(graph, node_args);

  // Replace the uses of node_args with corresponding new_node_args.
  std::vector<Node*> nodes = {comm_node};
  ReplaceNodeArgs(nodes, node_args, new_node_args); 

  // Create node_arg for event ID.
  auto event_node_arg = &CreateInt64NodeArg(graph, "event_forward_record_before_send");

  // Create node which produces new_node_args from event and node_arg.
  auto name = graph.GenerateNodeName("RecordEvent_ForwardRecordBeforeSend");
  CreateBottleneckNode(graph,
                       "RecordEvent",
                       name,
                       "",
                       event_node_arg,
                       node_args,
                       new_node_args);

  event_name = event_node_arg->Name();

  new_input_names.push_back(event_name);

  return Status::OK();
}

Status AddBackwardWaitAfterRecv1(
    Graph& graph,
    Node* comm_node,
    std::vector<std::string>& new_input_names,
    std::string& event_name) {
  if (!comm_node) {
    // No WaitEvent is inserted, so its input event name is empty.
    event_name = "";
    return Status::OK();
  }

  // Get outputs of Recv nodes. They will be connected to a WaitEvent.
  // Consumers of Recv's will be connected to the newly-added WaitEvent.
  std::vector<NodeArg*> node_args = comm_node->MutableOutputDefs();

  // Declare mirror variables. node_args[i] will be replaced with new_node_args[i].
  std::vector<NodeArg*> new_node_args = CreateNewNodeArgs(graph, node_args);

  // Replace the uses of node_args with corresponding new_node_args.
  std::vector<Node*> nodes;
  for (auto& node: graph.Nodes()) {
    nodes.push_back(&node);
  }
  ReplaceNodeArgs(nodes, node_args, new_node_args); 

  // Create node_arg for event ID.
  auto event_node_arg = &CreateInt64NodeArg(graph, "event_backward_wait_after_recv");

  // Create node which produces new_node_args from event and node_arg.
  auto name = graph.GenerateNodeName("WaitEvent_BackwardWaitAfterRecv");
  CreateBottleneckNode(graph,
                       "WaitEvent",
                       name,
                       "",
                       event_node_arg,
                       node_args,
                       new_node_args);

  event_name = event_node_arg->Name();

  new_input_names.push_back(event_name);

  return Status::OK();
}

Status AddBackwardRecordBeforeSend1(
    Graph& graph,
    Node* comm_node,
    std::vector<std::string>& new_input_names,
    std::string& event_name) {
  if (!comm_node) {
    // No RecordEvent is inserted, so its input event name is empty.
    event_name = "";
    return Status::OK();
  }

  // Get inputs of Send nodes. They will be connected to a RecordEvent as inputs.
  // Outputs of RecordEvent may be connected to Send's input list.
  std::vector<NodeArg*> node_args = comm_node->MutableInputDefs();

  // Declare mirror variables. node_args[i] will be replaced with new_node_args[i].
  std::vector<NodeArg*> new_node_args = CreateNewNodeArgs(graph, node_args);

  // Replace the uses of node_args with corresponding new_node_args.
  std::vector<Node*> nodes = {comm_node};
  ReplaceNodeArgs(nodes, node_args, new_node_args); 

  // Create node_arg for event ID.
  auto event_node_arg = &CreateInt64NodeArg(graph, "event_forward_record_before_send");

  // Create node which produces new_node_args from event and node_arg.
  auto name = graph.GenerateNodeName("RecordEvent_ForwardRecordBeforeSend");
  CreateBottleneckNode(graph,
                       "RecordEvent",
                       name,
                       "",
                       event_node_arg,
                       node_args,
                       new_node_args);

  event_name = event_node_arg->Name();

  new_input_names.push_back(event_name);

  return Status::OK();
}

Status TransformGraphForPipeline(
  Graph& graph,
  std::string& forward_waited_event_name,
  std::string& forward_recorded_event_name,
  std::string& backward_waited_event_name,
  std::string& backward_recorded_event_name,
  std::string& forward_waited_output_name,
  std::string& forward_recorded_output_name,
  std::string& backward_waited_output_name,
  std::string& backward_recorded_output_name,
  std::string& forward_waited_event_after_recv_name,
  std::string& forward_recorded_event_before_send_name,
  std::string& backward_waited_event_after_recv_name,
  std::string& backward_recorded_event_before_send_name) {
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

  ORT_RETURN_IF_ERROR(AddRecordBackward(
    graph,
    send_bw,
    new_input_names,
    new_output_names,
    backward_recorded_event_name,
    backward_recorded_output_name));
  ORT_RETURN_IF_ERROR(AddWaitForward(
    graph,
    recv_fw,
    new_input_names,
    forward_waited_event_name,
    forward_waited_output_name));
  ORT_RETURN_IF_ERROR(AddOrSkipRecordForwardWaitBackward(
    graph,
    send_fw,
    recv_bw,
    new_input_names,
    forward_recorded_event_name,
    backward_waited_event_name,
    forward_recorded_output_name,
    backward_waited_output_name));
  ORT_RETURN_IF_ERROR(AddForwardWaitAfterRecv1(
    graph,
    recv_fw,
    new_input_names,
    forward_waited_event_after_recv_name
  ));
  ORT_RETURN_IF_ERROR(AddForwardRecordBeforeSend1(
    graph,
    send_fw,
    new_input_names,
    forward_recorded_event_before_send_name
  ));
  ORT_RETURN_IF_ERROR(AddBackwardWaitAfterRecv1(
    graph,
    recv_bw,
    new_input_names,
    backward_waited_event_after_recv_name
  ));
  ORT_RETURN_IF_ERROR(AddBackwardRecordBeforeSend1(
    graph,
    send_bw,
    new_input_names,
    backward_recorded_event_before_send_name
  ));

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
  graph.Resolve();

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
