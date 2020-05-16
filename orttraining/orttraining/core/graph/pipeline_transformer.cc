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

NodeArg& CreateInt64NodeArg(Graph& graph, const std::string& name) {
  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  auto actual_name = graph.GenerateNodeArgName(name);
  auto& node_arg = graph.GetOrCreateNodeArg(actual_name, &type_proto);
  return node_arg;
}

void AddInputEvent(Graph& graph, const std::string& op_name,
                   bool is_forward,
                   std::vector<NodeArg*>& input_args,
                   std::vector<std::string>& new_input_names) {
  auto& event_id = CreateInt64NodeArg(graph, op_name + (is_forward ? "_fw" : "_bw") + "_event_id");
  new_input_names.push_back(event_id.Name());
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

// Return mirror variables for node_arg with a different name.
NodeArg& CreateNodeArg(Graph& graph, const NodeArg* base_arg) {
  const auto& new_name = graph.GenerateNodeArgName(base_arg->Name());
  ONNX_NAMESPACE::TypeProto type_proto(*(base_arg->TypeAsProto()));
  if (graph.GetNodeArg(new_name) != nullptr) {
    ORT_THROW("Node with name ", new_name, " already exists.");
  }
  return graph.GetOrCreateNodeArg(new_name, &type_proto);
}

// Return mirror variables for node_args.
// The i-th output element mirrors node_args[i] but with a different name.
std::vector<NodeArg*> CreateMirrorNodeArgs(
  Graph& graph,
  const std::vector<NodeArg*>& node_args) {
  // Declare output.
  std::vector<NodeArg*> new_node_args;

  for (auto& node_arg: node_args) {
    // new_node_arg is a mirror variable of node_arg. They have the same type.
    auto new_node_arg = &CreateNodeArg(graph, node_arg);
    new_node_args.push_back(new_node_arg);
  }

  return new_node_args;
}

// Create a node with input schema [event, input1, input2, ..., inputN] and
// output schema [input1, input2, ..., inputN]
void CreateBottleneckNode(Graph& graph,
                          const std::string& op_type,
                          const std::string& op_name,
                          const std::string& description,
                          NodeArg* event,
                          std::vector<NodeArg*> input_node_args,
                          std::vector<NodeArg*> output_node_args) {
  const auto name = graph.GenerateNodeName(op_name);
  if (event) {
    input_node_args.insert(input_node_args.begin(), event);
  }
  graph.AddNode(
    name,
    op_type,
    description,
    input_node_args,
    output_node_args,
    nullptr /* assume all bottleneck node have no attributes */,
    kMSDomain);
}
  
Node* AddRecordBackward(Graph& graph,
                        Node* backward_send,
                        std::vector<std::string>& new_input_names,
                        std::vector<std::string>& new_output_names,
                        std::string &event_id_tensor_name,
                        std::string &output_tensor_name) {
  std::vector<NodeArg*> input_args;
  AddInputEvent(graph, "RecordEvent", false /* is_forward */, input_args, new_input_names);
  std::vector<NodeArg*> output_args{};

  if (backward_send) {
    // if we have send op in backward pass (at the end of the graph), we make sure the RecordEvent happens
    // after that send by adding Send's outputs to RecordEvent's input list.
    input_args.insert(std::end(input_args),
                      std::begin(backward_send->MutableOutputDefs()),
                      std::end(backward_send->MutableOutputDefs()));
  }
  FindLeafNodes(graph, input_args);

  // Optimizer will be added after applying pipeline transformer. To support partial graph evaluation,
  // the added Record backward op will have its first passthrough input as output.
  ORT_ENFORCE(input_args.size() >= 2, "RecordEvent backward op at least have two inputs.");
  auto& new_output = CreateNodeArg(graph, input_args[1]); // the first input is signal, not passing through
  output_args.push_back(&new_output);
  new_output_names.push_back(new_output.Name());

  Node* record_node = &(graph.AddNode(
    graph.GenerateNodeName("RecordEvent"),
    "RecordEvent",
    "Backward pass",
    input_args,
    output_args,
    nullptr,
    kMSDomain));

  // First input argument is the recorded event ID tensor.
  event_id_tensor_name = input_args.front()->Name();
  // Use first output as output singnal. It will be fetched outside to make sure
  // event operator is computed.
  output_tensor_name = output_args.front()->Name();

  return record_node;
}

Node* AddWaitForward(Graph& graph,
                      Node* /* forward_recv */,
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
  Node* wait_node = &(graph.AddNode(
    graph.GenerateNodeName("WaitEvent"),
    "WaitEvent",
    "",
    input_args,
    output_args,
    nullptr,
    kMSDomain));
  forward_waited_event_name = input_args.front()->Name();
  output_tensor_name = output_args.front()->Name();
  return wait_node;
}

Status AddOrSkipRecordForwardWaitBackward(Graph& graph,
                                          Node* forward_send,
                                          Node* backward_recv,
                                          std::vector<std::string>& new_input_names,
                                          std::string& forward_recorded_event_name,
                                          std::string& backward_waited_event_name,
                                          std::string& forward_output_name,
                                          std::string& backward_output_name) {
  if (!forward_send != !backward_recv){
    ORT_THROW("Graph requires either having both send forward node "
      "and recv backword node, or none of them. Currently the graph "
      "has send forward: ", forward_send, " and recv backward: ", backward_recv);
  }

  if (!forward_send && !backward_recv){
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
    for (auto& output : forward_send->MutableOutputDefs()) {
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

    auto& input = backward_recv->MutableInputDefs()[0];
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

void ReplaceNodeArgs(std::vector<Node*>& nodes,
                     std::vector<NodeArg*>&& node_args,
                     std::vector<NodeArg*>&& new_node_args) {
  ReplaceNodeArgs(nodes, node_args, new_node_args);
}

std::string AddEventBeforeNode(
  Graph& graph,
  Node* node,
  const std::string& event_op_type,
  const std::string& event_op_name,
  const std::string& event_id_name) {
  if (!node) {
    // No event operator is be inserted, so we don't have its input event name.
    return "";
  }

  // Inputs of "node" should be detached.
  // "node" should consume outputs of the added event operator.
  std::vector<NodeArg*> node_args = node->MutableInputDefs();

  // Declare outputs of the added event operator.
  std::vector<NodeArg*> new_node_args = CreateMirrorNodeArgs(graph, node_args);

  // Convert Node* to std::vector<Node*>.
  std::vector<Node*> nodes = {node};

  // Replace node_args[i] with new_node_args[i] in nodes.
  ReplaceNodeArgs(nodes, node_args, new_node_args); 

  // Create node_arg for event ID.
  auto event_node_arg = &CreateInt64NodeArg(graph, event_id_name);

  // Create node which produces new_node_args from event ID and node_args.
  auto name = graph.GenerateNodeName(event_op_name);
  CreateBottleneckNode(graph,
                       event_op_type,
                       name,
                       "",
                       event_node_arg,
                       node_args,
                       new_node_args);

  return event_node_arg->Name();
}

std::string AddEventAfterNode(
  Graph& graph,
  Node* node,
  const std::string& event_op_type,
  const std::string& event_op_name,
  const std::string& event_id_name) {
  if (!node) {
    // No event operator is be inserted, so we don't have its input event name.
    return "";
  }

  // Outputs of "node" should be detached from its consumers.
  // Consumers of "node" should consume outputs of the added event operator.
  std::vector<NodeArg*> node_args = node->MutableOutputDefs();

  // Declare outputs of the added event operator.
  std::vector<NodeArg*> new_node_args = CreateMirrorNodeArgs(graph, node_args);

  // Find consumers of "node"
  for (size_t i = 0; i < node_args.size(); ++i) {
    // Find consumer of "node"'s i-th output.
    std::vector<Node*> consumer_nodes = graph.GetMutableConsumerNodes(
      node_args[i]->Name());
    // Replace node_args[i] with new_node_args[i] in nodes.
    ReplaceNodeArgs(consumer_nodes, {node_args[i]}, {new_node_args[i]}); 
  }

  // Create node_arg for event ID.
  auto event_node_arg = &CreateInt64NodeArg(graph, event_id_name);

  // Create node which produces new_node_args from event ID and node_args.
  auto name = graph.GenerateNodeName(event_op_name);
  CreateBottleneckNode(graph,
                       event_op_type,
                       name,
                       "",
                       event_node_arg,
                       node_args,
                       new_node_args);

  return event_node_arg->Name();
}

Status AddForwardWaitAfterRecv(
    Graph& graph,
    Node* comm_node,
    std::vector<std::string>& new_input_names,
    std::string& event_name) {
  event_name = AddEventAfterNode( 
    graph, comm_node,
    "WaitEvent", "forward_wait_after_recv",
    "forward_wait_after_recv_event_id"); 
  if (event_name.empty()) {
    return Status::OK();
  } else {
    new_input_names.push_back(event_name);
    return Status::OK();
  }
}

Status AddForwardRecordBeforeSend(
    Graph& graph,
    Node* comm_node,
    std::vector<std::string>& new_input_names,
    std::string& event_name) {
  event_name = AddEventBeforeNode(
    graph, comm_node,
    "RecordEvent", "forward_record_before_send",
    "forward_record_before_send_event_id");
  if (event_name.empty()) {
    return Status::OK();
  } else {
    new_input_names.push_back(event_name);
    return Status::OK();
  }
}

Status AddBackwardWaitAfterRecv(
    Graph& graph,
    Node* comm_node,
    std::vector<std::string>& new_input_names,
    std::string& event_name) {
  event_name = AddEventAfterNode(
    graph, comm_node,
    "WaitEvent", "backward_wait_after_recv",
    "backward_wait_after_recv_event_id");
  if (event_name.empty()) {
    return Status::OK();
  } else {
    new_input_names.push_back(event_name);
    return Status::OK();
  }
}

Status AddBackwardRecordBeforeSend(
    Graph& graph,
    Node* comm_node,
    std::vector<std::string>& new_input_names,
    std::string& event_name) {
  event_name = AddEventBeforeNode(
    graph, comm_node,
    "RecordEvent", "backward_record_before_send",
    "backward_record_before_send_event_id");
  if (event_name.empty()) {
    return Status::OK();
  } else {
    new_input_names.push_back(event_name);
    return Status::OK();
  }
}

// Insert WaitEvent and RecordEvent to the partition.
Status TransformGraphForPipeline(
  Graph& graph,
  std::string& forward_waited_event_name,
  std::string& forward_recorded_event_name,
  std::string& backward_waited_event_name,
  std::string& backward_recorded_event_name,
  std::string& forward_wait_output_name,
  std::string& forward_record_output_name,
  std::string& backward_wait_output_name,
  std::string& backward_record_output_name,
  std::string& forward_waited_event_after_recv_name,
  std::string& forward_recorded_event_before_send_name,
  std::string& backward_waited_event_after_recv_name,
  std::string& backward_recorded_event_before_send_name) {
  // Declare nodes according to their topological order.
  Node* forward_wait{nullptr};
  Node* forward_send{nullptr};
  Node* forward_recv{nullptr};
  Node* backward_recv{nullptr};
  Node* backward_send{nullptr};
  Node* backward_record{nullptr};
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Send") {
      if (IsBackward(node)) {
        backward_send = &node;
      } else {
        forward_send = &node;
      }
    } else if (node.OpType() == "Recv") {
      if (IsBackward(node)) {
        backward_recv = &node;
      } else {
        forward_recv = &node;
      }
    }
  }

  // Names to added into this graph's input list.
  // Their value may be provides as "feeds" when calling session.Run(...).
  std::vector<std::string> new_input_names;
  // Names to added into this graph's output list.
  // Their value may be provides as "feeds" when calling session.Run(...).
  std::vector<std::string> new_output_names;

  backward_record = AddRecordBackward(
    graph,
    backward_send,
    new_input_names,
    new_output_names,
    backward_recorded_event_name,
    backward_record_output_name);
  forward_wait = AddWaitForward(
    graph,
    forward_recv,
    new_input_names,
    forward_waited_event_name,
    forward_wait_output_name);
  ORT_RETURN_IF_ERROR(AddOrSkipRecordForwardWaitBackward(
    graph,
    forward_send,
    backward_recv,
    new_input_names,
    forward_recorded_event_name,
    backward_waited_event_name,
    forward_record_output_name,
    backward_wait_output_name));
  
  // Different stages have different patterns of Send & Recv.
  // For different patterns, we add different WaitEvent and Record.
  //
  // After applying all transformations above, the current patterns
  // are listed below.
  //
  // 1. First stage:
  //  Wait -----------------> FW -> --------> Send -> Record ->
  //  Wait -> Recv ---------> BW -> ----------------> Record
  // 2. Middle stage:
  //  Wait -> Recv -> Wait -> FW -> Record -> Send -> Record ->
  //  Wait -> Recv -> Wait -> BW -> Record -> Send -> Record
  // 3. Last stage:
  //  Wait -> Recv ---------> FW ----------------------------->
  //  ----------------------> BW -----------> Send -> Record
  //
  // After applying all transformations below, we will have
  // the following patterns.
  //
  // 1. First stage:
  //  Wait ---------> Wait -> FW -> Record -> Send -> Record ->
  //  Wait -> Recv -> Wait -> BW -> Record ---------> Record
  // 2. Middle stage:
  //  Wait -> Recv -> Wait -> FW -> Record -> Send -> Record ->
  //  Wait -> Recv -> Wait -> BW -> Record -> Send -> Record
  // 3. Last stage:
  //  Wait -> Recv -> Wait -> FW ----------------------------->
  //  ----------------------> BW -> Record -> Send -> Record
  const bool is_first_stage = !forward_recv && forward_send && backward_recv && !backward_send;
  const bool is_middle_stage = forward_recv && forward_send && backward_recv && backward_send;
  const bool is_last_stage = forward_recv && !forward_send && !backward_recv && backward_send;

  // One and only one of is_first_stage, is_middle_stage, and is_last_stage can be true.
  const unsigned int stage_flag_sum = is_first_stage + is_middle_stage + is_last_stage;
  ORT_RETURN_IF_NOT(stage_flag_sum == 1u,
    "The processed graph should be classified into an stage, "
    "but we see more than one true's in the following statements. ",
    "Is first stage? ", is_first_stage, ". ",
    "Is middle stage? ", is_middle_stage, ". ",
    "Is last stage? ", is_last_stage, ".");

  // Now, we add Wait's in parentheses shown below.
  // 1. First stage:
  //  Wait ---------> (Wait) -> FW -> Record -> Send -> Record ->
  //  Wait -> Recv -> Wait -> BW -> Record ---------> Record
  // 2. Middle stage:
  //  Wait -> Recv -> (Wait) -> FW -> Record -> Send -> Record ->
  //  Wait -> Recv -> Wait -> BW -> Record -> Send -> Record
  // 3. Last stage:
  //  Wait -> Recv -> (Wait) -> FW ----------------------------->
  //  ----------------------> BW -> Record -> Send -> Record
  if (is_first_stage) {
    // If first stage, insert after forward WaitEvent.
    ORT_RETURN_IF_ERROR(AddForwardWaitAfterRecv(
      graph,
      forward_wait,
      new_input_names,
      forward_waited_event_after_recv_name
    ));
  } else if (is_middle_stage || is_last_stage) {
    // If middle stage or last stage, insert after forward Recv.
    ORT_RETURN_IF_ERROR(AddForwardWaitAfterRecv(
      graph,
      forward_recv,
      new_input_names,
      forward_waited_event_after_recv_name
    ));
  }

  // Now, we add Record's in parentheses shown below.
  // 1. First stage:
  //  Wait ---------> Wait -> FW -> (Record) -> Send -> Record ->
  //  Wait -> Recv -> Wait -> BW -> Record ---------> Record
  // 2. Middle stage:
  //  Wait -> Recv -> Wait -> FW -> (Record) -> Send -> Record ->
  //  Wait -> Recv -> Wait -> BW -> Record -> Send -> Record
  // 3. Last stage (no change):
  //  Wait -> Recv -> Wait -> FW ----------------------------->
  //  ----------------------> BW -> Record -> Send -> Record
  if (is_first_stage || is_middle_stage) {
    ORT_RETURN_IF_ERROR(AddForwardRecordBeforeSend(
      graph,
      forward_send,
      new_input_names,
      forward_recorded_event_before_send_name
    ));
  }

  // Now, we add Wait's in parentheses shown below.
  // 1. First stage:
  //  Wait ---------> Wait -> FW -> Record -> Send -> Record ->
  //  Wait -> Recv -> (Wait) -> BW -> Record ---------> Record
  // 2. Middle stage:
  //  Wait -> Recv -> Wait -> FW -> Record -> Send -> Record ->
  //  Wait -> Recv -> (Wait) -> BW -> Record -> Send -> Record
  // 3. Last stage (no change):
  //  Wait -> Recv -> Wait -> FW ----------------------------->
  //  ----------------------> BW -> Record -> Send -> Record
  if (is_first_stage || is_middle_stage) {
    ORT_RETURN_IF_ERROR(AddBackwardWaitAfterRecv(
      graph,
      backward_recv,
      new_input_names,
      backward_waited_event_after_recv_name
    ));
  }

  // Now, we add Record's in parentheses shown below.
  // 1. First stage is:
  //  Wait ---------> Wait -> FW -> Record -> Send -> Record ->
  //  Wait -> Recv -> Wait -> BW -> (Record) ---------> Record
  // 2. Middle stage is:
  //  Wait -> Recv -> Wait -> FW -> Record -> Send -> Record ->
  //  Wait -> Recv -> Wait -> BW -> (Record) -> Send -> Record
  // 3. Final pattern of last stage is:
  //  Wait -> Recv -> Wait -> FW ----------------------------->
  //  ----------------------> BW -> (Record) -> Send -> Record
  if (is_first_stage) {
    ORT_RETURN_IF_ERROR(AddBackwardRecordBeforeSend(
      graph,
      backward_record,
      new_input_names,
      backward_recorded_event_before_send_name
    ));
  } else if (is_middle_stage || is_last_stage) {
    ORT_RETURN_IF_ERROR(AddBackwardRecordBeforeSend(
      graph,
      backward_send,
      new_input_names,
      backward_recorded_event_before_send_name
    ));
  }

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
