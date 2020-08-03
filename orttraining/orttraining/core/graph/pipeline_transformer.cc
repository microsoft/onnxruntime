// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/pipeline_transformer.h"
#include <queue>

#include "core/graph/graph_utils.h"

using namespace onnxruntime::common;
using namespace onnxruntime::graph_utils;

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

// Gradient graph can contain some dangling leaf nodes. This function collects
// their first output using the returned vector.
std::vector<NodeArg*> FindBackwardLeafNodes(Graph& graph) {
  // leaf_node_args[i] is the i-th leaf node's first output in the backward
  // pass.
  std::vector<NodeArg*> leaf_node_args;
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
      leaf_node_args.push_back(outputs[0]);
    }
  }

  return leaf_node_args;
};

// This function converts tensor NodeArg to a boolean scalar so that last
// backward RecordEvent doesn't block the early release of large gradient
// tensors. If we connect gradient tensors directly to that RecordEvent,
// we need a memory block (as large as a whole model) to store gradient
// for each trainable tensor until the end of backward pass.
//
// The newly created boolean scalar may be appended to signal_args. If
// signal_args is empty, the source of signal_args[i] would be tensor_args[i].
void ConvertTensorToBoolSignal(
    Graph& graph,
    const std::vector<NodeArg*>& tensor_args,
    std::vector<NodeArg*>& signal_args) {
  for (auto tensor_arg : tensor_args) {
    // Declare the scalar signal this "tensor_arg" will be converted into.
    auto signal_arg = &CreateTypedNodeArg(
        graph,
        ONNX_NAMESPACE::TensorProto_DataType_BOOL,
        "signal_" + tensor_arg->Name());

    // Add the new scalar to user-specified vector.
    signal_args.push_back(signal_arg);

    // Add tensor-to-scalar conversion node.
    const auto name = graph.GenerateNodeName("tensor_to_scalar_signal");
    std::vector<NodeArg*> input_args{tensor_arg};
    std::vector<NodeArg*> output_args{signal_arg};
    graph.AddNode(
        name,
        "Group",
        "",
        input_args,
        output_args,
        nullptr,
        kMSDomain);
  }
}

// Return mirror variables for node_args.
// The i-th output element mirrors node_args[i] but with a different name.
std::vector<NodeArg*> CreateMirrorNodeArgs(
    Graph& graph,
    const std::vector<NodeArg*>& node_args) {
  // Declare output.
  std::vector<NodeArg*> new_node_args;

  for (auto& node_arg : node_args) {
    // new_node_arg is a mirror variable of node_arg. They have the same type.
    assert(node_arg);
    auto new_node_arg = &CreateNodeArg(graph, *node_arg);
    new_node_args.push_back(new_node_arg);
  }

  return new_node_args;
}

// Create a node with input schema [event, input1, input2, ..., inputN] and
// output schema [input1, input2, ..., inputN]
Node& CreateBottleneckNode(Graph& graph,
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

  return graph.AddNode(
      name,
      op_type,
      description,
      input_node_args,
      output_node_args,
      nullptr /* assume all bottleneck node have no attributes */,
      kMSDomain);
}

Node* AddBackwardRecord(Graph& graph,
                        Node* backward_send,
                        std::vector<std::string>& new_input_names,
                        std::vector<std::string>& new_output_names,
                        std::string& event_id_tensor_name,
                        std::string& output_tensor_name) {
  std::vector<NodeArg*> input_args;
  AddNewNodeArg(graph, "backward_recorded_event_id", ONNX_NAMESPACE::TensorProto_DataType_INT64,
                input_args, new_input_names);
  std::vector<NodeArg*> output_args{};

  if (backward_send) {
    // if we have send op in backward pass (at the end of the graph), we make sure the RecordEvent happens
    // after that send by adding Send's outputs to RecordEvent's input list.
    input_args.insert(std::end(input_args),
                      std::begin(backward_send->MutableOutputDefs()),
                      std::end(backward_send->MutableOutputDefs()));
  }

  // Find all leaf nodes' frist inputs. They are used togehter as control edges
  // to determine if backward pass is finished.
  auto backward_leaf_node_args = FindBackwardLeafNodes(graph);

  // For each leaf tensor in the backward pass, we use "Group" operator to
  // convert it to a boolean scalar so that the original leaf's memory can be
  // released earlier.

  // TODO: use full list instead of the first element after changining
  // topological sort to depth-first from inputs.
  std::vector<NodeArg*> sub_backward_leaf_node_args{backward_leaf_node_args[0]};
  ConvertTensorToBoolSignal(graph, sub_backward_leaf_node_args, input_args);

  // Optimizer will be added after applying pipeline transformer. To support partial graph evaluation,
  // the added Record backward op will have its first passthrough input as output.
  ORT_ENFORCE(input_args.size() >= 2, "RecordEvent backward op at least have two inputs.");

  // RecordEvent doesn't have optional input, so it cannot be nullptr.
  assert(input_args[1]);
  auto& new_output = CreateNodeArg(graph, *input_args[1]);  // the first input is signal, not passing through
  output_args.push_back(&new_output);
  new_output_names.push_back(new_output.Name());

  Node* record_node = &CreateBottleneckNode(
      graph, "RecordEvent", "backward_record", "Backward pass", nullptr,
      input_args, output_args);

  // First input argument is the recorded event ID tensor.
  event_id_tensor_name = input_args.front()->Name();
  // Use first output as output singnal. It will be fetched outside to make sure
  // event operator is computed.
  output_tensor_name = output_args.front()->Name();

  return record_node;
}

Node* AddForwardWait(Graph& graph,
                     Node* /* forward_recv */,
                     std::vector<std::string>& new_input_names,
                     std::string& forward_waited_event_name,
                     std::string& output_tensor_name) {
  // Append old_input to input_args and return its pass-through value. Note that
  // input_args and output_args are Wait's inputs and outputs, respectively.
  auto update_wait_input_output = [&](NodeArg* old_input,
                                      std::vector<NodeArg*>& input_args,
                                      std::vector<NodeArg*>& output_args) -> NodeArg& {
    assert(old_input);
    input_args.push_back(old_input);

    auto& wait_output = CreateNodeArg(graph, *old_input);
    output_args.push_back(&wait_output);

    return wait_output;
  };

  std::vector<NodeArg*> input_args;
  std::vector<NodeArg*> output_args;
  AddNewNodeArg(graph, "forward_waited_event_id", ONNX_NAMESPACE::TensorProto_DataType_INT64,
                input_args, new_input_names);
  const std::vector<const NodeArg*>& graph_inputs = graph.GetInputsIncludingInitializers();

  if (graph_inputs.size() == 0) {
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

  Node* wait_node = &CreateBottleneckNode(
      graph, "WaitEvent", "backward_record", "", nullptr,
      input_args, output_args);

  forward_waited_event_name = input_args.front()->Name();
  output_tensor_name = output_args.front()->Name();

  return wait_node;
}

// If the input "graph" is the last pipeline stage, this function don't add any
// event operators.
Status AddOrSkipForwardRecordBackwardWait(Graph& graph,
                                          Node* forward_send,
                                          Node* backward_recv,
                                          std::vector<std::string>& new_input_names,
                                          std::string& forward_recorded_event_name,
                                          std::string& backward_waited_event_name,
                                          std::string& forward_output_name,
                                          std::string& backward_output_name) {
  if (!forward_send != !backward_recv) {
    ORT_THROW(
        "Graph requires either having both send forward node "
        "and recv backward node, or none of them. Currently the graph "
        "has send forward: ",
        forward_send, " and recv backward: ", backward_recv);
  }

  if (!forward_send && !backward_recv) {
    // Last partition doesn't have send forwrad and recv backward. No insert
    // needed.
    return Status::OK();
  }

  // if we have a send forward op followed by a recv backward op, insert
  // WaitEvent and RecordEvent in between.
  Node* record_node = nullptr;
  Node* wait_node = nullptr;

  // Insert RecordEvent
  {
    std::vector<NodeArg*> input_args;
    std::vector<NodeArg*> output_args;
    AddNewNodeArg(graph, "forward_recorded_event_id", ONNX_NAMESPACE::TensorProto_DataType_INT64,
                  input_args, new_input_names);

    // Add send forward op's output as record op's input and output
    for (auto& output : forward_send->MutableOutputDefs()) {
      // send doesn't have optional output, so the node cannot be nullptr.
      assert(output);
      auto& new_output = CreateNodeArg(graph, *output);
      output_args.push_back(&new_output);
      input_args.push_back(output);
    }

    record_node = &CreateBottleneckNode(
        graph, "RecordEvent", "forward_record", "", nullptr,
        input_args, output_args);

    forward_recorded_event_name = record_node->InputDefs()[0]->Name();
    forward_output_name = record_node->OutputDefs()[0]->Name();
  }

  // Insert WaitEvent
  {
    std::vector<NodeArg*> input_args;
    std::vector<NodeArg*> output_args;
    AddNewNodeArg(graph, "backward_waited_event_id", ONNX_NAMESPACE::TensorProto_DataType_INT64,
                  input_args, new_input_names);

    input_args.insert(std::end(input_args),
                      std::begin(record_node->MutableOutputDefs()),
                      std::end(record_node->MutableOutputDefs()));

    auto& input = backward_recv->MutableInputDefs()[0];

    // recv node doesn't have optional input, so the node cannot be nullptr.
    assert(input);
    auto& new_output = CreateNodeArg(graph, *input);
    output_args.push_back(&new_output);
    input = &new_output;

    wait_node = &CreateBottleneckNode(
        graph, "WaitEvent", "backward_wait", "Backward pass", nullptr,
        input_args, output_args);

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
    // Iteration for node_args[i] and new_node_args[i].
    ORT_ENFORCE(node_args[i]->Name() != new_node_args[i]->Name());
    ORT_ENFORCE(node_args[i]->Type() == new_node_args[i]->Type());

    for (auto& node : nodes) {
      for (auto& node_arg : node->MutableInputDefs()) {
        // Only replace when node's input name matches node_args[i].
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
  auto event_node_arg = &CreateTypedNodeArg(graph, ONNX_NAMESPACE::TensorProto_DataType_INT64, event_id_name);

  // Create node which produces new_node_args from event ID and node_args.
  CreateBottleneckNode(graph,
                       event_op_type,
                       event_op_name,
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
  auto event_node_arg = &CreateTypedNodeArg(graph, ONNX_NAMESPACE::TensorProto_DataType_INT64, event_id_name);

  // Create node which produces new_node_args from event ID and node_args.
  CreateBottleneckNode(graph,
                       event_op_type,
                       event_op_name,
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

  Graph::ResolveOptions options;
  // Reserve the training weights. In mixed precision case, without this field,
  // the original fp32 initializers could be removed due to not being used
  // at this point. But we still need to preserve them because later when optimizer is
  // is constructed, the isolated fp32 initializers will be inputs for optimizer.
  options.initializer_names_to_preserve = &weights_to_train;

  return graph.Resolve(options);
}

// This function inserts WaitEvent's and RecordEvent's to the input graph for
// controlling synchronization between (batch, pipeline stage)-pairs.
//
// The input graph is a pipeline's stage, which contains some Send's and Recv's.
//
// For diferent pipeline stages, they have different communication patterns as
// shown below.
//
//  1. First stage:
//                           FW -----------> Send ----------->
//   ------> Recv ---------> BW
//  2. Middle stage:
//           Recv ---------> FW -----------> Send ----------->
//   ------> Recv ---------> BW -----------> Send
//  3. Last stage:
//           Recv ---------> FW ----------------------------->
//   ----------------------> BW -----------> Send
//
// This function inserts some event operators and those patterns become
//
//  1. First stage:
//   Wait ---------> Wait -> FW -> Record -> Send -> Record ->
//   Wait -> Recv -> Wait -> BW -> Record ---------> Record
//  2. Middle stage:
//   Wait -> Recv -> Wait -> FW -> Record -> Send -> Record ->
//   Wait -> Recv -> Wait -> BW -> Record -> Send -> Record
//  3. Last stage:
//   Wait -> Recv -> Wait -> FW ----------------------------->
//   ----------------------> BW -> Record -> Send -> Record
//
// To explain the meaning of those operators, we take the middle stage's pattern
// as an example:
//
//   Wait-0 -> Recv -> Wait-1 -> FW -> Record-0 -> Send -> Record-1 ->
//   Wait-2 -> Recv -> Wait-3 -> BW -> Record-2 -> Send -> Record-3
//
// Their meanings are listed below.
//
//   Wait-0: Wait until we can start reciving forward data.
//   Wait-1: Wait until we can start forward pass.
//   Record-0: Tell others that forward pass is done.
//   Record-1: Tell others that forward result has been passed to another stage.
//   Wait-2: Wait until we can start reciving backward data.
//   Wait-3: Wait until we can start backward bass.
//   Record-2: Tell others that backward pass is done.
//   Record-3: Tell others that backward result has been passed to another stage.
Status TransformGraphForPipeline(
    Graph& graph,
    const std::unordered_set<std::string>& weights_to_train,
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
  // Their values may be provides as "feeds" when calling session.Run(...).
  std::vector<std::string> new_input_names;
  // Names to added into this graph's output list.
  // Their values may be returned as "fetches" when calling session.Run(...).
  std::vector<std::string> new_output_names;

  backward_record = AddBackwardRecord(
      graph,
      backward_send,
      new_input_names,
      new_output_names,
      backward_recorded_event_name,
      backward_record_output_name);
  forward_wait = AddForwardWait(
      graph,
      forward_recv,
      new_input_names,
      forward_waited_event_name,
      forward_wait_output_name);
  ORT_RETURN_IF_ERROR(AddOrSkipForwardRecordBackwardWait(
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
  const bool is_first_stage = !forward_recv && forward_send && backward_recv && !backward_send;
  const bool is_middle_stage = forward_recv && forward_send && backward_recv && backward_send;
  const bool is_last_stage = forward_recv && !forward_send && !backward_recv && backward_send;

  // One and only one of is_first_stage, is_middle_stage, and is_last_stage can be true.
  const unsigned int stage_flag_sum = is_first_stage + is_middle_stage + is_last_stage;
  ORT_RETURN_IF_NOT(stage_flag_sum == 1u,
                    "The processed graph should be classified into a stage, "
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
        forward_waited_event_after_recv_name));
  } else if (is_middle_stage || is_last_stage) {
    // If middle stage or last stage, insert after forward Recv.
    ORT_RETURN_IF_ERROR(AddForwardWaitAfterRecv(
        graph,
        forward_recv,
        new_input_names,
        forward_waited_event_after_recv_name));
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
        forward_recorded_event_before_send_name));
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
        backward_waited_event_after_recv_name));
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
        backward_recorded_event_before_send_name));
  } else if (is_middle_stage || is_last_stage) {
    ORT_RETURN_IF_ERROR(AddBackwardRecordBeforeSend(
        graph,
        backward_send,
        new_input_names,
        backward_recorded_event_before_send_name));
  }

  ORT_RETURN_IF_ERROR(SetInputsOutputsAndResolve(graph, weights_to_train, new_input_names, new_output_names));
  return Status::OK();
}

// This function is used when you want to create a scalar constant in a graph.
// It may create a NodeArg so that other Node can references its value.
// It also cerates an initializer to store its value.
template <typename T>
void AddNewScalarNodeArgAndInitializer(Graph& graph,
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
      ORT_THROW("pipeline partition unsupported 'type' value: ", type);
  }
  graph.AddInitializedTensor(proto_data);
}

// Given a node, this function finds all its connected nodes (consumer nodes and producer nodes) and
// connected inputs and outputs in the given graph, then adds them to the containers passed in.
Status FindAllConnectedNodes(Graph& graph,
                             Node* node,
                             std::vector<Node*>& connected_nodes,
                             std::set<NodeArg*>& connected_inputs,
                             std::set<NodeArg*>& connected_outputs
                             ) {
  assert(node);
  ORT_THROW_IF_ERROR(node->ForEachMutableWithIndex(
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

  ORT_THROW_IF_ERROR(node->ForEachMutableWithIndex(
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
    if (i > from_stage && node_groups[node_group_index].stage_id == (i + 1)) {
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

  ORT_ENFORCE(node_group_index == node_groups.size(), "Not all nodes are updated with new initializer.");

  return Status::OK();
}

// Traverse the graph to find out all connected elements in the graph from start_node. The traverse treats the graph as an
// undirected graph.
void TraverseGraphWithConnectedElement(Graph& graph,
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
      ORT_THROW_IF_ERROR(FindAllConnectedNodes(graph, node, connected_nodes, visited_inputs, visited_outputs));

      for (auto n : connected_nodes) {
        ORT_ENFORCE(n != nullptr, "Found nullptr in searching for connected nodes");
        node_queue.push(n);
      }
    }
  }
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
    TraverseGraphWithConnectedElement(graph,
                                      traverse_start_node,
                                      visited_nodes,
                                      visited_inputs,
                                      visited_outputs);

    for (const auto input : visited_inputs) {
      // If the node is an input instead of an initializer, continue
      if (!graph.IsInitializedTensor(input->Name())){
        continue;
      }

      // group all consumer nodes that shares the same input initializer in visited_consumer_nodes
      std::vector<Node*> consumer_nodes = graph.GetMutableConsumerNodes(input->Name());
      std::vector<Node*> visited_consumer_nodes;
      for(auto consumer_node : consumer_nodes){
        if (visited_nodes.count(consumer_node) != 0){
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
    AddNewScalarNodeArgAndInitializer<bool>(graph,
                                            "send_input_signal" + cut_index_str,
                                            ONNX_NAMESPACE::TensorProto_DataType_BOOL,
                                            true, /* initializer data */
                                            send_input_args,
                                            new_input_names);
    AddNewScalarNodeArgAndInitializer<bool>(graph,
                                            "recv_input_signal" + cut_index_str,
                                            ONNX_NAMESPACE::TensorProto_DataType_BOOL,
                                            true, /* initializer data */
                                            recv_input_args,
                                            new_input_names);

    AddNewScalarNodeArgAndInitializer<size_t>(graph,
                                              "send_dst_rank" + cut_index_str,
                                              ONNX_NAMESPACE::TensorProto_DataType_INT64,
                                              index + 1, /* initializer data */
                                              send_input_args,
                                              new_input_names);
    AddNewScalarNodeArgAndInitializer<size_t>(graph,
                                              "recv_src_rank" + cut_index_str,
                                              ONNX_NAMESPACE::TensorProto_DataType_INT64,
                                              index, /* initializer data */
                                              recv_input_args,
                                              new_input_names);
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
        ORT_THROW("Cannot find producer node of node_arg with name: ", id.node_arg_name, ". Wrong cutting infomation.");
      }

      // once we find out the producer node for id.node_arg_name, find which output index that leads
      // to id.node_arg_name
      int upstream_nodes_output_index{-1};
      producer_node->ForEachWithIndex(
          producer_node->OutputDefs(),
          [&](const NodeArg& def, size_t index) {
            if (def.Name() == id.node_arg_name) {
              upstream_nodes_output_index = static_cast<int>(index);
            }
            return Status::OK();
          });

      if (upstream_nodes_output_index < 0) {
        ORT_THROW("Node with name: ", producer_node->Name(),
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
      const auto old_shape = *(updated_node_arg->Shape());
      new_receive_output.SetShape(old_shape);
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
    const int num_attributes = 2;  // two attributes: tag and element_types
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
  TraverseGraphWithConnectedElement(graph, start_node,
                                    visited_nodes, visited_inputs, visited_outputs);

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
  // loss the hit to resolve symbolic shapes. For example, if an original
  // input's shape is [batch, sequence, 1024], that input should be provided as
  // a feed to all pipeline stages. Otherwise, we don't know the actual values
  // of "batch" and "sequence".
  //
  // graph.SetInputs({visited_inputs.begin(), visited_inputs.end()});

  // update the grah with only visited outputs
  graph.SetOutputs({visited_outputs.begin(), visited_outputs.end()});
  graph.SetGraphResolveNeeded();
  graph.SetGraphProtoSyncNeeded();

  return graph.Resolve();
}

Status ApplyPipelinePartitionToMainGraph(
    Graph& graph,
    const std::vector<TrainingSession::TrainingConfiguration::CutInfo>& cut_info,
    size_t pipeline_stage_id,
    size_t num_pipeline_stage) {
  size_t split_count = cut_info.size();

  if (num_pipeline_stage != split_count + 1) {
    ORT_THROW("Wrong pipeline partition cutting info. Total pipeline stage number is ",
              num_pipeline_stage,
              ", cut info length is: ",
              split_count);
  }

  std::vector<Node *> send_nodes, recv_nodes;
  send_nodes.reserve(split_count);
  recv_nodes.reserve(split_count);

  // Split the graph by cutting edges specified in cut_info. After this function, the graph will be
  // composed of several disconnected partitions.
  ORT_RETURN_IF_ERROR(SplitGraph(graph, cut_info, send_nodes, recv_nodes));

  if (send_nodes.size() != split_count || recv_nodes.size() != split_count) {
    ORT_THROW("Split error: not all cut has Send and Recv inserted. Send node count: ",
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

  if (pipeline_stage_id == 0){
    // For the first stage, there should be no recv node, and the send node contained in graph should match the first
    // send_node inserted during split.
    ORT_ENFORCE(recv_node == nullptr, "Error: first stage contains Recv node in forward pass.");
    ORT_ENFORCE(send_node == send_nodes[0],
                "Error: first stage doesn't contain the right Send node. Possibly CutInfo data is wrong.");
  }
  else if (pipeline_stage_id == split_count){
    // For the last stage, there should be no send node, and the recv node contained in graph should match the last
    // recv_node inserted during split.
    ORT_ENFORCE(recv_node == recv_nodes.back(),
                "Error: last stage doesn't contain the right Recv node. Possibly CutInfo data is wrong.");
    ORT_ENFORCE(send_node == nullptr, "Error: last stage contains Send node in forward pass.");
  } else {
    // For stages in the middle, i-th stage should contain recv node that matches the (i-1)-th inserted recv node, and the i-th
    // inserted send node.
    ORT_ENFORCE(recv_node == recv_nodes[pipeline_stage_id - 1],
                "Error: stage ", pipeline_stage_id, " doesn't contain the right Recv node. Possibly CutInfo data is wrong.");
    ORT_ENFORCE(send_node == send_nodes[pipeline_stage_id],
                "Error: stage ", pipeline_stage_id, " doesn't contain the right Send node. Possibly CutInfo data is wrong.");
  }

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
