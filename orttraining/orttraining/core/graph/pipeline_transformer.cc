// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/pipeline_transformer.h"
#include <queue>

#include "core/graph/graph_utils.h"
#include "orttraining/core/framework/distributed_run_context.h"

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
Node& CreateEventNode(Graph& graph,
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

// Create an event operator topologically before the input operator "node".
// All inputs of "node" would be re-wired to the passing-through outputs of the new event operator.
// That is,
//   upstream node -> node -> downstream node
// may become
//   upstream node -> event node (WaitEvent or RecordEvent) -> node -> downstream node
Node& PrependEventNode(
    Graph& graph,                                // Graph which contains "node" and the new event operator.
    Node* node,                                  // The anchor to prepend the new event operator.
    const std::string& event_op_type,            // Type of the new event operator, for example, "WaitEvent" or "RecordEvent".
    const std::string& event_op_name,            // Name's seed of the new event operator, for example, "WaitEvent" or "RecordEvent".
    const std::string& event_id_name,            // Name's seed of the event tensor consumed by the new event operator.
    std::vector<std::string>& new_input_names,   // Values to be added to input list of the transformed graph. Those values can be fed.
    std::vector<std::string>& new_output_names,  // Values to be added to output list of the transformed graph. Those values can be fatched.
    std::string& new_event_name,                 // Actual event name.
    std::string& new_output_name) {              // First output of the created event operator.
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

  // Let outer scope to know the newly-added event tensor name so that TrainingRunner can pass
  // event value in. We also returns the first output generated by this event operator so that
  // TrainingRunner can fetch that value to always run this event operator.
  new_event_name = event_node_arg->Name();
  new_output_name = new_node_args[0]->Name();

  // Allow outer scope to feed to and fetch from this event operator.
  new_input_names.push_back(new_event_name);
  new_output_names.push_back(new_output_name);

  // Create node which produces new_node_args from event ID and node_args.
  return CreateEventNode(graph,
                         event_op_type,
                         event_op_name,
                         "",
                         event_node_arg,
                         node_args,
                         new_node_args);
}

// Create an event operator topologically after the input operator "node".
// All cunsumers of "node" would be re-wired to the passing-through outputs of the new event operator.
// That is,
//   upstream node -> node -> downstream node
// may become
//   upstream node -> node -> event node (WaitEvent or RecordEvent) -> downstream node
Node& AppendEventNode(
    Graph& graph,                                // Graph which contains "node" and the new event operator.
    Node* node,                                  // The anchor to appended the new event operator.
    const std::string& event_op_type,            // Type of the new event operator, for example, "WaitEvent" or "RecordEvent".
    const std::string& event_op_name_seed,       // Name's seed of the new event operator, for example, "WaitEvent" or "RecordEvent".
    const std::string& event_id_name_seed,       // Name's seed of the event tensor consumed by the new event operator.
    std::vector<std::string>& new_input_names,   // Values to be added to input list of the transformed graph. Those values can be fed.
    std::vector<std::string>& new_output_names,  // Values to be added to output list of the transformed graph. Those values can be fatched.
    std::string& new_event_name,                 // Actual event name.
    std::string& new_output_name) {              // First output of the created event operator.
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
  auto event_node_arg = &CreateTypedNodeArg(graph, ONNX_NAMESPACE::TensorProto_DataType_INT64, event_id_name_seed);

  // Let outer scope to know the newly-added event tensor name so that TrainingRunner can pass
  // event value in. We also returns the first output generated by this event operator so that
  // TrainingRunner can fetch that value to always run this event operator.
  new_event_name = event_node_arg->Name();
  new_output_name = new_node_args[0]->Name();

  // Allow outer scope to feed to and fetch from this event operator.
  new_input_names.push_back(new_event_name);
  new_output_names.push_back(new_output_name);

  // Create node which produces new_node_args from event ID and node_args.
  return CreateEventNode(graph,
                         event_op_type,
                         event_op_name_seed,
                         "",
                         event_node_arg,
                         node_args,
                         new_node_args);
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

void FindPipelineLandmarks(
    Graph& graph,
    Node** forward_recv,
    Node** forward_send,
    Node** backward_recv,
    Node** backward_send,
    Node** first_node,
    Node** last_node) {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto idx : node_topology_list) {
    auto node = graph.GetNode(idx);
    if (node->OpType() == "Send") {
      if (IsBackward(*node)) {
        *backward_send = node;
      } else {
        *forward_send = node;
      }
    } else if (node->OpType() == "Recv") {
      if (IsBackward(*node)) {
        *backward_recv = node;
      } else {
        *forward_recv = node;
      }
    }
  }

  *first_node = graph.GetNode(node_topology_list.front());
  *last_node = graph.GetNode(node_topology_list.back());
}

// This function inserts WaitEvent's (or Wait's for short) and RecordEvent's
// (or Record's for short) to the input graph for controlling synchronization
// between (batch, pipeline stage)-pairs.
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
//                             Wait -> FW -> Record -> Wait -> Send -> Record ->
//   Wait -> Recv -> Record -> Wait -> BW -> Record
//  2. Middle stage:
//   Wait -> Recv -> Record -> Wait -> FW -> Record -> Wait -> Send -> Record ->
//   Wait -> Recv -> Record -> Wait -> BW -> Record -> Wait -> Send -> Record
//  3. Last stage:
//   Wait -> Recv -> Record -> Wait -> FW ->
//                                     BW -> Record -> Wait -> Send -> Record
//
// Each Recv, Send, FW, and BW, are surrounded by one Wait and one Record. Wait marks
// the beginning of the surrounded task and Record signals the end of that task.
//
// To explain the meaning of those operators, we take the middle stage's pattern
// as an example:
//
//   Wait-0 -> Recv -> Record-1 -> Wait-2 -> FW -> Record-3 -> Wait-4 -> Send -> Record-5 ->
//   Wait-6 -> Recv -> Record-7 -> Wait-8 -> BW -> Record-9 -> Wait-10 -> Send -> Record-11
//
// Their meanings are listed below.
//
//   Wait-0: Wait until we can start forward Recv.
//   Record-1: Tell others that forward Recv is done.
//
//   Wait-2: Wait until we can start forward pass.
//   Record-3: Tell others that forward computation is done.
//
//   Wait-4: Wait until we can start forward Send.
//   Record-5: Tell others that forward Send is done.
//
//   Wait-6: Wait until we can start backward Recv.
//   Record-7: Tell others that backward Recv is done.
//
//   Wait-8: Wait until we can start backward pass.
//   Record-9: Tell others that backward computation is done.
//
//   Wait-10: Wait until we can start backward Send.
//   Record-11: Tell others that backward Send is done.
Status TransformGraphForPipeline(
    Graph& graph,
    const std::unordered_set<std::string>& weights_to_train,
    pipeline::PipelineTensorNames& pipeline_tensor_names) {
  // Begin node of forward pass.
  Node* forward_recv{nullptr};
  // End node of forward pass.
  Node* forward_send{nullptr};
  // Begin node of backward pass.
  Node* backward_recv{nullptr};
  // End node of backward pass.
  Node* backward_send{nullptr};

  // First node in graph topology.
  Node* first_node{nullptr};
  // Last node in graph topology.
  Node* last_node{nullptr};

  // Find begin/end for Send, Recv, and computation in forward and backward.
  // If there is no Recv in forward/backward, the first forward/backward node is used.
  // If there is no Send in forward/backward, the last forward/backward node is used.
  FindPipelineLandmarks(graph, &forward_recv, &forward_send, &backward_recv, &backward_send, &first_node, &last_node);

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

  // For first and middle stages.
  Node* forward_send_wait{nullptr};
  Node* forward_send_record{nullptr};
  // For middle and last stages.
  Node* forward_recv_wait{nullptr};
  Node* forward_recv_record{nullptr};
  // For middle and last stages.
  Node* backward_send_wait{nullptr};
  Node* backward_send_record{nullptr};
  // For first and middle stages.
  Node* backward_recv_wait{nullptr};
  Node* backward_recv_record{nullptr};
  // For all stages.
  Node* forward_compute_wait{nullptr};
  // For first and middle stages.
  Node* forward_compute_record{nullptr};
  // For first and middle stages.
  Node* backward_compute_wait{nullptr};
  // For all stages.
  Node* backward_compute_record{nullptr};

  // Names to added into this graph's input list.
  // Their values may be provides as "feeds" when calling session.Run(...).
  std::vector<std::string> new_input_names;
  // Names to added into this graph's output list.
  // Their values may be returned as "fetches" when calling session.Run(...).
  std::vector<std::string> new_output_names;

  // Forward Recv
  if (is_middle_stage || is_last_stage) {
    // Insert Wait before Forward-Recv and all nodes.
    forward_recv_wait = &PrependEventNode(
        graph, forward_recv,
        "WaitEvent", "wait_forward_recv", "forward_recv_event_1",
        new_input_names, new_output_names,
        pipeline_tensor_names.forward_recv_waited_event_name,
        pipeline_tensor_names.forward_recv_wait_output_name);
    ORT_ENFORCE(forward_recv_wait);
    ResolveForTraining(graph, weights_to_train);

    // Insert Record after Forward-Recv.
    forward_recv_record = &AppendEventNode(
        graph, forward_recv,
        "RecordEvent", "record_forward_recv", "forward_recv_event_2",
        new_input_names, new_output_names,
        pipeline_tensor_names.forward_recv_recorded_event_name,
        pipeline_tensor_names.forward_recv_record_output_name);
    ORT_ENFORCE(forward_recv_record);
    ResolveForTraining(graph, weights_to_train);
  }

  // Forward Send
  if (is_first_stage || is_middle_stage) {
    // Insert Wait before Forward-Send.
    forward_send_wait = &PrependEventNode(
        graph, forward_send,
        "WaitEvent", "wait_forward_send", "forward_send_event_1",
        new_input_names, new_output_names,
        pipeline_tensor_names.forward_send_waited_event_name,
        pipeline_tensor_names.forward_send_wait_output_name);
    ORT_ENFORCE(forward_send_wait);
    ResolveForTraining(graph, weights_to_train);

    // Insert Record after Forward-Send.
    forward_send_record = &AppendEventNode(
        graph, forward_send,
        "RecordEvent", "record_forward_send", "forward_send_event_2",
        new_input_names, new_output_names,
        pipeline_tensor_names.forward_send_recorded_event_name,
        pipeline_tensor_names.forward_send_record_output_name);
    ORT_ENFORCE(forward_send_record);
    ResolveForTraining(graph, weights_to_train);
  }

  // Backward Recv
  if (is_first_stage || is_middle_stage) {
    // Insert Wait before Backward-Recv.
    backward_recv_wait = &PrependEventNode(
        graph, backward_recv,
        "WaitEvent", "wait_backward_recv", "backward_recv_event_1",
        new_input_names, new_output_names,
        pipeline_tensor_names.backward_recv_waited_event_name,
        pipeline_tensor_names.backward_recv_wait_output_name);
    ORT_ENFORCE(backward_recv_wait);
    ResolveForTraining(graph, weights_to_train);

    // Insert Record after Forward-Recv.
    backward_recv_record = &AppendEventNode(
        graph, backward_recv,
        "RecordEvent", "record_backward_recv", "backward_recv_event_2",
        new_input_names, new_output_names,
        pipeline_tensor_names.backward_recv_recorded_event_name,
        pipeline_tensor_names.backward_recv_record_output_name);
    ORT_ENFORCE(backward_recv_record);
    ResolveForTraining(graph, weights_to_train);
  }

  // Backward Send
  if (is_middle_stage || is_last_stage) {
    // Insert Wait before Backward-Send.
    backward_send_wait = &PrependEventNode(
        graph, backward_send,
        "WaitEvent", "wait_backward_send", "backward_send_event_1",
        new_input_names, new_output_names,
        pipeline_tensor_names.backward_send_waited_event_name,
        pipeline_tensor_names.backward_send_wait_output_name);
    ORT_ENFORCE(backward_send_wait);
    ResolveForTraining(graph, weights_to_train);

    // Insert Record after Backward-Send and all nodes.
    backward_send_record = &AppendEventNode(
        graph, backward_send,
        "RecordEvent", "record_backward_send", "backward_send_event_2",
        new_input_names, new_output_names,
        pipeline_tensor_names.backward_send_recorded_event_name,
        pipeline_tensor_names.backward_send_record_output_name);
    ORT_ENFORCE(backward_send_record);
    ResolveForTraining(graph, weights_to_train);
  }

  // Forward-Compute Wait.
  if (is_first_stage) {
    // Insert one Wait before all nodes.
    forward_compute_wait = &PrependEventNode(
        graph, first_node,
        "WaitEvent", "wait_forward_compute", "forward_compute_event_1",
        new_input_names, new_output_names,
        pipeline_tensor_names.forward_compute_waited_event_name,
        pipeline_tensor_names.forward_compute_wait_output_name);
    ORT_ENFORCE(forward_compute_wait);
    ResolveForTraining(graph, weights_to_train);
  } else {
    // Insert one Wait after Forward-Recv Record.
    forward_compute_wait = &AppendEventNode(
        graph, forward_recv_record,
        "WaitEvent", "wait_forward_compute", "forward_compute_event_1",
        new_input_names, new_output_names,
        pipeline_tensor_names.forward_compute_waited_event_name,
        pipeline_tensor_names.forward_compute_wait_output_name);
    ORT_ENFORCE(forward_compute_wait);
    ResolveForTraining(graph, weights_to_train);
  }

  // Forward-Compute Record
  if (is_first_stage || is_middle_stage) {
    // Insert one Record before Forward-Send Wait.
    forward_compute_record = &PrependEventNode(
        graph, forward_send_wait,
        "RecordEvent", "record_forward_compute", "forward_compute_event_2",
        new_input_names, new_output_names,
        pipeline_tensor_names.forward_compute_recorded_event_name,
        pipeline_tensor_names.forward_compute_record_output_name);
    ORT_ENFORCE(forward_compute_record);
    ResolveForTraining(graph, weights_to_train);
  }

  // Backward-Compute Wait.
  if (is_first_stage || is_middle_stage) {
    // Insert one Wait after Backward-Recv Record
    backward_compute_wait = &AppendEventNode(
        graph, backward_recv_record,
        "WaitEvent", "wait_backward_compute", "backward_compute_event_1",
        new_input_names, new_output_names,
        pipeline_tensor_names.backward_compute_waited_event_name,
        pipeline_tensor_names.backward_compute_wait_output_name);
    ORT_ENFORCE(backward_compute_wait);
    ResolveForTraining(graph, weights_to_train);
  }

  // Backward-Compute Record.
  if (is_first_stage) {
    // Insert one Record after all nodes.
    backward_compute_record = &AppendEventNode(
        graph, last_node,
        "RecordEvent", "record_backward_compute", "backward_compute_event_2",
        new_input_names, new_output_names,
        pipeline_tensor_names.backward_compute_recorded_event_name,
        pipeline_tensor_names.backward_compute_record_output_name);
    ORT_ENFORCE(backward_compute_record);
    ResolveForTraining(graph, weights_to_train);
  } else {
    // Insert one Record before Backward-Send Wait.
    backward_compute_record = &PrependEventNode(
        graph, backward_send_wait,
        "RecordEvent", "record_backward_compute", "backward_compute_event_2",
        new_input_names, new_output_names,
        pipeline_tensor_names.backward_compute_recorded_event_name,
        pipeline_tensor_names.backward_compute_record_output_name);
    ORT_ENFORCE(backward_compute_record);
    ResolveForTraining(graph, weights_to_train);
  }

  ORT_RETURN_IF_ERROR(SetInputsOutputsAndResolve(graph, weights_to_train, new_input_names, new_output_names));
  return Status::OK();
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

// Returns all the pointers to NodeArg in the graph, before applying any 
// partition transformation. 
std::vector<const NodeArg*> GetAllNodeArgs(Graph& graph) {
  std::vector<const NodeArg*> initial_node_args;
  for (size_t i = 0, t = graph.MaxNodeIndex(); i < t; ++i) {
    Node* node = graph.GetNode(i);
    auto& node_outputs = node->MutableOutputDefs();
    for (NodeArg* arg : node_outputs) {
      if (arg == nullptr || !arg->HasTensorOrScalarShape()) 
        continue;
      initial_node_args.push_back(arg);
    }
  }
  return initial_node_args;
}

common::Status AddMetaTensors(int current_rank, int next_rank,
                              Graph& graph,
                              std::vector<std::string>& new_input_names,
                              std::vector<std::string>& new_output_names,
                              std::vector<NodeArg*>& send_input_args,
                              std::vector<NodeArg*>& send_output_args,
                              std::vector<NodeArg*>& recv_input_args,
                              std::vector<NodeArg*>& recv_output_args) {
  std::string cut_index_str = std::to_string(current_rank);

  ORT_RETURN_IF_ERROR(
    AddNewScalarNodeArgAndInitializer<bool>(graph,
                                            "send_input_signal" + cut_index_str,
                                            ONNX_NAMESPACE::TensorProto_DataType_BOOL,
                                            true, /* initializer data */
                                            send_input_args,
                                            new_input_names));
  ORT_RETURN_IF_ERROR(
    AddNewScalarNodeArgAndInitializer<size_t>(graph,
                                              "send_dst_rank" + cut_index_str,
                                              ONNX_NAMESPACE::TensorProto_DataType_INT64,
                                              next_rank, /* initializer data */
                                              send_input_args,
                                              new_input_names));
  ORT_RETURN_IF_ERROR(
    AddNewScalarNodeArgAndInitializer<bool>(graph,
                                            "recv_input_signal" + cut_index_str,
                                            ONNX_NAMESPACE::TensorProto_DataType_BOOL,
                                            true, /* initializer data */
                                            recv_input_args,
                                            new_input_names));
  ORT_RETURN_IF_ERROR(
    AddNewScalarNodeArgAndInitializer<size_t>(graph,
                                              "recv_src_rank" + cut_index_str,
                                              ONNX_NAMESPACE::TensorProto_DataType_INT64,
                                              current_rank, /* initializer data */
                                              recv_input_args,
                                              new_input_names));

  // add output node_arg for send/recv
  AddNewNodeArg(graph, "send_output_signal" + cut_index_str,
                ONNX_NAMESPACE::TensorProto_DataType_BOOL,
                send_output_args, new_output_names);

  AddNewNodeArg(graph, "receive_output_signal" + cut_index_str,
                ONNX_NAMESPACE::TensorProto_DataType_BOOL,
                recv_output_args, new_output_names);

  return Status::OK();
}

common::Status SplitGraphWithMap(Graph& graph,
                                 std::map<Node*, int>& op_to_rank,
                                 int nstages,
                                 std::vector<std::pair<int, int>>& messages,
                                 std::vector<std::pair<Node*, Node*>>& send_nodes,
                                 std::vector<std::pair<Node*, Node*>>& recv_nodes) {

  // forward_messages[s]: all the tensors sent by stage s while executing
  // forward computation.
  std::vector<std::set<const NodeArg*>> forward_messages(nstages);
  // backward_messages[s]: all the tensors sent by stage s while executing
  // backward computation.
  // This may be empty, if the graph is forward only.
  std::vector<std::set<const NodeArg*>> backward_messages(nstages);

  // Tensors that need to be sent from one device to the other.
  // TODO: Should I consider weights here too? 
  // forwarded_tensors[i] = {t, {rank of producer, rank of the last consumer}}
  std::vector<std::pair<const NodeArg*, std::pair<int, int>>> forwarded_tensors;

  // All the tensors that are produced and consumed in the graph.
  auto initial_node_args = GetAllNodeArgs(graph);

  // We create all the tensor replicas in advance using this fuction.
  // A tensor produced in rank r and consumed in rank r', such that r' > r,
  // will have a replica in all ranks r'', such that r'' > r and r'' < r'.
  // tensor_replicas[t][r] contains a pointer to the the replica of t in rank r, 
  // it if exists, or to itself if r is the rank of the producer of r.
  std::map<const NodeArg*, std::vector<NodeArg*>> tensor_replicas;
  auto create_tensor_replica = [&tensor_replicas, &graph](const NodeArg* tensor,
                                                          int consumer_rank) {
    NodeArg& new_receive_output = CreateNodeArg(graph, *tensor);
    const auto* old_shape = tensor->Shape();
    if (old_shape != nullptr) {
      new_receive_output.SetShape(*old_shape);
    }
    // Add value info for this newly added receive_output, for shape propagation
    // when training this partition.
    graph.AddValueInfo(&new_receive_output);
    tensor_replicas[tensor][consumer_rank] = &new_receive_output;
  };

  // Checks whether the tensor is produced and consumed in the forward stage of
  // the computation. 
  auto is_forward = [](const int producer_rank, const int consumer_rank) {
    return producer_rank < consumer_rank;
  };

  // Checks whether the tensor is produced and consumed in the backward stage of
  // the computation. 
  auto is_backward = [](const int producer_rank, const int consumer_rank) {
    return producer_rank > consumer_rank;
  };

  // Find tensors that need to be sent and forwarded.
  for (const NodeArg* node_arg : initial_node_args) {
    // Initialize tensor_replicas data structure.
    auto inserted = tensor_replicas.emplace(
                      std::make_pair(node_arg, std::vector<NodeArg*>(nstages)));
    auto& replicas = (*inserted.first).second;

    // TODO: for now we pretend that inputs are produced in rank 0,
    // but I need to double check how they are handled.
    int producer_rank = 0;
    Node* producer_node = graph.GetMutableProducerNode(node_arg->Name());
    assert(producer_node != nullptr);
    producer_rank = op_to_rank.find(producer_node)->second;
    
    auto consumers = graph.GetMutableConsumerNodes(node_arg->Name());
    if (consumers.size() == 0) { // producer_node == nullptr || ?
      continue;
    }
    
    // This is only handling forwarding in the forward part of the graph.
    int last_consumer_rank_fwd = -1; 
    int last_consumer_rank_bwd = INT_MAX;
    for (Node* consumer : consumers) {
      auto found_rank = op_to_rank.find(consumer);
      assert(found_rank != op_to_rank.end());
      int consumer_rank = found_rank->second;
      // TODO: test case in which a tensor is produced by a fwd op, stashed and
      // sent to the previous rank by a bwd op.
      // For now, I'm assuming that if a tensor is produced by a fwd op and
      // consumed by a bwd op, then producer and consumer are both in the same
      // device. This will not always be the case though. 
      if (!IsBackward(*producer_node) && IsBackward(*consumer)) {
        // They must be on the same device. 
        ORT_ENFORCE(producer_rank == consumer_rank,
                    "Fwd producer and bwd consumer of a tensor must be in the same device.");
      }

      // It is impossible to have a bwd operator producing a tensor consumed
      // by a fwd operator. So, at this point, either both producer and consumer
      // are fwd or both are bwd. Either way, we want to know where are the last
      // consumers of a tensor. 
      if (is_forward(producer_rank, consumer_rank)) {
        last_consumer_rank_fwd = std::max(last_consumer_rank_fwd, consumer_rank);
      } else if (is_backward(producer_rank, consumer_rank)) {
        last_consumer_rank_bwd = std::min(last_consumer_rank_bwd, consumer_rank);
      }

      // Find which tensors need to be sent to the next rank (if it is a forward
      // message) and to previous rank (if it is a backward message).
      if (producer_rank + 1 == consumer_rank) {
        forward_messages.at(producer_rank).insert(node_arg);
      } else if (producer_rank - 1 == consumer_rank) { 
        backward_messages.at(producer_rank).insert(node_arg);
      } 
    }

    // Create all the replicas for this tensor now. We also keep track of which
    // tensors need to be forwarded, and their producer-consumer rank range.
    // The replica of the tensor in the producer rank, is the tensor itself.
    replicas[producer_rank] = const_cast<NodeArg *>(node_arg);
    if (is_forward(producer_rank, last_consumer_rank_fwd)) { 
      for (int r = producer_rank + 1; r <= last_consumer_rank_fwd; ++r) {
        create_tensor_replica(node_arg, r);
      }
      if (last_consumer_rank_fwd - producer_rank > 1) {
        forwarded_tensors.push_back({node_arg, 
                                    {producer_rank, last_consumer_rank_fwd}});
      }
    } else if (is_backward(producer_rank, last_consumer_rank_bwd)) {
      for (int r = producer_rank - 1; r >= last_consumer_rank_bwd; --r) {
        create_tensor_replica(node_arg, r);
      }
      if (producer_rank - last_consumer_rank_bwd > 1) {
        forwarded_tensors.push_back({node_arg, 
                                    {producer_rank, last_consumer_rank_bwd}});
      }
    }
  }

  std::vector<std::string> new_input_names;
  std::vector<std::string> new_output_names;

  for (auto& message : messages) {
    auto current_rank = message.first;
    auto next_rank = message.second;

    // for each pair of ranks, record the inserted input/output args.
    std::vector<NodeArg*> send_input_args;
    std::vector<NodeArg*> send_output_args;
    std::vector<NodeArg*> recv_input_args;
    std::vector<NodeArg*> recv_output_args;
    
    // add attribute data for send/recv
    ONNX_NAMESPACE::AttributeProto tag;
    tag.set_name("tag");
    tag.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
    // currently hard-coded all tag to be 0. May need to change when multiple GPU stream is used.
    tag.set_i(static_cast<int64_t>(0));

    ONNX_NAMESPACE::AttributeProto element_types;
    element_types.set_name("element_types");
    element_types.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);

    ORT_RETURN_IF_ERROR(
      AddMetaTensors(current_rank, next_rank, graph, new_input_names,
                     new_output_names, send_input_args, send_output_args,
                     recv_input_args, recv_output_args));

    // Get all the node_args that need to be sent to the next rank.
    auto& tensors_sent_in_fwd = forward_messages.at(current_rank);
    auto& tensors_sent_in_bwd = backward_messages.at(current_rank);
    auto& tensors_sent = current_rank + 1 == next_rank ? tensors_sent_in_fwd : tensors_sent_in_bwd;

    // Take care of tensors that need to be sent from one device to the other.
    for (const NodeArg* arg : tensors_sent) {
      send_input_args.push_back(const_cast<NodeArg *>(arg));

      // The tensor replica has been created in advance. We query it now
      // because it will be one of the outputs of the receive node in this
      // rank. We also need to add it to the graph.
      NodeArg* new_receive_output = tensor_replicas.at(arg).at(next_rank);
      recv_output_args.push_back(new_receive_output);

      auto dtype = arg->TypeAsProto()->tensor_type().elem_type();
      element_types.add_ints(static_cast<int64_t>(dtype));
    }

    // Take care of tensors that need to be forwarded.
    for (auto& fwding_entry : forwarded_tensors) {
      const NodeArg* tensor = fwding_entry.first;
      auto& range = fwding_entry.second;
      int start = range.first;
      int end = range.second; 

      if (start != current_rank) {
        continue; 
      }

      if (start == end) {
        continue; // Nothing else to do.
      }

      NodeArg* replica = tensor_replicas.at(tensor).at(current_rank);
      NodeArg* next_replica = tensor_replicas.at(tensor).at(next_rank);
      
      ORT_ENFORCE(replica != nullptr && next_replica != nullptr,
                  "Couldn't find replicas of tensor " + tensor->Name());
      if (!std::count(send_input_args.begin(), send_input_args.end(), replica)) {
        send_input_args.push_back(replica);
        recv_output_args.push_back(next_replica);
        auto dtype = tensor->TypeAsProto()->tensor_type().elem_type();
        element_types.add_ints(static_cast<int64_t>(dtype));
      } 

      if (start < end) {
        // Forwarding in forward stage of pipeline
        range.first = start + 1;
      } else if (start > end) {
        // Forwarding in backward stage of pipeline
        range.first = start - 1;
      } 
    }

    // Update the inputs of the next_rank consumers with the right replicas.
    for (auto& it : tensor_replicas) {
      const NodeArg* tensor = it.first; 
      auto& replicas = it.second;
      auto consumers = graph.GetMutableConsumerNodes(tensor->Name());
      for (Node* consumer : consumers) {
        auto found_rank = op_to_rank.find(consumer);
        if (found_rank->second != next_rank) {
          continue;
        }
        NodeArg* replica = replicas.at(next_rank);
        if (replica == nullptr) {
          continue;
        }
        for (auto& input : consumer->MutableInputDefs()) {
          if (input->Name() == tensor->Name()) {
            input = replica;
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

    // Add pair of Send?receive nodes.
    auto& send_node = graph.AddNode(graph.GenerateNodeName("Send"),
                                    "Send", "", send_input_args,
                                    send_output_args, /* output */
                                    &attributes,      /* attribute */
                                    kMSDomain);

    auto& recv_node = graph.AddNode(graph.GenerateNodeName("Recv"),
                                    "Recv", "", recv_input_args,
                                    recv_output_args, /* output */
                                    &attributes,      /* attribute */
                                    kMSDomain);

    ORT_ENFORCE(current_rank != next_rank,
                "Rank cannot send message to itself.");
    if (current_rank < next_rank) {
      send_nodes[current_rank].first = &send_node;
      recv_nodes[next_rank].first = &recv_node;
    } else if (current_rank > next_rank) {
      send_nodes[current_rank].second = &send_node;
      recv_nodes[next_rank].second = &recv_node;
    } 
  }

  ORT_RETURN_IF_ERROR(SetInputsOutputsAndResolve(graph, {}/* weights_to_train*/,
                                                 new_input_names,
                                                 new_output_names));

  return Status::OK();
}

Status ApplyPipelinePartitionToMainGraph(Graph& graph,
                                         std::map<Node*, int>& op_to_rank,
                                         bool is_training,
                                         int pipeline_stage_id,
                                         int nstages) {

  // TODO: we need to do some analysis on the graph and assignment of
  // operators to ranks, to find which messages will happen. For now, we assume
  // that 1) there are always tensors being copied from stage s to s+1,
  // or vice-versa, and that 2) if is_training, then the graph should always be
  //  partitioned into a pipeline in the shape of a V. 
  std::vector<std::pair<int, int>> messages;
  for (int s = 0; s < nstages - 1; ++s) {
    messages.emplace_back(s, s+1);
  }
  if (is_training) {
    for (int s = nstages - 1; s > 0; --s) {
      messages.emplace_back(s, s-1);
    }
  }

  // Get the nodes in topological order before spliting the graph.
  // This ordering will be useful later to remove nodes from the partition.
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // The first Node* of the pair in send_nodes[s] is the Send from s to s+1,
  // i.e., the send for the forward stage. The second is the Send from s to 
  // s-1, i.e., the send for the backward stage. 
  std::vector<std::pair<Node*, Node*>> send_nodes(nstages);
  std::vector<std::pair<Node*, Node*>> recv_nodes(nstages);

  // Split the graph given the mapping of operations to ranks.
  ORT_RETURN_IF_ERROR(
    SplitGraphWithMap(graph, op_to_rank, nstages, messages, send_nodes, recv_nodes));

  // TODO: handle weights that are shared accross many ranks
  // HandleSharedInitializer

  // Generate subgraph / Projection.
  // First remove Send nodes that do not belong to the `pipeline_stage_id`
  // partition. They don't have outgoing edges. Then remove computation nodes
  // that do not belong to the `pipeline_stage_id` partition, in their
  // topological order. Finally, remove Receive nodes that do not belong to the
  // `pipeline_stage_id` partition. At this point, they don't have outgoing
  // edges either.
  for (int s = 0; s < nstages; ++s) { 
    if (s == pipeline_stage_id) {
      continue; // These sends must be kept.
    }
    auto& pair = send_nodes.at(s);
    Node* fwd_send = pair.first;
    if (fwd_send != nullptr) {
      graph.RemoveNode(fwd_send->Index());
    } 
    Node* bwd_send = pair.second;
    if (bwd_send != nullptr) {
      graph.RemoveNode(bwd_send->Index());
    } 
  }

  // Collect all outputs of this partition too.
  std::set<NodeArg*> visited_outputs;
  for (auto it = node_topology_list.rbegin(); it != node_topology_list.rend(); ++it) {
    NodeIndex ni = *it;
    auto found = op_to_rank.find(graph.GetNode(ni));
    ORT_ENFORCE(found != op_to_rank.end(),
                "Found an op without rank.");
    
    if (found->second != pipeline_stage_id) {
      graph.RemoveNode(ni);
    } else {
      auto* node = graph.GetNode(ni);
      auto& consumers = node->MutableOutputDefs();
      for (auto consumer : consumers) {
        if (graph.IsOutput(consumer)) {
          visited_outputs.insert(consumer);
        }
      }
    }
  }

  for (int s = 0; s < nstages; ++s) {
    if (s == pipeline_stage_id) {
      // These receives must be kept.
      continue;
    }
    auto& pair = recv_nodes.at(s);
    Node* fwd_recv = pair.first;
    if (fwd_recv != nullptr) {
      graph.RemoveNode(fwd_recv->Index());
    } 
    Node* bwd_recv = pair.second;
    if (bwd_recv != nullptr) {
      graph.RemoveNode(bwd_recv->Index());
    } 
  }

  graph.SetOutputs({visited_outputs.begin(), visited_outputs.end()});
  graph.SetGraphResolveNeeded();
  graph.SetGraphProtoSyncNeeded();
  graph.Resolve();

  if (!is_training) {
    // In that case, we are done here. The remaining work is only needed when
    // partitioning training graphs.
    return Status::OK();
  }

  // Keep some information about the pipeline.
  auto forward_send = send_nodes.at(pipeline_stage_id).first;
  auto backward_recv = recv_nodes.at(pipeline_stage_id).second;

  if (pipeline_stage_id < nstages - 1) {
    ORT_ENFORCE(forward_send != nullptr, 
                "Something went wrong. Can't find fwd send in rank " +
                std::to_string(pipeline_stage_id));
    ORT_ENFORCE(backward_recv != nullptr,
                "Something went wrong. Can't find bwd recv in rank " + 
                std::to_string(pipeline_stage_id));

    // We need to make sure that the backward receive starts after the forward
    // send, or otherwise the computation will get stuck. We can do this by 
    // adding a control dependency between those two nodes.
    // graph.AddControlEdge(forward_send->Index(), backward_recv->Index());
    //
    // Or we could use the nodes' priorities. By default all the nodes seem to 
    // have a priority of 0, so we could give a high priority to the send and
    // a low priority to the receive. 
    // info.forward_send->SetPriority(static_cast<int>(ExecutionPriority::LOCAL_HIGH));
    // info.backward_recv->SetPriority(static_cast<int>(ExecutionPriority::LOCAL_LOW));
    //
    // However, none of these options seem to be enough. We need to check how 
    // these nodes are being handled in the InsertPipelineOps code. For now, we 
    // use a "hacky" solution, in which we add the output of the forward send 
    // as an input to the backward receive. 
    auto& send_outputs = forward_send->MutableOutputDefs();
    auto& recv_inputs = backward_recv->MutableInputDefs();
    for (int i = 0, s = recv_inputs.size(); i < s; ++s) {
      auto input = recv_inputs.at(i);
      if (input->Name().find("recv_input_signal") != std::string::npos) {
        for (auto output : send_outputs) {
          if (output->Name().find("send_output_signal") != std::string::npos) {
            recv_inputs[i] = output;
            break;
          }
        }
        break;
      }
    }
  }

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
