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

// Fill TensorProto with zeros.
void FillZeros(
    const ONNX_NAMESPACE::TensorProto_DataType& type,  // Type of tensor's elements.
    const size_t& size,                                // Number of scalar elements in the tensor.
    ONNX_NAMESPACE::TensorProto& tensor_proto) {
  std::vector<char> buffer;
  switch (type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      buffer.resize(size * sizeof(float));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      buffer.resize(size * sizeof(double));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      buffer.resize(size * sizeof(int16_t));
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      buffer.resize(size * sizeof(int8_t));
      break;
    default:
      ORT_THROW("This tensor type doesn't have default value.");
  }

  tensor_proto.set_raw_data(buffer.data(), buffer.size());
}

// When we partition the model into different pipeline stages,
// usually only the last pipeline stage has graph-level outputs
// such as loss. Here we create fake outputs for the first and
// all intermediate stages so that the output schema remains
// the same across all pipeline stages.
// The fake output's schema is determined by "sliced_schema[output_name]".
void CreateFakeOutput(
    Graph& graph,                   // the graph of a pipeline stage.
    const std::string output_name,  // The fake output's name to add to the graph.
    const std::unordered_map<std::string, std::vector<int>>& sliced_schema) {
  // Type of the considered graph output.
  const auto output_type_proto = graph.GetNodeArg(output_name)->TypeAsProto();
  ORT_ENFORCE(output_type_proto->has_tensor_type(), "Only tensors are supported.");
  ORT_ENFORCE(output_type_proto->tensor_type().has_elem_type(), "Tensor must have a valid element type.");

  // Element type of the considered graph output.
  const ONNX_NAMESPACE::TensorProto_DataType element_type =
      static_cast<ONNX_NAMESPACE::TensorProto_DataType>(output_type_proto->tensor_type().elem_type());

  // Create type for fake output.
  ONNX_NAMESPACE::TypeProto type_proto;
  type_proto.mutable_tensor_type()->set_elem_type(element_type);
  auto& seed_node_arg = graph.GetOrCreateNodeArg(output_name + "_seed", &type_proto);

  // Create fake output.
  ONNX_NAMESPACE::TensorProto tensor_proto;
  tensor_proto.set_name(seed_node_arg.Name());
  tensor_proto.set_data_type(element_type);
  int64_t reference_size = 1;

  // Shape of a variable can be found in the ONNX model or a dictionary defined by the user.
  // If the dictionary contains a shape, we use that shape as the actual output shape.
  // Otherwise, we extract the shape loaded from the ONNX model.
  ORT_ENFORCE(sliced_schema.find(output_name) != sliced_schema.end());
  // Get shape passed in by user.
  auto shape = sliced_schema.at(output_name);
  for (auto d : shape) {
    tensor_proto.add_dims(d);
    reference_size *= d;
  }

  FillZeros(element_type, reference_size, tensor_proto);

  // Assign dummy values.
  for (int64_t i = 0; i < reference_size; ++i) {
    tensor_proto.add_float_data(0.0f);
  }
  graph.AddInitializedTensor(tensor_proto);

  // Make a node to produce output.
  auto output_node_arg = graph.GetNodeArg(output_name);
  std::vector<NodeArg*> input_args{&seed_node_arg};
  std::vector<NodeArg*> output_args{output_node_arg};
  auto node_name = graph.GenerateNodeName("Identity");
  graph.AddNode(node_name, "Identity", "Fake loss node.", input_args, output_args);
}

void GetPipelineRecvInput(const Graph& graph, std::string& node_arg_name) {
  for (auto& node : graph.Nodes()) {
    if (!node.OpType().compare("Recv")) {
      node_arg_name = node.InputDefs()[0]->Name();
      return;
    }
  }
}

void GetPipelineSendOutput(const Graph& graph, std::string& node_arg_name) {
  for (auto& node : graph.Nodes()) {
    if (!node.OpType().compare("Send")) {
      // send op should always have an output, which is the OutputSignal.
      node_arg_name = node.OutputDefs()[0]->Name();
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
  // Avoid adding non-existent argumements as new inputs,
  // this would trigger a failure in the shape inference phase of graph resolve.
  std::vector<NodeArg*> node_args;
  std::copy_if(node->MutableOutputDefs().begin(), node->MutableOutputDefs().end(),
               std::back_inserter(node_args),
               [](NodeArg* arg) { return arg->Exists(); });

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
    const bool keep_original_output_schema,
    const std::unordered_set<std::string>& weights_to_train,
    const std::unordered_map<std::string, std::vector<int>>& sliced_schema,
    const std::vector<std::string>& expected_output_names,
    Graph& graph,
    pipeline::PipelineTensorNames& pipeline_tensor_names) {
  // If original outputs are not needed, sliced scheam and expected output
  // name list should be empty. Otherwise, for non-existing outputs, fake
  // output may be created according to shapes in sliced_schema. Note that
  // sliced_schema["X"] is the shape of sliced tensor named "X".
  if (!keep_original_output_schema) {
    // Enforce unused variable is empty.
    ORT_ENFORCE(sliced_schema.empty());
    // Enforce unused variable is empty.
    ORT_ENFORCE(expected_output_names.empty());
  }

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

  // If user wants to keep original outputs, we add fake outputs if the
  // current graph partition doesn't produce them.
  if (keep_original_output_schema) {
    for (size_t i = 0; i < expected_output_names.size(); ++i) {
      const std::string name = expected_output_names[i];

      auto producer = graph.GetProducerNode(name);
      if (producer) {
        // This partition generates original output.
        // There is no need to add a fake one.
        continue;
      }

      // For each graph output not produced by this pipeline stage,
      // we create a fake tensor with user-specified shape.
      CreateFakeOutput(graph, name, sliced_schema);
      new_output_names.push_back(name);
    }
  }

  ORT_RETURN_IF_ERROR(SetInputsOutputsAndResolve(graph, weights_to_train, new_input_names, new_output_names));
  return Status::OK();
}

// See header file for this function's doc.
void SetDataDependency(
    Graph& graph,
    Node& postponed_node,                             // node should happen after computing dependent_args.
    const std::vector<NodeArg*>& dependent_node_args  // extra data-dependency to add to "postponed_node"
) {
  // "postponed_node"'s original inputs + "dependent_args"
  std::vector<NodeArg*> pass_through_inputs;
  // the mirror of "postponed_node"'s original inputs + "dependent_args"
  std::vector<NodeArg*> pass_through_outputs;

  // Step 1: For each downstream operator, we add its original inputs to PassThrough.
  //          Then, we replace the original inputs with the corresponding outputs produced by PassThrough.
  for (auto& node_arg : postponed_node.MutableInputDefs()) {
    // Skip non-existing inputs.
    if (!node_arg->Exists())
      continue;
    ORT_ENFORCE(node_arg, "Non-existing NodeArg cannot be used as input of PassThrough.");
    NodeArg* mirror_node_arg = &CreateNodeArg(graph, *node_arg);
    pass_through_inputs.push_back(node_arg);
    pass_through_outputs.push_back(mirror_node_arg);
    node_arg = mirror_node_arg;
  }

  // Step 2: Add dependents to PassThrough so that PassThrough will be computed after generating dependents.
  for (auto& node_arg : dependent_node_args) {
    ORT_ENFORCE(node_arg->Exists(), "Non-existing NodeArg cannot be used as input of PassThrough.");
    // Retrieve NodeArg by name.
    NodeArg* mirror_node_arg = &CreateNodeArg(graph, *node_arg);
    pass_through_inputs.push_back(node_arg);
    // This mirror variable is not used by optimizer, so we don't need node_arg = mirror_node_arg.
    pass_through_outputs.push_back(mirror_node_arg);
  }

  // Step 3: Create PassThrough.
  graph.AddNode(graph.GenerateNodeName("OrderingPassThrough"),
                "PassThrough", "", pass_through_inputs, pass_through_outputs, nullptr, kMSDomain);

  graph.Resolve();
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
        std::vector<Node*> consumer_nodes = graph.GetMutableConsumerNodes(node_arg.Name());
        connected_nodes.insert(std::end(connected_nodes), consumer_nodes.begin(), consumer_nodes.end());
        if (graph.IsOutput(&node_arg)) {
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

  for (const auto node_arg : graph.GetInputs()) {
    if (!node_arg->Exists()) {
      continue;
    }
    NodeArg* mutable_node_arg = graph.GetNodeArg(node_arg->Name());
    visited_inputs.insert(mutable_node_arg);
  }

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
std::set<const NodeArg*> GetAllNodeArgs(const Graph& graph) {
  std::set<const NodeArg*> initial_node_args;
  const auto& all_nodes = graph.Nodes();
  for (const auto& node : all_nodes) {
    const auto& node_outputs = node.OutputDefs();
    for (const NodeArg* arg : node_outputs) {
      if (arg == nullptr || !arg->HasTensorOrScalarShape() || !arg->Exists())
        continue;
      initial_node_args.emplace(arg);
    }
  }
  return initial_node_args;
}

common::Status AddMetaTensors(const int current_stage, const int next_stage,
                              Graph& graph,
                              std::vector<std::string>& new_input_names,
                              std::vector<std::string>& new_output_names,
                              std::vector<NodeArg*>& send_input_args,
                              std::vector<NodeArg*>& send_output_args,
                              std::vector<NodeArg*>& recv_input_args,
                              std::vector<NodeArg*>& recv_output_args,
                              const std::vector<int>& stage_to_rank) {
  std::string cut_index_str = std::to_string(current_stage);

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
                                                stage_to_rank.at(next_stage), /* initializer data */
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
                                                stage_to_rank.at(current_stage), /* initializer data */
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

// Given the set of tensors sent (tensors_sent_in_forward) in one particular
// stage, add them as inputs of the send operator of that stage (send_input_args)
// and as outputs of the receive operator (recv_output_args) of the next stage.
// Note that the received tensor will be a replica, i.e., stored in tensor_replicas.
void SendProducedTensors(ONNX_NAMESPACE::AttributeProto& element_types,
                         const std::set<const NodeArg*>& tensors_sent_in_forward,
                         std::vector<NodeArg*>& send_input_args,
                         std::vector<NodeArg*>& recv_output_args,
                         const std::map<const NodeArg*, std::vector<NodeArg*>>& tensor_replicas,
                         const int next_stage) {
  for (const NodeArg* arg : tensors_sent_in_forward) {
    send_input_args.push_back(const_cast<NodeArg*>(arg));

    // The tensor replica has been created in advance. We query it now
    // because it will be one of the outputs of the receive node in this
    // stage. We also need to add it to the graph.
    NodeArg* new_receive_output = tensor_replicas.at(arg).at(next_stage);
    recv_output_args.push_back(new_receive_output);

    auto dtype = arg->TypeAsProto()->tensor_type().elem_type();
    element_types.add_ints(static_cast<int64_t>(dtype));
  }
}

// Sends tensors that need to be copied from stage to stage, from the stage of
// their producer to the stage of their last consumer. In particular,
// current_stage produces or receives the tensors it sends.
// The forwarded tensor is added as input of the send operator of current_stage
// (send_input_args) and as output of the receive operator of next_stage
// (recv_output_args). Note that the sent/received tensors will be a replicas,
// i.e., stored in tensor_replicas.
// forwarded_tensors contans tensors that need to be sent from one device to the
// other. For example,
// forwarded_tensors[i] = {t, {stage of producer - s0, stage of the last consumer - s1}}
// means that a tensor t, is produced in stage s0 and consumed for the last
// time in stage s1, where s1 > s0 + 1. This tensor will be copied from stage
// to stage until s1.
void SendForwardedTensors(ONNX_NAMESPACE::AttributeProto& element_types,
                          std::vector<NodeArg*>& send_input_args,
                          std::vector<NodeArg*>& recv_output_args,
                          const std::map<const NodeArg*, std::vector<NodeArg*>>& tensor_replicas,
                          std::vector<std::pair<const NodeArg*, std::pair<int, int>>>& forwarded_tensors,
                          const int current_stage,
                          const int next_stage) {
  for (auto& forwarding_entry : forwarded_tensors) {
    const NodeArg* tensor = forwarding_entry.first;
    auto& range = forwarding_entry.second;
    int start = range.first;
    int end = range.second;

    if (start != current_stage) {
      continue;
    }

    if (start == end) {
      continue;  // Nothing else to do.
    }

    NodeArg* replica = tensor_replicas.at(tensor).at(current_stage);
    NodeArg* next_replica = tensor_replicas.at(tensor).at(next_stage);

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
    }
    // TODO(jufranc): Forwarding in backward stage of pipeline.
    // else if (start > end) {
    //   range.first = start - 1;
    // }
  }
}

// Whenever a tensor is sent to a different stage, this function updates the
// inputs of the consumers of that tensor that are assigned to that stage.
void UpdateInputsOfConsumers(Graph& graph,
                             std::map<const NodeArg*, std::vector<NodeArg*>>& tensor_replicas,
                             const std::map<const Node*, int>& op_to_stage,
                             const int next_stage) {
  for (auto& it : tensor_replicas) {
    const NodeArg* tensor = it.first;
    auto& replicas = it.second;
    auto consumers = graph.GetMutableConsumerNodes(tensor->Name());
    for (Node* consumer : consumers) {
      const auto found_stage = op_to_stage.find(consumer);
      if (found_stage->second != next_stage) {
        continue;
      }
      NodeArg* replica = replicas.at(next_stage);
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
}

// Checks whether the tensor is produced and consumed in the forward stage of
// the computation.
bool IsForwardComputation(const int producer_stage, const int consumer_stage) {
  return producer_stage < consumer_stage;
};

// Checks whether the tensor is produced and consumed in the backward stage of
// the computation.
bool IsBackwardComputation(const int producer_stage, const int consumer_stage) {
  return producer_stage > consumer_stage;
};

// We create all the tensor replicas in advance using this function.
void CreateTensorReplica(Graph& graph,
                         std::map<const NodeArg*, std::vector<NodeArg*>>& tensor_replicas,
                         const NodeArg* tensor,
                         int consumer_stage) {
  auto type_proto = tensor->TypeAsProto();
  ORT_ENFORCE(type_proto->value_case() == TypeProto::kTensorType,
              "Only tensors are supported in training.");

  NodeArg& new_receive_output = CreateNodeArg(graph, *tensor);
  const auto* old_shape = tensor->Shape();
  if (old_shape != nullptr) {
    new_receive_output.SetShape(*old_shape);
  }
  // Add value info for this newly added receive_output, for shape propagation
  // when training this partition.
  graph.AddValueInfo(&new_receive_output);
  tensor_replicas.at(tensor).at(consumer_stage) = &new_receive_output;
}

// Splits the graph into disconnected subgraphs, each subgraph representing the
// a pipeline stage.
//
// Inputs:
//   - graph: a forward graph including the loss function.
//   - op_to_stage: a map between operators and stage identifiers. Each operator
// is represented by a pointer to a Node that belongs to `graph`.
//   - num_stages: the resulting number of pipeline stages.
//   - messages: a vector of pairs of stage identifiers describing the allowed
// communication. Currently, this will be {(0, 1),...,(num_stages - 2, num_stages - 1)},
// meaning that pipeline stage 0 is only allowed to send messages to stage 1,
// stage 1 to stage 2, and so on. Once we allow partition after AD, or more
// general forms of partition, this vector will contain other pairs of stages.
//   - send_nodes: a container for the Send nodes we add to the graph. E.g,
// send_nodes[i] represents the sending of tensors from stage i to stage i+1 and
// belongs to the partition i.
//   - receive_nodes: a container for the Recv nodes we add to the graph. E.g.,
// receive_nodes[i] represents the receiving of tensors from stage i in stage
// i+1 and belongs to the partition i+1.
//   - stage_to_rank: a mapping between stage id and rank id. This is needed
// because one stage may be composed of 2 ranks (due to horizontal parallelism).
//
// The algorithm used here can be divided into four major steps:
//   1. We collect all tensors that need be sent from stage to stage.
//   2. We collect all tensors that need to be forwarded, i.e., tensors that
// need to be sent by a stage even though it wasn't produced in that stage.
//   3. We create replicas of all the tensors that will be sent.
//   4. Finally, for each pair of stages (origin, destination), we perform the
// following steps:
//     a) Create all the meta tensors required (e.g., signals, and source and
// destination ranks).
//     b) Add as input of the Send operator, all the tensors collected in step 1,
// and as output of the respective Receive operator.
//     c) If origin, needs to send any received tensor (a forwarded tensor),
// mark the replica of that tensor as input and output of the same Send and
// Receive operators, respectively.
//     d) Update the consumers of the received tensors to read from the replicas
// of such tensors.
//     e) Finally, create the Send and Receive nodes.
common::Status SplitGraphWithOperatorToStageMap(Graph& graph,
                                                const std::map<const Node*, int>& op_to_stage,
                                                const int num_stages,
                                                const std::vector<std::pair<int, int>>& messages,
                                                std::vector<Node*>& send_nodes,
                                                std::vector<Node*>& receive_nodes,
                                                const std::vector<int>& stage_to_rank) {
  // forward_messages stores all the tensors that will be sent by any stage.
  // For example, forward_messages[s] is a set of all the tensors sent by stage
  // s while executing the forward computation.
  std::vector<std::set<const NodeArg*>> forward_messages(num_stages);
  // TODO(jufranc): once we start using this function on the training graph,
  // we need to keep backward_messages[s] too.

  // Tensors that need to be sent from one device to the other. For example,
  // forwarded_tensors[i] = {t, {stage of producer - s0, stage of the last consumer - s1}}
  // means that a tensor t, is produced in stage s0 and consumed for the last
  // time in stage s1, where s1 > s0 + 1. This tensor will be copied from stage
  // to stage until s1.
  // TODO: Should we consider weights here too?
  std::vector<std::pair<const NodeArg*, std::pair<int, int>>> forwarded_tensors;

  // All the tensors that are produced and consumed in the original graph.
  auto initial_node_args = GetAllNodeArgs(graph);

  // A tensor produced in stage r and consumed in stage r', such that r' > r,
  // will have a replica in all stages r'', such that r'' > r and r'' < r'.
  // tensor_replicas[t][r] contains a pointer to the the replica of t in stage r,
  // if it exists, or to itself if r is the stage of the producer of r.
  std::map<const NodeArg*, std::vector<NodeArg*>> tensor_replicas;

  // Find tensors that need to be sent and forwarded.
  for (const NodeArg* node_arg : initial_node_args) {
    // Initialize tensor_replicas data structure.
    auto inserted = tensor_replicas.emplace(
        std::make_pair(node_arg, std::vector<NodeArg*>(num_stages)));
    auto& replicas = (*inserted.first).second;

    const Node* producer_node = graph.GetProducerNode(node_arg->Name());
    assert(producer_node != nullptr);
    int producer_stage = op_to_stage.find(producer_node)->second;

    const auto consumers = graph.GetConsumerNodes(node_arg->Name());
    if (consumers.size() == 0) {
      continue;
    }

    // This is only handling forwarding in the forward part of the graph.
    int last_consumer_stage_forward = -1;
    for (const Node* consumer : consumers) {
      const auto found_stage = op_to_stage.find(consumer);
      assert(found_stage != op_to_stage.end());
      const int consumer_stage = found_stage->second;
      // TODO: test case in which a tensor is produced by a forward op, stashed
      // and sent to the previous stage by a backward op.
      // For now, I'm assuming that if a tensor is produced by a forward op and
      // consumed by a backward op, then producer and consumer are both in the
      // same device. This will not always be the case though.
      if (!IsBackward(const_cast<Node&>(*producer_node)) &&
          IsBackward(const_cast<Node&>(*consumer))) {
        // They must be on the same device.
        ORT_ENFORCE(producer_stage == consumer_stage,
                    "Forward producer and backward consumer of a tensor must be in the same device.");
      }

      // It is impossible to have a backward operator producing a tensor
      // consumed by a forward operator. So, at this point, either both producer
      // and consumer are forward or both are backward. Either way, we want
      // to know where are the last consumers of a tensor.
      if (IsForwardComputation(producer_stage, consumer_stage)) {
        last_consumer_stage_forward = std::max(last_consumer_stage_forward, consumer_stage);
      }
      ORT_ENFORCE(!IsBackwardComputation(producer_stage, consumer_stage),
                  "Forwarding backward tensors not supported yet: ", producer_stage, "->", consumer_stage);
      // TODO(jufranc): we will need something like the following, where
      // else if (IsBackwardComputation(producer_stage, consumer_stage)) {
      //  last_consumer_stage_backward is init to INT_MAX, for training graphs.
      //  last_consumer_stage_backward = std::min(last_consumer_stage_backward, consumer_stage);
      // }

      // Find which tensors need to be sent to the next stage (if it is a forward
      // message).
      if (producer_stage + 1 == consumer_stage) {
        forward_messages.at(producer_stage).insert(node_arg);
      }
      // TODO(jufranc): find which tensors need to be sent to the previous stage
      // (if it is a backward message). Something like:
      // else if (producer_stage - 1 == consumer_stage) {
      //   backward_messages.at(producer_stage).insert(node_arg);
      // }
    }

    // Create all the replicas for this tensor now. We also keep track of which
    // tensors need to be forwarded, and their producer-consumer stage range.
    // The replica of the tensor in the producer stage, is the tensor itself.
    replicas.at(producer_stage) = const_cast<NodeArg*>(node_arg);
    if (IsForwardComputation(producer_stage, last_consumer_stage_forward)) {
      for (int r = producer_stage + 1; r <= last_consumer_stage_forward; ++r) {
        CreateTensorReplica(graph, tensor_replicas, node_arg, r);
      }
      if (last_consumer_stage_forward - producer_stage > 1) {
        forwarded_tensors.push_back({node_arg,
                                     {producer_stage, last_consumer_stage_forward}});
      }
    }
    // TODO(jufranc): take care of IsBackwardComputation case.
    ORT_ENFORCE(last_consumer_stage_forward == -1 ||
                    !IsBackwardComputation(producer_stage, last_consumer_stage_forward),
                "Backward tensors (", node_arg->Name(), ") cannot be replicated yet");
  }

  std::vector<std::string> new_input_names;
  std::vector<std::string> new_output_names;

  for (auto& message : messages) {
    auto current_stage = message.first;
    auto next_stage = message.second;

    // for each pair of stages, record the inserted input/output args.
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
        AddMetaTensors(current_stage, next_stage, graph, new_input_names,
                       new_output_names, send_input_args, send_output_args,
                       recv_input_args, recv_output_args, stage_to_rank));

    // Get all the node_args that need to be sent to the next stage.
    auto& tensors_sent_in_forward = forward_messages.at(current_stage);
    // TODO(jufranc): consider tensors sent by backward ops.
    // auto& tensors_sent_in_backward = backward_messages.at(current_stage);
    // auto& tensors_sent = current_stage + 1 == next_stage ? tensors_sent_in_forward : tensors_sent_in_backward;

    // Take care of tensors that need to be sent from one device to the other.
    SendProducedTensors(element_types, tensors_sent_in_forward, send_input_args,
                        recv_output_args, tensor_replicas, next_stage);

    // Take care of tensors that need to be forwarded.
    SendForwardedTensors(element_types, send_input_args, recv_output_args,
                         tensor_replicas, forwarded_tensors, current_stage,
                         next_stage);

    // Update the inputs of the next_stage consumers with the right replicas.
    UpdateInputsOfConsumers(graph, tensor_replicas, op_to_stage, next_stage);

    const int num_attributes = 2;  // two attributes: tag and element_types
    NodeAttributes attributes;
    attributes.reserve(num_attributes);
    attributes[tag.name()] = tag;
    attributes[element_types.name()] = element_types;

    // Add pair of Send/Receive nodes.
    auto& send_node = graph.AddNode(graph.GenerateNodeName("Send"),
                                    "Send", "", send_input_args,
                                    send_output_args, /* output */
                                    &attributes,      /* attribute */
                                    kMSDomain);

    auto& receive_node = graph.AddNode(graph.GenerateNodeName("Recv"),
                                       "Recv", "", recv_input_args,
                                       recv_output_args, /* output */
                                       &attributes,      /* attribute */
                                       kMSDomain);

    ORT_ENFORCE(current_stage != next_stage,
                "Stage cannot send message to itself.");
    if (current_stage < next_stage) {
      send_nodes.at(current_stage) = &send_node;
      receive_nodes.at(next_stage - 1) = &receive_node;
    }
    // TODO(jufranc): consider backward sends and receives.
    // else if (current_stage > next_stage) {
    //   send_nodes[current_stage].second = &send_node;
    //   receive_nodes[next_stage].second = &receive_node;
    // }
  }

  ORT_RETURN_IF_ERROR(SetInputsOutputsAndResolve(graph, {} /* weights_to_train*/,
                                                 new_input_names,
                                                 new_output_names));

  return Status::OK();
}

// Generate subgraph / Projection.
// First remove Send nodes that do not belong to the `pipeline_stage_id`
// partition. They don't have outgoing edges. Then remove computation nodes
// that do not belong to the `pipeline_stage_id` partition, in their
// topological order. Finally, remove Receive nodes that do not belong to the
// `pipeline_stage_id` partition. At this point, they don't have outgoing
// edges either.
void GenerateSubgraph(Graph& graph, const int num_stages,
                      const std::map<const Node*, int>& op_to_stage,
                      int pipeline_stage_id,
                      std::vector<Node*>& send_nodes,
                      std::vector<Node*>& recv_nodes,
                      const std::vector<NodeIndex>& node_topology_list,
                      std::set<const NodeArg*>& visited_outputs) {
  for (int s = 0; s < num_stages - 1; ++s) {
    if (s == pipeline_stage_id) {
      continue;  // These sends must be kept.
    }
    Node* forward_send = send_nodes.at(s);
    ORT_ENFORCE(forward_send);
    graph.RemoveNode(forward_send->Index());
    // TODO(jufranc): once we enable partition of training graphs, we need to
    // remove the backward sends too.
  }

  // Collect all outputs of this partition too.
  for (auto it = node_topology_list.rbegin(); it != node_topology_list.rend(); ++it) {
    NodeIndex ni = *it;
    const auto found = op_to_stage.find(graph.GetNode(ni));
    ORT_ENFORCE(found != op_to_stage.end(),
                "Found an operator without stage.");

    if (found->second != pipeline_stage_id) {
      graph.RemoveNode(ni);
    } else {
      auto* node = graph.GetNode(ni);
      const auto& consumers = node->OutputDefs();
      for (const auto consumer : consumers) {
        if (graph.IsOutput(consumer)) {
          visited_outputs.insert(consumer);
        }
      }
    }
  }

  for (int s = 0; s < num_stages - 1; ++s) {
    if (s == pipeline_stage_id - 1) {
      // These receives must be kept.
      continue;
    }
    Node* forward_recv = recv_nodes.at(s);
    ORT_ENFORCE(forward_recv);
    graph.RemoveNode(forward_recv->Index());
    // TODO(jufranc): once we enable partition of training graphs, we need to
    // remove the backward sends too.
  }
}

Status ApplyPipelinePartitionToMainGraph(Graph& graph,
                                         const std::map<const Node*, int>& op_to_stage,
                                         const int pipeline_stage_id,
                                         const int num_stages,
                                         const std::vector<int32_t>& rank_ids) {
  // TODO(jufranc): in order to support more general pipeline shapes, we need to
  // do some analysis on the graph and assignment of operators to stages, to
  // find which messages will be sent. For now, we assume that 1) there are
  // always tensors being copied from stage s to s+1. Moreover, once we support
  // partition of training graphs, we need to let tensors be copied from s+1 to
  // s, as well.
  std::vector<int> stage_to_rank(num_stages);
  ORT_ENFORCE(static_cast<int>(rank_ids.size()) == num_stages);
  std::vector<std::pair<int, int>> messages;
  for (int s = 0; s < num_stages - 1; ++s) {
    messages.emplace_back(s, s + 1);
    stage_to_rank.at(s) = rank_ids.at(s);
  }
  stage_to_rank.at(num_stages - 1) = rank_ids.at(num_stages - 1);

  // Get the nodes in topological order before spliting the graph.
  // This ordering will be useful later to remove nodes from the partition.
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // send_nodes[s] is the Send node that copies tensors from stage s to stage s+1.
  // The last stage will not send anything.
  std::vector<Node*> send_nodes(num_stages - 1);
  // recv_nodes[s] is the Recv node that receives the replicas of tensors from
  // stage s, i.e., it is allocated to stage s+1.
  // The first stage does not receive anything.
  std::vector<Node*> recv_nodes(num_stages - 1);

  // TODO(jufranc): once we allow partition of training graphs, we need to keep
  // send and receive nodes for the backward computation. We can then use the
  // following type.
  // std::vector<std::pair<Node*, Node*>> send_nodes(num_stages);
  // std::vector<std::pair<Node*, Node*>> recv_nodes(num_stages);
  // The first Node* of the pair in send_nodes[s] is the Send from s to s+1,
  // i.e., the send for the forward stage. The second is the Send from s to
  // s-1, i.e., the send for the backward stage.

  // Split the graph into disconnected sub-graphs given the mapping of
  // operations to stages.
  ORT_RETURN_IF_ERROR(SplitGraphWithOperatorToStageMap(graph, op_to_stage,
                                                       num_stages, messages,
                                                       send_nodes, recv_nodes,
                                                       stage_to_rank));

  // Take care of weights that are shared accross stages.
  ORT_RETURN_IF_ERROR(HandleSharedInitializer(graph, send_nodes, recv_nodes));

  std::set<const NodeArg*> visited_outputs;
  GenerateSubgraph(graph, num_stages, op_to_stage, pipeline_stage_id,
                   send_nodes, recv_nodes, node_topology_list, visited_outputs);

  graph.SetOutputs({visited_outputs.begin(), visited_outputs.end()});
  graph.SetGraphResolveNeeded();
  graph.SetGraphProtoSyncNeeded();
  graph.Resolve();

  // TODO(jufranc): once we allow partition of training graphs, we need to add
  // some code to make sure that the backward receive starts after the forward
  // send, or otherwise the computation will get stuck.

  return Status::OK();
}

// Verifies the correctness of a given assignment of operators to stages.
// Input:
//   - stages[i] is the stage id assigned to the operator with Index() == i.
//   - num_stages is the total number of stages.
//   - graph is the graph being partitioned into multiple pipeline stages.
Status VerifyAssignment(const std::vector<int>& stages,
                        const int num_stages,
                        const Graph& graph) {
  // All stages are used.
  for (int s = 0; s < num_stages; ++s) {
    const auto stage_is_used = std::find(std::begin(stages), std::end(stages), s);
    ORT_RETURN_IF_NOT(stage_is_used != std::end(stages),
                      "Stage " + std::to_string(s) + " was not assigned to any node.");
  }

  // All nodes have been assigned to a stage.
  const auto op_assigned = std::find(std::begin(stages), std::end(stages), -1);
  ORT_RETURN_IF_NOT(op_assigned == std::end(stages),
                    "All ops must be assigned to a stage");

  // All assigned stages are within limits.
  for (const auto s : stages) {
    ORT_RETURN_IF_NOT(s >= 0 && s < num_stages,
                      "All stage ids must be in range.");
  }

  // Edges always go forward.
  for (size_t i = 0, t = graph.NumberOfNodes(); i < t; ++i) {
    const Node* node = graph.GetNode(i);
    const int node_stage = stages.at(i);
    const auto& node_outputs = node->OutputDefs();
    for (const NodeArg* arg : node_outputs) {
      if (arg == nullptr || !arg->HasTensorOrScalarShape() || !arg->Exists())
        continue;
      auto cs = graph.GetConsumerNodes(arg->Name());
      for (const Node* c : cs) {
        const int outgoing_stage = stages.at(c->Index());
        ORT_RETURN_IF_NOT(node_stage <= outgoing_stage, "node_stage > outgoing_stage");
      }
    }
  }

  return Status::OK();
}

Status GetDeviceAssignmentMap(const Graph& graph,
                              const std::map<std::string, int>& id_to_stage,
                              std::map<const Node*, int>& op_to_stage,
                              const int num_stages) {
  const int num_nodes = graph.NumberOfNodes();
  std::vector<int> stages(num_nodes);
  for (int i = 0; i < num_nodes; ++i) {
    const Node* node = graph.GetNode(i);
    bool found = false;
    const auto& node_outputs = node->OutputDefs();
    for (const NodeArg* arg : node_outputs) {
      if (id_to_stage.find(arg->Name()) != id_to_stage.end()) {
        stages.at(i) = id_to_stage.at(arg->Name());
        found = true;
        break;
      }
    }
    const std::string node_info = node->OpType() + "(" + node->Name() + ")";
    ORT_RETURN_IF_NOT(found, "Can't find node's stage: ", node_info);
  }

  ORT_RETURN_IF_ERROR(VerifyAssignment(stages, num_stages, graph));

  for (int i = 0; i < num_nodes; ++i) {
    op_to_stage.emplace(graph.GetNode(i), stages.at(i));
  }
  return Status::OK();
}

Status GetDeviceAssignmentMap(const Graph& graph,
                              const std::vector<TrainingSession::TrainingConfiguration::CutInfo>& cuts,
                              std::map<const Node*, int>& op_to_stage,
                              const int num_stages) {
  // The first step of this algorithm is to find all the producers and all the
  // the consumers of all the cut points, and keep them in the all_consumers and
  // all_producers containers, respectively. The producers of a cut point are the
  // operators that produce all the tensors defined in the cut point. The consumers
  // are those operators defined in the cut point.
  // Then, in order to find the ops assigned to stage 0, we visit the graph from
  // the producers of cut 0, not visiting the consumers of all cuts. All nodes
  // visited belong to stage 0. In order to find the mapping of stage s, for s>0,
  // we visit the graph from all the consumers of cut s-1, not visiting the
  // producers of that cut, and the consumers of the next cuts. All nodes visited
  // belong to stage s. Finally, we verify the assignment is valid.

  ORT_RETURN_IF(num_stages != static_cast<int>(cuts.size() + 1),
                "Number of cuts does not match number of pipeline stages.");

  const auto num_nodes = graph.NumberOfNodes();

  // Visits the graph ignoring direction of edges, i.e., it adds producers of
  // inputs and consumers of outputs to the queue of operators to be visited.
  // While it visits the graph, it assigns operators to stages. It takes a
  // stop_visit vector of booleans indicating which operators should not be
  // visited.
  auto visit_and_assign = [&](std::vector<const Node*>& roots, int stage,
                              std::vector<bool>& stop_visit,
                              std::vector<int>& stages) {
    std::vector<bool> visited(num_nodes, false);
    std::list<const Node*> q;
    // Start the visit from all the roots, which are the producers and consumers
    // of the NodeArgs in contents. If some of those nodes are not to be visited
    // because they belong to another partition, then we expect `stop_visit`
    // value to be true.
    q.insert(std::end(q), roots.begin(), roots.end());

    while (!q.empty()) {
      const Node* current = q.front();
      q.pop_front();
      if (visited.at(current->Index()) || stop_visit.at(current->Index())) {
        continue;  // This node has been processed.
      }

      // If the operator hasn't been visited, but has a stage already assigned,
      // then something went wront.
      // TODO: We should consider checking the cut is valid --- Cut Infos should
      // describe complete cuts between edges.
      ORT_RETURN_IF(stages.at(current->Index()) != -1,
                    "Trying to reassign an operator. Possibly, due to an invalid cut point");

      visited.at(current->Index()) = true;
      stages.at(current->Index()) = stage;

      // Add all ingoing edges to the queue.
      const auto& node_inputs = current->InputDefs();
      for (const NodeArg* arg : node_inputs) {
        if (arg == nullptr || !arg->HasTensorOrScalarShape() || !arg->Exists())
          continue;
        const auto producer = graph.GetProducerNode(arg->Name());
        if (producer != nullptr) {
          q.insert(std::end(q), producer);
        }
      }

      // Add all outgoing edges to the queue.
      const auto& node_outputs = current->OutputDefs();
      for (const NodeArg* arg : node_outputs) {
        if (arg == nullptr || !arg->HasTensorOrScalarShape() || !arg->Exists())
          continue;
        const auto consumers = graph.GetConsumerNodes(arg->Name());
        q.insert(std::end(q), consumers.begin(), consumers.end());
      }
    }

    return Status::OK();
  };

  const int num_cuts = static_cast<int>(cuts.size());
  // all_consumers[i] is the vector of consumers of cut i.
  std::vector<std::vector<const Node*>> all_consumers(num_cuts, std::vector<const Node*>());
  // all_producers[i] is the vector of producers of cut i.
  std::vector<std::vector<const Node*>> all_producers(num_cuts, std::vector<const Node*>());

  for (int cut_id = 0; cut_id < num_cuts; ++cut_id) {
    auto& cut = cuts.at(cut_id);
    // Find all consumers of this cut.
    auto& consumers = all_consumers.at(cut_id);
    auto& producers = all_producers.at(cut_id);
    for (auto& edge : cut) {
      const auto producer = graph.GetProducerNode(edge.node_arg_name);
      ORT_RETURN_IF(producer == nullptr, "Invalid cut point.");
      producers.emplace_back(producer);

      if (edge.consumer_nodes.has_value()) {
        auto& consumer_names = edge.consumer_nodes.value();
        for (auto& consumer_node_id : consumer_names) {
          consumers.emplace_back(graph.GetProducerNode(consumer_node_id));
        }
      } else {
        auto cs = graph.GetConsumerNodes(edge.node_arg_name);
        consumers.insert(std::end(consumers), cs.begin(), cs.end());
      }

      ORT_RETURN_IF(producers.size() == 0, "Invalid cut point.");
      ORT_RETURN_IF(consumers.size() == 0, "Invalid cut point.");
    }
  }

  std::vector<int> stages(num_nodes, -1);
  {  // Stage 0
    std::vector<bool> stop_visit(num_nodes, false);
    for (int cid = 0; cid < num_cuts; ++cid) {
      auto& consumers = all_consumers.at(cid);
      for (auto consumer : consumers) {
        stop_visit.at(consumer->Index()) = true;
      }
    }
    ORT_RETURN_IF_ERROR(
        visit_and_assign(all_producers.at(0), 0, stop_visit, stages));
  }

  // Stages 1 .. N-1
  for (int cid = 0; cid < num_cuts; ++cid) {
    std::vector<bool> stop_visit(num_nodes, false);

    auto& producers = all_producers.at(cid);
    for (auto producer : producers) {
      stop_visit.at(producer->Index()) = true;
    }

    for (int i = cid + 1; i < num_cuts; ++i) {
      auto& consumers = all_consumers.at(i);
      for (auto consumer : consumers) {
        stop_visit.at(consumer->Index()) = true;
      }
    }

    ORT_RETURN_IF_ERROR(
        visit_and_assign(all_consumers.at(cid), cid + 1, stop_visit, stages));
  }

  ORT_RETURN_IF_ERROR(VerifyAssignment(stages, num_cuts + 1, graph));

  for (size_t i = 0, t = graph.NumberOfNodes(); i < t; ++i) {
    op_to_stage.emplace(graph.GetNode(i), stages.at(i));
  }

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
