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

Status TransformGraphForPipeline(Graph& graph) {
  // insert WaitEvent and RecordEvent to the partition
  Node* send_fw{nullptr};
  Node* send_bw{nullptr};
  Node* recv_fw{nullptr};
  Node* recv_bw{nullptr};
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Send") {
      if (node.Description() == "Backward pass") {
        send_bw = &node;
      } else {
        send_fw = &node;
      }
    } else if (node.OpType() == "Recv") {
      if (node.Description() == "Backward pass") {
        recv_bw = &node;
      } else {
        recv_fw = &node;
      }
    }
  }

  ORT_RETURN_IF_NOT(
          (send_fw && recv_bw) || (!send_fw && !recv_bw),
          "Graph requires either having both send forward node "
          "and recv backword node, or none of them. Currently the graph "
          "has send forward: ", send_fw, " and recv backward: ", recv_bw);

  std::vector<std::string> new_input_names;
  auto add_input_event = [&](const std::string& op_name,
                             bool is_forward,
                             std::vector<NodeArg*>& input_args) {
    ONNX_NAMESPACE::TypeProto event_type_proto;
    event_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);

    auto event_id_name = graph.GenerateNodeArgName(op_name + (is_forward ? "_fw" : "_bw") + "_event_id");
    auto& event_id = graph.GetOrCreateNodeArg(event_id_name, &event_type_proto);
    new_input_names.push_back(event_id_name);

    input_args.push_back(&event_id);
  };

  if (send_bw) {
    // if we have send op in backward pass (at the end of the graph), add the RecordEvent op after that.
    std::vector<NodeArg*> input_args;
    add_input_event("RecordEvent", false /* is_forward */, input_args);
    input_args.insert(std::end(input_args),
                      std::begin(send_bw->MutableOutputDefs()),
                      std::end(send_bw->MutableOutputDefs()));
    graph.AddNode(graph.GenerateNodeName("RecordEvent"),
                  "RecordEvent",
                  "Backward pass",
                  input_args,
                  {},
                  nullptr,
                  kMSDomain);
  }

  auto create_node_args = [&](NodeArg* base_arg) -> NodeArg& {
    const auto& new_name = graph.GenerateNodeArgName(base_arg->Name());
    ONNX_NAMESPACE::TypeProto type_proto(*(base_arg->TypeAsProto()));
    return graph.GetOrCreateNodeArg(new_name, &type_proto);
  };

  if (recv_fw) {
    // if we have recv op in forward pass (at the begining of the graph), add the WaitEvent op before that.
    std::vector<NodeArg*> input_args;
    std::vector<NodeArg*> output_args;
    add_input_event("WaitEvent", true /* is_forward */, input_args);

    // recv's first input is the signal input. Re-direct it to WaitEvent's input.
    auto& input_signal = recv_fw->MutableInputDefs()[0];
    auto& wait_output = create_node_args(input_signal);
    output_args.push_back(&wait_output);
    input_args.push_back(input_signal);
    input_signal = &wait_output;

    graph.AddNode(graph.GenerateNodeName("WaitEvent"),
                  "WaitEvent",
                  "",
                  input_args,
                  output_args,
                  nullptr,
                  kMSDomain);
  }
  if (send_fw && recv_bw) {
    // if we have a send forward op followed by a recv backward op, insert WaitEvent and RecordEvent in between.
    Node* record_node = nullptr;

    // Insert RecordEvent
    {
      std::vector<NodeArg*> input_args;
      std::vector<NodeArg*> output_args;
      add_input_event("RecordEvent", true /* is_forward */, input_args);

      // Add send forward op's output as record op's input and output
      for (auto& output : send_fw->MutableOutputDefs()) {
        auto& new_output = create_node_args(output);
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
      add_input_event("WaitEvent", false /* is_forward */, input_args);

      input_args.insert(std::end(input_args),
                        std::begin(record_node->MutableOutputDefs()),
                        std::end(record_node->MutableOutputDefs()));

      auto& input = recv_bw->MutableInputDefs()[0];
      auto& new_output = create_node_args(input);
      output_args.push_back(&new_output);
      input = &new_output;

      graph.AddNode(graph.GenerateNodeName("WaitEvent"),
                    "WaitEvent",
                    "Backward pass",
                    input_args,
                    output_args, /* output */
                    {},          /* attribute */
                    kMSDomain);
    }
  }

  const std::vector<const NodeArg*>& graph_inputs = graph.GetInputsIncludingInitializers();
  std::vector<const NodeArg*> inputs_args_sets(graph_inputs.begin(), graph_inputs.end());

  for (auto& name : new_input_names) {
    inputs_args_sets.push_back(graph.GetNodeArg(name));
  }

  graph.SetInputs(inputs_args_sets);
  graph.SetGraphResolveNeeded();
  graph.SetGraphProtoSyncNeeded();
  return graph.Resolve();
}

}  // namespace training
}  // namespace onnxruntime
