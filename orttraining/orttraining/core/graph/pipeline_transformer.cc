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

void AddRecordBackward(Graph& graph, Node* send_bw, std::vector<std::string>& new_input_names) {
  // gradient graph can contain some dangling leaf nodes. Add them all to WaitEvent
  // backward node's input.
  auto add_leaf_node = [&](std::vector<NodeArg*>& input_args) {
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

  std::vector<NodeArg*> input_args;
  AddInputEvent(graph, "RecordEvent", false /* is_forward */, input_args, new_input_names);

  if (send_bw) {
    // if we have send op in backward pass (at the end of the graph), add the RecordEvent op after that.
    input_args.insert(std::end(input_args),
                      std::begin(send_bw->MutableOutputDefs()),
                      std::end(send_bw->MutableOutputDefs()));
  }
  add_leaf_node(input_args);
  graph.AddNode(graph.GenerateNodeName("RecordEvent"),
                "RecordEvent",
                "Backward pass",
                input_args,
                {},
                nullptr,
                kMSDomain);
}

Status AddWaitForward(Graph& graph, Node* recv_fw, std::vector<std::string>& new_input_names) {
  auto update_wait_input_output = [&](NodeArg* old_input,
                                      std::vector<NodeArg*>& input_args,
                                      std::vector<NodeArg*>& output_args) -> NodeArg& {
    const auto& new_name = graph.GenerateNodeArgName(old_input->Name());
    ONNX_NAMESPACE::TypeProto type_proto(*(old_input->TypeAsProto()));

    auto& wait_output = graph.GetOrCreateNodeArg(new_name, &type_proto);
    output_args.push_back(&wait_output);
    input_args.push_back(old_input);
    return wait_output;
  };

  if (recv_fw) {
    // if we have recv op in forward pass (at the begining of the graph), add the WaitEvent op before that.
    std::vector<NodeArg*> input_args;
    std::vector<NodeArg*> output_args;
    AddInputEvent(graph, "WaitEvent", true /* is_forward */, input_args, new_input_names);

    // recv's first input is the signal input. Re-direct it to WaitEvent's input.
    auto& input_signal = recv_fw->MutableInputDefs()[0];
    auto& wait_output = update_wait_input_output(input_signal, input_args, output_args);
    input_signal = &wait_output;

    graph.AddNode(graph.GenerateNodeName("WaitEvent"),
                  "WaitEvent",
                  "",
                  input_args,
                  output_args,
                  nullptr,
                  kMSDomain);
  } else {
    // the first stage doesn't have recv_fw. Add Wait for all inputs.
    std::vector<NodeArg*> input_args;
    std::vector<NodeArg*> output_args;
    AddInputEvent(graph, "WaitEvent", true /* is_forward */, input_args, new_input_names);
    const std::vector<const NodeArg*>& graph_inputs = graph.GetInputs();

    ORT_RETURN_IF_NOT(
      graph_inputs.size() > 0, "Graph ", graph.Name(), " doesn't have any inputs.");

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
  }
  return Status::OK();
}

Status AddRecordForwardWaitBackward(Graph& graph, Node* send_fw, Node* recv_bw, std::vector<std::string>& new_input_names) {
  ORT_RETURN_IF_NOT(
      !send_fw == !recv_bw,
      "Graph requires either having both send forward node "
      "and recv backword node, or none of them. Currently the graph "
      "has send forward: ",
      send_fw, " and recv backward: ", recv_bw);

  auto create_node_args = [&](const NodeArg* base_arg) -> NodeArg& {
    const auto& new_name = graph.GenerateNodeArgName(base_arg->Name());
    ONNX_NAMESPACE::TypeProto type_proto(*(base_arg->TypeAsProto()));
    return graph.GetOrCreateNodeArg(new_name, &type_proto);
  };
  if (send_fw && recv_bw) {
    // if we have a send forward op followed by a recv backward op, insert WaitEvent and RecordEvent in between.
    Node* record_node = nullptr;

    // Insert RecordEvent
    {
      std::vector<NodeArg*> input_args;
      std::vector<NodeArg*> output_args;
      AddInputEvent(graph, "RecordEvent", true /* is_forward */, input_args, new_input_names);

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
      AddInputEvent(graph, "WaitEvent", false /* is_forward */, input_args, new_input_names);

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
  return Status::OK();
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

  AddRecordBackward(graph, send_bw, new_input_names);
  ORT_RETURN_IF_ERROR(AddWaitForward(graph, recv_fw, new_input_names));
  ORT_RETURN_IF_ERROR(AddRecordForwardWaitBackward(graph, send_fw, recv_bw, new_input_names));

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
