// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/graph/pipeline_transformer.h"
#include <queue>

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
                         std::string& event_id_tensor_name) {
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
  auto& new_output = CreateNodeArg(graph, input_args[1]);  // the first input is signal, not passing through
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
  return Status::OK();
}

Status AddWaitForward(Graph& graph,
                      Node* /* recv_fw */,
                      std::vector<std::string>& new_input_names,
                      std::string& forward_waited_event_name) {
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
  graph.AddNode(graph.GenerateNodeName("WaitEvent"),
                "WaitEvent",
                "",
                input_args,
                output_args,
                nullptr,
                kMSDomain);
  forward_waited_event_name = input_args.front()->Name();
  return Status::OK();
}

Status AddOrSkipRecordForwardWaitBackward(Graph& graph,
                                          Node* send_fw,
                                          Node* recv_bw,
                                          std::vector<std::string>& new_input_names,
                                          std::string& forward_recorded_event_name,
                                          std::string& backward_waited_event_name) {
  if (!send_fw != !recv_bw) {
    ORT_THROW(
        "Graph requires either having both send forward node "
        "and recv backword node, or none of them. Currently the graph "
        "has send forward: ",
        send_fw, " and recv backward: ", recv_bw);
  }

  if (!send_fw && !recv_bw) {
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
  }

  return Status::OK();
}

Status TransformGraphForPipeline(
    Graph& graph,
    std::string& forward_waited_event_name,
    std::string& forward_recorded_event_name,
    std::string& backward_waited_event_name,
    std::string& backward_recorded_event_name) {
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
      backward_recorded_event_name));
  ORT_RETURN_IF_ERROR(AddWaitForward(
      graph,
      recv_fw,
      new_input_names,
      forward_waited_event_name));
  ORT_RETURN_IF_ERROR(AddOrSkipRecordForwardWaitBackward(
      graph,
      send_fw,
      recv_bw,
      new_input_names,
      forward_recorded_event_name,
      backward_waited_event_name));

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

// common::Status SplitGraphForPipeline(const Graph& graph,
//                                      std::vector<CutInfo> split_edge_groups,
//                                      size_t pipeline_stage_id,
//                                      const std::string& input_file_name,
//                                      std::string& pipeline_partition_file_name) {

void AddInputSignal(Graph& graph, const std::string& op_name,
                    size_t cut_index,
                    //  ::PROTOBUF_NAMESPACE_ID::int32 value
                    std::vector<NodeArg*>& input_args,
                    std::vector<std::string>& new_input_names) {
  ONNX_NAMESPACE::TypeProto signal_type_proto;
  signal_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);

  auto event_id_name = graph.GenerateNodeArgName(op_name + std::to_string(cut_index));
  auto& event_id = graph.GetOrCreateNodeArg(event_id_name, &signal_type_proto);

  ONNX_NAMESPACE::TensorProto data{};
  data.set_name(event_id_name);
  // data.add_dims(1);
  data.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);
  // data.set_bool_val(true);
  data.add_int32_data(true);
  graph.AddInitializedTensor(data);

  // ONNX_NAMESPACE::TensorProto data_0{};
  // data_0.set_name("data_0");
  // data_0.add_dims(1);
  // data_0.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  // data_0.add_float_data(0);
  // graph.AddInitializedTensor(data_0);

  new_input_names.push_back(event_id_name);
  input_args.push_back(&event_id);
}

void AddOutputSignal(Graph& graph, const std::string& op_name,
                     size_t cut_index,
                     //  ::PROTOBUF_NAMESPACE_ID::int32 value
                     std::vector<NodeArg*>& input_args,
                     std::vector<std::string>& new_input_names) {
  ONNX_NAMESPACE::TypeProto signal_type_proto;
  signal_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);

  auto event_id_name = graph.GenerateNodeArgName(op_name + std::to_string(cut_index));
  auto& event_id = graph.GetOrCreateNodeArg(event_id_name, &signal_type_proto);

  new_input_names.push_back(event_id_name);
  input_args.push_back(&event_id);
}

void AddInputRank(Graph& graph, const std::string& op_name,
                  size_t cut_index,
                  size_t value,
                  //  ::PROTOBUF_NAMESPACE_ID::int32 value
                  std::vector<NodeArg*>& input_args,
                  std::vector<std::string>& new_input_names) {
  ONNX_NAMESPACE::TypeProto signal_type_proto;
  signal_type_proto.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);

  auto event_id_name = graph.GenerateNodeArgName(op_name + std::to_string(cut_index));
  auto& event_id = graph.GetOrCreateNodeArg(event_id_name, &signal_type_proto);

  ONNX_NAMESPACE::TensorProto data{};
  data.set_name(event_id_name);
  // data.add_dims(1);
  data.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  data.add_int64_data(value);
  graph.AddInitializedTensor(data);

  new_input_names.push_back(event_id_name);
  input_args.push_back(&event_id);
}

// void add_expand_type(Graph& graph, std::string& name, type){
// for (const auto* value_info : graph.GetValueInfo()) {
//     *(graph_proto.mutable_value_info()->Add()) = value_info->ToProto();
//   }
// }
// // common::Status SplitGraphForPipeline(Graph& graph,
common::Status SplitGraphForPipeline(Graph&,
                                     std::vector<TrainingSession::TrainingConfiguration::CutInfo>,
                                     size_t,
                                     size_t,
                                     std::string&) {
  return Status::OK();
}
// // // common::Status SplitGraphForPipeline(Graph& graph,
// // // // common::Status SplitGraphForPipeline(Graph& ,
// // //                                      std::vector<TrainingSession::TrainingConfiguration::CutInfo> split_edge_groups,
// // //                                      size_t pipeline_stage_id,
// // //                                      size_t num_pipeline_stages,
// // //                                      std::string& pipeline_partition_file_name) {
// // //   if (num_pipeline_stages != split_edge_groups.size() + 1){
// // //     ORT_THROW("wrong cutting info");
// // //   }

// // //   std::cout<<pipeline_partition_file_name<<pipeline_stage_id<<num_pipeline_stages<<std::endl;
// // // // return Status::OK();
// // // // struct CutEdge {
// // // //       std::string node_arg_name;
// // // //       optional<std::string> consumer_node;

// // // //       CutEdge(std::string edge) : node_arg_name(edge){};
// // // //       CutEdge(std::string edge, std::string node) : node_arg_name(edge), consumer_node(node){};
// // // //     };
// // //   std::vector<std::string> new_input_names;
// // //   std::vector<std::string> new_output_names;

// // //   // using CInfo = TrainingSession::TrainingConfiguration::CutInfo;
// // //   for (size_t index = 0;index<split_edge_groups.size();++index){
// // //     auto& edgeIds = split_edge_groups[index];

// // //     std::vector<Node*> upstream_nodes;
// // //     std::vector<size_t> upstream_nodes_output_index;
// // //     std::vector<ONNX_NAMESPACE::DataType> element_types;

// // //     for(auto& id : edgeIds){
// // //       // find node whose output contains id.node_arg_name
// // //       auto producer_node = graph.GetMutableProducerNode(id.node_arg_name);
// // //       if(producer_node == nullptr){
// // //         ORT_THROW("wrong cutting info");
// // //       }
// // //       upstream_nodes.push_back(producer_node);

// // //       producer_node->ForEachWithIndex(
// // //           producer_node->OutputDefs(),
// // //           [&](const NodeArg& def, size_t index) {
// // //             if (def.Name() == id.node_arg_name) {
// // //               upstream_nodes_output_index.push_back(index);
// // //               element_types.push_back(def.Type());
// // //             }
// // //             return Status::OK();
// // //           });

// // //       std::vector<NodeArg*> send_input_args;
// // //       std::vector<NodeArg*> send_output_args;
// // //       std::vector<NodeArg*> recv_input_args;
// // //       std::vector<NodeArg*> recv_output_args;

// // //       AddInputSignal(graph, "send_input_signal", index, send_input_args, new_input_names);
// // //       AddInputSignal(graph, "recv_input_signal", index, recv_input_args, new_input_names);

// // //       AddInputRank(graph, "send_dst_rank", index, index +1, send_input_args, new_input_names);
// // //       AddInputRank(graph, "recv_src_rank", index, index, recv_input_args, new_input_names);

// // //       AddOutputSignal(graph, "send_output_signal", index, send_output_args, new_output_names);
// // //       AddOutputSignal(graph, "receive_output_signal", index, recv_output_args, new_output_names);

// // //       // Node* send_node = nullptr;
// // //       // Node* recv_node = nullptr;

// // //       for(size_t i=0;i<upstream_nodes.size();++i){
// // //         auto& n = upstream_nodes[i];
// // //         auto idx = upstream_nodes_output_index[i];
// // //         // // auto& output_type = element_types[i];
// // //         auto& output_edge_name = n->MutableOutputDefs()[idx]->Name();

// // //         std::vector<const Node*> consumer_nodes = graph.GetConsumerNodes(output_edge_name);

// // //         // deal with shape inference for newly added edge
// // //         std::string new_send_input_name = output_edge_name + "_send" + std::to_string(index);
// // //         // add_expand_type(model, new_send_input_name, output_type)
// // //         graph.AddNode(graph.GenerateNodeName("Send"),
// // //                                     "Send",
// // //                                     "",
// // //                                     send_input_args,
// // //                                     send_output_args, /* output */
// // //                                     {},          /* attribute */
// // //                                     kMSDomain);
// // //         // send_node = &new_node;

// // //         graph.AddNode(graph.GenerateNodeName("Recv"),
// // //                   "Recv",
// // //                   "",
// // //                   recv_input_args,
// // //                   recv_output_args, /* output */
// // //                   {},          /* attribute */
// // //                   kMSDomain);
// // //       }

// // //     }

// // //   }
// // //   // Model::Save(*model_, model_uri);
// // //   // auto mp = model->ToProto();
// // //   // std::ofstream ofs(output_file, std::ofstream::binary);
// // //   // mp.SerializeToOstream(&ofs);
// // //   // ofs.close();
// // //   return Status::OK();
// // // }
common::Status PipelineTransformer::SplitGraph(Graph& graph,
                                               std::vector<Node*>& send_nodes,
                                               std::vector<Node*>& recv_nodes) const {
  std::map<NodeArg*, NodeArg*> updated_node_args;  // = {{1,'a'},{2,'b'}};
  // std::cout << pipeline_partition_file_name_ << pipeline_stage_id_ << graph_level << num_pipeline_stages_ << std::endl;

  std::vector<std::string> new_input_names;
  std::vector<std::string> new_output_names;

  // using CInfo = TrainingSession::TrainingConfiguration::CutInfo;
  for (size_t index = 0; index < split_edge_groups_.size(); ++index) {
    auto& edgeIds = split_edge_groups_[index];

    // // std::vector<Node*> upstream_nodes;
    // // std::vector<size_t> upstream_nodes_output_index;
    // std::vector<ONNX_NAMESPACE::DataType> element_types;

    std::vector<NodeArg*> send_input_args;
    std::vector<NodeArg*> send_output_args;
    std::vector<NodeArg*> recv_input_args;
    std::vector<NodeArg*> recv_output_args;

    AddInputSignal(graph, "send_input_signal", index, send_input_args, new_input_names);
    AddInputSignal(graph, "recv_input_signal", index, recv_input_args, new_input_names);

    AddInputRank(graph, "send_dst_rank", index, index + 1, send_input_args, new_input_names);
    AddInputRank(graph, "recv_src_rank", index, index, recv_input_args, new_input_names);

    AddOutputSignal(graph, "send_output_signal", index, send_output_args, new_output_names);
    AddOutputSignal(graph, "receive_output_signal", index, recv_output_args, new_output_names);

    ONNX_NAMESPACE::AttributeProto keepdims;
    keepdims.set_name("tag");
    keepdims.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
    keepdims.set_i(static_cast<int64_t>(0));

    ONNX_NAMESPACE::AttributeProto element_types;
    element_types.set_name("element_types");
    element_types.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);

    for (auto& id : edgeIds) {
      // find node whose output contains id.node_arg_name
      auto producer_node = graph.GetMutableProducerNode(id.node_arg_name);
      if (producer_node == nullptr) {
        ORT_THROW("wrong cutting info");
      }
      // upstream_nodes.push_back(producer_node);
      size_t upstream_nodes_output_index{0};
      producer_node->ForEachWithIndex(
          producer_node->OutputDefs(),
          [&](const NodeArg& def, size_t index) {
            if (def.Name() == id.node_arg_name) {
              // upstream_nodes_output_index.push_back(index);
              // element_types.push_back(def.Type());
              upstream_nodes_output_index = index;
              return Status::OK();
            }
            // throw?
            ORT_THROW("Node with name ", id.node_arg_name, " already exists.");
          });

      // Node* send_node = nullptr;
      // Node* recv_node = nullptr;
      element_types.add_ints(static_cast<int64_t>(1));

      // for (size_t i = 0; i < upstream_nodes.size(); ++i) {
      // auto& n = upstream_nodes[i];
      auto& n = producer_node;
      auto idx = upstream_nodes_output_index;
      // // auto& output_type = element_types[i];
      auto& output_edge_name = n->MutableOutputDefs()[idx]->Name();  //output_edge_name should be id.node_arg_name. Consider remove this variable

      auto updated_node_arg = n->MutableOutputDefs()[idx];
      auto exiting_updated_node_arg = updated_node_args.find(updated_node_arg);
      if (exiting_updated_node_arg != updated_node_args.end()) {
        updated_node_arg = exiting_updated_node_arg->second;
      }

      send_input_args.push_back(updated_node_arg);
      auto& new_receive_output = CreateNodeArg(graph, n->MutableOutputDefs()[idx]);
      recv_output_args.push_back(&new_receive_output);

      updated_node_args[n->MutableOutputDefs()[idx]] = &new_receive_output;

      // deal with shape inference for newly added edge
      std::string new_send_input_name = output_edge_name + "_send" + std::to_string(index);
      // add_expand_type(model, new_send_input_name, output_type)

      // NodeProto tag;
      // attributes.set_name("tag");
      // num_scan_inputs.set_i(1);

      // for (int i = 0; i < num_attributes; ++i) {
      //   auto& attr = node_proto.attribute(i);
      //   attributes[attr.name()] = attr;
      // }

      std::vector<Node*> consumer_nodes;
      if (!id.consumer_node.has_value()) {
        consumer_nodes = graph.GetMutableConsumerNodes(output_edge_name);
      } else {
        consumer_nodes.push_back(graph.GetMutableProducerNode(id.consumer_node.value()));
        // for (auto& n : id.consumer_node.value()) {
        // consumer_nodes.push_back(graph.GetMutableProducerNode(n));
        // }
      }

      for (auto n : consumer_nodes) {
        for (auto& input : n->MutableInputDefs()) {
          if (input->Name() == output_edge_name) {
            // upstream_nodes_output_index.push_back(index);
            // element_types.push_back(def.Type());
            input = &new_receive_output;
            break;
          }
        }
      }
      // n->ForEachWithIndex(
      // n->MutableInputDefs(),
      // [&](NodeArg& def) {
      //   if (def.Name() == output_edge_name) {
      //     // upstream_nodes_output_index.push_back(index);
      //     // element_types.push_back(def.Type());
      //     def = new_receive_output;
      //     return Status::OK();
      //   }
      //   // Throw if not find?
      // });
    }
    const int num_attributes = 2;  // type and tag
    NodeAttributes attributes;
    attributes.reserve(num_attributes);
    attributes[keepdims.name()] = keepdims;
    attributes[element_types.name()] = element_types;

    auto& send_node = graph.AddNode(graph.GenerateNodeName("Send"),
                                    "Send",
                                    "",
                                    send_input_args,
                                    send_output_args, /* output */
                                    &attributes,      /* attribute */
                                    kMSDomain);
    // send_node = &new_node;
    send_nodes.push_back(&send_node);
    // inputs_args_sets.insert(inputs_args_sets.end(), send_input_args.begin(), send_input_args.end());
    // outputs_args_sets.insert(outputs_args_sets.end(), send_output_args.begin(), send_output_args.end());

    auto& recv_node = graph.AddNode(graph.GenerateNodeName("Recv"),
                                    "Recv",
                                    "",
                                    recv_input_args,
                                    recv_output_args, /* output */
                                    &attributes,      /* attribute */
                                    kMSDomain);
    recv_nodes.push_back(&recv_node);
    // inputs_args_sets.insert(inputs_args_sets.end(), send_input_args.begin(), send_input_args.end());
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

Status FindAllConnectedNodes(Graph& graph,
                             const Node* node,
                             std::vector<const Node*>& connected_nodes) {
  // std::vector<Node*> connected_nodes;


  ORT_THROW_IF_ERROR(node->ForEachWithIndex(
      node->InputDefs(),
      [&](const NodeArg& node_arg, size_t /*index*/) {
        if(!graph.IsInputsIncludingInitializers(&node_arg)){
          const Node* producer_node = graph.GetProducerNode(node_arg.Name());
          connected_nodes.push_back(producer_node);
        }

        return Status::OK();
      }));

  ORT_THROW_IF_ERROR(node->ForEachWithIndex(
      node->OutputDefs(),
      [&](const NodeArg& node_arg, size_t /*index*/) {

        if (!graph.IsOutput(&node_arg)) {
          std::vector<const Node*> consumer_nodes = graph.GetConsumerNodes(node_arg.Name());
          connected_nodes.insert(std::end(connected_nodes), consumer_nodes.begin(), consumer_nodes.end());

        }
        return Status::OK();
      }));
  return Status::OK();
}
common::Status PipelineTransformer::GenerateSubgraph(Graph& graph, const Node* start_node) const {
  std::queue<const Node*> node_queue;
  node_queue.push(start_node);

  std::set<const Node*> visited;
  std::vector<const NodeArg*> inputs;
  std::vector<const NodeArg*> outputs;
  while (!node_queue.empty()) {
    auto node = node_queue.front();
    node_queue.pop();
    if (visited.count(node) == 0) {
      visited.insert(node);
      std::vector<const Node*> connected_nodes;
      ORT_THROW_IF_ERROR(FindAllConnectedNodes(graph, node, connected_nodes));

      for (auto n : connected_nodes) {
        node_queue.push(n);
      }
    }
  }
  std::set<NodeIndex> visited_node_index;
  for (auto n : visited) {
    visited_node_index.insert(n->Index());
  }
  // reverse sort
  // std::sort(visited_node_index.begin(), visited_node_index.end(), std::greater<size_t>());

  // for (auto& node : graph.Nodes()) {
  for (auto it = graph.Nodes().cbegin(); it != graph.Nodes().cend(); it++){
    // auto node = *it;
    if (visited_node_index.count(it->Index())==0){
      graph.RemoveNode(it->Index() );
    }
  }
  // for (auto index : visited_node_index) {
  //   graph.RemoveNode(index);
  // }
  // graph.SetInputs(inputs_args_sets);
  // graph.SetOutputs(outputs_args_sets);
  graph.SetGraphResolveNeeded();
  graph.SetGraphProtoSyncNeeded();

  return graph.Resolve();
}

Status PipelineTransformer::ApplyImpl(Graph& graph,
                                      // Status PipelineTransformer::ApplyImpl(Graph&,
                                      bool& modified,
                                      int /*graph_level*/,
                                      const logging::Logger& /*logger*/) const {
  // std::vector<const NodeArg*> inputs_args_sets;
  // std::vector<const NodeArg*> outputs_args_sets;
  size_t split_count = split_edge_groups_.size();
  if (num_pipeline_stages_ != split_count + 1) {
    ORT_THROW("wrong cutting info");
  }

  std::vector<Node *> send_nodes, recv_nodes;
  send_nodes.reserve(split_count);
  recv_nodes.reserve(split_count);
  ORT_RETURN_IF_ERROR(SplitGraph(graph, send_nodes, recv_nodes));

  if (send_nodes.size() != split_count || recv_nodes.size() != split_count) {
    ORT_THROW("wrong split");
  }

  if (pipeline_stage_id_ < split_count) {
    ORT_RETURN_IF_ERROR(GenerateSubgraph(graph, send_nodes[pipeline_stage_id_]));
  } else {
    ORT_RETURN_IF_ERROR(GenerateSubgraph(graph, recv_nodes.back()));
  }
  modified = true;
  return Status::OK();
}
}  // namespace training
}  // namespace onnxruntime
