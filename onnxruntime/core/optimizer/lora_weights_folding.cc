// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/ortdevice.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/lora_weights_folding.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/framework/op_kernel.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

static Node& CreateCastNode(Graph& graph, NodeArg* cast_input, ONNX_NAMESPACE::TensorProto_DataType target_type) {
  ONNX_NAMESPACE::TypeProto cast_output_type_proto;
  cast_output_type_proto.mutable_tensor_type()->set_elem_type(target_type);

  auto& output_node_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("Cast"), &cast_output_type_proto);

  const std::array input_defs{cast_input};
  const std::array output_defs{&output_node_arg};

  Node& cast_node = graph.AddNode(
      graph.GenerateNodeName("Cast"),
      "Cast",
      "",
      input_defs,
      output_defs);
  cast_node.AddAttribute("to", static_cast<int64_t>(target_type));
  cast_node.SetExecutionProviderType(kCpuExecutionProvider);
  graph.SetOpSchemaFromRegistryForNode(cast_node);

  return cast_node;
}

static Node& CreateMatMulNode(Graph& graph, NodeArg* matmul_input_a, NodeArg* matmul_input_b) {
  auto& output_node_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("MatMul"), matmul_input_a->TypeAsProto());
  const std::array input_defs{matmul_input_a, matmul_input_b};
  const std::array output_defs{&output_node_arg};

  Node& matmul_node = graph.AddNode(
      graph.GenerateNodeName("MatMul"),
      "MatMul",
      "",
      input_defs,
      output_defs);
  matmul_node.SetExecutionProviderType(kCpuExecutionProvider);
  graph.SetOpSchemaFromRegistryForNode(matmul_node);

  return matmul_node;
}

static Node& CreateConcatNode(Graph& graph, gsl::span<NodeArg* const> input_defs, int64_t axis) {
  auto& output_node_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("Concat"), input_defs[0]->TypeAsProto());
  const std::array output_defs{&output_node_arg};

  Node& concat_node = graph.AddNode(
      graph.GenerateNodeName("Concat"),
      "Concat",
      "",
      input_defs,
      output_defs);
  concat_node.AddAttribute("axis", axis);
  concat_node.SetExecutionProviderType(kCpuExecutionProvider);
  concat_node.MutableInputArgsCount() = {static_cast<int>(input_defs.size())};
  graph.SetOpSchemaFromRegistryForNode(concat_node);

  return concat_node;
}

static Node& CreateAddNode(Graph& graph, NodeArg* original_qkv_input, NodeArg* lora_qkv_input) {
  auto& output_node_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("Add"), original_qkv_input->TypeAsProto());
  const std::array input_defs{original_qkv_input, lora_qkv_input};
  const std::array output_defs{&output_node_arg};

  Node& add_node = graph.AddNode(
      graph.GenerateNodeName("Add"),
      "Add",
      "",
      input_defs,
      output_defs);
  add_node.SetExecutionProviderType(kCpuExecutionProvider);
  graph.SetOpSchemaFromRegistryForNode(add_node);

  return add_node;
}

static Node& CreateReshapeNode(Graph& graph, NodeArg* reshape_input, NodeArg* reshape_shape) {
  auto& output_node_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("Reshape"), reshape_input->TypeAsProto());
  const std::array input_defs{reshape_input, reshape_shape};
  const std::array output_defs{&output_node_arg};

  Node& reshape_node = graph.AddNode(
      graph.GenerateNodeName("Reshape"),
      "Reshape",
      "",
      input_defs,
      output_defs);
  reshape_node.SetExecutionProviderType(kCpuExecutionProvider);
  graph.SetOpSchemaFromRegistryForNode(reshape_node);

  return reshape_node;
}

static Node& CreateSliceNode(Graph& graph, NodeArg* data, int64_t start, int64_t end) {
  auto& output_node_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("Slice"), data->TypeAsProto());

  ONNX_NAMESPACE::TensorProto starts_tensor_proto;
  starts_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  starts_tensor_proto.set_name(graph.GenerateNodeArgName("Reshape"));
  starts_tensor_proto.add_int64_data(start);
  starts_tensor_proto.add_dims(1);
  auto starts_arg = &graph_utils::AddInitializer(graph, starts_tensor_proto);

  ONNX_NAMESPACE::TensorProto ends_tensor_proto;
  ends_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  ends_tensor_proto.set_name(graph.GenerateNodeArgName("Reshape"));
  ends_tensor_proto.add_int64_data(end);
  ends_tensor_proto.add_dims(1);
  auto ends_arg = &graph_utils::AddInitializer(graph, ends_tensor_proto);

  Node& slice_node = graph.AddNode(
      graph.GenerateNodeName("Slice"),
      "Slice",
      "",
      {data, starts_arg, ends_arg},
      {&output_node_arg});
  slice_node.SetExecutionProviderType(kCpuExecutionProvider);
  graph.SetOpSchemaFromRegistryForNode(slice_node);

  return slice_node;
}

static Status RunSubgraph(
    Graph& graph,
    const IExecutionProvider& cpu_execution_provider,
    const std::unordered_map<std::string, const OrtValue*>& initializers_to_share_map,
    gsl::span<const std::string> subgraph_initializers,
    std::vector<const Node*> subgraph_nodes,
    const std::string& original_matmul_weight_name,
    const logging::Logger& logger) {
  InitializedTensorSet constant_inputs;

  std::vector<ONNX_NAMESPACE::TensorProto> initializer_overrides;
  initializer_overrides.reserve(original_matmul_weight_name.size());

  for (const std::string& initializer_name : subgraph_initializers) {
    auto iter = initializers_to_share_map.find(initializer_name);
    if (iter != initializers_to_share_map.end()) {
      auto tensor_proto = utils::TensorToTensorProto(iter->second->Get<Tensor>(), initializer_name);
      initializer_overrides.push_back(std::move(tensor_proto));
      constant_inputs[initializer_name] = &initializer_overrides.back();
    } else {
      constant_inputs[initializer_name] = graph.GetConstantInitializer(initializer_name, true);
    }
  }

  OptimizerExecutionFrame::Info info(subgraph_nodes, constant_inputs, graph.ModelPath(), cpu_execution_provider,
                                     [](std::string const&) { return false; });

  auto final_cast_node = subgraph_nodes.back();

  std::vector<int> fetch_mlvalue_idxs = {
      info.GetMLValueIndex(final_cast_node->OutputDefs()[0]->Name()),
  };

  OptimizerExecutionFrame frame(info, fetch_mlvalue_idxs);

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 6387)
#endif
  for (auto node : subgraph_nodes) {
    auto kernel = info.CreateKernel(node);
    OpKernelContext kernel_context(&frame, kernel.get(), nullptr, nullptr, logger);
    ORT_RETURN_IF_ERROR(kernel->Compute(&kernel_context));
  }
#ifdef _WIN32
#pragma warning(pop)
#endif

  std::vector<OrtValue> fetches;
  ORT_RETURN_IF_ERROR(frame.GetOutputs(fetches));

  auto new_tensor_proto = utils::TensorToTensorProto(fetches[0].Get<Tensor>(), original_matmul_weight_name);
  ORT_RETURN_IF_ERROR(graph.ReplaceInitializedTensor(std::move(new_tensor_proto)));

  // Remove all nodes that we just added
  for (const auto node : subgraph_nodes) {
    Node* mutable_node = graph.GetNode(node->Index());
    graph_utils::RemoveNodeOutputEdges(graph, *mutable_node);
    graph.RemoveNode(mutable_node->Index());
  }

  return Status::OK();
}

static Status MergeQKVWeights(
    Graph& graph,
    const IExecutionProvider& cpu_execution_provider,
    const std::unordered_map<std::string, const OrtValue*>& initializers_to_share_map,
    Node& original_qkv_node,
    Node& q_lora_down_node,
    Node& q_lora_up_node,
    Node& q_lora_reshape_node,
    Node& k_lora_down_node,
    Node& k_lora_up_node,
    Node& k_lora_reshape_node,
    Node& v_lora_down_node,
    Node& v_lora_up_node,
    Node& v_lora_reshape_node,
    Node& qkv_lora_reshape_node,
    const logging::Logger& logger) {
  Node& original_qkv_cast_node = CreateCastNode(graph, original_qkv_node.MutableInputDefs()[1], ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  Node& q_lora_down_cast_node = CreateCastNode(graph, q_lora_down_node.MutableInputDefs()[1], ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  Node& q_lora_up_cast_node = CreateCastNode(graph, q_lora_up_node.MutableInputDefs()[1], ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  Node& q_lora_matmul_node = CreateMatMulNode(graph, q_lora_down_cast_node.MutableOutputDefs()[0], q_lora_up_cast_node.MutableOutputDefs()[0]);
  Node& q_slice_node = CreateSliceNode(graph, q_lora_reshape_node.MutableInputDefs()[1], 1, INT_MAX);
  Node& q_reshape_node = CreateReshapeNode(graph, q_lora_matmul_node.MutableOutputDefs()[0], q_slice_node.MutableOutputDefs()[0]);

  Node& k_lora_down_cast_node = CreateCastNode(graph, k_lora_down_node.MutableInputDefs()[1], ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  Node& k_lora_up_cast_node = CreateCastNode(graph, k_lora_up_node.MutableInputDefs()[1], ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  Node& k_lora_matmul_node = CreateMatMulNode(graph, k_lora_down_cast_node.MutableOutputDefs()[0], k_lora_up_cast_node.MutableOutputDefs()[0]);
  Node& k_slice_node = CreateSliceNode(graph, k_lora_reshape_node.MutableInputDefs()[1], 1, INT_MAX);
  Node& k_reshape_node = CreateReshapeNode(graph, k_lora_matmul_node.MutableOutputDefs()[0], k_slice_node.MutableOutputDefs()[0]);

  Node& v_lora_down_cast_node = CreateCastNode(graph, v_lora_down_node.MutableInputDefs()[1], ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  Node& v_lora_up_cast_node = CreateCastNode(graph, v_lora_up_node.MutableInputDefs()[1], ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  Node& v_lora_matmul_node = CreateMatMulNode(graph, v_lora_down_cast_node.MutableOutputDefs()[0], v_lora_up_cast_node.MutableOutputDefs()[0]);
  Node& v_slice_node = CreateSliceNode(graph, v_lora_reshape_node.MutableInputDefs()[1], 1, INT_MAX);
  Node& v_reshape_node = CreateReshapeNode(graph, v_lora_matmul_node.MutableOutputDefs()[0], v_slice_node.MutableOutputDefs()[0]);

  std::array concat_inputs = {
      q_reshape_node.MutableOutputDefs()[0],
      k_reshape_node.MutableOutputDefs()[0],
      v_reshape_node.MutableOutputDefs()[0],
  };

  Node& concat_node = CreateConcatNode(graph, concat_inputs, 2);
  Node& slice_node = CreateSliceNode(graph, qkv_lora_reshape_node.MutableInputDefs()[1], 1, INT_MAX);
  Node& reshape_node = CreateReshapeNode(graph, concat_node.MutableOutputDefs()[0], slice_node.MutableOutputDefs()[0]);
  Node& add_node = CreateAddNode(graph, original_qkv_cast_node.MutableOutputDefs()[0], reshape_node.MutableOutputDefs()[0]);
  Node& final_cast_node = CreateCastNode(graph, add_node.MutableOutputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);

  std::vector<const Node*> subgraph_nodes = {
      &original_qkv_cast_node,
      &q_lora_down_cast_node,
      &q_lora_up_cast_node,
      &q_lora_matmul_node,
      &q_slice_node,
      &q_reshape_node,
      &k_lora_down_cast_node,
      &k_lora_up_cast_node,
      &k_lora_matmul_node,
      &k_slice_node,
      &k_reshape_node,
      &v_lora_down_cast_node,
      &v_lora_up_cast_node,
      &v_lora_matmul_node,
      &v_slice_node,
      &v_reshape_node,
      &concat_node,
      &slice_node,
      &reshape_node,
      &add_node,
      &final_cast_node,
  };

  std::array subgraph_initializer_names = {
      original_qkv_node.InputDefs()[1]->Name(),
      q_lora_down_node.InputDefs()[1]->Name(),
      q_lora_up_node.InputDefs()[1]->Name(),
      q_lora_reshape_node.InputDefs()[1]->Name(),
      q_slice_node.InputDefs()[1]->Name(),
      q_slice_node.InputDefs()[2]->Name(),
      k_lora_down_node.InputDefs()[1]->Name(),
      k_lora_up_node.InputDefs()[1]->Name(),
      k_lora_reshape_node.InputDefs()[1]->Name(),
      k_slice_node.InputDefs()[1]->Name(),
      k_slice_node.InputDefs()[2]->Name(),
      v_lora_down_node.InputDefs()[1]->Name(),
      v_lora_up_node.InputDefs()[1]->Name(),
      v_lora_reshape_node.InputDefs()[1]->Name(),
      v_slice_node.InputDefs()[1]->Name(),
      v_slice_node.InputDefs()[2]->Name(),
      qkv_lora_reshape_node.InputDefs()[1]->Name(),
      slice_node.InputDefs()[1]->Name(),
      slice_node.InputDefs()[2]->Name(),
  };

  return RunSubgraph(
      graph,
      cpu_execution_provider,
      initializers_to_share_map,
      subgraph_initializer_names,
      subgraph_nodes,
      original_qkv_node.InputDefs()[1]->Name(),
      logger);
}

static Status MergeQWeights(
    Graph& graph,
    const IExecutionProvider& cpu_execution_provider,
    const std::unordered_map<std::string, const OrtValue*>& initializers_to_share_map,
    Node& left_matmul_node,
    Node& q_lora_down_node,
    Node& q_lora_up_node,
    const logging::Logger& logger) {
  Node& left_matmul_cast_node = CreateCastNode(graph, left_matmul_node.MutableInputDefs()[1], ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  Node& q_lora_down_cast_node = CreateCastNode(graph, q_lora_down_node.MutableInputDefs()[1], ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  Node& q_lora_up_cast_node = CreateCastNode(graph, q_lora_up_node.MutableInputDefs()[1], ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  Node& q_lora_matmul_node = CreateMatMulNode(graph, q_lora_down_cast_node.MutableOutputDefs()[0], q_lora_up_cast_node.MutableOutputDefs()[0]);
  Node& add_node = CreateAddNode(graph, left_matmul_cast_node.MutableOutputDefs()[0], q_lora_matmul_node.MutableOutputDefs()[0]);
  Node& final_cast_node = CreateCastNode(graph, add_node.MutableOutputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);

  std::vector<const Node*> new_nodes = {
      &left_matmul_cast_node,
      &q_lora_down_cast_node,
      &q_lora_up_cast_node,
      &q_lora_matmul_node,
      &add_node,
      &final_cast_node,
  };

  std::array subgraph_initializer_names = {
      left_matmul_node.InputDefs()[1]->Name(),
      q_lora_down_node.InputDefs()[1]->Name(),
      q_lora_up_node.InputDefs()[1]->Name(),
  };

  return RunSubgraph(
      graph,
      cpu_execution_provider,
      initializers_to_share_map,
      subgraph_initializer_names,
      new_nodes,
      left_matmul_node.InputDefs()[1]->Name(),
      logger);
}

static Status MergeKVWeights(
    Graph& graph,
    const IExecutionProvider& cpu_execution_provider,
    const std::unordered_map<std::string, const OrtValue*>& initializers_to_share_map,
    Node& original_kv_node,
    Node& k_lora_down_node,
    Node& k_lora_up_node,
    Node& k_lora_reshape_node,
    Node& v_lora_down_node,
    Node& v_lora_up_node,
    Node& v_lora_reshape_node,
    Node& kv_lora_reshape_node,
    const logging::Logger& logger) {
  Node& original_kv_cast_node = CreateCastNode(graph, original_kv_node.MutableInputDefs()[1], ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  Node& k_lora_down_cast_node = CreateCastNode(graph, k_lora_down_node.MutableInputDefs()[1], ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  Node& k_lora_up_cast_node = CreateCastNode(graph, k_lora_up_node.MutableInputDefs()[1], ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  Node& k_lora_matmul_node = CreateMatMulNode(graph, k_lora_down_cast_node.MutableOutputDefs()[0], k_lora_up_cast_node.MutableOutputDefs()[0]);
  Node& k_slice_node = CreateSliceNode(graph, k_lora_reshape_node.MutableInputDefs()[1], 1, INT_MAX);
  Node& k_reshape_node = CreateReshapeNode(graph, k_lora_matmul_node.MutableOutputDefs()[0], k_slice_node.MutableOutputDefs()[0]);

  Node& v_lora_down_cast_node = CreateCastNode(graph, v_lora_down_node.MutableInputDefs()[1], ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  Node& v_lora_up_cast_node = CreateCastNode(graph, v_lora_up_node.MutableInputDefs()[1], ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  Node& v_lora_matmul_node = CreateMatMulNode(graph, v_lora_down_cast_node.MutableOutputDefs()[0], v_lora_up_cast_node.MutableOutputDefs()[0]);
  Node& v_slice_node = CreateSliceNode(graph, v_lora_reshape_node.MutableInputDefs()[1], 1, INT_MAX);
  Node& v_reshape_node = CreateReshapeNode(graph, v_lora_matmul_node.MutableOutputDefs()[0], v_slice_node.MutableOutputDefs()[0]);

  std::array concat_inputs = {
      k_reshape_node.MutableOutputDefs()[0],
      v_reshape_node.MutableOutputDefs()[0],
  };

  Node& concat_node = CreateConcatNode(graph, concat_inputs, 2);
  Node& slice_node = CreateSliceNode(graph, kv_lora_reshape_node.MutableInputDefs()[1], 1, INT_MAX);
  Node& reshape_node = CreateReshapeNode(graph, concat_node.MutableOutputDefs()[0], slice_node.MutableOutputDefs()[0]);
  Node& add_node = CreateAddNode(graph, original_kv_cast_node.MutableOutputDefs()[0], reshape_node.MutableOutputDefs()[0]);
  Node& final_cast_node = CreateCastNode(graph, add_node.MutableOutputDefs()[0], ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);

  std::vector<const Node*> subgraph_nodes = {
      &original_kv_cast_node,
      &k_lora_down_cast_node,
      &k_lora_up_cast_node,
      &k_lora_matmul_node,
      &k_slice_node,
      &k_reshape_node,
      &v_lora_down_cast_node,
      &v_lora_up_cast_node,
      &v_lora_matmul_node,
      &v_slice_node,
      &v_reshape_node,
      &concat_node,
      &slice_node,
      &reshape_node,
      &add_node,
      &final_cast_node,
  };

  std::array subgraph_initializer_names = {
      original_kv_node.InputDefs()[1]->Name(),
      k_lora_down_node.InputDefs()[1]->Name(),
      k_lora_up_node.InputDefs()[1]->Name(),
      k_lora_reshape_node.InputDefs()[1]->Name(),
      k_slice_node.InputDefs()[1]->Name(),
      k_slice_node.InputDefs()[2]->Name(),
      v_lora_down_node.InputDefs()[1]->Name(),
      v_lora_up_node.InputDefs()[1]->Name(),
      v_lora_reshape_node.InputDefs()[1]->Name(),
      v_slice_node.InputDefs()[1]->Name(),
      v_slice_node.InputDefs()[2]->Name(),
      kv_lora_reshape_node.InputDefs()[1]->Name(),
      slice_node.InputDefs()[1]->Name(),
      slice_node.InputDefs()[2]->Name(),
  };

  return RunSubgraph(
      graph,
      cpu_execution_provider,
      initializers_to_share_map,
      subgraph_initializer_names,
      subgraph_nodes,
      original_kv_node.InputDefs()[1]->Name(),
      logger);
}

struct QKVPath {
  Node* q_lora_down_node;
  Node* q_lora_up_node;
  Node* q_scale_node;
  Node* q_alpha_node;
  Node* q_reshape_node;
  Node* k_lora_down_node;
  Node* k_lora_up_node;
  Node* k_scale_node;
  Node* k_alpha_node;
  Node* k_reshape_node;
  Node* v_lora_down_node;
  Node* v_lora_up_node;
  Node* v_scale_node;
  Node* v_alpha_node;
  Node* v_reshape_node;
  Node* concat_node;
  Node* reshape_node;
  Node* qkv_matmul_node;
};

struct QPath {
  Node* q_lora_down_node;
  Node* q_lora_up_node;
  Node* q_scale_node;
  Node* q_alpha_node;
  Node* left_add_node;
  Node* left_matmul_node;
};

struct KVPath {
  Node* k_lora_down_node;
  Node* k_lora_up_node;
  Node* k_scale_node;
  Node* k_alpha_node;
  Node* k_reshape_node;
  Node* v_lora_down_node;
  Node* v_lora_up_node;
  Node* v_scale_node;
  Node* v_alpha_node;
  Node* v_reshape_node;
  Node* concat_node;
  Node* reshape_node;
  Node* kv_matmul_node;
};

static bool HaveConstantCpuWeights(
    Graph& graph,
    const std::unordered_map<std::string, const OrtValue*>& initializers_to_share_map,
    gsl::span<const Node* const> nodes) {
  for (const auto& node : nodes) {
    auto input_defs = node->InputDefs();

    if (input_defs.size() != 2) {
      return false;
    }

    const ONNX_NAMESPACE::TensorProto* initializer = graph.GetConstantInitializer(input_defs[1]->Name(), true);
    if (initializer == nullptr) {
      return false;
    }

    auto iter = initializers_to_share_map.find(input_defs[1]->Name());

    if (iter != initializers_to_share_map.end() && iter->second->Get<Tensor>().Location().device.Type() != OrtDevice::CPU) {
      return false;
    }
  }

  return true;
}

static bool MatchQKVPath(
    Graph& graph,
    const std::unordered_map<std::string, const OrtValue*>& initializers_to_share_map,
    Node& add_node,
    QKVPath& path,
    const logging::Logger& logger) {
  // Look for the self attention path that looks like the following:
  //
  //                  <AnyNode>
  //          ____________|_____________
  //         |      |         |         |
  //         |    MatMul    MatMul    MatMul
  //         |      |         |         |
  //         |    MatMul    MatMul    MatMul
  //         |      |         |         |
  //         |     Mul       Mul       Mul
  //         |      |         |         |
  //         |     Mul       Mul       Mul
  //         |      |         |         |
  //         |   Reshape   Reshape   Reshape
  //         |      |_________|_________|
  //         |                |
  //         |              Concat
  //         |                |
  //       MatMul          Reshape
  //         \_____   _______/
  //               \ /
  //               Add
  //
  // Essentially, we are able to fold all MatMul nodes to the right of the Add node into the left node
  // by adding their initializers together, as long as there's a node (called AnyNode here) that feeds
  // into their A input, and as long as their B inputs are initializers hardcoded into the model.

  std::vector<graph_utils::EdgeEndToMatch> add_left_path{
      {0, 1, "MatMul", {1, 9, 13}, kOnnxDomain},
  };

  std::vector<std::reference_wrapper<Node>> add_left_path_nodes;
  if (!graph_utils::FindPath(graph, add_node, true, add_left_path, add_left_path_nodes, logger)) {
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> add_right_path{
      {0, 0, "Reshape", {1, 5, 13, 14, 19}, kOnnxDomain},
      {0, 0, "Concat", {1, 4, 11, 13}, kOnnxDomain},
  };

  std::vector<std::reference_wrapper<Node>> add_right_path_nodes;
  if (!graph_utils::FindPath(graph, add_node, true, add_right_path, add_right_path_nodes, logger)) {
    return false;
  }

  path.reshape_node = &add_right_path_nodes[0].get();
  path.concat_node = &add_right_path_nodes[1].get();

  if (path.concat_node->InputDefs().size() != 3) {
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> concat_q_path{
      {0, 0, "Reshape", {1, 5, 13, 14, 19}, kOnnxDomain},
      {0, 0, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
      {0, 0, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
  };

  std::vector<std::reference_wrapper<Node>> concat_q_path_nodes;
  if (!graph_utils::FindPath(graph, *path.concat_node, true, concat_q_path, concat_q_path_nodes, logger)) {
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> concat_k_path{
      {0, 1, "Reshape", {1, 5, 13, 14, 19}, kOnnxDomain},
      {0, 0, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
      {0, 0, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
  };

  std::vector<std::reference_wrapper<Node>> concat_k_path_nodes;
  if (!graph_utils::FindPath(graph, *path.concat_node, true, concat_k_path, concat_k_path_nodes, logger)) {
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> concat_v_path{
      {0, 2, "Reshape", {1, 5, 13, 14, 19}, kOnnxDomain},
      {0, 0, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
      {0, 0, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
  };

  std::vector<std::reference_wrapper<Node>> concat_v_path_nodes;
  if (!graph_utils::FindPath(graph, *path.concat_node, true, concat_v_path, concat_v_path_nodes, logger)) {
    return false;
  }

  path.qkv_matmul_node = &add_left_path_nodes[0].get();

  path.q_reshape_node = &concat_q_path_nodes[0].get();
  path.q_scale_node = &concat_q_path_nodes[1].get();
  path.q_alpha_node = &concat_q_path_nodes[2].get();
  path.q_lora_up_node = &concat_q_path_nodes[3].get();
  path.q_lora_down_node = &concat_q_path_nodes[4].get();

  path.k_reshape_node = &concat_k_path_nodes[0].get();
  path.k_scale_node = &concat_k_path_nodes[1].get();
  path.k_alpha_node = &concat_k_path_nodes[2].get();
  path.k_lora_up_node = &concat_k_path_nodes[3].get();
  path.k_lora_down_node = &concat_k_path_nodes[4].get();

  path.v_reshape_node = &concat_v_path_nodes[0].get();
  path.v_scale_node = &concat_v_path_nodes[1].get();
  path.v_alpha_node = &concat_v_path_nodes[2].get();
  path.v_lora_up_node = &concat_v_path_nodes[3].get();
  path.v_lora_down_node = &concat_v_path_nodes[4].get();

  std::array all_nodes = {
      path.qkv_matmul_node,
      path.q_scale_node,
      path.q_alpha_node,
      path.q_lora_up_node,
      path.q_lora_down_node,
      path.k_scale_node,
      path.k_alpha_node,
      path.k_lora_up_node,
      path.k_lora_down_node,
      path.v_scale_node,
      path.v_alpha_node,
      path.v_lora_up_node,
      path.v_lora_down_node,
  };

  // Make sure that the B input of each MatMul and Mul node is a constant initializer (i.e. an initializer
  // that cannot be overridden by an input at runtime).
  if (!HaveConstantCpuWeights(graph, initializers_to_share_map, all_nodes)) {
    return false;
  }

  // Make sure that the root MatMul nodes all share the same input
  const bool share_same_root = path.qkv_matmul_node->MutableInputDefs()[0]->Name() == path.q_lora_down_node->MutableInputDefs()[0]->Name() &&
                               path.q_lora_down_node->MutableInputDefs()[0]->Name() == path.k_lora_down_node->MutableInputDefs()[0]->Name() &&
                               path.k_lora_down_node->MutableInputDefs()[0]->Name() == path.v_lora_down_node->MutableInputDefs()[0]->Name();

  if (!share_same_root) {
    return false;
  }

  return true;
}

static bool MatchQPath(
    Graph& graph,
    const std::unordered_map<std::string, const OrtValue*>& initializers_to_share_map,
    Node& add_node,
    QPath& path,
    const logging::Logger& logger) {
  // Look for the attention's output path that looks like one of the following graphs:
  //
  //          <AnyNode>                          <AnyNode>
  //          ____|____                          ____|____
  //         |         |                        |         |
  //         |       MatMul                     |       MatMul
  //         |         |                        |         |
  //         |       MatMul                     |       MatMul
  //         |         |                        |         |
  //         |        Mul                       |        Mul
  //         |         |                        |         |
  //       MatMul     Mul                       |        Mul
  //         |         |                        |         |
  //        Add      Reshape                  MatMul      Reshape
  //         |___   ___|                        |___   ___|
  //             \ /                                \ /
  //             Add                                Add
  //
  // Essentially, we are able to fold all MatMul nodes to the right of the Add node into the left MatMul
  // node by adding their initializers together, as long as there's a node (called AnyNode here) that feeds
  // into their A input, and as long as their B inputs are initializers hardcoded into the model.

  std::vector<graph_utils::EdgeEndToMatch> add_left_path{
      {0, 0, "Add", {1, 6, 7, 13, 14}, kOnnxDomain},
      {0, 1, "MatMul", {1, 9, 13}, kOnnxDomain},
  };

  std::vector<std::reference_wrapper<Node>> add_left_path_nodes;
  if (graph_utils::FindPath(graph, add_node, true, add_left_path, add_left_path_nodes, logger)) {
    path.left_add_node = &add_left_path_nodes[0].get();
    path.left_matmul_node = &add_left_path_nodes[1].get();
  } else {
    // The left Add node is optional, so try the path without it
    add_left_path = {
        {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
    };

    if (graph_utils::FindPath(graph, add_node, true, add_left_path, add_left_path_nodes, logger)) {
      path.left_add_node = nullptr;
      path.left_matmul_node = &add_left_path_nodes[0].get();
    } else {
      return false;
    }
  }

  std::vector<graph_utils::EdgeEndToMatch> add_right_path{
      {0, 1, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
      {0, 0, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
  };

  std::vector<std::reference_wrapper<Node>> add_right_path_nodes;
  if (!graph_utils::FindPath(graph, add_node, true, add_right_path, add_right_path_nodes, logger)) {
    return false;
  }

  path.q_scale_node = &add_right_path_nodes[0].get();
  path.q_alpha_node = &add_right_path_nodes[1].get();
  path.q_lora_up_node = &add_right_path_nodes[2].get();
  path.q_lora_down_node = &add_right_path_nodes[3].get();

  std::array all_nodes = {
      path.left_matmul_node,
      path.q_scale_node,
      path.q_alpha_node,
      path.q_lora_up_node,
      path.q_lora_down_node,
  };

  // Make sure that the B input of each MatMul and Mul node is a constant initializer (i.e. an initializer
  // that cannot be overridden by an input at runtime).
  if (!HaveConstantCpuWeights(graph, initializers_to_share_map, all_nodes)) {
    return false;
  }

  // Make sure that the root MatMul nodes all share the same input
  const bool share_same_root = path.left_matmul_node->MutableInputDefs()[0]->Name() == path.q_lora_down_node->MutableInputDefs()[0]->Name();

  if (!share_same_root) {
    return false;
  }

  return true;
}

static bool MatchKVPath(
    Graph& graph,
    const std::unordered_map<std::string, const OrtValue*>& initializers_to_share_map,
    Node& add_node, KVPath& path,
    const logging::Logger& logger) {
  // Look for the self attention path that looks like the following:
  //
  //                  <AnyNode>
  //          ___________|_________
  //         |           |         |
  //         |         MatMul    MatMul
  //         |           |         |
  //         |         MatMul    MatMul
  //         |           |         |
  //         |          Mul       Mul
  //         |           |         |
  //         |          Mul       Mul
  //         |           |         |
  //         |        Reshape   Reshape
  //         |           |_________|
  //         |                |
  //         |              Concat
  //         |                |
  //       MatMul          Reshape
  //         \_____   _______/
  //               \ /
  //               Add
  //
  // Essentially, we are able to fold all MatMul nodes to the right of the Add node into the left node
  // by adding their initializers together, as long as there's a node (called AnyNode here) that feeds
  // into their A input, and as long as their B inputs are initializers hardcoded into the model.

  std::vector<graph_utils::EdgeEndToMatch> add_left_path{
      {0, 1, "MatMul", {1, 9, 13}, kOnnxDomain},
  };

  std::vector<std::reference_wrapper<Node>> add_left_path_nodes;
  if (!graph_utils::FindPath(graph, add_node, true, add_left_path, add_left_path_nodes, logger)) {
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> add_right_path{
      {0, 0, "Reshape", {1, 5, 13, 14, 19}, kOnnxDomain},
      {0, 0, "Concat", {1, 4, 11, 13}, kOnnxDomain},
  };

  std::vector<std::reference_wrapper<Node>> add_right_path_nodes;
  if (!graph_utils::FindPath(graph, add_node, true, add_right_path, add_right_path_nodes, logger)) {
    return false;
  }

  path.reshape_node = &add_right_path_nodes[0].get();
  path.concat_node = &add_right_path_nodes[1].get();

  if (path.concat_node->InputDefs().size() != 2) {
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> concat_k_path{
      {0, 0, "Reshape", {1, 5, 13, 14, 19}, kOnnxDomain},
      {0, 0, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
      {0, 0, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
  };

  std::vector<std::reference_wrapper<Node>> concat_k_path_nodes;
  if (!graph_utils::FindPath(graph, *path.concat_node, true, concat_k_path, concat_k_path_nodes, logger)) {
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> concat_v_path{
      {0, 1, "Reshape", {1, 5, 13, 14, 19}, kOnnxDomain},
      {0, 0, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
      {0, 0, "Mul", {1, 6, 7, 13, 14}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
  };

  std::vector<std::reference_wrapper<Node>> concat_v_path_nodes;
  if (!graph_utils::FindPath(graph, *path.concat_node, true, concat_v_path, concat_v_path_nodes, logger)) {
    return false;
  }

  path.kv_matmul_node = &add_left_path_nodes[0].get();

  path.k_reshape_node = &concat_k_path_nodes[0].get();
  path.k_scale_node = &concat_k_path_nodes[1].get();
  path.k_alpha_node = &concat_k_path_nodes[2].get();
  path.k_lora_up_node = &concat_k_path_nodes[3].get();
  path.k_lora_down_node = &concat_k_path_nodes[4].get();

  path.v_reshape_node = &concat_v_path_nodes[0].get();
  path.v_scale_node = &concat_v_path_nodes[1].get();
  path.v_alpha_node = &concat_v_path_nodes[2].get();
  path.v_lora_up_node = &concat_v_path_nodes[3].get();
  path.v_lora_down_node = &concat_v_path_nodes[4].get();

  std::array all_nodes = {
      path.kv_matmul_node,
      path.k_scale_node,
      path.k_alpha_node,
      path.k_lora_up_node,
      path.k_lora_down_node,
      path.v_scale_node,
      path.v_alpha_node,
      path.v_lora_up_node,
      path.v_lora_down_node,
  };

  // Make sure that the B input of each MatMul and Mul node is a constant initializer (i.e. an initializer
  // that cannot be overridden by an input at runtime).
  if (!HaveConstantCpuWeights(graph, initializers_to_share_map, all_nodes)) {
    return false;
  }

  // Make sure that the root MatMul nodes all share the same input
  const bool share_same_root = path.kv_matmul_node->MutableInputDefs()[0]->Name() == path.k_lora_down_node->MutableInputDefs()[0]->Name() &&
                               path.k_lora_down_node->MutableInputDefs()[0]->Name() == path.v_lora_down_node->MutableInputDefs()[0]->Name();

  if (!share_same_root) {
    return false;
  }

  return true;
}

Status LoraWeightsFolding::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  InlinedVector<std::reference_wrapper<Node>> nodes_to_remove;
  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (!node_ptr || node_ptr->OpType() != "Add") {
      continue;  // node was removed
    }

    auto& add_node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(add_node, modified, graph_level, logger));

    QKVPath qkv_path;
    KVPath kv_path;
    QPath q_path;
    if (MatchQKVPath(graph, initializers_to_share_map_, add_node, qkv_path, logger)) {
      // Run a small graph that will perform the following operations on the CPU:
      // 1. MatMul operation between the up/down initializers of the LoRA Q/K/V weights
      // 2. Joining the results of the matmuls into a single tensor
      // 3. Adding the joined tensors to the original model's QKV weights
      ORT_RETURN_IF_ERROR(MergeQKVWeights(
          graph,
          cpu_execution_provider_,
          initializers_to_share_map_,
          *qkv_path.qkv_matmul_node,
          *qkv_path.q_lora_down_node,
          *qkv_path.q_lora_up_node,
          *qkv_path.q_reshape_node,
          *qkv_path.k_lora_down_node,
          *qkv_path.k_lora_up_node,
          *qkv_path.k_reshape_node,
          *qkv_path.v_lora_down_node,
          *qkv_path.v_lora_up_node,
          *qkv_path.v_reshape_node,
          *qkv_path.reshape_node,
          logger));

      nodes_to_remove.push_back(*qkv_path.q_lora_down_node);
      nodes_to_remove.push_back(*qkv_path.q_lora_up_node);
      nodes_to_remove.push_back(*qkv_path.q_scale_node);
      nodes_to_remove.push_back(*qkv_path.q_alpha_node);
      nodes_to_remove.push_back(*qkv_path.q_reshape_node);
      nodes_to_remove.push_back(*qkv_path.k_lora_down_node);
      nodes_to_remove.push_back(*qkv_path.k_lora_up_node);
      nodes_to_remove.push_back(*qkv_path.k_scale_node);
      nodes_to_remove.push_back(*qkv_path.k_alpha_node);
      nodes_to_remove.push_back(*qkv_path.k_reshape_node);
      nodes_to_remove.push_back(*qkv_path.v_lora_down_node);
      nodes_to_remove.push_back(*qkv_path.v_lora_up_node);
      nodes_to_remove.push_back(*qkv_path.v_scale_node);
      nodes_to_remove.push_back(*qkv_path.v_alpha_node);
      nodes_to_remove.push_back(*qkv_path.v_reshape_node);
      nodes_to_remove.push_back(*qkv_path.concat_node);
      nodes_to_remove.push_back(*qkv_path.reshape_node);
      nodes_to_remove.push_back(add_node);

      // Now that we removed the Add node, hook the left MatMul node directly to the output of the Add node
      graph_utils::ReplaceDownstreamNodeInput(graph, add_node, 0, *qkv_path.qkv_matmul_node, 0);
      modified = true;
    } else if (MatchQPath(graph, initializers_to_share_map_, add_node, q_path, logger)) {
      // Run a small graph that will perform the following operations on the CPU:
      // 1. MatMul operation between the up/down initializers of the LoRA Q weights
      // 2. Adding the result to the original model's Q weights
      ORT_RETURN_IF_ERROR(MergeQWeights(
          graph,
          cpu_execution_provider_,
          initializers_to_share_map_,
          *q_path.left_matmul_node,
          *q_path.q_lora_down_node,
          *q_path.q_lora_up_node,
          logger));

      nodes_to_remove.push_back(*q_path.q_lora_down_node);
      nodes_to_remove.push_back(*q_path.q_lora_up_node);
      nodes_to_remove.push_back(*q_path.q_scale_node);
      nodes_to_remove.push_back(*q_path.q_alpha_node);
      nodes_to_remove.push_back(add_node);

      if (q_path.left_add_node) {
        // Now that we removed the Add node, hook the left Add node directly to the output of the Add node
        graph_utils::ReplaceDownstreamNodeInput(graph, add_node, 0, *q_path.left_add_node, 0);
      } else {
        // Now that we removed the Add node, hook the left MatMul node directly to the output of the Add node
        graph_utils::ReplaceDownstreamNodeInput(graph, add_node, 0, *q_path.left_matmul_node, 0);
      }

      modified = true;
    } else if (MatchKVPath(graph, initializers_to_share_map_, add_node, kv_path, logger)) {
      // Run a small graph that will perform the following operations on the CPU:
      // 1. MatMul operation between the up/down initializers of the LoRA K/V weights
      // 2. Joining the results of the matmuls into a single tensor
      // 3. Adding the joined tensors to the original model's KV weights
      ORT_RETURN_IF_ERROR(MergeKVWeights(
          graph,
          cpu_execution_provider_,
          initializers_to_share_map_,
          *kv_path.kv_matmul_node,
          *kv_path.k_lora_down_node,
          *kv_path.k_lora_up_node,
          *kv_path.k_reshape_node,
          *kv_path.v_lora_down_node,
          *kv_path.v_lora_up_node,
          *kv_path.v_reshape_node,
          *kv_path.reshape_node,
          logger));

      nodes_to_remove.push_back(*kv_path.k_lora_down_node);
      nodes_to_remove.push_back(*kv_path.k_lora_up_node);
      nodes_to_remove.push_back(*kv_path.k_scale_node);
      nodes_to_remove.push_back(*kv_path.k_alpha_node);
      nodes_to_remove.push_back(*kv_path.k_reshape_node);
      nodes_to_remove.push_back(*kv_path.v_lora_down_node);
      nodes_to_remove.push_back(*kv_path.v_lora_up_node);
      nodes_to_remove.push_back(*kv_path.v_scale_node);
      nodes_to_remove.push_back(*kv_path.v_alpha_node);
      nodes_to_remove.push_back(*kv_path.v_reshape_node);
      nodes_to_remove.push_back(*kv_path.concat_node);
      nodes_to_remove.push_back(*kv_path.reshape_node);
      nodes_to_remove.push_back(add_node);

      // Now that we removed the Add node, hook the left MatMul node directly to the output of the Add node
      graph_utils::ReplaceDownstreamNodeInput(graph, add_node, 0, *kv_path.kv_matmul_node, 0);
      modified = true;
    }
  }

  // Once we're done, we can safely remove all LoRA nodes that were injected
  for (const auto& node : nodes_to_remove) {
    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.get().Index());
  }

  return Status::OK();
}

}  // namespace onnxruntime
