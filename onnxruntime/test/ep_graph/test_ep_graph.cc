// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <gtest/gtest.h>
#include <gsl/gsl>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "core/common/common.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "test/ep_graph/test_ep_graph_utils.h"
#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"

// defined in unittest_main/test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

// Checks that the producer of a OrtValueInfo obtained from the public C API is valid.
static void CheckValueInfoProducer(const GraphViewer& graph_viewer, const OrtValueInfo* value_info,
                                   const NodeArg* node_arg) {
  const OrtApi& ort_api = Ort::GetApi();

  if (!node_arg->Exists()) {
    return;
  }

  const OrtNode* api_producer_node = nullptr;
  size_t api_producer_output_index = 0;
  ASSERT_ORTSTATUS_OK(ort_api.ValueInfo_GetValueProducer(value_info, &api_producer_node, &api_producer_output_index));

  const Node* producer_node = graph_viewer.GetProducerNode(node_arg->Name());
  if (producer_node == nullptr) {
    ASSERT_EQ(api_producer_node, nullptr);
  } else {
    bool within_graph_viewer = graph_viewer.GetNode(producer_node->Index()) != nullptr;
    if (!within_graph_viewer) {
      ASSERT_EQ(api_producer_node, nullptr);  // Producer is outside the graph viewer, so C API should return null
    } else {
      ASSERT_EQ(std::string(ort_api.Node_Name(api_producer_node)), producer_node->Name());
      ASSERT_EQ(std::string(ort_api.Node_OperatorType(api_producer_node)), producer_node->OpType());
      ASSERT_EQ(std::string(ort_api.Node_Domain(api_producer_node)), producer_node->Domain());

      size_t output_index = 0;
      ASSERT_STATUS_OK(GetOutputIndex(*producer_node, node_arg->Name(), output_index));
      ASSERT_EQ(api_producer_output_index, output_index);
    }
  }
}

// Checks that consumers of a OrtValueInfo obtained from the public C API are valid by comparing to the original graph.
static void CheckValueInfoConsumers(const GraphViewer& graph_viewer, const OrtValueInfo* value_info,
                                    const NodeArg* node_arg) {
  const OrtApi& ort_api = Ort::GetApi();

  if (!node_arg->Exists()) {
    return;
  }

  std::vector<NodeArgConsumer> node_arg_consumers;
  ASSERT_STATUS_OK(GetNodeArgConsumers(graph_viewer, *node_arg, node_arg_consumers));

  size_t api_num_consumers = 0;
  ASSERT_ORTSTATUS_OK(ort_api.ValueInfo_GetValueNumConsumers(value_info, &api_num_consumers));
  ASSERT_EQ(api_num_consumers, node_arg_consumers.size());

  std::vector<const OrtNode*> api_node_consumers(api_num_consumers, nullptr);
  std::vector<int64_t> api_input_indices(api_num_consumers, 0);
  ASSERT_ORTSTATUS_OK(ort_api.ValueInfo_GetValueConsumers(value_info, api_node_consumers.data(),
                                                          api_input_indices.data(), api_num_consumers));

  for (size_t i = 0; i < api_num_consumers; i++) {
    ASSERT_EQ(std::string(ort_api.Node_Name(api_node_consumers[i])), node_arg_consumers[i].node->Name());
    ASSERT_EQ(std::string(ort_api.Node_OperatorType(api_node_consumers[i])), node_arg_consumers[i].node->OpType());
    ASSERT_EQ(std::string(ort_api.Node_Domain(api_node_consumers[i])), node_arg_consumers[i].node->Domain());
    ASSERT_EQ(api_input_indices[i], static_cast<int64_t>(node_arg_consumers[i].input_index));
  }
}

// Checks that the OrtValueInfos obtained from the public C API are "equivalent" to the NodeArgs
// in the original graph.
static void CheckValueInfosCApi(const GraphViewer& graph_viewer, gsl::span<const OrtValueInfo* const> value_infos,
                                gsl::span<const NodeArg* const> node_args) {
  ASSERT_EQ(value_infos.size(), node_args.size());
  const OrtApi& ort_api = Ort::GetApi();
  const auto& graph_viewer_inputs = graph_viewer.GetInputsIncludingInitializers();
  const auto& graph_viewer_outputs = graph_viewer.GetOutputs();

  for (size_t i = 0; i < value_infos.size(); i++) {
    const NodeArg* node_arg = node_args[i];
    const OrtValueInfo* value_info = value_infos[i];

    if (node_arg->Exists()) {
      ASSERT_NE(value_info, nullptr);

      const char* api_name = nullptr;
      ASSERT_ORTSTATUS_OK(ort_api.GetValueInfoName(value_info, &api_name));
      ASSERT_EQ(std::string(api_name), node_arg->Name());

      bool is_graph_input = std::any_of(graph_viewer_inputs.begin(), graph_viewer_inputs.end(),
                                        [&node_arg](const NodeArg* graph_input) {
                                          return node_arg->Name() == graph_input->Name();
                                        });
      bool api_is_graph_input = ort_api.ValueInfo_IsGraphInput(value_info);
      ASSERT_EQ(api_is_graph_input, is_graph_input);

      bool is_graph_output = std::any_of(graph_viewer_outputs.begin(), graph_viewer_outputs.end(),
                                         [&node_arg](const NodeArg* graph_output) {
                                           return node_arg->Name() == graph_output->Name();
                                         });
      bool api_is_graph_output = ort_api.ValueInfo_IsGraphOutput(value_info);
      ASSERT_EQ(api_is_graph_output, is_graph_output);

      bool is_initializer = graph_viewer.IsInitializedTensor(node_arg->Name());
      bool api_is_initializer = ort_api.ValueInfo_IsInitializer(value_info);
      ASSERT_EQ(api_is_initializer, is_initializer);

      bool is_outer_scope = graph_viewer.GetGraph().IsOuterScopeValue(node_arg->Name());
      bool api_is_outer_scope = ort_api.ValueInfo_IsFromOuterScope(value_info);
      ASSERT_EQ(api_is_outer_scope, is_outer_scope);

      auto node_arg_type_info = OrtTypeInfo::FromTypeProto(*node_arg->TypeAsProto());
      const OrtTypeInfo* api_type_info = nullptr;
      ASSERT_ORTSTATUS_OK(ort_api.GetValueInfoTypeInfo(value_info, &api_type_info));
      ASSERT_NE(api_type_info, nullptr);

      ONNXType api_onnx_type = ONNX_TYPE_UNKNOWN;
      ASSERT_ORTSTATUS_OK(ort_api.GetOnnxTypeFromTypeInfo(api_type_info, &api_onnx_type));
      ASSERT_EQ(api_onnx_type, node_arg_type_info->type);

      if (api_onnx_type == ONNX_TYPE_TENSOR) {
        // Only validating Tensors (not checking Map, Sequence, etc.) values because these C APIs for getting
        // type/shape information existed long before the new ORT graph IR APIs and are tested elsewhere.
        const OrtTensorTypeAndShapeInfo* api_type_shape = nullptr;
        ASSERT_ORTSTATUS_OK(ort_api.CastTypeInfoToTensorInfo(api_type_info, &api_type_shape));
        ASSERT_NE(api_type_shape, nullptr);

        ONNXTensorElementDataType api_elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
        ASSERT_ORTSTATUS_OK(ort_api.GetTensorElementType(api_type_shape, &api_elem_type));
        ASSERT_EQ(api_elem_type, node_arg_type_info->tensor_type_info->type);

        size_t api_num_dims = 0;
        ASSERT_ORTSTATUS_OK(ort_api.GetDimensionsCount(api_type_shape, &api_num_dims));
        ASSERT_EQ(api_num_dims, node_arg_type_info->tensor_type_info->shape.NumDimensions());

        std::vector<int64_t> api_dims(api_num_dims, 0);
        ASSERT_ORTSTATUS_OK(ort_api.GetDimensions(api_type_shape, api_dims.data(), api_dims.size()));
        ASSERT_EQ(gsl::span<const int64_t>(api_dims), node_arg_type_info->tensor_type_info->shape.GetDims());

        std::vector<const char*> api_dim_syms(api_num_dims, nullptr);
        ASSERT_ORTSTATUS_OK(ort_api.GetSymbolicDimensions(api_type_shape, api_dim_syms.data(), api_dim_syms.size()));
        const std::vector<std::string>& dim_syms = node_arg_type_info->tensor_type_info->dim_params;
        for (size_t dim_idx = 0; dim_idx < api_num_dims; dim_idx++) {
          ASSERT_EQ(std::string(api_dim_syms[dim_idx]), dim_syms[dim_idx]);
        }
      }

      CheckValueInfoProducer(graph_viewer, value_info, node_arg);
      CheckValueInfoConsumers(graph_viewer, value_info, node_arg);
    } else {
      ASSERT_EQ(value_info, nullptr);  // A missing optional input has a null OrtValueInfo.
    }
  }
}

// Checks that the contents of the original GraphViewer matches the contents of the OrtGraph.
// Uses the public C APIs to traverse the OrtGraph.
static void CheckGraphCApi(const GraphViewer& graph_viewer, const OrtGraph& api_graph) {
  const OrtApi& ort_api = Ort::GetApi();

  // Check graph inputs.
  const auto& graph_input_node_args = graph_viewer.GetInputsIncludingInitializers();
  size_t api_num_graph_inputs = ort_api.Graph_NumInputs(&api_graph);
  ASSERT_EQ(api_num_graph_inputs, graph_input_node_args.size());

  std::vector<const OrtValueInfo*> api_graph_inputs(api_num_graph_inputs, nullptr);
  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetInputs(&api_graph, api_graph_inputs.data(), api_graph_inputs.size()));
  CheckValueInfosCApi(graph_viewer, api_graph_inputs, graph_input_node_args);

  // Check graph outputs.
  const auto& graph_output_node_args = graph_viewer.GetOutputs();
  size_t api_num_graph_outputs = ort_api.Graph_NumOutputs(&api_graph);
  ASSERT_EQ(api_num_graph_outputs, graph_output_node_args.size());

  std::vector<const OrtValueInfo*> api_graph_outputs(api_num_graph_outputs, nullptr);
  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetOutputs(&api_graph, api_graph_outputs.data(), api_graph_outputs.size()));
  CheckValueInfosCApi(graph_viewer, api_graph_outputs, graph_output_node_args);

  // Check if it has a parent node.
  const Node* parent_node = graph_viewer.ParentNode();
  const bool has_parent_node = parent_node != nullptr;
  const OrtNode* api_parent_node = nullptr;
  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetParentNode(&api_graph, &api_parent_node));
  const bool api_has_parent_node = api_parent_node != nullptr;
  ASSERT_EQ(api_has_parent_node, has_parent_node);
  if (has_parent_node) {
    ASSERT_EQ(std::string(ort_api.Node_Name(api_parent_node)), parent_node->Name());
    ASSERT_EQ(std::string(ort_api.Node_OperatorType(api_parent_node)), parent_node->OpType());
    ASSERT_EQ(std::string(ort_api.Node_Domain(api_parent_node)), parent_node->Domain());
  }

  // Check all nodes.
  size_t num_nodes = ort_api.Graph_NumNodes(&api_graph);
  ASSERT_EQ(num_nodes, graph_viewer.NumberOfNodes());

  std::vector<const OrtNode*> api_nodes(num_nodes, nullptr);
  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetNodes(&api_graph, 0, api_nodes.data(), api_nodes.size()));

  std::vector<NodeIndex> node_indices = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::DEFAULT);
  for (size_t node_idx = 0; node_idx < num_nodes; node_idx++) {
    const Node* node = graph_viewer.GetNode(node_indices[node_idx]);
    ASSERT_NE(node, nullptr);

    const OrtNode* api_node = api_nodes[node_idx];
    ASSERT_EQ(std::string(ort_api.Node_Name(api_node)), node->Name());
    ASSERT_EQ(std::string(ort_api.Node_OperatorType(api_node)), node->OpType());
    ASSERT_EQ(std::string(ort_api.Node_Domain(api_node)), node->Domain());

    int api_since_version = 0;
    ASSERT_ORTSTATUS_OK(ort_api.Node_GetSinceVersion(api_node, &api_since_version));
    ASSERT_EQ(api_since_version, node->SinceVersion());

    const auto input_node_args = node->InputDefs();
    const size_t num_inputs = ort_api.Node_NumInputs(api_node);
    ASSERT_EQ(num_inputs, input_node_args.size());

    std::vector<const OrtValueInfo*> api_inputs(num_inputs, nullptr);
    ASSERT_ORTSTATUS_OK(ort_api.Node_GetInputs(api_node, api_inputs.data(), api_inputs.size()));
    CheckValueInfosCApi(graph_viewer, api_inputs, input_node_args);

    const auto output_node_args = node->OutputDefs();
    const size_t num_outputs = ort_api.Node_NumOutputs(api_node);
    ASSERT_EQ(num_outputs, output_node_args.size());

    std::vector<const OrtValueInfo*> api_outputs(num_outputs, nullptr);
    ASSERT_ORTSTATUS_OK(ort_api.Node_GetOutputs(api_node, api_outputs.data(), api_outputs.size()));
    CheckValueInfosCApi(graph_viewer, api_outputs, output_node_args);

    std::vector<gsl::not_null<const Graph*>> node_subgraphs = node->GetSubgraphs();
    size_t api_num_subgraphs = 0;
    ASSERT_ORTSTATUS_OK(ort_api.Node_GetNumSubgraphs(api_node, &api_num_subgraphs));
    ASSERT_EQ(api_num_subgraphs, node_subgraphs.size());

    if (api_num_subgraphs > 0) {
      const auto implicit_input_node_args = node->ImplicitInputDefs();
      size_t api_num_implicit_inputs = 0;
      ASSERT_ORTSTATUS_OK(ort_api.Node_GetNumImplicitInputs(api_node, &api_num_implicit_inputs));
      ASSERT_EQ(api_num_implicit_inputs, implicit_input_node_args.size());

      std::vector<const OrtValueInfo*> api_implicit_inputs(api_num_implicit_inputs, nullptr);
      ASSERT_ORTSTATUS_OK(ort_api.Node_GetImplicitInputs(api_node, api_implicit_inputs.data(),
                                                         api_implicit_inputs.size()));
      CheckValueInfosCApi(graph_viewer, api_implicit_inputs, implicit_input_node_args);

      std::vector<const OrtGraph*> api_node_subgraphs(api_num_subgraphs, nullptr);
      ASSERT_ORTSTATUS_OK(ort_api.Node_GetSubgraphs(api_node, api_node_subgraphs.data(), api_node_subgraphs.size()));
      for (size_t subgraph_idx = 0; subgraph_idx < api_num_subgraphs; subgraph_idx++) {
        auto subgraph_viewer = std::make_unique<GraphViewer>(*node_subgraphs[subgraph_idx]);
        CheckGraphCApi(*subgraph_viewer, *api_node_subgraphs[subgraph_idx]);
      }
    }
  }
}

// Checks that an OrtGraph is initialized correctly and tests basic usage of the C API
// by traversing the OrtGraph and checking validity of nodes and value infos.
TEST(EpGraphTest, BasicCApiUse) {
  auto test_graph = TestGraph::Load(ORT_TSTR("testdata/mnist.onnx"));
  ASSERT_NE(test_graph, nullptr) << "Failed to load test model";

  CheckGraphCApi(test_graph->GetGraphViewer(), test_graph->GetOrtGraph());
}

// Traverse OrtGraph with Scan nodes, which tests handling of subgraphs, implicit inputs, and variadic I/O.
TEST(EpGraphTest, CheckModelWithSubgraphs) {
  auto test_graph = TestGraph::Load(ORT_TSTR("testdata/scan_1.onnx"));
  ASSERT_NE(test_graph, nullptr) << "Failed to load test model";

  CheckGraphCApi(test_graph->GetGraphViewer(), test_graph->GetOrtGraph());
}

//
// Test implementation of Kahn's Topological sort using public C graph APIs and C++ STL.
//

#define RETURN_IF_API_ERROR(fn) \
  do {                          \
    Ort::Status status(fn);     \
    if (!status.IsOK()) {       \
      return status;            \
    }                           \
  } while (0)

template <typename T>
struct VisitorPriorityQueue {
  using ComparatorType = std::function<bool(T, T)>;
  std::list<T> list_;
  const ComparatorType comparator_ = nullptr;
  VisitorPriorityQueue(const ComparatorType& comp) : comparator_(comp) {}

  void push(T node) {
    list_.insert(
        std::upper_bound(list_.begin(), list_.end(), node, comparator_),
        node);
  }
  bool empty() { return list_.empty(); }
  T top() { return list_.back(); }
  void pop() { list_.pop_back(); }
};

// Get the number of input edges that come from another node upstream.
static Ort::Status GetNodeInputEdgeCount(const OrtNode* node, size_t& num_input_edges) {
  const OrtApi& ort_api = Ort::GetApi();

  const size_t num_inputs = ort_api.Node_NumInputs(node);
  std::vector<const OrtValueInfo*> inputs(num_inputs, nullptr);
  RETURN_IF_API_ERROR(ort_api.Node_GetInputs(node, inputs.data(), inputs.size()));

  // Sum the number of inputs with a producer node.
  num_input_edges = 0;
  for (const OrtValueInfo* input : inputs) {
    if (input == nullptr) continue;  // Skip missing optional input

    const OrtNode* producer_node = nullptr;
    RETURN_IF_API_ERROR(ort_api.ValueInfo_GetValueProducer(input, &producer_node, /*output_index*/ nullptr));
    num_input_edges += static_cast<size_t>(producer_node != nullptr);
  }

  return Ort::Status{nullptr};
}

// Get all output nodes that consume an output from the given node.
static Ort::Status GetOutputNodes(const OrtNode* node, std::vector<const OrtNode*>& result) {
  const OrtApi& ort_api = Ort::GetApi();

  const size_t num_outputs = ort_api.Node_NumOutputs(node);
  std::vector<const OrtValueInfo*> outputs(num_outputs, nullptr);
  RETURN_IF_API_ERROR(ort_api.Node_GetOutputs(node, outputs.data(), outputs.size()));

  std::vector<const OrtNode*> output_nodes;
  output_nodes.reserve(num_outputs);  // May have more than `num_outputs`

  // Gather the OrtNode consumers of every output.
  for (const OrtValueInfo* output : outputs) {
    if (output == nullptr) continue;  // Skip missing optional output

    size_t num_consumers = 0;
    RETURN_IF_API_ERROR(ort_api.ValueInfo_GetValueNumConsumers(output, &num_consumers));

    std::vector<const OrtNode*> node_consumers(num_consumers, nullptr);
    std::vector<int64_t> input_indices(num_consumers, 0);
    RETURN_IF_API_ERROR(ort_api.ValueInfo_GetValueConsumers(output, node_consumers.data(),
                                                            input_indices.data(), num_consumers));

    for (const OrtNode* consumer : node_consumers) {
      output_nodes.push_back(consumer);
    }
  }

  result = std::move(output_nodes);
  return Ort::Status{nullptr};
}

// Kahn's topological sort.
// Adapted from onnxruntime/core/graph/graph.cc to use public C API graph types.
static Ort::Status KahnsTopologicalSort(const OrtGraph& graph,
                                        const std::function<void(const OrtNode*)>& enter,
                                        const std::function<bool(const OrtNode*, const OrtNode*)>& comp) {
  const OrtApi& ort_api = Ort::GetApi();

  // Get all nodes
  const size_t num_nodes = ort_api.Graph_NumNodes(&graph);
  std::vector<const OrtNode*> nodes(num_nodes, nullptr);
  RETURN_IF_API_ERROR(ort_api.Graph_GetNodes(&graph, 0, nodes.data(), nodes.size()));

  // Get the maximum node ID. Not really required if we chose to represent the `in_degree` as a map instead of vector.
  size_t max_node_id = 0;
  for (const OrtNode* node : nodes) {
    max_node_id = std::max(max_node_id, ort_api.Node_Id(node));
  }

  std::vector<size_t> in_degree(max_node_id + 1, 0);
  std::vector<size_t> topo_order;
  VisitorPriorityQueue<const OrtNode*> to_visit(comp);

  topo_order.reserve(num_nodes);

  // Initialize in_degree and initial nodes to visit first.
  for (const OrtNode* node : nodes) {
    size_t input_edge_count = 0;
    RETURN_IF_API_ERROR(GetNodeInputEdgeCount(node, input_edge_count));
    in_degree[ort_api.Node_Id(node)] = input_edge_count;
    if (input_edge_count == 0) {
      to_visit.push(node);
    }
  }

  while (!to_visit.empty()) {
    const OrtNode* current = to_visit.top();
    to_visit.pop();

    if (!current) continue;

    if (enter) {
      enter(current);
    }

    std::vector<const OrtNode*> output_nodes;
    GetOutputNodes(current, output_nodes);

    for (const OrtNode* output_node : output_nodes) {
      auto& node_in_degree = in_degree[ort_api.Node_Id(output_node)];
      node_in_degree--;

      if (node_in_degree == 0) {
        to_visit.push(output_node);
      }
    }
    topo_order.push_back(ort_api.Node_Id(current));
  }

  if (num_nodes != static_cast<int>(topo_order.size())) {
    return Ort::Status("Some nodes are not included in the topological sort: graph has a cycle", ORT_FAIL);
  }

  return Ort::Status{nullptr};
}

// Node comparison functor copied from onnxruntime/core/graph/graph.cc
struct PriorityNodeCompare {
  inline bool IsHighPri(const OrtNode* n) const {
    // local statics so we can compare std::strings in the checks
    static constexpr std::string_view shape_op("Shape");
    static constexpr std::string_view size_op("Size");

    const std::string_view op_type = Ort::GetApi().Node_OperatorType(n);
    return op_type == shape_op || op_type == size_op;
  }

  // Used for std::priority_queue
  // If return false, n1 will be output first
  // If return true, n2 will be output first
  bool operator()(const OrtNode* n1, const OrtNode* n2) const {
    // nodes in global high priority list will be output first
    const bool isN1HighPri = IsHighPri(n1);
    const bool isN2HighPri = IsHighPri(n2);
    if (isN1HighPri != isN2HighPri) {
      return isN2HighPri;
    }

    // nodes with lower priority value will be output first
    const auto n1_priority = 0;  // n1->Priority(); // Looks to always be 0 inside ORT?
    const auto n2_priority = 0;  // n2->Priority(); // Looks to always be 0 inside ORT?
    if (n1_priority != n2_priority) {
      return n1_priority > n2_priority;
    }

    // otherwise, nodes with lower index will be output first
    return Ort::GetApi().Node_Id(n1) > Ort::GetApi().Node_Id(n2);
  }
};

TEST(EpGraphTest, BasicKahnTopoSort) {
  auto test_graph = TestGraph::Load(ORT_TSTR("testdata/bart_tiny.onnx"));
  ASSERT_NE(test_graph, nullptr) << "Failed to load test model";

  // Sort OrtGraph with a custom Kahn's topological sorting algorithm.
  std::vector<size_t> api_nodes_topo_sort_with_priority;
  Ort::Status status(KahnsTopologicalSort(
      test_graph->GetOrtGraph(),
      [&](const OrtNode* n) {
        api_nodes_topo_sort_with_priority.push_back(Ort::GetApi().Node_Id(n));
      },
      PriorityNodeCompare()));
  ASSERT_TRUE(status.IsOK()) << status.GetErrorMessage();

  // Use ORT's built in sorting with priority.
  std::vector<size_t> ort_topo_sort_with_priority = test_graph->GetGraphViewer()
                                                        .GetNodesInTopologicalOrder(ExecutionOrder::PRIORITY_BASED);

  // Check that they are equal.
  ASSERT_EQ(api_nodes_topo_sort_with_priority, ort_topo_sort_with_priority);
}
}  // namespace test
}  // namespace onnxruntime
