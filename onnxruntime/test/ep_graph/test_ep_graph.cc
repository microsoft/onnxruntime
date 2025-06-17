// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <gsl/gsl>
#include <memory>
#include <vector>

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
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

// forward-declaration for utility that uses public C APIs to check that an OrtGraph is equivalent
// to a graph represented by the internal ORT GraphViewer class.
static void CheckGraphCApi(const GraphViewer& graph_viewer, const OrtGraph& api_graph);

//
//  Tests
//

// Checks that an OrtGraph is initialized correctly and tests basic usage of the C API
// by traversing the OrtGraph and checking validity of nodes and value infos.
TEST(EpGraphTest, BasicCApiUse) {
  auto test_graph = TestGraph::Load(ORT_TSTR("testdata/mnist.onnx"));
  ASSERT_NE(test_graph, nullptr) << "Failed to load test model";

  CheckGraphCApi(test_graph->GetGraphViewer(), test_graph->GetOrtGraph());
}

// Use public C APIs to check that the OrtGraph for a model with subgraphs is correct.
// Traverse OrtGraph with Scan nodes, which tests handling of subgraphs, implicit inputs, and variadic I/O.
TEST(EpGraphTest, CheckModelWithSubgraphs) {
  auto test_graph = TestGraph::Load(ORT_TSTR("testdata/scan_1.onnx"));
  ASSERT_NE(test_graph, nullptr) << "Failed to load test model";

  CheckGraphCApi(test_graph->GetGraphViewer(), test_graph->GetOrtGraph());
}

// Use public C APIs to check that the OrtGraph for bart_tiny.onnx is correct.
// This model is used in an example topological sort implementation.
TEST(EpGraphTest, CheckModelBartTiny) {
  auto test_graph = TestGraph::Load(ORT_TSTR("testdata/bart_tiny.onnx"));
  ASSERT_NE(test_graph, nullptr) << "Failed to load test model";

  CheckGraphCApi(test_graph->GetGraphViewer(), test_graph->GetOrtGraph());
}

TEST(EpGraphTest, Check3LayerNestedSubgraph) {
  // The main graph contains a 'If' node: 'graph_0__if_0'
  // Inside the then-branch of 'graph_0__if_0', there is a nested 'If' node: 'graph_0__if_0__else__if_0'
  // This 3-layer nested graph consumes the same initializer in different subgraphs.
  auto test_graph = TestGraph::Load(ORT_TSTR("testdata/three_layer_nested_subgraph.onnx"));
  ASSERT_NE(test_graph, nullptr) << "Failed to load test model";

  CheckGraphCApi(test_graph->GetGraphViewer(), test_graph->GetOrtGraph());
}

//
// Utils for traversing an OrtGraph and checking against GraphViewer.
//

// Convert an OrtConstPointerArray into a span of Ort___ pointers.
template <typename T>
static void GetSpanFromConstPointerArray(const OrtConstPointerArray* ort_array,
                                         /*out*/ gsl::span<const T* const>& span) {
  const OrtApi& ort_api = Ort::GetApi();

  size_t size = 0;
  ASSERT_ORTSTATUS_OK(ort_api.ConstPointerArray_GetSize(ort_array, &size));

  const void* const* raw_data = nullptr;
  ASSERT_ORTSTATUS_OK(ort_api.ConstPointerArray_GetData(ort_array, &raw_data));

  auto data = reinterpret_cast<const T* const*>(raw_data);
  span = gsl::span<const T* const>(data, size);
}

// Checks that the OrtTypeInfo obtained from the public C API matches another OrtTypeInfo
// obtained from the internal ORT graph IR.
static void CheckTypeInfo(const OrtTypeInfo* api_type_info, const OrtTypeInfo* type_info) {
  const OrtApi& ort_api = Ort::GetApi();

  ASSERT_NE(api_type_info, nullptr);
  ASSERT_NE(type_info, nullptr);

  ONNXType api_onnx_type = ONNX_TYPE_UNKNOWN;
  ASSERT_ORTSTATUS_OK(ort_api.GetOnnxTypeFromTypeInfo(api_type_info, &api_onnx_type));
  ASSERT_EQ(api_onnx_type, type_info->type);

  if (api_onnx_type == ONNX_TYPE_TENSOR) {
    // Only validating Tensors (not checking Map, Sequence, etc.) values because these C APIs for getting
    // type/shape information existed long before the new ORT graph IR APIs and are tested elsewhere.
    const OrtTensorTypeAndShapeInfo* api_type_shape = nullptr;
    ASSERT_ORTSTATUS_OK(ort_api.CastTypeInfoToTensorInfo(api_type_info, &api_type_shape));
    ASSERT_NE(api_type_shape, nullptr);

    ONNXTensorElementDataType api_elem_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    ASSERT_ORTSTATUS_OK(ort_api.GetTensorElementType(api_type_shape, &api_elem_type));
    ASSERT_EQ(api_elem_type, type_info->tensor_type_info->type);

    size_t api_num_dims = 0;
    ASSERT_ORTSTATUS_OK(ort_api.GetDimensionsCount(api_type_shape, &api_num_dims));
    ASSERT_EQ(api_num_dims, type_info->tensor_type_info->shape.NumDimensions());

    std::vector<int64_t> api_dims(api_num_dims, 0);
    ASSERT_ORTSTATUS_OK(ort_api.GetDimensions(api_type_shape, api_dims.data(), api_dims.size()));
    ASSERT_EQ(gsl::span<const int64_t>(api_dims), type_info->tensor_type_info->shape.GetDims());

    std::vector<const char*> api_dim_syms(api_num_dims, nullptr);
    ASSERT_ORTSTATUS_OK(ort_api.GetSymbolicDimensions(api_type_shape, api_dim_syms.data(), api_dim_syms.size()));
    const std::vector<std::string>& dim_syms = type_info->tensor_type_info->dim_params;
    for (size_t dim_idx = 0; dim_idx < api_num_dims; dim_idx++) {
      ASSERT_EQ(std::string(api_dim_syms[dim_idx]), dim_syms[dim_idx]);
    }
  }
}

// Checks that the given OrtNode matches the onnxruntime::Node.
static void CheckNode(const Node* node, const OrtNode* api_node) {
  const OrtApi& ort_api = Ort::GetApi();

  size_t api_node_id = 0;
  const char* api_node_name = nullptr;
  const char* api_op_type = nullptr;
  const char* api_domain = nullptr;

  ASSERT_ORTSTATUS_OK(ort_api.Node_GetId(api_node, &api_node_id));
  ASSERT_ORTSTATUS_OK(ort_api.Node_GetName(api_node, &api_node_name));
  ASSERT_ORTSTATUS_OK(ort_api.Node_GetOperatorType(api_node, &api_op_type));
  ASSERT_ORTSTATUS_OK(ort_api.Node_GetDomain(api_node, &api_domain));

  ASSERT_EQ(api_node_id, node->Index());
  ASSERT_EQ(std::string(api_node_name), node->Name());
  ASSERT_EQ(std::string(api_op_type), node->OpType());
  ASSERT_EQ(std::string(api_domain), node->Domain());
}

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
      CheckNode(producer_node, api_producer_node);

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
    CheckNode(node_arg_consumers[i].node, api_node_consumers[i]);
    ASSERT_EQ(api_input_indices[i], static_cast<int64_t>(node_arg_consumers[i].input_index));
  }
}

static void CheckInitializerValueInfo(const OrtValueInfo* api_value_info,
                                      const ONNX_NAMESPACE::TensorProto* tensor_proto) {
  const OrtApi& ort_api = Ort::GetApi();

  const OrtValue* api_initializer_value = nullptr;
  ASSERT_ORTSTATUS_OK(ort_api.ValueInfo_GetInitializerValue(api_value_info, &api_initializer_value));
  ASSERT_NE(api_initializer_value, nullptr);

  const char* api_initializer_name = nullptr;
  ASSERT_ORTSTATUS_OK(ort_api.GetValueInfoName(api_value_info, &api_initializer_name));
  ASSERT_NE(api_initializer_name, nullptr);

  // Check initializer type.
  const ONNX_NAMESPACE::TypeProto type_proto = utils::TypeProtoFromTensorProto(*tensor_proto);
  auto type_info = OrtTypeInfo::FromTypeProto(type_proto);

  const OrtTypeInfo* api_type_info = nullptr;
  ASSERT_ORTSTATUS_OK(ort_api.GetValueInfoTypeInfo(api_value_info, &api_type_info));
  CheckTypeInfo(api_type_info, type_info.get());
}

static void CheckInitializerValueInfosCApi(gsl::span<const OrtValueInfo* const> initializer_value_infos,
                                           const InitializedTensorSet& initializer_tensor_protos) {
  const OrtApi& ort_api = Ort::GetApi();

  for (size_t i = 0; i < initializer_value_infos.size(); i++) {
    const OrtValueInfo* api_value_info = initializer_value_infos[i];

    const char* api_initializer_name = nullptr;
    ASSERT_ORTSTATUS_OK(ort_api.GetValueInfoName(api_value_info, &api_initializer_name));
    ASSERT_NE(api_initializer_name, nullptr);

    auto tensor_proto_iter = initializer_tensor_protos.find(api_initializer_name);
    ASSERT_NE(tensor_proto_iter, initializer_tensor_protos.end());

    const ONNX_NAMESPACE::TensorProto* tensor_proto = tensor_proto_iter->second;
    ASSERT_NE(tensor_proto, nullptr);

    CheckInitializerValueInfo(api_value_info, tensor_proto);
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
      const auto& value_name = node_arg->Name();

      ASSERT_NE(value_info, nullptr);

      const char* api_name = nullptr;
      ASSERT_ORTSTATUS_OK(ort_api.GetValueInfoName(value_info, &api_name));
      ASSERT_EQ(std::string(api_name), value_name);

      bool is_graph_input = std::any_of(graph_viewer_inputs.begin(), graph_viewer_inputs.end(),
                                        [&node_arg](const NodeArg* graph_input) {
                                          return node_arg->Name() == graph_input->Name();
                                        });

      bool is_graph_output = std::any_of(graph_viewer_outputs.begin(), graph_viewer_outputs.end(),
                                         [&node_arg](const NodeArg* graph_output) {
                                           return node_arg->Name() == graph_output->Name();
                                         });
      bool is_const_initializer = false;
      const ONNX_NAMESPACE::TensorProto* initializer = graph_viewer.GetGraph().GetInitializer(value_name, true,
                                                                                              is_const_initializer);
      bool can_override_initializer = graph_viewer.CanOverrideInitializer();

      bool api_is_req_graph_input = false;
      bool api_is_opt_graph_input = false;
      ASSERT_ORTSTATUS_OK(ort_api.ValueInfo_IsRequiredGraphInput(value_info, &api_is_req_graph_input));
      ASSERT_ORTSTATUS_OK(ort_api.ValueInfo_IsOptionalGraphInput(value_info, &api_is_opt_graph_input));
      ASSERT_EQ(api_is_req_graph_input, is_graph_input && (initializer == nullptr));
      ASSERT_EQ(api_is_opt_graph_input, can_override_initializer && (initializer != nullptr) && !is_const_initializer);

      bool api_is_graph_output = false;
      ASSERT_ORTSTATUS_OK(ort_api.ValueInfo_IsGraphOutput(value_info, &api_is_graph_output));
      ASSERT_EQ(api_is_graph_output, is_graph_output);

      bool is_outer_scope = graph_viewer.GetGraph().IsOuterScopeValue(node_arg->Name());
      bool api_is_outer_scope = false;
      ASSERT_ORTSTATUS_OK(ort_api.ValueInfo_IsFromOuterScope(value_info, &api_is_outer_scope));
      ASSERT_EQ(api_is_outer_scope, is_outer_scope);

      bool api_is_const_initializer = false;
      ASSERT_ORTSTATUS_OK(ort_api.ValueInfo_IsConstantInitializer(value_info, &api_is_const_initializer));
      ASSERT_EQ(api_is_const_initializer, is_const_initializer);

      if (is_const_initializer || api_is_opt_graph_input) {
        CheckInitializerValueInfo(value_info, initializer);
      } else {
        auto node_arg_type_info = OrtTypeInfo::FromTypeProto(*node_arg->TypeAsProto());
        const OrtTypeInfo* api_type_info = nullptr;
        ASSERT_ORTSTATUS_OK(ort_api.GetValueInfoTypeInfo(value_info, &api_type_info));
        CheckTypeInfo(api_type_info, node_arg_type_info.get());
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

  const OrtConstPointerArray* api_graph_inputs_container = nullptr;
  gsl::span<const OrtValueInfo* const> api_graph_inputs{};

  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetInputs(&api_graph, &api_graph_inputs_container));
  GetSpanFromConstPointerArray<OrtValueInfo>(api_graph_inputs_container, api_graph_inputs);

  ASSERT_EQ(api_graph_inputs.size(), graph_input_node_args.size());
  CheckValueInfosCApi(graph_viewer, api_graph_inputs, graph_input_node_args);

  // Check graph outputs.
  const auto& graph_output_node_args = graph_viewer.GetOutputs();

  const OrtConstPointerArray* api_graph_outputs_container = nullptr;
  gsl::span<const OrtValueInfo* const> api_graph_outputs{};

  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetOutputs(&api_graph, &api_graph_outputs_container));
  GetSpanFromConstPointerArray<OrtValueInfo>(api_graph_outputs_container, api_graph_outputs);

  ASSERT_EQ(api_graph_outputs.size(), graph_output_node_args.size());
  CheckValueInfosCApi(graph_viewer, api_graph_outputs, graph_output_node_args);

  // Check graph initializers
  const auto& graph_initializers = graph_viewer.GetAllInitializedTensors();

  const OrtConstPointerArray* api_initializers_container = nullptr;
  gsl::span<const OrtValueInfo* const> api_initializers{};

  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetInitializers(&api_graph, &api_initializers_container));
  GetSpanFromConstPointerArray<OrtValueInfo>(api_initializers_container, api_initializers);

  ASSERT_EQ(api_initializers.size(), graph_initializers.size());
  CheckInitializerValueInfosCApi(api_initializers, graph_initializers);

  // Check if it has a parent node.
  const Node* parent_node = graph_viewer.ParentNode();
  const bool has_parent_node = parent_node != nullptr;
  const OrtNode* api_parent_node = nullptr;

  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetParentNode(&api_graph, &api_parent_node));
  const bool api_has_parent_node = api_parent_node != nullptr;
  ASSERT_EQ(api_has_parent_node, has_parent_node);

  if (has_parent_node) {
    CheckNode(parent_node, api_parent_node);
  }

  // Check all nodes.
  const OrtConstPointerArray* api_nodes_container = nullptr;
  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetNodes(&api_graph, &api_nodes_container));

  size_t api_num_nodes = 0;
  ASSERT_ORTSTATUS_OK(ort_api.ConstPointerArray_GetSize(api_nodes_container, &api_num_nodes));
  ASSERT_EQ(api_num_nodes, graph_viewer.NumberOfNodes());

  std::vector<NodeIndex> node_indices = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::DEFAULT);
  for (size_t node_idx = 0; node_idx < api_num_nodes; node_idx++) {
    // Check basic node properties.
    const Node* node = graph_viewer.GetNode(node_indices[node_idx]);
    const OrtNode* api_node = nullptr;
    ASSERT_ORTSTATUS_OK(ort_api.ConstPointerArray_GetElementAt(api_nodes_container, node_idx,
                                                               reinterpret_cast<const void**>(&api_node)));
    CheckNode(node, api_node);

    int api_since_version = 0;
    ASSERT_ORTSTATUS_OK(ort_api.Node_GetSinceVersion(api_node, &api_since_version));
    ASSERT_EQ(api_since_version, node->SinceVersion());

    // Check node inputs
    const auto input_node_args = node->InputDefs();

    const OrtConstPointerArray* api_node_inputs_container = nullptr;
    gsl::span<const OrtValueInfo* const> api_node_inputs{};

    ASSERT_ORTSTATUS_OK(ort_api.Node_GetInputs(api_node, &api_node_inputs_container));
    GetSpanFromConstPointerArray<OrtValueInfo>(api_node_inputs_container, api_node_inputs);
    ASSERT_EQ(api_node_inputs.size(), input_node_args.size());

    CheckValueInfosCApi(graph_viewer, api_node_inputs, input_node_args);

    // Check node outputs
    const auto output_node_args = node->OutputDefs();
    const OrtConstPointerArray* api_node_outputs_container = nullptr;
    gsl::span<const OrtValueInfo* const> api_node_outputs{};

    ASSERT_ORTSTATUS_OK(ort_api.Node_GetOutputs(api_node, &api_node_outputs_container));
    GetSpanFromConstPointerArray<OrtValueInfo>(api_node_outputs_container, api_node_outputs);
    ASSERT_EQ(api_node_outputs.size(), output_node_args.size());

    CheckValueInfosCApi(graph_viewer, api_node_outputs, output_node_args);

    // Check node subgraphs
    std::vector<gsl::not_null<const Graph*>> node_subgraphs = node->GetSubgraphs();
    size_t api_num_subgraphs = 0;
    ASSERT_ORTSTATUS_OK(ort_api.Node_GetNumSubgraphs(api_node, &api_num_subgraphs));
    ASSERT_EQ(api_num_subgraphs, node_subgraphs.size());

    if (api_num_subgraphs > 0) {
      // Check node's implicit inputs to its subgraph nodes.
      const auto implicit_input_node_args = node->ImplicitInputDefs();
      const OrtConstPointerArray* api_node_implicit_inputs_container = nullptr;
      gsl::span<const OrtValueInfo* const> api_node_implicit_inputs{};

      ASSERT_ORTSTATUS_OK(ort_api.Node_GetImplicitInputs(api_node, &api_node_implicit_inputs_container));
      GetSpanFromConstPointerArray<OrtValueInfo>(api_node_implicit_inputs_container, api_node_implicit_inputs);
      ASSERT_EQ(api_node_implicit_inputs.size(), implicit_input_node_args.size());

      CheckValueInfosCApi(graph_viewer, api_node_implicit_inputs, implicit_input_node_args);

      // Recursively check subgraphs.
      std::vector<const OrtGraph*> api_node_subgraphs(api_num_subgraphs, nullptr);
      ASSERT_ORTSTATUS_OK(ort_api.Node_GetSubgraphs(api_node, api_node_subgraphs.data(), api_node_subgraphs.size()));

      for (size_t subgraph_idx = 0; subgraph_idx < api_num_subgraphs; subgraph_idx++) {
        auto subgraph_viewer = std::make_unique<GraphViewer>(*node_subgraphs[subgraph_idx]);
        CheckGraphCApi(*subgraph_viewer, *api_node_subgraphs[subgraph_idx]);
      }
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
