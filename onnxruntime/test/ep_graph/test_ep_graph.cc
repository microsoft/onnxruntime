// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <gsl/gsl>
#include <memory>

#include "core/common/common.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/graph/ep_api_types.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"

// defined in unittest_main/test_main.cc
extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

/// <summary>
/// Utility that loads a model from file and provides a OrtGraph view of the model for testing the public graph APIs.
/// </summary>
struct TestGraph {
  explicit TestGraph(std::shared_ptr<Model> model)
      : model(model), graph_viewer(model->MainGraph()) {
    api_graph = EpGraph::Create(graph_viewer);
  }

  static std::unique_ptr<TestGraph> Load(const ORTCHAR_T* model_path) {
    std::shared_ptr<Model> model;
    auto status = Model::Load(model_path, model, nullptr, DefaultLoggingManager().DefaultLogger());
    if (!status.IsOK()) {
      return nullptr;
    }

    return std::make_unique<TestGraph>(model);
  }

  const OrtGraph& GetOrtGraph() const { return *api_graph; }

  std::shared_ptr<Model> model;
  GraphViewer graph_viewer;
  std::unique_ptr<OrtGraph> api_graph;
};

// Gets the input or output index of a NodeArg by name.
static void GetInputOrOutputIndices(ConstPointerContainer<std::vector<NodeArg*>> node_args,
                                    const std::string& name,
                                    /*out*/ std::vector<size_t>& indices) {
  indices.reserve(node_args.size());

  bool found = false;
  for (size_t i = 0; i < node_args.size(); i++) {
    if (node_args[i]->Name() == name) {
      indices.push_back(i);
      found = true;
    }
  }

  ASSERT_TRUE(found) << "Did not find NodeArg's index";
}

struct NodeArgUse {
  NodeArgUse(const Node* node, size_t index) : consumer_node(node), input_index(index) {}
  const Node* consumer_node = nullptr;
  size_t input_index = 0;
};

// Returns "uses" (i.e., consumer node + input index) of a NodeArg from the original graph.
static void GetNodeArgUses(const GraphViewer& graph_viewer, const NodeArg& node_arg, std::vector<NodeArgUse>& uses) {
  std::vector<const Node*> nodes = graph_viewer.GetConsumerNodes(node_arg.Name());
  if (nodes.empty()) {
    return;
  }

  uses.reserve(nodes.size());
  for (const Node* node : nodes) {
    bool within_graph_viewer = node != nullptr && graph_viewer.GetNode(node->Index()) != nullptr;
    if (!within_graph_viewer) {
      continue;  // Node is not in this GraphViewer
    }

    std::vector<size_t> input_indices;
    GetInputOrOutputIndices(node->InputDefs(), node_arg.Name(), input_indices);

    for (size_t input_index : input_indices) {
      NodeArgUse use_info(node, input_index);
      uses.push_back(use_info);
    }
  }
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
  ASSERT_ORTSTATUS_OK(ort_api.GetValueProducer(value_info, &api_producer_node, &api_producer_output_index));

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

      std::vector<size_t> indices;
      GetInputOrOutputIndices(producer_node->OutputDefs(), node_arg->Name(), indices);
      ASSERT_EQ(indices.size(), 1);
      ASSERT_EQ(api_producer_output_index, indices[0]);
    }
  }
}

// Checks that "uses" of a OrtValueInfo obtained from the public C API are valid by comparing to the original graph.
static void CheckValueInfoUses(const GraphViewer& graph_viewer, const OrtValueInfo* value_info,
                               const NodeArg* node_arg) {
  const OrtApi& ort_api = Ort::GetApi();

  if (!node_arg->Exists()) {
    return;
  }

  std::vector<NodeArgUse> node_arg_uses;
  GetNodeArgUses(graph_viewer, *node_arg, node_arg_uses);

  size_t api_num_consumers = 0;
  ASSERT_ORTSTATUS_OK(ort_api.GetValueNumConsumers(value_info, &api_num_consumers));
  ASSERT_EQ(api_num_consumers, node_arg_uses.size());

  std::vector<const OrtNode*> api_node_consumers(api_num_consumers, nullptr);
  std::vector<int64_t> api_input_indices(api_num_consumers, 0);
  ASSERT_ORTSTATUS_OK(ort_api.GetValueConsumers(value_info, api_node_consumers.data(), api_input_indices.data(),
                                                api_num_consumers));

  for (size_t i = 0; i < api_num_consumers; i++) {
    ASSERT_EQ(std::string(ort_api.Node_Name(api_node_consumers[i])), node_arg_uses[i].consumer_node->Name());
    ASSERT_EQ(std::string(ort_api.Node_OperatorType(api_node_consumers[i])), node_arg_uses[i].consumer_node->OpType());
    ASSERT_EQ(std::string(ort_api.Node_Domain(api_node_consumers[i])), node_arg_uses[i].consumer_node->Domain());
    ASSERT_EQ(api_input_indices[i], static_cast<int64_t>(node_arg_uses[i].input_index));
  }
}

// Checks that the OrtValueInfos obtained from the public C API are "equivalent" to the NodeArgs
// in the original graph.
static void CheckValueInfosCApi(const GraphViewer& graph_viewer, gsl::span<const OrtValueInfo* const> value_infos,
                                gsl::span<const NodeArg* const> node_args) {
  ASSERT_EQ(value_infos.size(), node_args.size());
  const OrtApi& ort_api = Ort::GetApi();

  for (size_t i = 0; i < value_infos.size(); i++) {
    const NodeArg* node_arg = node_args[i];
    const OrtValueInfo* value_info = value_infos[i];

    if (node_arg->Exists()) {
      ASSERT_NE(value_info, nullptr);

      const char* api_name = nullptr;
      ASSERT_ORTSTATUS_OK(ort_api.GetValueInfoName(value_info, &api_name));
      ASSERT_EQ(std::string(api_name), node_arg->Name());

      auto node_arg_type_info = OrtTypeInfo::FromTypeProto(*node_arg->TypeAsProto());
      const OrtTypeInfo* api_type_info = nullptr;
      ASSERT_ORTSTATUS_OK(ort_api.GetValueInfoTypeInfo(value_info, &api_type_info));
      ASSERT_NE(api_type_info, nullptr);

      ONNXType api_onnx_type = ONNX_TYPE_UNKNOWN;
      ASSERT_ORTSTATUS_OK(ort_api.GetOnnxTypeFromTypeInfo(api_type_info, &api_onnx_type));
      ASSERT_EQ(api_onnx_type, node_arg_type_info->type);

      if (api_onnx_type == ONNX_TYPE_TENSOR) {
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
      } else {
        // TODO(adrianlizarraga): Check Map, Sequence, etc.
      }

      CheckValueInfoProducer(graph_viewer, value_info, node_arg);
      CheckValueInfoUses(graph_viewer, value_info, node_arg);
    } else {
      ASSERT_EQ(value_info, nullptr);  // A missing optional input has a null OrtValueInfo.
    }
  }
}

static std::vector<const NodeArg*> ToVector(ConstPointerContainer<std::vector<NodeArg*>> node_args) {
  std::vector<const NodeArg*> result;
  result.reserve(node_args.size());
  for (const NodeArg* node_arg : node_args) {
    result.push_back(node_arg);
  }
  return result;
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

    const auto input_node_args = node->InputDefs();
    const size_t num_inputs = ort_api.Node_NumInputs(api_node);
    ASSERT_EQ(num_inputs, input_node_args.size());

    std::vector<const OrtValueInfo*> api_inputs(num_inputs, nullptr);
    ASSERT_ORTSTATUS_OK(ort_api.Node_GetInputs(api_node, api_inputs.data(), api_inputs.size()));
    CheckValueInfosCApi(graph_viewer, api_inputs, ToVector(input_node_args));

    const auto output_node_args = node->OutputDefs();
    const size_t num_outputs = ort_api.Node_NumOutputs(api_node);
    ASSERT_EQ(num_outputs, output_node_args.size());

    std::vector<const OrtValueInfo*> api_outputs(num_outputs, nullptr);
    ASSERT_ORTSTATUS_OK(ort_api.Node_GetOutputs(api_node, api_outputs.data(), api_outputs.size()));
    CheckValueInfosCApi(graph_viewer, api_outputs, ToVector(output_node_args));
  }
}

// Checks that an OrtGraph is initialized correctly and tests basic usage of the C API
// by traversing the OrtGraph and checking validity of nodes and value infos.
TEST(EpGraphTest, BasicCApiUse) {
  auto test_graph = TestGraph::Load(ORT_TSTR("testdata/mnist.onnx"));
  ASSERT_NE(test_graph, nullptr) << "Failed to load test model";

  CheckGraphCApi(test_graph->graph_viewer, test_graph->GetOrtGraph());
}
}  // namespace test
}  // namespace onnxruntime
