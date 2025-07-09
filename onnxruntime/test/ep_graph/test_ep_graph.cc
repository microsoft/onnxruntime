// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <gsl/gsl>
#include <memory>
#include <vector>

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/session/onnxruntime_cxx_api.h"

#define ORT_EP_UTILS_ORT_GRAPH_TO_PROTO_IMPL
#include "core/providers/utils/ort_graph_to_proto.h"

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

static void RunMNISTModel(const ORTCHAR_T* model_path, std::vector<float>& output_data) {
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::SessionOptions sess_options;
  Ort::Session session(*ort_env, model_path, sess_options);

  std::vector<int64_t> input_shape = {1, 1, 28, 28};
  std::vector<float> input_data(28 * 28, 0.5f);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> ort_input_names;

  // Add 'Input3'
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()));
  ort_input_names.push_back("Input3");

  // Run session and get outputs
  std::array<const char*, 1> output_names{"Plus214_Output_0"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(),
                                                    ort_inputs.size(), output_names.data(), output_names.size());

  // Check output type and number of elements.
  Ort::Value& ort_output = ort_outputs[0];
  auto output_type_shape = ort_output.GetTensorTypeAndShapeInfo();
  size_t num_output_elems = output_type_shape.GetElementCount();

  ASSERT_EQ(output_type_shape.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ASSERT_EQ(num_output_elems, 10);

  // Return output data.
  const float* output_values = ort_output.GetTensorData<float>();
  output_data.assign(output_values, output_values + num_output_elems);
}

// Test serializing an OrtGraph (MNIST) to GraphProto. Saves initializers to external file.
// Checks that the outputs of the serialized and original models are identical.
TEST(EpGraphTest, SerializeToProto_Mnist) {
  const ORTCHAR_T* original_model_path = ORT_TSTR("testdata/mnist.onnx");
  const ORTCHAR_T* serialized_model_path = ORT_TSTR("mnist_serialized.onnx");
  std::filesystem::remove(serialized_model_path);

  {
    auto test_graph = TestGraph::Load(original_model_path);
    ASSERT_NE(test_graph, nullptr) << "Failed to load test model";

    // Serialize OrtGraph to GraphProto. Save initializers to external file.
    std::string ext_ini_file_path = "mnist_serialized.bin";
    std::filesystem::remove(ext_ini_file_path);
    std::ofstream ext_ini_ofs(ext_ini_file_path, std::ios::binary);
    auto handle_initializer_data = [&ext_ini_ofs, &ext_ini_file_path](const OrtValueInfo* value_info,
                                                                      const void* data, size_t bytes,
                                                                      bool& is_external, std::string& location,
                                                                      int64_t& offset) -> Ort::Status {
      // OrtValueInfo* could be used to query initializer's name, type, shape,
      // node consumers, etc.
      (void)value_info;

      if (bytes <= 127) {
        is_external = false;  // Keep small initializers stored inside the TensorProto.
        return Ort::Status{nullptr};
      }

      offset = ext_ini_ofs.tellp();
      location = ext_ini_file_path;
      ext_ini_ofs.write(static_cast<const char*>(data), bytes);
      ext_ini_ofs.flush();
      is_external = true;  // True if is external initializer.

      return Ort::Status{nullptr};
    };

    ONNX_NAMESPACE::ModelProto model_proto;
    OrtEpUtils::OrtGraphToProto(test_graph->GetOrtGraph(), model_proto, handle_initializer_data);

    std::ofstream ofs(serialized_model_path, std::ios::binary);
    model_proto.SerializeToOstream(&ofs);
    ofs.flush();

    ASSERT_TRUE(std::filesystem::exists(serialized_model_path));
    ASSERT_TRUE(std::filesystem::exists(ext_ini_file_path));
  }

  // Compare output of the original and serialized models. Should be identical.
  std::vector<float> output_original;
  std::vector<float> output_serialized;

  RunMNISTModel(original_model_path, output_original);
  RunMNISTModel(serialized_model_path, output_serialized);

  EXPECT_EQ(output_serialized, output_original);
}

static void Run3LayerModel(const ORTCHAR_T* model_path, bool input_cond, std::vector<float>& output_data) {
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::SessionOptions sess_options;
  Ort::Session session(*ort_env, model_path, sess_options);

  std::vector<int64_t> input_shape = {1};
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> ort_input_names;

  // Add 'if_cond_input'
  ort_inputs.emplace_back(Ort::Value::CreateTensor<bool>(
      memory_info, &input_cond, 1, input_shape.data(), input_shape.size()));
  ort_input_names.push_back("if_cond_input");

  // Run session and get outputs
  std::array<const char*, 1> output_names{"if_cond_output"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(),
                                                    ort_inputs.size(), output_names.data(), output_names.size());

  // Check output type and number of elements.
  Ort::Value& ort_output = ort_outputs[0];
  auto output_type_shape = ort_output.GetTensorTypeAndShapeInfo();
  size_t num_output_elems = output_type_shape.GetElementCount();

  ASSERT_EQ(output_type_shape.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ASSERT_EQ(num_output_elems, 1);

  // Return output data.
  const float* output_values = ort_output.GetTensorData<float>();
  output_data.assign(output_values, output_values + num_output_elems);
}

// Test serializing an OrtGraph to GraphProto. The model has 3 layers of nested subgraphs.
// Checks that the outputs of the serialized and original models are identical.
TEST(EpGraphTest, SerializeToProto_3LayerSubgraphs) {
  const ORTCHAR_T* original_model_path = ORT_TSTR("testdata/three_layer_nested_subgraph.onnx");
  const ORTCHAR_T* serialized_model_path = ORT_TSTR("three_layer_nested_subgraph_serialized.onnx");
  std::filesystem::remove(serialized_model_path);

  {
    auto test_graph = TestGraph::Load(original_model_path);
    ASSERT_NE(test_graph, nullptr) << "Failed to load test model";

    // Serialize OrtGraph to ModelProto (all initializers stored within TensorProtos).
    ONNX_NAMESPACE::ModelProto model_proto;
    OrtEpUtils::OrtGraphToProto(test_graph->GetOrtGraph(), model_proto);

    std::ofstream ofs(serialized_model_path, std::ios::binary);
    model_proto.SerializeToOstream(&ofs);
    ofs.flush();

    ASSERT_TRUE(std::filesystem::exists(serialized_model_path));
  }

  // Compare output of the original and serialized models. Should be identical.
  std::vector<float> output_original;
  std::vector<float> output_serialized;

  {
    Run3LayerModel(original_model_path, true, output_original);
    Run3LayerModel(serialized_model_path, true, output_serialized);
    EXPECT_EQ(output_serialized, output_original);
  }

  {
    Run3LayerModel(original_model_path, false, output_original);
    Run3LayerModel(serialized_model_path, false, output_serialized);
    EXPECT_EQ(output_serialized, output_original);
  }
}

//
// Utils for traversing an OrtGraph and checking against GraphViewer.
//

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
      OrtValue initializer_value;
      const ONNX_NAMESPACE::TensorProto* initializer = graph_viewer.GetGraph().GetInitializer(value_name,
                                                                                              initializer_value,
                                                                                              is_const_initializer,
                                                                                              /*check_outer_scope*/ true);
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

  size_t api_num_graph_inputs = 0;
  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetNumInputs(&api_graph, &api_num_graph_inputs));
  ASSERT_EQ(api_num_graph_inputs, graph_input_node_args.size());

  std::vector<const OrtValueInfo*> api_graph_inputs(api_num_graph_inputs);
  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetInputs(&api_graph, api_graph_inputs.data(), api_graph_inputs.size()));
  CheckValueInfosCApi(graph_viewer, api_graph_inputs, graph_input_node_args);

  // Check graph outputs.
  const auto& graph_output_node_args = graph_viewer.GetOutputs();

  size_t api_num_graph_outputs = 0;
  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetNumOutputs(&api_graph, &api_num_graph_outputs));
  ASSERT_EQ(api_num_graph_outputs, graph_output_node_args.size());

  std::vector<const OrtValueInfo*> api_graph_outputs(api_num_graph_outputs);
  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetOutputs(&api_graph, api_graph_outputs.data(), api_graph_outputs.size()));
  CheckValueInfosCApi(graph_viewer, api_graph_outputs, graph_output_node_args);

  // Check graph initializers
  const auto& graph_initializers = graph_viewer.GetAllInitializedTensors();

  size_t api_num_initializers = 0;
  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetNumInitializers(&api_graph, &api_num_initializers));
  ASSERT_EQ(api_num_initializers, graph_initializers.size());

  std::vector<const OrtValueInfo*> api_initializers(api_num_initializers);
  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetInitializers(&api_graph, api_initializers.data(), api_initializers.size()));
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
  size_t api_num_nodes = 0;
  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetNumNodes(&api_graph, &api_num_nodes));
  ASSERT_EQ(api_num_nodes, graph_viewer.NumberOfNodes());

  std::vector<const OrtNode*> api_nodes(api_num_nodes);
  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetNodes(&api_graph, api_nodes.data(), api_nodes.size()));

  std::vector<NodeIndex> node_indices = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::DEFAULT);
  for (size_t node_idx = 0; node_idx < api_num_nodes; node_idx++) {
    // Check basic node properties.
    const Node* node = graph_viewer.GetNode(node_indices[node_idx]);
    const OrtNode* api_node = api_nodes[node_idx];
    CheckNode(node, api_node);

    int api_since_version = 0;
    ASSERT_ORTSTATUS_OK(ort_api.Node_GetSinceVersion(api_node, &api_since_version));
    ASSERT_EQ(api_since_version, node->SinceVersion());

    // Check node inputs
    const auto input_node_args = node->InputDefs();

    size_t api_node_num_inputs = 0;
    ASSERT_ORTSTATUS_OK(ort_api.Node_GetNumInputs(api_node, &api_node_num_inputs));
    ASSERT_EQ(api_node_num_inputs, input_node_args.size());

    std::vector<const OrtValueInfo*> api_node_inputs(api_node_num_inputs);
    ASSERT_ORTSTATUS_OK(ort_api.Node_GetInputs(api_node, api_node_inputs.data(), api_node_inputs.size()));
    CheckValueInfosCApi(graph_viewer, api_node_inputs, input_node_args);

    // Check node outputs
    const auto output_node_args = node->OutputDefs();
    size_t api_node_num_outputs = 0;
    ASSERT_ORTSTATUS_OK(ort_api.Node_GetNumOutputs(api_node, &api_node_num_outputs));
    ASSERT_EQ(api_node_num_outputs, output_node_args.size());

    std::vector<const OrtValueInfo*> api_node_outputs(api_node_num_outputs);
    ASSERT_ORTSTATUS_OK(ort_api.Node_GetOutputs(api_node, api_node_outputs.data(), api_node_outputs.size()));
    CheckValueInfosCApi(graph_viewer, api_node_outputs, output_node_args);

    // Check node attributes
    const auto& node_attrs = node->GetAttributes();

    if (!node_attrs.empty()) {
      size_t api_num_node_attributes = 0;
      ASSERT_ORTSTATUS_OK(ort_api.Node_GetNumAttributes(api_node, &api_num_node_attributes));

      std::vector<const OrtOpAttr*> api_node_attributes(api_num_node_attributes);
      ASSERT_ORTSTATUS_OK(ort_api.Node_GetAttributes(api_node, api_node_attributes.data(), api_node_attributes.size()));

      size_t attr_idx = 0;
      for (const auto& node_attr : node_attrs) {
        const OrtOpAttr* api_node_attr = api_node_attributes[attr_idx];
        ASSERT_NE(api_node_attr, nullptr);

        api_node_attr = nullptr;
        ASSERT_ORTSTATUS_OK(ort_api.Node_GetAttributeByName(api_node, node_attr.first.c_str(), &api_node_attr));
        ASSERT_NE(api_node_attr, nullptr);

        const char* api_node_attr_name = nullptr;
        ASSERT_ORTSTATUS_OK(ort_api.OpAttr_GetName(api_node_attr, &api_node_attr_name));
        ASSERT_STREQ(api_node_attr_name, node_attr.first.c_str());

        OrtOpAttrType api_node_attr_type = OrtOpAttrType::ORT_OP_ATTR_UNDEFINED;

        // It's possible that the type is defined in ONNX::AttributeProto_AttributeType but not in OrtOpAttrType, since the two are not in a 1:1 mapping.
        // In such cases, OpAttr_GetType will return a non-null status, and we simply skip the check here.
        OrtStatusPtr status = ort_api.OpAttr_GetType(api_node_attr, &api_node_attr_type);
        if (status != nullptr) {
          Ort::GetApi().ReleaseStatus(status);
          continue;
        }

        ONNX_NAMESPACE::AttributeProto_AttributeType node_attr_type = node_attr.second.type();
        switch (node_attr_type) {
          case ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_UNDEFINED: {
            ASSERT_EQ(api_node_attr_type, OrtOpAttrType::ORT_OP_ATTR_UNDEFINED);
            break;
          }
          case ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT: {
            ASSERT_EQ(api_node_attr_type, OrtOpAttrType::ORT_OP_ATTR_INT);
            break;
          }
          case ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS: {
            ASSERT_EQ(api_node_attr_type, OrtOpAttrType::ORT_OP_ATTR_INTS);
            break;
          }
          case ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT: {
            ASSERT_EQ(api_node_attr_type, OrtOpAttrType::ORT_OP_ATTR_FLOAT);
            break;
          }
          case ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS: {
            ASSERT_EQ(api_node_attr_type, OrtOpAttrType::ORT_OP_ATTR_FLOATS);
            break;
          }
          case ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING: {
            ASSERT_EQ(api_node_attr_type, OrtOpAttrType::ORT_OP_ATTR_STRING);
            break;
          }
          case ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS: {
            ASSERT_EQ(api_node_attr_type, OrtOpAttrType::ORT_OP_ATTR_STRINGS);
            break;
          }
          default:
            // The unsupported type should be skipped by 'continue' above. It's unexpected so we force test to fail.
            ASSERT_ORTSTATUS_OK(ort_api.CreateStatus(ORT_FAIL, "The attribute type is not in AttributeProto_AttributeType and this case shouldn't be hit."));
        }
        attr_idx++;
      }
    }

    // Check node subgraphs
    std::unordered_map<std::string, gsl::not_null<const Graph*>> node_subgraphs_map =
        node->GetAttributeNameToSubgraphMap();

    if (!node_subgraphs_map.empty()) {
      // Check node's implicit inputs to its subgraph nodes.
      const auto implicit_input_node_args = node->ImplicitInputDefs();

      size_t api_num_node_implicit_inputs = 0;
      ASSERT_ORTSTATUS_OK(ort_api.Node_GetNumImplicitInputs(api_node, &api_num_node_implicit_inputs));
      ASSERT_EQ(api_num_node_implicit_inputs, implicit_input_node_args.size());

      std::vector<const OrtValueInfo*> api_node_implicit_inputs(api_num_node_implicit_inputs);
      ASSERT_ORTSTATUS_OK(ort_api.Node_GetImplicitInputs(api_node, api_node_implicit_inputs.data(),
                                                         api_node_implicit_inputs.size()));

      CheckValueInfosCApi(graph_viewer, api_node_implicit_inputs, implicit_input_node_args);

      // Recursively check subgraphs.
      size_t api_num_node_subgraphs = 0;
      ASSERT_ORTSTATUS_OK(ort_api.Node_GetNumSubgraphs(api_node, &api_num_node_subgraphs));
      ASSERT_EQ(api_num_node_subgraphs, node_subgraphs_map.size());

      std::vector<const OrtGraph*> api_node_subgraphs(api_num_node_subgraphs);
      std::vector<const char*> api_subgraph_attr_names(api_num_node_subgraphs);
      ASSERT_ORTSTATUS_OK(ort_api.Node_GetSubgraphs(api_node, api_node_subgraphs.data(), api_node_subgraphs.size(),
                                                    api_subgraph_attr_names.data()));

      for (const auto& [attr_name, subgraph] : node_subgraphs_map) {
        // find index of this subgraph.
        size_t api_subgraph_idx = api_num_node_subgraphs;
        for (size_t subgraph_idx = 0; subgraph_idx < api_num_node_subgraphs; subgraph_idx++) {
          if (api_subgraph_attr_names[subgraph_idx] == attr_name) {
            api_subgraph_idx = subgraph_idx;
            break;
          }
        }
        ASSERT_NE(api_subgraph_idx, api_num_node_subgraphs);

        // Recursively check the subgraph
        auto subgraph_viewer = std::make_unique<GraphViewer>(*subgraph);
        const OrtGraph* api_subgraph = api_node_subgraphs[api_subgraph_idx];
        CheckGraphCApi(*subgraph_viewer, *api_subgraph);
      }
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
