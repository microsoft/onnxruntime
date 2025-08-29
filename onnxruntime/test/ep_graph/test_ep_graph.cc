// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <gsl/gsl>
#include <memory>
#include <vector>
#include <fstream>

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/tensor_type_and_shape.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/graph/ep_api_types.h"
#include "core/graph/graph_proto_serializer.h"

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
static void Check_Graph_GetSubgraph(const OrtGraph& api_graph);

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

TEST(EpGraphTest, Check3LayerNestedSubgraphV2) {
  // The overall structure of this model is similar to the one used in "Check3LayerNestedSubgraph" test.
  // The model consists of a graph with subgraphs nested across three levels.
  // In this scenario, a third-layer subgraph consumes an input from the first-layer graph (not an initializer).
  auto test_graph = TestGraph::Load(ORT_TSTR("testdata/three_layer_nested_subgraph_v2.onnx"));
  ASSERT_NE(test_graph, nullptr) << "Failed to load test model";

  CheckGraphCApi(test_graph->GetGraphViewer(), test_graph->GetOrtGraph());
}

TEST(EpGraphTest, GetAttributeByName) {
  // Load model with a single Conv that has no explicit attributes set.
  auto test_graph = TestGraph::Load(ORT_TSTR("testdata/conv_default_attrs.onnx"));
  ASSERT_NE(test_graph, nullptr) << "Failed to load test model";

  //
  // Pre-check
  //

  // Original Conv has no explicit attributes but Graph::Resolve() fills in default values for
  // 'auto_pad' and 'group'. The other optional attributes (i.e. dilations, kernel_shape, pads, strides) do not
  // have statically computable default values, so will not be filled in by Graph::Resolve().
  const OrtGraph& ort_graph = test_graph->GetOrtGraph();
  Ort::ConstGraph graph{&ort_graph};

  auto nodes = graph.GetNodes();
  ASSERT_EQ(nodes.size(), 1);

  auto conv_node = nodes[0];
  auto op_type = conv_node.GetOperatorType();
  ASSERT_EQ(op_type, "Conv");

  auto attrs = conv_node.GetAttributes();
  ASSERT_EQ(attrs.size(), 2);

  for (const auto& attr : attrs) {
    auto attr_name = attr.GetName();
    ASSERT_TRUE(attr_name == "auto_pad" || attr_name == "group");  // Only 'auto_pad' and 'group' have been set
  }

  //
  // Test 1: Get optional attribute that is not set (e.g., dilations). Should not get an error.
  //
  {
    Ort::ConstOpAttr attr;
    auto status = conv_node.GetAttributeByName("dilations", attr);
    ASSERT_EQ(attr, nullptr);
  }

  //
  // Test 2: Get attribute that does not exist in operator schema. Should get a ORT_NOT_FOUND error.
  //
  {
    Ort::ConstOpAttr attr;
    Ort::Status status = conv_node.GetAttributeByName("_does_not_exist_", attr);
    ASSERT_FALSE(status.IsOK());
    ASSERT_EQ(status.GetErrorCode(), ORT_NOT_FOUND);
    ASSERT_EQ(attr, nullptr);
  }

  //
  // Test 3: Get attribute that is known to be set.
  //
  {
    Ort::ConstOpAttr attr;
    ASSERT_ORTSTATUS_OK(conv_node.GetAttributeByName("auto_pad", attr));
    ASSERT_NE(attr, nullptr);

    OrtOpAttrType type = attr.GetType();
    ASSERT_EQ(ORT_OP_ATTR_STRING, type);
    std::string auto_pad_val;
    ASSERT_ORTSTATUS_OK(attr.GetValue<std::string>(auto_pad_val));
    ASSERT_EQ(auto_pad_val, "NOTSET");
  }
}

// Check correctness of an OrtGraph that has external initializers.
TEST(EpGraphTest, CheckModelExternalInitializers) {
  auto test_graph = TestGraph::Load(ORT_TSTR("testdata/conv_qdq_external_ini.onnx"));
  ASSERT_NE(test_graph, nullptr) << "Failed to load test model";

  CheckGraphCApi(test_graph->GetGraphViewer(), test_graph->GetOrtGraph());
}

static void RunConvQDQExtIni(const ORTCHAR_T* model_path, std::vector<float>& output_data) {
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::SessionOptions sess_options;
  Ort::Session session(*ort_env, model_path, sess_options);

  std::vector<int64_t> input_shape = {1, 3, 24, 24};
  std::vector<float> input_data(3 * 24 * 24, 0.5f);
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> ort_input_names;

  // Add 'input'
  ort_inputs.emplace_back(Ort::Value::CreateTensor<float>(
      memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()));
  ort_input_names.push_back("input");

  // Run session and get outputs
  std::array<const char*, 1> output_names{"output"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(),
                                                    ort_inputs.size(), output_names.data(), output_names.size());

  // Check output type and number of elements.
  Ort::Value& ort_output = ort_outputs[0];
  auto output_type_shape = ort_output.GetTensorTypeAndShapeInfo();
  size_t num_output_elems = output_type_shape.GetElementCount();

  ASSERT_EQ(output_type_shape.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ASSERT_EQ(num_output_elems, 32 * 26 * 26);

  // Return output data.
  const float* output_values = ort_output.GetTensorData<float>();
  output_data.assign(output_values, output_values + num_output_elems);
}

// Test serializing an OrtGraph with external initializers to GraphProto.
// Checks that the outputs of the serialized and original models are identical.
TEST(EpGraphTest, SerializeToProto_InputModelHasExternalIni) {
  const ORTCHAR_T* original_model_path = ORT_TSTR("testdata/conv_qdq_external_ini.onnx");
  const ORTCHAR_T* serialized_model_path = ORT_TSTR("conv_qdq_ext_ini_serialized.onnx");
  std::filesystem::remove(serialized_model_path);

  {
    auto test_graph = TestGraph::Load(original_model_path);
    ASSERT_NE(test_graph, nullptr) << "Failed to load test model";

    // Serialize OrtGraph to GraphProto. Save initializers to external file.
    std::string ext_ini_file_path = "conv_qdq_ext_ini_serialized.bin";
    std::filesystem::remove(ext_ini_file_path);
    std::ofstream ext_ini_ofs(ext_ini_file_path, std::ios::binary);
    auto handle_initializer_data = [&ext_ini_ofs, &ext_ini_file_path](const OrtValueInfo* /* value_info */,
                                                                      const void* data, size_t bytes,
                                                                      bool& is_external, std::string& location,
                                                                      int64_t& offset) -> Ort::Status {
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
    ASSERT_CXX_ORTSTATUS_OK(OrtEpUtils::OrtGraphToProto(test_graph->GetOrtGraph(), model_proto,
                                                        handle_initializer_data));

    std::ofstream ofs(serialized_model_path, std::ios::binary);
    model_proto.SerializeToOstream(&ofs);
    ofs.flush();

    ASSERT_TRUE(std::filesystem::exists(serialized_model_path));
    ASSERT_TRUE(std::filesystem::exists(ext_ini_file_path));
  }

  // Compare output of the original and serialized models. Should be identical.
  std::vector<float> output_original;
  std::vector<float> output_serialized;

  RunConvQDQExtIni(original_model_path, output_original);
  RunConvQDQExtIni(serialized_model_path, output_serialized);

  EXPECT_EQ(output_serialized, output_original);
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

static void RunConstantOfShapeModel(const ORTCHAR_T* model_path, std::vector<float>& output_data) {
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::SessionOptions sess_options;
  Ort::Session session(*ort_env, model_path, sess_options);

  std::vector<int64_t> input_shape = {3};
  std::vector<int64_t> input_data = {2, 3, 4};
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> ort_input_names;

  // Add 'x'
  ort_inputs.emplace_back(Ort::Value::CreateTensor<int64_t>(
      memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size()));
  ort_input_names.push_back("x");

  // Run session and get outputs
  std::array<const char*, 1> output_names{"y"};
  std::vector<Ort::Value> ort_outputs = session.Run(Ort::RunOptions{nullptr}, ort_input_names.data(), ort_inputs.data(),
                                                    ort_inputs.size(), output_names.data(), output_names.size());

  // Check output type and number of elements.
  Ort::Value& ort_output = ort_outputs[0];
  auto output_type_shape = ort_output.GetTensorTypeAndShapeInfo();
  size_t num_output_elems = output_type_shape.GetElementCount();

  ASSERT_EQ(output_type_shape.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  ASSERT_EQ(num_output_elems, 24);

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
    ASSERT_CXX_ORTSTATUS_OK(OrtEpUtils::OrtGraphToProto(test_graph->GetOrtGraph(), model_proto,
                                                        handle_initializer_data));

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

// Test serializing an OrtGraph (MNIST) to GraphProto. Initializers are configured as "external" but point to
// existing data in memory (not standard ONNX).
TEST(EpGraphTest, SerializeToProto_ExternalInitializersInMemory) {
  const ORTCHAR_T* original_model_path = ORT_TSTR("testdata/mnist.onnx");
  auto test_graph = TestGraph::Load(original_model_path);
  ASSERT_NE(test_graph, nullptr) << "Failed to load test model";

  const OrtGraph& ort_graph = test_graph->GetOrtGraph();

  auto handle_initializer_data = [](const OrtValueInfo* value_info,
                                    const void* data, size_t bytes,
                                    bool& is_external, std::string& location,
                                    int64_t& offset) -> Ort::Status {
    (void)value_info;
    (void)bytes;

    offset = reinterpret_cast<int64_t>(data);
    location = "_MEM_ADDR_";
    is_external = true;  // True if is external initializer.

    return Ort::Status{nullptr};
  };

  ONNX_NAMESPACE::GraphProto graph_proto;
  ASSERT_CXX_ORTSTATUS_OK(OrtEpUtils::OrtGraphToProto(ort_graph, graph_proto, handle_initializer_data));

  // Verify that TensorProto objects within GraphProto point to memory owned by OrtValues in the OrtGraph.
  const OrtApi& ort_api = Ort::GetApi();

  size_t api_num_initializers = 0;
  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetNumInitializers(&ort_graph, &api_num_initializers));

  std::vector<const OrtValueInfo*> api_initializers(api_num_initializers);
  ASSERT_ORTSTATUS_OK(ort_api.Graph_GetInitializers(&ort_graph, api_initializers.data(), api_initializers.size()));

  const auto& tensor_protos = graph_proto.initializer();
  ASSERT_EQ(tensor_protos.size(), api_num_initializers);

  std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto*> tensor_proto_map;
  for (const auto& tensor_proto : tensor_protos) {
    tensor_proto_map.emplace(tensor_proto.name(), &tensor_proto);
  }

  for (size_t i = 0; i < api_num_initializers; ++i) {
    std::string value_name;
    Ort::ConstValue ort_value;

    Ort::ConstValueInfo vi(api_initializers[i]);
    value_name = vi.GetName();
    ASSERT_ORTSTATUS_OK(vi.GetInitializer(ort_value));
    const void* ort_value_data = ort_value.GetTensorRawData();

    auto iter = tensor_proto_map.find(value_name);
    ASSERT_NE(iter, tensor_proto_map.end());
    const ONNX_NAMESPACE::TensorProto* tensor_proto = iter->second;
    ONNX_NAMESPACE::TensorProto_DataLocation data_location = tensor_proto->data_location();
    ASSERT_EQ(data_location, ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);

    const auto& ext_data_entries = tensor_proto->external_data();
    const ONNX_NAMESPACE::StringStringEntryProto& location_entry = ext_data_entries[0];
    const ONNX_NAMESPACE::StringStringEntryProto& offset_entry = ext_data_entries[1];

    ASSERT_EQ(location_entry.key(), "location");
    ASSERT_EQ(location_entry.value(), "_MEM_ADDR_");
    ASSERT_EQ(offset_entry.key(), "offset");

    long long offset_int = std::stoll(offset_entry.value());
    ASSERT_EQ(offset_int, reinterpret_cast<long long>(ort_value_data));
  }
}

// Test serializing an OrtGraph (MNIST) to GraphProto. Saves initializers to external file.
// Checks that the outputs of the serialized and original models are identical.
TEST(EpGraphTest, SerializeToProto_ConstantOfShape) {
  const ORTCHAR_T* original_model_path = ORT_TSTR("testdata/ort_minimal_test_models/tensor_attribute.onnx");
  const ORTCHAR_T* serialized_model_path = ORT_TSTR("constant_of_shape.onnx");
  std::filesystem::remove(serialized_model_path);

  {
    auto test_graph = TestGraph::Load(original_model_path);
    ASSERT_NE(test_graph, nullptr) << "Failed to load test model";

    // Serialize OrtGraph to GraphProto. Save initializers to external file.
    std::string ext_ini_file_path = "constant_of_shape_serialized.bin";
    std::filesystem::remove(ext_ini_file_path);
    std::ofstream ext_ini_ofs(ext_ini_file_path, std::ios::binary);
    auto handle_initializer_data = [&ext_ini_ofs, &ext_ini_file_path](const OrtValueInfo* value_info,
                                                                      const void* data, size_t bytes,
                                                                      bool& is_external, std::string& location,
                                                                      int64_t& offset) -> Ort::Status {
      // OrtValueInfo* could be used to query initializer's name, type, shape,
      // node consumers, etc.
      static_cast<void>(value_info);

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
    ASSERT_CXX_ORTSTATUS_OK(OrtEpUtils::OrtGraphToProto(test_graph->GetOrtGraph(), model_proto,
                                                        handle_initializer_data));

    std::ofstream ofs(serialized_model_path, std::ios::binary);
    model_proto.SerializeToOstream(&ofs);
    ofs.flush();

    ASSERT_TRUE(std::filesystem::exists(serialized_model_path));
    ASSERT_TRUE(std::filesystem::exists(ext_ini_file_path));
  }

  // Compare output of the original and serialized models. Should be identical.
  std::vector<float> output_original;
  std::vector<float> output_serialized;

  RunConstantOfShapeModel(original_model_path, output_original);
  RunConstantOfShapeModel(serialized_model_path, output_serialized);

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
    ASSERT_CXX_ORTSTATUS_OK(OrtEpUtils::OrtGraphToProto(test_graph->GetOrtGraph(), model_proto));

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
                                      const ONNX_NAMESPACE::TensorProto* tensor_proto,
                                      const GraphViewer& graph_viewer) {
  Ort::ConstValueInfo vi(api_value_info);
  std::string api_initializer_name = vi.GetName();

  // Check external initializer info (if any).
  Ort::ExternalInitializerInfo api_ext_info{nullptr};
  auto external_status = vi.GetExternalInitializerInfo(api_ext_info);

  std::unique_ptr<ExternalDataInfo> ext_info = nullptr;
  bool has_ext_info = graph_viewer.GetGraph().GetExternalInitializerInfo(api_initializer_name, ext_info, true);

  if (has_ext_info) {
    ASSERT_NE(api_ext_info, nullptr);
    const std::basic_string<ORTCHAR_T> api_ext_file_path = api_ext_info.GetFilePath();
    int64_t api_ext_file_offset = api_ext_info.GetFileOffset();
    size_t api_ext_byte_size = api_ext_info.GetByteSize();

    ASSERT_EQ(PathString(api_ext_file_path), ext_info->GetRelPath());
    ASSERT_EQ(api_ext_file_offset, static_cast<int64_t>(ext_info->GetOffset()));
    ASSERT_EQ(api_ext_byte_size, ext_info->GetLength());
  } else {
    ASSERT_EQ(api_ext_info, nullptr);
    ASSERT_FALSE(utils::HasExternalDataInFile(*tensor_proto));
  }

  Ort::ConstValue api_initializer_value;
  ASSERT_ORTSTATUS_OK(vi.GetInitializer(api_initializer_value));
  ASSERT_NE(api_initializer_value, nullptr);

  // Check initializer type.
  const ONNX_NAMESPACE::TypeProto type_proto = utils::TypeProtoFromTensorProto(*tensor_proto);
  auto type_info = OrtTypeInfo::FromTypeProto(type_proto);

  Ort::ConstTypeInfo api_type_info = vi.TypeInfo();
  CheckTypeInfo(api_type_info, type_info.get());
}

static void CheckInitializerValueInfosCApi(gsl::span<Ort::ConstValueInfo> initializer_value_infos,
                                           const InitializedTensorSet& initializer_tensor_protos,
                                           const GraphViewer& graph_viewer) {
  for (size_t i = 0; i < initializer_value_infos.size(); i++) {
    Ort::ConstValueInfo vi(initializer_value_infos[i]);
    std::string api_initializer_name = vi.GetName();

    auto tensor_proto_iter = initializer_tensor_protos.find(api_initializer_name);
    ASSERT_NE(tensor_proto_iter, initializer_tensor_protos.end());

    const ONNX_NAMESPACE::TensorProto* tensor_proto = tensor_proto_iter->second;
    ASSERT_NE(tensor_proto, nullptr);
    CheckInitializerValueInfo(vi, tensor_proto, graph_viewer);
  }
}

// Checks that the OrtValueInfos obtained from the public C API are "equivalent" to the NodeArgs
// in the original graph.
static void CheckValueInfosCApi(const GraphViewer& graph_viewer, gsl::span<Ort::ConstValueInfo> value_infos,
                                gsl::span<const NodeArg* const> node_args) {
  ASSERT_EQ(value_infos.size(), node_args.size());
  const auto& graph_viewer_inputs = graph_viewer.GetInputsIncludingInitializers();
  const auto& graph_viewer_outputs = graph_viewer.GetOutputs();

  for (size_t i = 0; i < value_infos.size(); i++) {
    const NodeArg* node_arg = node_args[i];
    Ort::ConstValueInfo vi(value_infos[i]);

    if (node_arg->Exists()) {
      const auto& value_name = node_arg->Name();
      std::string api_name = vi.GetName();
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

      bool api_is_req_graph_input = vi.IsRequiredGraphInput();
      bool api_is_opt_graph_input = vi.IsOptionalGraphInput();
      ASSERT_EQ(api_is_req_graph_input, is_graph_input && (initializer == nullptr));
      ASSERT_EQ(api_is_opt_graph_input, can_override_initializer && (initializer != nullptr) && !is_const_initializer);

      bool api_is_graph_output = vi.IsGraphOutput();
      ASSERT_EQ(api_is_graph_output, is_graph_output);

      bool is_outer_scope = graph_viewer.GetGraph().IsOuterScopeValue(node_arg->Name());
      bool api_is_outer_scope = vi.IsFromOuterScope();
      ASSERT_EQ(api_is_outer_scope, is_outer_scope);

      bool api_is_const_initializer = vi.IsConstantInitializer();
      ASSERT_EQ(api_is_const_initializer, is_const_initializer);

      if (is_const_initializer || api_is_opt_graph_input) {
        CheckInitializerValueInfo(vi, initializer, graph_viewer);
      } else {
        auto node_arg_type_info = OrtTypeInfo::FromTypeProto(*node_arg->TypeAsProto());
        Ort::ConstTypeInfo api_type_info = vi.TypeInfo();
        CheckTypeInfo(api_type_info, node_arg_type_info.get());
      }

      CheckValueInfoProducer(graph_viewer, vi, node_arg);
      CheckValueInfoConsumers(graph_viewer, vi, node_arg);
    } else {
      ASSERT_EQ(vi, nullptr);  // A missing optional input has a null OrtValueInfo.
    }
  }
}

// Checks the Graph_GetSubgraph C API
static void Check_Graph_GetSubgraph(const OrtGraph& api_graph) {
  Ort::ConstGraph ort_graph{&api_graph};
  // Get all the nodes
  std::vector<Ort::ConstNode> nodes = ort_graph.GetNodes();

  // Select a half of nodes to create a OrtGraph
  size_t num_selected_nodes = std::max((nodes.size() >> 1), (size_t)1);
  std::vector<Ort::ConstNode> selected_nodes(num_selected_nodes);

  for (size_t i = 0; i < num_selected_nodes; i++) {
    selected_nodes[i] = nodes[i];
  }

  Ort::Graph sub_graph = ort_graph.GetGraphView(selected_nodes);

  // Convert OrtGraph/GraphViewer to ModelProto and dump it to disk.
  // If the GraphViewer associated with the OrtGraph somehow is incorrect, GraphViewerToProto() will throw.
  const GraphViewer& sub_graph_viewer = EpGraph::ToInternal(sub_graph)->GetGraphViewer();
  std::unique_ptr<Model> model = std::make_unique<Model>(sub_graph_viewer.Name(), true, sub_graph_viewer.GetGraph().GetLogger());
  auto model_proto = std::make_unique<ONNX_NAMESPACE::ModelProto>(model->ToProto());
  GraphViewerToProto(sub_graph_viewer, *model_proto->mutable_graph(), true, true, static_cast<ExecutionOrder>(1));
  model_proto->set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  auto graph_name = ort_graph.GetName();
  std::string name = graph_name;
  name += "_half.onnx";

  // Dump the graph for debugging
  // std::fstream dump(name, std::ios::out | std::ios::trunc | std::ios::binary);
  // model_proto->SerializeToOstream(&dump);
}

// Checks that the contents of the original GraphViewer matches the contents of the OrtGraph.
// Uses the public C APIs to traverse the OrtGraph.
static void CheckGraphCApi(const GraphViewer& graph_viewer, const OrtGraph& api_graph) {
  auto ort_cxx_graph = Ort::ConstGraph(&api_graph);
  // Check the path to model.
  const std::filesystem::path& model_path = graph_viewer.ModelPath();
  const auto api_model_path = ort_cxx_graph.GetModelPath();
  ASSERT_EQ(PathString(api_model_path), PathString(model_path.c_str()));
  // Check the model metadata
  Ort::AllocatorWithDefaultOptions default_allocator;
  auto ort_cxx_model_metadat = ort_cxx_graph.GetModelMetadata();
  auto& model = graph_viewer.GetGraph().GetModel();
  ASSERT_EQ(std::strcmp(ort_cxx_model_metadat.GetProducerNameAllocated(default_allocator).get(), model.ProducerName().c_str()), 0);
  ASSERT_EQ(std::strcmp(ort_cxx_model_metadat.GetGraphNameAllocated(default_allocator).get(), model.MainGraph().Name().c_str()), 0);
  ASSERT_EQ(std::strcmp(ort_cxx_model_metadat.GetDomainAllocated(default_allocator).get(), model.Domain().c_str()), 0);
  ASSERT_EQ(std::strcmp(ort_cxx_model_metadat.GetDescriptionAllocated(default_allocator).get(), model.DocString().c_str()), 0);
  ASSERT_EQ(std::strcmp(ort_cxx_model_metadat.GetGraphDescriptionAllocated(default_allocator).get(), model.GraphDocString().c_str()), 0);
  ASSERT_EQ(ort_cxx_model_metadat.GetVersion(), model.ModelVersion());
  auto model_meta_data = model.MetaData();
  for (auto& [k, v] : model_meta_data) {
    ASSERT_EQ(std::strcmp(ort_cxx_model_metadat.LookupCustomMetadataMapAllocated(k.c_str(), default_allocator).get(), v.c_str()), 0)
        << " key=" << k << "; value=" << v;
  }
  // Check graph inputs.
  const auto& graph_input_node_args = graph_viewer.GetInputsIncludingInitializers();

  std::vector<Ort::ConstValueInfo> api_graph_inputs = ort_cxx_graph.GetInputs();
  ASSERT_EQ(api_graph_inputs.size(), graph_input_node_args.size());

  CheckValueInfosCApi(graph_viewer, api_graph_inputs, graph_input_node_args);

  // Check graph outputs.
  const auto& graph_output_node_args = graph_viewer.GetOutputs();

  std::vector<Ort::ConstValueInfo> api_graph_outputs = ort_cxx_graph.GetOutputs();
  ASSERT_EQ(api_graph_outputs.size(), graph_output_node_args.size());

  CheckValueInfosCApi(graph_viewer, api_graph_outputs, graph_output_node_args);

  // Check graph initializers
  const auto& graph_initializers = graph_viewer.GetAllInitializedTensors();

  std::vector<Ort::ConstValueInfo> api_initializers = ort_cxx_graph.GetInitializers();
  ASSERT_EQ(api_initializers.size(), graph_initializers.size());
  CheckInitializerValueInfosCApi(api_initializers, graph_initializers, graph_viewer);

  // Check if it has a parent node.
  const Node* parent_node = graph_viewer.ParentNode();
  const bool has_parent_node = parent_node != nullptr;
  Ort::ConstNode api_parent_node = ort_cxx_graph.GetParentNode();
  const bool api_has_parent_node = api_parent_node != nullptr;
  ASSERT_EQ(api_has_parent_node, has_parent_node);

  if (has_parent_node) {
    CheckNode(parent_node, api_parent_node);
  }

  // Check all nodes.
  std::vector<Ort::ConstNode> api_nodes = ort_cxx_graph.GetNodes();
  ASSERT_EQ(api_nodes.size(), graph_viewer.NumberOfNodes());

  std::vector<NodeIndex> node_indices = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::DEFAULT);
  for (size_t node_idx = 0; node_idx < api_nodes.size(); node_idx++) {
    // Check basic node properties.
    const Node* node = graph_viewer.GetNode(node_indices[node_idx]);
    Ort::ConstNode api_node = api_nodes[node_idx];
    CheckNode(node, api_node);

    const int api_since_version = api_node.GetSinceVersion();
    ASSERT_EQ(api_since_version, node->SinceVersion());

    // Check node inputs
    const auto input_node_args = node->InputDefs();

    std::vector<Ort::ConstValueInfo> api_node_inputs = api_node.GetInputs();
    ASSERT_EQ(api_node_inputs.size(), input_node_args.size());
    CheckValueInfosCApi(graph_viewer, api_node_inputs, input_node_args);

    // Check node outputs
    const auto output_node_args = node->OutputDefs();
    std::vector<Ort::ConstValueInfo> api_node_outputs = api_node.GetOutputs();
    ASSERT_EQ(api_node_outputs.size(), output_node_args.size());
    CheckValueInfosCApi(graph_viewer, api_node_outputs, output_node_args);

    // Check node attributes
    const auto& node_attrs = node->GetAttributes();

    if (!node_attrs.empty()) {
      std::vector<Ort::ConstOpAttr> api_node_attributes = api_node.GetAttributes();

      size_t attr_idx = 0;
      for (const auto& node_attr : node_attrs) {
        auto api_node_attr = api_node_attributes[attr_idx];
        ASSERT_NE(api_node_attr, nullptr);

        auto status = api_node.GetAttributeByName(node_attr.first, api_node_attr);
        ASSERT_TRUE(status.IsOK());
        ASSERT_NE(api_node_attr, nullptr);

        auto api_node_attr_name = api_node_attr.GetName();
        ASSERT_EQ(api_node_attr_name, node_attr.first);

        // XXX: Investigate why not
        // It's possible that the type is defined in ONNX::AttributeProto_AttributeType but not in OrtOpAttrType, since the two are not in a 1:1 mapping.
        // In such cases, OpAttr_GetType will return a non-null status, and we simply skip the check here.
        // TODO: Once we add support for ORT_OP_ATTR_TENSOR, we should be able to just fail if OpAttr_GetType
        // returns an error.
        OrtOpAttrType api_node_attr_type = api_node_attr.GetType();

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
          case ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPH: {
            ASSERT_EQ(api_node_attr_type, OrtOpAttrType::ORT_OP_ATTR_GRAPH);
            break;
          }
          case ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR: {
            ASSERT_EQ(api_node_attr_type, OrtOpAttrType::ORT_OP_ATTR_TENSOR);
            break;
          }
          default:
            // The unsupported type should be skipped by 'continue' above. It's unexpected so we force test to fail.
            FAIL() << "The attribute type is not in AttributeProto_AttributeType and this case shouldn't be hit.";
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

      std::vector<Ort::ConstValueInfo> api_node_implicit_inputs = api_node.GetImplicitInputs();
      ASSERT_EQ(api_node_implicit_inputs.size(), implicit_input_node_args.size());
      CheckValueInfosCApi(graph_viewer, api_node_implicit_inputs, implicit_input_node_args);

      // Recursively check subgraphs.
      std::vector<Ort::AttrNameSubgraph> api_node_subgraphs = api_node.GetSubgraphs();
      ASSERT_EQ(api_node_subgraphs.size(), node_subgraphs_map.size());

      for (const auto& name_subgraph : api_node_subgraphs) {
        auto hit = node_subgraphs_map.find(name_subgraph.attr_name);
        ASSERT_NE(node_subgraphs_map.end(), hit);
        auto subgraph_viewer = std::make_unique<GraphViewer>(*hit->second);
        CheckGraphCApi(*subgraph_viewer, *name_subgraph.sub_graph);
      }
    }
  }

  // Check creating an OrtGraph from a subset of nodes in an OrtGraph
  Check_Graph_GetSubgraph(api_graph);
}

}  // namespace test
}  // namespace onnxruntime
