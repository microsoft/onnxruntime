// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/graph/contrib_ops/quantization_defs.h"
#include "core/graph/contrib_ops/contrib_defs.h"
#include "core/graph/contrib_ops/ms_opset.h"

namespace onnxruntime {
namespace test {

TEST(QuantizeBFPTest, CreateQuantizeGraph) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[onnxruntime::kMSDomain] = 1;
  // Generate the input & output def lists
  std::vector<ONNX_NAMESPACE::FunctionProto> model_specific_functions;
  auto p_model = std::make_unique<Model>(
      "test", true, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
      model_specific_functions, DefaultLoggingManager().DefaultLogger(), ModelOptions(true, true));
  onnxruntime::Graph& graph = p_model->MainGraph();

  ONNX_NAMESPACE::TypeProto x_float;
  x_float.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  x_float.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(128);
  x_float.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(16);
  std::vector<onnxruntime::NodeArg*> input_defs;
  auto& input_arg_x = graph.GetOrCreateNodeArg("x", &x_float);
  input_defs.push_back(&input_arg_x);

  NodeAttributes attributes;
  ONNX_NAMESPACE::AttributeProto bfp_type;
  bfp_type.set_name("bfp_type");
  bfp_type.set_i(static_cast<int64_t>(onnxruntime::contrib::BFPType::BFP_1_8_8_16));
  bfp_type.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
  attributes["bfp_type"] = bfp_type;
  ONNX_NAMESPACE::AttributeProto bounding_box_dims;
  bounding_box_dims.set_name("bounding_box_dims");
  bounding_box_dims.add_ints(1);  // bounding box is over dimension 1
  bounding_box_dims.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  attributes["bounding_box_dims"] = bounding_box_dims;

  std::vector<onnxruntime::NodeArg*> output_defs;
  ONNX_NAMESPACE::TypeProto y_byte;
  y_byte.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  auto& output_arg_y = graph.GetOrCreateNodeArg("y", &y_byte);
  output_defs.push_back(&output_arg_y);

  ONNX_NAMESPACE::TypeProto tensor_int;
  tensor_int.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  auto& output_arg_shape = graph.GetOrCreateNodeArg("shape", &tensor_int);
  auto& output_arg_strides = graph.GetOrCreateNodeArg("strides", &tensor_int);
  output_defs.push_back(&output_arg_shape);
  output_defs.push_back(&output_arg_strides);

  // Create a simple model
  graph.AddNode("node1", "QuantizeBFP", "quantizes float tensor to BFP", input_defs, output_defs, &attributes,
                onnxruntime::kMSDomain);
  Status status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
}

TEST(DequantizeBFPTest, CreateDequantizeGraph) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[onnxruntime::kMSDomain] = 1;
  // Generate the input & output def lists
  std::vector<ONNX_NAMESPACE::FunctionProto> model_specific_functions;
  auto p_model = std::make_unique<Model>(
      "test", true, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), domain_to_version,
      model_specific_functions, DefaultLoggingManager().DefaultLogger(), ModelOptions(true, true));
  onnxruntime::Graph& graph = p_model->MainGraph();

  ONNX_NAMESPACE::TypeProto x_byte;
  x_byte.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  x_byte.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(128);  // an arbitrary byte size
  std::vector<onnxruntime::NodeArg*> input_defs;
  auto& input_arg_x = graph.GetOrCreateNodeArg("x", &x_byte);
  input_defs.push_back(&input_arg_x);

  ONNX_NAMESPACE::TypeProto tensor_int;
  tensor_int.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  tensor_int.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);
  auto& input_arg_shape = graph.GetOrCreateNodeArg("shape", &tensor_int);
  auto& input_arg_strides = graph.GetOrCreateNodeArg("strides", &tensor_int);
  input_defs.push_back(&input_arg_shape);
  input_defs.push_back(&input_arg_strides);

  NodeAttributes attributes;
  ONNX_NAMESPACE::AttributeProto bfp_type;
  bfp_type.set_name("bfp_type");
  bfp_type.set_i(static_cast<int64_t>(onnxruntime::contrib::BFPType::BFP_1_8_8_16));
  bfp_type.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
  attributes["bfp_type"] = bfp_type;
  ONNX_NAMESPACE::AttributeProto bounding_box_dims;
  bounding_box_dims.set_name("bounding_box_dims");
  bounding_box_dims.add_ints(1);  // bounding box is over dimension 1
  bounding_box_dims.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
  attributes["bounding_box_dims"] = bounding_box_dims;
  ONNX_NAMESPACE::AttributeProto dtype;
  dtype.set_name("dtype");
  dtype.set_i(static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
  dtype.set_type(ONNX_NAMESPACE::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
  attributes["dtype"] = dtype;

  std::vector<onnxruntime::NodeArg*> output_defs;
  ONNX_NAMESPACE::TypeProto y_float;
  y_float.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  auto& output_arg_y = graph.GetOrCreateNodeArg("y", &y_float);
  output_defs.push_back(&output_arg_y);

  // Create a simple model
  graph.AddNode("node1", "DequantizeBFP", "dequantizes BFP tensor to float", input_defs, output_defs, &attributes,
                onnxruntime::kMSDomain);
  Status status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
}

TEST(QuantizeBFPTest, CheckFitsInsideBoundingBoxTest) {
  constexpr auto bfp_type = onnxruntime::contrib::BFPType::BFP_0_16_8_128;
  {
    ONNX_NAMESPACE::TensorShapeProto input_shape;
    google::protobuf::RepeatedField<int64_t> bounding_box_dims;
    // should succeed since input_shape is empty
    onnxruntime::contrib::CheckFitsInsideBoundingBox(input_shape, bounding_box_dims, bfp_type);
  }

  {
    ONNX_NAMESPACE::TensorShapeProto input_shape;
    input_shape.add_dim()->set_dim_value(2);
    google::protobuf::RepeatedField<int64_t> bounding_box_dims;
    // should succeed since bounding_box_dims is empty
    onnxruntime::contrib::CheckFitsInsideBoundingBox(input_shape, bounding_box_dims, bfp_type);
  }

  {
    ONNX_NAMESPACE::TensorShapeProto input_shape;
    input_shape.add_dim()->set_dim_value(2);
    google::protobuf::RepeatedField<int64_t> bounding_box_dims;
    bounding_box_dims.Add(0);
    // should succeed since 2 is less than the bounding box size of bfp_type
    onnxruntime::contrib::CheckFitsInsideBoundingBox(input_shape, bounding_box_dims, bfp_type);
  }

  {
    ONNX_NAMESPACE::TensorShapeProto input_shape;
    auto dim_too_large = onnxruntime::contrib::get_bounding_box_size(bfp_type) + 1;
    input_shape.add_dim()->set_dim_value(static_cast<int64_t>(dim_too_large));
    google::protobuf::RepeatedField<int64_t> bounding_box_dims;
    bounding_box_dims.Add(0);
    EXPECT_THROW(onnxruntime::contrib::CheckFitsInsideBoundingBox(input_shape, bounding_box_dims, bfp_type),
                 ONNX_NAMESPACE::InferenceError);
  }

  {
    ONNX_NAMESPACE::TensorShapeProto input_shape;
    google::protobuf::RepeatedField<int64_t> bounding_box_dims;
    // should succeed since the check is skipped for custom BFP types
    onnxruntime::contrib::CheckFitsInsideBoundingBox(input_shape, bounding_box_dims,
                                                     onnxruntime::contrib::BFPType::Custom_BFP_0);
  }
}
}  // namespace test
}  // namespace onnxruntime
