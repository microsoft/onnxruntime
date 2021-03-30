// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/provider_test_utils.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/checker.h"
#include "test/providers/cpu/tensor/shape_inference_test_helper.h"

namespace onnxruntime {
namespace test {

TEST(ShapeInferenceTests, embedding_float) {
  std::vector<int64_t> shape = {2, 4};
  ONNX_NAMESPACE::ValueInfoProto input;
  CreateValueInfo(input, "indices", ONNX_NAMESPACE::TensorProto_DataType_INT64, shape);
  std::vector<int64_t> shape_w = {10, 3};
  ONNX_NAMESPACE::ValueInfoProto weight;
  CreateValueInfo(weight, "weight", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape_w);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {weight, input};

  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  std::vector<int64_t> shape_y = {2, 4, 3};
  ONNX_NAMESPACE::ValueInfoProto output;
  CreateValueInfo(output, "Y", ONNX_NAMESPACE::TensorProto_DataType_FLOAT, shape_y);

  TestShapeInference("TorchEmbedding", kMSDomain, 1, 6, inputs, attributes, output);
}

TEST(ShapeInferenceTests, embedding_zero_dim_int) {
  std::vector<int64_t> shape = {0, 3};
  ONNX_NAMESPACE::ValueInfoProto input;
  CreateValueInfo(input, "X", ONNX_NAMESPACE::TensorProto_DataType_INT64, shape);
  std::vector<int64_t> shape_w = {10, 4};
  ONNX_NAMESPACE::ValueInfoProto weight;
  CreateValueInfo(weight, "W", ONNX_NAMESPACE::TensorProto_DataType_INT32, shape_w);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {weight, input};

  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  std::vector<int64_t> shape_y = {0, 3, 4};
  ONNX_NAMESPACE::ValueInfoProto output;
  CreateValueInfo(output, "Y", ONNX_NAMESPACE::TensorProto_DataType_INT32, shape_y);

  TestShapeInference("TorchEmbedding", kMSDomain, 1, 6, inputs, attributes, output);
}

TEST(ShapeInferenceTests, embedding_long) {
  std::vector<int64_t> shape = {2, 4};
  ONNX_NAMESPACE::ValueInfoProto input;
  CreateValueInfo(input, "X", ONNX_NAMESPACE::TensorProto_DataType_INT64, shape);
  std::vector<int64_t> shape_w = {10, 3};
  ONNX_NAMESPACE::ValueInfoProto weight;
  CreateValueInfo(weight, "W", ONNX_NAMESPACE::TensorProto_DataType_INT64, shape_w);
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {weight, input};

  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  std::vector<int64_t> shape_y = {2, 4, 3};
  ONNX_NAMESPACE::ValueInfoProto output;
  CreateValueInfo(output, "Y", ONNX_NAMESPACE::TensorProto_DataType_INT64, shape_y);

  TestShapeInference("TorchEmbedding", kMSDomain, 1, 6, inputs, attributes, output);
}

TEST(ShapeInferenceTests, embedding_with_padding) {
  std::vector<int64_t> shape = {2, 4};
  ONNX_NAMESPACE::ValueInfoProto input;
  CreateValueInfo(input, "indices", ONNX_NAMESPACE::TensorProto_DataType_INT64, shape);
  std::vector<int64_t> shape_w = {10, 3};
  ONNX_NAMESPACE::ValueInfoProto weight;
  CreateValueInfo(weight, "weight", ONNX_NAMESPACE::TensorProto_DataType_INT64, shape_w);
  ONNX_NAMESPACE::ValueInfoProto padding_idx;
  CreateValueInfo(padding_idx, "Pad", ONNX_NAMESPACE::TensorProto_DataType_INT64, {});
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {weight, input, padding_idx};

  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  std::vector<int64_t> shape_y = {2, 4, 3};
  ONNX_NAMESPACE::ValueInfoProto output;
  CreateValueInfo(output, "Y", ONNX_NAMESPACE::TensorProto_DataType_INT64, shape_y);

  TestShapeInference("TorchEmbedding", kMSDomain, 1, 6, inputs, attributes, output);
}

TEST(ShapeInferenceTests, embedding_with_scale_grad) {
  std::vector<int64_t> shape = {2, 8, 4};
  ONNX_NAMESPACE::ValueInfoProto input;
  CreateValueInfo(input, "indices", ONNX_NAMESPACE::TensorProto_DataType_INT64, shape);
  std::vector<int64_t> shape_w = {10, 3};
  ONNX_NAMESPACE::ValueInfoProto weight;
  CreateValueInfo(weight, "weight", ONNX_NAMESPACE::TensorProto_DataType_INT64, shape_w);
  ONNX_NAMESPACE::ValueInfoProto padding_idx;
  CreateValueInfo(padding_idx, "Pad", ONNX_NAMESPACE::TensorProto_DataType_INT64, {});
  ONNX_NAMESPACE::ValueInfoProto scale_grad;
  CreateValueInfo(scale_grad, "Scale", ONNX_NAMESPACE::TensorProto_DataType_BOOL, {});
  std::vector<ONNX_NAMESPACE::ValueInfoProto> inputs = {weight, input, padding_idx, scale_grad};

  std::vector<ONNX_NAMESPACE::AttributeProto> attributes = {};

  std::vector<int64_t> shape_y = {2, 8, 4, 3};
  ONNX_NAMESPACE::ValueInfoProto output;
  CreateValueInfo(output, "Y", ONNX_NAMESPACE::TensorProto_DataType_INT64, shape_y);

  TestShapeInference("TorchEmbedding", kMSDomain, 1, 6, inputs, attributes, output);
}

}  // namespace test
}  // namespace onnxruntime
