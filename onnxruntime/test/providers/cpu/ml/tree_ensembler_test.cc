// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static ONNX_NAMESPACE::TensorProto make_tensor(std::vector<double> array, std::string name) {
  ONNX_NAMESPACE::TensorProto array_as_tensor;
  array_as_tensor.set_name(name);
  array_as_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
  array_as_tensor.add_dims(array.size());
  for (auto v : array) {
    array_as_tensor.add_double_data(v);
  }

  return array_as_tensor;
}

static ONNX_NAMESPACE::TensorProto make_tensor(std::vector<uint8_t> array, std::string name) {
  ONNX_NAMESPACE::TensorProto array_as_tensor;
  array_as_tensor.set_name(name);
  array_as_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  array_as_tensor.add_dims(array.size());
  for (const auto v : array) {
    array_as_tensor.add_int32_data(v);
  }

  return array_as_tensor;
}

// static ONNX_NAMESPACE::TensorProto make_tensor(std::vector<float> array, std::string name) {
//   ONNX_NAMESPACE::TensorProto array_as_tensor;
//   array_as_tensor.set_name(name);
//   array_as_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
//   array_as_tensor.add_dims(array.size());
//   for (auto v : array) {
//     array_as_tensor.add_float_data(v);
//   }

//   return array_as_tensor;
// }

TEST(MLOpTest, TreeEnsembleOneTree) {
  OpTester test("TreeEnsemble", 5, onnxruntime::kMLDomain);
  int64_t n_targets = 2;

  int64_t aggregate_function = 1;
  int64_t post_transform = 0;
  std::vector<int64_t> tree_roots = {0};
  std::vector<int64_t> nodes_featureids = {0, 0, 0};
  std::vector<int64_t> nodes_truenodeids = {1, 0, 1};
  std::vector<int64_t> nodes_trueleafs = {0, 1, 1};
  std::vector<int64_t> nodes_falsenodeids = {2, 2, 3};
  std::vector<int64_t> nodes_falseleafs = {0, 1, 1};
  std::vector<int64_t> leaf_targetids = {0, 1, 0, 1};

  std::vector<uint8_t> nodes_modes = {0, 0, 0};
  auto nodes_modes_as_tensor = make_tensor(nodes_modes, "nodes_modes");

  std::vector<double> nodes_splits = {3.14, 1.2, 4.2};
  auto nodes_splits_as_tensor = make_tensor(nodes_splits, "nodes_splits");

  std::vector<double> leaf_weights = {5.23, 12.12, -12.23, 7.21};
  auto leaf_weights_as_tensor = make_tensor(leaf_weights, "leaf_weight");

  // add attributes
  test.AddAttribute("n_targets", n_targets);
  test.AddAttribute("aggregate_function", aggregate_function);
  test.AddAttribute("post_transform", post_transform);
  test.AddAttribute("tree_roots", tree_roots);
  test.AddAttribute("nodes_modes", nodes_modes_as_tensor);
  test.AddAttribute("nodes_featureids", nodes_featureids);
  test.AddAttribute("nodes_splits", nodes_splits_as_tensor);
  test.AddAttribute("nodes_truenodeids", nodes_truenodeids);
  test.AddAttribute("nodes_trueleafs", nodes_trueleafs);
  test.AddAttribute("nodes_falsenodeids", nodes_falsenodeids);
  test.AddAttribute("nodes_falseleafs", nodes_falseleafs);
  test.AddAttribute("leaf_targetids", leaf_targetids);
  test.AddAttribute("leaf_weights", leaf_weights_as_tensor);

  // fill input data
  std::vector<double> X = {1.2, 3.4, -0.12, 1.66, 4.14, 1.77};
  std::vector<double> Y = {5.23, 0, 5.23, 0, 0, 12.12};
  test.AddInput<double>("X", {3, 2}, X);
  test.AddOutput<double>("Y", {3, 2}, Y);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
