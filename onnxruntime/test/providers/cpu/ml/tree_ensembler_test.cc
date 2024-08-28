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

static ONNX_NAMESPACE::TensorProto make_tensor(std::vector<float> array, std::string name) {
  ONNX_NAMESPACE::TensorProto array_as_tensor;
  array_as_tensor.set_name(name);
  array_as_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  array_as_tensor.add_dims(array.size());
  for (auto v : array) {
    array_as_tensor.add_float_data(v);
  }

  return array_as_tensor;
}

template <typename T>
void _multiply_update_array(std::vector<T>& data, int n, T inc = 0) {
  std::vector<T> copy = data;
  data.resize(copy.size() * n);
  T cst = 0;
  for (int i = 0; i < n; ++i) {
    for (size_t j = 0; j < copy.size(); ++j) {
      data[j + i * copy.size()] = copy[j] + cst;
    }
    cst += inc;
  }
}

template <typename T>
void _multiply_update_childnode(std::vector<T>& childnodes, std::vector<T>& childleafs, std::vector<T>& otherchildleafs, int n) {
  int64_t leafs_cnt = 0;
  int64_t nodes_cnt = childnodes.size();
  for (auto& childleaf : childleafs) {
    if (childleaf) {
      leafs_cnt++;
    }
  }
  for (auto& childleaf : otherchildleafs) {
    if (childleaf) {
      leafs_cnt++;
    }
  }

  std::vector<T> copy = childnodes;
  childnodes.resize(copy.size() * n);
  T leafs_cst = 0;
  T nodes_cst = 0;
  for (int i = 0; i < n; ++i) {
    for (size_t j = 0; j < copy.size(); ++j) {
      T curr_inc = childleafs[j] ? leafs_cst : nodes_cst;
      childnodes[j + i * copy.size()] = copy[j] + curr_inc;
    }

    leafs_cst += leafs_cnt;
    nodes_cst += nodes_cnt;
  }
}

template <typename T>
void _multiply_arrays_values(std::vector<T>& data, int64_t val) {
  for (auto& curr : data) {
    curr *= val;
  }
}

template <typename T>
void GenTreeAndRunTest(const std::vector<T>& X, const std::vector<T>& Y, const int64_t& aggregate_function, int n_trees = 1) {
  OpTester test("TreeEnsemble", 5, onnxruntime::kMLDomain);
  int64_t n_targets = 2;

  int64_t post_transform = 0;
  std::vector<int64_t> tree_roots = {0};
  std::vector<int64_t> nodes_featureids = {0, 0, 0};
  std::vector<uint8_t> nodes_modes = {0, 0, 0};
  std::vector<T> nodes_splits = {3.14, 1.2, 4.2};
  std::vector<int64_t> nodes_truenodeids = {1, 0, 1};
  std::vector<int64_t> nodes_trueleafs = {0, 1, 1};
  std::vector<int64_t> nodes_falsenodeids = {2, 2, 3};
  std::vector<int64_t> nodes_falseleafs = {0, 1, 1};

  std::vector<int64_t> leaf_targetids = {0, 1, 0, 1};
  std::vector<T> leaf_weights = {5.23, 12.12, -12.23, 7.21};

  if (n_trees > 1) {
    // Multiplies the number of trees to test the parallelization by trees.
    _multiply_update_array(tree_roots, n_trees, (int64_t)nodes_truenodeids.size());
    _multiply_update_array(nodes_featureids, n_trees);
    _multiply_update_childnode(nodes_truenodeids, nodes_trueleafs, nodes_falseleafs, n_trees);
    _multiply_update_childnode(nodes_falsenodeids, nodes_falseleafs, nodes_trueleafs, n_trees);
    _multiply_update_array(nodes_trueleafs, n_trees);
    _multiply_update_array(nodes_falseleafs, n_trees);
    _multiply_update_array(leaf_targetids, n_trees);
    _multiply_update_array(nodes_modes, n_trees);
    _multiply_update_array(nodes_splits, n_trees);
    _multiply_update_array(leaf_weights, n_trees);
  }

  auto nodes_modes_as_tensor = make_tensor(nodes_modes, "nodes_modes");
  auto nodes_splits_as_tensor = make_tensor(nodes_splits, "nodes_splits");
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
  test.AddInput<T>("X", {3, 2}, X);
  test.AddOutput<T>("Y", {3, 2}, Y);
  test.Run();
}

template <typename T>
void GenTreeAndRunTestWithSetMembership(const std::vector<T>& X, const std::vector<T>& Y, const int64_t& aggregate_function, int n_trees = 1) {
  OpTester test("TreeEnsemble", 5, onnxruntime::kMLDomain);
  int64_t n_targets = 4;

  int64_t post_transform = 0;
  std::vector<int64_t> tree_roots = {0};
  std::vector<int64_t> nodes_featureids = {0, 0, 0};
  std::vector<int64_t> nodes_truenodeids = {1, 0, 1};
  std::vector<int64_t> nodes_trueleafs = {0, 1, 1};
  std::vector<int64_t> nodes_falsenodeids = {2, 2, 3};
  std::vector<int64_t> nodes_falseleafs = {1, 0, 1};
  std::vector<int64_t> leaf_targetids = {0, 1, 2, 3};

  std::vector<uint8_t> nodes_modes = {0, 6, 6};
  std::vector<T> nodes_splits = {11, 232344.0, NAN};
  std::vector<T> membership_values = {1.2, 3.7, 8, 9, NAN, 12, 7, NAN};
  std::vector<T> leaf_weights = {1, 10, 1000, 100};

  if (n_trees > 1) {
    // Multiplies the number of trees to test the parallelization by trees.
    _multiply_update_array(tree_roots, n_trees, (int64_t)nodes_truenodeids.size());
    _multiply_update_array(nodes_featureids, n_trees);
    _multiply_update_childnode(nodes_truenodeids, nodes_trueleafs, nodes_falseleafs, n_trees);
    _multiply_update_childnode(nodes_falsenodeids, nodes_falseleafs, nodes_trueleafs, n_trees);
    _multiply_update_array(nodes_trueleafs, n_trees);
    _multiply_update_array(nodes_falseleafs, n_trees);
    _multiply_update_array(leaf_targetids, n_trees);
    _multiply_update_array(nodes_modes, n_trees);
    _multiply_update_array(nodes_splits, n_trees);
    _multiply_update_array(membership_values, n_trees);
    _multiply_update_array(leaf_weights, n_trees);
  }

  auto nodes_modes_as_tensor = make_tensor(nodes_modes, "nodes_modes");
  auto nodes_splits_as_tensor = make_tensor(nodes_splits, "nodes_splits");
  auto membership_values_as_tensor = make_tensor(membership_values, "membership_values");
  auto leaf_weights_as_tensor = make_tensor(leaf_weights, "leaf_weight");

  // add attributes
  test.AddAttribute("n_targets", n_targets);
  test.AddAttribute("aggregate_function", aggregate_function);
  test.AddAttribute("post_transform", post_transform);
  test.AddAttribute("tree_roots", tree_roots);
  test.AddAttribute("nodes_modes", nodes_modes_as_tensor);
  test.AddAttribute("nodes_featureids", nodes_featureids);
  test.AddAttribute("nodes_splits", nodes_splits_as_tensor);
  test.AddAttribute("membership_values", membership_values_as_tensor);
  test.AddAttribute("nodes_truenodeids", nodes_truenodeids);
  test.AddAttribute("nodes_trueleafs", nodes_trueleafs);
  test.AddAttribute("nodes_falsenodeids", nodes_falsenodeids);
  test.AddAttribute("nodes_falseleafs", nodes_falseleafs);
  test.AddAttribute("leaf_targetids", leaf_targetids);
  test.AddAttribute("leaf_weights", leaf_weights_as_tensor);

  // fill input data
  test.AddInput<T>("X", {6, 1}, X);
  test.AddOutput<T>("Y", {6, 4}, Y);
  test.Run();
}

TEST(MLOpTest, TreeEnsembleFloat) {
  std::vector<float> X = {1.2, 3.4, -0.12, 1.66, 4.14, 1.77};
  std::vector<float> Y = {5.23, 0, 5.23, 0, 0, 12.12};
  GenTreeAndRunTest<float>(X, Y, 1, 1);

  Y = {15.69, 0, 15.69, 0, 0, 36.36};
  GenTreeAndRunTest<float>(X, Y, 1, 3);
}

TEST(MLOpTest, TreeEnsembleDouble) {
  std::vector<double> X = {1.2, 3.4, -0.12, 1.66, 4.14, 1.77};
  std::vector<double> Y = {5.23, 0, 5.23, 0, 0, 12.12};
  GenTreeAndRunTest<double>(X, Y, 1, 1);

  _multiply_arrays_values(Y, 3);
  GenTreeAndRunTest<double>(X, Y, 1, 3);
}

TEST(MLOpTest, TreeEnsembleSetMembership) {
  std::vector<double> X = {1.2, 3.4, -0.12, NAN, 12.0, 7.0};
  std::vector<double> Y = {
      1, 0, 0, 0,
      0, 0, 0, 100,
      0, 0, 0, 100,
      0, 0, 1000, 0,
      0, 0, 1000, 0,
      0, 10, 0, 0};
  GenTreeAndRunTestWithSetMembership<double>(X, Y, 1, 1);

  _multiply_arrays_values(Y, 5);
  GenTreeAndRunTestWithSetMembership<double>(X, Y, 1, 5);
}

TEST(MLOpTest, TreeEnsembleLeafOnly) {
  OpTester test("TreeEnsemble", 5, onnxruntime::kMLDomain);
  int64_t n_targets = 1;

  int64_t aggregate_function = 1;
  int64_t post_transform = 0;
  std::vector<int64_t> tree_roots = {0};
  std::vector<uint8_t> nodes_modes = {0};
  std::vector<int64_t> nodes_featureids = {0};
  std::vector<double> nodes_splits = {0};
  std::vector<int64_t> nodes_truenodeids = {0};
  std::vector<int64_t> nodes_trueleafs = {1};
  std::vector<int64_t> nodes_falsenodeids = {0};
  std::vector<int64_t> nodes_falseleafs = {1};

  std::vector<int64_t> leaf_targetids = {0};
  std::vector<double> leaf_weights = {6.23};

  auto nodes_modes_as_tensor = make_tensor(nodes_modes, "nodes_modes");
  auto nodes_splits_as_tensor = make_tensor(nodes_splits, "nodes_splits");
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
  std::vector<double> X = {1, 4};
  std::vector<double> Y = {6.23, 6.23};

  test.AddInput<double>("X", {2, 1}, X);
  test.AddOutput<double>("Y", {2, 1}, Y);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
