// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

void TreeEnsembleClassifierTest(int opsetml) {
  OpTester test("TreeEnsembleClassifier", opsetml, onnxruntime::kMLDomain);

  std::vector<int64_t> lefts = {1, -1, 3, -1, -1, 1, -1, 3, 4, -1, -1, -1, 1, 2, -1, 4, -1, -1, -1};
  std::vector<int64_t> rights = {2, -1, 4, -1, -1, 2, -1, 6, 5, -1, -1, -1, 6, 3, -1, 5, -1, -1, -1};
  std::vector<int64_t> treeids = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2};
  std::vector<int64_t> nodeids = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6};
  std::vector<int64_t> featureids = {2, -2, 0, -2, -2, 0, -2, 2, 1, -2, -2, -2, 0, 2, -2, 1, -2, -2, -2};
  std::vector<float> thresholds = {-172.f, -2.f, 2.5f, -2.f, -2.f, 1.5f, -2.f, -62.5f, 213.09999084f,
                                   -2.f, -2.f, -2.f, 27.5f, -172.f, -2.f, 8.10000038f, -2.f, -2.f, -2.f};
  std::vector<std::string> modes = {"BRANCH_LEQ", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_LEQ",
                                    "LEAF", "BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF",
                                    "BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF"};
  std::vector<int64_t> class_treeids = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  std::vector<int64_t> class_nodeids = {1, 3, 4, 1, 4, 5, 6, 2, 4, 5, 6};
  std::vector<int64_t> class_classids = {2, 0, 1, 0, 2, 3, 1, 2, 0, 1, 3};
  std::vector<float> class_weights = {1.f, 4.f, 1.f, 2.f, 1.f, 1.f, 2.f, 1.f, 1.f, 1.f, 3.f};
  std::vector<int64_t> classes = {0, 1, 2, 3};
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f,
                          11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<int64_t> results = {0, 1, 2, 2, 2, 2, 2, 3};
  std::vector<float> scores{7, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0,
                            0, 0, 3, 0, 0, 0, 2, 1, 0, 0, 3, 0, 0, 1, 0, 4};
  std::vector<float> probs = {};
  std::vector<float> log_probs = {};

  // define the context of the operator call
  constexpr int N = 8;
  test.AddAttribute("nodes_truenodeids", lefts);
  test.AddAttribute("nodes_falsenodeids", rights);
  test.AddAttribute("nodes_treeids", treeids);
  test.AddAttribute("nodes_nodeids", nodeids);
  test.AddAttribute("nodes_featureids", featureids);
  test.AddAttribute("nodes_values", thresholds);
  test.AddAttribute("nodes_modes", modes);
  test.AddAttribute("class_treeids", class_treeids);
  test.AddAttribute("class_nodeids", class_nodeids);
  test.AddAttribute("class_ids", class_classids);
  test.AddAttribute("class_weights", class_weights);
  test.AddAttribute("classlabels_int64s", classes);

  test.AddInput<float>("X", {N, 3}, X);
  test.AddOutput<int64_t>("Y", {N}, results);
  test.AddOutput<float>("Z", {N, static_cast<int64_t>(classes.size())}, scores);
  test.Run();
}

TEST(MLOpTest, TreeEnsembleClassifier) {
  TreeEnsembleClassifierTest(1);
  TreeEnsembleClassifierTest(3);
}

TEST(MLOpTest, TreeEnsembleClassifier_as_tensor) {
  OpTester test("TreeEnsembleClassifier", 3, onnxruntime::kMLDomain);

  std::vector<int64_t> lefts = {1, -1, 3, -1, -1, 1, -1, 3, 4, -1, -1, -1, 1, 2, -1, 4, -1, -1, -1};
  std::vector<int64_t> rights = {2, -1, 4, -1, -1, 2, -1, 6, 5, -1, -1, -1, 6, 3, -1, 5, -1, -1, -1};
  std::vector<int64_t> treeids = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2};
  std::vector<int64_t> nodeids = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6};
  std::vector<int64_t> featureids = {2, -2, 0, -2, -2, 0, -2, 2, 1, -2, -2, -2, 0, 2, -2, 1, -2, -2, -2};
  std::vector<double> thresholds = {-172.f, -2.f, 2.5f, -2.f, -2.f, 1.5f, -2.f, -62.5f, 213.09999084f,
                                    -2.f, -2.f, -2.f, 27.5f, -172.f, -2.f, 8.10000038f, -2.f, -2.f, -2.f};
  std::vector<std::string> modes = {"BRANCH_LEQ", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_LEQ",
                                    "LEAF", "BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF",
                                    "BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF"};
  std::vector<int64_t> class_treeids = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  std::vector<int64_t> class_nodeids = {1, 3, 4, 1, 4, 5, 6, 2, 4, 5, 6};
  std::vector<int64_t> class_classids = {2, 0, 1, 0, 2, 3, 1, 2, 0, 1, 3};
  std::vector<double> class_weights = {1.f, 4.f, 1.f, 2.f, 1.f, 1.f, 2.f, 1.f, 1.f, 1.f, 3.f};
  std::vector<int64_t> classes = {0, 1, 2, 3};
  std::vector<double> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f,
                           11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<int64_t> results = {0, 1, 2, 2, 2, 2, 2, 3};
  std::vector<float> scores{7, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0,
                            0, 0, 3, 0, 0, 0, 2, 1, 0, 0, 3, 0, 0, 1, 0, 4};
  std::vector<float> probs = {};
  std::vector<float> log_probs = {};

  ONNX_NAMESPACE::TensorProto nodes_values_as_tensor;
  nodes_values_as_tensor.set_name("nodes_values_as_tensor");
  nodes_values_as_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
  nodes_values_as_tensor.add_dims(thresholds.size());
  for (auto v : thresholds) {
    nodes_values_as_tensor.add_double_data(v);
  }

  ONNX_NAMESPACE::TensorProto class_weights_as_tensor;
  class_weights_as_tensor.set_name("class_weights_as_tensor");
  class_weights_as_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
  class_weights_as_tensor.add_dims(class_weights.size());
  for (auto v : class_weights) {
    class_weights_as_tensor.add_double_data(v);
  }

  // define the context of the operator call
  constexpr int N = 8;
  test.AddAttribute("nodes_truenodeids", lefts);
  test.AddAttribute("nodes_falsenodeids", rights);
  test.AddAttribute("nodes_treeids", treeids);
  test.AddAttribute("nodes_nodeids", nodeids);
  test.AddAttribute("nodes_featureids", featureids);
  test.AddAttribute("nodes_values_as_tensor", nodes_values_as_tensor);
  test.AddAttribute("nodes_modes", modes);
  test.AddAttribute("class_treeids", class_treeids);
  test.AddAttribute("class_nodeids", class_nodeids);
  test.AddAttribute("class_ids", class_classids);
  test.AddAttribute("class_weights_as_tensor", class_weights_as_tensor);
  test.AddAttribute("classlabels_int64s", classes);

  test.AddInput<double>("X", {N, 3}, X);
  test.AddOutput<int64_t>("Y", {N}, results);
  test.AddOutput<float>("Z", {N, static_cast<int64_t>(classes.size())}, scores);
  test.Run();
}

TEST(MLOpTest, TreeEnsembleClassifier_N1) {
  OpTester test("TreeEnsembleClassifier", 1, onnxruntime::kMLDomain);

  std::vector<int64_t> lefts = {1, -1, 3, -1, -1, 1, -1, 3, 4, -1, -1, -1, 1, 2, -1, 4, -1, -1, -1};
  std::vector<int64_t> rights = {2, -1, 4, -1, -1, 2, -1, 6, 5, -1, -1, -1, 6, 3, -1, 5, -1, -1, -1};
  std::vector<int64_t> treeids = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2};
  std::vector<int64_t> nodeids = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6};
  std::vector<int64_t> featureids = {2, -2, 0, -2, -2, 0, -2, 2, 1, -2, -2, -2, 0, 2, -2, 1, -2, -2, -2};
  std::vector<float> thresholds = {-172.f, -2.f, 2.5f, -2.f, -2.f, 1.5f, -2.f, -62.5f, 213.09999084f,
                                   -2.f, -2.f, -2.f, 27.5f, -172.f, -2.f, 8.10000038f, -2.f, -2.f, -2.f};
  std::vector<std::string> modes = {"BRANCH_LEQ", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_LEQ",
                                    "LEAF", "BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF",
                                    "BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF"};
  std::vector<int64_t> class_treeids = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  std::vector<int64_t> class_nodeids = {1, 3, 4, 1, 4, 5, 6, 2, 4, 5, 6};
  std::vector<int64_t> class_classids = {2, 0, 1, 0, 2, 3, 1, 2, 0, 1, 3};
  std::vector<float> class_weights = {1.f, 4.f, 1.f, 2.f, 1.f, 1.f, 2.f, 1.f, 1.f, 1.f, 3.f};
  std::vector<int64_t> classes = {0, 1, 2, 3};
  std::vector<float> X = {1.f, 0.0f, 0.4f};
  std::vector<int64_t> results = {0};
  std::vector<float> scores{7, 0, 0, 0};
  std::vector<float> probs = {};
  std::vector<float> log_probs = {};

  // define the context of the operator call
  constexpr int N = 1;
  test.AddAttribute("nodes_truenodeids", lefts);
  test.AddAttribute("nodes_falsenodeids", rights);
  test.AddAttribute("nodes_treeids", treeids);
  test.AddAttribute("nodes_nodeids", nodeids);
  test.AddAttribute("nodes_featureids", featureids);
  test.AddAttribute("nodes_values", thresholds);
  test.AddAttribute("nodes_modes", modes);
  test.AddAttribute("class_treeids", class_treeids);
  test.AddAttribute("class_nodeids", class_nodeids);
  test.AddAttribute("class_ids", class_classids);
  test.AddAttribute("class_weights", class_weights);
  test.AddAttribute("classlabels_int64s", classes);

  test.AddInput<float>("X", {N, 3}, X);
  test.AddOutput<int64_t>("Y", {N}, results);
  test.AddOutput<float>("Z", {N, static_cast<int64_t>(classes.size())}, scores);
  test.Run();
}

TEST(MLOpTest, TreeEnsembleClassifierFailShape) {
  OpTester test("TreeEnsembleClassifier", 1, onnxruntime::kMLDomain);

  std::vector<int64_t> lefts = {1, -1, 3, -1, -1, 1, -1, 3, 4, -1, -1, -1, 1, 2, -1, 4, -1, -1, -1};
  std::vector<int64_t> rights = {2, -1, 4, -1, -1, 2, -1, 6, 5, -1, -1, -1, 6, 3, -1, 5, -1, -1, -1};
  std::vector<int64_t> treeids = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2};
  std::vector<int64_t> nodeids = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6};
  std::vector<int64_t> featureids = {2, -2, 0, -2, -2, 0, -2, 2, 1, -2, -2, -2, 0, 2, -2, 1, -2, -2, -2};
  std::vector<float> thresholds = {-172.f, -2.f, 2.5f, -2.f, -2.f, 1.5f, -2.f, -62.5f, 213.09999084f,
                                   -2.f, -2.f, -2.f, 27.5f, -172.f, -2.f, 8.10000038f, -2.f, -2.f, -2.f};
  std::vector<std::string> modes = {"BRANCH_LEQ", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_LEQ",
                                    "LEAF", "BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF",
                                    "BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF"};
  std::vector<int64_t> class_treeids = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  std::vector<int64_t> class_nodeids = {1, 3, 4, 1, 4, 5, 6, 2, 4, 5, 6};
  std::vector<int64_t> class_classids = {2, 0, 1, 0, 2, 3, 1, 2, 0, 1, 3};
  std::vector<float> class_weights = {1.f, 4.f, 1.f, 2.f, 1.f, 1.f, 2.f, 1.f, 1.f, 1.f, 3.f};
  std::vector<int64_t> classes = {0, 1, 2, 3};
  std::vector<float> X = {1.f, 0.0f};
  std::vector<int64_t> results = {0};
  std::vector<float> scores{7, 0, 0, 0};
  std::vector<float> probs = {};
  std::vector<float> log_probs = {};

  constexpr int N = 1;
  test.AddAttribute("nodes_truenodeids", lefts);
  test.AddAttribute("nodes_falsenodeids", rights);
  test.AddAttribute("nodes_treeids", treeids);
  test.AddAttribute("nodes_nodeids", nodeids);
  test.AddAttribute("nodes_featureids", featureids);
  test.AddAttribute("nodes_values", thresholds);
  test.AddAttribute("nodes_modes", modes);
  test.AddAttribute("class_treeids", class_treeids);
  test.AddAttribute("class_nodeids", class_nodeids);
  test.AddAttribute("class_ids", class_classids);
  test.AddAttribute("class_weights", class_weights);
  test.AddAttribute("classlabels_int64s", classes);

  test.AddInput<float>("X", {N, 2}, X);
  test.AddOutput<int64_t>("Y", {N}, results);
  test.AddOutput<float>("Z", {N, static_cast<int64_t>(classes.size())}, scores);
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "One path in the graph requests feature 2 but input tensor has 2 features.");
}

TEST(MLOpTest, TreeEnsembleClassifierLabels) {
  OpTester test("TreeEnsembleClassifier", 1, onnxruntime::kMLDomain);

  std::vector<int64_t> lefts = {1, -1, 3, -1, -1, 1, -1, 3, 4, -1, -1, -1, 1, 2, -1, 4, -1, -1, -1};
  std::vector<int64_t> rights = {2, -1, 4, -1, -1, 2, -1, 6, 5, -1, -1, -1, 6, 3, -1, 5, -1, -1, -1};
  std::vector<int64_t> treeids = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2};
  std::vector<int64_t> nodeids = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6};
  std::vector<int64_t> featureids = {2, -2, 0, -2, -2, 0, -2, 2, 1, -2, -2, -2, 0, 2, -2, 1, -2, -2, -2};
  std::vector<float> thresholds = {-172.f, -2.f, 2.5f, -2.f, -2.f, 1.5f, -2.f, -62.5f, 213.09999084f,
                                   -2.f, -2.f, -2.f, 27.5f, -172.f, -2.f, 8.10000038f, -2.f, -2.f, -2.f};
  std::vector<std::string> modes = {"BRANCH_LEQ", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_LEQ",
                                    "LEAF", "BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF", "BRANCH_LEQ",
                                    "BRANCH_LEQ", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF"};

  std::vector<int64_t> class_treeids = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  std::vector<int64_t> class_nodeids = {1, 3, 4, 1, 4, 5, 6, 2, 4, 5, 6};
  std::vector<int64_t> class_classids = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> class_weights = {-1.f, 4.f, -1.f, 2.f, -1.f, +1.f, -2.f, 1.f, -1.f, 2.f, -3.f};
  std::vector<std::string> labels = {"label0", "label1"};
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f,
                          11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<std::string> results = {"label1", "label0", "label0", "label0", "label0", "label1", "label0", "label0"};
  std::vector<float> probs = {};
  std::vector<float> log_probs = {};
  std::vector<float> scores{-5, 5, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 3, -3};

  // define the context of the operator call
  constexpr int N = 8;
  test.AddAttribute("nodes_truenodeids", lefts);
  test.AddAttribute("nodes_falsenodeids", rights);
  test.AddAttribute("nodes_treeids", treeids);
  test.AddAttribute("nodes_nodeids", nodeids);
  test.AddAttribute("nodes_featureids", featureids);
  test.AddAttribute("nodes_values", thresholds);
  test.AddAttribute("nodes_modes", modes);
  test.AddAttribute("class_treeids", class_treeids);
  test.AddAttribute("class_nodeids", class_nodeids);
  test.AddAttribute("class_ids", class_classids);
  test.AddAttribute("class_weights", class_weights);
  test.AddAttribute("classlabels_strings", labels);

  test.AddInput<float>("X", {N, 3}, X);
  test.AddOutput<std::string>("Y", {N}, results);
  test.AddOutput<float>("Z", {N, static_cast<int64_t>(labels.size())}, scores);

  test.Run();
}

TEST(MLOpTest, TreeEnsembleClassifierBinary) {
  OpTester test("TreeEnsembleClassifier", 1, onnxruntime::kMLDomain);

  std::vector<int64_t> lefts = {1, -1, 3, -1, -1, 1, -1, 3, 4, -1, -1, -1, 1, 2, -1, 4, -1, -1, -1};
  std::vector<int64_t> rights = {2, -1, 4, -1, -1, 2, -1, 6, 5, -1, -1, -1, 6, 3, -1, 5, -1, -1, -1};
  std::vector<int64_t> treeids = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2};
  std::vector<int64_t> nodeids = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6};
  std::vector<int64_t> featureids = {2, -2, 0, -2, -2, 0, -2, 2, 1, -2, -2, -2, 0, 2, -2, 1, -2, -2, -2};
  std::vector<float> thresholds = {-172.f, -2.f, 2.5f, -2.f, -2.f, 1.5f, -2.f, -62.5f, 213.09999084f, -2.f,
                                   -2.f, -2.f, 27.5f, -172.f, -2.f, 8.10000038f, -2.f, -2.f, -2.f};
  std::vector<std::string> modes = {"BRANCH_LEQ", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_LEQ",
                                    "LEAF", "BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF",
                                    "BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF"};
  // std::vector<int64_t> classes = {0, 1, 2, 3};
  std::vector<int64_t> class_treeids = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  std::vector<int64_t> class_nodeids = {1, 3, 4, 1, 4, 5, 6, 2, 4, 5, 6};
  std::vector<int64_t> class_classids = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  std::vector<float> class_weights = {-1.f, 4.f, -1.f, 2.f, -1.f, +1.f, -2.f, 1.f, -1.f, 2.f, -3.f};
  std::vector<int64_t> classes = {1};
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f,
                          23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f,
                          -114.f};
  std::vector<int64_t> results = {1, 0, 0, 0, 0, 1, 0, 0};
  std::vector<float> probs = {};
  std::vector<float> log_probs = {};
  std::vector<float> scores{5, -1, -1, -1, -1, 1, -1, -3};

  // define the context of the operator call
  constexpr int N = 8;
  test.AddAttribute("nodes_truenodeids", lefts);
  test.AddAttribute("nodes_falsenodeids", rights);
  test.AddAttribute("nodes_treeids", treeids);
  test.AddAttribute("nodes_nodeids", nodeids);
  test.AddAttribute("nodes_featureids", featureids);
  test.AddAttribute("nodes_values", thresholds);
  test.AddAttribute("nodes_modes", modes);
  test.AddAttribute("class_treeids", class_treeids);
  test.AddAttribute("class_nodeids", class_nodeids);
  test.AddAttribute("class_ids", class_classids);
  test.AddAttribute("class_weights", class_weights);
  test.AddAttribute("classlabels_int64s", classes);

  test.AddInput<float>("X", {N, 3}, X);
  test.AddOutput<int64_t>("Y", {N}, results);
  test.AddOutput<float>("Z", {N, 1}, scores);

  test.Run();
}

TEST(MLOpTest, TreeEnsembleClassifierBinaryProbabilities) {
  OpTester test("TreeEnsembleClassifier", 1, onnxruntime::kMLDomain);

  std::vector<int64_t> lefts = {1, -1, 3, -1, -1, 1, -1, 3, 4, -1, -1, -1, 1, 2, -1, 4, -1, -1, -1};
  std::vector<int64_t> rights = {2, -1, 4, -1, -1, 2, -1, 6, 5, -1, -1, -1, 6, 3, -1, 5, -1, -1, -1};
  std::vector<int64_t> treeids = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2};
  std::vector<int64_t> nodeids = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6};
  std::vector<int64_t> featureids = {2, -2, 0, -2, -2, 0, -2, 2, 1, -2, -2, -2, 0, 2, -2, 1, -2, -2, -2};
  std::vector<float> thresholds = {-172.f, -2.f, 2.5f, -2.f, -2.f, 1.5f, -2.f, -62.5f, 213.09999084f, -2.f,
                                   -2.f, -2.f, 27.5f, -172.f, -2.f, 8.10000038f, -2.f, -2.f, -2.f};
  std::vector<std::string> modes = {"BRANCH_LEQ", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_LEQ",
                                    "LEAF", "BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF",
                                    "BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF"};
  // std::vector<int64_t> classes = {0, 1};
  std::vector<int64_t> class_treeids = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2};
  std::vector<int64_t> class_nodeids = {1, 3, 4, 1, 4, 5, 6, 2, 4, 5, 6};
  std::vector<int64_t> class_classids = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1};
  std::vector<float> class_weights = {-1.f, 4.f, -1.f, 2.f, -1.f, +1.f, -2.f, 1.f, -1.f, 2.f, -3.f};
  std::vector<int64_t> classes = {0, 1};
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f,
                          23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f,
                          -114.f};
  std::vector<int64_t> results = {1, 1, 0, 0, 0, 1, 0, 0};
  std::vector<float> probs = {};
  std::vector<float> log_probs = {};
  std::vector<float> scores{
      0.00669282f, 0.99330717f,
      0.04742586f, 0.88079702f,
      0.73105859f, 0.26894142f,
      0.73105859f, 0.26894142f,
      0.73105859f, 0.26894142f,
      0.26894140f, 0.73105859f,
      0.73105859f, 0.26894142f,
      0.5f, 0.04742586f};

  // define the context of the operator call
  constexpr int N = 8;
  test.AddAttribute("nodes_truenodeids", lefts);
  test.AddAttribute("nodes_falsenodeids", rights);
  test.AddAttribute("nodes_treeids", treeids);
  test.AddAttribute("nodes_nodeids", nodeids);
  test.AddAttribute("nodes_featureids", featureids);
  test.AddAttribute("nodes_values", thresholds);
  test.AddAttribute("nodes_modes", modes);
  test.AddAttribute("class_treeids", class_treeids);
  test.AddAttribute("class_nodeids", class_nodeids);
  test.AddAttribute("class_ids", class_classids);
  test.AddAttribute("class_weights", class_weights);
  test.AddAttribute("classlabels_int64s", classes);
  test.AddAttribute("post_transform", "LOGISTIC");

  test.AddInput<float>("X", {N, 3}, X);
  test.AddOutput<int64_t>("Y", {N}, results);
  test.AddOutput<float>("Z", {N, 2}, scores);

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
