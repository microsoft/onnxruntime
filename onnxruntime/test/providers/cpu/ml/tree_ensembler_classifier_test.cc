// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(MLOpTest, TreeEnsembleClassifier) {
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
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f,
                          11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<int64_t> results = {0, 1, 2, 2, 2, 2, 2, 3};
  std::vector<float> scores{7, 0, 0, 0, 0, 4, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0,
                            0, 0, 3, 0, 0, 0, 2, 1, 0, 0, 3, 0, 0, 1, 0, 4};
  std::vector<float> probs = {};
  std::vector<float> log_probs = {};

  //define the context of the operator call
  const int N = 8;
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
  std::vector<float> scores{-5, 5, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -3, 3};

  //define the context of the operator call
  const int N = 8;
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
  //std::vector<int64_t> classes = {0, 1, 2, 3};
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

  //define the context of the operator call
  const int N = 8;
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

TEST(MLOpTest, TreeEnsembleClassifierBinaryBaseValue) {
  OpTester test("TreeEnsembleClassifier", 1, onnxruntime::kMLDomain);

  // The example was generated by the following python script:
  // model = GradientBoostingClassifier(n_estimators = 1, max_depth = 2)
  // X, y = make_classification(10, n_features = 4, random_state = 42)
  // X = X[:, :2]
  // model.fit(X, y)
  // model.init_.class_prior_ = np.array([0.231, 0.231])

  std::vector<float> base_values = {-1.202673316001892f, -1.202673316001892f};
  std::vector<int64_t> class_ids = {0, 0, 0};
  std::vector<int64_t> class_nodeids = {2, 3, 4};
  std::vector<int64_t> class_treeids = {0, 0, 0};
  std::vector<float> class_weights = {-0.2f, -0.06f, 0.2f};
  std::vector<int64_t> classlabels_int64s = {0, 1};
  std::vector<int64_t> nodes_falsenodeids = {4, 3, 0, 0, 0};
  std::vector<int64_t> nodes_featureids = {0, 0, 0, 0, 0};
  std::vector<float> nodes_hitrates = {1, 1, 1, 1, 1};
  std::vector<int64_t> nodes_missing_value_tracks_true = {0, 0, 0, 0, 0};
  std::vector<std::string> nodes_modes = {"BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF"};
  std::vector<int64_t> nodes_nodeids = {0, 1, 2, 3, 4};
  std::vector<int64_t> nodes_treeids = {0, 0, 0, 0, 0};
  std::vector<int64_t> nodes_truenodeids = {1, 2, 0, 0, 0};
  std::vector<float> nodes_values = {0.21111594140529633f, -0.8440752029418945f, 0, 0, 0};
  std::string post_transform = "LOGISTIC";

  std::vector<float> X = {-0.92533575f, -1.14021544f, -0.46171143f, -0.58723065f, 1.44044386f, 1.77736657f};
  std::vector<int64_t> results = {0, 0, 0};
  std::vector<float> probs = {};
  std::vector<float> log_probs = {};
  std::vector<float> scores{0.802607834f, 0.197392166f, 0.779485941f, 0.220514059f, 0.731583834f, 0.268416166f};

  //define the context of the operator call
  const int N = 3;
  test.AddAttribute("base_values", base_values);
  test.AddAttribute("class_ids", class_ids);
  test.AddAttribute("class_nodeids", class_nodeids);
  test.AddAttribute("class_treeids", class_treeids);
  test.AddAttribute("class_weights", class_weights);
  test.AddAttribute("classlabels_int64s", classlabels_int64s);
  test.AddAttribute("nodes_falsenodeids", nodes_falsenodeids);
  test.AddAttribute("nodes_featureids", nodes_featureids);
  test.AddAttribute("nodes_hitrates", nodes_hitrates);
  test.AddAttribute("nodes_modes", nodes_modes);
  test.AddAttribute("nodes_nodeids", nodes_nodeids);
  test.AddAttribute("nodes_treeids", nodes_treeids);
  test.AddAttribute("nodes_truenodeids", nodes_truenodeids);
  test.AddAttribute("nodes_values", nodes_values);
  test.AddAttribute("post_transform", post_transform);

  test.AddInput<float>("X", {N, 2}, X);
  test.AddOutput<int64_t>("Y", {N}, results);
  test.AddOutput<float>("Z", {N, 2}, scores);

  test.Run();
}

TEST(MLOpTest, TreeEnsembleClassifierBinaryBaseValueNull) {
  OpTester test("TreeEnsembleClassifier", 1, onnxruntime::kMLDomain);

  // The example was generated by the following python script:
  // model = GradientBoostingClassifier(n_estimators = 1, max_depth = 2)
  // X, y = make_classification(10, n_features = 4, random_state = 42)
  // X = X[:, :2]
  // model.fit(X, y)

  std::vector<float> base_values = {0, 0};
  std::vector<int64_t> class_ids = {0, 0, 0};
  std::vector<int64_t> class_nodeids = {2, 3, 4};
  std::vector<int64_t> class_treeids = {0, 0, 0};
  std::vector<float> class_weights = {-0.2f, -0.0666f, 0.2f};
  std::vector<int64_t> classlabels_int64s = {0, 1};
  std::vector<int64_t> nodes_falsenodeids = {4, 3, 0, 0, 0};
  std::vector<int64_t> nodes_featureids = {0, 0, 0, 0, 0};
  std::vector<float> nodes_hitrates = {1, 1, 1, 1, 1};
  std::vector<int64_t> nodes_missing_value_tracks_true = {0, 0, 0, 0, 0};
  std::vector<std::string> nodes_modes = {"BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF"};
  std::vector<int64_t> nodes_nodeids = {0, 1, 2, 3, 4};
  std::vector<int64_t> nodes_treeids = {0, 0, 0, 0, 0};
  std::vector<int64_t> nodes_truenodeids = {1, 2, 0, 0, 0};
  std::vector<float> nodes_values = {0.24055418372154236f, -0.8440752029418945f, 0, 0, 0};
  std::string post_transform = "LOGISTIC";

  std::vector<float> X = {-0.92533575f, -1.14021544f, -0.46171143f, -0.58723065f, 1.44044386f, 1.77736657f};
  std::vector<int64_t> results = {0, 0, 1};
  std::vector<float> probs = {};
  std::vector<float> log_probs = {};
  std::vector<float> scores{0.549834f, 0.450166f, 0.5166605f, 0.4833395f, 0.450166f, 0.549834f};

  //define the context of the operator call
  const int N = 3;
  test.AddAttribute("base_values", base_values);
  test.AddAttribute("class_ids", class_ids);
  test.AddAttribute("class_nodeids", class_nodeids);
  test.AddAttribute("class_treeids", class_treeids);
  test.AddAttribute("class_weights", class_weights);
  test.AddAttribute("classlabels_int64s", classlabels_int64s);
  test.AddAttribute("nodes_falsenodeids", nodes_falsenodeids);
  test.AddAttribute("nodes_featureids", nodes_featureids);
  test.AddAttribute("nodes_hitrates", nodes_hitrates);
  test.AddAttribute("nodes_modes", nodes_modes);
  test.AddAttribute("nodes_nodeids", nodes_nodeids);
  test.AddAttribute("nodes_treeids", nodes_treeids);
  test.AddAttribute("nodes_truenodeids", nodes_truenodeids);
  test.AddAttribute("nodes_values", nodes_values);
  test.AddAttribute("post_transform", post_transform);

  test.AddInput<float>("X", {N, 2}, X);
  test.AddOutput<int64_t>("Y", {N}, results);
  test.AddOutput<float>("Z", {N, 2}, scores);

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
