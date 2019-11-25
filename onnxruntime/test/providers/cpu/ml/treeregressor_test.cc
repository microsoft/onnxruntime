// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

void GenTreeAndRunTest(const std::vector<float>& X, const std::vector<float>& base_values, const std::vector<float>& results, const std::string& aggFunction)
{
  OpTester test("TreeEnsembleRegressor", 1, onnxruntime::kMLDomain);

  //tree
  std::vector<int64_t> lefts = {1, 2, -1, -1, -1, 1, -1, 3, -1, -1, 1, -1, -1};
  std::vector<int64_t> rights = {4, 3, -1, -1, -1, 2, -1, 4, -1, -1, 2, -1, -1};
  std::vector<int64_t> treeids = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2};
  std::vector<int64_t> nodeids = {0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2};
  std::vector<int64_t> featureids = {2, 1, -2, -2, -2, 0, -2, 2, -2, -2, 1, -2, -2};
  std::vector<float> thresholds = {10.5f, 13.10000038f, -2.f, -2.f, -2.f, 1.5f, -2.f, -213.f, -2.f, -2.f, 13.10000038f, -2.f, -2.f};
  std::vector<std::string> modes = {"BRANCH_LEQ", "BRANCH_LEQ", "LEAF", "LEAF", "LEAF", "BRANCH_LEQ", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF"};

  std::vector<int64_t> target_treeids = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};
  std::vector<int64_t> target_nodeids = {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 1, 1, 2, 2};
  std::vector<int64_t> target_classids = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
  std::vector<float> target_weights = {1.5f, 27.5f, 2.25f, 20.75f, 2.f, 23.f, 3.f, 14.f, 0.f, 41.f, 1.83333333f, 24.5f, 0.f, 41.f, 2.75f, 16.25f, 2.f, 23.f, 3.f, 14.f, 2.66666667f, 17.f, 2.f, 23.f, 3.f, 14.f};
  std::vector<int64_t> classes = {0, 1};

  //add attributes
  test.AddAttribute("nodes_truenodeids", lefts);
  test.AddAttribute("nodes_falsenodeids", rights);
  test.AddAttribute("nodes_treeids", treeids);
  test.AddAttribute("nodes_nodeids", nodeids);
  test.AddAttribute("nodes_featureids", featureids);
  test.AddAttribute("nodes_values", thresholds);
  test.AddAttribute("nodes_modes", modes);
  test.AddAttribute("target_treeids", target_treeids);
  test.AddAttribute("target_nodeids", target_nodeids);
  test.AddAttribute("target_ids", target_classids);
  test.AddAttribute("target_weights", target_weights);
  test.AddAttribute("base_values", base_values);

  test.AddAttribute("n_targets", (int64_t)2);

  if (aggFunction == "AVERAGE") {
	test.AddAttribute("aggregate_function", "AVERAGE");
  } else if (aggFunction == "MIN") {
    test.AddAttribute("aggregate_function", "MIN");
  } else if (aggFunction == "MAX") {
    test.AddAttribute("aggregate_function", "MAX");
  } // default function is SUM

  //fill input data
  test.AddInput<float>("X", {8, 3}, X);
  test.AddOutput<float>("Y", {8, 2}, results);
  test.Run();
}

TEST(MLOpTest, TreeRegressorMultiTargetAverage) {
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<float> results = {1.33333333f, 29.f, 3.f, 14.f, 2.f, 23.f, 2.f, 23.f, 2.f, 23.f, 2.66666667f, 17.f, 2.f, 23.f, 3.f, 14.f};
  std::vector<float> base_values{0.f, 0.f};
  GenTreeAndRunTest(X, base_values, results, "AVERAGE");
}

TEST(MLOpTest, TreeRegressorMultiTargetMin) {
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<float> results = {5.f, 28.f, 8.f, 19.f, 7.f, 28.f, 7.f, 28.f, 7.f, 28.f, 7.f, 19.f, 7.f, 28.f, 8.f, 19.f};
  std::vector<float> base_values{5.f, 5.f};
  GenTreeAndRunTest(X, base_values, results, "MIN");
}

TEST(MLOpTest, TreeRegressorMultiTargetMax) {
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<float> results = {2.f, 41.f, 3.f, 14.f, 2.f, 23.f, 2.f, 23.f, 2.f, 23.f, 3.f, 23.f, 2.f, 23.f, 3.f, 14.f};
  std::vector<float> base_values{0.f, 0.f};
  GenTreeAndRunTest(X, base_values, results, "MAX");
}

TEST(MLOpTest, TreeRegressorSingleTargetSum) {
  OpTester test("TreeEnsembleRegressor", 1, onnxruntime::kMLDomain);

  //tree
  std::vector<int64_t> lefts = {1, 0, 0, 1, 0, 0, 1, 0 ,0};
  std::vector<int64_t> rights = {2,0,0,2,0,0,2,0,0};
  std::vector<int64_t> treeids = {0,0,0,1,1,1,2,2,2};
  std::vector<int64_t> nodeids = {0,1,2,0,1,2,0,1,2};
  std::vector<int64_t> featureids = {0,0,0,0,0,0,1,0,0};
  std::vector<float> thresholds = {1,0,0,0.5,0,0,0.5,0,0 };
  std::vector<std::string> modes = {"BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF"};

  std::vector<int64_t> target_treeids = {0,0,1,1,2,2};
  std::vector<int64_t> target_nodeids = {1,2,1,2,1,2};
  std::vector<int64_t> target_classids = {0,0,0,0,0,0};
  std::vector<float> target_weights = {33.33333f, 16.66666f, 33.33333f, -3.33333f, 16.66666f, -3.333333f};
  std::vector<int64_t> classes = {0, 1};

  //test data
  std::vector<float> X = {0,1,1,1,2,0};
  std::vector<float> results = {63.33333333f, 26.66666667f, 30.0f};

  //add attributes
  test.AddAttribute("nodes_truenodeids", lefts);
  test.AddAttribute("nodes_falsenodeids", rights);
  test.AddAttribute("nodes_treeids", treeids);
  test.AddAttribute("nodes_nodeids", nodeids);
  test.AddAttribute("nodes_featureids", featureids);
  test.AddAttribute("nodes_values", thresholds);
  test.AddAttribute("nodes_modes", modes);
  test.AddAttribute("target_treeids", target_treeids);
  test.AddAttribute("target_nodeids", target_nodeids);
  test.AddAttribute("target_ids", target_classids);
  test.AddAttribute("target_weights", target_weights);

  test.AddAttribute("n_targets", (int64_t)1);
  // SUM aggregation by default -- no need to add explicitly

  //fill input data
  test.AddInput<float>("X", {3, 2}, X);
  test.AddOutput<float>("Y", {3, 1}, results);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
