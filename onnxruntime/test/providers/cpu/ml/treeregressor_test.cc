// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

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

void _multiply_update_array_string(std::vector<std::string>& data, int n) {
  std::vector<std::string> copy = data;
  data.resize(copy.size() * n);
  for (int i = 0; i < n; ++i) {
    for (size_t j = 0; j < copy.size(); ++j) {
      data[j + i * copy.size()] = copy[j];
    }
  }
}

template <typename T>
void GenTreeAndRunTest(const std::vector<T>& X, const std::vector<float>& base_values, const std::vector<float>& results, const std::string& aggFunction,
                       bool one_obs = false, int64_t n_obs = 8, int n_trees = 1) {
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

  if (n_trees > 1) {
    // Multiplies the number of trees to test the parallelization by trees.
    _multiply_update_array(lefts, n_trees);
    _multiply_update_array(rights, n_trees);
    _multiply_update_array(treeids, n_trees, (int64_t)3);
    _multiply_update_array(nodeids, n_trees);
    _multiply_update_array(featureids, n_trees);
    _multiply_update_array(thresholds, n_trees);
    _multiply_update_array_string(modes, n_trees);
    _multiply_update_array(target_treeids, n_trees, (int64_t)3);
    _multiply_update_array(target_nodeids, n_trees);
    _multiply_update_array(target_classids, n_trees);
    _multiply_update_array(target_weights, n_trees);
  }

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
  }  // default function is SUM

  //fill input data
  std::vector<T> xn;
  std::vector<float> yn;
  if (one_obs) {
    auto X1 = X;
    auto results1 = results;
    X1.resize(3);
    results1.resize(2);
    test.AddInput<T>("X", {1, 3}, X1);
    test.AddOutput<float>("Y", {1, 2}, results1);
  } else if (n_obs == 8) {
    test.AddInput<T>("X", {8, 3}, X);
    test.AddOutput<float>("Y", {8, 2}, results);
  } else {
    int64_t i;
    size_t k;
    ASSERT_TRUE(n_obs % 8 == 0);
    xn.resize(n_obs * 3);
    yn.resize(n_obs * 2);
    for (i = 0; i < n_obs; i += 8) {
      for (k = 0; k < 24; ++k) {
        xn[i * 3 + k] = X[k];
      }
      for (k = 0; k < 16; ++k) {
        yn[i * 2 + k] = results[k];
      }
    }
    ASSERT_TRUE(i == n_obs);
    test.AddInput<T>("X", {n_obs, 3}, xn);
    test.AddOutput<float>("Y", {n_obs, 2}, yn);
  }

  test.Run();
}  // namespace test

TEST(MLOpTest, TreeRegressorMultiTargetBatchTreeA2) {
  // TreeEnsemble implements different paths depending on n_trees or N.
  // This test and the next ones go through all sections for multi-targets.
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<float> results = {1.33333333f, 29.f, 3.f, 14.f, 2.f, 23.f, 2.f, 23.f, 2.f, 23.f, 2.66666667f, 17.f, 2.f, 23.f, 3.f, 14.f};
  std::vector<float> base_values{0.f, 0.f};
  GenTreeAndRunTest(X, base_values, results, "AVERAGE", true, 8, 1);  // section A2
}

TEST(MLOpTest, TreeRegressorMultiTargetBatchTreeB2) {
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<float> results = {1.33333333f, 29.f, 3.f, 14.f, 2.f, 23.f, 2.f, 23.f, 2.f, 23.f, 2.66666667f, 17.f, 2.f, 23.f, 3.f, 14.f};
  std::vector<float> base_values{0.f, 0.f};
  GenTreeAndRunTest(X, base_values, results, "AVERAGE", true, 8, 130);  // section B2
}

TEST(MLOpTest, TreeRegressorMultiTargetBatchTreeC2) {
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<float> results = {1.33333333f, 29.f, 3.f, 14.f, 2.f, 23.f, 2.f, 23.f, 2.f, 23.f, 2.66666667f, 17.f, 2.f, 23.f, 3.f, 14.f};
  std::vector<float> base_values{0.f, 0.f};
  GenTreeAndRunTest(X, base_values, results, "AVERAGE", false, 200, 130);  // section C2
}

TEST(MLOpTest, TreeRegressorMultiTargetBatchTreeD2) {
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<float> results = {1.33333333f, 29.f, 3.f, 14.f, 2.f, 23.f, 2.f, 23.f, 2.f, 23.f, 2.66666667f, 17.f, 2.f, 23.f, 3.f, 14.f};
  std::vector<float> base_values{0.f, 0.f};
  GenTreeAndRunTest(X, base_values, results, "AVERAGE", true, 8, 30);     // section D2
  GenTreeAndRunTest(X, base_values, results, "AVERAGE", false, 200, 30);  // section D2
}

TEST(MLOpTest, TreeRegressorMultiTargetBatchTreeE2) {
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<float> results = {1.33333333f, 29.f, 3.f, 14.f, 2.f, 23.f, 2.f, 23.f, 2.f, 23.f, 2.66666667f, 17.f, 2.f, 23.f, 3.f, 14.f};
  std::vector<float> base_values{0.f, 0.f};
  GenTreeAndRunTest(X, base_values, results, "AVERAGE", false, 200, 1);  // section E2
}

TEST(MLOpTest, TreeRegressorMultiTargetAverage) {
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<float> results = {1.33333333f, 29.f, 3.f, 14.f, 2.f, 23.f, 2.f, 23.f, 2.f, 23.f, 2.66666667f, 17.f, 2.f, 23.f, 3.f, 14.f};
  std::vector<float> base_values{0.f, 0.f};
  GenTreeAndRunTest<float>(X, base_values, results, "AVERAGE", false);
  GenTreeAndRunTest<float>(X, base_values, results, "AVERAGE", true);
}

TEST(MLOpTest, TreeRegressorMultiTargetMin) {
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<float> results = {5.f, 28.f, 8.f, 19.f, 7.f, 28.f, 7.f, 28.f, 7.f, 28.f, 7.f, 19.f, 7.f, 28.f, 8.f, 19.f};
  std::vector<float> base_values{5.f, 5.f};
  GenTreeAndRunTest<float>(X, base_values, results, "MIN", false);
  GenTreeAndRunTest<float>(X, base_values, results, "MIN", true);
}

TEST(MLOpTest, TreeRegressorMultiTargetMax) {
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<float> results = {2.f, 41.f, 3.f, 14.f, 2.f, 23.f, 2.f, 23.f, 2.f, 23.f, 3.f, 23.f, 2.f, 23.f, 3.f, 14.f};
  std::vector<float> base_values{0.f, 0.f};
  GenTreeAndRunTest<float>(X, base_values, results, "MAX", false);
  GenTreeAndRunTest<float>(X, base_values, results, "MAX", true);
}

TEST(MLOpTest, TreeRegressorMultiTargetMaxDouble) {
  std::vector<double> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<float> results = {2.f, 41.f, 3.f, 14.f, 2.f, 23.f, 2.f, 23.f, 2.f, 23.f, 3.f, 23.f, 2.f, 23.f, 3.f, 14.f};
  std::vector<float> base_values{0.f, 0.f};
  GenTreeAndRunTest<double>(X, base_values, results, "MAX", false);
  GenTreeAndRunTest<double>(X, base_values, results, "MAX", true);
}

void GenTreeAndRunTest1(const std::string& aggFunction, bool one_obs, int64_t n_obs = 3, int n_trees = 1) {
  OpTester test("TreeEnsembleRegressor", 1, onnxruntime::kMLDomain);

  //tree
  std::vector<int64_t> lefts = {1, 0, 0, 1, 0, 0, 1, 0, 0};
  std::vector<int64_t> rights = {2, 0, 0, 2, 0, 0, 2, 0, 0};
  std::vector<int64_t> treeids = {0, 0, 0, 1, 1, 1, 2, 2, 2};
  std::vector<int64_t> nodeids = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  std::vector<int64_t> featureids = {0, 0, 0, 0, 0, 0, 1, 0, 0};
  std::vector<float> thresholds = {1, 0, 0, 0.5, 0, 0, 0.5, 0, 0};
  std::vector<std::string> modes = {"BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF", "BRANCH_LEQ", "LEAF", "LEAF"};

  std::vector<int64_t> target_treeids = {0, 0, 1, 1, 2, 2};
  std::vector<int64_t> target_nodeids = {1, 2, 1, 2, 1, 2};
  std::vector<int64_t> target_classids = {0, 0, 0, 0, 0, 0};
  std::vector<float> target_weights = {33.33333f, 16.66666f, 33.33333f, -3.33333f, 16.66666f, -3.333333f};
  std::vector<int64_t> classes = {0, 1};

  if (n_trees > 1) {
    // Multiplies the number of trees to test the parallelization by trees.
    _multiply_update_array(lefts, n_trees);
    _multiply_update_array(rights, n_trees);
    _multiply_update_array(treeids, n_trees, (int64_t)3);
    _multiply_update_array(nodeids, n_trees);
    _multiply_update_array(featureids, n_trees);
    _multiply_update_array(thresholds, n_trees);
    _multiply_update_array_string(modes, n_trees);
    _multiply_update_array(target_treeids, n_trees, (int64_t)3);
    _multiply_update_array(target_nodeids, n_trees);
    _multiply_update_array(target_classids, n_trees);
    _multiply_update_array(target_weights, n_trees);
  }

  std::vector<float> results;
  if (aggFunction == "AVERAGE") {
    test.AddAttribute("aggregate_function", "AVERAGE");
    results = {63.33333333f / 3, 26.66666667f / 3, 30.0f / 3};
  } else if (aggFunction == "MIN") {
    test.AddAttribute("aggregate_function", "MIN");
    results = {-3.33333f, -3.33333f, -3.33333f};
  } else if (aggFunction == "MAX") {
    test.AddAttribute("aggregate_function", "MAX");
    results = {33.33333f, 33.33333f, 16.66666f};
  } else {  // default function is SUM
    results = {63.33333333f, 26.66666667f, 30.0f};
  }

  //test data
  std::vector<float> X = {0, 1, 1, 1, 2, 0};

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
  std::vector<float> xn, yn;
  if (one_obs) {
    ASSERT_TRUE(n_obs == 3);
    auto X1 = X;
    auto results1 = results;
    X1.resize(2);
    results1.resize(1);
    test.AddInput<float>("X", {1, 2}, X1);
    test.AddOutput<float>("Y", {1, 1}, results1);
  } else if (n_obs == 3) {
    test.AddInput<float>("X", {3, 2}, X);
    test.AddOutput<float>("Y", {3, 1}, results);
  } else {
    ASSERT_TRUE(n_obs % 3 == 0);
    xn.resize(n_obs * 2);
    yn.resize(n_obs);
    for (int64_t i = 0; i < n_obs; i += 3) {
      for (size_t k = 0; k < 6; ++k) {
        xn[i * 2 + k] = X[k];
      }
      for (size_t k = 0; k < 3; ++k) {
        yn[i + k] = results[k];
      }
    }
    test.AddInput<float>("X", {n_obs, 2}, xn);
    test.AddOutput<float>("Y", {n_obs, 1}, yn);
  }
  test.Run();
}

TEST(MLOpTest, TreeRegressorSingleTargetSum) {
  GenTreeAndRunTest1("SUM", false);
  GenTreeAndRunTest1("SUM", true);
}

TEST(MLOpTest, TreeRegressorSingleTargetSumBatch) {
  GenTreeAndRunTest1("SUM", false, 201);
  GenTreeAndRunTest1("SUM", false, 40002);
}

TEST(MLOpTest, TreeRegressorSingleTargetBatchTreeA) {
  // TreeEnsemble implements different paths depending on n_trees or N.
  // This test and the next ones goe through all sections for one target.
  GenTreeAndRunTest1("SUM", true, 3, 1);  // section A
}

TEST(MLOpTest, TreeRegressorSingleTargetBatchTreeB) {
  GenTreeAndRunTest1("AVERAGE", true, 3, 30);  // section B
}

TEST(MLOpTest, TreeRegressorSingleTargetBatchTreeC) {
  GenTreeAndRunTest1("AVERAGE", false, 3, 1);  // section C
}

TEST(MLOpTest, TreeRegressorSingleTargetBatchTreeD) {
  GenTreeAndRunTest1("AVERAGE", false, 201, 30);   // section D
  GenTreeAndRunTest1("AVERAGE", false, 201, 130);  // section D
}

TEST(MLOpTest, TreeRegressorSingleTargetBatchTreeE) {
  GenTreeAndRunTest1("AVERAGE", false, 201, 1);  // section E
}

TEST(MLOpTest, TreeRegressorSingleTargetAverage) {
  GenTreeAndRunTest1("AVERAGE", false);
  GenTreeAndRunTest1("AVERAGE", true);
}

TEST(MLOpTest, TreeRegressorSingleTargetMin) {
  GenTreeAndRunTest1("MIN", false);
  GenTreeAndRunTest1("MIN", true);
}

TEST(MLOpTest, TreeRegressorSingleTargetMax) {
  GenTreeAndRunTest1("MAX", false);
  GenTreeAndRunTest1("MAX", true);
}

}  // namespace test
}  // namespace onnxruntime
