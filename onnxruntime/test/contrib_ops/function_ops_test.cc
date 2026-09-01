// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/graph/contrib_ops/contrib_defs.h"
#include "test/unittest_util/function_test_util.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {
namespace test {

static void RegisterSchemas() {
  static bool registered = false;
  if (!registered) {
    onnxruntime::contrib::RegisterContribSchemas();
    registered = true;
  }
}

class ContribFunExpansionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    RegisterSchemas();
  }
};

template <typename T, typename U, bool RunTest>
void CheckLayerNorm(bool compute_mean = true, bool compute_isd = true, std::vector<int64_t> shape1 = {8, 16}, std::vector<int64_t> shape2 = {16}, int64_t axis = -1) {
  FunctionTestCase testCase("LayerNormalization", kOnnxDomain);

  testCase.AddInput<T, RunTest>("x", shape1);
  testCase.AddInput<T, RunTest>("scale", shape2);
  testCase.AddInput<T, RunTest>("bias", shape2);
  testCase.AddOutput("y");
  testCase.AddOutput(compute_mean ? "mean" : "");
  testCase.AddOutput(compute_isd ? "invstddev" : "");
  testCase.AddAttribute("stash_type", utils::ToTensorProtoElementType<U>());
  if (axis != -1)
    testCase.AddAttribute("axis", axis);
  if (RunTest)
    testCase.RunTest();
  else
    testCase.CreateModel(true);
}

TEST_F(ContribFunExpansionTest, LayerNorm) {
  // Test expand-and-run
  CheckLayerNorm<float, float, true>();
  // Test expand-and-check-only
  CheckLayerNorm<MLFloat16, double, false>();
}

TEST_F(ContribFunExpansionTest, LayerNorm_OptionalOutputs) {
  // Test expand-and-run
  CheckLayerNorm<float, float, true>(false, false);
  CheckLayerNorm<float, float, true>(false, true);
  CheckLayerNorm<float, float, true>(true, false);
}

TEST_F(ContribFunExpansionTest, LayerNorm_OtherShapes) {
  // Test expand-and-run
  CheckLayerNorm<float, float, true>(true, true, {4, 2, 8}, {2, 8}, 1);
}

template <typename T>
void CheckGelu() {
  FunctionTestCase testCase("Gelu", kMSDomain);
  std::vector<int64_t> shape{8, 16};

  testCase.AddInput<T>("x", shape);
  testCase.AddOutput("y");

  // Only check expanded graph. Can't run it yet because no implementation of Erf is available yet.
  testCase.CreateModel(true);
}

TEST_F(ContribFunExpansionTest, Gelu) {
  CheckGelu<float>();
  CheckGelu<double>();
  CheckGelu<BFloat16>();
  CheckGelu<MLFloat16>();
}

template <typename T, bool RunTest = true>
void CheckFastGelu(bool withBias = true) {
  FunctionTestCase testCase("FastGelu", kMSDomain);
  std::vector<int64_t> shape{8, 16};
  std::vector<int64_t> bias_shape{16};

  testCase.AddInput<T, RunTest>("x", shape);
  if (withBias) {
    testCase.AddInput<T, RunTest>("bias", bias_shape);
  }
  testCase.AddOutput("y");

  if (RunTest)
    testCase.RunTest();
  else
    testCase.CreateModel(true);
}

TEST_F(ContribFunExpansionTest, FastGeluWithBias) {
  CheckFastGelu<float>(true);
  CheckFastGelu<BFloat16, false>(true);
  CheckFastGelu<MLFloat16, false>(true);
}

TEST_F(ContribFunExpansionTest, FastGeluWithoutBias) {
  CheckFastGelu<float>(false);
  CheckFastGelu<BFloat16, false>(false);
  CheckFastGelu<MLFloat16, false>(false);
}

void CheckSpaceDepthFunction(const char* op_type, std::vector<int64_t> input_shape,
                             int64_t blocksize, const char* mode) {
  FunctionTestCase test_case(op_type, kOnnxDomain);
  test_case.AddOpset(kOnnxDomain, 28);
  test_case.AddInput<float>("input", input_shape);
  test_case.AddOutput("output");
  test_case.AddAttribute("blocksize", blocksize);
  test_case.AddAttribute("mode", mode);
  test_case.RunTest();
}

TEST_F(ContribFunExpansionTest, SpaceDepthOpset28FunctionParity) {
  for (const auto* mode : {"DCR", "CRD"}) {
    CheckSpaceDepthFunction("SpaceToDepth", {1, 2, 4, 6}, 2, mode);
    CheckSpaceDepthFunction("SpaceToDepth", {1, 2, 6, 9}, 3, mode);
    CheckSpaceDepthFunction("DepthToSpace", {1, 8, 2, 3}, 2, mode);
    CheckSpaceDepthFunction("DepthToSpace", {1, 18, 2, 3}, 3, mode);
  }
}

}  // namespace test
}  // namespace onnxruntime
