// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

struct LinearRegressorParam {
  const char* post_transform;
  std::vector<float> expected_value;
  int64_t targets;
  LinearRegressorParam(const char* post_transform1, const std::vector<float>& expected_value1, int targets1)
      : post_transform(post_transform1), expected_value(expected_value1), targets(targets1) {}
};

class LinearRegressorTest : public testing::TestWithParam<LinearRegressorParam> {};

/* TEST MODEL TRAINING
    from sklearn import linear_model
    X = [[0., 0.5], [1., 1.5], [2., 2.9], [3., 13.3]]
    Z = [[41.], [32.], [23.], [14.]]
    model = linear_model.LinearRegression()
    r3 =model.fit(X, Z)
    r4 =model.predict ([[1, 0.],[3.,44.],[23.,11.3]])
    r3.coef_
    array([-9.00000000e+00, -1.99600736e-16])

    r3.intercept_
    array([4.10000000e+01])

    r4
    array([[32.],[14.],[-166.]])
  */
/* TEST MODEL TRAINING
from sklearn import linear_model
X = [[0., 0.5], [1., 1.5], [2., 2.9], [3., 13.3]]
Z = [[0., 41.], [1., 32.], [2., 23.], [3., 14.]]
model = linear_model.LinearRegression()
r3 =model.fit(X, Z)
r4 =model.predict ([[1, 0.],[3.,44.],[23.,11.3]])
r3.coef_
array([[ 1.00000000e+00, -2.49500920e-17],
[ -9.00000000e+00, -1.99600736e-16]])
r3.intercept_
array([ 2.22044605e-16, 4.10000000e+01])
r4
array([[ 1., 32.],[ 3., 14.],[ 23., -166.]])
*/
TEST_P(LinearRegressorTest, LinearRegressorUniTarget) {
  const LinearRegressorParam& param = GetParam();
  OpTester test("LinearRegressor", 1, onnxruntime::kMLDomain);
  std::vector<float> coefficients, intercepts;
  if (param.targets == 1) {
    coefficients = {-9.00000000f, -1.99600736e-16f};
    intercepts = {41.0000000f};
  } else {
    coefficients = {1.00000000f, -2.49500920e-17f, -9.00000000f, -1.99600736e-16f};
    intercepts = {2.22044605e-16f, 41.0000000f};
  }
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("coefficients", coefficients);
  if (strcmp(param.post_transform, "NONE") != 0) {
    test.AddAttribute("post_transform", param.post_transform);
  }
  test.AddAttribute("targets", param.targets);

  test.AddInput<float>("X", {3, 2}, {1.f, 0.f, 3.f, 44.f, 23.f, 11.3f});
  test.AddOutput<float>("Y", {static_cast<int64_t>(param.expected_value.size() / param.targets), param.targets},
                        param.expected_value);
  test.Run();
}

// For PROBIT, all the output values are NaN.
INSTANTIATE_TEST_SUITE_P(
    LinearRegressorTest, LinearRegressorTest,
    testing::Values(LinearRegressorParam("NONE", {32.0f, 14.0f, -166.0f}, 1),
                    LinearRegressorParam("SOFTMAX", {32.0f, 14.0f, -166.0f}, 1),
                    LinearRegressorParam("LOGISTIC", {32.0f, 14.0f, -166.0f}, 1),
                    LinearRegressorParam("SOFTMAX_ZERO", {32.0f, 14.0f, -166.0f}, 1),
                    LinearRegressorParam("NONE", {1.0f, 32.0f, 3.0f, 14.0f, 23.0f, -166.0f}, 2),
                    LinearRegressorParam("SOFTMAX", {3.442477e-14f, 1.f, 1.670142e-05f, 1.f, 1.0f, 0.f}, 2),
                    LinearRegressorParam("LOGISTIC", {0.731058f, 1.0f, 0.9525741f, 1.f, 1.0f, 0.f}, 2),
                    LinearRegressorParam("SOFTMAX_ZERO", {3.442477e-14f, 1.f, 1.670142e-05f, 1.f, 1.0f, 0.f}, 2)

                        ));
}  // namespace test
}  // namespace onnxruntime
