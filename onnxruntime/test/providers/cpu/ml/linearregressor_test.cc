// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(MLOpTest, LinearRegressorUniTarget) {
  /* TEST MODEL TRAINING
    from sklearn import linear_model
    X = [[0., 0.5], [1., 1.5], [2., 2.9], [3., 13.3]]
    Z = [[41.], [32.], [23.], [14.]]
    model = linear_model.LinearRegression()
    r3 =model.fit(X, Z)
    r4 =model.predict ([[1, 0.],[3.,44.],[23.,11.3]])
    r3.coef_
    array([-9.00000000e+00,  -1.99600736e-16])

    r3.intercept_
    array([4.10000000e+01])

    r4
    array([[32.], [14.], [-166.]])
  */

  OpTester test("LinearRegressor", 1, onnxruntime::kMLDomain);
  std::vector<float> coefficients = {-9.00000000f, -1.99600736e-16f};
  std::vector<float> intercepts = {41.0000000f};
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("coefficients", coefficients);

  test.AddInput<float>("X", {3, 2}, {1.f, 0.f, 3.f, 44.f, 23.f, 11.3f});
  test.AddOutput<float>("Y", {3, 1}, {32.0f, 14.0f, -166.0f});
  test.Run();
}

TEST(MLOpTest, LinearRegressorMultiTarget) {
  /* TEST MODEL TRAINING
    from sklearn import linear_model
    X = [[0., 0.5], [1., 1.5], [2., 2.9], [3., 13.3]]
    Z = [[0., 41.], [1., 32.], [2., 23.], [3., 14.]]
    model = linear_model.LinearRegression()
    r3 =model.fit(X, Z)
    r4 =model.predict ([[1, 0.],[3.,44.],[23.,11.3]])
    r3.coef_
    array([[  1.00000000e+00,  -2.49500920e-17],
    [ -9.00000000e+00,  -1.99600736e-16]])
    r3.intercept_
    array([  2.22044605e-16,   4.10000000e+01])
    r4
    array([[   1.,   32.], [   3.,   14.], [  23., -166.]])
  */

  OpTester test("LinearRegressor", 1, onnxruntime::kMLDomain);
  std::vector<float> coefficients = {1.00000000f, -2.49500920e-17f, -9.00000000f, -1.99600736e-16f};
  std::vector<float> intercepts = {2.22044605e-16f, 41.0000000f};
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("coefficients", coefficients);
  int64_t targets = 2;
  test.AddAttribute("targets", targets);

  test.AddInput<float>("X", {3, 2}, {1.f, 0.f, 3.f, 44.f, 23.f, 11.3f});
  test.AddOutput<float>("Y", {3, 2}, {1.0f, 32.0f, 3.0f, 14.0f, 23.0f, -166.0f});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
