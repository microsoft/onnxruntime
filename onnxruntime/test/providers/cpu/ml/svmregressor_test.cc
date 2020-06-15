// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(MLOpTest, SVMRegressorSVC) {
  OpTester test("SVMRegressor", 1, onnxruntime::kMLDomain);

  std::vector<float> dual_coefficients = {-1.54236563f, 0.53485162f, -1.5170623f, 0.69771864f, 1.82685767f};
  std::vector<float> support_vectors = {0.f, 0.5f, 32.f, 1.f, 1.5f, 1.f, 2.f, 2.9f, -32.f, 12.f, 12.9f, -312.f, 43.f, 413.3f, -114.f};
  std::vector<float> rho = {1.96292297f};
  std::vector<float> kernel_params = {0.001f, 0.f, 3.f};  //gamma, coef0, degree

  //three estimates, for 3 points each, so 9 predictions
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<float> predictions = {1.40283655f, 1.86065906f, 2.66064161f, 1.96311014f, 1.96311014f, 1.96292297f, 1.96311014f, 3.78978065f};

  test.AddAttribute("kernel_type", std::string("RBF"));
  test.AddAttribute("coefficients", dual_coefficients);
  test.AddAttribute("support_vectors", support_vectors);
  test.AddAttribute("rho", rho);
  test.AddAttribute("kernel_params", kernel_params);
  test.AddAttribute("n_supports", static_cast<int64_t>(5));

  test.AddInput<float>("X", {8, 3}, X);
  test.AddOutput<float>("Y", {8, 1}, predictions);

  test.Run();
}

TEST(MLOpTest, SVMRegressorNuSVC) {
  OpTester test("SVMRegressor", 1, onnxruntime::kMLDomain);

  std::vector<float> dual_coefficients = {-1.7902966f, 1.05962596f, -1.54324389f, -0.43658884f, 0.79025169f, 1.92025169f};
  std::vector<float> support_vectors = {0.f, 0.5f, 32.f, 1.f, 1.5f, 1.f, 2.f, 2.9f, -32.f, 3.f, 13.3f, -11.f, 12.f, 12.9f, -312.f, 43.f, 413.3f, -114.f};
  std::vector<float> rho = {1.96923464f};
  std::vector<float> kernel_params = {0.001f, 0.f, 3.f};  //gamma, coef0, degree

  //three estimates, for 3 points each, so 9 predictions
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<float> predictions = {1.51230766f, 1.77893206f, 2.75948633f, 1.96944663f, 1.96944663f, 1.96923464f, 1.96944663f, 3.88948633f};

  test.AddAttribute("kernel_type", std::string("RBF"));
  test.AddAttribute("coefficients", dual_coefficients);
  test.AddAttribute("support_vectors", support_vectors);
  test.AddAttribute("rho", rho);
  test.AddAttribute("kernel_params", kernel_params);
  test.AddAttribute("n_supports", static_cast<int64_t>(6));

  test.AddInput<float>("X", {8, 3}, X);
  test.AddOutput<float>("Y", {8, 1}, predictions);

  test.Run();
}

TEST(MLOpTest, SVMRegressorNuSVCPolyKernel) {
  OpTester test("SVMRegressor", 1, onnxruntime::kMLDomain);

  std::vector<float> dual_coefficients = {-2.74322388e+01f, 5.81893108e+01f, -1.00000000e+02f,
                                          6.91693781e+01f, 7.62161261e-02f, -2.66618042e-03f};
  std::vector<float> support_vectors = {0.f, 0.5f, 32.f,
                                        1.f, 1.5f, 1.f,
                                        2.f, 2.9f, -32.f,
                                        3.f, 13.3f, -11.f,
                                        12.f, 12.9f, -312.f,
                                        43.f, 413.3f, -114.f};
  std::vector<float> rho = {1.5004596f};
  std::vector<float> kernel_params = {0.001f, 0.f, 3.f};  //gamma, coef0, degree

  // 8 batches with 3 features in each
  std::vector<float> X = {1.f, 0.0f, 0.4f,
                          3.0f, 44.0f, -3.f,
                          12.0f, 12.9f, 112.f,
                          23.0f, 11.3f, -222.f,
                          23.0f, 11.3f, -222.f,
                          23.0f, 3311.3f, -222.f,
                          23.0f, 11.3f, -222.f,
                          43.0f, 413.3f, -114.f};

  /*
  # batched_dot_product calculations
  first = np.matmul(X, np.transpose(support_vectors))
  first *= gamma
  first += coef0
  POLY_dot_product = np.power(first, degree)

  # second GEMM call in SVMRegressor code
  predictions = np.matmul(POLY_dot_product, np.transpose(coefficients))
  predictions += rho
  */
  std::vector<float> predictions = {1.50041862e+00f,
                                    3.49624789e-01f,
                                    -1.36680453e+02f,
                                    -2.28659315e+02f,
                                    -2.28659315e+02f,
                                    -6.09640827e+05f,
                                    -2.28659315e+02f,
                                    3.89055458e+00f};

  test.AddAttribute("kernel_type", std::string("POLY"));
  test.AddAttribute("coefficients", dual_coefficients);
  test.AddAttribute("support_vectors", support_vectors);
  test.AddAttribute("rho", rho);
  test.AddAttribute("kernel_params", kernel_params);
  test.AddAttribute("n_supports", static_cast<int64_t>(6));

  test.AddInput<float>("X", {8, 3}, X);
  test.AddOutput<float>("Y", {8, 1}, predictions);
  test.SetOutputRelErr("Y", 0.01f);
  test.Run();
}

TEST(MLOpTest, SVMRegressorLinear) {
  OpTester test("SVMRegressor", 1, onnxruntime::kMLDomain);
  std::vector<float> coefficients = {0.28290501f, -0.0266512f, 0.01674867f};
  std::vector<float> rho = {1.24032312f};
  std::vector<float> kernel_params = {0.001f, 0.f, 3.f};  //gamma, coef0, degree

  //three estimates, for 3 points each, so 9 predictions
  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<float> predictions = {1.52992759f, 0.8661395f, -0.93420165f, 3.72777548f, 3.72777548f, -84.22117216f, 3.72777548f, 0.48095091f};

  test.AddAttribute("kernel_type", std::string("LINEAR"));
  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("rho", rho);
  test.AddAttribute("kernel_params", kernel_params);
  test.AddAttribute("n_supports", static_cast<int64_t>(0));

  test.AddInput<float>("X", {8, 3}, X);
  test.AddOutput<float>("Y", {8, 1}, predictions);

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
