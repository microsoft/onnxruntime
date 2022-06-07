// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(MLOpTest, SVMClassifierMulticlassSVC) {
  OpTester test("SVMClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> dual_coefficients = {1.14360327f, 1.95968249f, -1.175683f, -1.92760275f, -1.32575698f,
                                          -1.32575698f, 0.66332785f, 0.66242913f, 0.53120854f, 0.53510444f,
                                          -1.06631298f, -1.06631298f, 0.66332785f, 0.66242913f, 0.53120854f,
                                          0.53510444f, 1.f, -1.f};
  std::vector<float> support_vectors = {0.f, 0.5f, 32.f, 2.f, 2.9f, -32.f, 1.f, 1.5f, 1.f, 3.f,
                                        13.3f, -11.f, 12.f, 12.9f, -312.f, 43.f, 413.3f, -114.f};
  std::vector<int64_t> classes = {0, 1, 2, 3};
  std::vector<int64_t> vectors_per_class = {2, 2, 1, 1};
  std::vector<float> rho = {0.5279583f, 0.32605162f, 0.32605162f, 0.06663721f, 0.06663721f, 0.f};
  std::vector<float> kernel_params = {0.001f, 0.f, 3.f};  //gamma, coef0, degree

  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f,
                          11.3f, -222.f, 23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f, 23.0f,
                          11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<int64_t> predictions = {1, 1, 2, 0, 0, 0, 0, 3};
  std::vector<float> scores = {
      -0.956958294f, 0.799815655f, 0.799815655f, 0.988598406f, 0.988598406f, 0,
      -0.159782529f, 0.407864451f, 0.407864451f, 0.347750872f, 0.347750872f, 0,
      0.527958274f, -0.999705434f, 0.326051623f, -0.999675810f, 0.0666372105f, 1.00000000f,
      0.527958274f, 0.325695992f, 0.326051623f, 0.0663511604f, 0.0666372105f, 0.000268258271f,
      0.527958274f, 0.325695992f, 0.326051623f, 0.0663511604f, 0.0666372105f, 0.000268258271f,
      0.527958274f, 0.326051623f, 0.326051623f, 0.0666372105f, 0.0666372105f, 0,
      0.527958274f, 0.325695992f, 0.326051623f, 0.0663511604f, 0.0666372105f, 0.000268258271f,
      0.527958274f, 0.326051623f, -0.999705434f, 0.0666372105f, -0.999675810f, -1.00000000f};

  test.AddAttribute("kernel_type", std::string("RBF"));
  test.AddAttribute("coefficients", dual_coefficients);
  test.AddAttribute("support_vectors", support_vectors);
  test.AddAttribute("vectors_per_class", vectors_per_class);
  test.AddAttribute("rho", rho);
  test.AddAttribute("kernel_params", kernel_params);
  test.AddAttribute("classlabels_ints", classes);

  test.AddInput<float>("X", {8, 3}, X);
  test.AddOutput<int64_t>("Y", {8}, predictions);
  test.AddOutput<float>("Z", {8, 6}, scores);

  test.Run();
}

TEST(MLOpTest, SVMClassifierMulticlassLinearSVC) {
  OpTester test("SVMClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> dual_coefficients = {-1.55181212e-01f, 2.42698956e-01f, 7.01893432e-03f,
                                          4.07614474e-01f, -3.24927823e-02f, 2.79897536e-04f,
                                          -1.95771302e-01f, -3.52437368e-01f, -2.15973096e-02f,
                                          -4.38190277e-01f, 4.56869105e-02f, -1.29375499e-02f};
  std::vector<int64_t> classes = {0, 1, 2, 3};
  std::vector<float> rho = {-0.07489691f, -0.1764396f, -0.21167431f, -0.51619097f};
  std::vector<float> kernel_params = {0.001f, 0.f, 3.f};  //gamma, coef0, degree

  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f,
                          12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f,
                          23.0f, 11.3f, -222.f, 23.0f, 3311.3f, -222.f,
                          23.0f, 11.3f, -222.f, 43.0f, 413.3f, -114.f};
  std::vector<int64_t> predictions = {1, 0, 1, 1, 1, 0, 1, 0};
  std::vector<float> scores = {
      -0.227270544f, 0.332829535f, -0.279307127f, -0.518262208f,
      10.1172562f, -0.282575697f, -16.1046638f, 0.659568906f,
      -0.996162534f, 4.30999184f, -0.232234091f, -0.707304120f,
      -2.45976996f, 8.87092972f, -3.76557732f, -6.76487541f,
      -2.45976996f, 8.87092972f, -3.76557732f, -6.76487541f,
      798.446777f, -98.3552551f, -1166.80896f, 144.001923f,
      -2.45976996f, 8.87092972f, -3.76557732f, -6.76487541f,
      92.7596283f, 3.99134970f, -151.693329f, 1.44020212f};

  test.AddAttribute("kernel_type", std::string("LINEAR"));
  test.AddAttribute("coefficients", dual_coefficients);
  test.AddAttribute("rho", rho);
  test.AddAttribute("kernel_params", kernel_params);
  test.AddAttribute("classlabels_ints", classes);

  test.AddInput<float>("X", {8, 3}, X);
  test.AddOutput<int64_t>("Y", {8}, predictions);
  test.AddOutput<float>("Z", {8, 4}, scores);
  test.SetOutputRelErr("Z", 0.00001f);

  test.Run();
}

TEST(MLOpTest, SVMClassifierSVCProbabilities) {
  OpTester test("SVMClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> coefficients = {1.14360327f, 1.95968249f, -1.175683f, -1.92760275f, -1.32575698f, -1.32575698f,
                                     0.66332785f, 0.66242913f, 0.53120854f, 0.53510444f, -1.06631298f, -1.06631298f,
                                     0.66332785f, 0.66242913f, 0.53120854f, 0.53510444f, 1.f, -1.f};
  // 6 support vectors x 3 features
  std::vector<float> support_vectors = {0.f, 0.5f, 32.f,
                                        2.f, 2.9f, -32.f,
                                        1.f, 1.5f, 1.f,
                                        3.f, 13.3f, -11.f,
                                        12.f, 12.9f, -312.f,
                                        43.f, 413.3f, -114.f};

  std::vector<float> rho = {0.5279583f, 0.32605162f, 0.32605162f, 0.06663721f, 0.06663721f, 0.f};
  std::vector<float> kernel_params = {0.001f, 0.f, 3.f};  //gamma, coef0, degree
  std::vector<float> proba = {-3.8214362f, 1.82177748f, 1.82177748f, 7.17655643f, 7.17655643f, 0.69314718f};
  std::vector<float> probb = {-1.72839673e+00f, -1.12863030e+00f, -1.12863030e+00f, -6.48340925e+00f, -6.48340925e+00f, 2.39189538e-16f};
  std::vector<int64_t> classes = {0, 1, 2, 3};
  std::vector<int64_t> vectors_per_class = {2, 2, 1, 1};

  std::vector<float> X = {1.f, 0.0f, 0.4f, 3.0f, 44.0f, -3.f, 12.0f, 12.9f, -312.f, 23.0f, 11.3f, -222.f, 23.0f, 11.3f, -222.f};
  std::vector<float> prob_predictions = {
      0.13766955f, 0.21030431f, 0.32596754f, 0.3260586f,
      0.45939931f, 0.26975416f, 0.13539588f, 0.13545066f,
      0.71045899f, 0.07858939f, 0.05400437f, 0.15694726f,
      0.58274772f, 0.10203105f, 0.15755227f, 0.15766896f,
      0.58274772f, 0.10203105f, 0.15755227f, 0.15766896f};
  std::vector<int64_t> class_predictions = {1, 1, 2, 0, 0};

  test.AddAttribute("kernel_type", std::string("RBF"));
  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("support_vectors", support_vectors);
  test.AddAttribute("vectors_per_class", vectors_per_class);
  test.AddAttribute("rho", rho);
  test.AddAttribute("kernel_params", kernel_params);
  test.AddAttribute("classlabels_ints", classes);
  test.AddAttribute("prob_a", proba);
  test.AddAttribute("prob_b", probb);

  test.AddInput<float>("X", {5, 3}, X);
  test.AddOutput<int64_t>("Y", {5}, class_predictions);
  test.AddOutput<float>("Z", {5, 4}, prob_predictions);

  test.Run();
}

TEST(MLOpTest, SVMClassifierSVC) {
  OpTester test("SVMClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> coefficients = {1.14360327f, 1.95968249f, -1.175683f, -1.92760275f, -1.32575698f, -1.32575698f,
                                     0.66332785f, 0.66242913f, 0.53120854f, 0.53510444f, -1.06631298f, -1.06631298f,
                                     0.66332785f, 0.66242913f, 0.53120854f, 0.53510444f, 1.f, -1.f};
  std::vector<float> support_vectors = {0.f, 0.5f, 32.f, 2.f, 2.9f, -32.f,
                                        1.f, 1.5f, 1.f, 3.f, 13.3f, -11.f,
                                        12.f, 12.9f, -312.f, 43.f, 413.3f, -114.f};
  std::vector<float> rho = {0.5279583f};
  std::vector<float> kernel_params = {0.001f, 0.f, 3.f};  //gamma, coef0, degree
  std::vector<int64_t> classes = {0, 1};
  std::vector<int64_t> vectors_per_class = {3, 3};

  std::vector<float> X = {1.f, 0.0f, 0.4f,
                          3.0f, 44.0f, -3.f,
                          12.0f, 12.9f, -312.f,
                          23.0f, 11.3f, -222.f,
                          23.0f, 11.3f, -222.f};
  std::vector<float> scores_predictions = {
      0.95695829391479492f, -0.95695829391479492f,
      0.1597825288772583f, -0.1597825288772583f,
      0.797798752784729f, -0.797798752784729f,
      -0.52760261297225952f, 0.52760261297225952f,
      -0.52760261297225952f, 0.52760261297225952f};
  std::vector<int64_t> class_predictions = {1, 1, 1, 0, 0};

  test.AddAttribute("kernel_type", std::string("RBF"));
  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("support_vectors", support_vectors);
  test.AddAttribute("vectors_per_class", vectors_per_class);
  test.AddAttribute("rho", rho);
  test.AddAttribute("kernel_params", kernel_params);
  test.AddAttribute("classlabels_ints", classes);

  test.AddInput<float>("X", {5, 3}, X);
  test.AddOutput<int64_t>("Y", {5}, class_predictions);
  test.AddOutput<float>("Z", {5, 2}, scores_predictions);

  test.Run();
}

TEST(MLOpTest, SVMClassifierSVCDouble) {
  OpTester test("SVMClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> coefficients = {1.14360327f, 1.95968249f, -1.175683f, -1.92760275f, -1.32575698f, -1.32575698f,
                                     0.66332785f, 0.66242913f, 0.53120854f, 0.53510444f, -1.06631298f, -1.06631298f,
                                     0.66332785f, 0.66242913f, 0.53120854f, 0.53510444f, 1.f, -1.f};
  std::vector<float> support_vectors = {0.f, 0.5f, 32.f, 2.f, 2.9f, -32.f,
                                        1.f, 1.5f, 1.f, 3.f, 13.3f, -11.f,
                                        12.f, 12.9f, -312.f, 43.f, 413.3f, -114.f};
  std::vector<float> rho = {0.5279583f};
  std::vector<float> kernel_params = {0.001f, 0.f, 3.f};  //gamma, coef0, degree
  std::vector<int64_t> classes = {0, 1};
  std::vector<int64_t> vectors_per_class = {3, 3};

  std::vector<double> X = {1.f, 0.0f, 0.4f,
                           3.0f, 44.0f, -3.f,
                           12.0f, 12.9f, -312.f,
                           23.0f, 11.3f, -222.f,
                           23.0f, 11.3f, -222.f};
  std::vector<float> scores_predictions = {
      0.95695829391479492f, -0.95695829391479492f,
      0.1597825288772583f, -0.1597825288772583f,
      0.797798752784729f, -0.797798752784729f,
      -0.52760261297225952f, 0.52760261297225952f,
      -0.52760261297225952f, 0.52760261297225952f};
  std::vector<int64_t> class_predictions = {1, 1, 1, 0, 0};

  test.AddAttribute("kernel_type", std::string("RBF"));
  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("support_vectors", support_vectors);
  test.AddAttribute("vectors_per_class", vectors_per_class);
  test.AddAttribute("rho", rho);
  test.AddAttribute("kernel_params", kernel_params);
  test.AddAttribute("classlabels_ints", classes);

  test.AddInput<double>("X", {5, 3}, X);
  test.AddOutput<int64_t>("Y", {5}, class_predictions);
  test.AddOutput<float>("Z", {5, 2}, scores_predictions);

  test.Run();
}

TEST(MLOpTest, SVMClassifierLinear) {
  OpTester test("SVMClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> coefficients = {0.766398549079895f, 0.0871576070785522f, 0.110420741140842f,
                                     -0.963976919651031f};
  std::vector<float> support_vectors = {4.80000019073486f, 3.40000009536743f, 1.89999997615814f,
                                        5.f, 3.f, 1.60000002384186f,
                                        4.5f, 2.29999995231628f, 1.29999995231628f,
                                        5.09999990463257f, 2.5f, 3.f};
  std::vector<float> rho = {2.23510527610779f};
  std::vector<float> kernel_params = {0.122462183237076f, 0.f, 3.f};  //gamma, coef0, degree
  std::vector<int64_t> classes = {0, 1};
  std::vector<int64_t> vectors_per_class = {3, 1};

  std::vector<float> X = {5.1f, 3.5f, 1.4f,
                          4.9f, 3.f, 1.4f,
                          4.7f, 3.2f, 1.3f,
                          4.6f, 3.1f, 1.5f,
                          5.f, 3.6f, 1.4f};
  std::vector<float> scores_predictions = {-1.5556798f, 1.5556798f,
                                           -1.2610321f, 1.2610321f,
                                           -1.5795376f, 1.5795376f,
                                           -1.3083477f, 1.3083477f,
                                           -1.6572928f, 1.6572928f};

  std::vector<int64_t> class_predictions = {0, 0, 0, 0, 0};

  test.AddAttribute("kernel_type", std::string("LINEAR"));
  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("support_vectors", support_vectors);
  test.AddAttribute("vectors_per_class", vectors_per_class);
  test.AddAttribute("rho", rho);
  test.AddAttribute("kernel_params", kernel_params);
  test.AddAttribute("classlabels_ints", classes);

  test.AddInput<float>("X", {5, 3}, X);
  test.AddOutput<int64_t>("Y", {5}, class_predictions);
  test.AddOutput<float>("Z", {5, 2}, scores_predictions);

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
