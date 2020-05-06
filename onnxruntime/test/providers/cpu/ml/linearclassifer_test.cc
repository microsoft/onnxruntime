// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(MLOpTest, LinearClassifierMulticlass) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> coefficients = {-0.22562418f, 0.34188559f, 0.68346153f, -0.68051993f, -0.1975279f, 0.03748541f};
  std::vector<int64_t> classes = {1, 2, 3};
  int64_t multi_class = 0;
  std::vector<float> X = {1.f, 0.f, 3.f, 44.f, 23.f, 11.3f};

  //three estimates, for 3 points each, so 9 predictions
  std::vector<float> predictions = {-4.14164229f, 1.1092185f, -0.06021539f,
                                    10.45007543f, -27.46673545f, 1.19408663f,
                                    -5.24206713f, 8.45549693f, -3.98224414f};
  std::vector<float> intercepts = {-3.91601811f, 0.42575697f, 0.13731251f};
  std::vector<int64_t> predicted_class = {2, 1, 2};

  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("classlabels_ints", classes);
  test.AddAttribute("multi_class", multi_class);

  test.AddInput<float>("X", {3, 2}, X);
  test.AddOutput<int64_t>("Y", {3}, predicted_class);
  test.AddOutput<float>("Z", {3, 3}, predictions);

  test.Run();
}

TEST(MLOpTest, LinearClassifierMulticlassProb) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> coefficients = {-0.22562418f, 0.34188559f, 0.68346153f,
                                     -0.68051993f, -0.1975279f, 0.03748541f};
  std::vector<int64_t> classes = {1, 2, 3};
  std::vector<float> X = {1.f, 0.f, 3.f, 44.f, 23.f, 11.3f};

  //three estimates, for 3 points each, so 9 predictions
  std::vector<float> predictions = {-4.14164229f, 1.1092185f, -0.06021539f,
                                    10.45007543f, -27.46673545f, 1.19408663f,
                                    -5.24206713f, 8.45549693f, -3.98224414f};
  std::vector<float> intercepts = {-3.91601811f, 0.42575697f, 0.13731251f};
  std::vector<int64_t> predicted_class = {2, 1, 2};

  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("classlabels_ints", classes);

  test.AddInput<float>("X", {3, 2}, X);
  test.AddOutput<int64_t>("Y", {3}, predicted_class);
  test.AddOutput<float>("Z", {3, 3}, predictions);
  test.SetOutputAbsErr("Z", 0.00001f);
  test.Run();
}

TEST(MLOpTest, LinearClassifierMulticlassProbSigmoid) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> coefficients = {-0.22562418f, 0.34188559f, 0.68346153f,
                                     -0.68051993f, -0.1975279f, 0.03748541f};
  std::vector<int64_t> classes = {1, 2, 3};
  std::vector<float> X = {1.f, 0.f, 3.f, 44.f, 23.f, 11.3f};

  //three estimates, for 3 points each, so 9 predictions
  std::vector<float> predictions = {0.015647972f, 0.751983387f, 0.484950699f,
                                    0.999971055f, 1.17855E-12f, 0.767471158f,
                                    0.005261482f, 0.999787317f, 0.018302525f};
  std::vector<float> intercepts = {-3.91601811f, 0.42575697f, 0.13731251f};
  std::vector<int64_t> predicted_class = {2, 1, 2};

  std::string trans("LOGISTIC");
  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("classlabels_ints", classes);
  test.AddAttribute("post_transform", trans);

  test.AddInput<float>("X", {3, 2}, X);
  test.AddOutput<int64_t>("Y", {3}, predicted_class);
  test.AddOutput<float>("Z", {3, 3}, predictions);
  test.SetOutputAbsErr("Z", 0.0001f);
  test.Run();
}

TEST(MLOpTest, LinearClassifierBinary) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> coefficients = {0.00085401f, -0.00314063f};
  std::vector<float> X = {1.f, 0.f, 3.f, 44.f, 23.f, 11.3f};
  std::vector<float> intercepts = {0.03930598f};
  std::vector<int64_t> predicted_class = {1, 0, 1};
  std::vector<float> scores = {0.0401599929f, -0.0963197052f, 0.0234590918f};

  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);

  test.AddInput<float>("X", {3, 2}, X);
  test.AddOutput<int64_t>("Y", {3}, predicted_class);
  test.AddOutput<float>("Z", {3, 1}, scores);
  test.Run();
}

TEST(MLOpTest, LinearClassifierBinaryWithLabels) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> coefficients = {0.00085401f, -0.00314063f};
  std::vector<float> X = {1.f, 0.f, 3.f, 44.f, 23.f, 11.3f};
  std::vector<float> intercepts = {0.03930598f};
  std::vector<std::string> labels = {"not_so_good", "pretty_good"};
  std::vector<std::string> predicted_class = {"pretty_good", "not_so_good", "pretty_good"};
  std::vector<float> scores = {0.959840000f, 0.0401599929f, 1.09631968f, -0.0963197052f, 0.976540923f, 0.0234590918f};

  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("classlabels_strings", labels);

  test.AddInput<float>("X", {3, 2}, X);
  test.AddOutput<std::string>("Y", {3}, predicted_class);
  test.AddOutput<float>("Z", {3, 2}, scores);
  test.Run();
}

template <typename T>
void LinearClassifierMulticlass() {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> coefficients = {-0.22562418f, 0.34188559f, 0.68346153f,
                                     -0.68051993f, -0.1975279f, 0.03748541f};
  std::vector<int64_t> classes = {1, 2, 3};
  int64_t multi_class = 0;
  std::vector<T> X = {1, 0, 3, 44, 23, 11};

  //three estimates, for 3 points each, so 9 predictions
  std::vector<float> predictions = {-4.14164229f, 1.1092185f, -0.06021539f,
                                    10.45007543f, -27.46673545f, 1.19408663f,
                                    -5.3446321487426758f, 8.6596536636352539f, -3.9934897422790527};
  std::vector<float> intercepts = {-3.91601811f, 0.42575697f, 0.13731251f};
  std::vector<int64_t> predicted_class = {2, 1, 2};

  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("classlabels_ints", classes);
  test.AddAttribute("multi_class", multi_class);

  test.AddInput<T>("X", {3, 2}, X);
  test.AddOutput<int64_t>("Y", {3}, predicted_class);
  test.AddOutput<float>("Z", {3, 3}, predictions);

  test.Run();
}

TEST(MLOpTest, LinearClassifierMulticlassInt64Input) {
  LinearClassifierMulticlass<int64_t>();
}

TEST(MLOpTest, LinearClassifierMulticlassInt32Input) {
  LinearClassifierMulticlass<int32_t>();
}

TEST(MLOpTest, LinearClassifierMulticlassDoubleInput) {
  LinearClassifierMulticlass<double>();
}
}  // namespace test
}  // namespace onnxruntime
