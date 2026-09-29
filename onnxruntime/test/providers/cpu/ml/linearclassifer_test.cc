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

  // three estimates, for 3 points each, so 9 predictions
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

  // three estimates, for 3 points each, so 9 predictions
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

  // three estimates, for 3 points each, so 9 predictions
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

#if !defined(ORT_NO_EXCEPTIONS)
// coefficients size (3) is not a multiple of class_count (2) - caught at construction time.
TEST(MLOpTest, LinearClassifierInvalidCoefficientsSize) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  test.AddAttribute("coefficients", std::vector<float>{1.f, 2.f, 3.f});
  test.AddAttribute("intercepts", std::vector<float>{0.f, 0.f});
  test.AddAttribute("classlabels_ints", std::vector<int64_t>{0, 1});

  test.AddInput<float>("X", {1, 2}, {1.f, 2.f});
  test.AddOutput<int64_t>("Y", {1}, {0});
  test.AddOutput<float>("Z", {1, 2}, {0.f, 0.f});

  test.Run(OpTester::ExpectResult::kExpectFailure,
           "coefficients size (3) must be a multiple of the number of classes (2)");
}
#endif  // !defined(ORT_NO_EXCEPTIONS)

template <typename T>
void LinearClassifierMulticlass() {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> coefficients = {-0.22562418f, 0.34188559f, 0.68346153f,
                                     -0.68051993f, -0.1975279f, 0.03748541f};
  std::vector<int64_t> classes = {1, 2, 3};
  int64_t multi_class = 0;
  std::vector<T> X = {1, 0, 3, 44, 23, 11};

  // three estimates, for 3 points each, so 9 predictions
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

// Regression test: coefficients size doesn't match class_count * num_features.
TEST(MLOpTest, LinearClassifierInvalidCoefficientsSizeFails) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  // 3 intercepts => class_count = 3, input has 2 features => expects 6 coefficients.
  std::vector<float> coefficients = {-0.22562418f, 0.34188559f, 0.68346153f};
  std::vector<int64_t> classes = {1, 2, 3};
  std::vector<float> intercepts = {-3.91601811f, 0.42575697f, 0.13731251f};

  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("classlabels_ints", classes);

  test.AddInput<float>("X", {1, 2}, {1.f, 0.f});
  test.AddOutput<int64_t>("Y", {1}, {0LL});
  test.AddOutput<float>("Z", {1, 3}, {0.f, 0.f, 0.f});

  test.Run(OpTester::ExpectResult::kExpectFailure,
           "coefficients size (3) is less than class_count (3) * num_features (2)");
}

TEST(MLOpTest, LinearClassifierExtraCoefficientsAreIgnored) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> coefficients = {-0.22562418f, 0.34188559f, 0.68346153f,
                                     -0.68051993f, -0.1975279f, 0.03748541f,
                                     101.f, 102.f, 103.f};
  std::vector<int64_t> classes = {1, 2, 3};
  std::vector<float> intercepts = {-3.91601811f, 0.42575697f, 0.13731251f};

  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("classlabels_ints", classes);

  test.AddInput<float>("X", {1, 2}, {1.f, 0.f});
  test.AddOutput<int64_t>("Y", {1}, {2LL});
  test.AddOutput<float>("Z", {1, 3}, {-4.14164229f, 1.1092185f, -0.06021539f});

  test.Run();
}

#if !defined(ORT_NO_EXCEPTIONS)
// Regression test: coefficients not divisible by class_count - caught at construction time.
TEST(MLOpTest, LinearClassifierCoefficientsSizeNotDivisibleByClassCountFails) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  // 3 intercepts => class_count = 3, but 5 coefficients is not divisible by 3.
  std::vector<float> coefficients = {1.f, 2.f, 3.f, 4.f, 5.f};
  std::vector<int64_t> classes = {1, 2, 3};
  std::vector<float> intercepts = {0.1f, 0.2f, 0.3f};

  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("classlabels_ints", classes);

  test.AddInput<float>("X", {1, 2}, {1.f, 0.f});
  test.AddOutput<int64_t>("Y", {1}, {0LL});
  test.AddOutput<float>("Z", {1, 3}, {0.f, 0.f, 0.f});

  test.Run(OpTester::ExpectResult::kExpectFailure,
           "coefficients size (5) must be a multiple of the number of classes (3)");
}
#endif  // !defined(ORT_NO_EXCEPTIONS)

TEST(MLOpTest, LinearClassifierInputFeatureCountMismatchFails) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> coefficients = {-0.22562418f, 0.34188559f, 0.68346153f,
                                     -0.68051993f, -0.1975279f, 0.03748541f};
  std::vector<int64_t> classes = {1, 2, 3};
  std::vector<float> intercepts = {-3.91601811f, 0.42575697f, 0.13731251f};

  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("classlabels_ints", classes);

  test.AddInput<float>("X", {1, 3}, {1.f, 0.f, 0.f});
  test.AddOutput<int64_t>("Y", {1}, {0LL});
  test.AddOutput<float>("Z", {1, 3}, {0.f, 0.f, 0.f});

  test.Run(OpTester::ExpectResult::kExpectFailure,
           "coefficients size (6) is less than class_count (3) * num_features (3)");
}

#if !defined(ORT_NO_EXCEPTIONS)
// Regression test: classlabels_ints has fewer elements than classes defined by intercepts.
TEST(MLOpTest, LinearClassifierClassLabelsIntsTooFewFails) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  // 2 intercepts => class_count = 2, but only 1 class label provided.
  std::vector<float> coefficients = {1.f, 2.f, 3.f, 4.f};
  std::vector<int64_t> classes = {42};
  std::vector<float> intercepts = {0.f, 100.f};

  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("classlabels_ints", classes);

  test.AddInput<float>("X", {1, 2}, {1.f, 2.f});
  test.AddOutput<int64_t>("Y", {1}, {0LL});
  test.AddOutput<float>("Z", {1, 2}, {0.f, 0.f});

  test.Run(OpTester::ExpectResult::kExpectFailure,
           "classlabels_ints has 1 elements but intercepts defines 2 classes");
}

// Regression test: classlabels_strings has fewer elements than classes defined by intercepts.
TEST(MLOpTest, LinearClassifierClassLabelsStringsTooFewFails) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  // 3 intercepts => class_count = 3, but only 2 class labels provided.
  std::vector<float> coefficients = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<std::string> labels = {"cat", "dog"};
  std::vector<float> intercepts = {0.f, 0.f, 100.f};

  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("classlabels_strings", labels);

  test.AddInput<float>("X", {1, 2}, {1.f, 2.f});
  test.AddOutput<std::string>("Y", {1}, {std::string("cat")});
  test.AddOutput<float>("Z", {1, 3}, {0.f, 0.f, 0.f});

  test.Run(OpTester::ExpectResult::kExpectFailure,
           "classlabels_strings has 2 elements but intercepts defines 3 classes");
}

// Regression test: both classlabels_ints and classlabels_strings specified.
TEST(MLOpTest, LinearClassifierBothClassLabelsFails) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> coefficients = {1.f, 2.f, 3.f, 4.f};
  std::vector<float> intercepts = {0.f, 0.f};

  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("classlabels_ints", std::vector<int64_t>{0, 1});
  test.AddAttribute("classlabels_strings", std::vector<std::string>{"a", "b"});

  test.AddInput<float>("X", {1, 2}, {1.f, 2.f});
  test.AddOutput<std::string>("Y", {1}, {std::string("a")});
  test.AddOutput<float>("Z", {1, 2}, {0.f, 0.f});

  test.Run(OpTester::ExpectResult::kExpectFailure,
           "only one of classlabels_strings or classlabels_ints may be specified");
}

// Regression test: multi-class with no classlabels at all would index an empty vector.
TEST(MLOpTest, LinearClassifierMulticlassNoClassLabelsFails) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  // 3 intercepts => class_count = 3, but no classlabels provided.
  std::vector<float> coefficients = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  std::vector<float> intercepts = {0.f, 0.f, 0.f};

  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);

  test.AddInput<float>("X", {1, 2}, {1.f, 2.f});
  test.AddOutput<int64_t>("Y", {1}, {0LL});
  test.AddOutput<float>("Z", {1, 3}, {0.f, 0.f, 0.f});

  test.Run(OpTester::ExpectResult::kExpectFailure,
           "classlabels_ints or classlabels_strings must be provided");
}

// Regression test: binary classification with 1 label (not 2) should be rejected.
TEST(MLOpTest, LinearClassifierBinaryWrongLabelCountFails) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  // 1 intercept => binary, but only 1 label instead of required 2.
  std::vector<float> coefficients = {1.f, 2.f};
  std::vector<float> intercepts = {0.f};

  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("classlabels_ints", std::vector<int64_t>{42});

  test.AddInput<float>("X", {1, 2}, {1.f, 2.f});
  test.AddOutput<int64_t>("Y", {1}, {0LL});
  test.AddOutput<float>("Z", {1, 1}, {0.f});

  test.Run(OpTester::ExpectResult::kExpectFailure,
           "classlabels_ints must have exactly 2 elements for binary classification");
}

// Regression test: binary classification with 3 string labels should be rejected.
TEST(MLOpTest, LinearClassifierBinaryTooManyStringLabelsFails) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> coefficients = {1.f, 2.f};
  std::vector<float> intercepts = {0.f};

  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("classlabels_strings", std::vector<std::string>{"a", "b", "c"});

  test.AddInput<float>("X", {1, 2}, {1.f, 2.f});
  test.AddOutput<std::string>("Y", {1}, {std::string("a")});
  test.AddOutput<float>("Z", {1, 1}, {0.f});

  test.Run(OpTester::ExpectResult::kExpectFailure,
           "classlabels_strings must have exactly 2 elements for binary classification");
}
#endif  // !defined(ORT_NO_EXCEPTIONS)

// Input must be 1-D or 2-D. 3-D input should fail at runtime.
TEST(MLOpTest, LinearClassifierInput3DFails) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  std::vector<float> coefficients = {1.f, 2.f};
  std::vector<float> intercepts = {0.f};
  std::vector<int64_t> classes = {0, 1};

  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("classlabels_ints", classes);

  // Bypass ONNX shape inference so we exercise the kernel's own runtime rank validation.
  test.AddShapeToTensorData(false);
  test.AddInput<float>("X", {1, 1, 2}, {1.f, 2.f});
  test.AddOutput<int64_t>("Y", {1}, {0LL});
  test.AddOutput<float>("Z", {1, 2}, {0.f, 0.f});

  test.Run(OpTester::ExpectResult::kExpectFailure, "input must be 1-D or 2-D");
}

// 1-D input should be treated as [1, C].
TEST(MLOpTest, LinearClassifier1DInput) {
  OpTester test("LinearClassifier", 1, onnxruntime::kMLDomain);

  // 1 intercept => binary, 2 features
  std::vector<float> coefficients = {1.f, -1.f};
  std::vector<float> intercepts = {0.f};

  test.AddAttribute("coefficients", coefficients);
  test.AddAttribute("intercepts", intercepts);
  test.AddAttribute("classlabels_ints", std::vector<int64_t>{10, 20});

  // Input [2] treated as [1,2]. score = 1*1 + (-1)*2 + 0 = -1 < 0 => class 0 (label 10)
  test.AddInput<float>("X", {2}, {1.f, 2.f});
  test.AddOutput<int64_t>("Y", {1}, {10LL});
  test.AddOutput<float>("Z", {1, 2}, {2.f, -1.f});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
