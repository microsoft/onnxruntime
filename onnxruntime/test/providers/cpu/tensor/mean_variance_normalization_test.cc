// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime::test {

TEST(MeanVarianceNormalizationTest, DefaultAxes) {
  constexpr int64_t N = 2, C = 2, H = 2, W = 3;

  std::vector<float> N1C1 = {3.0f, -3.0f, -1.0f,
                             1.0f, 2.0f, -1.0f};
  std::vector<float> N1C2 = {-2.0f, -2.0f, -2.0f,
                             4.0f, 1.0f, 4.0f};
  std::vector<float> N2C1 = {
      0.0f,
      -2.0f,
      -2.0f,
      -4.0f,
      5.0f,
      7.0f,
  };
  std::vector<float> N2C2 = {
      5.0f,
      -5.0f,
      -5.0f,
      3.0f,
      4.0f,
      4.0f,
  };

  std::vector<float> X;
  X.reserve(N * C * H * W);
  X.insert(X.end(), N1C1.begin(), N1C1.end());
  X.insert(X.end(), N1C2.begin(), N1C2.end());
  X.insert(X.end(), N2C1.begin(), N2C1.end());
  X.insert(X.end(), N2C2.begin(), N2C2.end());

  std::vector<float> C1;
  C1.reserve(N * H * W);
  C1.insert(C1.end(), N1C1.begin(), N1C1.end());
  C1.insert(C1.end(), N2C1.begin(), N2C1.end());
  auto C1_meam_stdev = MeanStdev(C1);

  std::vector<float> C2;
  C2.reserve(N * H * W);
  C2.insert(C2.end(), N1C2.begin(), N1C2.end());
  C2.insert(C2.end(), N2C2.begin(), N2C2.end());
  auto C2_meam_stdev = MeanStdev(C2);

  std::vector<float> N1C1_result(N1C1), N1C2_result(N1C2),
      N2C1_result(N2C1), N2C2_result(N2C2);
  Normalize(N1C1_result, C1_meam_stdev, true);
  Normalize(N2C1_result, C1_meam_stdev, true);
  Normalize(N1C2_result, C2_meam_stdev, true);
  Normalize(N2C2_result, C2_meam_stdev, true);

  std::vector<float> result;
  result.reserve(N * C * H * W);
  result.insert(result.end(), N1C1_result.begin(), N1C1_result.end());
  result.insert(result.end(), N1C2_result.begin(), N1C2_result.end());
  result.insert(result.end(), N2C1_result.begin(), N2C1_result.end());
  result.insert(result.end(), N2C2_result.begin(), N2C2_result.end());

  OpTester test("MeanVarianceNormalization", 9);
  test.AddInput<float>("input", {N, C, H, W}, X);
  test.AddOutput<float>("output", {N, C, H, W}, result);
  test.Run();
}

static void TestMeanVarianceNormalizationOverAllAxes(const std::vector<int64_t>& shape) {
  SCOPED_TRACE(MakeString("shape: ", TensorShape(shape)));

  FixedPatternValueGenerator generator{};

  const auto X_value_candidates = ValueRange<float>(11, -5.0f);
  const auto X = generator.Discrete<float>(shape, X_value_candidates);
  const auto mean_stdev = MeanStdev(X);

  std::vector<float> Y(X);
  Normalize(Y, mean_stdev, true);

  OpTester test("MeanVarianceNormalization", 9);
  const auto all_axes = ValueRange<int64_t>(shape.size());
  test.AddAttribute("axes", all_axes);
  test.AddInput<float>("input", shape, X);
  test.AddOutput<float>("output", shape, Y);

  test.Run();
}

TEST(MeanVarianceNormalizationTest, AllAxes) {
  TestMeanVarianceNormalizationOverAllAxes({2, 2, 4});
  TestMeanVarianceNormalizationOverAllAxes({2, 2, 2, 3});
  TestMeanVarianceNormalizationOverAllAxes({2, 2, 2, 2, 2});
}

TEST(MeanVarianceNormalizationTest, AxesSubsets5D) {
  // test data was generated with:
  //   python onnxruntime/test/providers/cpu/tensor/gen_mvn_test_data.py --shape 2 2 2 2 2 --axes <axes values>

  auto axes_to_str = [](gsl::span<const int64_t> axes) {
    std::ostringstream s;
    s << "{ ";
    std::copy(axes.begin(), axes.end(), std::ostream_iterator<int64_t>(s, " "));
    s << "}";
    return s.str();
  };

  auto test_with_axes = [&](gsl::span<const int64_t> axes, gsl::span<const float> Y) {
    SCOPED_TRACE(axes_to_str(axes));

    constexpr std::array X = {
        0.6369617f,
        0.2697867f,
        0.0409735f,
        0.0165276f,
        0.8132702f,
        0.9127556f,
        0.6066358f,
        0.7294966f,
        0.5436250f,
        0.9350724f,
        0.8158536f,
        0.0027385f,
        0.8574043f,
        0.0335856f,
        0.7296554f,
        0.1756556f,
        0.8631789f,
        0.5414612f,
        0.2997119f,
        0.4226872f,
        0.0283197f,
        0.1242833f,
        0.6706244f,
        0.6471895f,
        0.6153851f,
        0.3836776f,
        0.9972099f,
        0.9808353f,
        0.6855420f,
        0.6504593f,
        0.6884467f,
        0.3889214f,
    };

    const std::vector<int64_t> shape{2, 2, 2, 2, 2};

    OpTester test("MeanVarianceNormalization", 9);
    test.AddAttribute("axes", axes);
    test.AddInput<float>("input", shape, X.data(), X.size());
    test.AddOutput<float>("output", shape, Y.data(), Y.size());

    test.Run();
  };

  test_with_axes(
      AsSpan<int64_t>({0, 2, 4}),
      AsSpan<float>({
          0.3508345f,
          -0.7870349f,
          -1.4605863f,
          -1.5525494f,
          0.8972119f,
          1.2055154f,
          0.6673803f,
          1.1295706f,
          -0.1683330f,
          1.3134559f,
          0.6321192f,
          -1.7208749f,
          1.0194501f,
          -2.0990413f,
          0.3826789f,
          -1.2204870f,
          1.0518781f,
          0.0548801f,
          -0.4872377f,
          -0.0246164f,
          -1.5353374f,
          -1.2379477f,
          0.9080993f,
          0.8199395f,
          0.1033084f,
          -0.7737996f,
          1.1569287f,
          1.1095439f,
          0.3688809f,
          0.2360785f,
          0.2634291f,
          -0.6033379f,
      }));

  test_with_axes(
      AsSpan<int64_t>({1, 2, 3}),
      AsSpan<float>({
          0.0260567f,
          -0.3008327f,
          -2.3950341f,
          -0.9652744f,
          0.7422773f,
          1.3860379f,
          -0.0971367f,
          0.9052460f,
          -0.3531062f,
          1.4445876f,
          0.7527716f,
          -1.0014510f,
          0.9215636f,
          -0.9205217f,
          0.4026078f,
          -0.5477917f,
          0.8924309f,
          0.1015335f,
          -1.0632416f,
          -0.4004898f,
          -2.0051854f,
          -1.6617567f,
          0.2241158f,
          0.5484163f,
          0.0323921f,
          -0.5653723f,
          1.3576236f,
          1.9586404f,
          0.2758914f,
          0.5622366f,
          0.2859732f,
          -0.5432080f,
      }));

  test_with_axes(
      AsSpan<int64_t>({0, 1, 4}),
      AsSpan<float>({
          0.1843672f,
          -1.5822912f,
          -1.0098907f,
          -1.0706838f,
          0.8350127f,
          1.1118552f,
          0.1472972f,
          0.8161319f,
          -0.2647213f,
          1.6187237f,
          0.9171133f,
          -1.1049752f,
          0.9578265f,
          -1.3346525f,
          0.8169968f,
          -2.1988905f,
          1.2728089f,
          -0.2751323f,
          -0.3664493f,
          -0.0606291f,
          -1.3493062f,
          -1.0822638f,
          0.4956413f,
          0.3680653f,
          0.0805517f,
          -1.0343067f,
          1.3681179f,
          1.3273969f,
          0.4795772f,
          0.3819509f,
          0.5926631f,
          -1.0379052f,
      }));
}

}  // namespace onnxruntime::test
