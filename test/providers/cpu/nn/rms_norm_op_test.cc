#include "gtest/gtest.h"
#include "gmock/gmock.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"

using namespace std;

namespace onnxruntime {
namespace test {

// All tests in this file are for the CPU provider and
// CUDA provider

TEST(RMSNormalizationOpTest, RMSNorm) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  std::vector<int64_t> input_dims{1, 2, 3};
  test.AddInput<float>("X", input_dims, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  vector<int64_t> scale_dims = {3};
  test.AddInput<float>("scale", scale_dims, {1.F, 1.F, 1.F});
  test.AddOutput<float>("Y", input_dims, {0.4629f, 0.9258f, 1.3887f, 0.7895f, 0.9869f, 1.1843f});
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider});
}

TEST(RMSNormalizationOpTest, RMSNorm_float16) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  std::vector<int64_t> input_dims{1, 2, 3};
  test.AddInput<MLFloat16>("X", input_dims, ToFloat16({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
  vector<int64_t> scale_dims = {3};
  test.AddInput<MLFloat16>("scale", scale_dims, ToFloat16({1.F, 1.F, 1.F}));
  test.AddOutput<MLFloat16>("Y", input_dims, ToFloat16({0.4629f, 0.9258f, 1.3887f, 0.7895f, 0.9869f, 1.1843f}));
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider});
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  std::vector<int64_t> input_dims{2, 2, 2};
  test.AddInput<float>("X", input_dims, {-10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f});
  vector<int64_t> scale_dims = {2};
  test.AddInput<float>("scale", scale_dims, {-0.6953f, 5.1824f});
  test.AddOutput<float>("Y", input_dims, {0.7521f, 4.7215f, -0.9832f, -0.1089f, 0.9832f, 0.1018f, -0.9806f, 0.5388f});
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider});
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_Float16) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  std::vector<int64_t> input_dims{2, 2, 2};
  test.AddInput<MLFloat16>("X", input_dims, ToFloat16({-10.264f, 8.6453f, 43.1561f, -0.641239f, -8.2164f, 0.11412f, 41.3156f, 3.0458f}));
  vector<int64_t> scale_dims = {2};
  test.AddInput<MLFloat16>("scale", scale_dims, ToFloat16({-0.6953f, 5.1824f}));
  test.AddOutput<MLFloat16>("Y", input_dims, ToFloat16({0.7521f, 4.7215f, -0.9832f, -0.1089f, 0.9832f, 0.1018f, -0.9806f, 0.5388f}));
  // TRT, DNNL, OpenVINO and NNAPI, CoreML don't support this combination of datatypes
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider});
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_Scalar_Axis2) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);
  std::vector<float> x(2 * 5 * 3);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>(i);
  test.AddInput<float>("X", {2, 5, 3}, x);
  test.AddInput<float>("scale", {}, {1.5f}, true);
  test.AddOutput<float>(
      "Y", {2, 5, 3},
      {0.0000f, 1.1619f, 2.3238f,
       1.1023f, 1.4697f, 1.8371f,
       1.2771f, 1.4899f, 1.7027f,
       1.3455f, 1.4950f, 1.6445f,
       1.3819f, 1.4971f, 1.6122f,

       1.4044f, 1.4981f, 1.5917f,
       1.4197f, 1.4986f, 1.5775f,
       1.4308f, 1.4990f, 1.5671f,
       1.4392f, 1.4992f, 1.5592f,
       1.4458f, 1.4994f, 1.5529f});
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_1x1x1_Axis2) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);
  std::vector<float> x(2 * 2 * 2);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    x[i] = static_cast<float>(i);
  }
  test.AddInput<float>("X", {2, 2, 2}, x);

  test.AddInput<float>("scale", {1, 1, 1}, {1.0f}, true);

  test.AddOutput<float>("Y", {2, 2, 2},
                        {
                            0.0000f,
                            1.4142f,
                            0.7845f,
                            1.1767f,

                            0.8835f,
                            1.1043f,
                            0.9204f,
                            1.0738f,
                        });

  test.SetOutputAbsErr("Y", 1e-4f);

  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_Vector3_Axis2) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);

  std::vector<float> x(2 * 5 * 3);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>(i);
  test.AddInput<float>("X", {2, 5, 3}, x);

  test.AddInput<float>("scale", {3}, {1.5f, 1.5f, 1.5f}, true);

  test.AddOutput<float>("Y", {2, 5, 3},
                        {0.0000f, 1.1619f, 2.3238f,
                         1.1023f, 1.4697f, 1.8371f,
                         1.2771f, 1.4899f, 1.7027f,
                         1.3455f, 1.4950f, 1.6445f,
                         1.3819f, 1.4971f, 1.6122f,

                         1.4044f, 1.4981f, 1.5917f,
                         1.4197f, 1.4986f, 1.5775f,
                         1.4308f, 1.4990f, 1.5671f,
                         1.4392f, 1.4992f, 1.5592f,
                         1.4458f, 1.4994f, 1.5529f});

  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_1x1x3_Axis2) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);

  std::vector<float> x(2 * 5 * 3);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>(i);
  test.AddInput<float>("X", {2, 5, 3}, x);

  test.AddInput<float>("scale", {1, 1, 3}, {1.5f, 1.5f, 1.5f}, true);

  test.AddOutput<float>("Y", {2, 5, 3},
                        {0.0000f, 1.1619f, 2.3238f,
                         1.1023f, 1.4697f, 1.8371f,
                         1.2771f, 1.4899f, 1.7027f,
                         1.3455f, 1.4950f, 1.6445f,
                         1.3819f, 1.4971f, 1.6122f,

                         1.4044f, 1.4981f, 1.5917f,
                         1.4197f, 1.4986f, 1.5775f,
                         1.4308f, 1.4990f, 1.5671f,
                         1.4392f, 1.4992f, 1.5592f,
                         1.4458f, 1.4994f, 1.5529f});

  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_Bx1x3_Axis2) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-5f);
  test.AddAttribute<int64_t>("axis", 2);

  std::vector<float> x(3 * 2 * 3);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    x[i] = static_cast<float>(i);
  }
  test.AddInput<float>("X", {3, 2, 3}, x);

  test.AddInput<float>(
      "scale", {3, 1, 3},
      {
          1.0f,
          1.0f,
          1.0f,
          1.2f,
          1.2f,
          1.2f,
          1.4f,
          1.4f,
          1.4f,
      },
      true);

  test.AddOutput<float>(
      "Y", {3, 2, 3},
      {
          0.0000f,
          0.7746f,
          1.5492f,
          0.7348f,
          0.9798f,
          1.2247f,

          1.0216f,
          1.1919f,
          1.3622f,
          1.0764f,
          1.1960f,
          1.3156f,

          1.2898f,
          1.3972f,
          1.5047f,
          1.3108f,
          1.3982f,
          1.4856f,
      });

  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_1xSx3_Axis2) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-5f);
  test.AddAttribute<int64_t>("axis", 2);

  std::vector<float> x(2 * 4 * 3);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) {
    x[i] = static_cast<float>(i);
  }
  test.AddInput<float>("X", {2, 4, 3}, x);

  test.AddInput<float>("scale",
                       {1, 4, 3},
                       {
                           1.1f,
                           1.1f,
                           1.1f,
                           1.2f,
                           1.2f,
                           1.2f,
                           1.3f,
                           1.3f,
                           1.3f,
                           1.4f,
                           1.4f,
                           1.4f,
                       },
                       true);

  test.AddOutput<float>(
      "Y", {2, 4, 3},
      {
          0.0000f,
          0.8521f,
          1.7041f,
          0.8818f,
          1.1758f,
          1.4697f,
          1.1068f,
          1.2912f,
          1.4757f,
          1.2558f,
          1.3954f,
          1.5349f,

          1.0134f,
          1.0978f,
          1.1823f,
          1.1235f,
          1.1984f,
          1.2733f,
          1.2304f,
          1.2988f,
          1.3672f,
          1.3354f,
          1.3990f,
          1.4626f,
      });

  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_NoBroadcast_BxSx3_Axis2) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);

  std::vector<float> x(2 * 5 * 3);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>(i);
  test.AddInput<float>("X", {2, 5, 3}, x);

  std::vector<float> scale(2 * 5 * 3, 1.5f);
  test.AddInput<float>("scale", {2, 5, 3}, scale, true);

  test.AddOutput<float>("Y", {2, 5, 3},
                        {0.0000f, 1.1619f, 2.3238f,
                         1.1023f, 1.4697f, 1.8371f,
                         1.2771f, 1.4899f, 1.7027f,
                         1.3455f, 1.4950f, 1.6445f,
                         1.3819f, 1.4971f, 1.6122f,

                         1.4044f, 1.4981f, 1.5917f,
                         1.4197f, 1.4986f, 1.5775f,
                         1.4308f, 1.4990f, 1.5671f,
                         1.4392f, 1.4992f, 1.5592f,
                         1.4458f, 1.4994f, 1.5529f});

  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_1xCx1x1_Axis1) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 1);
  std::vector<float> x(1 * 4 * 2 * 2);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>(i);
  test.AddInput<float>("X", {1, 4, 2, 2}, x);

  test.AddInput<float>("scale", {1, 4, 1, 1},
                       {1.1f, 1.2f, 1.3f, 1.4f},
                       true);

  test.AddOutput<float>("Y", {1, 4, 2, 2}, {0.0000000, 0.1249516, 0.2499032, 0.3748548,

                                            0.5452434, 0.6815542, 0.8178651, 0.9541759,

                                            1.1813605, 1.3290305, 1.4767007, 1.6243708,

                                            1.9083518, 2.0673811, 2.2264102, 2.3854396});
  test.SetOutputAbsErr("Y", 1e-4f);
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_1xCx1_Axis1) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 1);
  std::vector<float> x(2 * 3 * 2);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>(i);
  test.AddInput<float>("X", {2, 3, 2}, x);
  test.AddInput<float>("scale", {1, 3, 1}, {1.0f, 1.2f, 1.4f}, true);
  test.AddOutput<float>(
      "Y", {2, 3, 2},
      {0.0f, 0.33028895f,
       0.79269350f, 1.18904030f,
       1.84961808f, 2.31202269f,
       0.69205177f, 0.80739373f,
       1.10728300f, 1.24569333f,
       1.61478746f, 1.77626622f});

  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_1x3x2x1_Axis1) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 1);
  std::vector<float> x(1 * 3 * 2 * 2);
  for (int i = 0; i < static_cast<int>(x.size()); ++i)
    x[i] = static_cast<float>(i);
  test.AddInput<float>("X", {1, 3, 2, 2}, x);
  test.AddInput<float>(
      "scale", {1, 3, 2, 1},
      {
          1.0f,
          1.1f,
          1.2f,
          1.3f,
          1.4f,
          1.5f,
      },
      true);
  test.AddOutput<float>(
      "Y", {1, 3, 2, 2},
      {0.0f, 0.15399808f,
       0.33879578f, 0.50819367f,
       0.73919082f, 0.92398852f,
       1.20118499f, 1.40138257f,
       1.72477841f, 1.94037580f,
       2.30997109f, 2.54096842f});

  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_1xSx1xW_Axis2) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);
  std::vector<float> x(1 * 2 * 2 * 2);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>(i);
  test.AddInput<float>("X", {1, 2, 2, 2}, x);

  test.AddInput<float>("scale", {1, 2, 1, 2},
                       {1.0f, 1.2f,
                        1.4f, 1.6f},
                       true);
  test.AddOutput<float>("Y", {1, 2, 2, 2},
                        {0.0000f, 0.6414f,
                         1.0690f, 1.9243f,

                         0.9978f, 1.4254f,
                         1.4967f, 1.9956f});

  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_1x1xHx1_Axis2) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);

  std::vector<float> x(1 * 2 * 2 * 2);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>(i);
  test.AddInput<float>("X", {1, 2, 2, 2}, x);

  test.AddInput<float>("scale", {1, 1, 2, 1}, {1.0f, 1.3f}, true);

  test.AddOutput<float>("Y", {1, 2, 2, 2},
                        {0.0000f, 0.5345f,
                         1.3898f, 2.0846f,

                         0.7127f, 0.8909f,
                         1.3898f, 1.6214f});

  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_1x1x1xW_Axis2) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);

  std::vector<float> x(1 * 2 * 2 * 3);
  for (int i = 0; i < (int)x.size(); ++i) x[i] = (float)i;
  test.AddInput<float>("X", {1, 2, 2, 3}, x);

  test.AddInput<float>("scale", {1, 1, 1, 3}, {1.0f, 1.2f, 1.4f}, true);

  test.AddOutput<float>("Y", {1, 2, 2, 3},
                        {0.0000f, 0.3963f, 0.9248f,
                         0.9909f, 1.5854f, 2.3120f,
                         0.6921f, 0.9689f, 1.2918f,
                         1.0381f, 1.3841f, 1.7763f});
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_1xSx1x1_Axis2) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);

  std::vector<float> x(1 * 3 * 2 * 2);
  for (int i = 0; i < (int)x.size(); ++i) x[i] = (float)i;
  test.AddInput<float>("X", {1, 3, 2, 2}, x);

  test.AddInput<float>("scale", {1, 3, 1, 1}, {1.0f, 1.2f, 1.4f}, true);

  test.AddOutput<float>("Y", {1, 3, 2, 2},
                        {0.0000f, 0.5345f, 1.0690f, 1.6036f,
                         0.8552f, 1.0690f, 1.2829f, 1.4967f,
                         1.1709f, 1.3172f, 1.4636f, 1.6099f});
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_Bx1x1xW_Axis2) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);

  std::vector<float> x(2 * 1 * 2 * 2);
  for (int i = 0; i < (int)x.size(); ++i) x[i] = (float)i;
  test.AddInput<float>("X", {2, 1, 2, 2}, x);

  test.AddInput<float>("scale", {2, 1, 1, 2}, {1.0f, 1.1f, 1.3f, 1.4f}, true);

  test.AddOutput<float>("Y", {2, 1, 2, 2},
                        {0.0000f, 0.5880f, 1.0690f, 1.7639f,
                         0.9265f, 1.2472f, 1.3898f, 1.7461f});
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_1x1xHxW_Axis2) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);

  test.AddInput<float>("X", {1, 2, 2, 3},
                       {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

  test.AddInput<float>("scale", {1, 1, 2, 3},
                       {1.0f, 1.1f, 1.2f,
                        1.3f, 1.4f, 1.5f},
                       true);

  test.AddOutput<float>("Y", {1, 2, 2, 3},
                        {0.0000f, 0.3633f, 0.7927f, 1.2881f, 1.8496f, 2.4772f,
                         0.6921f, 0.8881f, 1.1073f, 1.3495f, 1.6148f, 1.9031f});
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_1xSx1xW_AxisNeg2) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", -2);

  std::vector<float> x(1 * 2 * 2 * 2);
  for (int i = 0; i < (int)x.size(); ++i) x[i] = (float)i;
  test.AddInput<float>("X", {1, 2, 2, 2}, x);

  test.AddInput<float>("scale", {1, 2, 1, 2},
                       {1.0f, 1.2f, 1.4f, 1.6f}, true);

  test.AddOutput<float>("Y", {1, 2, 2, 2},
                        {0.0000f, 0.6414f,
                         1.0690f, 1.9243f,

                         0.9978f, 1.4254f,
                         1.4967f, 1.9956f});
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_1xSx1x1xC_Axis3) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 3);

  const int B = 1, S = 2, H = 2, W = 2, C = 3;
  std::vector<float> x(B * S * H * W * C);
  for (int i = 0; i < (int)x.size(); ++i) x[i] = (float)i;
  test.AddInput<float>("X", {B, S, H, W, C}, x);

  test.AddInput<float>("scale", {1, S, 1, 1, C},
                       {1.0f, 1.1f, 1.2f,
                        1.3f, 1.4f, 1.5f},
                       true);

  test.AddOutput<float>("Y", {B, S, H, W, C},
                        {
                            0.0000f,
                            0.3633f,
                            0.7927f,
                            0.9909f,
                            1.4533f,
                            1.9817f,
                            0.6921f,
                            0.8881f,
                            1.1073f,
                            1.0381f,
                            1.2688f,
                            1.5225f,
                            1.0685f,
                            1.2466f,
                            1.4383f,
                            1.3356f,
                            1.5342f,
                            1.7465f,
                            1.1375f,
                            1.2931f,
                            1.4584f,
                            1.3271f,
                            1.4973f,
                            1.6771f,
                        });
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_Float16_OuterInnerBroadcast_Axis1) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 1);
  std::vector<float> x_f(24);
  for (int i = 0; i < 24; ++i) x_f[i] = static_cast<float>(i);

  std::vector<MLFloat16> x_half(x_f.size());
  for (size_t i = 0; i < x_f.size(); ++i)
    x_half[i] = MLFloat16(x_f[i]);

  test.AddInput<MLFloat16>("X", {2, 3, 4}, x_half);

  std::vector<float> scale_f = {1.0f, 2.0f, 3.0f};
  std::vector<MLFloat16> scale_half(scale_f.size());
  for (size_t i = 0; i < scale_f.size(); ++i)
    scale_half[i] = MLFloat16(scale_f[i]);

  test.AddInput<MLFloat16>("scale", {1, 3, 1}, scale_half, true);

  std::vector<float> y_f = {
      0.0000f, 0.1540f, 0.3080f, 0.4620f,
      1.2320f, 1.5400f, 1.8480f, 2.1560f,
      3.6960f, 4.1579f, 4.6199f, 5.0819f,
      0.6728f, 0.7288f, 0.7849f, 0.8409f,
      1.7940f, 1.9061f, 2.0183f, 2.1304f,
      3.3638f, 3.5319f, 3.7001f, 3.8683f};

  std::vector<MLFloat16> y_half(y_f.size());
  for (size_t i = 0; i < y_f.size(); ++i)
    y_half[i] = MLFloat16(y_f[i]);

  test.AddOutput<MLFloat16>("Y", {2, 3, 4}, y_half);
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}
TEST(RMSNormalizationOpTest, RMSNorm_Scale_Float16_OuterBroadcast_BxSx1_Axis2) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 2);
  std::vector<float> x_f(2 * 2 * 3);
  for (int i = 0; i < static_cast<int>(x_f.size()); ++i) {
    x_f[static_cast<size_t>(i)] = static_cast<float>(i);
  }
  std::vector<MLFloat16> x_half(x_f.size());
  for (size_t i = 0; i < x_f.size(); ++i) {
    x_half[i] = MLFloat16(x_f[i]);
  }
  test.AddInput<MLFloat16>("X", {2, 2, 3}, x_half);
  std::vector<float> scale_f = {
      1.0f, 2.0f,
      3.0f, 4.0f};
  std::vector<MLFloat16> scale_half(scale_f.size());
  for (size_t i = 0; i < scale_f.size(); ++i) {
    scale_half[i] = MLFloat16(scale_f[i]);
  }
  test.AddInput<MLFloat16>("scale", {2, 2, 1}, scale_half, true);
  std::vector<float> y_f = {
      0.0000f, 0.7746f, 1.5492f,
      1.4697f, 1.9596f, 2.4495f,

      2.5541f, 2.9798f, 3.4055f,
      3.5881f, 3.9867f, 4.3854f};

  std::vector<MLFloat16> y_half(y_f.size());
  for (size_t i = 0; i < y_f.size(); ++i) {
    y_half[i] = MLFloat16(y_f[i]);
  }

  test.AddOutput<MLFloat16>("Y", {2, 2, 3}, y_half);
  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}
TEST(RMSNormalizationOpTest, RMSNorm_Scale_Broadcast_Inner_Mixed) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 1);
  std::vector<int64_t> dims{1, 2, 4};
  std::vector<float> x = {
      0.0f, 1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f, 7.0f};
  test.AddInput<float>("X", dims, x);
  std::vector<float> scale = {1.0f, 0.5f, 1.0f, 0.5f};
  test.AddInput<float>("Scale", {1, 4}, scale);
  std::vector<float> expected = {
      0.0f,
      0.119527f,
      0.478108f,
      0.358581f,
      0.956216f,
      0.597635f,
      1.434324f,
      0.836689f};

  test.AddOutput<float>("Y", dims, expected);

  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) GTEST_SKIP() << "CPU EP not available in this build.";
  test.ConfigEp(std::move(cpu)).RunWithConfig();
}

TEST(RMSNormalizationOpTest, RMSNorm_InvalidScaleRank_GreaterThanInputRank_ShouldFail) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-05f);
  test.AddAttribute<int64_t>("axis", 1);

  std::vector<int64_t> x_dims{2, 4};
  std::vector<float> x = {
      0.0f, 1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f, 7.0f};
  test.AddInput<float>("X", x_dims, x);

  std::vector<int64_t> scale_dims{1, 2, 4};
  std::vector<float> scale(1 * 2 * 4, 1.0f);
  test.AddInput<float>("Scale", scale_dims, scale);

  // Dummy output so model builds; failure is expected during shape check.
  std::vector<float> dummy_y(x.size(), 0.0f);
  test.AddOutput<float>("Y", x_dims, dummy_y);

  auto cpu = DefaultCpuExecutionProvider();
  if (!cpu) {
    GTEST_SKIP() << "CPU EP not available in this build.";
  }
  test.ConfigEp(std::move(cpu));

  test.Run(OpTester::ExpectResult::kExpectFailure,
           "Scale/Bias rank cannot exceed Input rank.");
}

}  // namespace test
}  // namespace onnxruntime
