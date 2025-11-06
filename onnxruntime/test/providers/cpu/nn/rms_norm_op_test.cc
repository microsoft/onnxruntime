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
//------------
TEST(RMSNormalizationOpTest, RMSNorm_Scale_Scalar_Broadcast_ShouldPass) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-5f);
  test.AddAttribute<int64_t>("axis", 2);

  // X: values 0..29 reshaped to (2,5,3)
  std::vector<float> x(2 * 5 * 3);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>(i);
  test.AddInput<float>("X", {2, 5, 3}, x);

  test.AddInput<float>("scale", /*dims*/ {}, /*vals*/ {1.5f}, /*is_initializer*/ true);

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

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider});
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_1x1x1_Broadcast_ShouldPass) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-5f);
  test.AddAttribute<int64_t>("axis", 2);

  std::vector<float> x(2 * 5 * 3);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>(i);
  test.AddInput<float>("X", {2, 5, 3}, x);

  test.AddInput<float>("scale", {1, 1, 1}, {1.0f}, /*is_initializer*/ true);

  test.AddOutput<float>(
      "Y", {2, 5, 3},
      {0.0000f, 0.7746f, 1.5492f,
       0.7348f, 0.9798f, 1.2247f,
       0.8514f, 0.9933f, 1.1352f,
       0.8970f, 0.9967f, 1.0964f,
       0.9213f, 0.9980f, 1.0748f,

       0.9363f, 0.9987f, 1.0611f,
       0.9465f, 0.9991f, 1.0517f,
       0.9539f, 0.9993f, 1.0447f,
       0.9595f, 0.9995f, 1.0394f,
       0.9639f, 0.9996f, 1.0353f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kTensorrtExecutionProvider, kDnnlExecutionProvider, kOpenVINOExecutionProvider,
            kNnapiExecutionProvider, kQnnExecutionProvider});
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_3_ShouldPass_NoBroadcast) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-5f);
  test.AddAttribute<int64_t>("axis", 2);

  std::vector<float> x(2 * 5 * 3);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>(i);
  test.AddInput<float>("X", {2, 5, 3}, x);

  test.AddInput<float>("scale", {3}, {1.5f, 1.5f, 1.5f}, /*is_initializer*/ true);

  test.AddOutput<float>("Y", {2, 5, 3},
    {
      0.0000f, 1.1619f, 2.3238f,
      1.1023f, 1.4697f, 1.8371f,
      1.2771f, 1.4899f, 1.7027f,
      1.3455f, 1.4950f, 1.6445f,
      1.3819f, 1.4971f, 1.6122f,

      1.4044f, 1.4981f, 1.5917f,
      1.4197f, 1.4986f, 1.5775f,
      1.4308f, 1.4990f, 1.5671f,
      1.4392f, 1.4992f, 1.5592f,
      1.4458f, 1.4994f, 1.5529f
    });

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_1x1x3_ShouldPass_WhenBroadcastSupported) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-5f);
  test.AddAttribute<int64_t>("axis", 2);

  std::vector<float> x(2 * 5 * 3);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>(i);
  test.AddInput<float>("X", {2, 5, 3}, x);

  test.AddInput<float>("scale", {1, 1, 3}, {1.5f, 1.5f, 1.5f}, /*is_initializer*/ true);

  test.AddOutput<float>("Y", {2, 5, 3},
    {
      0.0000f, 1.1619f, 2.3238f,
      1.1023f, 1.4697f, 1.8371f,
      1.2771f, 1.4899f, 1.7027f,
      1.3455f, 1.4950f, 1.6445f,
      1.3819f, 1.4971f, 1.6122f,

      1.4044f, 1.4981f, 1.5917f,
      1.4197f, 1.4986f, 1.5775f,
      1.4308f, 1.4990f, 1.5671f,
      1.4392f, 1.4992f, 1.5592f,
      1.4458f, 1.4994f, 1.5529f
    });

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(RMSNormalizationOpTest, RMSNorm_Scale_Bx1x3_ShouldPass_WhenBroadcastSupported) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-5f);
  test.AddAttribute<int64_t>("axis", 2);

  std::vector<float> x(2 * 5 * 3);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>(i);
  test.AddInput<float>("X", {2, 5, 3}, x);

  // batch 0: 1.25; batch 1: 1.75
  test.AddInput<float>("scale", {2, 1, 3},
                       {1.25f,1.25f,1.25f,  1.75f,1.75f,1.75f}, /*is_initializer*/ true);

  test.AddOutput<float>("Y", {2, 5, 3},
    {
      // batch 0 (S=0..4)
      0.0000f, 0.9682f, 1.9365f,
      0.9186f, 1.2247f, 1.5309f,
      1.0642f, 1.2416f, 1.4190f,
      1.1213f, 1.2459f, 1.3704f,
      1.1516f, 1.2475f, 1.3435f,

      // batch 1 (S=0..4)
      1.6385f, 1.7477f, 1.8570f,
      1.6564f, 1.7484f, 1.8404f,
      1.6693f, 1.7488f, 1.8283f,
      1.6791f, 1.7491f, 1.8190f,
      1.6868f, 1.7493f, 1.8117f
    });

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}
TEST(RMSNormalizationOpTest, RMSNorm_Scale_1xSx3_ShouldPass_WhenBroadcastSupported) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-5f);
  test.AddAttribute<int64_t>("axis", 2);

  std::vector<float> x(2 * 5 * 3);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>(i);
  test.AddInput<float>("X", {2, 5, 3}, x);

  test.AddInput<float>("scale", {1, 5, 3},
                       {
                         1.1f,1.1f,1.1f,  1.2f,1.2f,1.2f,  1.3f,1.3f,1.3f,
                         1.4f,1.4f,1.4f,  1.5f,1.5f,1.5f
                       }, /*is_initializer*/ true);

  test.AddOutput<float>("Y", {2, 5, 3},
    {
      // batch 0
      0.0000f, 0.8521f, 1.7041f,
      0.8818f, 1.1758f, 1.4697f,
      1.1068f, 1.2912f, 1.4757f,
      1.2558f, 1.3954f, 1.5349f,
      1.3819f, 1.4971f, 1.6122f,

      // batch 1
      1.0299f, 1.0986f, 1.1672f,
      1.1358f, 1.1989f, 1.2620f,
      1.2401f, 1.2991f, 1.3582f,
      1.3433f, 1.3993f, 1.4552f,
      1.4458f, 1.4994f, 1.5529f
    });

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}
TEST(RMSNormalizationOpTest, RMSNorm_Scale_BxSx3_ShouldPass_NoBroadcast) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-5f);
  test.AddAttribute<int64_t>("axis", 2);

  std::vector<float> x(2 * 5 * 3);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>(i);
  test.AddInput<float>("X", {2, 5, 3}, x);

  std::vector<float> scale(2 * 5 * 3, 1.5f);
  test.AddInput<float>("scale", {2, 5, 3}, scale, /*is_initializer*/ true);

  test.AddOutput<float>("Y", {2, 5, 3},
    {
      0.0000f, 1.1619f, 2.3238f,
      1.1023f, 1.4697f, 1.8371f,
      1.2771f, 1.4899f, 1.7027f,
      1.3455f, 1.4950f, 1.6445f,
      1.3819f, 1.4971f, 1.6122f,

      1.4044f, 1.4981f, 1.5917f,
      1.4197f, 1.4986f, 1.5775f,
      1.4308f, 1.4990f, 1.5671f,
      1.4392f, 1.4992f, 1.5592f,
      1.4458f, 1.4994f, 1.5529f
    });

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}


TEST(RMSNormalizationOpTest, RMSNorm_Scale_1xCx1x1_ShouldPass_WhenBroadcastSupported) {
  OpTester test("RMSNormalization", 23);
  test.AddAttribute<float>("epsilon", 1e-5f);
  test.AddAttribute<int64_t>("axis", 1);  // normalize over [C,H,W]

  // X: 0..15 reshaped to (1,4,2,2)
  std::vector<float> x(1 * 4 * 2 * 2);
  for (int i = 0; i < static_cast<int>(x.size()); ++i) x[i] = static_cast<float>(i);
  test.AddInput<float>("X", {1, 4, 2, 2}, x);

  // scale: [1,4,1,1]
  test.AddInput<float>("scale", {1, 4, 1, 1},
                       {1.1f, 1.2f, 1.3f, 1.4f},
                       /*is_initializer*/ true);

  // expected Y (מחושב מראש)
  test.AddOutput<float>("Y", {1, 4, 2, 2}, {
    // c=0 (scale=1.1)
    0.0000f, 0.1250f,
    0.2499f, 0.3749f,

    // c=1 (scale=1.2)
    0.5452f, 0.6816f,
    0.8179f, 0.9542f,

    // c=2 (scale=1.3)
    1.1814f, 1.3290f,
    1.4767f, 1.6244f,

    // c=3 (scale=1.4)
    1.9084f, 2.0674f,
    2.2264f, 2.3854f
  });

  test.Run(OpTester::ExpectResult::kExpectSuccess);
}









}  // namespace test
}  // namespace onnxruntime
