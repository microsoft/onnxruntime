// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/util/include/default_providers.h"

#ifdef COREML_ENABLE_MLPROGRAM
using namespace std;
namespace onnxruntime {
namespace test {

template <typename T>
class GroupNormalizationOpTest : public ::testing::Test {
};
using GroupNormalizationOpTestTypes = ::testing::Types<float, MLFloat16>;
TYPED_TEST_SUITE(GroupNormalizationOpTest, GroupNormalizationOpTestTypes);

template <typename T>
static std::vector<T> GetTypedArray(std::vector<float> inputs, [[maybe_unused]] T v = T(0.f)) {
  if constexpr (std::is_same<T, float>::value) {
    return inputs;
  } else {
    return ToFloat16(inputs);
  }
}

// Disable TensorRT on some of the tests because its parser doesn't support weight as input

TYPED_TEST(GroupNormalizationOpTest, InstanceNorm) {
  OpTester test("GroupNormalization", 18);
  test.AddAttribute("epsilon", 0.3F);
  test.AddAttribute("num_groups", int64_t(3));

  vector<float> input = {3.1513367F, 9.283596F, 1.4546119F, 5.4617004F,
                         8.519701F, 1.2382338F, 1.7930176F, 5.1099434F,
                         7.9195533F, 7.638727F, 8.065445F, 3.8082376F,

                         2.3667817F, 2.8248506F, 3.7754705F, 5.861325F,
                         5.058735F, 3.2787242F, 3.6843839F, 9.755121F,
                         2.7902672F, 7.3974323F, 8.283609F, 8.488337F};
  vector<int64_t> input_dims = {2, 3, 4};
  test.AddInput<TypeParam>("X", input_dims, GetTypedArray<TypeParam>(input));

  vector<float> scale = {1.F, 1.F, 1.F};
  vector<int64_t> scale_dims = {3};
  test.AddInput<TypeParam>("scale", scale_dims, GetTypedArray<TypeParam>(scale), true);

  vector<float> B = {0.F, 0.F, 0.F};
  vector<int64_t> B_dims = {3};
  test.AddInput<TypeParam>("bias", B_dims, GetTypedArray<TypeParam>(B), true);

  vector<float> expected_output = {-0.56495477f, 1.48930046f, -1.13334329f, 0.20899761f,
                                   1.46688162f, -0.98600774f, -0.79911913f, 0.31824524f,
                                   0.57370438f, 0.42193634f, 0.6525492f, -1.64818992f,

                                   -0.92380346f, -0.60808484f, 0.04711878f, 1.48476953f,
                                   -0.14644464f, -0.82262872f, -0.66852817f, 1.63760153f,
                                   -1.65898662f, 0.27618144f, 0.64840618f, 0.734399f,};
  test.AddOutput<TypeParam>("Y", input_dims, GetTypedArray<TypeParam>(expected_output));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCoreMLExecutionProvider(true));
  // coreml EP requires weight and bias to be initializers
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}


TYPED_TEST(GroupNormalizationOpTest, LayerNorm17_opset) {
  auto run_test = [](bool is_initializer) {
    OpTester test("GroupNormalization", 18);
    test.AddAttribute<float>("epsilon", 1e-5f);
    test.AddAttribute("num_groups", int64_t(1));

    std::vector<int64_t> dims{1, 2, 3};
    test.AddInput<TypeParam>("x", dims, GetTypedArray<TypeParam>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}));
    test.AddInput<TypeParam>("scale", {3}, GetTypedArray<TypeParam>({1.0f, 1.0f, 1.0f}), is_initializer);
    test.AddInput<TypeParam>("bias", {3}, GetTypedArray<TypeParam>({.0f, .0f, .0f}), is_initializer);
    test.AddOutput<TypeParam>("output", dims, GetTypedArray<TypeParam>({-1.4638f, -0.8783f, -0.2928f, 0.2928f,  0.8783f,  1.4638f}));

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCoreMLExecutionProvider(true));
    // coreml EP requires weight and bias to be initializers
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  };

  // gamma as initialized will fail for CPUEP
  run_test(true);
}

TYPED_TEST(GroupNormalizationOpTest, groupsizen) {
  OpTester test("GroupNormalization", 18);
  test.AddAttribute("epsilon", 0.3F);
  test.AddAttribute("num_groups", int64_t(2));

  vector<float> input = {-1.1258f, -1.1524f, -0.2506f, -0.4339f,
                        0.8487f,  0.6920f, -0.3160f, -2.1152f,
                        0.3223f, -1.2633f,  0.3500f,  0.3081f,
                        0.1198f,  1.2377f,  1.1168f, -0.2473f,
                        -1.3527f, -1.6959f,  0.5667f,  0.7935f,
                        0.5988f, -1.5551f, -0.3414f,  1.8530f,

                        0.7502f, -0.5855f, -0.1734f,  0.1835f,
                        1.3894f,  1.5863f,  0.9463f, -0.8437f,
                        -0.6136f,  0.0316f, -0.4927f,  0.2484f,
                        0.4397f,  0.1124f,  0.6408f,  0.4412f,
                        -0.1023f,  0.7924f, -0.2897f,  0.0525f,
                        0.5229f,  2.3022f, -1.4689f, -1.5867f};
  vector<int64_t> input_dims = {2, 6, 4};
  test.AddInput<TypeParam>("X", input_dims, GetTypedArray<TypeParam>(input));

  vector<float> scale = {1.F, 1.F, 1.F, 1.F, 1.F, 1.F};
  vector<int64_t> scale_dims = {6};
  test.AddInput<TypeParam>("scale", scale_dims, GetTypedArray<TypeParam>(scale), true);

  vector<float> B = {.0F, .0F, .0F, .0F, .0F, .0F};
  vector<int64_t> B_dims = {6};
  test.AddInput<TypeParam>("bias", B_dims, GetTypedArray<TypeParam>(B), true);

  vector<float> expected_output = {-0.7590f, -0.7848f,  0.0914f, -0.0867f,
                                  1.1595f,  1.0073f,  0.0278f, -1.7203f,
                                  0.6480f, -0.8926f,  0.6749f,  0.6343f,
                                  0.0232f,  0.9274f,  0.8296f, -0.2738f,
                                  -1.1679f, -1.4456f,  0.3846f,  0.5681f,
                                  0.4107f, -1.3317f, -0.3499f,  1.4252f,

                                  0.5772f, -0.8298f, -0.3957f, -0.0198f,
                                  1.2505f,  1.4580f,  0.7838f, -1.1017f,
                                  -0.8594f, -0.1798f, -0.7320f,  0.0486f,
                                  0.2541f, -0.0377f,  0.4334f,  0.2554f,
                                  -0.2291f,  0.5686f, -0.3962f, -0.0911f,
                                  0.3282f,  1.9145f, -1.4475f, -1.5525f,};
  test.AddOutput<TypeParam>("Y", input_dims, GetTypedArray<TypeParam>(expected_output));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCoreMLExecutionProvider(true));
  // coreml EP requires weight and bias to be initializers
  if constexpr (std::is_same<TypeParam, float>::value) {
    test.SetOutputTolerance(1e-4f);
  } else {
    test.SetOutputTolerance(0.005f);
  }
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}


}  // namespace test
}  // namespace onnxruntime
#endif
