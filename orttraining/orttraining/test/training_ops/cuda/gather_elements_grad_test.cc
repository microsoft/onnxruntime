// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace cuda {
namespace test {

TEST(GatherElementsGrad, WithoutAxis) {
  onnxruntime::test::OpTester test("GatherElementsGrad", 1, kMSDomain);
  test.AddInput<float>("dY", {2, 3},
                       {1.0f, 1.1f, 1.2f,
                        2.0f, 2.1f, 2.2f});
  std::vector<int64_t> data_shape = {3, 3};
  test.AddInput<int64_t>("data_shape", {2}, data_shape);
  test.AddInput<int64_t>("indices", {2, 3},
                         {1, 0, 2,
                          0, 2, 1});
  test.AddOutput<float>("dX", {3, 3},
                        {2.0f, 1.1f, 0.0f,
                         1.0f, 0.0f, 2.2f,
                         0.0f, 2.1f, 1.2f});
  test.Run();
}

TEST(GatherElementsGrad, WithAxis) {
  onnxruntime::test::OpTester test("GatherElementsGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<float>("dY", {1, 2}, {1.1f, 2.1f});
  std::vector<int64_t> data_shape = {1, 5};
  test.AddInput<int64_t>("data_shape", {2}, data_shape);
  test.AddInput<int64_t>("indices", {1, 2}, {1, 3});
  test.AddOutput<float>("dX", {1, 5}, {0.0f, 1.1f, 0.0f, 2.1f, 0.0f});
  test.Run();
}

TEST(GatherElementsGrad, ThreeDimsWithAxis_0) {
  onnxruntime::test::OpTester test("GatherElementsGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", 0);

  test.AddInput<float>("dY", {1, 3, 3},
                       {11.0f, 12.0f, 13.0f,
                        14.0f, 15.0f, 16.0f,
                        17.0f, 18.0f, 19.0f});

  std::vector<int64_t> data_shape = {1, 3, 3};
  test.AddInput<int64_t>("data_shape", {3}, data_shape);

  // Because axis 0 is only 1 dimension it should be all zeros
  test.AddInput<int64_t>("indices", {1, 3, 3},
                         {0, 0, 0,
                          0, 0, 0,
                          0, 0, 0});

  test.AddOutput<float>("dX", {1, 3, 3},
                        {11.0f, 12.0f, 13.0f,
                         14.0f, 15.0f, 16.0f,
                         17.0f, 18.0f, 19.0f});
  test.Run();
}

TEST(GatherElementsGrad, ThreeDimsWithAxis_2) {
  onnxruntime::test::OpTester test("GatherElementsGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", 2);

  test.AddInput<float>("dY", {1, 3, 3},
                       {11, 12, 13,
                        14, 15, 16,
                        17, 18, 19});

  std::vector<int64_t> data_shape = {1, 3, 3};
  test.AddInput<int64_t>("data_shape", {3}, data_shape);

  test.AddInput<int64_t>("indices", {1, 3, 3},
                         {2, 1, 0,
                          2, 1, 0,
                          2, 1, 0});

  test.AddOutput<float>("dX", {1, 3, 3},
                        {13, 12, 11,
                         16, 15, 14,
                         19, 18, 17});
  test.Run();
}

TEST(GatherElementsGrad, NegativeAxis) {
  onnxruntime::test::OpTester test("GatherElementsGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", -1);
  test.AddInput<float>("dY", {1, 2}, {1.1f, 2.1f});
  std::vector<int64_t> data_shape = {1, 5};
  test.AddInput<int64_t>("data_shape", {2}, data_shape);
  test.AddInput<int64_t>("indices", {1, 2}, {1, 3});
  test.AddOutput<float>("dX", {1, 5}, {0.0f, 1.1f, 0.0f, 2.1f, 0.0f});
  test.Run();
}

TEST(GatherElementsGrad, IndicesUpdatesDontMatch) {
  onnxruntime::test::OpTester test("GatherElementsGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<float>("dY", {1, 2}, {1.1f, 2.1f});
  std::vector<int64_t> data_shape = {1, 5};
  test.AddInput<int64_t>("data_shape", {2}, data_shape);
  test.AddInput<int64_t>("indices", {1, 3}, {1, 3, 3});
  test.AddOutput<float>("dX", {1, 5}, {1.0f, 3.1f, 3.0f, 6.1f, 5.0f});
  test.Run(onnxruntime::test::OpTester::ExpectResult::kExpectFailure, "Indices vs dY dimensions differs at position=1 3 vs 2");
}

TEST(GatherElementsGrad, ValidAxis) {
  onnxruntime::test::OpTester test("GatherElementsGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddInput<float>("dY", {1, 1, 1}, {5.0f});
  std::vector<int64_t> data_shape = {4, 2, 1};
  test.AddInput<int64_t>("data_shape", {3}, data_shape);
  test.AddInput<int64_t>("indices", {1, 1, 1}, {3});
  test.AddOutput<float>("dX", {4, 2, 1}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f});
  test.Run();
}

TEST(GatherElementsGrad, ValidNegativeIndex) {
  onnxruntime::test::OpTester test("GatherElementsGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddInput<float>("dY", {1, 1, 1}, {5.0f});
  std::vector<int64_t> data_shape = {4, 2, 1};
  test.AddInput<int64_t>("data_shape", {3}, data_shape);
  test.AddInput<int64_t>("indices", {1, 1, 1}, {-1});
  test.AddOutput<float>("dX", {4, 2, 1}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f});
  test.Run();
}

TEST(GatherElementsGrad, SameUpdateWithoutAxis) {
  onnxruntime::test::OpTester test("GatherElementsGrad", 1, kMSDomain);
  test.AddInput<float>("dY", {2, 2},
                       {11.0f, 22.0f,
                        33.0f, 44.0f});

  std::vector<int64_t> data_shape = {3, 3};
  test.AddInput<int64_t>("data_shape", {2}, data_shape);

  test.AddInput<int32_t>("indices", {2, 2},
                         {1, 1,
                          1, 1},
                         true);

  test.AddOutput<float>("dX", {3, 3},
                        {0.0f, 0.0f, 0.0f,
                         44.0f, 66.0f, 0.0f,
                         0.0f, 0.0f, 0.0f});
  test.Run();
}

TEST(GatherElementsGrad, SameUpdateWithAxis) {
  onnxruntime::test::OpTester test("GatherElementsGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<float>("dY", {2, 3},
                       {11.0f, 22.0f, 33.0f,
                        44.0f, 55.0f, 66.0f});

  std::vector<int64_t> data_shape = {3, 3};
  test.AddInput<int64_t>("data_shape", {2}, data_shape);

  test.AddInput<int32_t>("indices", {2, 3},
                         {1, 1, 1,
                          1, 1, 1},
                         true);

  test.AddOutput<float>("dX", {3, 3},
                        {0.0f, 66.0f, 0.0f,
                         0.0f, 165.0f, 0.0f,
                         0.0f, 0.0f, 0.0f});
  test.Run();
}

TEST(GatherElementsGrad, SameUpdateWithNegativeAxis) {
  onnxruntime::test::OpTester test("GatherElementsGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", -1);
  test.AddInput<float>("dY", {2, 3},
                       {11.0f, 22.0f, 33.0f,
                        44.0f, 55.0f, 66.0f});

  std::vector<int64_t> data_shape = {3, 3};
  test.AddInput<int64_t>("data_shape", {2}, data_shape);

  test.AddInput<int32_t>("indices", {2, 3},
                         {1, 0, 1,
                          1, 0, 1},
                         true);

  test.AddOutput<float>("dX", {3, 3},
                        {22.0f, 44.0f, 0.0f,
                         55.0f, 110.0f, 0.0f,
                         0.0f, 0.0f, 0.0f});
  test.Run();
}

TEST(GatherElementsGrad, SameUpdateWithoutAxisMLFloat16) {
  onnxruntime::test::OpTester test("GatherElementsGrad", 1, kMSDomain);
  std::vector<float> update = {11.0f, 22.0f,
                               33.0f, 44.0f};
  std::vector<MLFloat16> fp16_update(update.size());
  onnxruntime::test::ConvertFloatToMLFloat16(update.data(), fp16_update.data(), static_cast<int>(update.size()));

  std::vector<float> output = {0.0f, 0.0f, 0.0f,
                               44.0f, 66.0f, 0.0f,
                               0.0f, 0.0f, 0.0f};
  std::vector<MLFloat16> fp16_output(output.size());
  onnxruntime::test::ConvertFloatToMLFloat16(output.data(), fp16_output.data(), static_cast<int>(output.size()));

  test.AddInput<MLFloat16>("dY", {2, 2}, fp16_update);

  std::vector<int64_t> data_shape = {3, 3};
  test.AddInput<int64_t>("data_shape", {2}, data_shape);

  test.AddInput<int32_t>("indices", {2, 2},
                         {1, 1,
                          1, 1},
                         true);

  test.AddOutput<MLFloat16>("dX", {3, 3}, fp16_output);

  test.Run();
}

TEST(GatherElementsGrad, LargerIndicesOnAxis) {
  onnxruntime::test::OpTester test("GatherElementsGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<float>("dY", {1, 4}, {1.1f, 2.2f, 3.3f, 4.4f});
  std::vector<int64_t> data_shape = {1, 2};
  test.AddInput<int64_t>("data_shape", {2}, data_shape);
  test.AddInput<int64_t>("indices", {1, 4}, {0, 1, 0, 1});
  test.AddOutput<float>("dX", {1, 2}, {4.4f, 6.6f});
  test.Run();
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
