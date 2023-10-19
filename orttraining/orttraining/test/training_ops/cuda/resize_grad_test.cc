// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {

#if defined(USE_CUDA) || defined(USE_ROCM)

namespace {

void AddResizeGradAttributes(OpTester& test, const std::string& coordinate_transformation_mode) {
  test.AddAttribute<std::string>("mode", "linear");
  test.AddAttribute<std::string>("coordinate_transformation_mode", coordinate_transformation_mode);
}

}  // namespace

TEST(ResizeGradTest, ResizeGradWithSizes) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
#ifdef USE_CUDA
  providers.emplace_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  providers.emplace_back(DefaultRocmExecutionProvider());
#endif

  OpTester test("ResizeGrad", 1, onnxruntime::kMSDomain);

  AddResizeGradAttributes(test, "half_pixel");

  std::vector<float> dY(128, 1.0f);
  std::vector<int64_t> dY_shape = {1, 2, 8, 8};

  std::vector<float> X(32, 1.0f);
  std::vector<int64_t> X_shape = {1, 2, 4, 4};

  std::vector<float> dX(32, 4.0f);
  std::vector<int64_t> dX_shape = X_shape;

  test.AddInput<float>("dY", dY_shape, dY);
  test.AddInput<float>("X", X_shape, X);

  test.AddOutput<float>("dX", dX_shape, dX);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(ResizeGradTest, ResizeGradWithSizesHalf) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
#ifdef USE_CUDA
  providers.emplace_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  providers.emplace_back(DefaultRocmExecutionProvider());
#endif

  OpTester test("ResizeGrad", 1, onnxruntime::kMSDomain);

  AddResizeGradAttributes(test, "half_pixel");

  std::vector<float> dY(128, 1.0f);
  std::vector<MLFloat16> dY_half(dY.size());
  ConvertFloatToMLFloat16(dY.data(), dY_half.data(), static_cast<int>(dY.size()));
  std::vector<int64_t> dY_shape = {1, 2, 8, 8};

  std::vector<float> X(32, 1.0f);
  std::vector<MLFloat16> X_half(X.size());
  ConvertFloatToMLFloat16(X.data(), X_half.data(), static_cast<int>(X.size()));
  std::vector<int64_t> X_shape = {1, 2, 4, 4};

  std::vector<float> dX(32, 4.0f);
  std::vector<MLFloat16> dX_half(dX.size());
  ConvertFloatToMLFloat16(dX.data(), dX_half.data(), static_cast<int>(dX.size()));
  std::vector<int64_t> dX_shape = X_shape;

  test.AddInput<MLFloat16>("dY", dY_shape, dY_half);
  test.AddInput<MLFloat16>("X", X_shape, X_half);

  test.AddOutput<MLFloat16>("dX", dX_shape, dX_half);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(ResizeGradTest, ResizeGradWithSizesAndAlignCorners) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
#ifdef USE_CUDA
  providers.emplace_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  providers.emplace_back(DefaultRocmExecutionProvider());
#endif

  OpTester test("ResizeGrad", 1, onnxruntime::kMSDomain);

  AddResizeGradAttributes(test, "align_corners");

  std::vector<float> dY(128, 1.0f);
  std::vector<int64_t> dY_shape = {1, 2, 8, 8};

  std::vector<float> X(32, 1.0f);
  std::vector<int64_t> X_shape = {1, 2, 4, 4};

  std::vector<float> dX({2.9388f, 3.9184f, 3.9184f, 2.9388f, 3.9184f, 5.2245f, 5.2245f, 3.9184f,
                         3.9184f, 5.2245f, 5.2245f, 3.9184f, 2.9388f, 3.9184f, 3.9184f, 2.9388f,
                         2.9388f, 3.9184f, 3.9184f, 2.9388f, 3.9184f, 5.2245f, 5.2245f, 3.9184f,
                         3.9184f, 5.2245f, 5.2245f, 3.9184f, 2.9388f, 3.9184f, 3.9184f, 2.9388f});
  std::vector<int64_t> dX_shape = X_shape;

  test.AddInput<float>("dY", dY_shape, dY);
  test.AddInput<float>("X", X_shape, X);

  test.AddOutput<float>("dX", dX_shape, dX);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(ResizeGradTest, ResizeGradWithScales) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
#ifdef USE_CUDA
  providers.emplace_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  providers.emplace_back(DefaultRocmExecutionProvider());
#endif

  OpTester test("ResizeGrad", 1, onnxruntime::kMSDomain);

  AddResizeGradAttributes(test, "half_pixel");

  std::vector<float> dY(72, 1.0f);
  std::vector<int64_t> dY_shape = {1, 2, 6, 6};

  std::vector<float> X(32, 1.0f);
  std::vector<int64_t> X_shape = {1, 2, 4, 4};

  std::vector<float> dX({2.7128f, 2.9550f, 2.7612f, 1.4533f, 2.9550f, 3.2189f, 3.0078f, 1.5830f,
                         2.7612f, 3.0078f, 2.8106f, 1.4792f, 1.4533f, 1.5830f, 1.4792f, 0.7785f,
                         2.7128f, 2.9550f, 2.7612f, 1.4533f, 2.9550f, 3.2189f, 3.0078f, 1.5830f,
                         2.7612f, 3.0078f, 2.8106f, 1.4792f, 1.4533f, 1.5830f, 1.4792f, 0.7785f});
  std::vector<int64_t> dX_shape = X_shape;

  test.AddInput<float>("dY", dY_shape, dY);
  test.AddInput<float>("X", X_shape, X);
  test.AddInput<float>("", {0}, {});
  test.AddInput<float>("scales", {4}, {1.0f, 1.0f, 1.7f, 1.7f});

  test.AddOutput<float>("dX", dX_shape, dX);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(ResizeGradTest, ResizeGradWithScalesHalf) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
#ifdef USE_CUDA
  providers.emplace_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  providers.emplace_back(DefaultRocmExecutionProvider());
#endif

  OpTester test("ResizeGrad", 1, onnxruntime::kMSDomain);

  AddResizeGradAttributes(test, "half_pixel");

  std::vector<float> dY(72, 1.0f);
  std::vector<MLFloat16> dY_half(dY.size());
  ConvertFloatToMLFloat16(dY.data(), dY_half.data(), static_cast<int>(dY.size()));
  std::vector<int64_t> dY_shape = {1, 2, 6, 6};

  std::vector<float> X(32, 1.0f);
  std::vector<MLFloat16> X_half(X.size());
  ConvertFloatToMLFloat16(X.data(), X_half.data(), static_cast<int>(X.size()));
  std::vector<int64_t> X_shape = {1, 2, 4, 4};

  std::vector<float> dX({2.7128f, 2.9550f, 2.7612f, 1.4533f, 2.9550f, 3.2189f, 3.0078f, 1.5830f,
                         2.7612f, 3.0078f, 2.8106f, 1.4792f, 1.4533f, 1.5830f, 1.4792f, 0.7785f,
                         2.7128f, 2.9550f, 2.7612f, 1.4533f, 2.9550f, 3.2189f, 3.0078f, 1.5830f,
                         2.7612f, 3.0078f, 2.8106f, 1.4792f, 1.4533f, 1.5830f, 1.4792f, 0.7785f});
  std::vector<MLFloat16> dX_half(dX.size());
  ConvertFloatToMLFloat16(dX.data(), dX_half.data(), static_cast<int>(dX.size()));
  std::vector<int64_t> dX_shape = X_shape;

  test.AddInput<MLFloat16>("dY", dY_shape, dY_half);
  test.AddInput<MLFloat16>("X", X_shape, X_half);
  test.AddInput<float>("", {0}, {});
  test.AddInput<float>("scales", {4}, {1.0f, 1.0f, 1.7f, 1.7f});

  test.AddOutput<MLFloat16>("dX", dX_shape, dX_half);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(ResizeGradTest, ResizeGradWithScalesAndAlignCorners) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
#ifdef USE_CUDA
  providers.emplace_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  providers.emplace_back(DefaultRocmExecutionProvider());
#endif

  OpTester test("ResizeGrad", 1, onnxruntime::kMSDomain);

  AddResizeGradAttributes(test, "align_corners");

  std::vector<float> dY(72, 1.0f);
  std::vector<int64_t> dY_shape = {1, 2, 6, 6};

  std::vector<float> X(32, 1.0f);
  std::vector<int64_t> X_shape = {1, 2, 4, 4};

  std::vector<float> dX({1.9600f, 2.2400f, 2.2400f, 1.9600f, 2.2400f, 2.5600f, 2.5600f, 2.2400f,
                         2.2400f, 2.5600f, 2.5600f, 2.2400f, 1.9600f, 2.2400f, 2.2400f, 1.9600f,
                         1.9600f, 2.2400f, 2.2400f, 1.9600f, 2.2400f, 2.5600f, 2.5600f, 2.2400f,
                         2.2400f, 2.5600f, 2.5600f, 2.2400f, 1.9600f, 2.2400f, 2.2400f, 1.9600f});
  std::vector<int64_t> dX_shape = X_shape;

  test.AddInput<float>("dY", dY_shape, dY);
  test.AddInput<float>("X", X_shape, X);
  test.AddInput<float>("", {0}, {});
  test.AddInput<float>("scales", {4}, {1.0f, 1.0f, 1.7f, 1.7f});

  test.AddOutput<float>("dX", dX_shape, dX);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

#endif  // defined(USE_CUDA) || defined(USE_ROCM)

}  // namespace onnxruntime::test
