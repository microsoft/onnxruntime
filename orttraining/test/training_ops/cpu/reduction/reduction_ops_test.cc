// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>
#include <cmath>
#include <type_traits>
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
//#include "test/providers/cpu/reduction/reduction_test_cases.h"
#ifdef USE_CUDA
#include "core/providers/cuda/reduction/reduction_functions.h"
#endif

namespace onnxruntime {
namespace test {

#ifdef USE_CUDA

void test_all_1d_true(size_t size) {
  std::unique_ptr<bool[]> p_data(new bool[size]);
  for (size_t i = 0; i < size; ++i) {
    p_data[i] = true;
  }

  OpTester test("All", 9);
  test.AddInput<bool>("data", {static_cast<int64_t>(size)}, p_data.get(), size);
  test.AddOutput<bool>("result", {1}, {true});
  test.Run();
}

void test_all_1d_false(size_t size) {
  std::unique_ptr<bool[]> p_data(new bool[size]);
  for (size_t i = 0; i < size; ++i) {
    p_data[i] = false;
  }

  OpTester test("All", 9);
  test.AddInput<bool>("data", {static_cast<int64_t>(size)}, p_data.get(), size);
  test.AddOutput<bool>("result", {1}, {false});
  test.Run();
}

void test_all_1d_first_false(size_t size) {
  std::unique_ptr<bool[]> p_data(new bool[size]);
  for (size_t i = 0; i < size; ++i) {
    p_data[i] = true;
  }
  p_data[0] = false;

  OpTester test("All", 9);
  test.AddInput<bool>("data", {static_cast<int64_t>(size)}, p_data.get(), size);
  test.AddOutput<bool>("result", {1}, {false});
  test.Run();
}

void test_all_1d_last_false(size_t size) {
  std::unique_ptr<bool[]> p_data(new bool[size]);
  for (size_t i = 0; i < size; ++i) {
    p_data[i] = true;
  }
  p_data[size - 1] = false;

  OpTester test("All", 9);
  test.AddInput<bool>("data", {static_cast<int64_t>(size)}, p_data.get(), size);
  test.AddOutput<bool>("result", {1}, {false});
  test.Run();
}

TEST(AllOpTest, All_1d_small) {
  for (size_t i = 1; i < 256; ++i) {
    test_all_1d_false(i);
    test_all_1d_first_false(i);
    test_all_1d_last_false(i);
    test_all_1d_true(i);
  }
}

TEST(AllOpTest, All_1d_large) {
  std::vector<int> centers = {1228, 8877};
  for (auto it = centers.begin(); it != centers.end(); ++it) {
    for (int j = -32; j <= 32; ++j) {
      test_all_1d_first_false(*it + j);
      test_all_1d_last_false(*it + j);
    }
  }
}

TEST(ReductionOpTest, ReduceAllL2) {
  OpTester test("ReduceAllL2", 9, onnxruntime::kOnnxDomain, true);
  std::vector<float> data0 = {1.0f, 2.0f, 3.0f};
  std::vector<float> data1 = {-1.0f, -2.0f};

  test.AddInput<float>("data0", {3}, data0);
  test.AddInput<float>("data1", {2}, data1);
  test.AddOutput<float>("reduced", {}, {4.358898943540674f});
  test.Run();
}

TEST(ReductionOpTest, ReduceAllL2HalfHalf) {
  OpTester test("ReduceAllL2", 9, onnxruntime::kOnnxDomain, true);

  std::vector<float> data0 = {1.0f, 2.0f, 3.0f};
  std::vector<MLFloat16> data0_half(3);
  ConvertFloatToMLFloat16(data0.data(), data0_half.data(), 3);

  std::vector<float> data1 = {-1.0f, -2.0f};
  std::vector<MLFloat16> data1_half(2);
  ConvertFloatToMLFloat16(data1.data(), data1_half.data(), 2);

  std::vector<float> result = {4.358898943540674f};
  std::vector<MLFloat16> result_half(1);
  ConvertFloatToMLFloat16(result.data(), result_half.data(), 1);

  test.AddInput<MLFloat16>("data0", {3}, data0_half);
  test.AddInput<MLFloat16>("data1", {2}, data1_half);

  test.AddOutput<MLFloat16>("reduced", {}, result_half);
  test.Run();
}

TEST(ReductionOpTest, ReduceAllL2FloatHalf) {
  OpTester test("ReduceAllL2", 9, onnxruntime::kOnnxDomain, true);

  std::vector<float> data0 = {1.0f, 2.0f, 3.0f};
  std::vector<float> data1 = {-1.0f, -2.0f};

  test.AddInput<float>("data0", {3}, data0);
  test.AddInput<float>("data1", {2}, data1);

  std::vector<float> result = {4.358898943540674f};
  std::vector<MLFloat16> result_half(1);
  ConvertFloatToMLFloat16(result.data(), result_half.data(), 1);

  test.AddOutput<MLFloat16>("reduced", {}, result_half);
  test.Run();
}

TEST(ReductionOpTest, ReduceAllL2HalfFloat) {
  OpTester test("ReduceAllL2", 9, onnxruntime::kOnnxDomain, true);

  std::vector<float> data0 = {1.0f, 2.0f, 3.0f};
  std::vector<MLFloat16> data0_half(3);
  ConvertFloatToMLFloat16(data0.data(), data0_half.data(), 3);

  std::vector<float> data1 = {-1.0f, -2.0f};
  std::vector<MLFloat16> data1_half(2);
  ConvertFloatToMLFloat16(data1.data(), data1_half.data(), 2);

  std::vector<float> result = {4.358898943540674f};

  test.AddInput<MLFloat16>("data0", {3}, data0_half);
  test.AddInput<MLFloat16>("data1", {2}, data1_half);

  test.AddOutput<float>("reduced", {}, result);
  test.Run();
}

void TestMultiTensorReduce(
    const int tensor_count,
    const int min_tensor_size,
    const int max_tensor_size,
    const float min,
    const float max) {
  OpTester test("ReduceAllL2", 9, onnxruntime::kOnnxDomain, true);

  // Set up random number generator.
  std::random_device random_device;
  std::mt19937 random_engine(0);
  std::uniform_real_distribution<float> dist(min, max);
  std::uniform_int_distribution<int64_t> dist_int(min_tensor_size, max_tensor_size);

  // Initialize tensor-related variables.
  std::vector<int64_t> sizes(tensor_count);
  std::vector<std::vector<int64_t>> shapes(tensor_count);
  std::vector<std::vector<float>> ws(tensor_count);

  double result = 0.f;

  // Generate tensors and compute their reduction result.
  for (int64_t i = 0; i < tensor_count; ++i) {
    const auto size = dist_int(random_engine);
    sizes[i] = size;
    shapes[i] = std::vector<int64_t>(1, size);
    ws[i] = std::vector<float>(sizes[i]);

    for (int64_t j = 0; j < sizes[i]; ++j) {
      ws[i][j] = 1.f;  //dist(random_engine);
      result += ws[i][j] * ws[i][j];
    }

    std::string w_name = "data_" + std::to_string(i);
    test.AddInput<float>(w_name.c_str(), shapes[i], ws[i]);
  }
  test.AddOutput<float>("reduced", {}, {static_cast<float>(std::sqrt(result))});

  test.Run();
}

TEST(ReductionOpTest, ReduceAllL2LargeOne) {
  TestMultiTensorReduce(16, 1, 131072, 1.f, 1.f);
}

TEST(ReductionOpTest, ReduceAllL2Large) {
  TestMultiTensorReduce(16, 1, 131072, 1.2f, 1.3f);
}

TEST(ReductionOpTest, ReduceAllL2ManyOne) {
  TestMultiTensorReduce(4096, 1, 8, 1.f, 1.f);
}

TEST(ReductionOpTest, ReduceAllL2Many) {
  TestMultiTensorReduce(4096, 1, 8, 1.2f, 1.3f);
}

#endif

}  // namespace test
}  // namespace onnxruntime
