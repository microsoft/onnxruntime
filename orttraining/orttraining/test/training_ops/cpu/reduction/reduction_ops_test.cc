// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>
#include <cmath>
#include <type_traits>
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(USE_CUDA) || defined(USE_ROCM)

void test_all_1d_true(size_t size) {
  std::unique_ptr<bool[]> p_data(new bool[size]);
  for (size_t i = 0; i < size; ++i) {
    p_data[i] = true;
  }

  OpTester test("All", 1, kMSDomain);
  test.AddInput<bool>("data", {static_cast<int64_t>(size)}, p_data.get(), size);
  test.AddOutput<bool>("result", {1}, {true});
  test.Run();
}

void test_all_1d_false(size_t size) {
  std::unique_ptr<bool[]> p_data(new bool[size]);
  for (size_t i = 0; i < size; ++i) {
    p_data[i] = false;
  }

  OpTester test("All", 1, kMSDomain);
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

  OpTester test("All", 1, kMSDomain);
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

  OpTester test("All", 1, kMSDomain);
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
#endif

class ReductionOpTest : public ::testing::TestWithParam<bool> {
 protected:
  bool use_determinism;
};

TEST_P(ReductionOpTest, ReduceAllL2) {
  OpTester test("ReduceAllL2", 1, onnxruntime::kMSDomain, true);
  test.SetDeterminism(GetParam());
  std::vector<float> data0 = {1.0f, 2.0f, 3.0f};
  std::vector<float> data1 = {-1.0f, -2.0f};

  test.AddInput<float>("data0", {3}, data0);
  test.AddInput<float>("data1", {2}, data1);
  test.AddOutput<float>("reduced", {}, {4.358898943540674f});
  test.Run();
}

#if defined(USE_CUDA) || defined(USE_ROCM)
TEST_P(ReductionOpTest, ReduceAllL2HalfHalf) {
  OpTester test("ReduceAllL2", 1, onnxruntime::kMSDomain, true);
  test.SetDeterminism(GetParam());

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

TEST_P(ReductionOpTest, ReduceAllL2FloatHalf) {
  OpTester test("ReduceAllL2", 1, onnxruntime::kMSDomain, true);
  test.SetDeterminism(GetParam());

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

TEST_P(ReductionOpTest, ReduceAllL2HalfFloat) {
  OpTester test("ReduceAllL2", 1, onnxruntime::kMSDomain, true);
  test.SetDeterminism(GetParam());

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
#endif

void TestMultiTensorReduce(
    const int tensor_count,
    const int min_tensor_size,
    const int max_tensor_size,
    const float min,
    const float max,
    bool use_determinism) {
  OpTester test("ReduceAllL2", 1, onnxruntime::kMSDomain, true);
  test.SetDeterminism(use_determinism);

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

TEST_P(ReductionOpTest, ReduceAllL2LargeOne) {
  TestMultiTensorReduce(16, 1, 131072, 1.f, 1.f, GetParam());
}

TEST_P(ReductionOpTest, ReduceAllL2Large) {
  TestMultiTensorReduce(16, 1, 131072, 1.2f, 1.3f, GetParam());
}

TEST_P(ReductionOpTest, ReduceAllL2ManyOne) {
  TestMultiTensorReduce(4096, 1, 8, 1.f, 1.f, GetParam());
}

TEST_P(ReductionOpTest, ReduceAllL2Many) {
  TestMultiTensorReduce(4096, 1, 8, 1.2f, 1.3f, GetParam());
}

// invoke with and without use_determinism flag for session
INSTANTIATE_TEST_SUITE_P(ReductionOpTestWrapper, ReductionOpTest, ::testing::Bool());

TEST(ReductionOpTest, ReduceSumTraining_int32) {
  OpTester test("ReduceSumTraining", 1, onnxruntime::kMSDomain);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<int32_t>("data", {3, 2, 2},
                         {1, 2,
                          3, 4,

                          5, 6,
                          7, 8,

                          9, 10,
                          11, 12});
  test.AddInput<int64_t>("axes", {2}, {0, 2}, true /*is_initializer*/);
  test.AddOutput<int32_t>("reduced", {1, 2, 1}, {33, 45});
  test.Run();
}

TEST(ReductionOpTest, ReduceSumTraining_fast_matrix_reduction) {
  OpTester test("ReduceSumTraining", 1, onnxruntime::kMSDomain);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 4},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddInput<int64_t>("axes", {2}, {0, 1}, true /*is_initializer*/);
  test.AddOutput<float>("reduced", {1, 1}, {78.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSumTraining_default_axes_keepdims) {
  OpTester test("ReduceSumTraining", 1, onnxruntime::kMSDomain);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true /*is_initializer*/);
  test.AddOutput<float>("reduced", {1, 1, 1}, {78.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSumTraining_axes_not_initializer) {
  OpTester test("ReduceSumTraining", 1, onnxruntime::kMSDomain);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddInput<int64_t>("axes", {0}, {});
  test.AddOutput<float>("reduced", {1, 1, 1}, {78.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSumTraining_empty_axes_noop) {
  OpTester test("ReduceSumTraining", 1, onnxruntime::kMSDomain);
  test.AddAttribute("keepdims", (int64_t)1);
  test.AddAttribute("noop_with_empty_axes", (int64_t)1);
  test.AddInput<float>("data", {3, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f,

                        5.0f, 6.0f,
                        7.0f, 8.0f,

                        9.0f, 10.0f,
                        11.0f, 12.0f});
  test.AddInput<int64_t>("axes", {0}, {}, true /*is_initializer*/);
  test.AddOutput<float>("reduced", {3, 2, 2},
                        {1.0f, 2.0f,
                         3.0f, 4.0f,

                         5.0f, 6.0f,
                         7.0f, 8.0f,

                         9.0f, 10.0f,
                         11.0f, 12.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSumTraining_do_not_keepdims) {
  OpTester test("ReduceSumTraining", 1, onnxruntime::kMSDomain);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {1}, {1}, true /*is_initializer*/);
  test.AddOutput<float>("reduced", {1, 2}, {4.0f, 6.0f});
  test.Run();
}

TEST(ReductionOpTest, ReduceSumTraining_neg_axis) {
  OpTester test("ReduceSumTraining", 1, onnxruntime::kMSDomain);
  test.AddAttribute("keepdims", (int64_t)0);
  test.AddInput<float>("data", {1, 2, 2},
                       {1.0f, 2.0f,
                        3.0f, 4.0f});
  test.AddInput<int64_t>("axes", {1}, {-2}, true /*is_initializer*/);
  test.AddOutput<float>("reduced", {1, 2}, {4.0f, 6.0f});
  test.Run();
}

#if defined(USE_CUDA) || defined(USE_ROCM)
TEST(ReductionOpTest, ReduceSumTrainingHalfHalf) {
  OpTester test("ReduceSumTraining", 1, onnxruntime::kMSDomain);
  test.AddAttribute("keepdims", (int64_t)0);

  std::vector<float> data = {1.0f, 2.0f,
                             3.0f, 4.0f,

                             5.0f, 6.0f,
                             7.0f, 8.0f,

                             9.0f, 10.0f,
                             11.0f, 12.0f};
  std::vector<MLFloat16> data_half(12);
  ConvertFloatToMLFloat16(data.data(), data_half.data(), 12);

  std::vector<float> result = {36.0f, 42.0f};
  std::vector<MLFloat16> result_half(2);
  ConvertFloatToMLFloat16(result.data(), result_half.data(), 2);
  test.AddInput<MLFloat16>("data", {3, 2, 2}, data_half);
  test.AddInput<int64_t>("axes", {2}, {0, 1}, true /*is_initializer*/);
  test.AddOutput<MLFloat16>("reduced", {2}, result_half);
  test.Run();
}
#endif

}  // namespace test
}  // namespace onnxruntime
