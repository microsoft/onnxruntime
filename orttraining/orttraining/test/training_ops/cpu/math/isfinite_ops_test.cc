
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>
#include <string>
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

#if defined(USE_CUDA) || defined(USE_ROCM)
TEST(IsFiniteTest, Float) {
  OpTester test("IsFinite", 1, kMSDomain);

  std::vector<int64_t> shape = {3};
  std::vector<float> input = {std::numeric_limits<float>::infinity(),
                              1.0f, std::numeric_limits<float>::quiet_NaN()};

  test.AddInput<float>("X", shape, input);
  test.AddOutput<bool>("Y", shape, {false, true, false});

  test.Run();
}

TEST(IsFiniteTest, Double) {
  OpTester test("IsFinite", 1, kMSDomain);

  std::vector<int64_t> shape = {3};
  std::vector<double> input = {std::numeric_limits<double>::infinity(),
                               1.0f, std::numeric_limits<double>::quiet_NaN()};

  test.AddInput<double>("X", shape, input);
  test.AddOutput<bool>("Y", shape, {false, true, false});

  test.Run();
}

TEST(IsFiniteTest, MLFloat16) {
  OpTester test("IsFinite", 1, kMSDomain);

  std::vector<int64_t> shape = {3};
  std::vector<float> input = {std::numeric_limits<float>::infinity(),
                              1.0f, std::numeric_limits<float>::quiet_NaN()};
  std::vector<MLFloat16> input_half(input.size());
  ConvertFloatToMLFloat16(input.data(), input_half.data(), int(input.size()));

  test.AddInput<MLFloat16>("X", shape, input_half);
  test.AddOutput<bool>("Y", shape, {false, true, false});

  test.Run();
}

TEST(IsAllFiniteTest, FalseFloat) {
  OpTester test("IsAllFinite", 1, kMSDomain);

  std::vector<int64_t> shape = {3};
  std::vector<float> input0 = {9.4f, 1.7f, 3.6f};
  std::vector<float> input1 = {1.2f, 2.8f, std::numeric_limits<float>::infinity()};

  test.AddInput<float>("X0", shape, input0);
  test.AddInput<float>("X1", shape, input1);
  test.AddOutput<bool>("Y", {}, {false});

  test.Run();
}

TEST(IsAllFiniteTest, TrueFloat) {
  OpTester test("IsAllFinite", 1, kMSDomain);

  std::vector<int64_t> shape = {3};
  std::vector<float> input0 = {9.4f, 1.7f, 3.6f};
  std::vector<float> input1 = {7.5f, 1.2f, 2.8f};

  test.AddInput<float>("X0", shape, input0);
  test.AddInput<float>("X1", shape, input1);
  test.AddOutput<bool>("Y", {}, {true});

  test.Run();
}

TEST(IsAllFiniteTest, IsInfOnly) {
  OpTester test("IsAllFinite", 1, kMSDomain);

  std::vector<int64_t> shape = {3};
  std::vector<float> input0 = {9.4f, 1.7f, 3.6f};
  std::vector<float> input1 = {7.5f, 1.2f, std::numeric_limits<float>::quiet_NaN()};

  test.AddInput<float>("X0", shape, input0);
  test.AddInput<float>("X1", shape, input1);
  test.AddAttribute("isinf_only", static_cast<int64_t>(1));
  test.AddOutput<bool>("Y", {}, {true});

  test.Run();
}

TEST(IsAllFiniteTest, IsNaNOnly) {
  OpTester test("IsAllFinite", 1, kMSDomain);

  std::vector<int64_t> shape = {3};
  std::vector<float> input0 = {9.4f, 1.7f, 3.6f};
  std::vector<float> input1 = {7.5f, 1.2f, std::numeric_limits<float>::infinity()};

  test.AddInput<float>("X0", shape, input0);
  test.AddInput<float>("X1", shape, input1);
  test.AddAttribute("isnan_only", static_cast<int64_t>(1));
  test.AddOutput<bool>("Y", {}, {true});

  test.Run();
}

std::vector<std::vector<float>> generate_is_all_finite_test_data(
    const int tensor_count,
    const int max_tensor_size,
    const float infinite_probability, int seed) {
  std::vector<std::vector<float>> tensors(tensor_count);
  std::mt19937 rd(seed);
  std::uniform_int_distribution<int> dist(1, max_tensor_size + 1);
  std::uniform_real_distribution<float> dist_data(0.0f, 1.0f);

  for (int i = 0; i < tensor_count; ++i) {
    const auto tensor_size = dist(rd);
    tensors[i] = std::vector<float>(tensor_size);
    for (int j = 0; j < tensor_size; ++j) {
      if (dist_data(rd) < infinite_probability) {
        tensors[i][j] = std::numeric_limits<float>::infinity();
      } else {
        tensors[i][j] = dist_data(rd);
      }
    }
  }

  if (infinite_probability != 0.0f) {
    auto tensor_index = dist(rd) % tensor_count;
    auto size = tensors[tensor_index].size();
    tensors[tensor_index][size - 1] = std::numeric_limits<float>::infinity();
  }

  return tensors;
}

std::vector<std::vector<float>> generate_is_all_finite_test_data(
    const int tensor_count,
    const int max_tensor_size,
    const bool answer, const int seed) {
  std::vector<std::vector<float>> tensors(tensor_count);
  std::mt19937 rd(seed);
  std::uniform_int_distribution<int> dist(1, max_tensor_size + 1);
  std::uniform_real_distribution<float> dist_data(0.0f, 1.0f);

  for (int i = 0; i < tensor_count; ++i) {
    const auto tensor_size = dist(rd);
    tensors[i] = std::vector<float>(tensor_size);
    for (int j = 0; j < tensor_size; ++j) {
      tensors[i][j] = dist_data(rd);
    }
  }

  if (answer == false) {
    auto tensor_index = dist(rd) % tensor_count;
    auto size = tensors[tensor_index].size();
    tensors[tensor_index][size - 1] = std::numeric_limits<float>::infinity();
  }

  return tensors;
}

TEST(IsAllFiniteTest, MoreFalseFloatTensorLarge) {
  for (int test_count = 0; test_count < 10; ++test_count) {
    OpTester test("IsAllFinite", 1, kMSDomain);
    bool expected_answer = false;
    auto tensors = generate_is_all_finite_test_data(13, 941736, expected_answer, test_count);
    for (int i = 0; i < tensors.size(); ++i) {
      auto name = std::string("X") + std::to_string(i);
      auto size = static_cast<int64_t>(tensors[i].size());
      test.AddInput<float>(name.c_str(), {size}, tensors[i]);
    }
    test.AddOutput<bool>("Y", {}, {expected_answer});
    test.Run();
  }
}

TEST(IsAllFiniteTest, MoreFalseFloatManyBlock) {
  for (int test_count = 0; test_count < 10; ++test_count) {
    OpTester test("IsAllFinite", 1, kMSDomain);
    bool expected_answer = false;
    auto tensors = generate_is_all_finite_test_data(894, 17, expected_answer, test_count);
    for (int i = 0; i < tensors.size(); ++i) {
      auto name = std::string("X") + std::to_string(i);
      auto size = static_cast<int64_t>(tensors[i].size());
      test.AddInput<float>(name.c_str(), {size}, tensors[i]);
    }
    test.AddOutput<bool>("Y", {}, {expected_answer});
    test.Run();
  }
}

TEST(IsAllFiniteTest, MoreFalseFloatMultipleFalse) {
  OpTester test("IsAllFinite", 1, kMSDomain);
  auto tensors = generate_is_all_finite_test_data(1234, 1987, 0.1f, 0);
  for (int i = 0; i < tensors.size(); ++i) {
    auto name = std::string("X") + std::to_string(i);
    auto size = static_cast<int64_t>(tensors[i].size());
    test.AddInput<float>(name.c_str(), {size}, tensors[i]);
  }
  test.AddOutput<bool>("Y", {}, {false});
  test.Run();
}

TEST(IsAllFiniteTest, MoreTrueFloatTensorLarge) {
  OpTester test("IsAllFinite", 1, kMSDomain);
  bool expected_answer = true;
  auto tensors = generate_is_all_finite_test_data(12, 941736, expected_answer, 0);
  for (int i = 0; i < tensors.size(); ++i) {
    auto name = std::string("X") + std::to_string(i);
    auto size = static_cast<int64_t>(tensors[i].size());
    test.AddInput<float>(name.c_str(), {size}, tensors[i]);
  }

  test.AddOutput<bool>("Y", {}, {expected_answer});

  test.Run();
}

TEST(IsAllFiniteTest, MoreFalseFloatManyBlockFloat16) {
  OpTester test("IsAllFinite", 1, kMSDomain);
  bool expected_answer = false;
  auto tensors = generate_is_all_finite_test_data(894, 17, expected_answer, 0);
  for (int i = 0; i < tensors.size(); ++i) {
    auto name = std::string("X") + std::to_string(i);
    auto size = static_cast<int64_t>(tensors[i].size());
    std::vector<MLFloat16> buffer_half(tensors[i].size());
    ConvertFloatToMLFloat16(tensors[i].data(), buffer_half.data(), int(size));
    test.AddInput<MLFloat16>(name.c_str(), {size}, buffer_half);
  }
  test.AddOutput<bool>("Y", {}, {expected_answer});
  test.Run();
}

TEST(IsAllFiniteTest, MoreFalseFloatTensorLargeFloat16) {
  OpTester test("IsAllFinite", 1, kMSDomain);
  bool expected_answer = false;
  auto tensors = generate_is_all_finite_test_data(12, 941736, expected_answer, 0);
  for (int i = 0; i < tensors.size(); ++i) {
    auto name = std::string("X") + std::to_string(i);
    auto size = static_cast<int64_t>(tensors[i].size());
    std::vector<MLFloat16> buffer_half(tensors[i].size());
    ConvertFloatToMLFloat16(tensors[i].data(), buffer_half.data(), int(size));
    test.AddInput<MLFloat16>(name.c_str(), {size}, buffer_half);
  }
  test.AddOutput<bool>("Y", {}, {expected_answer});
  test.Run();
}

#endif

}  // namespace test
}  // namespace onnxruntime