// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {
// Returns a vector of `count` values which start at `start` and change by increments of `step`.
template <typename T>
std::vector<T> ValueRange(
    size_t count, T start = static_cast<T>(0), T step = static_cast<T>(1)) {
  std::vector<T> result;
  result.reserve(count);
  T curr = start;
  for (size_t i = 0; i < count; ++i) {
    result.emplace_back(curr);
    curr += step;
  }
  return result;
}
}  // namespace

TEST(GatherNDOpTest, GatherND_scalar_string_int32) {
  OpTester test1("GatherND", 1, onnxruntime::kOnnxDomain);
  test1.AddInput<std::string>("data", {2, 2}, {"h", "k", "o", "z"});
  test1.AddInput<int32_t>("indices", {2}, {0, 1});
  test1.AddOutput<std::string>("output", {}, {"k"});
  test1.Run();

  OpTester test2("GatherND", 1, onnxruntime::kOnnxDomain);
  test2.AddInput<std::string>("data", {6}, {"h", "k", "o", "z", "l", "t"});
  test2.AddInput<int32_t>("indices", {1}, {3});
  test2.AddOutput<std::string>("output", {}, {"z"});
  test2.Run();

  OpTester test3("GatherND", 1, onnxruntime::kOnnxDomain);
  test3.AddInput<std::string>("data", {3, 2}, {"h", "k", "o", "z", "l", "t"});
  test3.AddInput<int32_t>("indices", {2}, {2, 1});
  test3.AddOutput<std::string>("output", {}, {"t"});
  test3.Run();
}

TEST(GatherNDOpTest, GatherND_matrix_int64_int64) {
  OpTester test("GatherND", 1, onnxruntime::kOnnxDomain);
  test.AddInput<int64_t>("data", {2, 2}, {0LL, 1LL, 2LL, 3LL});
  test.AddInput<int64_t>("indices", {2, 2}, {0LL, 0LL, 1LL, 1LL});
  test.AddOutput<int64_t>("output", {2}, {0LL, 3LL});
  test.Run();
}

TEST(GatherNDOpTest, GatherND_matrix_string_int64) {
  OpTester test("GatherND", 1, onnxruntime::kOnnxDomain);
  test.AddInput<std::string>("data", {2, 2}, {"a", "b", "c", "d"});
  test.AddInput<int64_t>("indices", {2, 2}, {0LL, 0LL, 1LL, 1LL});
  test.AddOutput<std::string>("output", {2}, {"a", "d"});
  test.Run();
}

TEST(GatherNDOpTest, GatherND_matrix_int64_int32) {
  OpTester test("GatherND", 1, onnxruntime::kOnnxDomain);
  test.AddInput<int64_t>("data", {2, 2}, {0LL, 1LL, 2LL, 3LL});
  test.AddInput<int32_t>("indices", {2, 2}, {0, 0, 1, 1});
  test.AddOutput<int64_t>("output", {2}, {0LL, 3LL});
  test.Run();
}

TEST(GatherNDOpTest, GatherND_matrix_string_int32) {
  OpTester test1("GatherND", 1, onnxruntime::kOnnxDomain);
  test1.AddInput<std::string>("data", {2, 2, 2}, {"egg", "dance", "air", "bob", "terry", "smart", "laugh", "kite"});
  test1.AddInput<int32_t>("indices", {2, 1, 2}, {0, 1, 1, 0});
  test1.AddOutput<std::string>("output", {2, 1, 2}, {"air", "bob", "terry", "smart"});
  test1.Run();

  OpTester test2("GatherND", 1, onnxruntime::kOnnxDomain);
  test2.AddInput<std::string>("data", {3, 3}, {"egg", "dance", "air", "bob", "terry", "smart", "laugh", "kite", "hop"});
  test2.AddInput<int32_t>("indices", {3, 2}, {2, 1, 1, 0, 0, 1});
  test2.AddOutput<std::string>("output", {3}, {"kite", "bob", "dance"});
  test2.Run();
}

TEST(GatherNDOpTest, GatherND_slice_float_int64_t) {
  OpTester test("GatherND", 1, onnxruntime::kOnnxDomain);
  test.AddInput<float>("data", {2, 2}, {0.0f, 0.1f, 0.2f, 0.3f});
  test.AddInput<int64_t>("indices", {2, 1}, {1LL, 0LL});
  test.AddOutput<float>("output", {2, 2}, {0.2f, 0.3f, 0.0f, 0.1f});
  test.Run();
}

TEST(GatherNDOpTest, GatherND_slice_float_int64_t_axis_0) {
  OpTester test("GatherND", 1, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddInput<float>("data", {2, 3, 4}, ValueRange(24, 1.0f));
  test.AddInput<int64_t>("indices", {3, 2, 2}, {0LL, 1LL, 0LL, 2LL, 1LL, 0LL, 0LL, 0LL, 1LL, 1LL, 1LL, 2LL});
  test.AddOutput<float>("output", {3, 2, 4}, {5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 1.0, 2.0, 3.0, 4.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0});
  test.Run();
}

TEST(GatherNDOpTest, GatherND_slice_float_int64_t_axis_1) {
  OpTester test("GatherND", 1, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<float>("data", {2, 3, 4}, ValueRange(24, 1.0f));
  test.AddInput<int64_t>("indices", {2, 2, 2}, {0LL, 1LL, 0LL, 2LL, 1LL, 0LL, 0LL, 0LL});
  test.AddOutput<float>("output", {2, 2}, {2.0, 3.0, 17.0, 13.0});
  test.Run();
}

TEST(GatherNDOpTest, GatherND_slice_float_int32_t_axis_2) {
  OpTester test("GatherND", 1, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<float>("data", {2, 2, 2}, ValueRange(8, 0.0f, 0.1f));
  test.AddInput<int32_t>("indices", {2, 1}, {1LL, 0LL});
  test.AddOutput<float>("output", {2, 2}, {0.2f, 0.3f, 0.4f, 0.5f});
  test.Run();
}

#ifdef USE_CUDA
#if __CUDA_ARCH__ >= 600
TEST(GatherNDOpTest, GatherND_slice_double_int64_t_axis_3) {
  OpTester test("GatherND", 1, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<double>("data", {2, 2, 2}, ValueRange(8, 0.0f, 0.1f));
  test.AddInput<int64_t>("indices", {2, 1, 1}, {1LL, 0LL});
  test.AddOutput<double>("output", {2, 1, 2}, {0.2f, 0.3f, 0.4f, 0.5f});
  test.Run();
}

TEST(GatherNDOpTest, GatherND_slice_double_int32_t) {
  OpTester test("GatherND", 1, onnxruntime::kOnnxDomain);
  test.AddInput<double>("data", {2, 2}, {0.0f, 0.1f, 0.2f, 0.3f});
  test.AddInput<int32_t>("indices", {2, 1}, {1LL, 0LL});
  test.AddOutput<double>("output", {2, 2}, {0.2f, 0.3f, 0.0f, 0.1f});
  test.Run();
}
#endif
#endif

TEST(GatherNDOpTest, GatherND_slice_float_int64_t_axis_4) {
  OpTester test("GatherND", 1, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<float>("data", {2, 2, 2}, ValueRange(8, 0.0f, 0.1f));
  test.AddInput<int64_t>("indices", {2, 1, 2}, {1LL, 0LL, 0LL, 1LL});
  test.AddOutput<float>("output", {2, 1}, {0.2f, 0.5f});
  test.Run();
}

TEST(GatherNDOpTest, GatherND_3tensor_int64) {
  OpTester test1("GatherND", 1, onnxruntime::kOnnxDomain);
  test1.AddInput<int64_t>("data", {2, 2, 2}, ValueRange<int64_t>(8));
  test1.AddInput<int64_t>("indices", {2, 2}, {0LL, 1LL, 1LL, 0LL});
  test1.AddOutput<int64_t>("output", {2, 2}, {2LL, 3LL, 4LL, 5LL});
  test1.Run();

  OpTester test2("GatherND", 1, onnxruntime::kOnnxDomain);
  test2.AddInput<int8_t>("data", {2, 2, 2}, ValueRange<int8_t>(8));
  test2.AddInput<int32_t>("indices", {2, 3}, {0, 0, 1, 1, 0, 1});
  test2.AddOutput<int8_t>("output", {2}, {1, 5});
  test2.Run();

  OpTester test3("GatherND", 1, onnxruntime::kOnnxDomain);
  test3.AddInput<int16_t>("data", {2, 2, 2}, ValueRange<int16_t>(8));
  test3.AddInput<int64_t>("indices", {1, 1}, {1LL});
  test3.AddOutput<int16_t>("output", {1, 2, 2}, {4, 5, 6, 7});
  test3.Run();
}

TEST(GatherNDOpTest, GatherND_batched_index_int64) {
  OpTester test("GatherND", 1, onnxruntime::kOnnxDomain);
  test.AddInput<int64_t>("data", {2, 2}, {0LL, 1LL, 2LL, 3LL});
  test.AddInput<int64_t>("indices", {2, 1, 2}, {0LL, 0LL, 0LL, 1LL});
  test.AddOutput<int64_t>("output", {2, 1}, {0LL, 1LL});
  test.Run();
}

TEST(GatherNDOpTest, GatherND_batched_index_bool_int64) {
  OpTester test("GatherND", 1, onnxruntime::kOnnxDomain);
  test.AddInput<bool>("data", {2, 2}, {true, false, false, true});
  test.AddInput<int64_t>("indices", {2, 1, 2}, {0LL, 0LL, 0LL, 1LL});
  test.AddOutput<bool>("output", {2, 1}, {true, false});
  test.Run();
}

TEST(GatherNDOpTest, GatherND_sliced_index_int64) {
  OpTester test("GatherND", 1, onnxruntime::kOnnxDomain);
  test.AddInput<int64_t>("data", {2, 2}, {0LL, 1LL, 2LL, 3LL});
  test.AddInput<int64_t>("indices", {2, 1, 1}, {1LL, 0LL});
  test.AddOutput<int64_t>("output", {2, 1, 2}, {2LL, 3LL, 0LL, 1LL});
  test.Run();
}

TEST(GatherNDOpTest, GatherND_sliced_index_string_int32) {
  OpTester test("GatherND", 1, onnxruntime::kOnnxDomain);
  test.AddInput<std::string>("data", {2, 2}, {"ab", "cde", "f", "ghi"});
  test.AddInput<int32_t>("indices", {2, 1, 1}, {1LL, 0LL});
  test.AddOutput<std::string>("output", {2, 1, 2}, {"f", "ghi", "ab", "cde"});
  test.Run();
}

TEST(GatherNDOpTest, GatherND_batched_3tensor_int64) {
  OpTester test1("GatherND", 1, onnxruntime::kOnnxDomain);
  test1.AddInput<uint32_t>("data", {2, 2, 2}, ValueRange<uint32_t>(8));
  test1.AddInput<int64_t>("indices", {2, 2, 2}, {0LL, 1LL, 1LL, 0LL, 0LL, 0LL, 1LL, 1LL});
  test1.AddOutput<uint32_t>("output", {2, 2, 2}, {2, 3, 4, 5, 0, 1, 6, 7});
  test1.Run();

  OpTester test2("GatherND", 1, onnxruntime::kOnnxDomain);
  test2.AddInput<uint32_t>("data", {2, 2, 2}, ValueRange<uint32_t>(8));
  test2.AddInput<int32_t>("indices", {2, 2, 3}, {0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0});
  test2.AddOutput<uint32_t>("output", {2, 2}, {1, 5, 3, 6});
  test2.Run();

  OpTester test3("GatherND", 1, onnxruntime::kOnnxDomain);
  test3.AddInput<int64_t>("data", {2, 2, 2}, ValueRange<int64_t>(8));
  test3.AddInput<int32_t>("indices", {2, 1, 1}, {1, 0});
  test3.AddOutput<int64_t>("output", {2, 1, 2, 2}, {4LL, 5LL, 6LL, 7LL, 0LL, 1LL, 2LL, 3LL});
  test3.Run();
}

#ifdef USE_CUDA
TEST(GatherNDOpTest, GatherNDGrad_slice_float_int64_t_axis_1) {
  OpTester test("GatherNDGrad", 1, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddInput<int64_t>("shape", {3}, {2LL, 2LL, 3LL});
  test.AddInput<int64_t>("indices", {2, 2}, {0LL, 1LL, 1LL, 0LL});
  test.AddInput<float>("update", {2, 3}, ValueRange(6, 1.0f));
  test.AddOutput<float>("output", {2, 2, 3}, {0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0});
  test.Run();
}
#endif

#ifdef USE_CUDA
#if __CUDA_ARCH__ >= 600
TEST(GatherNDOpTest, GatherNDGrad_slice_double_int32_t_axis_3) {
  OpTester test("GatherNDGrad", 1, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<int64_t>("shape", {3}, {2LL, 2LL, 3LL});
  test.AddInput<int32_t>("indices", {2, 1, 1}, {1LL, 0LL});
  test.AddInput<double>("update", {2, 3}, ValueRange(6, 1.0));
  test.AddOutput<double>("output", {2, 2, 3}, {0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0});
  test.Run();
}

TEST(GatherNDOpTest, GatherND_slice_double_int64_t_axis_3) {
  OpTester test("GatherND", 1, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<double>("data", {2, 2, 2}, ValueRange(8, 0.0, 0.1));
  test.AddInput<int64_t>("indices", {2, 1, 1}, {1LL, 0LL});
  test.AddOutput<double>("output", {2, 1, 2}, {0.2f, 0.3f, 0.4f, 0.5f});
  test.Run();
}
#endif

#if __CUDA_ARCH__ >= 700
TEST(GatherNDOpTest, GatherNDGrad_slice_half_int32_t_axis_3) {
  OpTester test("GatherNDGrad", 1, onnxruntime::kOnnxDomain);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddInput<int64_t>("shape", {3}, {2LL, 2LL, 3LL});
  test.AddInput<int32_t>("indices", {2, 1, 1}, {1LL, 0LL});
  std::vector<float> updates_f = ValueRange(6, 1.0f);
  std::vector<float> outputs_f({0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0});
  std::vector<MLFloat16> updates(6);
  std::vector<MLFloat16> outputs(12);
  ConvertFloatToMLFloat16(updates_f.data(), updates.data(), 6);
  ConvertFloatToMLFloat16(outputs_f.data(), outputs.data(), 12);
  test.AddInput<MLFloat16>("update", {2, 3}, updates);
  test.AddOutput<MLFloat16>("output", {2, 2, 3}, outputs);
  test.Run();
}

TEST(GatherNDOpTest, GatherND_slice_half_int32_t) {
  OpTester test("GatherND", 1, onnxruntime::kOnnxDomain);
  std::vector<float> data_f({0.0f, 0.1f, 0.2f, 0.3f});
  std::vector<float> outputs_f({0.2f, 0.3f, 0.0f, 0.1f});
  std::vector<MLFloat16> data(4);
  std::vector<MLFloat16> outputs(4);
  ConvertFloatToMLFloat16(data_f.data(), data.data(), 4);
  ConvertFloatToMLFloat16(outputs_f.data(), outputs.data(), 4);
  test.AddInput<MLFloat16>("data", {2, 2}, data);
  test.AddInput<int32_t>("indices", {2, 1}, {1LL, 0LL});
  test.AddOutput<MLFloat16>("output", {2, 2}, outputs);
  test.Run();
}
#endif
#endif

#ifdef USE_CUDA
TEST(GatherNDOpTest, GatherND_axis_of_2) {
  OpTester test("GatherND", 1, kOnnxDomain);
  test.AddAttribute<int64_t>("axis", 2);
  test.AddInput<int32_t>("data", {2, 2, 2, 2, 3}, ValueRange<int32_t>(48));
  test.AddInput<int32_t>(
      "indices", {2, 2, 1, 2},
      {
          0, 0,  // batch 0
          1, 0,  // batch 1
          1, 1,  // batch 2
          0, 1,  // batch 3
      });
  test.AddOutput<int32_t>(
      "output", {2, 2, 1, 3},
      {
          0, 1, 2,     // batch 0
          18, 19, 20,  // batch 1
          33, 34, 35,  // batch 2
          39, 40, 41,  // batch 3
      });
  test.Run();
}

TEST(GatherNDOpTest, GatherNDGrad_axis_of_2) {
  OpTester test("GatherNDGrad", 1, kOnnxDomain);
  test.AddAttribute<int64_t>("axis", 2);
  test.AddInput<int64_t>("shape", {4}, {2, 2, 2, 3});
  test.AddInput<int64_t>(
      "indices", {2, 2, 1},
      {
          1,  // batch 0
          1,  // batch 1
          0,  // batch 2
          1,  // batch 3
      });
  test.AddInput<float>("update", {2, 2, 3}, ValueRange<float>(12));
  test.AddOutput<float>(
      "output", {2, 2, 2, 3},
      {
          0, 0, 0, 0, 1, 2,    // batch 0
          0, 0, 0, 3, 4, 5,    // batch 1
          6, 7, 8, 0, 0, 0,    // batch 2
          0, 0, 0, 9, 10, 11,  // batch 3
      });
  test.Run();
}
#endif

}  // namespace test
}  // namespace onnxruntime
