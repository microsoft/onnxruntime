// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

// From ONNX spec example: tensorscatter (4D, linear mode)
TEST(TensorScatterTest, Linear_4D) {
  OpTester test("TensorScatter", 24);
  test.AddAttribute<std::string>("mode", "linear");

  // past_cache: shape (2, 1, 4, 5)
  test.AddInput<float>("past_cache", {2, 1, 4, 5},
                       {1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 4, 3, 2, 1, 0,
                        1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 4, 3, 2, 1, 0});

  // update: shape (2, 1, 1, 5)
  test.AddInput<float>("update", {2, 1, 1, 5},
                       {5, 5, 5, 5, 5,
                        1, 1, 1, 1, 1});

  // write_indices: shape (2,)
  test.AddInput<int64_t>("write_indices", {2}, {1, 2});

  // present_cache: shape (2, 1, 4, 5)
  test.AddOutput<float>("present_cache", {2, 1, 4, 5},
                        {1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 8, 7, 6, 5, 4, 4, 3, 2, 1, 0,
                         1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 1, 1, 1, 1, 1, 4, 3, 2, 1, 0});

  test.Run();
}

// From ONNX spec example: tensorscatter_3d (3D, default axis=-2 -> axis=1)
TEST(TensorScatterTest, Linear_3D) {
  OpTester test("TensorScatter", 24);

  // past_cache: shape (3, 4, 5)
  test.AddInput<float>("past_cache", {3, 4, 5},
                       {1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 5, 4, 3, 2, 1,
                        1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 5, 4, 3, 2, 1,
                        1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 5, 4, 3, 2, 1});

  // update: shape (3, 2, 5)
  test.AddInput<float>("update", {3, 2, 5},
                       {4, 4, 4, 4, 4, 5, 5, 5, 5, 5,
                        6, 6, 6, 6, 6, 7, 7, 7, 7, 7,
                        2, 2, 2, 2, 2, 3, 3, 3, 3, 3});

  // write_indices: shape (3,)
  test.AddInput<int64_t>("write_indices", {3}, {1, 2, 0});

  // present_cache: shape (3, 4, 5)
  test.AddOutput<float>("present_cache", {3, 4, 5},
                        {1, 2, 3, 4, 5, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 4, 3, 2, 1,
                         1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7,
                         2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 8, 7, 6, 5, 4, 5, 4, 3, 2, 1});

  test.Run();
}

// From ONNX spec example: tensorscatter_circular (4D, circular mode)
TEST(TensorScatterTest, Circular_4D) {
  OpTester test("TensorScatter", 24);
  test.AddAttribute<std::string>("mode", "circular");

  // past_cache: shape (2, 1, 4, 5)
  test.AddInput<float>("past_cache", {2, 1, 4, 5},
                       {1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 4, 3, 2, 1, 0,
                        1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 4, 3, 2, 1, 0});

  // update: shape (2, 1, 2, 5)
  test.AddInput<float>("update", {2, 1, 2, 5},
                       {5, 5, 5, 5, 5, 6, 6, 6, 6, 6,
                        1, 1, 1, 1, 1, 2, 2, 2, 2, 2});

  // write_indices: shape (2,)
  test.AddInput<int64_t>("write_indices", {2}, {1, 3});

  // present_cache: shape (2, 1, 4, 5)
  // Batch 0: wi=1, seq_len=2 -> positions 1,2 (no wrap)
  // Batch 1: wi=3, seq_len=2 -> positions 3, 0 (wraps around)
  test.AddOutput<float>("present_cache", {2, 1, 4, 5},
                        {1, 2, 3, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 4, 3, 2, 1, 0,
                         2, 2, 2, 2, 2, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 1, 1, 1, 1, 1});

  test.Run();
}

// No write_indices (defaults to zero) — prefill scenario.
TEST(TensorScatterTest, Linear_NoWriteIndices) {
  OpTester test("TensorScatter", 24);
  test.AddAttribute<std::string>("mode", "linear");

  // past_cache: shape (1, 1, 4, 3)
  test.AddInput<float>("past_cache", {1, 1, 4, 3},
                       {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});

  // update: shape (1, 1, 2, 3) — writes at position 0 (default)
  test.AddInput<float>("update", {1, 1, 2, 3},
                       {1, 2, 3, 4, 5, 6});

  test.AddOptionalInputEdge<int64_t>();

  // present_cache: positions 0,1 filled, 2,3 untouched
  test.AddOutput<float>("present_cache", {1, 1, 4, 3},
                        {1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0});

  test.Run();
}

// Float16 type test
TEST(TensorScatterTest, Linear_Float16) {
  OpTester test("TensorScatter", 24);
  test.AddAttribute<std::string>("mode", "linear");

  std::vector<float> past_f = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  std::vector<float> update_f = {99, 98, 97};
  std::vector<float> expected_f = {1, 2, 3, 99, 98, 97, 7, 8, 9, 10, 11, 12};

  std::vector<MLFloat16> past_fp16(past_f.size());
  std::vector<MLFloat16> update_fp16(update_f.size());
  std::vector<MLFloat16> expected_fp16(expected_f.size());
  for (size_t i = 0; i < past_f.size(); ++i) past_fp16[i] = MLFloat16(past_f[i]);
  for (size_t i = 0; i < update_f.size(); ++i) update_fp16[i] = MLFloat16(update_f[i]);
  for (size_t i = 0; i < expected_f.size(); ++i) expected_fp16[i] = MLFloat16(expected_f[i]);

  // shape (1, 4, 3), axis=-2 -> axis=1
  test.AddInput<MLFloat16>("past_cache", {1, 4, 3}, past_fp16);
  test.AddInput<MLFloat16>("update", {1, 1, 3}, update_fp16);
  test.AddInput<int64_t>("write_indices", {1}, {1});
  test.AddOutput<MLFloat16>("present_cache", {1, 4, 3}, expected_fp16);

  test.Run();
}

// Explicit axis attribute test (axis=1 on a 3D tensor)
TEST(TensorScatterTest, Linear_ExplicitAxis) {
  OpTester test("TensorScatter", 24);
  test.AddAttribute<std::string>("mode", "linear");
  test.AddAttribute<int64_t>("axis", 1);

  // shape (2, 3, 2) — axis=1 means the seq dim is dim 1 (size 3)
  test.AddInput<float>("past_cache", {2, 3, 2},
                       {0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0});

  // update: shape (2, 1, 2)
  test.AddInput<float>("update", {2, 1, 2},
                       {1, 2,
                        3, 4});

  test.AddInput<int64_t>("write_indices", {2}, {0, 2});

  test.AddOutput<float>("present_cache", {2, 3, 2},
                        {1, 2, 0, 0, 0, 0,
                         0, 0, 0, 0, 3, 4});

  test.Run();
}

// Circular wrap-around with multi-position update
TEST(TensorScatterTest, Circular_WrapAround) {
  OpTester test("TensorScatter", 24);
  test.AddAttribute<std::string>("mode", "circular");

  // shape (1, 4, 2), axis=-2 -> axis=1, max_seq=4
  test.AddInput<float>("past_cache", {1, 4, 2},
                       {10, 11, 20, 21, 30, 31, 40, 41});

  // update: 3 positions starting at wi=2 -> positions 2, 3, 0 (wraps)
  test.AddInput<float>("update", {1, 3, 2},
                       {1, 2, 3, 4, 5, 6});

  test.AddInput<int64_t>("write_indices", {1}, {2});

  // pos 2->1,2  pos 3->3,4  pos 0->5,6 (wrapped)
  test.AddOutput<float>("present_cache", {1, 4, 2},
                        {5, 6, 20, 21, 1, 2, 3, 4});

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
