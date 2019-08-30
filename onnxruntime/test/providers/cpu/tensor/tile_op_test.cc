// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
void RunTest(std::initializer_list<T> input,
             std::initializer_list<int64_t> input_dims,
             std::initializer_list<int64_t> repeat,
             std::initializer_list<int64_t> repeat_dims,
             std::initializer_list<T> output,
             std::initializer_list<int64_t> output_dims) {
  OpTester test("Tile");
  test.AddInput<T>("input", input_dims, input);
  test.AddInput<int64_t>("repeats", repeat_dims, repeat);
  test.AddOutput<T>("output", output_dims, output);
  test.Run();
}

template <typename T>
void RunTestWrapper() {
  // Tile1DWithZeroRepeats
  RunTest<T>({1, 2, 3}, {3}, {0}, {1}, {}, {0});

  // Tile2DWithZeroRepeats
  RunTest<T>({11, 12, 21, 22}, {2, 2}, {2, 0}, {2}, {}, {4, 0});

  // Tile1D
  RunTest<T>({1, 2, 3}, {3}, {3}, {1}, {1, 2, 3, 1, 2, 3, 1, 2, 3}, {9});

  // Tile2D_1Axis
  RunTest<T>({11, 12, 21, 22}, {2, 2}, {2, 1}, {2}, {11, 12, 21, 22, 11, 12, 21, 22}, {4, 2});

  // Tile2D_2Axes
  RunTest<T>({11, 12, 21, 22}, {2, 2}, {2, 2}, {2}, {11, 12, 11, 12, 21, 22, 21, 22, 11, 12, 11, 12, 21, 22, 21, 22}, {4, 4});

  // Tile3D
  RunTest<T>({111, 112, 113, 122, 123, 124}, {2, 1, 3}, {1, 2, 1}, {3}, {111, 112, 113, 111, 112, 113, 122, 123, 124, 122, 123, 124}, {2, 2, 3});
}

template <>
void RunTestWrapper<bool>() {
  // Tile1DWithZeroRepeats
  RunTest<bool>({true, false, true}, {3}, {0}, {1}, {}, {0});

  // Tile2DWithZeroRepeats
  RunTest<bool>({true, false, true, false}, {2, 2}, {2, 0}, {2}, {}, {4, 0});

  // Tile1D
  RunTest<bool>({true, false, true}, {3}, {3}, {1}, {true, false, true, true, false, true, true, false, true}, {9});

  // Tile2D_1Axis
  RunTest<bool>({true, false, true, false}, {2, 2}, {2, 1}, {2}, {true, false, true, false, true, false, true, false}, {4, 2});

  // Tile2D_2Axes
  RunTest<bool>({true, false, true, false}, {2, 2}, {2, 2}, {2}, {true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false}, {4, 4});

  // Tile3D
  RunTest<bool>({true, false, true, false, true, false}, {2, 1, 3}, {1, 2, 1}, {3}, {true, false, true, true, false, true, false, true, false, false, true, false}, {2, 2, 3});
}

TEST(TensorOpTest, TileFloatType) {
  RunTestWrapper<float>();
}

TEST(TensorOpTest, TileDoubleType) {
  RunTestWrapper<double>();
}

TEST(TensorOpTest, TileInt8Type) {
  RunTestWrapper<int8_t>();
}

TEST(TensorOpTest, TileUint8Type) {
  RunTestWrapper<uint8_t>();
}

TEST(TensorOpTest, TileInt16Type) {
  RunTestWrapper<int16_t>();
}

TEST(TensorOpTest, TileUint16Type) {
  RunTestWrapper<uint16_t>();
}

TEST(TensorOpTest, TileInt32Type) {
  RunTestWrapper<int32_t>();
}

TEST(TensorOpTest, TileUint32Type) {
  RunTestWrapper<uint32_t>();
}

TEST(TensorOpTest, TileInt64Type) {
  RunTestWrapper<int64_t>();
}

TEST(TensorOpTest, TileUint64Type) {
  RunTestWrapper<uint64_t>();
}

TEST(TensorOpTest, TileBoolType) {
  RunTestWrapper<bool>();
}
}  // namespace test
}  // namespace onnxruntime
