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
  if (std::is_same<T, int8_t>::value)
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});  //TensorRT reports error: Assertion Error in makePaddedScale: 0 (regionRanges != nullptr)
  else
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

  // Tile4D
  RunTest<T>(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},  // input
      {1, 2, 3, 4},                                                                            // input dims
      {2, 1, 2, 1},                                                                            // repeat
      {4},                                                                                     // repeat dims
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
       12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
       12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23},  // output
      {2, 2, 6, 4}                                                                                       // output dims
  );

  // Tile5D
  RunTest<T>(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
       36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
       54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71},  // input
      {2, 3, 2, 3, 2},                                                           // input dims
      {2, 1, 2, 1, 2},                                                           // repeat
      {5},                                                                       // repeat dims
      {0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9,
       8, 9, 10, 11, 10, 11, 0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5,
       6, 7, 6, 7, 8, 9, 8, 9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15,
       14, 15, 16, 17, 16, 17, 18, 19, 18, 19, 20, 21, 20, 21, 22, 23, 22, 23,
       12, 13, 12, 13, 14, 15, 14, 15, 16, 17, 16, 17, 18, 19, 18, 19, 20, 21,
       20, 21, 22, 23, 22, 23, 24, 25, 24, 25, 26, 27, 26, 27, 28, 29, 28, 29,
       30, 31, 30, 31, 32, 33, 32, 33, 34, 35, 34, 35, 24, 25, 24, 25, 26, 27,
       26, 27, 28, 29, 28, 29, 30, 31, 30, 31, 32, 33, 32, 33, 34, 35, 34, 35,
       36, 37, 36, 37, 38, 39, 38, 39, 40, 41, 40, 41, 42, 43, 42, 43, 44, 45,
       44, 45, 46, 47, 46, 47, 36, 37, 36, 37, 38, 39, 38, 39, 40, 41, 40, 41,
       42, 43, 42, 43, 44, 45, 44, 45, 46, 47, 46, 47, 48, 49, 48, 49, 50, 51,
       50, 51, 52, 53, 52, 53, 54, 55, 54, 55, 56, 57, 56, 57, 58, 59, 58, 59,
       48, 49, 48, 49, 50, 51, 50, 51, 52, 53, 52, 53, 54, 55, 54, 55, 56, 57,
       56, 57, 58, 59, 58, 59, 60, 61, 60, 61, 62, 63, 62, 63, 64, 65, 64, 65,
       66, 67, 66, 67, 68, 69, 68, 69, 70, 71, 70, 71, 60, 61, 60, 61, 62, 63,
       62, 63, 64, 65, 64, 65, 66, 67, 66, 67, 68, 69, 68, 69, 70, 71, 70, 71,
       0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9,
       8, 9, 10, 11, 10, 11, 0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5,
       6, 7, 6, 7, 8, 9, 8, 9, 10, 11, 10, 11, 12, 13, 12, 13, 14, 15,
       14, 15, 16, 17, 16, 17, 18, 19, 18, 19, 20, 21, 20, 21, 22, 23, 22, 23,
       12, 13, 12, 13, 14, 15, 14, 15, 16, 17, 16, 17, 18, 19, 18, 19, 20, 21,
       20, 21, 22, 23, 22, 23, 24, 25, 24, 25, 26, 27, 26, 27, 28, 29, 28, 29,
       30, 31, 30, 31, 32, 33, 32, 33, 34, 35, 34, 35, 24, 25, 24, 25, 26, 27,
       26, 27, 28, 29, 28, 29, 30, 31, 30, 31, 32, 33, 32, 33, 34, 35, 34, 35,
       36, 37, 36, 37, 38, 39, 38, 39, 40, 41, 40, 41, 42, 43, 42, 43, 44, 45,
       44, 45, 46, 47, 46, 47, 36, 37, 36, 37, 38, 39, 38, 39, 40, 41, 40, 41,
       42, 43, 42, 43, 44, 45, 44, 45, 46, 47, 46, 47, 48, 49, 48, 49, 50, 51,
       50, 51, 52, 53, 52, 53, 54, 55, 54, 55, 56, 57, 56, 57, 58, 59, 58, 59,
       48, 49, 48, 49, 50, 51, 50, 51, 52, 53, 52, 53, 54, 55, 54, 55, 56, 57,
       56, 57, 58, 59, 58, 59, 60, 61, 60, 61, 62, 63, 62, 63, 64, 65, 64, 65,
       66, 67, 66, 67, 68, 69, 68, 69, 70, 71, 70, 71, 60, 61, 60, 61, 62, 63,
       62, 63, 64, 65, 64, 65, 66, 67, 66, 67, 68, 69, 68, 69, 70, 71, 70, 71},  // output
      {4, 3, 4, 3, 4}                                                            // output dims
  );

  // Tile1DWithOneRepeats
  RunTest<T>({111, 112, 113, 122, 123, 124}, {2, 1, 3}, {1, 1, 1}, {3}, {111, 112, 113, 122, 123, 124}, {2, 1, 3});

  // TileWhichIsBasicallyCopiesOfInputBuffer - 1
  // This will trigger the MemCpy optimization path
  RunTest<T>({111, 112, 113}, {1, 1, 3}, {2, 2, 1}, {3}, {111, 112, 113, 111, 112, 113, 111, 112, 113, 111, 112, 113}, {2, 2, 3});

  // TileWhichIsBasicallyCopiesOfInputBuffer - 2
  // This will trigger the MemCpy optimization path
  RunTest<T>({111, 112, 113}, {1, 1, 3}, {3, 1, 1}, {3}, {111, 112, 113, 111, 112, 113, 111, 112, 113}, {3, 1, 3});

  // TileWhichIsBasicallyCopiesOfInputBuffer - 3 (batch > 1 and batch_repeat == 1)
  // This will trigger the (Batched) MemCpy optimization path
  RunTest<T>({111, 112, 113, 11, 12, 13}, {2, 1, 3}, {1, 2, 1}, {3}, {111, 112, 113, 111, 112, 113, 11, 12, 13, 11, 12, 13}, {2, 2, 3});

  // TileWhichIsBasicallyCopiesOfInputBuffer - 3 (batch > 1 and batch_repeat > 1)
  // This will trigger the (Batched) MemCpy optimization path
  RunTest<T>({111, 112, 113, 11, 12, 13}, {2, 1, 3}, {2, 2, 1}, {3},
             {111, 112, 113, 111, 112, 113, 11, 12, 13, 11, 12, 13, 111, 112, 113, 111, 112, 113, 11, 12, 13, 11, 12, 13}, {4, 2, 3});
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

  // Tile1DWithOneRepeats
  RunTest<bool>({true, false, true, false, true, true}, {2, 1, 3}, {1, 1, 1}, {3}, {true, false, true, false, true, true}, {2, 1, 3});

  // TileWhichIsBasicallyCopiesOfInputBuffer - 1
  // This will trigger the MemCpy optimization path
  RunTest<bool>({true, false, true}, {1, 1, 3}, {2, 2, 1}, {3}, {true, false, true, true, false, true, true, false, true, true, false, true}, {2, 2, 3});

  // TileWhichIsBasicallyCopiesOfInputBuffer - 2
  // This will trigger the MemCpy optimization path
  RunTest<bool>({true, false, true}, {1, 1, 3}, {3, 1, 1}, {3}, {true, false, true, true, false, true, true, false, true}, {3, 1, 3});

  // TileWhichIsBasicallyCopiesOfInputBuffer - 3 (batch > 1 and batch_repeat == 1)
  // This will trigger the (Batched) MemCpy optimization path
  RunTest<bool>({true, false, true, true, false, true}, {2, 1, 3}, {1, 2, 1}, {3},
                {true, false, true, true, false, true, true, false, true, true, false, true}, {2, 2, 3});

  // TileWhichIsBasicallyCopiesOfInputBuffer - 3 (batch > 1 and batch_repeat > 1)
  // This will trigger the (Batched) MemCpy optimization path
  RunTest<bool>({true, false, true, true, false, true}, {2, 1, 3}, {2, 2, 1}, {3},
                {true, false, true, true, false, true, true, false, true, true, false, true, true, false, true, true, false, true, true, false, true, true, false, true},
                {4, 2, 3});
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
