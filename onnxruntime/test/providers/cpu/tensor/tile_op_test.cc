// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename T>
std::vector<T> InputData(size_t size) {
  std::vector<T> result(size);
  for (size_t i = 0; i < size; i++) {
    result[i] = static_cast<T>(i);
  }
  return result;
}

template <>
std::vector<MLFloat16> InputData<MLFloat16>(size_t size) {
  std::vector<MLFloat16> result(size);
  for (size_t i = 0; i < size; i++) {
    result[i] = MLFloat16(static_cast<float>(i));
  }
  return result;
}

template <typename T>
void RunTest(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& repeats) {
  size_t input_size =
      static_cast<size_t>(std::accumulate(input_dims.begin(), input_dims.end(), 1LL, std::multiplies<int64_t>()));
  std::vector<T> input_data = InputData<T>(input_size);
  size_t rank = input_dims.size();
  std::vector<int64_t> repeats_dims(1);
  repeats_dims[0] = static_cast<int64_t>(rank);
  std::vector<int64_t> output_dims(rank);
  for (size_t i = 0; i < rank; ++i) {
    output_dims[i] = input_dims[i] * repeats[i];
  }
  size_t output_size =
      static_cast<size_t>(std::accumulate(output_dims.begin(), output_dims.end(), 1LL, std::multiplies<int64_t>()));
  std::vector<T> output_data(output_size);
  std::vector<int64_t> input_strides(rank);
  std::vector<int64_t> output_strides(rank);
  input_strides[rank - 1] = output_strides[rank - 1] = 1;
  if (rank > 1) {
    for (size_t i = rank - 2;; --i) {
      input_strides[i] = input_dims[i + 1] * input_strides[i + 1];
      output_strides[i] = output_dims[i + 1] * output_strides[i + 1];
      if (i == 0) break;
    }
  }
  for (size_t i = 0; i < output_size; ++i) {
    int64_t index = 0;
    int64_t remain = static_cast<int64_t>(i);
    for (size_t j = 0; j < rank; ++j) {
      index += (((remain / output_strides[j]) % input_dims[j]) * input_strides[j]);
      remain = remain % output_strides[j];
    }
    output_data[i] = input_data[static_cast<size_t>(index)];
  }
  OpTester test("Tile");
  test.AddInput<T>("input", input_dims, input_data);
  test.AddInput<int64_t>("repeats", repeats_dims, repeats);
  test.AddOutput<T>("output", output_dims, output_data);
  if (std::is_same<T, int8_t>::value) {
    // TensorRT reports error: Assertion Error in makePaddedScale: 0 (regionRanges != nullptr)
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  } else {
    test.Run();
  }
}

template <typename T>
void RunTestWrapper() {
  // Tile1DWithZeroRepeats
  RunTest<T>({3}, {0});

  // Tile2DWithZeroRepeats
  RunTest<T>({2, 2}, {2, 0});

  // Tile1D
  RunTest<T>({3}, {3});

  // Tile2D_1Axis
  RunTest<T>({2, 2}, {2, 1});
  RunTest<T>({2, 3}, {2, 1});

  // Tile2D_2Axes
  RunTest<T>({2, 2}, {2, 2});
  RunTest<T>({2, 4}, {2, 2});

  // Tile3D
  RunTest<T>({2, 1, 3}, {1, 2, 1});

  // Tile4D
  RunTest<T>({1, 2, 3, 4}, {2, 1, 2, 1});

  // Tile5D
  RunTest<T>({2, 3, 2, 3, 2}, {2, 1, 2, 1, 2});

  // Tile1DWithOneRepeats
  RunTest<T>({2, 1, 3}, {1, 1, 1});

  // TileWhichIsBasicallyCopiesOfInputBuffer - 1
  // This will trigger the MemCpy optimization path
  RunTest<T>({1, 1, 3}, {2, 2, 1});

  // TileWhichIsBasicallyCopiesOfInputBuffer - 2
  // This will trigger the MemCpy optimization path
  RunTest<T>({1, 1, 3}, {3, 1, 1});

  // TileWhichIsBasicallyCopiesOfInputBuffer - 3 (batch > 1 and batch_repeat == 1)
  // This will trigger the (Batched) MemCpy optimization path
  RunTest<T>({2, 1, 3}, {1, 2, 1});

  // TileWhichIsBasicallyCopiesOfInputBuffer - 3 (batch > 1 and batch_repeat > 1)
  // This will trigger the (Batched) MemCpy optimization path
  RunTest<T>({2, 1, 3}, {2, 2, 1});

#if defined(USE_CUDA) || defined(USE_ROCM)
  // _TileMemcpyKernelFromInput, vectorized 4
  RunTest<T>({256, 512}, {3, 1});

  // _TileMemcpyKernelFromInput, vectorized 2
  RunTest<T>({258, 257}, {4, 1});

  // _TileMemcpyKernelFromInput, non-vectorized
  RunTest<T>({129, 257}, {5, 1});

  // _TileBatchedMemcpyKernelFromInput, vectorized 4
  RunTest<T>({512, 256}, {1, 3});

  // _TileBatchedMemcpyKernelFromInput, vectorized 2
  RunTest<T>({257, 258}, {2, 2});

  // _TileBatchedMemcpyKernelFromInput, non-vectorized
  RunTest<T>({129, 257}, {3, 2});
#endif
}

// OpTester's AddInput and AddOutput do not support std::vector<bool>.
void RunTestForBool(std::initializer_list<bool> input_data, std::initializer_list<int64_t> input_dims,
                    std::initializer_list<int64_t> repeats, std::initializer_list<int64_t> repeats_dims,
                    std::initializer_list<bool> output_data, std::initializer_list<int64_t> output_dims) {
  OpTester test("Tile");
  test.AddInput<bool>("input", input_dims, input_data);
  test.AddInput<int64_t>("repeats", repeats_dims, repeats);
  test.AddOutput<bool>("output", output_dims, output_data);
  test.Run();
}

void RunTestWrapperForBool() {
  // Tile1DWithZeroRepeats
  RunTestForBool({true, false, true}, {3}, {0}, {1}, {}, {0});

  // Tile2DWithZeroRepeats
  RunTestForBool({true, false, true, false}, {2, 2}, {2, 0}, {2}, {}, {4, 0});

  // Tile1D
  RunTestForBool({true, false, true}, {3}, {3}, {1}, {true, false, true, true, false, true, true, false, true}, {9});

  // Tile2D_1Axis
  RunTestForBool({true, false, true, false}, {2, 2}, {2, 1}, {2}, {true, false, true, false, true, false, true, false},
                 {4, 2});

  // Tile2D_2Axes
  RunTestForBool(
      {true, false, true, false}, {2, 2}, {2, 2}, {2},
      {true, false, true, false, true, false, true, false, true, false, true, false, true, false, true, false}, {4, 4});

  // Tile3D
  RunTestForBool({true, false, true, false, true, false}, {2, 1, 3}, {1, 2, 1}, {3},
                 {true, false, true, true, false, true, false, true, false, false, true, false}, {2, 2, 3});

  // Tile1DWithOneRepeats
  RunTestForBool({true, false, true, false, true, true}, {2, 1, 3}, {1, 1, 1}, {3},
                 {true, false, true, false, true, true}, {2, 1, 3});

  // TileWhichIsBasicallyCopiesOfInputBuffer - 1
  // This will trigger the MemCpy optimization path
  RunTestForBool({true, false, true}, {1, 1, 3}, {2, 2, 1}, {3},
                 {true, false, true, true, false, true, true, false, true, true, false, true}, {2, 2, 3});

  // TileWhichIsBasicallyCopiesOfInputBuffer - 2
  // This will trigger the MemCpy optimization path
  RunTestForBool({true, false, true}, {1, 1, 3}, {3, 1, 1}, {3},
                 {true, false, true, true, false, true, true, false, true}, {3, 1, 3});

  // TileWhichIsBasicallyCopiesOfInputBuffer - 3 (batch > 1 and batch_repeat == 1)
  // This will trigger the (Batched) MemCpy optimization path
  RunTestForBool({true, false, true, true, false, true}, {2, 1, 3}, {1, 2, 1}, {3},
                 {true, false, true, true, false, true, true, false, true, true, false, true}, {2, 2, 3});

  // TileWhichIsBasicallyCopiesOfInputBuffer - 3 (batch > 1 and batch_repeat > 1)
  // This will trigger the (Batched) MemCpy optimization path
  RunTestForBool({true, false, true, true, false, true}, {2, 1, 3}, {2, 2, 1}, {3},
                 {true, false, true, true, false, true, true, false, true, true, false, true,
                  true, false, true, true, false, true, true, false, true, true, false, true},
                 {4, 2, 3});
}

TEST(TensorOpTest, TileFloatType) { RunTestWrapper<float>(); }

TEST(TensorOpTest, TileDoubleType) { RunTestWrapper<double>(); }

TEST(TensorOpTest, TileInt8Type) { RunTestWrapper<int8_t>(); }

TEST(TensorOpTest, TileUint8Type) { RunTestWrapper<uint8_t>(); }

TEST(TensorOpTest, TileInt16Type) { RunTestWrapper<int16_t>(); }

TEST(TensorOpTest, TileUint16Type) { RunTestWrapper<uint16_t>(); }

TEST(TensorOpTest, TileInt32Type) { RunTestWrapper<int32_t>(); }

TEST(TensorOpTest, TileUint32Type) { RunTestWrapper<uint32_t>(); }

TEST(TensorOpTest, TileInt64Type) { RunTestWrapper<int64_t>(); }

TEST(TensorOpTest, TileUint64Type) { RunTestWrapper<uint64_t>(); }

TEST(TensorOpTest, TileBoolType) { RunTestWrapperForBool(); }

#if defined(USE_CUDA) || defined(USE_ROCM)
TEST(TensorOpTest, TileMLFloat16Type) { RunTestWrapper<MLFloat16>(); }
#endif

}  // namespace test
}  // namespace onnxruntime
