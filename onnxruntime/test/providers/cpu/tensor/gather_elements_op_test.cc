// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <ctime>
#include <cstdlib>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

template <typename T, typename TIndex>
void RunTest(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& indices_dims, bool has_axis = false,
             int64_t axis = 0LL) {
  size_t input_size =
      static_cast<size_t>(std::accumulate(input_dims.begin(), input_dims.end(), 1LL, std::multiplies<int64_t>()));
  size_t indices_size =
      static_cast<size_t>(std::accumulate(indices_dims.begin(), indices_dims.end(), 1LL, std::multiplies<int64_t>()));
  std::vector<T> input_data = ValueRange<T>(input_size, static_cast<T>(1.0f), static_cast<T>(1.0f));
  size_t rank = input_dims.size();
  std::vector<int64_t> input_strides(rank);
  std::vector<int64_t> indices_strides(rank);
  input_strides[rank - 1] = indices_strides[rank - 1] = 1;
  if (rank > 1) {
    for (size_t i = rank - 1; i > 0; --i) {
      input_strides[i - 1] = input_dims[i] * input_strides[i];
      indices_strides[i - 1] = indices_dims[i] * indices_strides[i];
    }
  }

  int64_t new_axis = axis < 0 ? axis + static_cast<int64_t>(rank) : axis;
  std::vector<TIndex> indices_data(indices_size);
  std::vector<T> output_data(indices_size);
  std::srand(static_cast<unsigned>(std::time(0)));
  for (size_t i = 0; i < indices_size; ++i) {
    // Negative index possible.
    int64_t index = (static_cast<int64_t>(std::rand()) % (input_dims[new_axis] * 2)) - input_dims[new_axis];
    indices_data[i] = static_cast<TIndex>(index);
    int64_t offset = 0;
    int64_t remain = static_cast<int64_t>(i);
    for (size_t j = 0; j < rank; ++j) {
      int64_t pos = static_cast<int64_t>(j) == new_axis ? (index < 0 ? index + input_dims[new_axis] : index)
                                                        : (remain / indices_strides[j]);
      offset += pos * input_strides[j];
      remain = remain % indices_strides[j];
    }
    output_data[i] = input_data[offset];
  }

  OpTester test("GatherElements", 11);
  if (has_axis) test.AddAttribute<int64_t>("axis", axis);
  test.AddInput<T>("data", input_dims, input_data);
  test.AddInput<TIndex>("indices", indices_dims, indices_data);
  test.AddOutput<T>("output", indices_dims, output_data);
  // Skip tensorrt for INT8 tests.
  if (std::is_same<T, int8_t>::value) {
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  } else {
    test.Run();
  }
}

template <typename T>
void RunTestWrapper() {
  RunTest<T, int32_t>({2, 3}, {1, 2});
  RunTest<T, int32_t>({2, 2}, {2, 2}, true, 1LL);
  RunTest<T, int64_t>({2, 2}, {2, 2}, true, -1LL);
  RunTest<T, int32_t>({2, 2, 2}, {1, 2, 1}, true, 1LL);
  RunTest<T, int64_t>({2, 2, 2}, {1, 2, 1}, true, 2LL);
  RunTest<T, int32_t>({3, 3}, {3, 2}, true, 1LL);
  RunTest<T, int64_t>({3, 3}, {3, 2});
  RunTest<T, int32_t>({3}, {2}, true, 0LL);

  // ([2,2,2],[2,2,2],axis=0) coalesce to ([2,4],[2,4],axis=0)
  RunTest<T, int32_t>({2, 2, 2}, {2, 2, 2});

  // ([2,2,2],[3,2,2],axis=0) coalesce to ([2,4],[3,4],axis=0)
  RunTest<T, int64_t>({2, 2, 2}, {3, 2, 2}, true);

  // ([2,2,2,3],[3,2,2,2],axis=0) coalesce to ([2,4,3],[3,4,2],axis=0)
  RunTest<T, int32_t>({2, 2, 2, 3}, {3, 2, 2, 2}, true, 0LL);

  // ([2,2,3,3,2],[2,2,3,2,2],axis=0) coalesce to ([2,6,3,2],[2,6,3,2],axis=0)
  RunTest<T, int64_t>({2, 2, 3, 3, 2}, {2, 2, 3, 2, 2});

  // ([2,2,1,3,1],[2,2,1,2,1],axis=0) coalesce to ([2,2,3],[2,2,2],axis=0)
  RunTest<T, int32_t>({2, 2, 1, 3, 1}, {2, 2, 1, 2, 1});

  // No coalesce
  RunTest<T, int64_t>({2, 3, 2}, {2, 3, 2}, true, -2LL);

  // ([2,2,2],[2,2,3],axis=2) coalesce to ([4,2],[4,3],axis=1)
  RunTest<T, int64_t>({2, 2, 2}, {2, 2, 3}, true, 2LL);

  // ([2,2,3,2],[2,2,2,3],axis=-1) coalesce to ([4,3,2],[4,2,3],axis=2)
  RunTest<T, int32_t>({2, 2, 3, 2}, {2, 2, 2, 3}, true, -1LL);

  // ([2,1,2,3,1,2],[2,1,2,2,1,2],axis=5) coalesce to ([4,3,2],[4,2,2],axis=2)
  RunTest<T, int64_t>({2, 1, 2, 3, 1, 2}, {2, 1, 2, 2, 1, 2}, true, 5LL);

  // ([2,1,2,2,3,2,2],[2,1,2,3,2,2,2],axis=3) coalesce to ([4,2,3,4],[4,3,2,4],axis=1)
  RunTest<T, int32_t>({2, 1, 2, 2, 3, 2, 2}, {2, 1, 2, 3, 2, 2, 2}, true, 3LL);

  // ([2,1,1,2,3,2,3],[2,1,1,2,3,2,2],axis=-5) coalesce to ([2,1,12,3],[2,1,12,2],axis=1)
  RunTest<T, int64_t>({2, 1, 1, 2, 3, 2, 3}, {2, 1, 1, 2, 3, 2, 2}, true, -5LL);
}

template <>
void RunTestWrapper<bool>() {
  // 3D input - axis 2
  OpTester test1("GatherElements", 11);
  test1.AddAttribute<int64_t>("axis", 2LL);
  test1.AddInput<bool>("data", {2, 2, 2}, {true, false, true, false, true, false, true, false});
  test1.AddInput<int64_t>("indices", {1, 2, 1}, {0, 1});
  test1.AddOutput<bool>("output", {1, 2, 1}, {true, false});
  test1.Run();
}

template <>
void RunTestWrapper<std::string>() {
  // int32_t indices - axis 0
  OpTester test1("GatherElements", 11);
  test1.AddAttribute<int64_t>("axis", 0LL);
  test1.AddInput<std::string>("data", {2, 3}, {"a", "b", "c", "d", "e", "f"});
  test1.AddInput<int32_t>("indices", {1, 2}, {0, 1});
  test1.AddOutput<std::string>("output", {1, 2}, {"a", "e"});
  test1.Run();

  // int32_t indices - axis 1
  OpTester test2("GatherElements", 11);
  test2.AddAttribute<int64_t>("axis", 1LL);
  test2.AddInput<std::string>("data", {2, 2}, {"a", "b", "c", "d"});
  test2.AddInput<int32_t>("indices", {2, 2}, {0, 0, 1, 0});
  test2.AddOutput<std::string>("output", {2, 2}, {"a", "a", "d", "c"});
  test2.Run();

  // negative indices - axis 1
  OpTester test3("GatherElements", 11);
  test3.AddAttribute<int64_t>("axis", 1LL);
  test3.AddInput<std::string>("data", {2, 2}, {"a", "b", "c", "d"});
  test3.AddInput<int32_t>("indices", {2, 2}, {0, 0, -1, -1});
  test3.AddOutput<std::string>("output", {2, 2}, {"a", "a", "d", "d"});
  test3.Run();

  // indices out of bounds
  OpTester test4("GatherElements", 11);
  test4.AddAttribute<int64_t>("axis", 1LL);
  test4.AddInput<std::string>("data", {2, 2}, {"a", "b", "c", "d"});
  test4.AddInput<int32_t>("indices", {2, 2}, {0, 0, -3, -3});
  test4.AddOutput<std::string>("output", {2, 2}, {"a", "a", "c", "c"});
  // skip nuphar, which will not throw error message but will ensure no out-of-bound access
  // skip Openvino, which will not throw error message but will ensure no out-of-bound access
  test4.Run(OpTester::ExpectResult::kExpectFailure, "GatherElements op: Out of range value in index tensor",
            {kNupharExecutionProvider, kOpenVINOExecutionProvider});

  // 3D input - axis 1
  OpTester test5("GatherElements", 11);
  test5.AddAttribute<int64_t>("axis", 1LL);
  test5.AddInput<std::string>("data", {2, 2, 2}, {"a", "b", "c", "d", "e", "f", "g", "h"});
  test5.AddInput<int32_t>("indices", {1, 2, 1}, {0, 1});
  test5.AddOutput<std::string>("output", {1, 2, 1}, {"a", "c"});
  test5.Run();

  // 3D input - axis 2
  OpTester test6("GatherElements", 11);
  test6.AddAttribute<int64_t>("axis", 2LL);
  test6.AddInput<std::string>("data", {2, 2, 2}, {"a", "b", "c", "d", "e", "f", "g", "h"});
  test6.AddInput<int32_t>("indices", {1, 2, 1}, {0, 1});
  test6.AddOutput<std::string>("output", {1, 2, 1}, {"a", "d"});
  test6.Run();

  // 2D input - axis 1
  OpTester test7("GatherElements", 11);
  test7.AddAttribute<int64_t>("axis", 1LL);
  test7.AddInput<std::string>("data", {3, 3}, {"a", "b", "c", "d", "e", "f", "g", "h", "i"});
  test7.AddInput<int64_t>("indices", {3, 2}, {1, 0, 0, 1, 0, 1});
  test7.AddOutput<std::string>("output", {3, 2}, {"b", "a", "d", "e", "g", "h"});
  test7.Run();

  // 2D input - axis 2
  OpTester test8("GatherElements", 11);
  test8.AddAttribute<int64_t>("axis", 0LL);
  test8.AddInput<std::string>("data", {3, 3}, {"a", "b", "c", "d", "e", "f", "g", "h", "i"});
  test8.AddInput<int64_t>("indices", {3, 2}, {1, 0, 0, 1, 0, 1});
  test8.AddOutput<std::string>("output", {3, 2}, {"d", "b", "a", "e", "a", "e"});
  test8.Run();
}

}  // namespace

// Disable TensorRT due to missing int8 calibrator
TEST(GatherElementsOpTest, int8_t) { RunTestWrapper<int8_t>(); }

TEST(GatherElementsOpTest, int16_t) { RunTestWrapper<int16_t>(); }

TEST(GatherElementsOpTest, int32_t) { RunTestWrapper<int32_t>(); }

TEST(GatherElementsOpTest, int64_t) { RunTestWrapper<int64_t>(); }

TEST(GatherElementsOpTest, uint8_t) { RunTestWrapper<uint8_t>(); }

TEST(GatherElementsOpTest, uint16_t) { RunTestWrapper<uint16_t>(); }

TEST(GatherElementsOpTest, uint32_t) { RunTestWrapper<uint32_t>(); }

TEST(GatherElementsOpTest, uint64_t) { RunTestWrapper<uint64_t>(); }

TEST(GatherElementsOpTest, float) { RunTestWrapper<float>(); }

TEST(GatherElementsOpTest, double) { RunTestWrapper<double>(); }

TEST(GatherElementsOpTest, MLFloat16) { RunTestWrapper<MLFloat16>(); }

TEST(GatherElementsOpTest, bool) { RunTestWrapper<bool>(); }

TEST(GatherElementsOpTest, string) { RunTestWrapper<std::string>(); }

TEST(GatherElementsOpTest, IndicesOutOfBounds) {
  // indices out of bounds
  OpTester test("GatherElements", 11);
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<float>("data", {2, 2}, {1, 2, 3, 4});
  test.AddInput<int64_t>("indices", {2, 2}, {0, 0, 2, 2});
  test.AddOutput<float>("output", {2, 2}, {1, 1, 3, 3});
  // skip nuphar, which will not throw error message but will ensure no out-of-bound access
  // skip cuda as the cuda kernel won't throw the error message
  // skip openvino which will not throw error message but will ensure no out-of-bound access
  // skip TensorRT because it doesn't support out of bounds indices
  test.Run(OpTester::ExpectResult::kExpectFailure, "",
           {kNupharExecutionProvider, kCudaExecutionProvider, kRocmExecutionProvider, kOpenVINOExecutionProvider,
            kTensorrtExecutionProvider});
}

TEST(GatherElementsOpTest, BigIndices) {
  // int32_t indices - axis 0
  OpTester test1("GatherElements", 11);

  test1.AddAttribute<int64_t>("axis", 0LL);
  constexpr int kNumIndices = 10 * 1000;  // must be >= kParallelizationThreshold in gather_elements.cc
  std::vector<float> input(2 * kNumIndices);
  std::iota(std::begin(input), std::end(input), 0.f);
  test1.AddInput<float>("data", {2, kNumIndices}, input);

  std::vector<int32_t> indices(kNumIndices, 0);
  std::vector<float> output(kNumIndices);
  std::iota(std::begin(output), std::end(output), 0.f);
  test1.AddInput<int32_t>("indices", {1, kNumIndices}, indices);
  test1.AddOutput<float>("output", {1, kNumIndices}, output);
  test1.Run();
}

}  // namespace test
}  // namespace onnxruntime
