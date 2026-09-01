// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <memory>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/util/include/default_providers.h"

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
  std::vector<T> input_data = ValueRange<T>(input_size, static_cast<T>(0.0f), static_cast<T>(2.0f));
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
  std::vector<T> updates_data = ValueRange<T>(indices_size, static_cast<T>(1.0f), static_cast<T>(2.0f));
  std::vector<T> output_data(input_data);
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
    // It's possible that one position is updated multiple times, to make sure it generate certain result,
    // set all the corresponding places in updates_data to same value.
    if (output_data[offset] != input_data[offset]) {
      updates_data[i] = output_data[offset];
    } else {
      output_data[offset] = updates_data[i];
    }
  }

  onnxruntime::test::OpTester test("Scatter", 9);
  if (has_axis) test.AddAttribute<int64_t>("axis", axis);
  test.AddInput<T>("data", input_dims, input_data);
  test.AddInput<TIndex>("indices", indices_dims, indices_data);
  test.AddInput<T>("updates", indices_dims, updates_data);
  test.AddOutput<T>("y", input_dims, output_data);
  // OpenVINO and QNN doesn't support negative indices value.
  // Disable TensorRT due to missing int8 calibrator.
  if (std::is_same<T, int8_t>::value) {
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kQnnExecutionProvider});
  } else if (std::is_same<T, MLFloat16>::value) {
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider, kQnnExecutionProvider});
  } else {
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider, kQnnExecutionProvider});
  }

  onnxruntime::test::OpTester test1("ScatterElements", 11);
  if (has_axis) test1.AddAttribute<int64_t>("axis", axis);
  test1.AddInput<T>("data", input_dims, input_data);
  test1.AddInput<TIndex>("indices", indices_dims, indices_data);
  test1.AddInput<T>("updates", indices_dims, updates_data);
  test1.AddOutput<T>("y", input_dims, output_data);
  if (std::is_same<T, int8_t>::value) {
    test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kQnnExecutionProvider});
  } else if (std::is_same<T, MLFloat16>::value) {
    test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider, kQnnExecutionProvider});
  } else {
    test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider, kQnnExecutionProvider});
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

  // ([2,2,3,3,2],[2,2,3,2,2],axis=0) coalesce to ([2,6,3,2],[2,6,2,2],axis=0)
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

}  // namespace

TEST(Scatter, int8_t) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

  RunTestWrapper<int8_t>();
}

TEST(Scatter, int16_t) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

  RunTestWrapper<int16_t>();
}

TEST(Scatter, int32_t) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

  RunTestWrapper<int32_t>();
}

TEST(Scatter, int64_t) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

  RunTestWrapper<int64_t>();
}

TEST(Scatter, uint8_t) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

  RunTestWrapper<uint8_t>();
}

TEST(Scatter, uint16_t) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

  RunTestWrapper<uint16_t>();
}

TEST(Scatter, uint32_t) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

  RunTestWrapper<uint32_t>();
}

TEST(Scatter, uint64_t) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

  RunTestWrapper<uint64_t>();
}

TEST(Scatter, float) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

  RunTestWrapper<float>();
}

TEST(Scatter, double) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

  RunTestWrapper<double>();
}

TEST(Scatter, MLFloat16) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

  RunTestWrapper<MLFloat16>();
}

static void scatter_indices_updates_dont_match(const char* op_name, int op_version) {
  OpTester test(op_name, op_version);
  test.AddAttribute<int64_t>("axis", 1);

  test.AddInput<float>("data", {1, 5}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f});
  test.AddInput<int64_t>("indices", {1, 3}, {1, 3, 3});
  test.AddInput<float>("updates", {1, 2}, {1.1f, 2.1f});
  test.AddOutput<float>("y", {1, 5}, {1.0f, 1.1f, 3.0f, 2.1f, 5.0f});
  test.Run(OpTester::ExpectResult::kExpectFailure, "Indices vs updates dimensions differs at position=1 3 vs 2",
           {kTensorrtExecutionProvider, kWebGpuExecutionProvider});
}

TEST(Scatter, IndicesUpdatesDontMatch) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

  scatter_indices_updates_dont_match("Scatter", 9);
  scatter_indices_updates_dont_match("ScatterElements", 11);
}

static void scatter_invalid_index(const char* op_name, int op_version) {
  OpTester test(op_name, op_version);
  test.AddAttribute<int64_t>("axis", 0);

  test.AddInput<float>("data", {4, 2, 1}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddInput<int64_t>("indices", {1, 1, 1}, {4});
  test.AddInput<float>("updates", {1, 1, 1}, {5.0f});
  test.AddOutput<float>("y", {4, 2, 1}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 0.0f});
  test.Run(OpTester::ExpectResult::kExpectFailure,
           "indices element out of data bounds, idx=4 must be within the inclusive range [-4,3]",
           {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kTensorrtExecutionProvider, kQnnExecutionProvider, kWebGpuExecutionProvider});
}

TEST(Scatter, InvalidIndex) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: AbiCustomRegistry.cpp(507): The parameter is incorrect.";
  }

  scatter_invalid_index("Scatter", 9);
  scatter_invalid_index("ScatterElements", 11);
}

static void scatter_bool_with_axis_tests(const char* op_name, int op_version) {
  OpTester test(op_name, op_version);
  test.AddAttribute<int64_t>("axis", 1);

  test.AddInput<bool>("data", {1, 5}, {false, false, false, true, false});
  test.AddInput<int64_t>("indices", {1, 2}, {1, 3});
  test.AddInput<bool>("updates", {1, 2}, {true, false});
  test.AddOutput<bool>("y", {1, 5}, {false, true, false, false, false});
#if defined(OPENVINO_CONFIG_GPU)
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaNHWCExecutionProvider, kOpenVINOExecutionProvider});  // OpenVINO: Disabled due to failure for GPU
#else
  test.Run(OpTester::ExpectResult::kExpectSuccess, "",
           {kCudaNHWCExecutionProvider});  // OpenVINO: Disabled due to failure for GPU
#endif
}

TEST(Scatter, BoolInputWithAxis) {
  scatter_bool_with_axis_tests("Scatter", 9);
  scatter_bool_with_axis_tests("ScatterElements", 11);
}

TEST(ScatterElements, AddReduction) {
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", "add");

  test.AddInput<float>("data", {3, 3}, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
  test.AddInput<int64_t>("indices", {2, 3}, {1, 0, 2, 0, 2, 1});
  test.AddInput<float>("updates", {2, 3}, {1.0f, 1.1f, 1.2f, 2.0f, 2.1f, 2.2f});
  test.AddOutput<float>("y", {3, 3}, {3.0f, 1.1f, 0.0f, 1.0f, 0.0f, 2.2f, 0.0f, 2.1f, 1.2f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

#if defined(CUDA_VERSION)
// Operation on float16 (MLFloat16) is not implemented on CPU.
TEST(ScatterElements, AddReduction_MLFloat16) {
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", "add");

  test.AddInput<MLFloat16>("data", {2, 3}, ToFloat16(std::vector<float>({-9.f, -4.f, -1.f, -7.f, -3.f, -6.f})));
  test.AddInput<int64_t>("indices", {4, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  test.AddInput<MLFloat16>("updates", {4, 3}, ToFloat16(std::vector<float>({1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f, 4.f, 4.f, 4.f})));
  test.AddOutput<MLFloat16>("y", {2, 3}, ToFloat16(std::vector<float>({-9.f, -4.f, -1.f, -7.f + (1.f + 2.f + 3.f + 4.f), -3.f + (1.f + 2.f + 3.f + 4.f), -6.f + (1.f + 2.f + 3.f + 4.f)})));

  // exclude CPU Execution Provider as MLFloat16 is not supported in CPU
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}
#endif

TEST(ScatterElements, AddReductionAxis1) {
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<std::string>("reduction", "add");

  // update's slice shape is {2, 1}
  test.AddInput<float>("data", {2, 3}, {9.f, 4.f, 1.f, 7.f, 3.f, 6.f});
  test.AddInput<int64_t>("indices", {2, 4}, {1, 1, 1, 1, 1, 1, 1, 1});
  test.AddInput<float>("updates", {2, 4}, {2.f, 5.f, 3.f, 6.f, 7.f, 9.f, 8.f, 10.f});
  test.AddOutput<float>("y", {2, 3}, {9.f, 4.f + (2.f + 5.f + 3.f + 6.f), 1.f, 7.f, 3.f + (7.f + 9.f + 8.f + 10.f), 6.f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kQnnExecutionProvider});
}

TEST(ScatterElements, MulReduction) {
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", "mul");

  test.AddInput<float>("data", {2, 3}, {-9.f, -4.f, -1.f, -7.f, -3.f, -6.f});
  test.AddInput<int64_t>("indices", {2, 3}, {1, 1, 1, 1, 1, 1});
  test.AddInput<float>("updates", {2, 3}, {7.f, 3.f, 6.f, 7.f, 3.f, 6.f});
  test.AddOutput<float>("y", {2, 3}, {-9.f, -4.f, -1.f, -7.f * 7.f * 7.f, -3.f * 3.f * 3.f, -6.f * 6.f * 6.f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ScatterElements, MulReductionInt32) {
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", "mul");

  test.AddInput<int32_t>("data", {1}, {2});
  test.AddInput<int64_t>("indices", {1}, {0});
  test.AddInput<int32_t>("updates", {1}, {3});
  test.AddOutput<int32_t>("y", {1}, {6});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ScatterElements, AddReductionInt32SemanticDispatch) {
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", "add");

  test.AddInput<int32_t>("data", {1}, {0x3f800000});
  test.AddInput<int64_t>("indices", {1}, {0});
  test.AddInput<int32_t>("updates", {1}, {0x3f800000});
  test.AddOutput<int32_t>("y", {1}, {0x7f000000});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ScatterElements, AddReductionInt32DuplicateIndices) {
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", "add");

  test.AddInput<int32_t>("data", {1}, {1});
  test.AddInput<int64_t>("indices", {2}, {0, 0});
  test.AddInput<int32_t>("updates", {2}, {2, 3});
  test.AddOutput<int32_t>("y", {1}, {6});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ScatterElements, NoneReductionInt32) {
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", "none");

  test.AddInput<int32_t>("data", {1}, {2});
  test.AddInput<int64_t>("indices", {1}, {0});
  test.AddInput<int32_t>("updates", {1}, {3});
  test.AddOutput<int32_t>("y", {1}, {3});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ScatterElements, MulReductionAxis1) {
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<std::string>("reduction", "mul");

  // update's slice shape is {2, 1}
  test.AddInput<float>("data", {2, 3}, {9.f, 4.f, 1.f, 7.f, 3.f, 6.f});
  test.AddInput<int64_t>("indices", {2, 4}, {1, 1, 1, 1, 1, 1, 1, 1});
  test.AddInput<float>("updates", {2, 4}, {2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});
  test.AddOutput<float>("y", {2, 3}, {9.f, 4.f * (2.f * 3.f * 4.f * 5.f), 1.f, 7.f, 3.f * (6.f * 7.f * 8.f * 9.f), 6.f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider, kQnnExecutionProvider});
}

TEST(ScatterElements, MaxReduction_MLFloat16) {
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", "max");

  test.AddInput<MLFloat16>("data", {2, 3}, ToFloat16({-9.f, -4.f, -1.f, -7.f, -3.f, -6.f}));
  test.AddInput<int64_t>("indices", {2, 3}, {1, 1, 1, 1, 1, 1});
  test.AddInput<MLFloat16>("updates", {2, 3}, ToFloat16({1.f, 5.f, 3.f, 7.f, 3.f, 6.f}));
  test.AddOutput<MLFloat16>("y", {2, 3}, ToFloat16({-9.f, -4.f, -1.f, 7.f, 5.f, 6.f}));

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ScatterElements, MaxReduction_Float) {
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", "max");

  test.AddInput<float>("data", {2, 3}, {-9.f, -4.f, -1.f, -7.f, -3.f, -6.f});
  test.AddInput<int64_t>("indices", {2, 3}, {1, 1, 1, 1, 1, 1});
  test.AddInput<float>("updates", {2, 3}, {1.f, 5.f, 3.f, 7.f, 3.f, 6.f});
  test.AddOutput<float>("y", {2, 3}, {-9.f, -4.f, -1.f, 7.f, 5.f, 6.f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ScatterElements, MaxReduction_Double) {
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", "max");

  test.AddInput<double>("data", {2, 3}, {-9.f, -4.f, -1.f, -7.f, -3.f, -6.f});
  test.AddInput<int64_t>("indices", {2, 3}, {1, 1, 1, 1, 1, 1});
  test.AddInput<double>("updates", {2, 3}, {1.f, 5.f, 3.f, 7.f, 3.f, 6.f});
  test.AddOutput<double>("y", {2, 3}, {-9.f, -4.f, -1.f, 7.f, 5.f, 6.f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ScatterElements, MinReduction_MLFloat16) {
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", "min");

  test.AddInput<MLFloat16>("data", {2, 3}, ToFloat16({-9.f, -4.f, -1.f, 8.f, -3.f, 5.f}));
  test.AddInput<int64_t>("indices", {2, 3}, {1, 1, 1, 1, 1, 1});
  test.AddInput<MLFloat16>("updates", {2, 3}, ToFloat16({1.f, 5.f, 3.f, 7.f, 3.f, 6.f}));
  test.AddOutput<MLFloat16>("y", {2, 3}, ToFloat16({-9.f, -4.f, -1.f, 1.f, -3.f, 3.f}));

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ScatterElements, MinReduction_Float) {
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", "min");

  test.AddInput<float>("data", {2, 3}, {-9.f, -4.f, -1.f, 8.f, -3.f, 5.f});
  test.AddInput<int64_t>("indices", {2, 3}, {1, 1, 1, 1, 1, 1});
  test.AddInput<float>("updates", {2, 3}, {1.f, 5.f, 3.f, 7.f, 3.f, 6.f});
  test.AddOutput<float>("y", {2, 3}, {-9.f, -4.f, -1.f, 1.f, -3.f, 3.f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ScatterElements, MinReduction_Double) {
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", "min");

  test.AddInput<double>("data", {2, 3}, {-9.f, -4.f, -1.f, 8.f, -3.f, 5.f});
  test.AddInput<int64_t>("indices", {2, 3}, {1, 1, 1, 1, 1, 1});
  test.AddInput<double>("updates", {2, 3}, {1.f, 5.f, 3.f, 7.f, 3.f, 6.f});
  test.AddOutput<double>("y", {2, 3}, {-9.f, -4.f, -1.f, 1.f, -3.f, 3.f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

namespace {

template <typename T, typename TIndex = int64_t>
void RunCudaScatterElementsReduction(const char* dtype_name, const char* reduction, T data, T update, T expected,
                                     int opset = 18) {
#if !defined(USE_CUDA)
  GTEST_SKIP() << "CUDA is required for CUDA ScatterElements dtype coverage.";
#endif
  SCOPED_TRACE(testing::Message() << "dtype=" << dtype_name << ", reduction=" << reduction << ", opset=" << opset);
  OpTester test("ScatterElements", opset);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", reduction);
  test.AddInput<T>("data", {1}, {data});
  test.AddInput<TIndex>("indices", {1}, {0});
  test.AddInput<T>("updates", {1}, {update});
  test.AddOutput<T>("y", {1}, {expected});
#if defined(USE_CUDA)
  auto cuda_ep = DefaultCudaExecutionProvider();
  ASSERT_NE(cuda_ep, nullptr);
  test.ConfigEp(std::move(cuda_ep)).RunWithConfig();
#endif
}

template <typename T, typename TIndex = int64_t>
void RunCudaScatterElementsAddContention(const char* dtype_name, T data, T update, T expected,
                                         size_t update_count) {
#if !defined(USE_CUDA)
  GTEST_SKIP() << "CUDA is required for CUDA ScatterElements contention coverage.";
#endif
  SCOPED_TRACE(testing::Message() << "dtype=" << dtype_name << ", updates=" << update_count);
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", "add");
  test.AddInput<T>("data", {1}, {data});
  test.AddInput<TIndex>("indices", {static_cast<int64_t>(update_count)}, std::vector<TIndex>(update_count, 0));
  test.AddInput<T>("updates", {static_cast<int64_t>(update_count)}, std::vector<T>(update_count, update));
  test.AddOutput<T>("y", {1}, {expected});
#if defined(USE_CUDA)
  auto cuda_ep = DefaultCudaExecutionProvider();
  ASSERT_NE(cuda_ep, nullptr);
  test.ConfigEp(std::move(cuda_ep)).RunWithConfig();
#endif
}

void RunCudaScatterElementsBoolReduction(const char* reduction, bool data, bool update, bool expected) {
#if !defined(USE_CUDA)
  GTEST_SKIP() << "CUDA is required for CUDA ScatterElements bool coverage.";
#endif
  SCOPED_TRACE(testing::Message() << "dtype=bool, reduction=" << reduction);
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", reduction);
  test.AddInput<bool>("data", {1}, {data});
  test.AddInput<int64_t>("indices", {1}, {0});
  test.AddInput<bool>("updates", {1}, {update});
  test.AddOutput<bool>("y", {1}, {expected});
  test.SetCustomOutputVerifier([expected](const std::vector<OrtValue>& fetches, const std::string& provider_type) {
    ASSERT_EQ(provider_type, kCudaExecutionProvider);
    ASSERT_EQ(fetches.size(), 1u);
    ASSERT_TRUE(fetches[0].IsTensor());
    const auto& output = fetches[0].Get<Tensor>();
    ASSERT_EQ(output.Shape().Size(), 1);
    EXPECT_EQ(output.Data<bool>()[0], expected);
    EXPECT_EQ(static_cast<const uint8_t*>(output.DataRaw())[0], expected ? 1u : 0u);
  });
#if defined(USE_CUDA)
  auto cuda_ep = DefaultCudaExecutionProvider();
  ASSERT_NE(cuda_ep, nullptr);
  test.ConfigEp(std::move(cuda_ep)).RunWithConfig();
#endif
}

}  // namespace

TEST(ScatterElements, FullDTypeInt8Smoke) {
  RunCudaScatterElementsReduction<int8_t>("int8", "max", -5, 3, 3);
}

TEST(ScatterElements, FullDTypeUInt8) {
  RunCudaScatterElementsReduction<uint8_t>("uint8", "add", 200, 20, 220);
  RunCudaScatterElementsReduction<uint8_t>("uint8", "mul", 12, 11, 132);
  RunCudaScatterElementsReduction<uint8_t>("uint8", "min", 200, 100, 100);
  RunCudaScatterElementsReduction<uint8_t>("uint8", "max", 200, 100, 200);
}

TEST(ScatterElements, FullDTypeInt16) {
  RunCudaScatterElementsReduction<int16_t>("int16", "add", 0x3c00, 0x3c00, 0x7800);
  RunCudaScatterElementsReduction<int16_t>("int16", "mul", 2, 3, 6);
  RunCudaScatterElementsReduction<int16_t>("int16", "min", -17408, -16384, -17408);
  RunCudaScatterElementsReduction<int16_t>("int16", "max", -17408, -16384, -16384);
}

TEST(ScatterElements, FullDTypeUInt16) {
  RunCudaScatterElementsReduction<uint16_t>("uint16", "add", 0x3c00, 0x3c00, 0x7800);
  RunCudaScatterElementsReduction<uint16_t>("uint16", "mul", 2, 3, 6);
  RunCudaScatterElementsReduction<uint16_t>("uint16", "mul", 50000, 2, 34464);
  RunCudaScatterElementsReduction<uint16_t>("uint16", "min", 0xbc00, 0xc000, 0xbc00);
  RunCudaScatterElementsReduction<uint16_t>("uint16", "max", 0xbc00, 0xc000, 0xc000);
}

// These IEEE 1.0 bit patterns are intentional: they detect accidental float/double representative dispatch.
TEST(ScatterElements, FullDTypeInt32) {
  RunCudaScatterElementsReduction<int32_t>("int32", "add", 0x3f800000, 0x3f800000, 0x7f000000);
  RunCudaScatterElementsReduction<int32_t>("int32", "add", std::numeric_limits<int32_t>::max(), 1,
                                           std::numeric_limits<int32_t>::min());
  RunCudaScatterElementsReduction<int32_t>("int32", "mul", 2, 3, 6);
  RunCudaScatterElementsReduction<int32_t>("int32", "min", -1082130432, -1073741824, -1082130432);
  RunCudaScatterElementsReduction<int32_t>("int32", "max", -1082130432, -1073741824, -1073741824);
}

TEST(ScatterElements, FullDTypeUInt32) {
  RunCudaScatterElementsReduction<uint32_t>("uint32", "add", 0x3f800000u, 0x3f800000u, 0x7f000000u);
  RunCudaScatterElementsReduction<uint32_t>("uint32", "mul", 2u, 3u, 6u);
  RunCudaScatterElementsReduction<uint32_t>("uint32", "min", 0xbf800000u, 0x3f800000u, 0x3f800000u);
  RunCudaScatterElementsReduction<uint32_t>("uint32", "max", 0xbf800000u, 0x3f800000u, 0xbf800000u);
}

TEST(ScatterElements, FullDTypeInt64) {
  RunCudaScatterElementsReduction<int64_t>("int64", "add", 0x3ff0000000000000LL, 0x3ff0000000000000LL,
                                           0x7fe0000000000000LL);
  RunCudaScatterElementsReduction<int64_t>("int64", "mul", 2, 3, 6);
  RunCudaScatterElementsReduction<int64_t>("int64", "mul", std::numeric_limits<int64_t>::max(), 2, -2);
  RunCudaScatterElementsReduction<int64_t>("int64", "min", -4616189618054758400LL, -4611686018427387904LL,
                                           -4616189618054758400LL);
  RunCudaScatterElementsReduction<int64_t>("int64", "max", -4616189618054758400LL, -4611686018427387904LL,
                                           -4611686018427387904LL);
}

TEST(ScatterElements, FullDTypeUInt64) {
  RunCudaScatterElementsReduction<uint64_t>("uint64", "add", 0x3ff0000000000000ULL, 0x3ff0000000000000ULL,
                                            0x7fe0000000000000ULL);
  RunCudaScatterElementsReduction<uint64_t>("uint64", "mul", 2, 3, 6);
  RunCudaScatterElementsReduction<uint64_t>("uint64", "min", 0xbff0000000000000ULL, 0xc000000000000000ULL,
                                            0xbff0000000000000ULL);
  RunCudaScatterElementsReduction<uint64_t>("uint64", "max", 0xbff0000000000000ULL, 0xc000000000000000ULL,
                                            0xc000000000000000ULL);
}

TEST(ScatterElements, FullDTypeBFloat16) {
  RunCudaScatterElementsReduction<BFloat16>("BFloat16", "add", BFloat16::FromBits(0x3f80),
                                            BFloat16::FromBits(0x3f80), BFloat16::FromBits(0x4000));
  RunCudaScatterElementsReduction<BFloat16>("BFloat16", "mul", BFloat16::FromBits(0x4000),
                                            BFloat16::FromBits(0x4040), BFloat16::FromBits(0x40c0));
  RunCudaScatterElementsReduction<BFloat16>("BFloat16", "min", BFloat16::FromBits(0x7f80),
                                            BFloat16::FromBits(0x3f80), BFloat16::FromBits(0x3f80));
  RunCudaScatterElementsReduction<BFloat16>("BFloat16", "max", BFloat16::FromBits(0x3f80),
                                            BFloat16::FromBits(0x7f80), BFloat16::FromBits(0x7f80));
}

TEST(ScatterElements, BoolReductionsAndCanonicalStorage) {
  RunCudaScatterElementsBoolReduction("add", false, true, true);
  RunCudaScatterElementsBoolReduction("add", true, true, true);
  RunCudaScatterElementsBoolReduction("mul", true, false, false);
  RunCudaScatterElementsBoolReduction("min", true, false, false);
  RunCudaScatterElementsBoolReduction("min", true, true, true);
  RunCudaScatterElementsBoolReduction("max", false, true, true);
  RunCudaScatterElementsBoolReduction("max", false, false, false);
}

TEST(ScatterElements, Opset16Int32IndicesAdd) {
  RunCudaScatterElementsReduction<uint32_t, int32_t>("uint32", "add", 0x3f800000u, 0x3f800000u, 0x7f000000u, 16);
}

TEST(ScatterElements, Opset17Int64IndicesMul) {
  RunCudaScatterElementsReduction<int64_t, int64_t>("int64", "mul", 2, 3, 6, 17);
}

TEST(ScatterElements, ContentionPacked8Bit) {
  RunCudaScatterElementsAddContention<uint8_t>("uint8", 1, 1, 201, 200);
}

TEST(ScatterElements, ContentionAdjacentPacked8BitLanes) {
#if !defined(USE_CUDA)
  GTEST_SKIP() << "CUDA is required for CUDA ScatterElements contention coverage.";
#endif
  constexpr size_t updates_per_lane = 64;
  constexpr size_t lane_count = 4;
  std::vector<int64_t> indices;
  indices.reserve(updates_per_lane * lane_count);
  for (size_t i = 0; i < updates_per_lane; ++i) {
    for (size_t lane = 0; lane < lane_count; ++lane) {
      indices.push_back(static_cast<int64_t>(lane));
    }
  }

  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", "add");
  test.AddInput<uint8_t>("data", {4}, {1, 2, 3, 4});
  test.AddInput<int64_t>("indices", {static_cast<int64_t>(indices.size())}, indices);
  test.AddInput<uint8_t>("updates", {static_cast<int64_t>(indices.size())},
                         std::vector<uint8_t>(indices.size(), 1));
  test.AddOutput<uint8_t>("y", {4}, {65, 66, 67, 68});
#if defined(USE_CUDA)
  auto cuda_ep = DefaultCudaExecutionProvider();
  ASSERT_NE(cuda_ep, nullptr);
  test.ConfigEp(std::move(cuda_ep)).RunWithConfig();
#endif
}

TEST(ScatterElements, ContentionPacked16Bit) {
  RunCudaScatterElementsAddContention<uint16_t>("uint16", 1, 1, 513, 512);
}

TEST(ScatterElements, ContentionNative32Bit) {
  RunCudaScatterElementsAddContention<int32_t>("int32", 1, 1, 4097, 4096);
}

TEST(ScatterElements, ContentionNative64Bit) {
  RunCudaScatterElementsAddContention<uint64_t>("uint64", 1, 1, 4097, 4096);
}

TEST(ScatterElements, ContentionBFloat16) {
  RunCudaScatterElementsAddContention<BFloat16>("BFloat16", BFloat16(0.0f), BFloat16(1.0f), BFloat16(128.0f), 128);
}

TEST(ScatterElements, ContentionBoolLogical) {
#if !defined(USE_CUDA)
  GTEST_SKIP() << "CUDA is required for CUDA ScatterElements contention coverage.";
#endif
  constexpr size_t update_count = 512;
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 0);
  test.AddAttribute<std::string>("reduction", "mul");
  test.AddInput<bool>("data", {1}, {true});
  test.AddInput<int64_t>("indices", {static_cast<int64_t>(update_count)}, std::vector<int64_t>(update_count, 0));
  auto updates = std::make_unique<bool[]>(update_count);
  std::fill_n(updates.get(), update_count, true);
  updates[update_count / 2] = false;
  test.AddInput<bool>("updates", {static_cast<int64_t>(update_count)}, updates.get(), update_count);
  test.AddOutput<bool>("y", {1}, {false});
#if defined(USE_CUDA)
  auto cuda_ep = DefaultCudaExecutionProvider();
  ASSERT_NE(cuda_ep, nullptr);
  test.ConfigEp(std::move(cuda_ep)).RunWithConfig();
#endif
}

}  // namespace test
}  // namespace onnxruntime
