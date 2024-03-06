// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <ctime>
#include <cstdlib>

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
  // OpenVINO doesn't support negative indices value.
  // Disable TensorRT due to missing int8 calibrator.
  if (std::is_same<T, int8_t>::value) {
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
  } else if (std::is_same<T, MLFloat16>::value) {
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
  } else {
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
  }

  onnxruntime::test::OpTester test1("ScatterElements", 11);
  if (has_axis) test1.AddAttribute<int64_t>("axis", axis);
  test1.AddInput<T>("data", input_dims, input_data);
  test1.AddInput<TIndex>("indices", indices_dims, indices_data);
  test1.AddInput<T>("updates", indices_dims, updates_data);
  test1.AddOutput<T>("y", input_dims, output_data);
  if (std::is_same<T, int8_t>::value) {
    test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
  } else if (std::is_same<T, MLFloat16>::value) {
    test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
  } else {
    test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
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
           {kTensorrtExecutionProvider});
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
           {kCudaExecutionProvider, kCudaNHWCExecutionProvider, kTensorrtExecutionProvider});
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
#if defined(OPENVINO_CONFIG_GPU_FP32) || defined(OPENVINO_CONFIG_GPU_FP16)
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

  test.AddInput<float>("data", {2, 3}, {-9.f, -4.f, -1.f, -7.f, -3.f, -6.f});
  test.AddInput<int64_t>("indices", {4, 3}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  test.AddInput<float>("updates", {4, 3}, {1.f, 1.f, 1.f, 2.f, 2.f, 2.f, 3.f, 3.f, 3.f, 4.f, 4.f, 4.f});
  test.AddOutput<float>("y", {2, 3}, {-9.f, -4.f, -1.f, -7.f + (1.f + 2.f + 3.f + 4.f), -3.f + (1.f + 2.f + 3.f + 4.f), -6.f + (1.f + 2.f + 3.f + 4.f)});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(ScatterElements, AddReductionAxis1) {
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<std::string>("reduction", "add");

  // update's slice shape is {2, 1}
  test.AddInput<float>("data", {2, 3}, {9.f, 4.f, 1.f, 7.f, 3.f, 6.f});
  test.AddInput<int64_t>("indices", {2, 4}, {1, 1, 1, 1, 1, 1, 1, 1});
  test.AddInput<float>("updates", {2, 4}, {2.f, 5.f, 3.f, 6.f, 7.f, 9.f, 8.f, 10.f});
  test.AddOutput<float>("y", {2, 3}, {9.f, 4.f + (2.f + 5.f + 3.f + 6.f), 1.f, 7.f, 3.f + (7.f + 9.f + 8.f + 10.f), 6.f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
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

TEST(ScatterElements, MulReductionAxis1) {
  OpTester test("ScatterElements", 18);
  test.AddAttribute<int64_t>("axis", 1);
  test.AddAttribute<std::string>("reduction", "mul");

  // update's slice shape is {2, 1}
  test.AddInput<float>("data", {2, 3}, {9.f, 4.f, 1.f, 7.f, 3.f, 6.f});
  test.AddInput<int64_t>("indices", {2, 4}, {1, 1, 1, 1, 1, 1, 1, 1});
  test.AddInput<float>("updates", {2, 4}, {2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f});
  test.AddOutput<float>("y", {2, 3}, {9.f, 4.f * (2.f * 3.f * 4.f * 5.f), 1.f, 7.f, 3.f * (6.f * 7.f * 8.f * 9.f), 6.f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
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

}  // namespace test
}  // namespace onnxruntime
