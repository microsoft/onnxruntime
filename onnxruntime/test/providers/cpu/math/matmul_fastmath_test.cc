// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/run_options_config_keys.h"
#include "test/common/dnnl_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "default_providers.h"

#if defined(__aarch64__) && defined(__linux__)

namespace onnxruntime {
namespace test {

namespace {

const onnxruntime::RunOptions run_options = []() {
  onnxruntime::RunOptions options{};
  ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOpTesterRunOptionsConfigTestTunableOp, "true"));
  return options;
}();

const constexpr auto run_with_tunable_op = &run_options;

}  // namespace

template <typename T>
struct MatMulTestData {
  std::string name;
  std::vector<int64_t> input0_dims;
  std::vector<int64_t> input1_dims;
  std::vector<int64_t> expected_dims;
  std::vector<T> expected_vals;
};

template <typename T>
std::vector<MatMulTestData<T>> GenerateTestCases() {
  std::vector<MatMulTestData<T>> test_cases;
  test_cases.push_back(
      {"test padding and broadcast A > B",
       {3, 1, 1, 6},
       {2, 6, 7},
       {3, 2, 1, 7},
       {385, 400, 415, 430, 445, 460, 475, 1015, 1030, 1045, 1060, 1075, 1090, 1105, 1015, 1066, 1117, 1168, 1219, 1270, 1321, 3157, 3208, 3259, 3310, 3361, 3412, 3463, 1645, 1732, 1819, 1906, 1993, 2080, 2167, 5299, 5386, 5473, 5560, 5647, 5734, 5821}});

  test_cases.push_back(
      {"test padding and broadcast B > A",
       {2, 3, 12},
       {3, 2, 12, 3},
       {3, 2, 3, 3},
       {1518, 1584, 1650, 3894, 4104, 4314, 6270, 6624, 6978, 26574, 27072, 27570, 34134, 34776, 35418, 41694, 42480, 43266, 6270, 6336, 6402, 19014, 19224, 19434, 31758, 32112, 32466, 62430, 62928, 63426, 80358, 81000, 81642, 98286, 99072, 99858, 11022, 11088, 11154, 34134, 34344, 34554, 57246, 57600, 57954, 98286, 98784, 99282, 126582, 127224, 127866, 154878, 155664, 156450}});

  test_cases.push_back(
      {"test 2D",
       {8, 6},
       {6, 6},
       {8, 6},
       {330, 345, 360, 375, 390, 405, 870, 921, 972, 1023, 1074, 1125, 1410, 1497, 1584, 1671, 1758, 1845, 1950, 2073, 2196, 2319, 2442, 2565, 2490, 2649, 2808, 2967, 3126, 3285, 3030, 3225, 3420, 3615, 3810, 4005, 3570, 3801, 4032, 4263, 4494, 4725, 4110, 4377, 4644, 4911, 5178, 5445}});

  test_cases.push_back(
      {"test 2D special",
       {2, 2, 16},
       {16, 4},
       {2, 2, 4},
       {4960, 5080, 5200, 5320, 12640, 13016, 13392, 13768, 20320, 20952, 21584, 22216, 28000, 28888, 29776, 30664}});

  test_cases.push_back(
      {"test 2D special 2",
       {2, 2, 9},
       {1, 9, 4},
       {2, 2, 4},
       {816, 852, 888, 924, 2112, 2229, 2346, 2463, 3408, 3606, 3804, 4002, 4704, 4983, 5262, 5541}});

  test_cases.push_back(
      {"test 2D special 3",
       {2, 12},
       {1, 1, 12, 3},
       {1, 1, 2, 3},
       {1518, 1584, 1650, 3894, 4104, 4314}});

  test_cases.push_back(
      {"test 3D batch",
       {3, 1, 18},
       {3, 18, 2},
       {3, 1, 2},
       {
           // clang-format off
            3570,  3723,
           26250, 26727,
           72258, 73059,
           // clang-format on
       }});

  test_cases.push_back(
      {"test 4D batch",
       {2, 2, 1, 20},
       {2, 2, 20, 2},
       {2, 2, 1, 2},
       {
           // clang-format off
            4940,  5130,
           36140, 36730,
           99340, 100330,
           194540, 195930,
           // clang-format on
       }});

  return test_cases;
}

template <typename T>
void RunMatMulTest(int32_t opset_version, bool is_a_constant, bool is_b_constant, bool disable_fastmath) {
  for (auto t : GenerateTestCases<T>()) {
    SCOPED_TRACE("test case: " + t.name);

    OpTester test("MatMul", opset_version);

    int64_t size0 = TensorShape::FromExistingBuffer(t.input0_dims).SizeHelper(0, t.input0_dims.size());
    std::vector<T> input0_vals = ValueRange<T>(size0);

    test.AddInput<T>("A", t.input0_dims, input0_vals, is_a_constant);

    int64_t size1 = TensorShape::FromExistingBuffer(t.input1_dims).SizeHelper(0, t.input1_dims.size());
    std::vector<T> input1_vals = ValueRange<T>(size1);
    test.AddInput<T>("B", t.input1_dims, input1_vals, is_b_constant);

    test.AddOutput<T>("Y", t.expected_dims, t.expected_vals);

    // OpenVINO EP: Disabled temporarily matmul broadcasting not fully supported
    // Disable TensorRT because of unsupported data type
    std::unordered_set<std::string> excluded_providers{kTensorrtExecutionProvider, kOpenVINOExecutionProvider};
    if (t.name == "test 2D empty input") {
      // NNAPI: currently fails for the "test 2D empty input" case
      excluded_providers.insert(kNnapiExecutionProvider);
    }

    if ("test padding and broadcast A > B" == t.name || "test 2D empty input" == t.name) {
      // QNN can't handle 0 shap
      excluded_providers.insert(kQnnExecutionProvider);
    }

    SessionOptions so;
    ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
        kOrtSessionOptionsMlasGemmFastMathArm64Bfloat16, "1"));

    test.ConfigExcludeEps(excluded_providers)
        .Config(run_with_tunable_op)
        .Config(so)
        .RunWithConfig();

    if (disable_fastmath) {
      ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
          kOrtSessionOptionsMlasGemmFastMathArm64Bfloat16, "0"));

      test.ConfigExcludeEps(excluded_providers)
          .Config(run_with_tunable_op)
          .Config(so)
          .RunWithConfig();
    }
  }
}

template <typename T>
void RunMatMulTest(int32_t opset_version) {
  RunMatMulTest<T>(opset_version, false, false, false);
}

TEST(MathOpTest, MatMulFloatType_FastMath) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: Assertion failed: m_bufferTensorDesc.TotalTensorSizeInBytes >= ComputeByteSizeFromDimensions(nonBroadcastDimensions, dataType)";
  }
  RunMatMulTest<float>(7, false, false, false);
}

TEST(MathOpTest, MatMulFloatTypeInitializer_FastMath) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: Assertion failed: m_bufferTensorDesc.TotalTensorSizeInBytes >= ComputeByteSizeFromDimensions(nonBroadcastDimensions, dataType)";
  }
  RunMatMulTest<float>(7, false, true, false);
}

TEST(MathOpTest, MatMulInt32Type_FastMath) {
  RunMatMulTest<int32_t>(9);
}

TEST(MathOpTest, MatMulUint32Type_FastMath) {
  RunMatMulTest<uint32_t>(9);
}

TEST(MathOpTest, MatMulInt64Type_FastMath) {
  RunMatMulTest<int64_t>(9);
}

TEST(MathOpTest, MatMulUint64Type_FastMath) {
  RunMatMulTest<uint64_t>(9);
}

#ifndef ENABLE_TRAINING
// Prepacking is disabled in full training build so no need to test the feature in a training build.
TEST(MathOpTest, MatMulSharedPrepackedWeights_FastMath) {
  OpTester test("MatMul");

  std::vector<float> b_init_values(32, 1.0f);
  test.AddInput<float>("A", {8, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f,
                        1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  // B is to be an initializer for triggering pre-packing
  test.AddInput<float>("B", {4, 8}, b_init_values, true);

  test.AddOutput<float>("Y", {8, 8},
                        {10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f,
                         -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f,
                         10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f,
                         -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f,
                         10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f,
                         -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f,
                         10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f, 10.0f,
                         -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f, -10.0f});

  OrtValue b;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape({4, 8}),
                       b_init_values.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), b);

  SessionOptions so;
  // Set up B as a shared initializer to be shared between sessions
  ASSERT_EQ(so.AddInitializer("B", &b), Status::OK());
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(
      kOrtSessionOptionsMlasGemmFastMathArm64Bfloat16, "1"));

  // We want all sessions running using this OpTester to be able to share pre-packed weights if applicable
  test.EnableSharingOfPrePackedWeightsAcrossSessions();

  // Pre-packing is limited just to the CPU EP for now and we will only test the CPU EP
  // and we want to ensure that it is available in this build
  auto cpu_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    return execution_providers;
  };

  size_t number_of_pre_packed_weights_counter_session_1 = 0;
  size_t number_of_shared_pre_packed_weights_counter = 0;

  // Session 1
  {
    test.Config(so)
        .Config(run_with_tunable_op)
        .ConfigEps(cpu_ep())
        .RunWithConfig(&number_of_pre_packed_weights_counter_session_1, &number_of_shared_pre_packed_weights_counter);
    // Assert that no pre-packed weights have been shared thus far
    ASSERT_EQ(number_of_shared_pre_packed_weights_counter, static_cast<size_t>(0));
  }

  auto number_of_elements_in_shared_prepacked_buffers_container =
      test.GetNumPrePackedWeightsShared();
  // Assert that the number of elements in the shared container
  // is the same as the number of weights that have been pre-packed
  ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_elements_in_shared_prepacked_buffers_container);

  // On some platforms/architectures MLAS may choose to not do any pre-packing and the number of elements
  // that have been pre-packed will be zero in which case we do not continue with the testing
  // of "sharing" of pre-packed weights as there are no pre-packed weights to be shared at all.
  if (number_of_pre_packed_weights_counter_session_1 == 0)
    return;

  // Session 2
  {
    size_t number_of_pre_packed_weights_counter_session_2 = 0;
    test.Config(so)
        .Config(run_with_tunable_op)
        .ConfigEps(cpu_ep())
        .RunWithConfig(&number_of_pre_packed_weights_counter_session_2, &number_of_shared_pre_packed_weights_counter);

    // Assert that the same number of weights were pre-packed in both sessions
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_pre_packed_weights_counter_session_2);

    // Assert that the number of pre-packed weights that were shared equals
    // the number of pre-packed weights in the second session
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_2,
              static_cast<size_t>(number_of_shared_pre_packed_weights_counter));
  }
}

#endif

// Dummy run to disable the FastMath mode for the current session
TEST(MathOpTest, MatMulUint64Type_DisableFastMath) {
  RunMatMulTest<uint64_t>(9, false, false, true);
}

}  // namespace test
}  // namespace onnxruntime
#endif  // defined(__aarch64__) && defined(__linux__)
