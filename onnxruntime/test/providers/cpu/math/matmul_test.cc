// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/providers/run_options_config_keys.h"
#include "test/common/dnnl_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "default_providers.h"

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
       {3, 1, 1, 2},
       {2, 2, 2},
       {3, 2, 1, 2},
       {2, 3, 6, 7, 6, 11, 26, 31, 10, 19, 46, 55}});

  test_cases.push_back(
      {"test padding and broadcast B > A",
       {2, 3, 2},
       {3, 2, 2, 1},
       {3, 2, 3, 1},
       {1, 3, 5, 33, 43, 53, 5, 23, 41, 85, 111, 137, 9, 43, 77, 137, 179, 221}});

  test_cases.push_back(
      {"test left 1D",
       {2},
       {3, 2, 1},
       {3, 1},
       {1, 3, 5}});

  test_cases.push_back(
      {"test right 1D",
       {3, 1, 2},
       {2},
       {3, 1},
       {1, 3, 5}});

  test_cases.push_back(
      {"test left 1D right 2D",
       {2},
       {2, 3},
       {3},
       {3, 4, 5}});

  test_cases.push_back(
      {"test scalar output",
       {3},
       {3},
       {},
       {5}});

  test_cases.push_back(
      {"test 2D",
       {3, 4},
       {4, 3},
       {3, 3},
       {42, 48, 54, 114, 136, 158, 186, 224, 262}});

  test_cases.push_back(
      {"test 2D special",
       {2, 2, 3},
       {3, 4},
       {2, 2, 4},
       {20, 23, 26, 29, 56, 68, 80, 92, 92, 113, 134, 155, 128, 158, 188, 218}});

  test_cases.push_back(
      {"test 2D special 2",
       {2, 2, 3},
       {1, 3, 4},
       {2, 2, 4},
       {20, 23, 26, 29, 56, 68, 80, 92, 92, 113, 134, 155, 128, 158, 188, 218}});

  test_cases.push_back(
      {"test 2D special 3",
       {2, 6},
       {1, 1, 6, 1},
       {1, 1, 2, 1},
       {55, 145}});

  test_cases.push_back(
      {"test 2D empty input",
       {3, 4},
       {4, 0},
       {3, 0},
       {}});

  test_cases.push_back(
      {"test 3D batch",
       {3, 1, 3},
       {3, 3, 2},
       {3, 1, 2},
       {
           // clang-format off
            10,  13,
           100, 112,
           298, 319,
           // clang-format on
       }});

  test_cases.push_back(
      {"test 4D batch",
       {2, 2, 1, 3},
       {2, 2, 3, 2},
       {2, 2, 1, 2},
       {
           // clang-format off
            10,  13,
           100, 112,
           298, 319,
           604, 634,
           // clang-format on
       }});

  return test_cases;
}

template <typename T>
void RunMatMulTest(int32_t opset_version, bool is_a_constant, bool is_b_constant) {
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
#if defined(__linux__)
    if (t.name == "test padding and broadcast B > A") {
      // Accuracy error with QNN SDK 2.17.0 on CPU backend.
      excluded_providers.insert(kQnnExecutionProvider);
    }
#endif
    test.ConfigExcludeEps(excluded_providers)
        .Config(run_with_tunable_op)
        .RunWithConfig();
  }
}

template <typename T>
void RunMatMulTest(int32_t opset_version) {
  RunMatMulTest<T>(opset_version, false, false);
}

TEST(MathOpTest, MatMulFloatType) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: Assertion failed: m_bufferTensorDesc.TotalTensorSizeInBytes >= ComputeByteSizeFromDimensions(nonBroadcastDimensions, dataType)";
  }
  RunMatMulTest<float>(7, false, false);
}

TEST(MathOpTest, MatMulDoubleType) {
  RunMatMulTest<double>(7);
}

TEST(MathOpTest, MatMulFloatTypeInitializer) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: Assertion failed: m_bufferTensorDesc.TotalTensorSizeInBytes >= ComputeByteSizeFromDimensions(nonBroadcastDimensions, dataType)";
  }
  RunMatMulTest<float>(7, false, true);
}

TEST(MathOpTest, MatMulInt32Type) {
  RunMatMulTest<int32_t>(9);
}

TEST(MathOpTest, MatMulUint32Type) {
  RunMatMulTest<uint32_t>(9);
}

TEST(MathOpTest, MatMulInt64Type) {
  RunMatMulTest<int64_t>(9);
}

TEST(MathOpTest, MatMulUint64Type) {
  RunMatMulTest<uint64_t>(9);
}

#if defined(USE_CUDA) || defined(USE_ROCM)
TEST(MathOpTest, MatMul_Float16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support FP16";
    return;
  }
#endif
  OpTester test("MatMul", 14);

  std::vector<float> A{1.0f, 2.0f, 3.0f, 4.0f,
                       -1.0f, -2.0f, -3.0f, -4.0f};
  std::vector<float> B(12, 1.0f);
  std::vector<float> Y{10.0f, 10.0f, 10.0f,
                       -10.0f, -10.0f, -10.0f};

  std::vector<MLFloat16> f_A(8);
  std::vector<MLFloat16> f_B(12);
  std::vector<MLFloat16> f_Y(6);
  ConvertFloatToMLFloat16(A.data(), f_A.data(), 8);
  ConvertFloatToMLFloat16(B.data(), f_B.data(), 12);
  ConvertFloatToMLFloat16(Y.data(), f_Y.data(), 6);

  test.AddInput<MLFloat16>("A", {2, 4}, f_A);
  test.AddInput<MLFloat16>("B", {4, 3}, f_B);
  test.AddOutput<MLFloat16>("Y", {2, 3}, f_Y);
  test.ConfigExcludeEps({kTensorrtExecutionProvider})  // TensorRT: fp16 is not supported
      .Config(run_with_tunable_op)
      .RunWithConfig();
}
#endif

#if defined(USE_CUDA) || defined(USE_ROCM) || defined(USE_DNNL)
TEST(MathOpTest, MatMul_bfloat16) {
#ifdef USE_CUDA
  int min_cuda_architecture = 530;
  if (!HasCudaEnvironment(min_cuda_architecture)) {
    LOGS_DEFAULT(WARNING) << "Hardware NOT support BFP16";
    return;
  }
#endif
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  OpTester test("MatMul", 14);

  test.AddInput<BFloat16>("A", {2, 4}, MakeBFloat16({1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f}));
  test.AddInput<BFloat16>("B", {4, 3}, MakeBFloat16({1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f}));
  test.AddOutput<BFloat16>("Y", {2, 3}, MakeBFloat16({10.0f, 10.0f, 10.0f, -10.0f, -10.0f, -10.0f}));
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  test.Config(run_with_tunable_op);
#ifdef USE_CUDA
  execution_providers.emplace_back(DefaultCudaExecutionProvider());
#elif USE_ROCM
  execution_providers.emplace_back(DefaultRocmExecutionProvider(/*test_tunable_op=*/true));
  test.ConfigEps(std::move(execution_providers))
      .RunWithConfig();

  execution_providers.clear();
  execution_providers.emplace_back(DefaultRocmExecutionProvider(/*test_tunable_op=*/false));
#elif USE_DNNL
  execution_providers.emplace_back(DefaultDnnlExecutionProvider());
#endif
  test.ConfigEps(std::move(execution_providers))
      .RunWithConfig();
}
#endif

#ifndef ENABLE_TRAINING
// Prepacking is disabled in full training build so no need to test the feature in a training build.
TEST(MathOpTest, MatMulSharedPrepackedWeights) {
  OpTester test("MatMul");

  std::vector<float> b_init_values(12, 1.0f);
  test.AddInput<float>("A", {2, 4},
                       {1.0f, 2.0f, 3.0f, 4.0f,
                        -1.0f, -2.0f, -3.0f, -4.0f});
  // B is to be an initializer for triggering pre-packing
  test.AddInput<float>("B", {4, 3}, b_init_values, true);

  test.AddOutput<float>("Y", {2, 3},
                        {10.0f, 10.0f, 10.0f,
                         -10.0f, -10.0f, -10.0f});

  OrtValue b;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape({4, 3}),
                       b_init_values.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), b);

  SessionOptions so;
  // Set up B as a shared initializer to be shared between sessions
  ASSERT_EQ(so.AddInitializer("B", &b), Status::OK());

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

}  // namespace test
}  // namespace onnxruntime
