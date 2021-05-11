// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {

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
      {"test padding and broadcast",
       {3, 1, 1, 2},
       {2, 2, 2},
       {3, 2, 1, 2},
       {2, 3, 6, 7, 6, 11, 26, 31, 10, 19, 46, 55}});

  test_cases.push_back(
      {"test padding and broadcast",
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

  return test_cases;
}

template <typename T>
void RunMatMulTest(int32_t opset_version, bool is_b_constant = false) {
  std::vector<T> common_input_vals{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  for (auto t : GenerateTestCases<T>()) {
    OpTester test("MatMul", opset_version);

    int64_t size0 = TensorShape::ReinterpretBaseType(t.input0_dims).SizeHelper(0, t.input0_dims.size());
    std::vector<T> input0_vals(common_input_vals.cbegin(), common_input_vals.cbegin() + size0);
    test.AddInput<T>("A", t.input0_dims, input0_vals);

    int64_t size1 = TensorShape::ReinterpretBaseType(t.input1_dims).SizeHelper(0, t.input1_dims.size());
    std::vector<T> input1_vals(common_input_vals.cbegin(), common_input_vals.cbegin() + size1);
    test.AddInput<T>("B", t.input1_dims, input1_vals, is_b_constant);

    test.AddOutput<T>("Y", t.expected_dims, t.expected_vals);

    // OpenVINO EP: Disabled temporarily matmul broadcasting not fully supported
    // Disable TensorRT because of unsupported data type
    std::unordered_set<std::string> excluded_providers{kTensorrtExecutionProvider, kOpenVINOExecutionProvider};
    if (is_b_constant) {
      // NNAPI: currently fails for the "test 2D empty input" case
      excluded_providers.insert(kNnapiExecutionProvider);
    }
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", excluded_providers);
  }
}

TEST(MathOpTest, MatMulFloatType) {
  RunMatMulTest<float>(7, false);
  RunMatMulTest<float>(7, true);
}

TEST(MathOpTest, MatMulDoubleType) {
  RunMatMulTest<double>(7);
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

#ifndef ENABLE_TRAINING  // Prepacking is enabled only on non-training builds
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

  auto p_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<float>(), TensorShape({4, 3}),
                                           b_init_values.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator));
  OrtValue b;

  b.Init(p_tensor.release(), DataTypeImpl::GetType<Tensor>(),
         DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  SessionOptions so;
  // Set up B as a shared initializer to be shared between sessions
  ASSERT_EQ(so.AddInitializer("B", &b), Status::OK());

  // We want all sessions running using this OpTester to be able to share pre-packed weights if applicable
  test.AddPrePackedSharedContainerToSessions();

  size_t used_cached_pre_packed_weights_counter = 0;

  // Pre-packing is limited just to the CPU EP for now and we will only test the CPU EP
  // and we want to ensure that it is available in this build
  auto cpu_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    return execution_providers;
  };

  // Session 1
  {
    auto ep_vec = cpu_ep();
    test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &ep_vec, {}, &used_cached_pre_packed_weights_counter);
    ASSERT_EQ(used_cached_pre_packed_weights_counter, static_cast<size_t>(0));  // No pre-packed weights have been shared thus far
  }

  // On some platforms/architectures MLAS may choose to not do any pre-packing and the number of elements
  // in the shared container will be zero in which case this test will be a no-op
  auto number_of_elements_in_shared_prepacked_buffers_container =
      test.GetNumberOfElementsInPrePackedSharedContainer();

  // Session 2
  {
    auto ep_vec = cpu_ep();
    test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &ep_vec, {}, &used_cached_pre_packed_weights_counter);
    ASSERT_EQ(used_cached_pre_packed_weights_counter,
              static_cast<size_t>(number_of_elements_in_shared_prepacked_buffers_container));
  }
}

#endif

}  // namespace test
}  // namespace onnxruntime
