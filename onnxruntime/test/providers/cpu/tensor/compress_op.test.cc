// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#ifdef USE_CUDA
#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/default_providers.h"
#endif

namespace onnxruntime {
namespace test {

TEST(CompressTest, Compress0) {
  OpTester test("Compress", 9);

  test.AddAttribute("axis", int64_t(0));

  test.AddInput<float>("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<bool>("condition", {3}, {0, 1, 1});
  test.AddOutput<float>("output", {2, 2}, {3.0f, 4.0f, 5.0f, 6.0f});
  test.Run();
}

TEST(CompressTest, Compress1) {
  OpTester test("Compress", 9);

  test.AddAttribute("axis", int64_t(1));

  test.AddInput<float>("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<bool>("condition", {2}, {0, 1});
  test.AddOutput<float>("output", {3, 1}, {2.0f, 4.0f, 6.0f});
  test.Run();
}

TEST(CompressTest, Compress_3dims) {
  OpTester test("Compress", 9);

  test.AddAttribute("axis", int64_t(1));

  test.AddInput<float>("input", {2, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                                            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  test.AddInput<bool>("condition", {2}, {0, 1});
  test.AddOutput<float>("output", {2, 1, 3}, {4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f});
  test.Run();
}

TEST(CompressTest, Compress_condition_all_false) {
  OpTester test("Compress", 9);

  test.AddAttribute("axis", int64_t(1));

  test.AddInput<float>("input", {2, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                                            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  test.AddInput<bool>("condition", {2}, {0, 0});
  test.AddOutput<float>("output", {2, 0, 3}, {});
  test.Run();
}

TEST(CompressTest, Compress_3dims_has_extra_condition) {
  OpTester test("Compress", 9);

  test.AddAttribute("axis", int64_t(1));

  test.AddInput<float>("input", {2, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                                            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  // has condition length = 3 > input_dim[axis] = 2
  test.AddInput<bool>("condition", {3}, {0, 1, 1});
  test.AddOutput<float>("output", {2, 1, 3}, {4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f});
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
}

TEST(CompressTest, Compress_3dims_has_extra_input) {
  OpTester test("Compress", 9);

  test.AddAttribute("axis", int64_t(1));

  test.AddInput<float>("input", {2, 3, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,

                                            10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f});
  // has condition length = 2 < input_dim[axis] = 3
  test.AddInput<bool>("condition", {2}, {0, 1});
  test.AddOutput<float>("output", {2, 1, 3}, {4.0f, 5.0f, 6.0f, 13.0f, 14.0f, 15.0f});
  test.Run();
}

TEST(CompressTest, Compress_default_axis) {
  OpTester test("Compress", 9);

  test.AddInput<float>("input", {3, 2}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  test.AddInput<bool>("condition", {5}, {0, 1, 0, 0, 1});
  test.AddOutput<float>("output", {2}, {2.0f, 5.0f});
  test.Run();
}

// Test that we accumulate to a buffer that does not overflow
TEST(CompressTest, Compress_default_axis_issue_9247_cumulative_sum_overflow) {
  OpTester test("Compress", 9);

  // Generate input longer than 127
  constexpr size_t elements = 150;
  std::vector<float> input;
  for (size_t i = 0; i < elements; ++i) {
    input.push_back(static_cast<float>(i));
  }

  // Let's select all of the elements
  std::unique_ptr<bool[]> all_true = std::make_unique<bool[]>(elements);
  std::fill_n(all_true.get(), elements, true);
  std::vector<int64_t> output_shape{static_cast<int64_t>(elements)};

  test.AddInput<float>("input", {2, 75}, input);
  test.AddInput<bool>("condition", output_shape, all_true.get(), elements);
  // Should get all of the input
  test.AddOutput<float>("output", output_shape, input);
  test.Run();
}

TEST(CompressTest, Compress0_string) {
  OpTester test("Compress", 9);

  test.AddAttribute("axis", int64_t(0));

  test.AddInput<std::string>("input", {3, 2}, {"1", "2", "3", "4", "5", "6"});
  test.AddInput<bool>("condition", {3}, {0, 1, 1});
  test.AddOutput<std::string>("output", {2, 2}, {"3", "4", "5", "6"});
  test.Run();
}

TEST(CompressTest, Compress_default_axis_string) {
  OpTester test("Compress", 9);

  test.AddInput<std::string>("input", {3, 2}, {"1", "2", "3", "4", "5", "6"});
  test.AddInput<bool>("condition", {5}, {0, 1, 0, 0, 1});
  test.AddOutput<std::string>("output", {2}, {"2", "5"});
  test.Run();
}

TEST(CompressTest, Compress_3dims_neg_axis) {
  OpTester test("Compress", 11);

  test.AddAttribute("axis", int64_t(-2));

  test.AddInput<float>("input", {2, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,

                                            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f});
  test.AddInput<bool>("condition", {2}, {0, 1});
  test.AddOutput<float>("output", {2, 1, 3}, {4.0f, 5.0f, 6.0f, 10.0f, 11.0f, 12.0f});
  test.Run();
}

#ifdef USE_CUDA
// Regression test for the CUDA Compress prefix-sum sizing path. A bool condition byte may hold a
// non-canonical value (e.g. 0xFF) at runtime — initializers are normalized to {0, 1} on unpack,
// but runtime-produced bool tensors are not. Without normalizing the byte before the prefix sum,
// 0xFF would be summed as 255 (sizing the output for 255 selected elements) while _CompressKernel
// selects it as a single element via truthiness, so sizing and selection would disagree. This
// test feeds a raw 0xFF condition byte (which OpTester cannot produce, since it normalizes bool
// inputs to {0, 1}) and asserts the output is sized by truthiness.
TEST(CompressTest, Compress_cuda_non_canonical_bool_condition) {
  // Build: output = Compress(input, condition, axis=0)
  auto model = std::make_unique<onnxruntime::Model>("compress_non_canonical_bool", false,
                                                    DefaultLoggingManager().DefaultLogger());
  onnxruntime::Graph& graph = model->MainGraph();

  ONNX_NAMESPACE::TypeProto tensor_float;
  tensor_float.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  ONNX_NAMESPACE::TypeProto tensor_bool;
  tensor_bool.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_BOOL);

  auto& input_arg = graph.GetOrCreateNodeArg("input", &tensor_float);
  auto& condition_arg = graph.GetOrCreateNodeArg("condition", &tensor_bool);
  auto& output_arg = graph.GetOrCreateNodeArg("output", &tensor_float);

  std::vector<onnxruntime::NodeArg*> input_defs{&input_arg, &condition_arg};
  std::vector<onnxruntime::NodeArg*> output_defs{&output_arg};
  auto& node = graph.AddNode("compress", "Compress", "Compress", input_defs, output_defs, nullptr,
                             onnxruntime::kOnnxDomain);
  node.AddAttribute("axis", static_cast<int64_t>(0));
  ASSERT_STATUS_OK(graph.Resolve());

  SessionOptions so;
  so.session_logid = "CompressTest.Compress_cuda_non_canonical_bool_condition";
  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(DefaultCudaExecutionProvider()));

  std::string serialized_model;
  ASSERT_TRUE(model->ToProto().SerializeToString(&serialized_model));
  std::stringstream sstr(serialized_model);
  ASSERT_STATUS_OK(session_object.Load(sstr));
  ASSERT_STATUS_OK(session_object.Initialize());

  AllocatorPtr cpu_allocator = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  OrtValue input_value;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape({3, 2}), cpu_allocator, input_value);
  const float input_data[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  memcpy(input_value.GetMutable<Tensor>()->MutableData<float>(), input_data, sizeof(input_data));

  // Condition {false, non-canonical-true, true}: write a raw 0xFF byte for the middle element to
  // emulate a runtime-produced bool tensor outside the canonical {0, 1} set.
  OrtValue condition_value;
  Tensor::InitOrtValue(DataTypeImpl::GetType<bool>(), TensorShape({3}), cpu_allocator, condition_value);
  auto* condition_bytes =
      reinterpret_cast<uint8_t*>(condition_value.GetMutable<Tensor>()->MutableDataRaw());
  condition_bytes[0] = 0x00;
  condition_bytes[1] = 0xFF;
  condition_bytes[2] = 0x01;

  std::vector<OrtValue> fetches;
  ASSERT_STATUS_OK(session_object.Run(
      std::unordered_map<std::string, OrtValue>{{"input", input_value}, {"condition", condition_value}},
      std::vector<std::string>{"output"}, &fetches));

  ASSERT_EQ(fetches.size(), 1u);
  const Tensor& output = fetches[0].Get<Tensor>();
  // Two non-zero condition bytes select two rows along axis 0 (not 256).
  EXPECT_EQ(output.Shape(), TensorShape({2, 2}));
  const auto output_span = output.DataAsSpan<float>();
  const std::vector<float> expected{3.0f, 4.0f, 5.0f, 6.0f};
  ASSERT_EQ(output_span.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(output_span[i], expected[i]);
  }
}
#endif  // USE_CUDA

}  // namespace test
}  // namespace onnxruntime
