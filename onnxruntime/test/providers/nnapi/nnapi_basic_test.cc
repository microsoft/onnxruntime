// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
#include "core/common/logging/logging.h"
#include "core/common/span_utils.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_builder.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_execution_provider.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/NeuralNetworksTypes.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_lib/nnapi_implementation.h"
#include "core/providers/nnapi/nnapi_builtin/nnapi_api_helper.h"
#include "core/session/inference_session.h"
#include "core/framework/tensorprotoutils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test/test_environment.h"
#include "test/util/include/test_utils.h"
#include "core/framework/data_types_internal.h"

#if !defined(ORT_MINIMAL_BUILD)
// if this is a full build we need the provider test utils
#include "test/providers/provider_test_utils.h"
#include "test/optimizer/qdq_test_utils.h"
#endif

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

namespace onnxruntime {
namespace test {

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)

// Since NNAPI EP handles Reshape and Flatten differently,
// Please see ReshapeOpBuilder::CanSkipReshape in
// <repo_root>/onnxruntime/core/providers/nnapi/nnapi_builtin/builders/op_builder.cc
// We have a separated test for these skip reshape scenarios
TEST(NnapiExecutionProviderTest, ReshapeFlattenTest) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/nnapi_reshape_flatten_test.onnx");

#if defined(__ANDROID__)
  std::vector<int64_t> dims_mul_x = {2, 1, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<int64_t> dims_mul_y = {3, 2, 2};
  std::vector<float> values_mul_y = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  OrtValue ml_value_x;
  AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
  CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x,
                       &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(cpu_allocator, dims_mul_y, values_mul_y,
                       &ml_value_y);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));

  RunAndVerifyOutputsWithEP(model_file_name, "NnapiExecutionProviderTest.ReshapeFlattenTest",
                            std::make_unique<NnapiExecutionProvider>(0),
                            feeds);
#else
  // test load only
  TestModelLoad(model_file_name, std::make_unique<NnapiExecutionProvider>(0), ExpectedEPNodeAssignment::Some);
#endif
}

TEST(NnapiExecutionProviderTest, SigmoidSupportedInputRankTest) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/nnapi_sigmoid_input_rank_test.onnx");

#if defined(__ANDROID__)
  std::vector<int64_t> dims_mul_x = {2, 1, 2, 1, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};

  OrtValue ml_value_x;
  AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
  CreateMLValue<float>(std::move(cpu_allocator), dims_mul_x, values_mul_x,
                       &ml_value_x);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  RunAndVerifyOutputsWithEP(model_file_name, "NnapiExecutionProviderTest.SigmoidSupportedInputRankTest",
                            std::make_unique<NnapiExecutionProvider>(0),
                            feeds, {ExpectedEPNodeAssignment::None} /* params */);
#else
  // test load only
  TestModelLoad(model_file_name, std::make_unique<NnapiExecutionProvider>(0), ExpectedEPNodeAssignment::None);
#endif
}

// Since NNAPI EP does not support dynamic shape input and we now switch from the approach of immediately rejecting
// the whole graph in NNAPI EP if it has a dynamic input to check at individual operator support check level, we have a
// separated test here.
// Please see BaseOpBuilder::HasSupportedInputs in <repo_root>/onnxruntime/core/providers/nnapi/nnapi_builtin/builders/op_support_checker.cc
TEST(NnapiExecutionProviderTest, DynamicGraphInputTest) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/ep_dynamic_graph_input_test.onnx");

#if defined(__ANDROID__)
  std::vector<int64_t> dims_mul_x = {1, 1, 4, 4};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  OrtValue ml_value_x;
  AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
  CreateMLValue<float>(std::move(cpu_allocator), dims_mul_x, values_mul_x,
                       &ml_value_x);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  RunAndVerifyOutputsWithEP(model_file_name, "NnapiExecutionProviderTest.DynamicGraphInputTest",
                            std::make_unique<NnapiExecutionProvider>(0),
                            feeds);
#else
  TestModelLoad(model_file_name, std::make_unique<NnapiExecutionProvider>(0),
                [](const Graph& graph) { ASSERT_EQ(CountAssignedNodes(graph, kNnapiExecutionProvider), 1)
                                             << "Exactly one node (Add) should have been taken by the NNAPI EP"; });
#endif
}

// This is to test the uint8 handling of operators without "QLinear" such as Concat and Transpose
// NNAPI will require scale and zero point for inputs of all quantized operations
// For these operators without "Qlinear", there is no information about the scale and zero point, we can
// only fetch these from the output of the previous node
// So uint8 support of these operators will only be enabled when they are internal to the graph
// by not consuming graph inputs
TEST(NnapiExecutionProviderTest, InternalUint8SupportTest) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/nnapi_internal_uint8_support.onnx");

#if defined(__ANDROID__)
  std::vector<int64_t> dims_x = {1, 1, 1, 3};
  std::vector<float> values_x = {0.0f, 256.0f, 512.0f};
  OrtValue ml_value_x;
  AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
  CreateMLValue<float>(std::move(cpu_allocator), dims_x, values_x,
                       &ml_value_x);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));

  RunAndVerifyOutputsWithEP(model_file_name, "NnapiExecutionProviderTest.InternalUint8SupportTest",
                            std::make_unique<NnapiExecutionProvider>(0),
                            feeds);
#else
  TestModelLoad(model_file_name, std::make_unique<NnapiExecutionProvider>(0), ExpectedEPNodeAssignment::Some);
#endif
}

TEST(NnapiExecutionProviderTest, FunctionTest) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("nnapi_execution_provider_test_graph.onnx");

  {  // Create the model with 2 add nodes
    onnxruntime::Model model("graph_1", false, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();
    std::vector<onnxruntime::NodeArg*> inputs;
    std::vector<onnxruntime::NodeArg*> outputs;

    // FLOAT tensor.
    ONNX_NAMESPACE::TypeProto float_tensor;
    float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(2);

    auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
    auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
    inputs.push_back(&input_arg_1);
    inputs.push_back(&input_arg_2);
    auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
    outputs.push_back(&output_arg);
    graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

    auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
    inputs.clear();
    inputs.push_back(&output_arg);
    inputs.push_back(&input_arg_3);
    auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &float_tensor);
    outputs.clear();
    outputs.push_back(&output_arg_2);
    graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

    ASSERT_STATUS_OK(graph.Resolve());
    ASSERT_STATUS_OK(onnxruntime::Model::Save(model, model_file_name));
  }

#if defined(__ANDROID__)
  std::vector<int64_t> dims_mul_x = {1, 1, 3, 2};
  std::vector<float> values_mul_x = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;
  AllocatorPtr cpu_allocator = std::make_shared<CPUAllocator>();
  CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x,
                       &ml_value_x);
  OrtValue ml_value_y;
  CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x,
                       &ml_value_y);
  OrtValue ml_value_z;
  CreateMLValue<float>(cpu_allocator, dims_mul_x, values_mul_x,
                       &ml_value_z);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value_x));
  feeds.insert(std::make_pair("Y", ml_value_y));
  feeds.insert(std::make_pair("Z", ml_value_z));

  RunAndVerifyOutputsWithEP(model_file_name, "NnapiExecutionProviderTest.FunctionTest",
                            std::make_unique<NnapiExecutionProvider>(0),
                            feeds);
#else
  TestModelLoad(model_file_name, std::make_unique<NnapiExecutionProvider>(0), ExpectedEPNodeAssignment::Some);
#endif
}

TEST(NnapiExecutionProviderTest, TestNoShapeInputModel) {
  const ORTCHAR_T* model_file_name = ORT_TSTR("input_with_no_shape_test_graph.onnx");

  {  // Create the model with 2 add nodes, the graph has 2 inputs with no shape
    onnxruntime::Model model("graph_1", false, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();
    std::vector<onnxruntime::NodeArg*> inputs;
    std::vector<onnxruntime::NodeArg*> outputs;

    // FLOAT tensor without shape
    ONNX_NAMESPACE::TypeProto float_tensor;
    float_tensor.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

    auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
    auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
    inputs.push_back(&input_arg_1);
    inputs.push_back(&input_arg_2);
    auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
    outputs.push_back(&output_arg);
    graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

    auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
    inputs.clear();
    inputs.push_back(&output_arg);
    inputs.push_back(&input_arg_3);
    auto& output_arg_2 = graph.GetOrCreateNodeArg("M", &float_tensor);
    outputs.clear();
    outputs.push_back(&output_arg_2);
    graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

    ASSERT_STATUS_OK(graph.Resolve());
    ASSERT_STATUS_OK(onnxruntime::Model::Save(model, model_file_name));
  }

  // test load only
  // since we know NNAPI supports Add op, but both Add ops in the graph has no input shape
  // verify the entire graph will not be assigned to NNAPI EP
  TestModelLoad(model_file_name, std::make_unique<NnapiExecutionProvider>(0), ExpectedEPNodeAssignment::None);
}

static void RunQDQModelTest(
    const GetQDQTestCaseFn& build_test_case,
    const char* test_description,
    const EPVerificationParams& params = EPVerificationParams()) {
  onnxruntime::Model model(test_description, false, DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  build_test_case(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);
  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

#if defined(__ANDROID__)
  RunAndVerifyOutputsWithEP(model_data_span, "NnapiExecutionProviderTest.TestQDQModel",
                            std::make_unique<NnapiExecutionProvider>(0),
                            helper.feeds_, params);
#else
  // test load only
  TestModelLoad(model_data_span, std::make_unique<NnapiExecutionProvider>(0), params.ep_node_assignment);
#endif
}

TEST(NnapiExecutionProviderTest, DISABLED_TestQDQConv) {
  RunQDQModelTest(BuildQDQConvTestCase<uint8_t /* InputType */,
                                       uint8_t /* WeightType */,
                                       int32_t /* BiasType */,
                                       uint8_t /* OutputType */>(
                      {1, 1, 5, 5} /* input_shape */,
                      {1, 1, 3, 3} /* weights_shape */),
                  "nnapi_qdq_test_graph_conv",
                  {ExpectedEPNodeAssignment::All});
}

TEST(NnapiExecutionProviderTest, DISABLED_TestQDQResizeNCHW) {
  // NNAPI EP does not support the default setting of Resize Op
  // Use bi-linear and asymmetric for NNAPI EP only
  auto Mode = ExpectedEPNodeAssignment::None;
  const auto* nnapi_handle = NnApiImplementation();
  if (nnapi_handle && nnapi::GetNNAPIEffectiveFeatureLevelFromTargetDeviceOption(
                          *nnapi_handle, nnapi::TargetDeviceOption::ALL_DEVICES) >= ANEURALNETWORKS_FEATURE_LEVEL_3) {
    Mode = ExpectedEPNodeAssignment::All;
  }
  RunQDQModelTest(BuildQDQResizeTestCase({1, 3, 64, 64} /* input_shape */,
                                         {1, 3, 32, 32} /* sizes_data */,
                                         "linear" /* mode */,
                                         "asymmetric" /* coordinate_transformation_mode */),
                  "nnapi_qdq_test_graph_resize",
                  {Mode});
}

TEST(NnapiExecutionProviderTest, DISABLED_TestQDQResizeNHWC) {
  // NNAPI EP does not support the default setting of Resize Op
  // Use bi-linear and asymmetric for NNAPI EP only
  RunQDQModelTest(BuildQDQResizeTestCase({1, 64, 64, 3} /* input_shape */,
                                         {1, 32, 32, 3} /* sizes_data */,
                                         "linear" /* mode */,
                                         "asymmetric" /* coordinate_transformation_mode */),
                  "nnapi_qdq_test_graph_resize",
                  {ExpectedEPNodeAssignment::All});
}

TEST(NnapiExecutionProviderTest, DISABLED_TestQDQResize_UnsupportedDefaultSettingNCHW) {
  RunQDQModelTest(BuildQDQResizeTestCase({1, 3, 64, 64} /* input_shape */,
                                         {1, 3, 32, 32} /* sizes_data */),
                  "nnapi_qdq_test_graph_resize_unsupported",
                  {ExpectedEPNodeAssignment::None});
}

TEST(NnapiExecutionProviderTest, DISABLED_TestQDQResize_UnsupportedDefaultSettingNHWC) {
  RunQDQModelTest(BuildQDQResizeTestCase({1, 64, 64, 3} /* input_shape */,
                                         {1, 32, 32, 3} /* sizes_data */),
                  "nnapi_qdq_test_graph_resize_unsupported",
                  {ExpectedEPNodeAssignment::None});
}

TEST(NnapiExecutionProviderTest, DISABLED_TestQDQAveragePool) {
  // NNAPI use different rounding, which may cause ~1% difference in the result
  RunQDQModelTest(BuildQDQAveragePoolTestCase<uint8_t /* InputType */,
                                              uint8_t /* OutputType */>(
                      {1, 3, 32, 32} /* input_shape */),
                  "nnapi_qdq_test_graph_averagepool",
                  {
                      ExpectedEPNodeAssignment::All,
                      1e-2f /* fp32_abs_err */,
                  });
}

TEST(NnapiExecutionProviderTest, DISABLED_TestQDQAdd) {
  RunQDQModelTest(BuildBinaryOpTestCase<uint8_t /* Input1Type */,
                                        uint8_t /* Input2Type */,
                                        uint8_t /* OutputType */>(
                      {1, 23, 13, 13} /* input_shape */,
                      "Add" /* op_type */),
                  "nnapi_qdq_test_graph_add",
                  {ExpectedEPNodeAssignment::All});
}

TEST(NnapiExecutionProviderTest, DISABLED_TestQDQMul) {
  // NNAPI use different rounding, which may cause ~1% difference in the result
  RunQDQModelTest(BuildBinaryOpTestCase<uint8_t /* Input1Type */,
                                        uint8_t /* Input2Type */,
                                        uint8_t /* OutputType */>(
                      {1, 23, 13, 13} /* input_shape */,
                      "Mul" /* op_type */),
                  "nnapi_qdq_test_graph_mul",
                  {
                      ExpectedEPNodeAssignment::All,
                      1e-2f /* fp32_abs_err */
                  });
}

TEST(NnapiExecutionProviderTest, TestQDQTranspose) {
  RunQDQModelTest(BuildQDQTransposeTestCase<uint8_t /* InputType */,
                                            uint8_t /* OutputType */>(
                      {1, 3, 32, 32} /* input_shape */,
                      {0, 3, 1, 2} /* perms */),
                  "nnapi_qdq_test_graph_transpose",
                  {ExpectedEPNodeAssignment::All});
}

TEST(NnapiExecutionProviderTest, DISABLED_TestQDQReshape) {
  RunQDQModelTest(BuildQDQReshapeTestCase({1, 3, 64, 64} /* input_shape */,
                                          {1, 64, 64, 3} /* reshape_shape */),
                  "nnapi_qdq_test_graph_reshape",
                  {ExpectedEPNodeAssignment::All});
}

TEST(NnapiExecutionProviderTest, TestQDQSoftMax) {
  RunQDQModelTest(BuildQDQSoftMaxTestCase<uint8_t, uint8_t>(
                      {1, 32} /* input_shape */,
                      static_cast<int64_t>(1) /* axis */,
                      1.f / 256 /* output_scales */,
                      0 /* output_zp */),
                  "nnapi_qdq_test_graph_softmax",
                  {ExpectedEPNodeAssignment::All});
}

// This is to verify when Nnapi required scale and zero point are not satisfied
// the model can work as expected. (no nodes should be handled by Nnapi)
TEST(NnapiExecutionProviderTest, TestQDQSoftMax_UnsupportedOutputScaleAndZp) {
  RunQDQModelTest(BuildQDQSoftMaxTestCase<uint8_t, uint8_t>(
                      {1, 32} /* input_shape */,
                      static_cast<int64_t>(1) /* axis */,
                      0.002f /* output_scales */,
                      1 /* output_zp */),
                  "nnapi_qdq_test_graph_softmax_unsupported",
                  {ExpectedEPNodeAssignment::None});
}

TEST(NnapiExecutionProviderTest, DISABLED_TestQDQConcat) {
  RunQDQModelTest(BuildQDQConcatTestCase(
                      {
                          {1, 6, 36},
                          {1, 6, 8},
                          {1, 6, 2},
                      } /* input_shapes */,
                      2 /* axis */),
                  "nnapi_qdq_test_graph_concat",
                  {ExpectedEPNodeAssignment::All});
}

#if defined(__ANDROID__)
TEST(NnapiExecutionProviderTest, DISABLED_TestQDQConcat_UnsupportedInputScalesAndZp) {
  // This is to verify all the inputs have the same scale and zp as input 0 for API 28-
  // Currently, this test can only be run locally with a android emulator with API < 29
  // See https://developer.android.com/studio/run/emulator-commandline for some info on
  // starting a testing android emulator in command line. (Run an android build with emulator started)
  // TODO: consider to configure this and enable it to run in Android CI.
  const auto* nnapi_handle = NnApiImplementation();
  if (nnapi_handle && nnapi::GetNNAPIEffectiveFeatureLevelFromTargetDeviceOption(
                          *nnapi_handle, nnapi::TargetDeviceOption::ALL_DEVICES) < ANEURALNETWORKS_FEATURE_LEVEL_3) {
    RunQDQModelTest(BuildQDQConcatTestCaseUnsupportedInputScaleZp(),
                    "nnapi_qdq_test_graph_concat_unsupported",
                    {ExpectedEPNodeAssignment::None});
  }
}
#endif

TEST(NnapiExecutionProviderTest, DISABLED_TestQDQGemm) {
  RunQDQModelTest(BuildQDQGemmTestCase<uint8_t, uint8_t, uint8_t>(
                      {2, 2} /* input_shape1 */,
                      {2, 2} /* input_shape2 */,
                      true /* has_bias */,
                      1 /* transB */),
                  "nnapi_qdq_test_graph_gemm_1",
                  {ExpectedEPNodeAssignment::All});
}

TEST(NnapiExecutionProviderTest, DISABLED_TestQDQGemm_NoTransB) {
  RunQDQModelTest(BuildQDQGemmTestCase<uint8_t, uint8_t, uint8_t>(
                      {2, 2} /* input_shape1 */,
                      {2, 2} /* input_shape2 */,
                      true /* has_bias */,
                      0 /* transB */),
                  "nnapi_qdq_test_graph_gemm_2",
                  {ExpectedEPNodeAssignment::All});
}

TEST(NnapiExecutionProviderTest, DISABLED_TestQDQGemm_NoBias) {
  RunQDQModelTest(BuildQDQGemmTestCase<uint8_t, uint8_t, uint8_t>(
                      {2, 2} /* input_shape1 */,
                      {2, 2} /* input_shape2 */,
                      false /* has_bias */,
                      1 /* transB */),
                  "nnapi_qdq_test_graph_gemm_3",
                  {ExpectedEPNodeAssignment::All});
}

TEST(NnapiExecutionProviderTest, DISABLED_TestQDQMatMul) {
  RunQDQModelTest(BuildQDQMatMulTestCase(
                      {2, 2} /* input_shape1 */,
                      {2, 2} /* input_shape2 */),
                  "nnapi_qdq_test_graph_matmul",
                  {ExpectedEPNodeAssignment::All});
}

// zero inputs test
TEST(NnapiExecutionProviderTest, DISABLED_TestCast) {
  std::vector<int64_t> input1_shape{1, 2, 3, 4};
  auto build_func = [input1_shape](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInitializer<float>(input1_shape, -100.f, 100.f);
    auto* output_arg = builder.MakeOutput();

    builder.AddNode("CastLike", {input_arg, input_arg}, {output_arg});
  };
  RunQDQModelTest(build_func, "nnapi_qdq_test_graph_cast", {ExpectedEPNodeAssignment::None});
}

TEST(NnapiExecutionProviderTest, TestGather) {
  auto BuildGatherTestCase = [](const std::vector<int64_t>& input_shape,
                                bool scalar_indices) {
    return [input_shape, scalar_indices](ModelTestBuilder& builder) {
      auto* input_arg = builder.MakeInput<float>(input_shape,
                                                 std::numeric_limits<float>::min(),
                                                 std::numeric_limits<float>::max());
      auto* output_arg = builder.MakeOutput();
      auto* indices = builder.MakeScalarInitializer<int64_t>(1);
      if (!scalar_indices) {
        indices = builder.Make1DInitializer<int64_t>({1});
      }
      auto& gather_node = builder.AddNode("Gather", {input_arg, indices}, {output_arg});
      gather_node.AddAttribute("axis", int64_t(1));
    };
  };

  RunQDQModelTest(BuildGatherTestCase({10, 5, 5} /* input_shape */, true /* scalar_indices */),
                  "nnapi_test_graph_gather_scalar", {ExpectedEPNodeAssignment::All});

  RunQDQModelTest(BuildGatherTestCase({10, 5, 5} /* input_shape */, false /* not scalar_indices */),
                  "nnapi_test_graph_gather",
                  {ExpectedEPNodeAssignment::All});
}

#endif  // !(ORT_MINIMAL_BUILD)

TEST(NnapiExecutionProviderTest, NNAPIFlagsTest) {
  uint32_t nnapi_flags = NNAPI_FLAG_USE_NONE;
  nnapi_flags |= NNAPI_FLAG_USE_FP16;
  onnxruntime::NnapiExecutionProvider nnapi_ep(nnapi_flags);
  const auto flags = nnapi_ep.GetNNAPIFlags();
  ASSERT_TRUE(flags & NNAPI_FLAG_USE_FP16);
  ASSERT_FALSE(flags & NNAPI_FLAG_USE_NCHW);
}

TEST(NnapiExecutionProviderTest, TestOrtFormatModel) {
  // mnist model that has only had basic optimizations applied. nnapi should be able to take at least some of the nodes
  const ORTCHAR_T* model_file_name = ORT_TSTR("testdata/mnist.basic.ort");

// The execution can only be performed on Android
#if defined(__ANDROID__)
  RandomValueGenerator random{};
  const std::vector<int64_t> dims = {1, 1, 28, 28};
  std::vector<float> data = random.Gaussian<float>(dims, 0.0f, 1.f);

  OrtValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->CreatePreferredAllocators()[0], dims, data,
                       &ml_value);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("Input3", ml_value));

  RunAndVerifyOutputsWithEP(model_file_name, "NnapiExecutionProviderTest.TestOrtFormatModel",
                            std::make_unique<NnapiExecutionProvider>(0),
                            feeds);
#else
  // test load only
  TestModelLoad(model_file_name, std::make_unique<NnapiExecutionProvider>(0), ExpectedEPNodeAssignment::Some);
#endif
}

// test that NNAPI EP can process an activation node that is outside of its partition
TEST(NnapiExecutionProviderTest, ActivationOutsideOfPartition) {
  // model starts with Conv -> Relu
  constexpr auto* model_file_name = ORT_TSTR("testdata/mnist.basic.ort");
  // stop NNAPI partitioning at Relu so NNAPI EP only takes first Conv
  const auto nnapi_partitioning_stop_ops = "Relu";
  TestModelLoad(model_file_name, std::make_unique<NnapiExecutionProvider>(0, nnapi_partitioning_stop_ops),
                // expect one NNAPI partition
                [](const Graph& graph) { ASSERT_EQ(CountAssignedNodes(graph, kNnapiExecutionProvider), 1)
                                             << "Exactly one node should have been taken by the NNAPI EP"; });
}

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
