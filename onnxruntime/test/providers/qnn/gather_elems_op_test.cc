// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <optional>
#include <string>
#include <unordered_map>

#include "core/graph/node_attr_utils.h"
#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "onnx/onnx_pb.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Creates a graph with a single Q/DQ GatherElements operator. Used for testing HTP backend.
template <typename QuantType, typename IndexType>
static GetTestQDQModelFn<QuantType> BuildQDQGatherElemsTestCase(const TestInputDef<float>& input_def,
                                                                const TestInputDef<IndexType>& indices_def,
                                                                const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                                                bool use_contrib_qdq = false) {
  return [input_def, indices_def, attrs, use_contrib_qdq](ModelTestBuilder& builder,
                                                          std::vector<QuantParams<QuantType>>& output_qparams) {
    // input -> Q -> DQ ->
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair<QuantType>(builder, input, input_qparams.scale, input_qparams.zero_point,
                                                   use_contrib_qdq);

    // indices input
    NodeArg* indices_input = MakeTestInput(builder, indices_def);

    // GatherElements op
    NodeArg* gather_output = builder.MakeIntermediate();
    Node& gather_node = builder.AddNode("GatherElements", {input_qdq, indices_input}, {gather_output});

    for (const auto& attr : attrs) {
      gather_node.AddAttributeProto(attr);
    }

    // op_output -> Q -> DQ -> output
    // NOTE: Input and output quantization parameters must be equal for GatherElements.
    output_qparams[0] = input_qparams;  // Overwrite!
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, gather_output, input_qparams.scale,
                                                     input_qparams.zero_point, use_contrib_qdq);
  };
}

// Runs an GatherElements model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
template <typename DataType, typename IndexType>
static void RunCPUGatherElemsOpTest(const TestInputDef<float>& input_def,
                                    const TestInputDef<IndexType>& indices_def,
                                    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                    ExpectedEPNodeAssignment expected_ep_assignment,
                                    int opset = 13) {
  ProviderOptions provider_options;
  float fp32_abs_err = 1e-5f;  // default tolerance

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildOpTestCase<DataType, IndexType>("GatherElements", {input_def}, {indices_def}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

// Runs a QDQ GatherElements model on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match with expected accuracy.
template <typename QuantType, typename IndexType>
static void RunHTPQDQGatherElemsOpTest(const TestInputDef<float>& input_def,
                                       const TestInputDef<IndexType>& indices_def,
                                       const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                       ExpectedEPNodeAssignment expected_ep_assignment,
                                       int opset = 13,
                                       bool use_contrib_qdq = false) {
  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  auto f32_model_builder = BuildOpTestCase<float, IndexType>("GatherElements", {input_def}, {indices_def}, attrs);
  auto qdq_model_builder = BuildQDQGatherElemsTestCase<QuantType, IndexType>(input_def, indices_def, attrs,
                                                                             use_contrib_qdq);

  TestQDQModelAccuracy<QuantType>(f32_model_builder,
                                  qdq_model_builder,
                                  provider_options,
                                  opset,
                                  expected_ep_assignment);
}

// Runs a non-quantized GatherElements model on the QNN HTP backend. Checks the graph node assignment,
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType, typename IndexType>
static void RunHTPGatherElemsOpTest(const TestInputDef<DataType>& input_def,
                                    const TestInputDef<IndexType>& indices_def,
                                    const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                                    ExpectedEPNodeAssignment expected_ep_assignment,
                                    int opset = 13) {
  ProviderOptions provider_options;
  float fp32_abs_err = 1e-5f;  // default tolerance

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  RunQnnModelTest(BuildOpTestCase<DataType, IndexType>("GatherElements", {input_def}, {indices_def}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

//
// CPU tests:
//

// Test GatherElements op on CPU backend:
// positive, dynamic, int64 indices.
TEST_F(QnnCPUBackendTests, GatherElems_DataF32_IndicesInt64) {
  RunCPUGatherElemsOpTest<float, int64_t>(
      TestInputDef<float>({3, 3}, false, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}),
      TestInputDef<int64_t>({2, 3}, false, {1, 2, 0, 2, 0, 0}),
      {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
      ExpectedEPNodeAssignment::All);
}

// Test GatherElements op on CPU backend:
// positive, dynamic, int32 indices.
TEST_F(QnnCPUBackendTests, GatherElems_DataF32_IndicesInt32) {
  RunCPUGatherElemsOpTest<float, int32_t>(
      TestInputDef<float>({3, 3}, false, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}),
      TestInputDef<int32_t>({2, 3}, false, {1, 2, 0, 2, 0, 0}),
      {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
      ExpectedEPNodeAssignment::All);
}

// Test GatherElements op on CPU backend:
// positive, static, int64 indices.
TEST_F(QnnCPUBackendTests, GatherElems_DataF32_StaticIndicesInt64) {
  RunCPUGatherElemsOpTest<float, int64_t>(
      TestInputDef<float>({3, 3}, false, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}),
      TestInputDef<int64_t>({2, 3}, true, {1, 2, 0, 2, 0, 0}),
      {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
      ExpectedEPNodeAssignment::All);
}

// Test GatherElements op on CPU backend:
// positive, static, int32 indices.
TEST_F(QnnCPUBackendTests, GatherElems_DataF32_StaticIndicesInt32) {
  RunCPUGatherElemsOpTest<float, int32_t>(
      TestInputDef<float>({3, 3}, false, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}),
      TestInputDef<int32_t>({2, 3}, true, {1, 2, 0, 2, 0, 0}),
      {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
      ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Test non-quantized GatherElements op on HTP backend:
// Input[0] is int32. Indices are int32, positive, and dynamic. Both inputs are rank 1.
TEST_F(QnnHTPBackendTests, GatherElems_DataInt32_IndicesInt32_Rank1) {
  RunHTPGatherElemsOpTest<int32_t, int32_t>(
      TestInputDef<int32_t>({3}, false, {1, 2, 3}),
      TestInputDef<int32_t>({2}, false, {1, 2}),
      {utils::MakeAttribute("axis", static_cast<int64_t>(0))},
      ExpectedEPNodeAssignment::All);
}

// Test uint8 QDQ GatherElements op on HTP backend:
// positive, dynamic, int32 indices.
TEST_F(QnnHTPBackendTests, GatherElems_DataUint8_IndicesInt32) {
  RunHTPQDQGatherElemsOpTest<uint8_t, int32_t>(
      TestInputDef<float>({3, 3}, false, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}),
      TestInputDef<int32_t>({2, 3}, false, {1, 2, 0, 2, 0, 0}),
      {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
      ExpectedEPNodeAssignment::All);
}

// Test uint8 QDQ GatherElements op on HTP backend:
// positive, static, int32 indices.
TEST_F(QnnHTPBackendTests, GatherElems_DataUint8_StaticIndicesInt32) {
  RunHTPQDQGatherElemsOpTest<uint8_t, int32_t>(
      TestInputDef<float>({3, 3}, false, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}),
      TestInputDef<int32_t>({2, 3}, true, {1, 2, 0, 2, 0, 0}),
      {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
      ExpectedEPNodeAssignment::All);
}

// Test GatherElements op on HTP backend:
// positive, static, int64 indices.
// HTP does not support int64, but QNN EP converts static int64 indices into int32.
TEST_F(QnnHTPBackendTests, GatherElems_DataUint8_StaticIndicesInt64) {
  RunHTPQDQGatherElemsOpTest<uint8_t, int64_t>(
      TestInputDef<float>({3, 3}, false, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}),
      TestInputDef<int64_t>({2, 3}, true, {1, 2, 0, 2, 0, 0}),
      {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
      ExpectedEPNodeAssignment::All);
}

// Test GatherElements op on HTP backend:
// negative, static, int32 indices.
TEST_F(QnnHTPBackendTests, GatherElems_DataUint8_StaticNegIndicesInt32) {
  RunHTPQDQGatherElemsOpTest<uint8_t, int32_t>(
      TestInputDef<float>({3, 3}, false, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}),
      TestInputDef<int32_t>({2, 3}, true, {1, 2, -3, 2, 0, 0}),
      {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
      ExpectedEPNodeAssignment::All);
}

// Test GatherElements op on HTP backend:
// negative, static, int64 indices.
TEST_F(QnnHTPBackendTests, GatherElems_DataUint8_StaticNegIndicesInt64) {
  RunHTPQDQGatherElemsOpTest<uint8_t, int64_t>(
      TestInputDef<float>({3, 3}, false, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}),
      TestInputDef<int64_t>({2, 3}, true, {1, -1, -3, 2, 0, 0}),
      {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
      ExpectedEPNodeAssignment::All);
}

// Test QDQ GatherElements op on HTP backend:
// Input[0] is uint16_t, and rank 4. Indices are int64_t, rank 4, negative, and static.
// Axis is negative (points to last dim).
TEST_F(QnnHTPBackendTests, GatherElems_DataUint16_StaticNegIndicesInt64) {
  const std::vector<int64_t> input_shape = {1, 2, 3, 3};
  const std::vector<float> input_data = GetSequentialFloatData(input_shape, -8.0f, 1.0f);
  RunHTPQDQGatherElemsOpTest<uint16_t, int64_t>(
      TestInputDef<float>(input_shape, false, input_data),
      TestInputDef<int64_t>({1, 1, 2, 2}, true, {0, -1, -3, 2}),
      {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All,
      /*opset*/ 21);
}

// Test QDQ GatherElements op on HTP backend with large number of indices:
TEST_F(QnnHTPBackendTests, GatherElems_DataUint16_StaticNegIndicesInt64_Large) {
  const std::vector<int64_t> input_shape = {12, 1024, 512};
  std::vector<float> input_data(12 * 1024 * 512);
  for (size_t i = 0; i < input_data.size(); i++) {
    input_data[i] = static_cast<float>((static_cast<int64_t>(i) % 8));
  }

  RunHTPQDQGatherElemsOpTest<uint16_t, int64_t>(
      TestInputDef<float>(input_shape, false, input_data),
      TestInputDef<int64_t>({12, 1024, 1024}, true, -512, 511),
      {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All,
      /*opset*/ 21);
}

// Test QDQ GatherElements op on HTP backend with large number of indices.
// TODO: Investigate inaccuracy.
// Negative input[0] values seem to cause inaccuracies with 2M+ indices.
// Inaccuracy detected for output 'output_0', element 131072
// output_range=300, tolerance=0.40000000596046448%.
// Expected val (f32@CPU_EP): 121
// qdq@QNN_EP val: -97.999542236328125 (err: 218.99954223632812, err/output_range: 72.999847412109375%)
// qdq@CPU_EP val: 120.99794006347656 (err: 0.0020599365234375, err/output_range: 0.0006866455078125%)
// abs(qdq@QNN_EP - qdq@CPU_EP) / output_range = 72.999160766601562%
TEST_F(QnnHTPBackendTests, DISABLED_GatherElems_DataUint16_StaticNegIndicesInt64_Large2) {
  // Input data with sequential values from -98.0f to 202.0f
  const std::vector<int64_t> input_shape = {12, 1024, 512};
  std::vector<float> input_data(12 * 1024 * 512);
  for (size_t i = 0; i < input_data.size(); i++) {
    int32_t int_val = -98 + (static_cast<int32_t>(i) % 301);
    input_data[i] = static_cast<float>(int_val);
  }

  // Indices with values between -512 to 511.
  const std::vector<int64_t> indices_shape = {12, 1024, 1024};
  std::vector<int64_t> indices(12 * 1024 * 1024);
  for (size_t i = 0; i < indices.size(); i++) {
    indices[i] = static_cast<int64_t>(-512 + (static_cast<int32_t>(i) % 1024));
  }

  RunHTPQDQGatherElemsOpTest<uint16_t, int64_t>(
      TestInputDef<float>(input_shape, false, input_data),
      TestInputDef<int64_t>(indices_shape, true, indices),
      {utils::MakeAttribute("axis", static_cast<int64_t>(-1))},
      ExpectedEPNodeAssignment::All,
      /*opset*/ 21);
}

// Test GatherElements op on HTP backend:
// Tests that dynamic int64 indices are supported on HTP backend if the indices are a graph input.
// QNN SDK 2.23 added support for Cast from int64 to int32.
TEST_F(QnnHTPBackendTests, GatherElems_DynamicInt64IndicesSupportedAsGraphInput) {
  RunHTPQDQGatherElemsOpTest<uint8_t, int64_t>(
      TestInputDef<float>({3, 3}, false, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f}),
      TestInputDef<int64_t>({2, 3}, false, {1, 2, 0, 2, 0, 0}),
      {utils::MakeAttribute("axis", static_cast<int64_t>(1))},
      ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
