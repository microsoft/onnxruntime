// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include "core/graph/graph.h"
#include "core/graph/node_attr_utils.h"

#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

// Function that builds a float32 model with a Where operator.
GetTestModelFn BuildWhereTestCase(const TestInputDef<bool>& condition_def,
                                  const TestInputDef<float>& x_def,
                                  const TestInputDef<float>& y_def) {
  return [condition_def, x_def, y_def](ModelTestBuilder& builder) {
    NodeArg* condition = MakeTestInput(builder, condition_def);
    NodeArg* x = MakeTestInput(builder, x_def);
    NodeArg* y = MakeTestInput(builder, y_def);

    NodeArg* output = builder.MakeOutput();
    builder.AddNode("Where", {condition, x, y}, {output});
  };
}

// Function that builds a QDQ model with a Where operator.
template <typename QuantType>
static GetTestQDQModelFn<QuantType> BuildQDQWhereTestCase(const TestInputDef<bool>& condition_def,
                                                          const TestInputDef<float>& x_def,
                                                          const TestInputDef<float>& y_def) {
  return [condition_def, x_def, y_def](ModelTestBuilder& builder,
                                       std::vector<QuantParams<QuantType>>& output_qparams) {
    // condition
    NodeArg* condition = MakeTestInput(builder, condition_def);

    // x => Q => DQ =>
    NodeArg* x = MakeTestInput(builder, x_def);
    QuantParams<QuantType> x_qparams = GetTestInputQuantParams<QuantType>(x_def);
    NodeArg* x_qdq = AddQDQNodePair(builder, x, x_qparams.scale, x_qparams.zero_point);

    // y => Q => DQ =>
    NodeArg* y = MakeTestInput(builder, y_def);
    QuantParams<QuantType> y_qparams = GetTestInputQuantParams<QuantType>(y_def);
    NodeArg* y_qdq = AddQDQNodePair(builder, y, y_qparams.scale, y_qparams.zero_point);

    // Where operator.
    auto* where_output = builder.MakeIntermediate();
    builder.AddNode("Where", {condition, x_qdq, y_qdq}, {where_output});

    // Add output -> Q -> output_u8
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, where_output, output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

/**
 * Runs an Where model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param condition_def The condition input's definition (shape, is_initializer, data).
 * \param x_def The x input's definition.
 * \param y_def The y input's definition.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 */
template <typename QuantType = uint8_t>
static void RunWhereQDQTest(const TestInputDef<bool>& condition_def,
                            const TestInputDef<float>& x_def,
                            const TestInputDef<float>& y_def,
                            ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Runs model with DQ-> Where -> Q and compares the outputs of the CPU and QNN EPs.
  TestQDQModelAccuracy(BuildWhereTestCase(condition_def, x_def, y_def),
                       BuildQDQWhereTestCase<QuantType>(condition_def, x_def, y_def),
                       provider_options,
                       18,
                       expected_ep_assignment);
}

// Check that QNN compiles DQ -> Where -> Q as a single unit.
TEST_F(QnnHTPBackendTests, WhereQDQU8) {
  RunWhereQDQTest(TestInputDef<bool>({4, 3, 2}, false,
                                     {true, false, true, false, true, false,
                                      true, false, true, false, true, false,
                                      true, false, true, false, true, false,
                                      true, false, true, false, true, false}),
                  TestInputDef<float>({4, 3, 2}, true, 0.0f, 2.0f),
                  TestInputDef<float>({4, 3, 2}, true, 2.0f, 3.0),
                  ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Where -> Q as a single unit.
// Check QNN Where works with broadcast
TEST_F(QnnHTPBackendTests, WhereBroadcastU8) {
  RunWhereQDQTest(TestInputDef<bool>({2}, false, {true, false}),
                  TestInputDef<float>({4, 3, 2}, true, -2.0f, 2.0f),
                  TestInputDef<float>({1}, true, {3.0f}),
                  ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Where -> Q as a single unit.
// Large data broadcast, QNN v2.13 failed
TEST_F(QnnHTPBackendTests, WhereLargeDataU8) {
  RunWhereQDQTest(TestInputDef<bool>({5120}, false, false, true),
                  TestInputDef<float>({1, 16, 64, 5120}, true, -5000.0f, 0.0f),
                  TestInputDef<float>({1, 16, 64, 5120}, true, 0.0f, 5000.0f),
                  ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Where -> Q as a single unit.
// Large data broadcast, QNN v2.13 failed to finalize graph
// Worked with QNN v2.16
TEST_F(QnnHTPBackendTests, WhereLargeDataBroadcastU8) {
  RunWhereQDQTest(TestInputDef<bool>({5120}, false, false, true),
                  TestInputDef<float>({1, 16, 64, 5120}, true, 0.0f, 1.0f),
                  TestInputDef<float>({1}, true, {3.0f}),
                  ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, WhereLargeDataBroadcastTransformedU8) {
  RunWhereQDQTest(TestInputDef<bool>({1, 1, 5120, 1}, false, false, true),
                  TestInputDef<float>({1, 64, 5120, 16}, true, 0.0f, 1.0f),
                  TestInputDef<float>({1, 1, 1, 1}, true, {3.0f}),
                  ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif