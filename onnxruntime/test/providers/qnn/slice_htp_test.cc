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

// Function that builds a model with a Slice operator.
template <typename DataType>
GetTestModelFn BuildSliceTestCase(const TestInputDef<DataType>& data_def,
                                  const TestInputDef<int64_t>& starts_def,
                                  const TestInputDef<int64_t>& ends_def,
                                  const TestInputDef<int64_t>& axes_def,
                                  const TestInputDef<int64_t>& steps_def) {
  return [data_def, starts_def, ends_def, axes_def, steps_def](ModelTestBuilder& builder) {
    NodeArg* data = MakeTestInput(builder, data_def);
    NodeArg* starts = MakeTestInput(builder, starts_def);
    NodeArg* ends = MakeTestInput(builder, ends_def);
    NodeArg* axes = MakeTestInput(builder, axes_def);
    NodeArg* steps = MakeTestInput(builder, steps_def);

    NodeArg* output = builder.MakeOutput();
    builder.AddNode("Slice", {data, starts, ends, axes, steps}, {output});
  };
}

// Function that builds a QDQ model with a Slice operator.
template <typename QuantType>
static GetTestQDQModelFn<QuantType> BuildQDQSliceTestCase(const TestInputDef<float>& data_def,
                                                          const TestInputDef<int64_t>& starts_def,
                                                          const TestInputDef<int64_t>& ends_def,
                                                          const TestInputDef<int64_t>& axes_def,
                                                          const TestInputDef<int64_t>& steps_def) {
  return [data_def, starts_def, ends_def, axes_def, steps_def](ModelTestBuilder& builder,
                                                               std::vector<QuantParams<QuantType>>& output_qparams) {
    NodeArg* data = MakeTestInput(builder, data_def);
    QuantParams<QuantType> data_qparams = GetTestInputQuantParams(data_def);
    NodeArg* data_qdq = AddQDQNodePair(builder, data, data_qparams.scale, data_qparams.zero_point);

    NodeArg* starts = MakeTestInput(builder, starts_def);
    NodeArg* ends = MakeTestInput(builder, ends_def);
    NodeArg* axes = MakeTestInput(builder, axes_def);
    NodeArg* steps = MakeTestInput(builder, steps_def);

    auto* slice_output = builder.MakeIntermediate();
    builder.AddNode("Slice", {data_qdq, starts, ends, axes, steps}, {slice_output});

    // Add output -> Q -> output_u8
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, slice_output, output_qparams[0].scale, output_qparams[0].zero_point);
  };
}

/**
 * Runs an Slice model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param data_def The data input's definition (shape, is_initializer, data).
 * \param starts_def The starts input's definition.
 * \param ends_def The ends input's definition.
 * \param axes_def The axes input's definition.
 * \param steps_def The steps input's definition.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 */
template <typename QuantType = uint8_t>
static void RunSliceQDQTest(const TestInputDef<float>& data_def,
                            const TestInputDef<int64_t>& starts_def,
                            const TestInputDef<int64_t>& ends_def,
                            const TestInputDef<int64_t>& axes_def,
                            const TestInputDef<int64_t>& steps_def,
                            ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  // Runs model with DQ-> Slice -> Q and compares the outputs of the CPU and QNN EPs.
  TestQDQModelAccuracy(BuildSliceTestCase<float>(data_def, starts_def, ends_def, axes_def, steps_def),
                       BuildQDQSliceTestCase<QuantType>(data_def, starts_def, ends_def, axes_def, steps_def),
                       provider_options,
                       18,
                       expected_ep_assignment,
                       1e-5f);
}

/**
 * Runs an Slice model on the QNN HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param data_def The data (int32_t) input's definition (shape, is_initializer, data).
 * \param starts_def The starts input's definition.
 * \param ends_def The ends input's definition.
 * \param axes_def The axes input's definition.
 * \param steps_def The steps input's definition.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 */
template <typename DataType>
static void RunSliceNonQDQOnHTP(const TestInputDef<DataType>& data_def,
                                const TestInputDef<int64_t>& starts_def,
                                const TestInputDef<int64_t>& ends_def,
                                const TestInputDef<int64_t>& axes_def,
                                const TestInputDef<int64_t>& steps_def,
                                ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  RunQnnModelTest(BuildSliceTestCase<DataType>(data_def, starts_def, ends_def, axes_def, steps_def),
                  provider_options,
                  13,
                  expected_ep_assignment,
                  1e-5f);
}

// Check that QNN compiles DQ -> Slice -> Q as a single unit.
// Inaccuracy detected for output 'output', element 1.
// Output quant params: scale=0.017182711511850357, zero_point=0.
// Expected val: 4.1360063552856445
// QNN QDQ val: 4.0035719871520996 (err 0.13243436813354492)
// CPU QDQ val: 4.1238508224487305 (err 0.012155532836914062)
TEST_F(QnnHTPBackendTests, DISABLED_SliceSmallDataQDQU8) {
  RunSliceQDQTest(TestInputDef<float>({8}, false, 0.0f, 10.0f),
                  TestInputDef<int64_t>({1}, true, {0}),
                  TestInputDef<int64_t>({1}, true, {-1}),
                  TestInputDef<int64_t>({1}, true, {0}),
                  TestInputDef<int64_t>({1}, true, {2}),
                  ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Slice -> Q as a single unit.
// Inaccuracy detected for output 'output', element 8.
// Output quant params: scale=0.39207255840301514, zero_point=0.
// Expected val: 91.531997680664062
// QNN QDQ val: 92.137054443359375 (err 0.6050567626953125)
// CPU QDQ val: 91.3529052734375 (err 0.1790924072265625)
TEST_F(QnnHTPBackendTests, DISABLED_SliceLargePositiveDataQDQU8) {
  RunSliceQDQTest(TestInputDef<float>({5120}, false, 0.0f, 100.0f),
                  TestInputDef<int64_t>({1}, true, {0}),
                  TestInputDef<int64_t>({1}, true, {-1}),
                  TestInputDef<int64_t>({1}, true, {0}),
                  TestInputDef<int64_t>({1}, true, {2}),
                  ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Slice -> Q as a single unit.
// Inaccuracy detected for output 'output', element 1.
// Output quant params: scale=0.78414511680603027, zero_point=128.
// Expected val: -17.279869079589844
// QNN QDQ val: 0 (err 17.279869079589844)
// CPU QDQ val: -17.251192092895508 (err 0.028676986694335938)
TEST_F(QnnHTPBackendTests, DISABLED_SliceLargeNegativeDataQDQU8) {
  RunSliceQDQTest(TestInputDef<float>({5120}, false, -100.0f, 100.0f),
                  TestInputDef<int64_t>({1}, true, {0}),
                  TestInputDef<int64_t>({1}, true, {-1}),
                  TestInputDef<int64_t>({1}, true, {0}),
                  TestInputDef<int64_t>({1}, true, {2}),
                  ExpectedEPNodeAssignment::All);
}

// Check that QNN supports Slice with int32 data input on HTP
TEST_F(QnnHTPBackendTests, SliceInt32OnHTP) {
  RunSliceNonQDQOnHTP<int32_t>(TestInputDef<int32_t>({5120}, false, -100, 100),
                               TestInputDef<int64_t>({1}, true, {0}),
                               TestInputDef<int64_t>({1}, true, {-1}),
                               TestInputDef<int64_t>({1}, true, {0}),
                               TestInputDef<int64_t>({1}, true, {2}),
                               ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif