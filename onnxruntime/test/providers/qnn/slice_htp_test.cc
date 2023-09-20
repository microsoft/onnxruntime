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

  const std::vector<TestInputDef<float>> f32_inputs = {data_def};
  const std::vector<TestInputDef<int64_t>> int64_inputs = {starts_def, ends_def, axes_def, steps_def};

  TestQDQModelAccuracy(BuildOpTestCase<float, int64_t>("Slice", f32_inputs, int64_inputs, {}),
                       BuildQDQOpTestCase<QuantType, int64_t>("Slice", f32_inputs, int64_inputs, {}),
                       provider_options,
                       18,
                       expected_ep_assignment);
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
  auto f32_model_builder = BuildOpTestCase<DataType, int64_t>("Slice", {data_def},
                                                              {starts_def, ends_def, axes_def, steps_def}, {});
  RunQnnModelTest(f32_model_builder,
                  provider_options,
                  13,
                  expected_ep_assignment);
}

// Check that QNN compiles DQ -> Slice -> Q as a single unit.
TEST_F(QnnHTPBackendTests, SliceSmallDataQDQU8) {
  RunSliceQDQTest(TestInputDef<float>({8}, false, 0.0f, 1.0f),
                  TestInputDef<int64_t>({1}, true, {0}),
                  TestInputDef<int64_t>({1}, true, {-1}),
                  TestInputDef<int64_t>({1}, true, {0}),
                  TestInputDef<int64_t>({1}, true, {2}),
                  ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Slice -> Q as a single unit.
TEST_F(QnnHTPBackendTests, SliceLargePositiveDataQDQU8) {
  RunSliceQDQTest(TestInputDef<float>({5120}, false, 0.0f, 1.0f),
                  TestInputDef<int64_t>({1}, true, {0}),
                  TestInputDef<int64_t>({1}, true, {-1}),
                  TestInputDef<int64_t>({1}, true, {0}),
                  TestInputDef<int64_t>({1}, true, {2}),
                  ExpectedEPNodeAssignment::All);
}

// Check that QNN compiles DQ -> Slice -> Q as a single unit.
TEST_F(QnnHTPBackendTests, SliceLargeNegativeDataQDQU8) {
  RunSliceQDQTest(TestInputDef<float>({5120}, false, 0.0f, 1.0f),
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