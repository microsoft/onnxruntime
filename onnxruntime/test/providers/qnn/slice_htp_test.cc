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

// Test for "index-out-of-bounds" bug that occurred when a Slice operator
// shared one of its initializer inputs with another op that was processed by QNN EP first.
TEST_F(QnnCPUBackendTests, Slice_SharedInitializersBugFix) {
  // Model with an Add that processes a shared initializer before Slice is processed.
  GetTestModelFn model_fn = [](ModelTestBuilder& builder) {
    NodeArg* input0 = builder.MakeInput<int32_t>({2, 2}, {1, 2, 3, 4});

    // Initializers
    NodeArg* starts_input = builder.Make1DInitializer<int32_t>({1, 0});  // Shared by Add
    NodeArg* ends_input = builder.Make1DInitializer<int32_t>({2, 2});
    NodeArg* axes_input = builder.Make1DInitializer<int32_t>({0, 1});
    NodeArg* steps_input = builder.Make1DInitializer<int32_t>({1, 1});

    // Add input0 with a shared initializer.
    NodeArg* add_output = builder.MakeIntermediate();
    builder.AddNode("Add", {input0, starts_input}, {add_output});

    // Cast Add's output to float.
    NodeArg* cast_output = builder.MakeIntermediate();
    Node& cast_node = builder.AddNode("Cast", {add_output}, {cast_output});
    cast_node.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT));

    // Slice Cast's output
    NodeArg* slice0_out = builder.MakeOutput();
    builder.AddNode("Slice", {cast_output, starts_input, ends_input, axes_input, steps_input}, {slice0_out});
  };

  ProviderOptions provider_options;

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(model_fn,
                  provider_options,
                  13,  // opset
                  ExpectedEPNodeAssignment::All);
}

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
 * \param use_contrib_qdq Force Q/DQ ops to use the com.microsoft domain (enable 16-bit).
 */
template <typename QuantType = uint8_t>
static void RunSliceQDQTest(const TestInputDef<float>& data_def,
                            const TestInputDef<int64_t>& starts_def,
                            const TestInputDef<int64_t>& ends_def,
                            const TestInputDef<int64_t>& axes_def,
                            const TestInputDef<int64_t>& steps_def,
                            ExpectedEPNodeAssignment expected_ep_assignment,
                            bool use_contrib_qdq = false) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  const std::vector<TestInputDef<float>> f32_inputs = {data_def};
  const std::vector<TestInputDef<int64_t>> int64_inputs = {starts_def, ends_def, axes_def, steps_def};

  TestQDQModelAccuracy(BuildOpTestCase<float, int64_t>("Slice", f32_inputs, int64_inputs, {}),
                       BuildQDQOpTestCase<QuantType, int64_t>("Slice", f32_inputs, int64_inputs, {}, kOnnxDomain,
                                                              use_contrib_qdq),
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

// Test 8-bit QDQ Slice with more than 1 axis.
TEST_F(QnnHTPBackendTests, SliceU8_MultAxes) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  RunSliceQDQTest<uint8_t>(TestInputDef<float>({2, 4}, false, input_data),
                           TestInputDef<int64_t>({2}, true, {1, 0}),  // starts
                           TestInputDef<int64_t>({2}, true, {2, 3}),  // ends
                           TestInputDef<int64_t>({2}, true, {0, 1}),  // axes
                           TestInputDef<int64_t>({2}, true, {1, 2}),  // steps
                           ExpectedEPNodeAssignment::All);
}

// Test 16-bit QDQ Slice with more than 1 axis.
TEST_F(QnnHTPBackendTests, SliceU16_MultAxes) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  RunSliceQDQTest<uint16_t>(TestInputDef<float>({2, 4}, false, input_data),
                            TestInputDef<int64_t>({2}, true, {1, 0}),  // starts
                            TestInputDef<int64_t>({2}, true, {2, 3}),  // ends
                            TestInputDef<int64_t>({2}, true, {0, 1}),  // axes
                            TestInputDef<int64_t>({2}, true, {1, 2}),  // steps
                            ExpectedEPNodeAssignment::All,
                            true);  // Use com.microsoft Q/DQ ops for 16-bit
}

// Test 8-bit QDQ Slice with more than 1 axis and an end value that exceeds the associated dimension size.
TEST_F(QnnHTPBackendTests, SliceU8_MultAxes_LargeEnd) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  RunSliceQDQTest<uint8_t>(TestInputDef<float>({2, 4}, false, input_data),
                           TestInputDef<int64_t>({2}, true, {0, 1}),      // starts
                           TestInputDef<int64_t>({2}, true, {-1, 1000}),  // ends
                           TestInputDef<int64_t>({2}, true, {0, 1}),      // axes
                           TestInputDef<int64_t>({2}, true, {1, 1}),      // steps
                           ExpectedEPNodeAssignment::All);
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif