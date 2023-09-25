// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>

#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"

#include "onnx/onnx_pb.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

/**
 * Creates a graph with a single Cast operator.
 *
 * \param shape The shape of the input and output. Input data is randomly generated with this shape.
 * \param dst_type The destination type as an instance of the DataType enum in TensorProto.
 *
 * \return A function that builds the graph with the provided builder.
 */
template <typename InputType>
static GetTestModelFn BuildCastTestCase(const std::vector<int64_t>& shape,
                                        ONNX_NAMESPACE::TensorProto_DataType dst_type) {
  return [shape, dst_type](ModelTestBuilder& builder) {
    // Random input data
    auto input = builder.MakeInput<InputType>(shape, static_cast<InputType>(0), static_cast<InputType>(20));

    auto* output = builder.MakeOutput();
    Node& cast_node = builder.AddNode("Cast", {input}, {output});
    cast_node.AddAttribute("to", static_cast<int64_t>(dst_type));
  };
}

/**
 * Runs a Cast model on the QNN CPU or HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param shape The shape of the input and output. Input data is randomly generated with this shape.
 * \param dst_type The destination type as an instance of the DataType enum in TensorProto.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 * \param use_htp True to run on HTP backend. Otherwise, runs on CPU.
 */
template <typename InputType>
static void RunCastOpTest(const std::vector<int64_t>& shape, ONNX_NAMESPACE::TensorProto_DataType dst_type,
                          ExpectedEPNodeAssignment expected_ep_assignment,
                          bool use_htp) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = use_htp ? "QnnHtp.dll" : "QnnCpu.dll";
#else
  provider_options["backend_path"] = use_htp ? "libQnnHtp.so" : "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildCastTestCase<InputType>(shape, dst_type),
                  provider_options,
                  13,  // opset
                  expected_ep_assignment);
}

//
// CPU tests:
//

// Cast int32_t to float on CPU
TEST_F(QnnCPUBackendTests, TestCastInt32ToFloat) {
  RunCastOpTest<int32_t>({2, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT, ExpectedEPNodeAssignment::All,
                         false);
}

// Cast uint8_t to float on CPU
TEST_F(QnnCPUBackendTests, TestCastUInt8ToFloat) {
  RunCastOpTest<uint8_t>({2, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT, ExpectedEPNodeAssignment::All,
                         false);
}

// Cast float to int32_t on CPU
TEST_F(QnnCPUBackendTests, TestCastFloatToInt32) {
  RunCastOpTest<float>({2, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32, ExpectedEPNodeAssignment::All,
                       false);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Cast int32_t to float on HTP
TEST_F(QnnHTPBackendTests, TestCastInt32ToFloatHTP) {
  RunCastOpTest<int32_t>({3, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT, ExpectedEPNodeAssignment::All,
                         true);
}

// Cast uint8_t to float on HTP
TEST_F(QnnHTPBackendTests, TestCastUInt8ToFloatHTP) {
  RunCastOpTest<uint8_t>({3, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT, ExpectedEPNodeAssignment::All,
                         true);
}

// Cast float to int32_t on HTP
TEST_F(QnnHTPBackendTests, TestCastFloatToInt32HTP) {
  RunCastOpTest<float>({3, 3}, ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_INT32, ExpectedEPNodeAssignment::All,
                       true);
}
#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)