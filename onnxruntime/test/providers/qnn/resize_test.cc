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
 * Creates a graph with a single Resize operator.
 *
 * \param shape The shape of the input and output. Input data is randomly generated with this shape.
 * \param sizes_data The sizes input which determines the output shape.
 * \param mode The resize mode (e.g., nearest, linear).
 * \param coordinate_transformation_mode The coordinate transformation mode (e.g., half_pixel, pytorch_half_pixel).
 * \param nearest_mode The rounding for "nearest" mode (e.g., round_prefer_floor, floor).
 *
 * \return A function that builds the graph with the provided builder.
 */
static GetTestModelFn BuildResizeTestCase(const std::vector<int64_t>& shape,
                                          const std::vector<int64_t>& sizes_data,
                                          const std::string& mode = "nearest",
                                          const std::string& coordinate_transformation_mode = "half_pixel",
                                          const std::string& nearest_mode = "round_prefer_floor") {
  return [shape, sizes_data, mode, coordinate_transformation_mode, nearest_mode](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>(shape, 0.0f, 20.0f);
    auto* roi = builder.MakeInitializer<float>({0}, {});
    auto* scales = builder.MakeInitializer<float>({0}, {});
    auto* sizes = builder.Make1DInitializer<int64_t>(sizes_data);

    auto* output = builder.MakeOutput();
    Node& resize_node = builder.AddNode("Resize", {input, roi, scales, sizes}, {output});
    resize_node.AddAttribute("mode", mode);
    resize_node.AddAttribute("coordinate_transformation_mode", coordinate_transformation_mode);

    if (mode == "nearest") {
      resize_node.AddAttribute("nearest_mode", nearest_mode);
    }
  };
}

/**
 * Runs a Resize model on the QNN CPU backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param shape The shape of the input and output. Input data is randomly generated with this shape.
 * \param sizes_data The sizes input which determines the output shape.
 * \param mode The resize mode (e.g., nearest, linear).
 * \param coordinate_transformation_mode The coordinate transformation mode (e.g., half_pixel, pytorch_half_pixel).
 * \param nearest_mode The rounding for "nearest" mode (e.g., round_prefer_floor, floor).
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 * \param test_description Description of the test for error reporting.
 * \param opset The opset version to use.
 */
static void RunCPUResizeOpTest(const std::vector<int64_t>& shape, const std::vector<int64_t>& sizes_data,
                            const std::string& mode, const std::string& coordinate_transformation_mode,
                            const std::string& nearest_mode,
                            ExpectedEPNodeAssignment expected_ep_assignment, const char* test_description,
                            int opset = 18) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  constexpr int expected_nodes_in_partition = 1;
  RunQnnModelTest(BuildResizeTestCase(shape, sizes_data, mode, coordinate_transformation_mode, nearest_mode),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  expected_nodes_in_partition,
                  test_description);
}

template <typename QuantType>
static void RunQDQResizeOpTest(const std::vector<int64_t>& shape, const std::vector<int64_t>& sizes_data,
                               const std::string& mode, const std::string& coordinate_transformation_mode,
                               const std::string& nearest_mode,
                               ExpectedEPNodeAssignment expected_ep_assignment, float fp32_abs_err,
                               const char* test_description) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  constexpr int expected_nodes_in_partition = 1;
  RunQnnModelTest(BuildQDQResizeTestCase<QuantType>(shape, sizes_data, mode, coordinate_transformation_mode,
                                                    nearest_mode, true),
                  provider_options,
                  18,  // opset
                  expected_ep_assignment,
                  expected_nodes_in_partition,
                  test_description,
                  fp32_abs_err);
}

//
// CPU tests:
//

TEST(QnnCPUBackendTests, DISABLED_TestResize2xNearestHalfPixel) {
  RunCPUResizeOpTest({1, 2, 2, 2}, {1, 2, 4, 4}, "nearest", "half_pixel", "round_prefer_floor",
                     ExpectedEPNodeAssignment::All, "TestResize2xNearestHalfPixel");
}

// TODO: Investigate difference in results, and then enable this test.
// onnxruntime\onnxruntime\test\util\test_utils.cc(51): error: Value of: ltensor.DataAsSpan<float>()
// Expected : contains 3 values, where the value pair(13.2128859, 13.362051) at index #0 don't match,
// which is 0.149165 from 13.2129 Google Test trace : onnxruntime\onnxruntime\test\common\tensor_op_test_utils.cc(14) :
// ORT test random seed : 2345
TEST(QnnCPUBackendTests, DISABLED_TestResizeDownSampleNearestHalfPixel) {
  RunCPUResizeOpTest({1, 1, 2, 4}, {1, 1, 1, 3}, "nearest", "half_pixel", "round_prefer_floor",
                     ExpectedEPNodeAssignment::All, "TestResizeDownSampleNearestHalfPixel");
}

TEST(QnnCPUBackendTests, DISABLED_TestResize2xNearestHalfAlignCorners) {
  RunCPUResizeOpTest({1, 2, 2, 2}, {1, 2, 4, 4}, "nearest", "align_corners", "round_prefer_floor",
                     ExpectedEPNodeAssignment::All, "TestResize2xNearestHalfAlignCorners");
}

TEST(QnnCPUBackendTests, DISABLED_TestResize2xLinearHalfPixel) {
  RunCPUResizeOpTest({1, 2, 2, 2}, {1, 2, 4, 4}, "linear", "half_pixel", "",
                     ExpectedEPNodeAssignment::All, "TestResize2xLinearHalfPixel");
}

TEST(QnnCPUBackendTests, DISABLED_TestResize2xLinearAlignCorners) {
  RunCPUResizeOpTest({1, 2, 2, 2}, {1, 2, 4, 4}, "linear", "align_corners", "",
                     ExpectedEPNodeAssignment::All, "TestResize2xLinearAlignCorners");
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//
TEST_F(QnnHTPBackendTests, TestQDQU8Resize2xLinearPytorchHalfPixel) {
  RunQDQResizeOpTest<uint8_t>({1, 3, 4, 4}, {1, 3, 8, 8}, "linear", "pytorch_half_pixel", "",
                              ExpectedEPNodeAssignment::All, 0.0031f,
                              "TestQDQU8Resize2xLinearPytorchHalfPixel");
}

TEST_F(QnnHTPBackendTests, TestQDQU8Resize2xNearestHalfPixelRoundPreferFloor) {
  RunQDQResizeOpTest<uint8_t>({1, 3, 4, 4}, {1, 3, 8, 8}, "nearest", "half_pixel", "round_prefer_floor",
                              ExpectedEPNodeAssignment::All, 1e-5f,
                              "TestQDQU8Resize2xNearestHalfPixelRoundPreferFloor");
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)