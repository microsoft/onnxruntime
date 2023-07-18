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

static GetTestModelFn BuildResizeTestCaseWithScales(const std::vector<int64_t>& shape,
                                                    const std::vector<float>& scales_data,
                                                    const std::string& mode = "nearest",
                                                    const std::string& coordinate_transformation_mode = "half_pixel",
                                                    const std::string& nearest_mode = "round_prefer_floor") {
  return [shape, scales_data, mode, coordinate_transformation_mode, nearest_mode](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>(shape, 0.0f, 20.0f);
    auto* roi = builder.MakeInitializer<float>({0}, {});
    auto* scales = builder.Make1DInitializer<float>(scales_data);

    auto* output = builder.MakeOutput();
    Node& resize_node = builder.AddNode("Resize", {input, roi, scales}, {output});
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
 * \param opset The opset version to use.
 */
static void RunCPUResizeOpTest(const std::vector<int64_t>& shape, const std::vector<int64_t>& sizes_data,
                               const std::string& mode, const std::string& coordinate_transformation_mode,
                               const std::string& nearest_mode,
                               ExpectedEPNodeAssignment expected_ep_assignment,
                               int opset = 11) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildResizeTestCase(shape, sizes_data, mode, coordinate_transformation_mode, nearest_mode),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

static void RunCPUResizeOpTestWithScales(const std::vector<int64_t>& shape, const std::vector<float>& scales_data,
                                         const std::string& mode, const std::string& coordinate_transformation_mode,
                                         const std::string& nearest_mode,
                                         ExpectedEPNodeAssignment expected_ep_assignment,
                                         int opset = 11) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildResizeTestCaseWithScales(shape, scales_data, mode, coordinate_transformation_mode, nearest_mode),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

template <typename QuantType>
static void RunQDQResizeOpTest(const std::vector<int64_t>& shape, const std::vector<int64_t>& sizes_data,
                               const std::string& mode, const std::string& coordinate_transformation_mode,
                               const std::string& nearest_mode,
                               ExpectedEPNodeAssignment expected_ep_assignment, float fp32_abs_err) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  RunQnnModelTest(BuildQDQResizeTestCase<QuantType>(shape, sizes_data, mode, coordinate_transformation_mode,
                                                    nearest_mode, true),
                  provider_options,
                  18,  // opset
                  expected_ep_assignment,
                  fp32_abs_err);
}

//
// CPU tests:
//

// TODO: Enable QnnCPU tests that use "nearest" mode.
//
// Our non-quantized implementation of Resize uses QNN's ResizeNearestNeighbor operator,
// which is __not__ equivalent to ONNX's Resize operator with a single specific "nearest_mode".
// The following disabled unit tests would pass if we removed the check in QNN EP that expects the
// "nearest_mode" to be "floor". Sometimes, ResizeNearestNeighbor is equivalent to ONNX Resize with
// "round_prefer_floor", and other times it is equivalent to ONNX Resize with "round_prefer_ceil".

// Upsample that uses "round_prefer_floor" as the "nearest_mode".
// coordinate_transformation_mode: "half_pixel"
TEST_F(QnnCPUBackendTests, DISABLED_TestResizeUpsampleNearestHalfPixel_rpf) {
  RunCPUResizeOpTest({1, 2, 7, 5}, {1, 2, 21, 10}, "nearest", "half_pixel", "round_prefer_floor",
                     ExpectedEPNodeAssignment::All);
}

// Upsample that uses "round_prefer_ceil" as the "nearest_mode".
// coordinate_transformation_mode: "half_pixel"
TEST_F(QnnCPUBackendTests, DISABLED_TestResizeUpsampleNearestHalfPixel_rpc) {
  RunCPUResizeOpTest({1, 1, 2, 4}, {1, 1, 7, 5}, "nearest", "half_pixel", "round_prefer_ceil",
                     ExpectedEPNodeAssignment::All);
}

// Downsample that uses "round_prefer_ceil" as the "nearest_mode".
// coordinate_transformation_mode: "half_pixel"
TEST_F(QnnCPUBackendTests, DISABLED_TestResizeDownsampleNearestHalfPixel_rpc) {
  RunCPUResizeOpTest({1, 1, 2, 4}, {1, 1, 1, 3}, "nearest", "half_pixel", "round_prefer_ceil",
                     ExpectedEPNodeAssignment::All);
}

// Downsample that uses "round_prefer_floor" as the "nearest_mode".
// coordinate_transformation_mode: "half_pixel"
TEST_F(QnnCPUBackendTests, DISABLED_TestResizeDownsampleNearestHalfPixel_rpf) {
  RunCPUResizeOpTest({1, 1, 2, 4}, {1, 1, 1, 2}, "nearest", "half_pixel", "round_prefer_ceil",
                     ExpectedEPNodeAssignment::All);
}

// Upsample that uses "round_prefer_floor" as the "nearest_mode".
// coordinate_transformation_mode: "align_corners"
TEST_F(QnnCPUBackendTests, DISABLED_TestResizeUpsampleNearestAlignCorners_rpf) {
  RunCPUResizeOpTest({1, 2, 7, 5}, {1, 2, 21, 10}, "nearest", "align_corners", "round_prefer_floor",
                     ExpectedEPNodeAssignment::All);
}

// Upsample that uses "round_prefer_ceil" as the "nearest_mode".
// coordinate_transformation_mode: "align_corners"
TEST_F(QnnCPUBackendTests, DISABLED_TestResizeUpsampleNearestAlignCorners_rpc) {
  RunCPUResizeOpTest({1, 1, 2, 4}, {1, 1, 7, 5}, "nearest", "align_corners", "round_prefer_ceil",
                     ExpectedEPNodeAssignment::All);
}

// Downsample that uses "round_prefer_ceil" as the "nearest_mode".
// coordinate_transformation_mode: "align_corners"
TEST_F(QnnCPUBackendTests, DISABLED_TestResizeDownsampleNearestAlignCorners_rpc) {
  RunCPUResizeOpTest({1, 1, 2, 4}, {1, 1, 1, 3}, "nearest", "align_corners", "round_prefer_ceil",
                     ExpectedEPNodeAssignment::All);
}

// Downsample that uses "round_prefer_floor" as the "nearest_mode".
// coordinate_transformation_mode: "align_corners"
TEST_F(QnnCPUBackendTests, DISABLED_TestResizeDownsampleNearestAlignCorners_rpf) {
  RunCPUResizeOpTest({1, 1, 2, 4}, {1, 1, 1, 2}, "nearest", "align_corners", "round_prefer_floor",
                     ExpectedEPNodeAssignment::All);
}

//
// Cpu tests that use the "linear" mode.
//

TEST_F(QnnCPUBackendTests, TestResize2xLinearHalfPixel) {
  RunCPUResizeOpTest({1, 3, 4, 5}, {1, 3, 8, 10}, "linear", "half_pixel", "",
                     ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, TestResize2xLinearHalfPixel_scales) {
  RunCPUResizeOpTestWithScales({1, 3, 4, 5}, {1.0f, 1.0f, 2.0f, 2.0f}, "linear", "half_pixel", "",
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, TestResize2xLinearAlignCorners) {
  RunCPUResizeOpTest({1, 3, 4, 5}, {1, 3, 8, 10}, "linear", "align_corners", "",
                     ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, TestResize2xLinearAlignCorners_scales) {
  RunCPUResizeOpTestWithScales({1, 3, 4, 5}, {1.0f, 1.0f, 2.0f, 2.0f}, "linear", "align_corners", "",
                               ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

TEST_F(QnnHTPBackendTests, TestQDQU8Resize2xLinearPytorchHalfPixel) {
  RunQDQResizeOpTest<uint8_t>({1, 3, 4, 4}, {1, 3, 8, 8}, "linear", "pytorch_half_pixel", "",
                              ExpectedEPNodeAssignment::All, 0.0031f);
}

TEST_F(QnnHTPBackendTests, TestQDQU8Resize2xNearestHalfPixelRoundPreferFloor) {
  RunQDQResizeOpTest<uint8_t>({1, 3, 4, 4}, {1, 3, 8, 8}, "nearest", "half_pixel", "round_prefer_floor",
                              ExpectedEPNodeAssignment::All, 1e-5f);
}

TEST_F(QnnHTPBackendTests, TestQDQU8Resize2xNearestAsymmetricFloor) {
  RunQDQResizeOpTest<uint8_t>({1, 3, 4, 4}, {1, 3, 8, 8}, "nearest", "asymmetric", "floor",
                              ExpectedEPNodeAssignment::All, 1e-5f);
}

// TODO: Investigate with Qualcomm. The qnn-onnx-converter tool translates ONNX Resize [nearest, asymmetric, ceil] to
// QNN ResizeNearestNeighbor {align_corners: 0, half_pixel: 0}, which is NOT equivalent. It would be better to use
// QNN's own Resize operator (instead of ResizeNearestNeighbor), but it doesn't support the "asymmetric" coordinate
// transform mode.
//
// Expected: contains 192 values, where each value and its corresponding value in 16-byte object
// <C0-00 00-00 00-00 00-00 40-05 D6-27 BB-01 00-00> are an almost-equal pair
// Actual : 16 - byte object<C0 - 00 00 - 00 00 - 00 00 - 00 40 - 04 E9 - 1B BB - 01 00 - 00>,
// where the value pair(0.15, 0.501) at index #1 don't match, which is 0.351 from 0.15
TEST_F(QnnHTPBackendTests, DISABLED_TestQDQU8Resize2xNearestAsymmetricCeil) {
  RunQDQResizeOpTest<uint8_t>({1, 3, 4, 4}, {1, 3, 8, 8}, "nearest", "asymmetric", "ceil",
                              ExpectedEPNodeAssignment::All, 1e-5f);
}

TEST_F(QnnHTPBackendTests, TestQDQU8Resize3xNearestAsymmetricFloor) {
  RunQDQResizeOpTest<uint8_t>({1, 3, 4, 4}, {1, 3, 12, 12}, "nearest", "asymmetric", "floor",
                              ExpectedEPNodeAssignment::All, 1e-5f);
}

TEST_F(QnnHTPBackendTests, TestQDQU8ResizeHalfNearestAsymmetricFloor) {
  RunQDQResizeOpTest<uint8_t>({1, 3, 4, 4}, {1, 3, 2, 2}, "nearest", "asymmetric", "floor",
                              ExpectedEPNodeAssignment::All, 1e-5f);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
