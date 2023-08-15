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
static GetTestModelFn GetResizeModelBuilder(const TestInputDef<float>& input_def,
                                            const std::vector<int64_t>& sizes_data,
                                            const std::string& mode = "nearest",
                                            const std::string& coordinate_transformation_mode = "half_pixel",
                                            const std::string& nearest_mode = "round_prefer_floor") {
  return [input_def, sizes_data, mode, coordinate_transformation_mode, nearest_mode](ModelTestBuilder& builder) {
    NodeArg* input = MakeTestInput(builder, input_def);
    NodeArg* roi = builder.MakeInitializer<float>({0}, {});
    NodeArg* scales = builder.MakeInitializer<float>({0}, {});
    NodeArg* sizes = builder.Make1DInitializer<int64_t>(sizes_data);

    NodeArg* output = builder.MakeOutput();
    Node& resize_node = builder.AddNode("Resize", {input, roi, scales, sizes}, {output});
    resize_node.AddAttribute("mode", mode);
    resize_node.AddAttribute("coordinate_transformation_mode", coordinate_transformation_mode);

    if (mode == "nearest") {
      resize_node.AddAttribute("nearest_mode", nearest_mode);
    }
  };
}

static GetTestModelFn GetResizeModelBuilderWithScales(const TestInputDef<float>& input_def,
                                                      const std::vector<float>& scales_data,
                                                      const std::string& mode = "nearest",
                                                      const std::string& coordinate_transformation_mode = "half_pixel",
                                                      const std::string& nearest_mode = "round_prefer_floor") {
  return [input_def, scales_data, mode, coordinate_transformation_mode, nearest_mode](ModelTestBuilder& builder) {
    NodeArg* input = MakeTestInput(builder, input_def);
    NodeArg* roi = builder.MakeInitializer<float>({0}, {});
    NodeArg* scales = builder.Make1DInitializer<float>(scales_data);

    NodeArg* output = builder.MakeOutput();
    Node& resize_node = builder.AddNode("Resize", {input, roi, scales}, {output});
    resize_node.AddAttribute("mode", mode);
    resize_node.AddAttribute("coordinate_transformation_mode", coordinate_transformation_mode);

    if (mode == "nearest") {
      resize_node.AddAttribute("nearest_mode", nearest_mode);
    }
  };
}

template <typename QuantType = uint8_t>
static GetTestQDQModelFn<QuantType> GetQDQResizeModelBuilder(const TestInputDef<float>& input_def,
                                                             const std::vector<int64_t>& sizes_data,
                                                             const std::string& mode = "nearest",
                                                             const std::string& coordinate_transformation_mode = "half_pixel",
                                                             const std::string& nearest_mode = "round_prefer_floor") {
  return [input_def, sizes_data, mode,
          coordinate_transformation_mode, nearest_mode](ModelTestBuilder& builder,
                                                        std::vector<QuantParams<QuantType>>& output_qparams) {
    // input -> Q -> DQ ->
    NodeArg* input = MakeTestInput(builder, input_def);
    QuantParams<QuantType> input_qparams = GetTestInputQuantParams<QuantType>(input_def);
    NodeArg* input_qdq = AddQDQNodePair<QuantType>(builder, input, input_qparams.scale, input_qparams.zero_point);

    NodeArg* roi = builder.MakeInitializer<float>({0}, {});
    NodeArg* scales = builder.MakeInitializer<float>({0}, {});
    NodeArg* sizes = builder.Make1DInitializer<int64_t>(sizes_data);

    NodeArg* resize_output = builder.MakeIntermediate();
    Node& resize_node = builder.AddNode("Resize", {input_qdq, roi, scales, sizes}, {resize_output});
    resize_node.AddAttribute("mode", mode);
    resize_node.AddAttribute("coordinate_transformation_mode", coordinate_transformation_mode);

    if (mode == "nearest") {
      resize_node.AddAttribute("nearest_mode", nearest_mode);
    }

    // Resize requires the output quantization parameters to match the input.
    output_qparams[0] = input_qparams;
    AddQDQNodePairWithOutputAsGraphOutput<QuantType>(builder, resize_output, output_qparams[0].scale,
                                                     output_qparams[0].zero_point);
  };
}

/**
 * Runs a Resize model on the QNN CPU backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param input_def The input definition (shape, data, etc).
 * \param sizes_data The sizes input which determines the output shape.
 * \param mode The resize mode (e.g., nearest, linear).
 * \param coordinate_transformation_mode The coordinate transformation mode (e.g., half_pixel, pytorch_half_pixel).
 * \param nearest_mode The rounding for "nearest" mode (e.g., round_prefer_floor, floor).
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 * \param opset The opset version to use.
 */
static void RunCPUResizeOpTest(const TestInputDef<float>& input_def, const std::vector<int64_t>& sizes_data,
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

  RunQnnModelTest(GetResizeModelBuilder(input_def, sizes_data, mode, coordinate_transformation_mode, nearest_mode),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

static void RunCPUResizeOpTestWithScales(const TestInputDef<float>& input_def, const std::vector<float>& scales_data,
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

  RunQnnModelTest(GetResizeModelBuilderWithScales(input_def, scales_data, mode, coordinate_transformation_mode, nearest_mode),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

template <typename QuantType>
static void RunQDQResizeOpTest(const TestInputDef<float>& input_def,
                               const std::vector<int64_t>& sizes_data,
                               const std::string& mode, const std::string& coordinate_transformation_mode,
                               const std::string& nearest_mode,
                               ExpectedEPNodeAssignment expected_ep_assignment) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  TestQDQModelAccuracy(GetResizeModelBuilder(input_def, sizes_data, mode, coordinate_transformation_mode, nearest_mode),
                       GetQDQResizeModelBuilder<QuantType>(input_def, sizes_data, mode, coordinate_transformation_mode,
                                                           nearest_mode),
                       provider_options,
                       18,  // opset
                       expected_ep_assignment,
                       1e-5f);
}

//
// CPU tests:
//

// TODO: Our QNN CPU translation of ONNX Resize with "nearest" mode uses QNN's ResizeNearestNeighbor
// operator, which does not have a way to specify rounding (i.e., "nearest_mode" in ONNX). It is not clear
// what kind of rounding QNN's ResizeNearestNeighbor uses. Therefore, we do not yet know how to compare
// ONNX Resize to QNN ResizeNearestNeighbor. These tests should remain disabled until this behavior is
// clarified. If, for example, it turns out that ResizeNearestNeighbor uses "floor" rounding, then we should
// only compare against ONNX resize with "floor" rounding.

// Upsample that uses "round_prefer_floor" as the "nearest_mode".
// coordinate_transformation_mode: "half_pixel"
TEST_F(QnnCPUBackendTests, DISABLED_ResizeUpsampleNearestHalfPixel_rpf) {
  RunCPUResizeOpTest(TestInputDef<float>({1, 2, 7, 5}, false, -10.0f, 10.0f),  // Random input w/ range [-10, 10]
                     {1, 2, 21, 10},                                           // Sizes
                     "nearest",
                     "half_pixel",
                     "round_prefer_floor",
                     ExpectedEPNodeAssignment::All);
}

// Upsample that uses "round_prefer_ceil" as the "nearest_mode".
// coordinate_transformation_mode: "half_pixel"
TEST_F(QnnCPUBackendTests, DISABLED_ResizeUpsampleNearestHalfPixel_rpc) {
  RunCPUResizeOpTest(TestInputDef<float>({1, 1, 2, 4}, false, -10.0f, 10.0f),
                     {1, 1, 7, 5}, "nearest", "half_pixel", "round_prefer_ceil",
                     ExpectedEPNodeAssignment::All);
}

// Downsample that uses "round_prefer_ceil" as the "nearest_mode".
// coordinate_transformation_mode: "half_pixel"
TEST_F(QnnCPUBackendTests, DISABLED_ResizeDownsampleNearestHalfPixel_rpc) {
  RunCPUResizeOpTest(TestInputDef<float>({1, 1, 2, 4}, false, -10.0f, 10.0f),
                     {1, 1, 1, 3}, "nearest", "half_pixel", "round_prefer_ceil",
                     ExpectedEPNodeAssignment::All);
}

// Downsample that uses "round_prefer_floor" as the "nearest_mode".
// coordinate_transformation_mode: "half_pixel"
TEST_F(QnnCPUBackendTests, DISABLED_ResizeDownsampleNearestHalfPixel_rpf) {
  RunCPUResizeOpTest(TestInputDef<float>({1, 1, 2, 4}, false, -10.0f, 10.0f),
                     {1, 1, 1, 2}, "nearest", "half_pixel", "round_prefer_ceil",
                     ExpectedEPNodeAssignment::All);
}

// Upsample that uses "round_prefer_floor" as the "nearest_mode".
// coordinate_transformation_mode: "align_corners"
// QNN v2.13: index #50 don't match, which is 4.67152 from -1.93515
TEST_F(QnnCPUBackendTests, DISABLED_ResizeUpsampleNearestAlignCorners_rpf) {
  RunCPUResizeOpTest(TestInputDef<float>({1, 2, 7, 5}, false, -10.0f, 10.0f),
                     {1, 2, 21, 10}, "nearest", "align_corners", "round_prefer_floor",
                     ExpectedEPNodeAssignment::All);
}

// Upsample that uses "round_prefer_ceil" as the "nearest_mode".
// coordinate_transformation_mode: "align_corners"
TEST_F(QnnCPUBackendTests, DISABLED_ResizeUpsampleNearestAlignCorners_rpc) {
  RunCPUResizeOpTest(TestInputDef<float>({1, 1, 2, 4}, false, -10.0f, 10.0f),
                     {1, 1, 7, 5}, "nearest", "align_corners", "round_prefer_ceil",
                     ExpectedEPNodeAssignment::All);
}

// Downsample that uses "round_prefer_ceil" as the "nearest_mode".
// coordinate_transformation_mode: "align_corners"
TEST_F(QnnCPUBackendTests, DISABLED_ResizeDownsampleNearestAlignCorners_rpc) {
  RunCPUResizeOpTest(TestInputDef<float>({1, 1, 2, 4}, false, -10.0f, 10.0f),
                     {1, 1, 1, 3}, "nearest", "align_corners", "round_prefer_ceil",
                     ExpectedEPNodeAssignment::All);
}

// Downsample that uses "round_prefer_floor" as the "nearest_mode".
// coordinate_transformation_mode: "align_corners"
TEST_F(QnnCPUBackendTests, DISABLED_ResizeDownsampleNearestAlignCorners_rpf) {
  RunCPUResizeOpTest(TestInputDef<float>({1, 1, 2, 4}, false, -10.0f, 10.0f),
                     {1, 1, 1, 2}, "nearest", "align_corners", "round_prefer_floor",
                     ExpectedEPNodeAssignment::All);
}

//
// Cpu tests that use the "linear" mode.
//

TEST_F(QnnCPUBackendTests, Resize2xLinearHalfPixel) {
  RunCPUResizeOpTest(TestInputDef<float>({1, 3, 4, 5}, false, -10.0f, 10.0f),
                     {1, 3, 8, 10}, "linear", "half_pixel", "",
                     ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, Resize2xLinearHalfPixel_scales) {
  RunCPUResizeOpTestWithScales(TestInputDef<float>({1, 3, 4, 5}, false, -10.0f, 10.0f),
                               {1.0f, 1.0f, 2.0f, 2.0f}, "linear", "half_pixel", "",
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, Resize2xLinearAlignCorners) {
  RunCPUResizeOpTest(TestInputDef<float>({1, 3, 4, 5}, false, -10.0f, 10.0f),
                     {1, 3, 8, 10}, "linear", "align_corners", "",
                     ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, Resize2xLinearAlignCorners_scales) {
  RunCPUResizeOpTestWithScales(TestInputDef<float>({1, 3, 4, 5}, false, -10.0f, 10.0f),
                               {1.0f, 1.0f, 2.0f, 2.0f}, "linear", "align_corners", "",
                               ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

TEST_F(QnnHTPBackendTests, ResizeU8_2xLinearPytorchHalfPixel) {
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                              {1, 3, 8, 8}, "linear", "pytorch_half_pixel", "",
                              ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, ResizeU8_2xNearestHalfPixelRoundPreferFloor) {
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                              {1, 3, 8, 8}, "nearest", "half_pixel", "round_prefer_floor",
                              ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, ResizeU8_2xNearestAsymmetricFloor) {
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                              {1, 3, 8, 8}, "nearest", "asymmetric", "floor",
                              ExpectedEPNodeAssignment::All);
}

// TODO: Investigate with Qualcomm. The qnn-onnx-converter tool translates ONNX Resize [nearest, asymmetric, ceil] to
// QNN ResizeNearestNeighbor {align_corners: 0, half_pixel: 0}, which is NOT equivalent. It would be better to use
// QNN's own Resize operator (instead of ResizeNearestNeighbor), but it doesn't support the "asymmetric" coordinate
// transform mode.
//
// QNN v2.13: Inaccuracy detected for output 'output', element 189.
// Output quant params: scale=0.078431375324726105, zero_point=127.
// Expected val: -2.663428783416748
// QNN QDQ val: 7.4509806632995605 (err 10.114409446716309)
// CPU QDQ val: -2.6666667461395264 (err 0.0032379627227783203)
TEST_F(QnnHTPBackendTests, DISABLED_ResizeU8_2xNearestAsymmetricCeil) {
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                              {1, 3, 8, 8}, "nearest", "asymmetric", "ceil",
                              ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, ResizeU8_3xNearestAsymmetricFloor) {
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                              {1, 3, 12, 12}, "nearest", "asymmetric", "floor",
                              ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, ResizeU8_HalfNearestAsymmetricFloor) {
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, -10.0f, 10.0f),
                              {1, 3, 2, 2}, "nearest", "asymmetric", "floor",
                              ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
