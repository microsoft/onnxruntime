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
                               int opset = 19) {
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
                                         int opset = 19) {
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
                               ExpectedEPNodeAssignment expected_ep_assignment,
                               int opset = 19,
                               QDQTolerance tolerance = QDQTolerance()) {
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
                       opset,
                       expected_ep_assignment,
                       tolerance);
}

//
// CPU tests:
//

// Upsample that uses "round_prefer_floor" as the "nearest_mode".
// coordinate_transformation_mode: "half_pixel"
TEST_F(QnnCPUBackendTests, ResizeUpsampleNearestHalfPixel_rpf) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 70);
  RunCPUResizeOpTest(TestInputDef<float>({1, 2, 7, 5}, false, input_data),
                     {1, 2, 21, 10},  // Sizes
                     "nearest",
                     "half_pixel",
                     "round_prefer_floor",
                     ExpectedEPNodeAssignment::All);
}

// Upsample that uses "round_prefer_ceil" as the "nearest_mode".
// coordinate_transformation_mode: "half_pixel"
TEST_F(QnnCPUBackendTests, ResizeUpsampleNearestHalfPixel_rpc) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  RunCPUResizeOpTest(TestInputDef<float>({1, 1, 2, 4}, false, input_data),
                     {1, 1, 7, 5}, "nearest", "half_pixel", "round_prefer_ceil",
                     ExpectedEPNodeAssignment::All);
}

// Downsample that uses "round_prefer_ceil" as the "nearest_mode".
// coordinate_transformation_mode: "half_pixel"
TEST_F(QnnCPUBackendTests, ResizeDownsampleNearestHalfPixel_rpc) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  RunCPUResizeOpTest(TestInputDef<float>({1, 1, 2, 4}, false, input_data),
                     {1, 1, 1, 3}, "nearest", "half_pixel", "round_prefer_ceil",
                     ExpectedEPNodeAssignment::All);
}

// Downsample that uses "round_prefer_floor" as the "nearest_mode".
// coordinate_transformation_mode: "half_pixel"
TEST_F(QnnCPUBackendTests, ResizeDownsampleNearestHalfPixel_rpf) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  RunCPUResizeOpTest(TestInputDef<float>({1, 1, 2, 4}, false, input_data),
                     {1, 1, 1, 2}, "nearest", "half_pixel", "round_prefer_ceil",
                     ExpectedEPNodeAssignment::All);
}

// Upsample that uses "round_prefer_floor" as the "nearest_mode".
// coordinate_transformation_mode: "align_corners"
TEST_F(QnnCPUBackendTests, ResizeUpsampleNearestAlignCorners_rpf) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 70);
  RunCPUResizeOpTest(TestInputDef<float>({1, 2, 7, 5}, false, input_data),
                     {1, 2, 21, 10}, "nearest", "align_corners", "round_prefer_floor",
                     ExpectedEPNodeAssignment::All);
}

// Upsample that uses "round_prefer_floor" as the "nearest_mode".
// coordinate_transformation_mode: "asymmetric"
TEST_F(QnnCPUBackendTests, ResizeUpsampleNearestAsymmetric_rpf) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 70);
  RunCPUResizeOpTest(TestInputDef<float>({1, 2, 7, 5}, false, input_data),
                     {1, 2, 21, 10}, "nearest", "asymmetric", "round_prefer_floor",
                     ExpectedEPNodeAssignment::All);
}

// Upsample that uses "round_prefer_ceil" as the "nearest_mode".
// coordinate_transformation_mode: "align_corners"
TEST_F(QnnCPUBackendTests, ResizeUpsampleNearestAlignCorners_rpc) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  RunCPUResizeOpTest(TestInputDef<float>({1, 1, 2, 4}, false, input_data),
                     {1, 1, 7, 5}, "nearest", "align_corners", "round_prefer_ceil",
                     ExpectedEPNodeAssignment::All);
}

// Downsample that uses "round_prefer_ceil" as the "nearest_mode".
// coordinate_transformation_mode: "align_corners"
TEST_F(QnnCPUBackendTests, ResizeDownsampleNearestAlignCorners_rpc) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  RunCPUResizeOpTest(TestInputDef<float>({1, 1, 2, 4}, false, input_data),
                     {1, 1, 1, 3}, "nearest", "align_corners", "round_prefer_ceil",
                     ExpectedEPNodeAssignment::All);
}

// Downsample that uses "round_prefer_floor" as the "nearest_mode".
// coordinate_transformation_mode: "align_corners"
TEST_F(QnnCPUBackendTests, ResizeDownsampleNearestAlignCorners_rpf) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  RunCPUResizeOpTest(TestInputDef<float>({1, 1, 2, 4}, false, input_data),
                     {1, 1, 1, 2}, "nearest", "align_corners", "round_prefer_floor",
                     ExpectedEPNodeAssignment::All);
}

//
// Cpu tests that use the "linear" mode.
//

TEST_F(QnnCPUBackendTests, Resize2xLinearHalfPixel) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 60);
  RunCPUResizeOpTest(TestInputDef<float>({1, 3, 4, 5}, false, input_data),
                     {1, 3, 8, 10}, "linear", "half_pixel", "",
                     ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, Resize2xLinearHalfPixel_scales) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 60);
  RunCPUResizeOpTestWithScales(TestInputDef<float>({1, 3, 4, 5}, false, input_data),
                               {1.0f, 1.0f, 2.0f, 2.0f}, "linear", "half_pixel", "",
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, Resize2xLinearAlignCorners) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 60);
  RunCPUResizeOpTest(TestInputDef<float>({1, 3, 4, 5}, false, input_data),
                     {1, 3, 8, 10}, "linear", "align_corners", "",
                     ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, Resize2xLinearAlignCorners_scales) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 60);
  RunCPUResizeOpTestWithScales(TestInputDef<float>({1, 3, 4, 5}, false, input_data),
                               {1.0f, 1.0f, 2.0f, 2.0f}, "linear", "align_corners", "",
                               ExpectedEPNodeAssignment::All);
}

// Test Resize downsample with mode: "linear", coordinate_transformation_mode: "align_corners"
TEST_F(QnnCPUBackendTests, Resize_DownSample_Linear_AlignCorners_scales) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  RunCPUResizeOpTestWithScales(TestInputDef<float>({1, 1, 2, 4}, false, input_data),
                               {1.0f, 1.0f, 0.6f, 0.6f}, "linear", "align_corners", "",
                               ExpectedEPNodeAssignment::All);
}

// Test Resize downsample with mode: "linear", coordinate_transformation_mode: "half_pixel"
TEST_F(QnnCPUBackendTests, Resize_DownSample_Linear_HalfPixel_scales) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  RunCPUResizeOpTestWithScales(TestInputDef<float>({1, 1, 2, 4}, false, input_data),
                               {1.0f, 1.0f, 0.6f, 0.6f}, "linear", "half_pixel", "",
                               ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Test QDQ Resize downsample with mode: "linear", coordinate_transformation_mode: "align_corners"
TEST_F(QnnHTPBackendTests, Resize_DownSample_Linear_AlignCorners) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 1, 2, 4}, false, input_data),
                              {1, 1, 1, 2}, "linear", "align_corners", "",
                              ExpectedEPNodeAssignment::All);
}

// Test QDQ Resize downsample with mode: "linear", coordinate_transformation_mode: "half_pixel"
TEST_F(QnnHTPBackendTests, Resize_DownSample_Linear_HalfPixel) {
  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 1, 2, 4}, false, input_data),
                              {1, 1, 1, 2}, "linear", "half_pixel", "",
                              ExpectedEPNodeAssignment::All,
                              19,
                              // Need tolerance of 0.539% of output range after QNN SDK 2.17
                              QDQTolerance(0.00539f));
}

// Test 2x QDQ Resize mode: "linear", coordinate_transformation_mode: "pytorch_half_pixel"
// QNN EP uses QNN's Resize op.
TEST_F(QnnHTPBackendTests, ResizeU8_2xLinearPytorchHalfPixel) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                              {1, 3, 8, 8}, "linear", "pytorch_half_pixel", "",
                              ExpectedEPNodeAssignment::All,
                              19,
                              // Need tolerance of 0.609% of output range after QNN SDK 2.17
                              QDQTolerance(0.00609f));
}

// Test 2x QDQ Resize mode: "linear", coordinate_transformation_mode: "half_pixel"
// QNN EP uses QNN's Resize op.
TEST_F(QnnHTPBackendTests, ResizeU8_2xLinearHalfPixel) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                              {1, 3, 8, 8}, "linear", "half_pixel", "",
                              ExpectedEPNodeAssignment::All,
                              19,
                              // Need tolerance of 0.609% of output range after QNN SDK 2.17
                              QDQTolerance(0.00609f));
}

// Test 2x QDQ Resize mode: "linear", coordinate_transformation_mode: "align_corners"
// QNN EP uses QNN's Resize op.
TEST_F(QnnHTPBackendTests, ResizeU8_2xLinearAlignCorners) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                              {1, 3, 8, 8}, "linear", "align_corners", "",
                              ExpectedEPNodeAssignment::All,
                              19,
                              // Need tolerance of 0.533% of output range after QNN SDK 2.17
                              QDQTolerance(0.00533f));
}

// Test 2x QDQ Resize mode: "linear", coordinate_transformation_mode: "asymmetric"
// QNN EP uses QNN's Resize op.
TEST_F(QnnHTPBackendTests, ResizeU8_2xLinearAsymmetric) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                              {1, 3, 8, 8}, "linear", "asymmetric", "",
                              ExpectedEPNodeAssignment::All,
                              19,
                              // Need tolerance of 0.619% of output range after QNN SDK 2.17
                              QDQTolerance(0.00619f));
}

// Test 2x QDQ Resize mode: "nearest", coordinate_transformation_mode: "half_pixel", nearest_mode: "round_prefer_floor"
// QNN EP uses QNN's Resize op.
TEST_F(QnnHTPBackendTests, ResizeU8_2xNearestHalfPixelRoundPreferFloor) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                              {1, 3, 8, 8}, "nearest", "half_pixel", "round_prefer_floor",
                              ExpectedEPNodeAssignment::All);
}

// Test that the nearest_mode "ceil" is not supported on the HTP backend.
TEST_F(QnnHTPBackendTests, ResizeU8_NearestModeCeil_Unsupported) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                              {1, 3, 8, 8}, "nearest", "asymmetric", "ceil",
                              ExpectedEPNodeAssignment::None);
}

// Test 3x QDQ Resize mode: "nearest", coordinate_transformation_mode: "asymmetric", nearest_mode: "floor".
// QNN EP uses QNN's ResizeNearestNeighbor op.
TEST_F(QnnHTPBackendTests, ResizeU8_3xNearestAsymmetricFloor) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                              {1, 3, 12, 12}, "nearest", "asymmetric", "floor",
                              ExpectedEPNodeAssignment::All);
}

// Test 2x QDQ Resize mode: "nearest", coordinate_transformation_mode: "asymmetric", nearest_mode: "round_prefer_floor"
// QNN EP uses QNN's Resize op.
TEST_F(QnnHTPBackendTests, ResizeU8_2xNearestAsymmetricRoundPreferFloor) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 8);
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 2, 2, 2}, false, input_data),
                              {1, 2, 4, 4}, "nearest", "asymmetric", "round_prefer_floor",
                              ExpectedEPNodeAssignment::All);
}

// Test 3x QDQ Resize mode: "nearest", coordinate_transformation_mode: "asymmetric", nearest_mode: "round_prefer_floor"
// QNN EP uses QNN's Resize op.
//
// TODO: Inaccuracy detected for output 'output_0', element 2.
// Output quant params: scale=0.078431375324726105, zero_point=127.
// Expected val: -3.3333334922790527
// QNN QDQ val: -9.960784912109375 (err 6.6274514198303223)
// CPU QDQ val: -3.2941176891326904 (err 0.039215803146362305)
//
// More debugging info:
// Input elements f32[1,1,2,2] = -10.0000000 -3.33333349 3.33333302 10.0000000
// ORT CPU EP (f32 model) outputs: -10.0000000 -10.0000000 -3.33333349 -3.33333349 -3.33333349 -3.33333349 -10.00 ...
// ORT CPU EP (qdq model) outputs: -9.96078491 -9.96078491 -3.29411769 -3.29411769 -3.29411769 -3.29411769 -9.961 ...
// ORT QNN EP (qdq model) outputs: -9.96078491 -9.96078491 -9.96078491 -3.37254906 -3.37254906 -3.37254906 -9.961 ...
TEST_F(QnnHTPBackendTests, DISABLED_ResizeU8_3xNearestAsymmetricRoundPreferFloor) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 4);
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 1, 2, 2}, false, input_data),
                              {1, 1, 6, 6}, "nearest", "asymmetric", "round_prefer_floor",
                              ExpectedEPNodeAssignment::All);
}

// Test 0.5x QDQ Resize mode: "nearest", coordinate_transformation_mode: "asymmetric", nearest_mode: "floor"
// QNN EP uses QNN's ResizeNearestNeighbor op.
TEST_F(QnnHTPBackendTests, ResizeU8_HalfNearestAsymmetricFloor) {
  std::vector<float> input_data = GetFloatDataInRange(-10.0f, 10.0f, 48);
  RunQDQResizeOpTest<uint8_t>(TestInputDef<float>({1, 3, 4, 4}, false, input_data),
                              {1, 3, 2, 2}, "nearest", "asymmetric", "floor",
                              ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
