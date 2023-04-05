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
static GetTestModelFn BuildCastTestCase(const std::vector<int64_t>& shape,
                                        const std::vector<int64_t>& sizes_data,
                                        const std::string& mode = "nearest",
                                        const std::string& coordinate_transformation_mode = "half_pixel",
                                        const std::string& nearest_mode = "floor") {
  return [shape, sizes_data, mode, coordinate_transformation_mode, nearest_mode](ModelTestBuilder& builder) {

    // Random input data
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
 * Runs a Cast model on the QNN CPU or HTP backend. Checks the graph node assignment, and that inference
 * outputs for QNN and CPU match.
 *
 * \param shape The shape of the input and output. Input data is randomly generated with this shape.
 * \param dst_type The destination type as an instance of the DataType enum in TensorProto.
 * \param test_description Description of the test for error reporting.
 * \param expected_ep_assignment How many nodes are expected to be assigned to QNN (All, Some, or None).
 * \param use_htp True to run on HTP backend. Otherwise, runs on CPU.
 */
static void RunResizeOpTest(const std::vector<int64_t>& shape, const std::vector<int64_t>& sizes_data,
                            const std::string& mode, const std::string& coordinate_transformation_mode,
                            const std::string& nearest_mode,
                            ExpectedEPNodeAssignment expected_ep_assignment, const char* test_description) {
  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  constexpr int expected_nodes_in_partition = 1;
  RunQnnModelTest(BuildCastTestCase(shape, sizes_data, mode, coordinate_transformation_mode, nearest_mode),
                  provider_options,
                  18,  // opset
                  expected_ep_assignment,
                  expected_nodes_in_partition,
                  test_description);
}

//
// CPU tests:
//

// Cast int32_t to float on CPU
TEST(QnnCPUBackendTests, TestResize1) {
  RunResizeOpTest({1, 2, 3, 3}, {1, 2, 3, 3}, "nearest", "half_pixel", "floor", ExpectedEPNodeAssignment::All,
                  "TestResize1");
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//


#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)