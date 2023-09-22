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

// Creates a graph with a single GatherND operator. Used for testing CPU backend.
template <typename DataType = float>
static GetTestModelFn BuildGatherNDTestCase(const std::vector<int64_t>& data_shape,
                                            const std::vector<DataType>& data,
                                            const std::vector<int64_t>& indices_shape,
                                            const std::vector<int64_t>& indices,
                                            bool indices_are_static,
                                            int64_t batch_dims) {
  return [data_shape, data, indices_shape, indices,
          indices_are_static, batch_dims](ModelTestBuilder& builder) {
    auto* data_input = builder.MakeInput<DataType>(data_shape, data);
    auto* indices_input = (indices_are_static ? builder.MakeInitializer<int64_t>(indices_shape, indices)
                                              : builder.MakeInput<int64_t>(indices_shape, indices));
    auto* output = builder.MakeOutput();

    Node& gather_nd_node = builder.AddNode("GatherND", {data_input, indices_input}, {output});
    gather_nd_node.AddAttribute("batch_dims", batch_dims);
  };
}

// Creates a graph with a single Q/DQ GatherND operator. Used for testing HTP backend.
template <typename DataType = float, typename DataQType = uint8_t>
static GetTestModelFn BuildQDQGatherNDTestCase(const std::vector<int64_t>& data_shape,
                                               const std::vector<DataType>& data,
                                               const std::vector<int64_t>& indices_shape,
                                               const std::vector<int64_t>& indices,
                                               bool indices_are_static,
                                               int64_t batch_dims) {
  return [data_shape, data, indices_shape, indices,
          indices_are_static, batch_dims](ModelTestBuilder& builder) {
    constexpr float qdq_scale = 0.0038f;
    const DataQType zero_point = (std::numeric_limits<DataQType>::max() - std::numeric_limits<DataQType>::min()) / 2;

    auto* data_input = builder.MakeInput<DataType>(data_shape, data);
    auto* indices_input = (indices_are_static ? builder.MakeInitializer<int64_t>(indices_shape, indices)
                                              : builder.MakeInput<int64_t>(indices_shape, indices));
    auto* output = builder.MakeOutput();

    // data_input -> Q -> DQ -> GatherND
    auto* qdq_output = AddQDQNodePair<DataQType>(builder, data_input, qdq_scale, zero_point);
    auto* gather_output = builder.MakeIntermediate();

    Node& gather_nd_node = builder.AddNode("GatherND", {qdq_output, indices_input}, {gather_output});
    gather_nd_node.AddAttribute("batch_dims", batch_dims);

    // -> Q -> DQ -> output
    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<DataQType>(gather_output, qdq_scale, zero_point, q_output);
    builder.AddDequantizeLinearNode<DataQType>(q_output, qdq_scale, zero_point, output);
  };
}

// Runs an GatherND model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
template <typename DataType = float>
static void RunCPUGatherNDOpTest(const std::vector<int64_t>& data_shape,
                                 const std::vector<DataType>& data,
                                 const std::vector<int64_t>& indices_shape,
                                 const std::vector<int64_t>& indices,
                                 bool indices_are_static,
                                 int64_t batch_dims,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 13) {
  ProviderOptions provider_options;
  float fp32_abs_err = 1e-5f;  // default tolerance

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  RunQnnModelTest(BuildGatherNDTestCase<DataType>(data_shape, data, indices_shape, indices,
                                                  indices_are_static, batch_dims),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

// Runs an GatherND model on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
template <typename DataType = float, typename DataQType = uint8_t>
static void RunHTPGatherNDOpTest(const std::vector<int64_t>& data_shape,
                                 const std::vector<DataType>& data,
                                 const std::vector<int64_t>& indices_shape,
                                 const std::vector<int64_t>& indices,
                                 bool indices_are_static,
                                 int64_t batch_dims,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 13) {
  ProviderOptions provider_options;
  float fp32_abs_err = 1e-5f;  // default tolerance

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  RunQnnModelTest(BuildQDQGatherNDTestCase<DataType, DataQType>(data_shape, data, indices_shape, indices,
                                                                indices_are_static, batch_dims),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  fp32_abs_err);
}

//
// CPU tests:
//

// Test GatherND op on CPU backend:
// positive, dynamic indices.
// QNN EP should support by adding a Cast operator (to int32) after the indices input.
TEST_F(QnnCPUBackendTests, GatherND_f32_DynamicIndices_BatchDim0) {
  RunCPUGatherNDOpTest<float>({2, 2, 2},                                        // data_shape
                              {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f},  // data
                              {2, 1, 2},                                        // indices_shape
                              {0, 1, 1, 0},                                     // indices
                              false,                                            // indices_are_static
                              0,                                                // batch_dims
                              ExpectedEPNodeAssignment::All);
}

// Test GatherND op on CPU backend:
// positive, static indices.
// QNN EP should support by converting static weights to int32_t.
TEST_F(QnnCPUBackendTests, GatherND_f32_StaticIndices_BatchDim0) {
  RunCPUGatherNDOpTest<float>({2, 2, 2},                                        // data_shape
                              {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f},  // data
                              {2, 1, 2},                                        // indices_shape
                              {0, 1, 1, 0},                                     // indices
                              true,                                             // indices_are_static
                              0,                                                // batch_dims
                              ExpectedEPNodeAssignment::All);
}

// Test GatherND op on CPU backend:
// - positive, dynamic indices.
// - batch_dims = 1
// QNN EP should support by adding a Cast operator (to int32) after the indices input.
//
// TODO: Enable when QNN fixes GatherNd with batch_dims != 0
// QNN graph fails to finalized.
TEST_F(QnnCPUBackendTests, DISABLED_GatherND_f32_DynamicIndices_BatchDim1) {
  RunCPUGatherNDOpTest<float>({2, 2, 2},                                        // data_shape
                              {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f},  // data
                              {2, 1},                                           // indices_shape
                              {1, 0},                                           // indices
                              false,                                            // indices_are_static
                              1,                                                // batch_dims
                              ExpectedEPNodeAssignment::All);
}

// Test GatherND op on CPU backend:
// - positive, static indices.
// - batch_dims = 1
// QNN EP should support by converting static weights to int32_t.
//
// TODO: Enable when QNN fixes GatherNd with batch_dims != 0
// QNN graph fails to finalized.
TEST_F(QnnCPUBackendTests, DISABLED_GatherND_f32_StaticIndices_BatchDim1) {
  RunCPUGatherNDOpTest<float>({2, 2, 2},                                        // data_shape
                              {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f},  // data
                              {2, 1},                                           // indices_shape
                              {1, 0},                                           // indices
                              true,                                             // indices_are_static
                              1,                                                // batch_dims
                              ExpectedEPNodeAssignment::All);
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Test GatherND op on CPU backend:
// positive, dynamic indices.
// QNN EP's HTP backend does not support int64 data types.
// Thefore, HTP does not support Dynamic int64_t indices at all.
TEST_F(QnnHTPBackendTests, GatherND_f32_DynamicIndices_BatchDim0) {
  RunHTPGatherNDOpTest<float>({2, 2, 2},                                        // data_shape
                              {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f},  // data
                              {2, 1, 2},                                        // indices_shape
                              {0, 1, 1, 0},                                     // indices
                              false,                                            // indices_are_static
                              0,                                                // batch_dims
                              ExpectedEPNodeAssignment::Some);                  // QDQ GatherND not assigned to QNN EP.
}

// Test GatherND op on HTP backend:
// positive, static indices.
// HTP does not support int64, but QNN EP converts static int64 indices into int32.
TEST_F(QnnHTPBackendTests, GatherND_u8_StaticIndices_BatchDim0) {
  RunHTPGatherNDOpTest<float, uint8_t>({2, 2, 2},                                        // data_shape
                                       {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f},  // data
                                       {2, 1, 2},                                        // indices_shape
                                       {0, 1, 1, 0},                                     // indices
                                       true,                                             // indices_are_static
                                       0,                                                // batch_dims
                                       ExpectedEPNodeAssignment::All);
}

// Test GatherND op on HTP backend:
// - positive, static indices.
// - batch_dims = 1
// QNN EP should support by converting static weights to int32_t.
//
// TODO: Enable when QNN fixes GatherNd with batch_dims != 0
// Expected value: [[0.2, 0.3],[0.4, 0.5]], Actual (incorrect) output: [[0.2, 0.3], [0.0, 0.1]]
TEST_F(QnnHTPBackendTests, DISABLED_GatherND_f32_StaticIndices_BatchDim1) {
  RunHTPGatherNDOpTest<float>({2, 2, 2},                                        // data_shape
                              {0.f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f},  // data
                              {2, 1},                                           // indices_shape
                              {1, 0},                                           // indices
                              true,                                             // indices_are_static
                              1,                                                // batch_dims
                              ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
