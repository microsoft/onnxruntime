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

// Creates a graph with a single GatherElements operator. Used for testing CPU backend.
template <typename DataType = float, typename IndexType = int32_t>
static GetTestModelFn BuildGatherElemsTestCase(const std::vector<int64_t>& data_shape,
                                               const std::vector<DataType>& data,
                                               const std::vector<int64_t>& indices_shape,
                                               const std::vector<IndexType>& indices,
                                               bool indices_are_static,
                                               int64_t axis) {
  return [data_shape, data, indices_shape, indices,
          indices_are_static, axis](ModelTestBuilder& builder) {
    auto* data_input = builder.MakeInput<DataType>(data_shape, data);
    auto* indices_input = indices_are_static ? builder.MakeInitializer<IndexType>(indices_shape, indices) : builder.MakeInput<IndexType>(indices_shape, indices);
    auto* output = builder.MakeOutput();

    Node& gather_elems_node = builder.AddNode("GatherElements", {data_input, indices_input}, {output});
    gather_elems_node.AddAttribute("axis", axis);
  };
}

// Creates a graph with a single Q/DQ GatherElements operator. Used for testing HTP backend.
template <typename DataType = float, typename DataQType = uint8_t, typename IndexType = int32_t>
static GetTestModelFn BuildQDQGatherElemsTestCase(const std::vector<int64_t>& data_shape,
                                                  const std::vector<DataType>& data,
                                                  const std::vector<int64_t>& indices_shape,
                                                  const std::vector<IndexType>& indices,
                                                  bool indices_are_static,
                                                  int64_t axis) {
  return [data_shape, data, indices_shape, indices,
          indices_are_static, axis](ModelTestBuilder& builder) {
    constexpr float qdq_scale = 0.05f;
    const DataQType zero_point = std::numeric_limits<DataQType>::max() / 2;

    auto* data_input = builder.MakeInput<DataType>(data_shape, data);
    auto* indices_input = indices_are_static ? builder.MakeInitializer<IndexType>(indices_shape, indices) : builder.MakeInput<IndexType>(indices_shape, indices);
    auto* output = builder.MakeOutput();

    // data_input -> Q -> DQ -> GatherElements
    auto* qdq_output = AddQDQNodePair<DataQType>(builder, data_input, qdq_scale, zero_point);
    auto* gather_output = builder.MakeIntermediate();

    Node& gather_elems_node = builder.AddNode("GatherElements", {qdq_output, indices_input}, {gather_output});
    gather_elems_node.AddAttribute("axis", axis);

    // -> Q -> DQ -> output
    auto* q_output = builder.MakeIntermediate();
    builder.AddQuantizeLinearNode<DataQType>(gather_output, qdq_scale, zero_point, q_output);
    builder.AddDequantizeLinearNode<DataQType>(q_output, qdq_scale, zero_point, output);
  };
}

// Runs an GatherElements model on the QNN CPU backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
template <typename DataType = float, typename IndexType = int32_t>
static void RunCPUGatherElemsOpTest(const std::vector<int64_t>& data_shape,
                                    const std::vector<DataType>& data,
                                    const std::vector<int64_t>& indices_shape,
                                    const std::vector<IndexType>& indices,
                                    bool indices_are_static,
                                    int64_t axis,
                                    ExpectedEPNodeAssignment expected_ep_assignment, const char* test_description,
                                    int opset = 13) {
  ProviderOptions provider_options;
  float fp32_abs_err = 1e-5f;  // default tolerance

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnCpu.dll";
#else
  provider_options["backend_path"] = "libQnnCpu.so";
#endif

  constexpr int expected_nodes_in_partition = 1;
  RunQnnModelTest(BuildGatherElemsTestCase<DataType, IndexType>(data_shape, data, indices_shape, indices,
                                                                indices_are_static, axis),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  expected_nodes_in_partition,
                  test_description,
                  fp32_abs_err);
}

// Runs an GatherElements model on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
template <typename DataType = float, typename DataQType = uint8_t, typename IndexType = int32_t>
static void RunHTPGatherElemsOpTest(const std::vector<int64_t>& data_shape,
                                    const std::vector<DataType>& data,
                                    const std::vector<int64_t>& indices_shape,
                                    const std::vector<IndexType>& indices,
                                    bool indices_are_static,
                                    int64_t axis,
                                    ExpectedEPNodeAssignment expected_ep_assignment, const char* test_description,
                                    int opset = 13) {
  ProviderOptions provider_options;
  float fp32_abs_err = 1e-5f;  // default tolerance

#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif

  constexpr int expected_nodes_in_partition = 1;
  RunQnnModelTest(BuildQDQGatherElemsTestCase<DataType, DataQType, IndexType>(data_shape, data, indices_shape, indices,
                                                                              indices_are_static, axis),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  expected_nodes_in_partition,
                  test_description,
                  fp32_abs_err);
}

//
// CPU tests:
//

// Test GatherElements op on CPU backend:
// positive, dynamic, int32 indices.
TEST_F(QnnCPUBackendTests, GatherElems_f32_IndicesInt32) {
  RunCPUGatherElemsOpTest<float, int32_t>(
      {2, 2}, {1.f, 2.f, 3.f, 4.f}, {2, 2}, {0, 0, 1, 0}, false, 1,
      ExpectedEPNodeAssignment::All, "GatherElems_f32_IndicesInt32");
}

// Test GatherElements op on CPU backend:
// positive, static, int64 indices.
TEST_F(QnnCPUBackendTests, GatherElems_f32_StaticIndicesInt64) {
  RunCPUGatherElemsOpTest<float, int64_t>(
      {2, 2}, {1.f, 2.f, 3.f, 4.f}, {2, 2}, {0, 0, 1, 0}, true, 1,
      ExpectedEPNodeAssignment::All, "GatherElems_f32_StaticIndicesInt64");
}

// Test GatherElements op on CPU backend:
// positive, static, int32 indices.
TEST_F(QnnCPUBackendTests, GatherElems_f32_StaticIndicesInt32) {
  RunCPUGatherElemsOpTest<float, int32_t>(
      {2, 2}, {1.f, 2.f, 3.f, 4.f}, {2, 2}, {0, 0, 1, 0}, true, 1,
      ExpectedEPNodeAssignment::All, "GatherElems_f32_StaticIndicesInt32");
}

// Test GatherElements op on CPU backend:
// positive, static, int64 indices.
TEST_F(QnnCPUBackendTests, GatherElems_f32_StaticNegIndicesInt64) {
  RunCPUGatherElemsOpTest<float, int64_t>(
      {2, 2}, {1.f, 2.f, 3.f, 4.f}, {2, 2}, {-1, 0, -1, 0}, true, 1,
      ExpectedEPNodeAssignment::All, "GatherElems_f32_StaticNegIndicesInt64");
}

// Test GatherElements op on CPU backend:
// positive, static, int32 indices.
TEST_F(QnnCPUBackendTests, GatherElems_f32_StaticNegIndicesInt32) {
  RunCPUGatherElemsOpTest<float, int32_t>(
      {2, 2}, {1.f, 2.f, 3.f, 4.f}, {2, 2}, {-1, 0, -1, 0}, true, 1,
      ExpectedEPNodeAssignment::All, "GatherElems_f32_StaticNegIndicesInt32");
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Test GatherElements op on HTP backend:
// positive, dynamic, int32 indices.
TEST_F(QnnHTPBackendTests, GatherElems_u8_IndicesInt32) {
  RunHTPGatherElemsOpTest<float, uint8_t, int32_t>(
      {3, 3}, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 9.f},
      {2, 3}, {1, 2, 0, 2, 0, 0}, false, 1,
      ExpectedEPNodeAssignment::All, "GatherElems_u8_IndicesInt32");
}

// Test GatherElements op on HTP backend:
// positive, static, int64 indices.
TEST_F(QnnHTPBackendTests, GatherElems_u8_StaticIndicesInt64) {
  RunHTPGatherElemsOpTest<float, uint8_t, int64_t>(
      {2, 2}, {1.f, 2.f, 3.f, 4.f}, {2, 2}, {0, 0, 1, 0}, true, 1,
      ExpectedEPNodeAssignment::All, "GatherElems_u8_StaticIndicesInt64");
}

// Test GatherElements op on HTP backend:
// positive, static, int32 indices.
TEST_F(QnnHTPBackendTests, GatherElems_u8_StaticIndicesInt32) {
  RunHTPGatherElemsOpTest<float, uint8_t, int32_t>(
      {2, 2}, {1.f, 2.f, 3.f, 4.f}, {2, 2}, {0, 0, 1, 0}, true, 1,
      ExpectedEPNodeAssignment::All, "GatherElems_u8_StaticIndicesInt32");
}

// Test GatherElements op on HTP backend:
// negative, static, int64 indices.
TEST_F(QnnHTPBackendTests, GatherElems_u8_StaticNegIndicesInt64) {
  RunHTPGatherElemsOpTest<float, uint8_t, int64_t>(
      {2, 2}, {1.f, 2.f, 3.f, 4.f}, {2, 2}, {0, 0, -1, 0}, true, 1,
      ExpectedEPNodeAssignment::All, "GatherElems_u8_StaticNegIndicesInt64");
}

// Test GatherElements op on HTP backend:
// negative, static, int32 indices.
TEST_F(QnnHTPBackendTests, GatherElems_u8_StaticNegIndicesInt32) {
  RunHTPGatherElemsOpTest<float, uint8_t, int32_t>(
      {2, 2}, {1.f, 2.f, 3.f, 4.f}, {2, 2}, {0, 0, -1, 0}, true, 1,
      ExpectedEPNodeAssignment::All, "GatherElems_u8_StaticNegIndicesInt32");
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
