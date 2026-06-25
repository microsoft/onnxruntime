// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "core/session/inference_session.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"

namespace onnxruntime {
namespace test {

template <typename T>
static void RunTest(
    T start,
    T limit,
    T delta,
    const std::vector<int64_t>& output_dims,
    const std::vector<T>& output) {
  // ONNX domain opset-11
  OpTester test1("Range", 11);
  test1.AddInput<T>("start", {}, {start});
  test1.AddInput<T>("limit", {}, {limit});
  test1.AddInput<T>("delta", {}, {delta});
  test1.AddOutput<T>("output", output_dims, output);
  // TensorRT do not yet support opset-11 and builds break on this test, hence exclude the EP
  test1.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

#ifndef DISABLE_CONTRIB_OPS

  // MSFT domain opset-1 (contrib op)
  OpTester test2("Range", 1, kMSDomain);
  test2.AddInput<T>("start", {}, {start});
  test2.AddInput<T>("limit", {}, {limit});

  if (delta != T{1})  // only contrib schema allows optional 'delta' input
    test2.AddInput<T>("delta", {}, {delta});

  test2.AddOutput<T>("output", output_dims, output);
  // TensorRT doesn't fully support opset 11 yet
  test2.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

#endif
}  // namespace test

TEST(RangeTest, Int32_DeltaDefault) {
  RunTest<int32_t>(0, 5, 1, {5}, {0, 1, 2, 3, 4});
}

TEST(RangeTest, Int64_DeltaDefault) {
  RunTest<int64_t>(0, 5, 1, {5}, {0, 1, 2, 3, 4});
}

TEST(RangeTest, Float_DeltaDefault) {
  RunTest<float>(0.f, 5.f, 1.f, {5}, {0.f, 1.f, 2.f, 3.f, 4.f});
}

TEST(RangeTest, Double_DeltaDefault) {
  RunTest<double>(0., 5., 1., {5}, {0., 1., 2., 3., 4.});
}

TEST(RangeTest, Int32_Delta_NonDefault) {
  RunTest<int32_t>(0, 10, 2, {5}, {0, 2, 4, 6, 8});
}

TEST(RangeTest, Int64_Delta_NonDefault_0) {
  RunTest<int64_t>(0, 9, 2, {5}, {0, 2, 4, 6, 8});
}

TEST(RangeTest, Int64_Delta_NonDefault_1) {
  RunTest<int64_t>(1, 2, 2, {1}, {1});
}

TEST(RangeTest, Int32_NegativeDelta_0) {
  RunTest<int32_t>(2, -9, -2, {6}, {2, 0, -2, -4, -6, -8});
}

TEST(RangeTest, Int32_NegativeDelta_1) {
  RunTest<int32_t>(2, 9, -2, {0}, {});
}

TEST(RangeTest, Float_NegativeDelta_0) {
  RunTest<float>(2.0f, -8.1f, -2.0f, {6}, {2.0f, 0.0f, -2.0f, -4.0f, -6.0f, -8.0f});
}

TEST(RangeTest, Float_SameStartAndLimit) {
  RunTest<float>(2.0f, 2.0f, 1, {0}, {});
}

TEST(RangeTest, AlmostSameStartAndLimitHighDelta) {
  RunTest<float>(2.0f, 2.01f, 1000000.0f, {1}, {2.0f});
}

#ifndef DISABLE_CONTRIB_OPS

namespace {
// Adds a scalar double graph initializer whose raw_data is set to the provided bytes.
// The bytes length is intentionally controllable so tests can supply a truncated payload
// (shorter than sizeof(double)) to exercise input validation in shape inference.
void AddDoubleScalarInitializerWithRawData(ONNX_NAMESPACE::GraphProto* graph,
                                           const std::string& name,
                                           const std::string& raw_bytes) {
  auto* initializer = graph->add_initializer();
  initializer->set_name(name);
  initializer->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);
  initializer->set_raw_data(raw_bytes);
}

std::string DoubleToRawData(double value) {
  return std::string(reinterpret_cast<const char*>(&value), sizeof(double));
}

// Builds a single com.microsoft Range node model whose start/limit/delta inputs are
// supplied as graph initializers, then loads and initializes it in a session.
common::Status BuildAndInitializeContribRangeModel(const std::string& start_raw_data,
                                                    const std::string& limit_raw_data,
                                                    const std::string& delta_raw_data,
                                                    InferenceSession& session_object) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  model.add_opset_import()->set_version(11);
  auto* ms_opset = model.add_opset_import();
  ms_opset->set_domain(kMSDomain);
  ms_opset->set_version(1);

  auto* graph = model.mutable_graph();
  graph->set_name("ContribRangeGraph");

  AddDoubleScalarInitializerWithRawData(graph, "start", start_raw_data);
  AddDoubleScalarInitializerWithRawData(graph, "limit", limit_raw_data);
  AddDoubleScalarInitializerWithRawData(graph, "delta", delta_raw_data);

  auto* output = graph->add_output();
  output->set_name("Y");
  output->mutable_type()->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_DOUBLE);

  auto* range_node = graph->add_node();
  range_node->set_name("Range");
  range_node->set_op_type("Range");
  range_node->set_domain(kMSDomain);
  range_node->add_input("start");
  range_node->add_input("limit");
  range_node->add_input("delta");
  range_node->add_output("Y");

  std::string serialized_model;
  if (!model.SerializeToString(&serialized_model)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to serialize test model.");
  }

  std::stringstream model_stream(serialized_model);
  ORT_RETURN_IF_ERROR(session_object.Load(model_stream));
  return session_object.Initialize();
}
}  // namespace

// Verifies that Range shape inference rejects an initializer whose raw_data is shorter than
// the element size for its declared data type, returning a clean failure status instead of
// reading past the end of the buffer. The model must fail to load/initialize.
TEST(RangeTest, ContribOp_TruncatedRawData_FailsCleanly) {
  // 'start' carries fewer bytes than sizeof(double); 'limit'/'delta' are well formed.
  const std::string truncated_start(sizeof(double) / 2, '\0');
  SessionOptions so;
  so.session_logid = "RangeTest.ContribOp_TruncatedRawData_FailsCleanly";
  InferenceSession session_object{so, GetEnvironment()};
  const auto status = BuildAndInitializeContribRangeModel(truncated_start,
                                                          DoubleToRawData(5.0),
                                                          DoubleToRawData(1.0),
                                                          session_object);
  ASSERT_FALSE(status.IsOK());
}

// Verifies that shape inference clamps an empty/backward range to a zero-sized dimension,
// matching the CPU kernel behavior, instead of emitting a negative dimension value.
TEST(RangeTest, ContribOp_BackwardRange_InfersZeroDim) {
  SessionOptions so;
  so.session_logid = "RangeTest.ContribOp_BackwardRange_InfersZeroDim";
  InferenceSession session_object{so, GetEnvironment()};
  // start > limit with a positive delta yields an empty range.
  const auto status = BuildAndInitializeContribRangeModel(DoubleToRawData(5.0),
                                                          DoubleToRawData(0.0),
                                                          DoubleToRawData(1.0),
                                                          session_object);
  ASSERT_STATUS_OK(status);

  const auto outputs = session_object.GetModelOutputs();
  ASSERT_STATUS_OK(outputs.first);
  ASSERT_EQ(outputs.second->size(), 1u);
  const auto* shape = (*outputs.second)[0]->Shape();
  ASSERT_NE(shape, nullptr);
  ASSERT_EQ(shape->dim_size(), 1);
  ASSERT_TRUE(shape->dim(0).has_dim_value());
  ASSERT_EQ(shape->dim(0).dim_value(), 0);
}

#endif  // DISABLE_CONTRIB_OPS

}  // namespace test
}  // namespace onnxruntime
