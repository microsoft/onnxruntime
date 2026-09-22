// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ORT_MINIMAL_BUILD

#include "core/common/span_utils.h"
#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/default_providers.h"
#include "core/util/qmath.h"

#include <chrono>
#include <random>

#include "gtest/gtest.h"
#include "onnx/shape_inference/implementation.h"

namespace onnxruntime {
namespace test {

namespace {

void RunInvalidShapeInferenceTest(const std::vector<int64_t>& b_dims,
                                  const std::vector<uint8_t>& b_data,
                                  const std::vector<int64_t>& b_shape_dims,
                                  const std::vector<int64_t>& b_shape_data,
                                  const std::string& expected_error) {
  OpTester test("MatMulFpQ4", 1, kMSDomain);
  test.AddAttribute<int64_t>("blk_quant_type", BlkQ4Zp8);
  test.AddInput<float>("A", {1, 4}, std::vector<float>(4), true);
  test.AddInput<uint8_t>("B", b_dims, b_data, true);
  test.AddInput<int64_t>("B_shape", b_shape_dims, b_shape_data, true);
  test.AddOutput<float>("Y", {1, 1}, {0.0f});
  test.Run(OpTester::ExpectResult::kExpectFailure, expected_error);
}

void RunSymbolicInputLengthsShapeInferenceTest() {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);

  auto* default_opset = model.add_opset_import();
  default_opset->set_domain(kOnnxDomain);
  default_opset->set_version(21);
  auto* ms_opset = model.add_opset_import();
  ms_opset->set_domain(kMSDomain);
  ms_opset->set_version(1);

  auto* graph = model.mutable_graph();
  graph->set_name("symbolic_input_lengths");

  auto add_input = [graph](const char* name, int32_t elem_type) {
    auto* input = graph->add_input();
    input->set_name(name);
    auto* tensor_type = input->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(elem_type);
    return tensor_type->mutable_shape();
  };

  auto* a_shape = add_input("A", ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  a_shape->add_dim()->set_dim_value(1);
  a_shape->add_dim()->set_dim_value(4);
  add_input("B", ONNX_NAMESPACE::TensorProto_DataType_UINT8)->add_dim()->set_dim_param("packed_size");
  add_input("B_shape", ONNX_NAMESPACE::TensorProto_DataType_INT64)->add_dim()->set_dim_param("shape_length");

  auto* node = graph->add_node();
  node->set_op_type("MatMulFpQ4");
  node->set_domain(kMSDomain);
  node->add_input("A");
  node->add_input("B");
  node->add_input("B_shape");
  node->add_output("Y");
  auto* attribute = node->add_attribute();
  attribute->set_name("blk_quant_type");
  attribute->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  attribute->set_i(BlkQ4Zp8);

  ONNX_NAMESPACE::shape_inference::InferShapes(
      model, ONNX_NAMESPACE::OpSchemaRegistry::Instance());
}

}  // namespace

TEST(MatMulFpQ4, RejectsScalarBShape) {
  RunInvalidShapeInferenceTest({1}, {0}, {}, {4},
                               "B_shape input for MatMulFpQ4 must be a 1-D int64 tensor");
}

TEST(MatMulFpQ4, AllowsSymbolicInputLengths) {
  RunSymbolicInputLengthsShapeInferenceTest();
}

TEST(MatMulFpQ4, RejectsKnownWrongBShapeLength) {
  RunInvalidShapeInferenceTest({1}, {0}, {3}, {4, 1, 0},
                               "B_shape input for MatMulFpQ4 must be a 1-D int64 tensor of length 2");
}

TEST(MatMulFpQ4, RejectsNegativeBShapeDimension) {
  RunInvalidShapeInferenceTest({1}, {0}, {2}, {-1, 1},
                               "B_shape initializer for MatMulFpQ4 must contain non-negative dimensions");
}

TEST(MatMulFpQ4, RejectsScalarPackedB) {
  RunInvalidShapeInferenceTest({}, {0}, {2}, {4, 1}, "B input for MatMulFpQ4 must be a 1-D tensor");
}

TEST(MatMulFpQ4, RejectsWrongPackedBSize) {
  const size_t expected_pack_size = MlasQ4GemmPackBSize(BlkQ4Zp8, 1, 4);
  if (expected_pack_size == 0) {
    GTEST_SKIP();
  }

  const size_t wrong_pack_size = expected_pack_size + 1;
  RunInvalidShapeInferenceTest({static_cast<int64_t>(wrong_pack_size)},
                               std::vector<uint8_t>(wrong_pack_size),
                               {2}, {4, 1}, "Input q4 tensors of wrong size");
}

TEST(MatMulFpQ4, MatMul2DSym) {
  // (100 x 52) X (52 x 288)
  constexpr int64_t M = 100;
  constexpr int64_t N = 288;
  constexpr int64_t K = 52;

  const auto buf_size = MlasQ4GemmPackBSize(BlkQ4Sym, (size_t)N, (size_t)K);
  if (buf_size == 0) {
    GTEST_SKIP();  // operation not supported on this hardware platform yet.
  }

  OpTester test("MatMulFpQ4", 1, kMSDomain);
  test.AddAttribute<int64_t>("blk_quant_type", BlkQ4Sym);

  std::vector<float> input0_vals(M * K);
  float fv = -135.f;
  for (auto& f : input0_vals) {
    f = fv / 128;
    fv++;
    if (fv > 135.f) {
      fv = -135.f;
    }
  }

  std::vector<float> input1_f_vals(N * K);
  int v = -2;
  for (size_t i = 0; i < N * K; i++) {
    if (v == 0 || v == -3 || v == 3) v++;
    input1_f_vals[i] = (float)v;
    if (++v >= 8) {
      v = -8;
    }
  }
  std::vector<uint8_t> input1_vals(buf_size);
  MlasQ4GemmPackB(BlkQ4Sym, input1_vals.data(), input1_f_vals.data(), (size_t)N, (size_t)K, (size_t)N);

  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += input0_vals[m * K + k] * input1_f_vals[k * N + n];
      }
      expected_vals[m * N + n] = sum;
    }
  }

  test.AddInput<float>("A", {M, K}, input0_vals, false);
  test.AddInput<uint8_t>("B", {(int64_t)input1_vals.size()}, input1_vals, true);
  test.AddInput<int64_t>("B_shape", {(int64_t)2}, {(int64_t)K, (int64_t)N}, true);

  test.AddOutput<float>("Y", {M, N}, expected_vals);

  test.Run();
}

TEST(MatMulFpQ4, MatMul2DBlkZp) {
  // (100 x 41) X (41 x 288)
  constexpr int64_t M = 100;
  constexpr int64_t N = 288;
  constexpr int64_t K = 41;

  const auto buf_size = MlasQ4GemmPackBSize(BlkQ4Zp8, (size_t)N, (size_t)K);
  if (buf_size == 0) {
    GTEST_SKIP();  // operation not yet supported on this hardware platform.
  }

  OpTester test("MatMulFpQ4", 1, kMSDomain);
  test.AddAttribute<int64_t>("blk_quant_type", BlkQ4Zp8);

  std::vector<float> input0_vals(M * K);
  float fv = -135.f;
  for (auto& f : input0_vals) {
    f = fv / 128;
    fv++;
    if (fv > 135.f) {
      fv = -135.f;
    }
  }

  std::vector<float> input1_f_vals(N * K);
  int v = 0;
  for (size_t i = 0; i < N * K; i++) {
    input1_f_vals[i] = (float)v;
    if (++v >= 16) {
      v = 0;
    }
  }
  std::vector<uint8_t> input1_vals(buf_size);
  MlasQ4GemmPackB(BlkQ4Zp8, input1_vals.data(), input1_f_vals.data(), (size_t)N, (size_t)K, (size_t)N);

  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += input0_vals[m * K + k] * input1_f_vals[k * N + n];
      }
      expected_vals[m * N + n] = sum;
    }
  }

  test.AddInput<float>("A", {M, K}, input0_vals, false);
  test.AddInput<uint8_t>("B", {(int64_t)input1_vals.size()}, input1_vals, true);
  test.AddInput<int64_t>("B_shape", {(int64_t)2}, {(int64_t)K, (int64_t)N}, true);

  test.AddOutput<float>("Y", {M, N}, expected_vals);

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
