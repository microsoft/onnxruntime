// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Unit tests for the DQMatMulNBitsFusion graph transformer.
// Tests Pattern 1: DQ(3D,axis=2)->Reshape->Transpose([1,0])->[Cast]->MatMul/Gemm -> MatMulNBits
// Tests Pattern 2: DQ(2D,axis=0)->MatMul/Gemm -> MatMulNBits

#include "core/common/span_utils.h"
#include "core/framework/int4.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/dq_matmulnbits_fusion.h"

#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/util/include/asserts.h"

#include "gtest/gtest.h"

#if !defined(DISABLE_CONTRIB_OPS)

namespace onnxruntime {
namespace test {

// =============================================================================
// Helper: create packed UINT4 data (N logical 4-bit elements -> (N+1)/2 bytes)
// =============================================================================
static std::vector<UInt4x2> MakePackedUint4(const std::vector<uint8_t>& values) {
  std::vector<UInt4x2> packed(UInt4x2::CalcNumInt4Pairs(values.size()));
  UInt4x2::Pack(packed, values);
  return packed;
}

// =============================================================================
// Pattern 1 tests:
//   DQ(3D, axis=2, UINT4, block_size) --> Reshape(2D) --> Transpose([1,0])
//     --> MatMul(A, transposed_weight) --> output
//
// Weight shape for DQ: [N, num_blocks, block_size] (UINT4)
// Scale shape:         [N, num_blocks, 1]
// ZP shape (opt):      [N, num_blocks, 1] (UINT4)
// Reshape target:      [N, K]   (K = num_blocks * block_size)
// Transpose perm:      [1, 0]   -> [K, N]
// MatMul:              [M, K] x [K, N] -> [M, N]
// =============================================================================

// Build a Pattern 1 test graph: DQ -> Reshape -> Transpose -> MatMul
static void BuildPattern1Graph(ModelTestBuilder& builder,
                               int64_t M, int64_t N, int64_t K,
                               int64_t block_size,
                               bool with_zp,
                               bool with_cast,
                               bool use_gemm,
                               const std::vector<uint8_t>* weight_values = nullptr,
                               const std::vector<float>* scale_values = nullptr,
                               const std::vector<uint8_t>* zp_values = nullptr) {
  const int64_t num_blocks = K / block_size;

  // Input A: [M, K] float
  auto* input_a = builder.MakeInput<float>({M, K}, -1.0f, 1.0f);
  auto* output = builder.MakeOutput();

  // Weight initializer: [N, num_blocks, block_size] UINT4
  const int64_t weight_elems = N * num_blocks * block_size;
  std::vector<uint8_t> w_vals;
  if (weight_values) {
    w_vals = *weight_values;
  } else {
    w_vals.resize(static_cast<size_t>(weight_elems));
    for (size_t i = 0; i < w_vals.size(); ++i) {
      w_vals[i] = static_cast<uint8_t>(i % 16);
    }
  }
  auto w_packed = MakePackedUint4(w_vals);
  auto* weight_arg = builder.MakeInitializer<UInt4x2>(
      {N, num_blocks, block_size}, w_packed);

  // Scale initializer: [N, num_blocks, 1] float
  std::vector<float> s_vals;
  if (scale_values) {
    s_vals = *scale_values;
  } else {
    s_vals.resize(static_cast<size_t>(N * num_blocks));
    for (size_t i = 0; i < s_vals.size(); ++i) {
      s_vals[i] = 0.1f + 0.01f * static_cast<float>(i % 10);
    }
  }
  auto* scale_arg = builder.MakeInitializer<float>({N, num_blocks, 1}, s_vals);

  // DQ node
  NodeAttributes dq_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(2)), dq_attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);

  auto* dq_output = builder.MakeIntermediate();
  if (with_zp) {
    std::vector<uint8_t> z_vals;
    if (zp_values) {
      z_vals = *zp_values;
    } else {
      z_vals.resize(static_cast<size_t>(N * num_blocks));
      for (size_t i = 0; i < z_vals.size(); ++i) {
        z_vals[i] = 8;  // default ZP for UINT4 in MatMulNBits
      }
    }
    auto zp_packed = MakePackedUint4(z_vals);
    auto* zp_arg = builder.MakeInitializer<UInt4x2>({N, num_blocks, 1}, zp_packed);
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg, zp_arg}, {dq_output}, "", &dq_attrs);
  } else {
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &dq_attrs);
  }

  // Reshape: [N, num_blocks, block_size] -> [N, K]
  auto* reshape_shape = builder.MakeInitializer<int64_t>({2}, {N, K});
  auto* reshape_output = builder.MakeIntermediate();
  builder.AddNode("Reshape", {dq_output, reshape_shape}, {reshape_output});

  // Transpose: [N, K] -> [K, N] with perm=[1, 0]
  NodeAttributes tp_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("perm", std::vector<int64_t>{1, 0}), tp_attrs);
  auto* transpose_output = builder.MakeIntermediate();
  builder.AddNode("Transpose", {reshape_output}, {transpose_output}, "", &tp_attrs);

  NodeArg* matmul_b = transpose_output;

  // Optional Cast (identity cast, same type)
  if (with_cast) {
    auto* cast_output = builder.MakeIntermediate();
    NodeAttributes cast_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("to", static_cast<int64_t>(1) /*FLOAT*/), cast_attrs);
    builder.AddNode("Cast", {transpose_output}, {cast_output}, "", &cast_attrs);
    matmul_b = cast_output;
  }

  // MatMul or Gemm
  if (use_gemm) {
    builder.AddNode("Gemm", {input_a, matmul_b}, {output});
  } else {
    builder.AddNode("MatMul", {input_a, matmul_b}, {output});
  }
}

// Build a Pattern 1 test graph with Gemm + bias
static void BuildPattern1GemmBiasGraph(ModelTestBuilder& builder,
                                       int64_t M, int64_t N, int64_t K,
                                       int64_t block_size,
                                       bool with_zp) {
  const int64_t num_blocks = K / block_size;

  auto* input_a = builder.MakeInput<float>({M, K}, -1.0f, 1.0f);
  auto* output = builder.MakeOutput();

  // Weight: [N, num_blocks, block_size] UINT4
  const int64_t weight_elems = N * num_blocks * block_size;
  std::vector<uint8_t> w_vals(static_cast<size_t>(weight_elems));
  for (size_t i = 0; i < w_vals.size(); ++i) w_vals[i] = static_cast<uint8_t>(i % 16);
  auto w_packed = MakePackedUint4(w_vals);
  auto* weight_arg = builder.MakeInitializer<UInt4x2>({N, num_blocks, block_size}, w_packed);

  // Scale: [N, num_blocks, 1] float
  std::vector<float> s_vals(static_cast<size_t>(N * num_blocks));
  for (size_t i = 0; i < s_vals.size(); ++i) s_vals[i] = 0.1f;
  auto* scale_arg = builder.MakeInitializer<float>({N, num_blocks, 1}, s_vals);

  // DQ
  NodeAttributes dq_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(2)), dq_attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);
  auto* dq_output = builder.MakeIntermediate();

  if (with_zp) {
    std::vector<uint8_t> z_vals(static_cast<size_t>(N * num_blocks), 8);
    auto zp_packed = MakePackedUint4(z_vals);
    auto* zp_arg = builder.MakeInitializer<UInt4x2>({N, num_blocks, 1}, zp_packed);
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg, zp_arg}, {dq_output}, "", &dq_attrs);
  } else {
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &dq_attrs);
  }

  // Reshape->Transpose
  auto* reshape_shape = builder.MakeInitializer<int64_t>({2}, {N, K});
  auto* reshape_output = builder.MakeIntermediate();
  builder.AddNode("Reshape", {dq_output, reshape_shape}, {reshape_output});

  NodeAttributes tp_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("perm", std::vector<int64_t>{1, 0}), tp_attrs);
  auto* transpose_output = builder.MakeIntermediate();
  builder.AddNode("Transpose", {reshape_output}, {transpose_output}, "", &tp_attrs);

  // Gemm with bias
  auto* bias_arg = builder.MakeInitializer<float>({N}, std::vector<float>(static_cast<size_t>(N), 0.5f));
  builder.AddNode("Gemm", {input_a, transpose_output, bias_arg}, {output});
}

// =============================================================================
// Pattern 2 tests:
//   DQ(2D, axis=0, UINT4, block_size) --> MatMul(A, DQ_output) --> output
//
// Weight shape for DQ: [K, N] (UINT4)
// Scale shape:         [k_blocks, N]
// ZP shape (opt):      [k_blocks, N] (UINT4)
// =============================================================================

static void BuildPattern2Graph(ModelTestBuilder& builder,
                               int64_t M, int64_t N, int64_t K,
                               int64_t block_size,
                               bool with_zp,
                               bool use_gemm) {
  const int64_t k_blocks = K / block_size;

  auto* input_a = builder.MakeInput<float>({M, K}, -1.0f, 1.0f);
  auto* output = builder.MakeOutput();

  // Weight: [K, N] UINT4
  std::vector<uint8_t> w_vals(static_cast<size_t>(K * N));
  for (size_t i = 0; i < w_vals.size(); ++i) w_vals[i] = static_cast<uint8_t>(i % 16);
  auto w_packed = MakePackedUint4(w_vals);
  auto* weight_arg = builder.MakeInitializer<UInt4x2>({K, N}, w_packed);

  // Scale: [k_blocks, N] float
  std::vector<float> s_vals(static_cast<size_t>(k_blocks * N));
  for (size_t i = 0; i < s_vals.size(); ++i) s_vals[i] = 0.1f + 0.01f * static_cast<float>(i % 10);
  auto* scale_arg = builder.MakeInitializer<float>({k_blocks, N}, s_vals);

  // DQ
  NodeAttributes dq_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), dq_attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);
  auto* dq_output = builder.MakeIntermediate();

  if (with_zp) {
    std::vector<uint8_t> z_vals(static_cast<size_t>(k_blocks * N), 8);
    auto zp_packed = MakePackedUint4(z_vals);
    auto* zp_arg = builder.MakeInitializer<UInt4x2>({k_blocks, N}, zp_packed);
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg, zp_arg}, {dq_output}, "", &dq_attrs);
  } else {
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &dq_attrs);
  }

  if (use_gemm) {
    builder.AddNode("Gemm", {input_a, dq_output}, {output});
  } else {
    builder.AddNode("MatMul", {input_a, dq_output}, {output});
  }
}

static void BuildPattern2GemmBiasGraph(ModelTestBuilder& builder,
                                       int64_t M, int64_t N, int64_t K,
                                       int64_t block_size,
                                       bool with_zp) {
  const int64_t k_blocks = K / block_size;

  auto* input_a = builder.MakeInput<float>({M, K}, -1.0f, 1.0f);
  auto* output = builder.MakeOutput();

  // Weight: [K, N] UINT4
  std::vector<uint8_t> w_vals(static_cast<size_t>(K * N));
  for (size_t i = 0; i < w_vals.size(); ++i) w_vals[i] = static_cast<uint8_t>(i % 16);
  auto w_packed = MakePackedUint4(w_vals);
  auto* weight_arg = builder.MakeInitializer<UInt4x2>({K, N}, w_packed);

  // Scale: [k_blocks, N] float
  std::vector<float> s_vals(static_cast<size_t>(k_blocks * N));
  for (size_t i = 0; i < s_vals.size(); ++i) s_vals[i] = 0.1f;
  auto* scale_arg = builder.MakeInitializer<float>({k_blocks, N}, s_vals);

  NodeAttributes dq_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), dq_attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);
  auto* dq_output = builder.MakeIntermediate();

  if (with_zp) {
    std::vector<uint8_t> z_vals(static_cast<size_t>(k_blocks * N), 8);
    auto zp_packed = MakePackedUint4(z_vals);
    auto* zp_arg = builder.MakeInitializer<UInt4x2>({k_blocks, N}, zp_packed);
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg, zp_arg}, {dq_output}, "", &dq_attrs);
  } else {
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &dq_attrs);
  }

  auto* bias_arg = builder.MakeInitializer<float>({N}, std::vector<float>(static_cast<size_t>(N), 0.5f));
  builder.AddNode("Gemm", {input_a, dq_output, bias_arg}, {output});
}

// =============================================================================
// Negative test graph builders
// =============================================================================

// Pattern 1 with wrong axis (axis=0 instead of axis=2)
static void BuildPattern1WrongAxis(ModelTestBuilder& builder,
                                   int64_t M, int64_t N, int64_t K,
                                   int64_t block_size) {
  const int64_t num_blocks = K / block_size;
  auto* input_a = builder.MakeInput<float>({M, K}, -1.0f, 1.0f);
  auto* output = builder.MakeOutput();

  std::vector<uint8_t> w_vals(static_cast<size_t>(N * num_blocks * block_size));
  for (size_t i = 0; i < w_vals.size(); ++i) w_vals[i] = static_cast<uint8_t>(i % 16);
  auto w_packed = MakePackedUint4(w_vals);
  auto* weight_arg = builder.MakeInitializer<UInt4x2>({N, num_blocks, block_size}, w_packed);

  std::vector<float> s_vals(static_cast<size_t>(N * num_blocks), 0.1f);
  auto* scale_arg = builder.MakeInitializer<float>({N, num_blocks, 1}, s_vals);

  // Wrong axis = 0 (Pattern 1 expects axis=2)
  NodeAttributes dq_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), dq_attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);
  auto* dq_output = builder.MakeIntermediate();
  builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &dq_attrs);

  auto* reshape_shape = builder.MakeInitializer<int64_t>({2}, {N, K});
  auto* reshape_output = builder.MakeIntermediate();
  builder.AddNode("Reshape", {dq_output, reshape_shape}, {reshape_output});

  NodeAttributes tp_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("perm", std::vector<int64_t>{1, 0}), tp_attrs);
  auto* transpose_output = builder.MakeIntermediate();
  builder.AddNode("Transpose", {reshape_output}, {transpose_output}, "", &tp_attrs);

  builder.AddNode("MatMul", {input_a, transpose_output}, {output});
}

// Pattern 1 with missing block_size attribute (should not match)
static void BuildPattern1MissingBlockSize(ModelTestBuilder& builder,
                                          int64_t M, int64_t N, int64_t K,
                                          int64_t block_size) {
  const int64_t num_blocks = K / block_size;
  auto* input_a = builder.MakeInput<float>({M, K}, -1.0f, 1.0f);
  auto* output = builder.MakeOutput();

  std::vector<uint8_t> w_vals(static_cast<size_t>(N * num_blocks * block_size));
  for (size_t i = 0; i < w_vals.size(); ++i) w_vals[i] = static_cast<uint8_t>(i % 16);
  auto w_packed = MakePackedUint4(w_vals);
  auto* weight_arg = builder.MakeInitializer<UInt4x2>({N, num_blocks, block_size}, w_packed);

  std::vector<float> s_vals(static_cast<size_t>(N * num_blocks), 0.1f);
  auto* scale_arg = builder.MakeInitializer<float>({N, num_blocks, 1}, s_vals);

  // DQ with axis=2 but NO block_size attribute => should not match
  NodeAttributes dq_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(2)), dq_attrs);
  auto* dq_output = builder.MakeIntermediate();
  builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &dq_attrs);

  auto* reshape_shape = builder.MakeInitializer<int64_t>({2}, {N, K});
  auto* reshape_output = builder.MakeIntermediate();
  builder.AddNode("Reshape", {dq_output, reshape_shape}, {reshape_output});

  NodeAttributes tp_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("perm", std::vector<int64_t>{1, 0}), tp_attrs);
  auto* transpose_output = builder.MakeIntermediate();
  builder.AddNode("Transpose", {reshape_output}, {transpose_output}, "", &tp_attrs);

  builder.AddNode("MatMul", {input_a, transpose_output}, {output});
}

// Pattern 2 with non-constant weight (input instead of initializer)
static void BuildPattern2NonConstWeight(ModelTestBuilder& builder,
                                        int64_t M, int64_t N, int64_t K,
                                        int64_t block_size) {
  const int64_t k_blocks = K / block_size;
  auto* input_a = builder.MakeInput<float>({M, K}, -1.0f, 1.0f);
  auto* output = builder.MakeOutput();

  // Weight is an input, NOT a constant initializer
  auto* weight_arg = builder.MakeInput<UInt4x2>({K, N},
                                                UInt4x2(UInt4x2::min_val, 0),
                                                UInt4x2(UInt4x2::max_val, 0));

  std::vector<float> s_vals(static_cast<size_t>(k_blocks * N), 0.1f);
  auto* scale_arg = builder.MakeInitializer<float>({k_blocks, N}, s_vals);

  NodeAttributes dq_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), dq_attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", block_size), dq_attrs);
  auto* dq_output = builder.MakeIntermediate();
  builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &dq_attrs);

  builder.AddNode("MatMul", {input_a, dq_output}, {output});
}

// Pattern 2 with non-power-of-2 block size
static void BuildPattern2BadBlockSize(ModelTestBuilder& builder,
                                      int64_t M, int64_t N, int64_t K) {
  // block_size = 12 (not power of 2)
  const int64_t block_size = 12;
  const int64_t k_blocks = K / block_size;
  auto* input_a = builder.MakeInput<float>({M, K}, -1.0f, 1.0f);
  auto* output = builder.MakeOutput();

  std::vector<uint8_t> w_vals(static_cast<size_t>(K * N));
  for (size_t i = 0; i < w_vals.size(); ++i) w_vals[i] = static_cast<uint8_t>(i % 16);
  auto w_packed = MakePackedUint4(w_vals);
  auto* weight_arg = builder.MakeInitializer<UInt4x2>({K, N}, w_packed);

  std::vector<float> s_vals(static_cast<size_t>(k_blocks * N), 0.1f);
  auto* scale_arg = builder.MakeInitializer<float>({k_blocks, N}, s_vals);

  NodeAttributes dq_attrs;
  utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), dq_attrs);
  utils::SetNodeAttribute(utils::MakeAttribute("block_size", static_cast<int64_t>(12)), dq_attrs);
  auto* dq_output = builder.MakeIntermediate();
  builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &dq_attrs);

  builder.AddNode("MatMul", {input_a, dq_output}, {output});
}

// =============================================================================
// Helper to count ops by type, including domain-qualified names
// =============================================================================

static std::map<std::string, int> CountOpsInGraphByDomain(const Graph& graph) {
  std::map<std::string, int> op_counts;
  for (const auto& node : graph.Nodes()) {
    std::string key = node.OpType();
    if (!node.Domain().empty() && node.Domain() != kOnnxDomain) {
      key = node.Domain() + "." + key;
    }
    op_counts[key]++;
  }
  return op_counts;
}

// =============================================================================
// Test class
// =============================================================================

class DQMatMulNBitsFusionTest : public GraphTransformationTests {};

// ---------------------------------------------------------------------------
// Pattern 1: Basic MatMul fusion (no ZP, no Cast)
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Pattern1_MatMul_NoZP) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;
  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern1Graph(builder, M, N, K, block_size, /*with_zp=*/false,
                       /*with_cast=*/false, /*use_gemm=*/false);
  };

  auto pre_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["DequantizeLinear"], 1);
    EXPECT_EQ(ops["Reshape"], 1);
    EXPECT_EQ(ops["Transpose"], 1);
    EXPECT_EQ(ops["MatMul"], 1);
    return Status::OK();
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    // After fusion: DQ, Reshape, Transpose, MatMul should all be gone
    EXPECT_EQ(ops.count("DequantizeLinear"), 0);
    EXPECT_EQ(ops.count("Reshape"), 0);
    EXPECT_EQ(ops.count("Transpose"), 0);
    EXPECT_EQ(ops.count("MatMul"), 0);
    // MatMulNBits should appear
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);

    // Verify MatMulNBits attributes
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        const auto& attrs = node.GetAttributes();
        EXPECT_EQ(attrs.at("K").i(), K);
        EXPECT_EQ(attrs.at("N").i(), N);
        EXPECT_EQ(attrs.at("bits").i(), 4);
        EXPECT_EQ(attrs.at("block_size").i(), block_size);

        // Without ZP, DQ default ZP=0 requires explicit zeros for MatMulNBits
        // (DQ default is 0, MatMulNBits default is 8), so expect 4 inputs (A, B, scales, zp)
        EXPECT_EQ(node.InputDefs().size(), static_cast<size_t>(4));
      }
    }
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(/*accuracy_level=*/4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, pre_check, post_check));
}

// ---------------------------------------------------------------------------
// Pattern 1: MatMul with ZP (default value 8 => should be elided)
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Pattern1_MatMul_WithDefaultZP8) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern1Graph(builder, M, N, K, block_size, /*with_zp=*/true,
                       /*with_cast=*/false, /*use_gemm=*/false);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(ops.count("DequantizeLinear"), 0);

    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        // Default ZP=8 should be elided, so only 3 inputs (A, B, scales)
        EXPECT_EQ(node.InputDefs().size(), static_cast<size_t>(3));
      }
    }
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(/*accuracy_level=*/4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

// ---------------------------------------------------------------------------
// Pattern 1: MatMul with non-default ZP
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Pattern1_MatMul_WithNonDefaultZP) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;
  const int64_t num_blocks = K / block_size;

  // ZP values are not all 8 => should NOT be elided
  std::vector<uint8_t> zp_vals(static_cast<size_t>(N * num_blocks), 3);

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern1Graph(builder, M, N, K, block_size, /*with_zp=*/true,
                       /*with_cast=*/false, /*use_gemm=*/false,
                       nullptr, nullptr, &zp_vals);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);

    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        // Non-default ZP should be preserved, 4 inputs (A, B, scales, zp)
        EXPECT_EQ(node.InputDefs().size(), static_cast<size_t>(4));
      }
    }
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(/*accuracy_level=*/4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

// ---------------------------------------------------------------------------
// Pattern 1: MatMul with Cast (identity cast float->float)
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Pattern1_MatMul_WithCast) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern1Graph(builder, M, N, K, block_size, /*with_zp=*/false,
                       /*with_cast=*/true, /*use_gemm=*/false);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(ops.count("Cast"), 0);
    EXPECT_EQ(ops.count("MatMul"), 0);
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(/*accuracy_level=*/4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

// ---------------------------------------------------------------------------
// Pattern 1: Gemm with bias
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Pattern1_Gemm_WithBias) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern1GemmBiasGraph(builder, M, N, K, block_size, /*with_zp=*/true);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(ops.count("Gemm"), 0);

    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        // With bias: A, B, scales, [zp omitted if default 8], empty, empty, bias
        // The bias should be at index 5, with empties at 3 and 4
        // Since default ZP=8 is elided, inputs should be:
        // 0:A, 1:B, 2:scales, 3:empty, 4:empty, 5:bias
        EXPECT_GE(node.InputDefs().size(), static_cast<size_t>(6));
        // Bias should exist and have a name
        EXPECT_TRUE(node.InputDefs()[5] != nullptr);
        EXPECT_TRUE(node.InputDefs()[5]->Exists());
      }
    }
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(/*accuracy_level=*/4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

// ---------------------------------------------------------------------------
// Pattern 1: Gemm with no transpose → should fuse
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Pattern1_Gemm_NoZP) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern1Graph(builder, M, N, K, block_size, /*with_zp=*/false,
                       /*with_cast=*/false, /*use_gemm=*/true);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(ops.count("Gemm"), 0);
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(/*accuracy_level=*/4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

// ---------------------------------------------------------------------------
// Pattern 1: Larger block size = 32
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Pattern1_BlockSize32) {
  constexpr int64_t M = 2, N = 4, K = 64, block_size = 32;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern1Graph(builder, M, N, K, block_size, /*with_zp=*/false,
                       /*with_cast=*/false, /*use_gemm=*/false);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        EXPECT_EQ(node.GetAttributes().at("block_size").i(), block_size);
        EXPECT_EQ(node.GetAttributes().at("K").i(), K);
        EXPECT_EQ(node.GetAttributes().at("N").i(), N);
      }
    }
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(/*accuracy_level=*/4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

// ---------------------------------------------------------------------------
// Pattern 2: Basic MatMul fusion (no ZP)
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Pattern2_MatMul_NoZP) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern2Graph(builder, M, N, K, block_size, /*with_zp=*/false, /*use_gemm=*/false);
  };

  auto pre_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["DequantizeLinear"], 1);
    EXPECT_EQ(ops["MatMul"], 1);
    return Status::OK();
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops.count("DequantizeLinear"), 0);
    EXPECT_EQ(ops.count("MatMul"), 0);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);

    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        EXPECT_EQ(node.GetAttributes().at("K").i(), K);
        EXPECT_EQ(node.GetAttributes().at("N").i(), N);
        // DQ default ZP=0 != MNB default ZP=8, so explicit zeros emitted => 4 inputs
        EXPECT_EQ(node.InputDefs().size(), static_cast<size_t>(4));
      }
    }
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(/*accuracy_level=*/4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, pre_check, post_check));
}

// ---------------------------------------------------------------------------
// Pattern 2: MatMul with default ZP=8 (should be elided)
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Pattern2_MatMul_WithDefaultZP8) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern2Graph(builder, M, N, K, block_size, /*with_zp=*/true, /*use_gemm=*/false);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);

    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        // Default ZP=8 is elided => 3 inputs
        EXPECT_EQ(node.InputDefs().size(), static_cast<size_t>(3));
      }
    }
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(/*accuracy_level=*/4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

// ---------------------------------------------------------------------------
// Pattern 2: Gemm fusion
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Pattern2_Gemm) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern2Graph(builder, M, N, K, block_size, /*with_zp=*/false, /*use_gemm=*/true);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(ops.count("Gemm"), 0);
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(/*accuracy_level=*/4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

// ---------------------------------------------------------------------------
// Pattern 2: Gemm with bias
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Pattern2_Gemm_WithBias) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern2GemmBiasGraph(builder, M, N, K, block_size, /*with_zp=*/false);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);
    EXPECT_EQ(ops.count("Gemm"), 0);

    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        // Bias fused: A, B, scales, zp(explicit 0), empty, bias = 6 inputs
        EXPECT_GE(node.InputDefs().size(), static_cast<size_t>(6));
      }
    }
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(/*accuracy_level=*/4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

// ---------------------------------------------------------------------------
// Pattern 2: Larger block_size = 32
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Pattern2_BlockSize32) {
  constexpr int64_t M = 2, N = 4, K = 64, block_size = 32;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern2Graph(builder, M, N, K, block_size, /*with_zp=*/true, /*use_gemm=*/false);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops["com.microsoft.MatMulNBits"], 1);
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "MatMulNBits") {
        EXPECT_EQ(node.GetAttributes().at("block_size").i(), block_size);
      }
    }
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(/*accuracy_level=*/4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

// ---------------------------------------------------------------------------
// Pattern 1: accuracy_level parameter propagation
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Pattern1_AccuracyLevel) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  for (int64_t acc_level : {0, 1, 2, 3, 4}) {
    auto build = [&](ModelTestBuilder& builder) {
      BuildPattern1Graph(builder, M, N, K, block_size, /*with_zp=*/false,
                         /*with_cast=*/false, /*use_gemm=*/false);
    };

    auto post_check = [&](Graph& graph) -> Status {
      for (const auto& node : graph.Nodes()) {
        if (node.OpType() == "MatMulNBits") {
          EXPECT_EQ(node.GetAttributes().at("accuracy_level").i(), acc_level);
        }
      }
      return Status::OK();
    };

    auto transformer = std::make_unique<DQMatMulNBitsFusion>(acc_level);
    ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                          TransformerLevel::Level1, 1, nullptr, post_check));
  }
}

// =============================================================================
// Negative tests: transformations should NOT fire
// =============================================================================

// ---------------------------------------------------------------------------
// Negative: Pattern 1 with wrong DQ axis (axis!=2)
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Negative_Pattern1_WrongAxis) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern1WrongAxis(builder, M, N, K, block_size);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    // Should NOT fuse — DQ, Reshape, Transpose, MatMul all remain
    EXPECT_EQ(ops.count("com.microsoft.MatMulNBits"), 0);
    EXPECT_EQ(ops["MatMul"], 1);
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(/*accuracy_level=*/4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

// ---------------------------------------------------------------------------
// Negative: Pattern 1 with missing block_size (no block_size attr)
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Negative_Pattern1_MissingBlockSize) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern1MissingBlockSize(builder, M, N, K, block_size);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops.count("com.microsoft.MatMulNBits"), 0);
    EXPECT_EQ(ops["MatMul"], 1);
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(/*accuracy_level=*/4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

// ---------------------------------------------------------------------------
// Negative: Pattern 2 with non-constant weight
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Negative_Pattern2_NonConstWeight) {
  constexpr int64_t M = 4, N = 8, K = 32, block_size = 16;

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern2NonConstWeight(builder, M, N, K, block_size);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops.count("com.microsoft.MatMulNBits"), 0);
    EXPECT_EQ(ops["DequantizeLinear"], 1);
    EXPECT_EQ(ops["MatMul"], 1);
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(/*accuracy_level=*/4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

// ---------------------------------------------------------------------------
// Negative: Pattern 2 with non-power-of-2 block_size (12)
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Negative_Pattern2_BadBlockSize) {
  constexpr int64_t M = 4, N = 8, K = 48;  // K divisible by 12

  auto build = [&](ModelTestBuilder& builder) {
    BuildPattern2BadBlockSize(builder, M, N, K);
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops.count("com.microsoft.MatMulNBits"), 0);
    EXPECT_EQ(ops["MatMul"], 1);
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(/*accuracy_level=*/4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

// ---------------------------------------------------------------------------
// Negative: Pattern 2 with block_size < 16
// ---------------------------------------------------------------------------
TEST_F(DQMatMulNBitsFusionTest, Negative_Pattern2_SmallBlockSize) {
  constexpr int64_t M = 4, N = 8, K = 32;

  auto build = [&](ModelTestBuilder& builder) {
    const int64_t block_size = 8;  // < 16
    const int64_t k_blocks = K / block_size;
    auto* input_a = builder.MakeInput<float>({M, K}, -1.0f, 1.0f);
    auto* output = builder.MakeOutput();

    std::vector<uint8_t> w_vals(static_cast<size_t>(K * N));
    for (size_t i = 0; i < w_vals.size(); ++i) w_vals[i] = static_cast<uint8_t>(i % 16);
    auto w_packed = MakePackedUint4(w_vals);
    auto* weight_arg = builder.MakeInitializer<UInt4x2>({K, N}, w_packed);

    std::vector<float> s_vals(static_cast<size_t>(k_blocks * N), 0.1f);
    auto* scale_arg = builder.MakeInitializer<float>({k_blocks, N}, s_vals);

    NodeAttributes dq_attrs;
    utils::SetNodeAttribute(utils::MakeAttribute("axis", static_cast<int64_t>(0)), dq_attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("block_size", static_cast<int64_t>(8)), dq_attrs);
    auto* dq_output = builder.MakeIntermediate();
    builder.AddNode("DequantizeLinear", {weight_arg, scale_arg}, {dq_output}, "", &dq_attrs);

    builder.AddNode("MatMul", {input_a, dq_output}, {output});
  };

  auto post_check = [&](Graph& graph) -> Status {
    auto ops = CountOpsInGraphByDomain(graph);
    EXPECT_EQ(ops.count("com.microsoft.MatMulNBits"), 0);
    EXPECT_EQ(ops["MatMul"], 1);
    return Status::OK();
  };

  auto transformer = std::make_unique<DQMatMulNBitsFusion>(/*accuracy_level=*/4);
  ASSERT_STATUS_OK(TestGraphTransformer(build, 21, *logger_, std::move(transformer),
                                        TransformerLevel::Level1, 1, nullptr, post_check));
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(DISABLE_CONTRIB_OPS)
