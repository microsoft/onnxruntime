// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Unit tests for the two-level (partition-time + kernel-instance) MatMulNBits workspace estimation
// pilot (Phase-A memory roadmap, issue microsoft/onnxruntime#29775). These exercise the pure,
// stateless building blocks that do NOT require a CUDA device:
//   - ComputeFpAIntBGemmWorkspaceSize : the shared workspace-size formula (single source of truth);
//   - EffectiveFpAIntBWorkspaceSm     : the device_sm/weight_prepacked -> effective-arch mapping;
//   - CheckFpAIntBEligibility         : the shared fpA_intB eligibility decision.
// Tests A (EffectiveFpAIntBWorkspaceSm) and C (CheckFpAIntBEligibility) are the regression guard
// against the Level-1 / constructor drift described in the issue.

#include "gtest/gtest.h"

#if defined(USE_FPA_INTB_GEMM) && USE_FPA_INTB_GEMM

#include "core/graph/onnx_protobuf.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm/fpA_intB_gemm.h"
#include "contrib_ops/cuda/quantization/matmul_nbits.h"

namespace onnxruntime {
namespace test {

using onnxruntime::contrib::cuda::CheckFpAIntBEligibility;
using onnxruntime::contrib::cuda::EffectiveFpAIntBWorkspaceSm;
using onnxruntime::contrib::cuda::FpAIntBEligibility;
using onnxruntime::contrib::cuda::kMatMulNBitsWeightNotPrepacked;
using onnxruntime::contrib::cuda::kMatMulNBitsWeightPrepackedSm80;
using onnxruntime::contrib::cuda::kMatMulNBitsWeightPrepackedSm90;
using onnxruntime::llm::kernels::cutlass_kernels::ComputeFpAIntBGemmWorkspaceSize;

namespace {
constexpr int32_t kFp16 = ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
constexpr int32_t kBf16 = ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
constexpr int32_t kFp32 = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;

// A representative eligible configuration: fp16, int4, block_size 32, aligned N/K, unprepacked,
// fpA_intB option ON, SM80.
FpAIntBEligibility CheckDefault(int32_t elem_type = kFp16, int64_t N = 256, int64_t K = 1024,
                                int64_t nbits = 4, int64_t block_size = 32,
                                int64_t weight_prepacked = kMatMulNBitsWeightNotPrepacked,
                                bool has_g_idx = false, int device_sm = 80, int option = 1) {
  return CheckFpAIntBEligibility(elem_type, N, K, nbits, block_size, weight_prepacked, has_g_idx,
                                 device_sm, option);
}
}  // namespace

// ---------------------------------------------------------------------------
// Shared workspace-size formula (ComputeFpAIntBGemmWorkspaceSize).
// The non-SM90 branch is device-independent so it can be asserted with exact values.
// ---------------------------------------------------------------------------
TEST(MatMulNBitsWorkspace, FormulaNonSm90ExactValues) {
  // ceil_div(m, 16) * ceil_div(n, 64) * SPLIT_K_LIMIT(7) * 4 bytes.
  // m=16, n=64 -> 1 * 1 * 7 * 4 = 28.
  auto ws = ComputeFpAIntBGemmWorkspaceSize(/*m=*/16, /*n=*/64, /*k=*/1024, /*sm=*/80, /*mpc=*/100);
  ASSERT_TRUE(ws.has_value());
  EXPECT_EQ(*ws, static_cast<size_t>(28));

  // m=17, n=65 -> ceil(17/16)=2, ceil(65/64)=2 -> 2*2*7*4 = 112.
  ws = ComputeFpAIntBGemmWorkspaceSize(17, 65, 0, 80, 100);
  ASSERT_TRUE(ws.has_value());
  EXPECT_EQ(*ws, static_cast<size_t>(112));

  // k is unused by the formula: two different k values must give the same result.
  EXPECT_EQ(ComputeFpAIntBGemmWorkspaceSize(128, 512, 1, 80, 8),
            ComputeFpAIntBGemmWorkspaceSize(128, 512, 999999, 80, 8));
}

#ifndef EXCLUDE_SM_90
TEST(MatMulNBitsWorkspace, FormulaSm90DependsOnMultiProcessorCount) {
  // sk_tiles = 2*mpc; sk_units = mpc; scaled = 2*(2*mpc)+mpc = 5*mpc.
  // bytes = 5*mpc * MAX_M_TILE_SM90(128) * MAX_N_TILE_SM90(256) * sizeof(float)(4).
  const size_t mpc = 10;
  const size_t expected = 5 * mpc * 128 * 256 * 4;  // 6,553,600
  auto ws = ComputeFpAIntBGemmWorkspaceSize(/*m=*/512, /*n=*/512, /*k=*/1, /*sm=*/90,
                                            static_cast<int>(mpc));
  ASSERT_TRUE(ws.has_value());
  EXPECT_EQ(*ws, expected);

  // On SM90 the result is independent of m and n (only mpc drives it).
  EXPECT_EQ(ComputeFpAIntBGemmWorkspaceSize(1, 1, 0, 90, 20),
            ComputeFpAIntBGemmWorkspaceSize(9999, 9999, 0, 90, 20));
}
#endif

TEST(MatMulNBitsWorkspace, FormulaReturnsNulloptOnInvalidNegativeDim) {
  // A negative dimension cannot be represented as an unsigned size and must yield nullopt (not throw).
  EXPECT_FALSE(ComputeFpAIntBGemmWorkspaceSize(-1, 64, 0, 80, 100).has_value());
}

// ---------------------------------------------------------------------------
// Test A: EffectiveFpAIntBWorkspaceSm drift guard.
// Native SM90 arch only when device is SM90 AND weights were prepacked for the Hopper layout.
// ---------------------------------------------------------------------------
TEST(MatMulNBitsWorkspace, EffectiveArchSelection) {
  EXPECT_EQ(EffectiveFpAIntBWorkspaceSm(90, kMatMulNBitsWeightPrepackedSm90), 90);
  EXPECT_EQ(EffectiveFpAIntBWorkspaceSm(90, kMatMulNBitsWeightPrepackedSm80), 80);
  EXPECT_EQ(EffectiveFpAIntBWorkspaceSm(90, kMatMulNBitsWeightNotPrepacked), 80);
  EXPECT_EQ(EffectiveFpAIntBWorkspaceSm(80, kMatMulNBitsWeightPrepackedSm90), 80);
  EXPECT_EQ(EffectiveFpAIntBWorkspaceSm(80, kMatMulNBitsWeightNotPrepacked), 80);
  EXPECT_EQ(EffectiveFpAIntBWorkspaceSm(75, kMatMulNBitsWeightNotPrepacked), 80);
}

// ---------------------------------------------------------------------------
// Test C: CheckFpAIntBEligibility single-source-of-truth guard.
// ---------------------------------------------------------------------------
TEST(MatMulNBitsWorkspace, EligibilityBasic) {
  EXPECT_TRUE(CheckDefault().eligible);
  EXPECT_TRUE(CheckDefault(kBf16).eligible);
}

TEST(MatMulNBitsWorkspace, EligibilityRejectsFp32) {
  // CUDA registers an FP32 MatMulNBits variant that must never be reported as fpA_intB-eligible.
  EXPECT_FALSE(CheckDefault(kFp32).eligible);
}

TEST(MatMulNBitsWorkspace, EligibilityOptionGate) {
  // Unprepacked + option OFF -> not eligible.
  EXPECT_FALSE(CheckDefault(kFp16, 256, 1024, 4, 32, kMatMulNBitsWeightNotPrepacked, false, 80, 0)
                   .eligible);
  // Prepacked weights force the path ON regardless of the option.
  EXPECT_TRUE(CheckDefault(kFp16, 256, 1024, 4, 64, kMatMulNBitsWeightPrepackedSm80, false, 80, 0)
                  .eligible);
}

TEST(MatMulNBitsWorkspace, EligibilityRejectsUnsupportedShapesAndAttrs) {
  EXPECT_FALSE(CheckDefault(kFp16, 256, 1024, 4, 32, kMatMulNBitsWeightNotPrepacked, /*g_idx*/ true)
                   .eligible);
  EXPECT_FALSE(CheckDefault(kFp16, 256, 1024, /*nbits*/ 3).eligible);
  EXPECT_FALSE(CheckDefault(kFp16, 256, 1024, 4, /*block_size*/ 256).eligible);
  // int4 requires N % 64 == 0.
  EXPECT_FALSE(CheckDefault(kFp16, /*N*/ 100, 1024, 4, 32).eligible);
  // K must be a multiple of block_size.
  EXPECT_FALSE(CheckDefault(kFp16, 256, /*K*/ 1000, 4, 32).eligible);
  // Device below SM75.
  EXPECT_FALSE(CheckDefault(kFp16, 256, 1024, 4, 32, kMatMulNBitsWeightNotPrepacked, false, /*sm*/ 70)
                   .eligible);
}

TEST(MatMulNBitsWorkspace, EligibilityInt8Alignment) {
  // int8 requires N % 32 == 0 (looser than int4's N % 64).
  EXPECT_TRUE(CheckDefault(kFp16, /*N*/ 32, 1024, /*nbits*/ 8, 32).eligible);
  EXPECT_FALSE(CheckDefault(kFp16, /*N*/ 16, 1024, /*nbits*/ 8, 32).eligible);
}

TEST(MatMulNBitsWorkspace, EligibilitySm90PrepackedConstraints) {
  // weight_prepacked=2 requires an SM90 device and block_size in {64,128}.
  EXPECT_FALSE(CheckDefault(kFp16, 256, 1024, 4, 64, kMatMulNBitsWeightPrepackedSm90, false, /*sm*/ 80)
                   .eligible);
  EXPECT_FALSE(CheckDefault(kFp16, 256, 1024, 4, /*block*/ 32, kMatMulNBitsWeightPrepackedSm90, false, 90)
                   .eligible);
  EXPECT_TRUE(CheckDefault(kFp16, 256, 1024, 4, /*block*/ 64, kMatMulNBitsWeightPrepackedSm90, false, 90)
                  .eligible);
  EXPECT_TRUE(CheckDefault(kFp16, 256, 1024, 4, /*block*/ 128, kMatMulNBitsWeightPrepackedSm90, false, 90)
                  .eligible);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // USE_FPA_INTB_GEMM
