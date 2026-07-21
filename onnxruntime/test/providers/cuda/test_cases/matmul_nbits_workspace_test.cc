// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Unit tests for the two-level (partition-time + kernel-instance) MatMulNBits workspace estimation
// pilot (Phase-A memory roadmap, issue microsoft/onnxruntime#29775).
//
// This translation unit holds:
//   1. Pure host-side unit tests (MatMulNBitsWorkspace.Formula* / EffectiveArch* / Eligibility*)
//      that exercise the stateless building blocks and DO NOT strictly require a CUDA device:
//        - ComputeFpAIntBGemmWorkspaceSize : the shared workspace-size formula (single source of truth);
//        - EffectiveFpAIntBWorkspaceSm     : the device_sm/weight_prepacked -> effective-arch mapping;
//        - CheckFpAIntBEligibility         : the shared fpA_intB eligibility decision.
//      They live in this CUDA-only test target because they reach into CUDA-provider internals, but
//      no kernel launch happens. Tests EffectiveArchSelection and Eligibility* are the regression
//      guard against the Level-1 / constructor drift described in the issue.
//   2. GetMatMulNBitsLastComputeWorkspaceBytes(): a small provider-world probe used by the companion
//      end-to-end test (matmul_nbits_e2e_workspace_test.cc) to read the runtime workspace request off
//      a constructed MatMulNBits<MLFloat16> kernel.
//
// This is a plain .cc (host-compiled), NOT a .cu: it contains no device code, and including
// matmul_nbits.h drags in gtest's re2-backed regex support, which the CUDA front-end (nvcc) cannot
// parse. The end-to-end test that actually runs a session lives in a SEPARATE file because that one
// needs the core framework headers, which cannot coexist with the shared-provider bridge headers
// pulled in here (both define the same logging macros / ONNX enums).

#include "gtest/gtest.h"

#if defined(USE_FPA_INTB_GEMM) && USE_FPA_INTB_GEMM

#include <cstring>

// NOTE: do NOT include core/graph/onnx_protobuf.h here. matmul_nbits.h pulls in the CUDA-provider
// shared-provider bridge (provider_api.h), which defines its own copies of the ONNX enums
// (ONNX_NAMESPACE::TensorProto_DataType / AttributeProto_AttributeType). Including the real ONNX
// protobuf headers as well would redefine those enums and fail to compile. The enum constants used
// below (TensorProto_DataType_FLOAT16 etc.) come from provider_api.h via matmul_nbits.h.
#include "contrib_ops/cuda/llm/fpA_intB_gemm/fpA_intB_gemm.h"
#include "contrib_ops/cuda/quantization/matmul_nbits.h"
#include "test/providers/cuda/test_cases/matmul_nbits_workspace_test_probe.h"

namespace onnxruntime {
namespace test {

using onnxruntime::contrib::cuda::CheckFpAIntBEligibility;
using onnxruntime::contrib::cuda::EffectiveFpAIntBWorkspaceSm;
using onnxruntime::contrib::cuda::kMatMulNBitsWeightNotPrepacked;
using onnxruntime::contrib::cuda::kMatMulNBitsWeightPrepackedSm80;
using onnxruntime::contrib::cuda::kMatMulNBitsWeightPrepackedSm90;
using onnxruntime::contrib::cuda::MatMulNBits;
using onnxruntime::llm::kernels::cutlass_kernels::ComputeFpAIntBGemmWorkspaceSize;

namespace {
constexpr int32_t kFp16 = ONNX_NAMESPACE::TensorProto_DataType_FLOAT16;
constexpr int32_t kBf16 = ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16;
constexpr int32_t kFp32 = ONNX_NAMESPACE::TensorProto_DataType_FLOAT;

// A representative eligible configuration: fp16, int4, block_size 32, aligned N/K, unprepacked,
// fpA_intB option ON, SM80. Returns true iff the node is fpA_intB-eligible.
bool CheckDefault(int32_t elem_type = kFp16, int64_t N = 256, int64_t K = 1024,
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
  EXPECT_TRUE(CheckDefault());
  EXPECT_TRUE(CheckDefault(kBf16));
}

TEST(MatMulNBitsWorkspace, EligibilityRejectsFp32) {
  // CUDA registers an FP32 MatMulNBits variant that must never be reported as fpA_intB-eligible.
  EXPECT_FALSE(CheckDefault(kFp32));
}

TEST(MatMulNBitsWorkspace, EligibilityOptionGate) {
  // Unprepacked + option OFF -> not eligible.
  EXPECT_FALSE(CheckDefault(kFp16, 256, 1024, 4, 32, kMatMulNBitsWeightNotPrepacked, false, 80, 0));
  // Prepacked weights force the path ON regardless of the option.
  EXPECT_TRUE(CheckDefault(kFp16, 256, 1024, 4, 64, kMatMulNBitsWeightPrepackedSm80, false, 80, 0));
}

TEST(MatMulNBitsWorkspace, EligibilityRejectsUnsupportedShapesAndAttrs) {
  EXPECT_FALSE(CheckDefault(kFp16, 256, 1024, 4, 32, kMatMulNBitsWeightNotPrepacked, /*g_idx*/ true));
  EXPECT_FALSE(CheckDefault(kFp16, 256, 1024, /*nbits*/ 3));
  EXPECT_FALSE(CheckDefault(kFp16, 256, 1024, 4, /*block_size*/ 256));
  // int4 requires N % 64 == 0.
  EXPECT_FALSE(CheckDefault(kFp16, /*N*/ 100, 1024, 4, 32));
  // K must be a multiple of block_size.
  EXPECT_FALSE(CheckDefault(kFp16, 256, /*K*/ 1000, 4, 32));
  // Device below SM75.
  EXPECT_FALSE(CheckDefault(kFp16, 256, 1024, 4, 32, kMatMulNBitsWeightNotPrepacked, false, /*sm*/ 70));
}

TEST(MatMulNBitsWorkspace, EligibilityInt8Alignment) {
  // int8 requires N % 32 == 0 (looser than int4's N % 64).
  EXPECT_TRUE(CheckDefault(kFp16, /*N*/ 32, 1024, /*nbits*/ 8, 32));
  EXPECT_FALSE(CheckDefault(kFp16, /*N*/ 16, 1024, /*nbits*/ 8, 32));
}

TEST(MatMulNBitsWorkspace, EligibilitySm90PrepackedConstraints) {
  // weight_prepacked=2 requires an SM90 device and block_size in {64,128}.
  EXPECT_FALSE(CheckDefault(kFp16, 256, 1024, 4, 64, kMatMulNBitsWeightPrepackedSm90, false, /*sm*/ 80));
  EXPECT_FALSE(CheckDefault(kFp16, 256, 1024, 4, /*block*/ 32, kMatMulNBitsWeightPrepackedSm90, false, 90));
  EXPECT_TRUE(CheckDefault(kFp16, 256, 1024, 4, /*block*/ 64, kMatMulNBitsWeightPrepackedSm90, false, 90));
  EXPECT_TRUE(CheckDefault(kFp16, 256, 1024, 4, /*block*/ 128, kMatMulNBitsWeightPrepackedSm90, false, 90));
}

// ---------------------------------------------------------------------------
// Provider-world probe (Major 1). The end-to-end test that proves Level 1 == Level 2 == runtime
// lives in matmul_nbits_e2e_workspace_test.cc, which runs a real InferenceSession and therefore
// must include core framework headers. Those headers cannot coexist in one translation unit with
// the CUDA-provider headers pulled in above (the shared-provider bridge redefines logging macros),
// so this small helper is the only piece that needs the concrete MatMulNBits type. It reaches the
// per-kernel runtime workspace record via a narrow declaration in the probe header (no core headers
// here, no provider headers there). MatMulNBits<MLFloat16> is single, non-virtual inheritance from
// OpKernel, so the downcast is a fixed offset; the caller guarantees the kernel is a fp16
// MatMulNBits before calling.
// ---------------------------------------------------------------------------
size_t GetMatMulNBitsLastComputeWorkspaceBytes(const OpKernel* kernel) {
  const auto* mm_kernel = static_cast<const MatMulNBits<MLFloat16>*>(kernel);
  return mm_kernel->LastComputeWorkspaceBytes();
}

}  // namespace test
}  // namespace onnxruntime

#endif  // USE_FPA_INTB_GEMM
