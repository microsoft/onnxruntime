// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Verifies that the CUTLASS weight-only (fpA_intB) path reproduces the dequantized weight matrix
// exactly, element for element, rather than only approximately.
//
// The trick is to run MatMulNBits with A = I and M = K, so Y[k][n] == dequant(B)[n][k]. Every
// quantity is chosen to be exact in fp16 (unit scales, no zero points), so any difference is a
// layout error, not rounding. This is the only check that pins down the offline weight transform
// -- the LDSM row permutation, the sub-byte transpose, the column interleave and the pair
// interleave -- because a GEMM that silently reuses one K tile still produces plausible numbers
// and passes a tolerance-based comparison on some shapes.
#if !defined(ORT_MINIMAL_BUILD) && defined(USE_CUDA)

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime {
namespace test {
namespace {

// K spans four 64-element threadblock K tiles, which is what caught the B pipeline stalling on
// the first tile. N is a multiple of 128 so 2-bit weights are fpA_intB-eligible.
constexpr int64_t kN = 128, kK = 256, kBlockSize = 128;

void ExpectExactDequantizedWeights(int64_t bits) {
  SCOPED_TRACE("bits=" + std::to_string(bits));
  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FPA_INTB_GEMM", "1"}}};

  const int64_t k_blocks = kK / kBlockSize;
  const int64_t blob_size = kBlockSize * bits / 8;
  const int per_byte = static_cast<int>(8 / bits);
  const int mask = (1 << bits) - 1;
  const int zero_point = 1 << (bits - 1);

  std::vector<int> code(static_cast<size_t>(kN * kK));
  std::vector<uint8_t> b(static_cast<size_t>(kN * kK * bits / 8), 0);
  unsigned seed = 7;
  for (int64_t i = 0; i < kN * kK; ++i) {
    seed = seed * 1103515245u + 12345u;
    int c = static_cast<int>((seed >> 16) & mask);
    code[static_cast<size_t>(i)] = c;
    b[static_cast<size_t>(i / per_byte)] |= static_cast<uint8_t>(c << (bits * (i % per_byte)));
  }

  std::vector<MLFloat16> scales(static_cast<size_t>(kN * k_blocks), MLFloat16(1.0f));
  std::vector<MLFloat16> a(static_cast<size_t>(kK * kK), MLFloat16(0.0f));
  for (int64_t i = 0; i < kK; ++i) a[static_cast<size_t>(i * kK + i)] = MLFloat16(1.0f);

  std::vector<MLFloat16> expected(static_cast<size_t>(kK * kN));
  for (int64_t k = 0; k < kK; ++k) {
    for (int64_t n = 0; n < kN; ++n) {
      expected[static_cast<size_t>(k * kN + n)] =
          MLFloat16(static_cast<float>(code[static_cast<size_t>(n * kK + k)] - zero_point));
    }
  }

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", kK);
  test.AddAttribute<int64_t>("N", kN);
  test.AddAttribute<int64_t>("block_size", kBlockSize);
  test.AddAttribute<int64_t>("bits", bits);
  test.AddAttribute<int64_t>("accuracy_level", static_cast<int64_t>(0));
  test.AddInput<MLFloat16>("A", {kK, kK}, a, false);
  test.AddInput<uint8_t>("B", {kN, k_blocks, blob_size}, b, true);
  test.AddInput<MLFloat16>("scales", {kN, k_blocks}, scales, true);
  test.AddOptionalInputEdge<uint8_t>();
  test.AddOptionalInputEdge<int32_t>();
  test.AddOptionalInputEdge<MLFloat16>();
  test.AddOutput<MLFloat16>("Y", {kK, kN}, expected);
  test.SetOutputAbsErr("Y", 0.0f);
  test.SetOutputRelErr("Y", 0.0f);

  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.emplace_back(DefaultCudaExecutionProvider());
  test.ConfigEps(std::move(eps));
  test.RunWithConfig();
}

}  // namespace

TEST(MatMulNBitsFpAIntBLayout, Int4WeightsRoundTripExactly) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "fpA_intB layout test requires SM80+";
  ExpectExactDequantizedWeights(4);
}

TEST(MatMulNBitsFpAIntBLayout, Int2WeightsRoundTripExactly) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "fpA_intB layout test requires SM80+";
  ExpectExactDequantizedWeights(2);
}

}  // namespace test
}  // namespace onnxruntime

#endif
