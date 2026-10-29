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
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime {
namespace test {
namespace {

// K spans four 64-element threadblock K tiles, which is what caught the B pipeline stalling on
// the first tile. N is a multiple of 128 so 2-bit weights are fpA_intB-eligible.
constexpr int64_t kN = 128, kK = 256;
// A = I makes M = K, so 512 reaches the large-M INT8 path (dequantize + one cuBLAS GEMM, see
// ep.cuda.fpa_intb_dequant_gemm_min_m).
constexpr int64_t kLargeMK = 512;

// dequant_gemm_min_m is the ep.cuda.fpa_intb_dequant_gemm_min_m session option; nullptr leaves it unset. A non-null
// expected_error expects session creation to fail with that message.
void ExpectExactDequantizedWeights(int64_t bits, bool deterministic = false, int64_t k_dim = kK,
                                   const char* dequant_gemm_min_m = "512", const char* expected_error = nullptr) {
  SCOPED_TRACE("bits=" + std::to_string(bits) + " K=" + std::to_string(k_dim) + " dequant_gemm_min_m=" +
               (dequant_gemm_min_m != nullptr ? dequant_gemm_min_m : "<unset>"));
  ScopedEnvironmentVariables scoped_env_vars{EnvVarMap{{"ORT_FPA_INTB_GEMM", "1"}}};

  const int64_t block_size = bits == 2 ? 64 : 32;
  const int64_t k_blocks = k_dim / block_size;
  const int64_t blob_size = block_size * bits / 8;
  const int per_byte = static_cast<int>(8 / bits);
  const int mask = (1 << bits) - 1;
  const int zero_point = 1 << (bits - 1);

  std::vector<int> code(static_cast<size_t>(kN * k_dim));
  std::vector<uint8_t> b(static_cast<size_t>(kN * k_dim * bits / 8), 0);
  unsigned seed = 7;
  for (int64_t i = 0; i < kN * k_dim; ++i) {
    seed = seed * 1103515245u + 12345u;
    int c = static_cast<int>((seed >> 16) & mask);
    code[static_cast<size_t>(i)] = c;
    b[static_cast<size_t>(i / per_byte)] |= static_cast<uint8_t>(c << (bits * (i % per_byte)));
  }

  std::vector<MLFloat16> scales(static_cast<size_t>(kN * k_blocks), MLFloat16(1.0f));
  std::vector<MLFloat16> a(static_cast<size_t>(k_dim * k_dim), MLFloat16(0.0f));
  for (int64_t i = 0; i < k_dim; ++i) a[static_cast<size_t>(i * k_dim + i)] = MLFloat16(1.0f);

  std::vector<MLFloat16> expected(static_cast<size_t>(k_dim * kN));
  for (int64_t k = 0; k < k_dim; ++k) {
    for (int64_t n = 0; n < kN; ++n) {
      expected[static_cast<size_t>(k * kN + n)] =
          MLFloat16(static_cast<float>(code[static_cast<size_t>(n * k_dim + k)] - zero_point));
    }
  }

  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", k_dim);
  test.AddAttribute<int64_t>("N", kN);
  test.AddAttribute<int64_t>("block_size", block_size);
  test.AddAttribute<int64_t>("bits", bits);
  test.AddAttribute<int64_t>("accuracy_level", static_cast<int64_t>(0));
  test.AddInput<MLFloat16>("A", {k_dim, k_dim}, a, false);
  test.AddInput<uint8_t>("B", {kN, k_blocks, blob_size}, b, true);
  test.AddInput<MLFloat16>("scales", {kN, k_blocks}, scales, true);
  test.AddOptionalInputEdge<uint8_t>();
  test.AddOptionalInputEdge<int32_t>();
  test.AddOptionalInputEdge<MLFloat16>();
  test.AddOutput<MLFloat16>("Y", {k_dim, kN}, expected);
  test.SetOutputAbsErr("Y", 0.0f);
  test.SetOutputRelErr("Y", 0.0f);

  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.emplace_back(DefaultCudaExecutionProvider());
  SessionOptions session_options;
  session_options.use_per_session_threads = false;
  session_options.use_deterministic_compute = deterministic;
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  if (dequant_gemm_min_m != nullptr) {
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsCudaFpAIntBDequantGemmMinM,
                                                                   dequant_gemm_min_m));
  }
  test.Config(session_options);
  test.ConfigEps(std::move(eps));
  if (expected_error != nullptr) {
    test.Config(OpTester::ExpectResult::kExpectFailure, expected_error);
  }
  test.RunWithConfig();
}

}  // namespace

TEST(MatMulNBitsFpAIntBLayout, TensorCoreSessionOptions) {
  if (!HasCudaEnvironment(900)) GTEST_SKIP() << "Tensor-core INT8 GEMV requires SM90";
  for (const char* disable : {"0", "1"}) {
    ScopedEnvironmentVariables env{EnvVarMap{{"ORT_DISABLE_FPA_INTB_GEMV_TC", "invalid"}}};
    OpTester test("MatMulNBits", 1, kMSDomain);
    test.AddAttribute<int64_t>("K", kK);
    test.AddAttribute<int64_t>("N", kN);
    test.AddAttribute<int64_t>("block_size", 32);
    test.AddAttribute<int64_t>("bits", 8);
    std::vector<BFloat16> a(8 * kK, BFloat16(0.0f));
    for (int i = 0; i < 8; ++i) a[i * kK + i] = BFloat16(1.0f);
    test.AddInput<BFloat16>("A", {8, kK}, a);
    test.AddInput<uint8_t>("B", {kN, kK / 32, 32}, std::vector<uint8_t>(kN * kK, 129), true);
    test.AddInput<BFloat16>("scales", {kN, kK / 32}, std::vector<BFloat16>(kN * kK / 32, BFloat16(1.0f)), true);
    test.AddOutput<BFloat16>("Y", {8, kN}, std::vector<BFloat16>(8 * kN, BFloat16(1.0f)));
    test.SetOutputTolerance(0.0f, 0.0f);
    SessionOptions options;
    options.use_deterministic_compute = true;
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("ep.cuda.fpa_intb_gemm", "1"));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry("ep.cuda.disable_fpa_intb_gemv_tc", disable));
    ASSERT_STATUS_OK(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.emplace_back(DefaultCudaExecutionProvider());
    test.Run(options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  }
}

TEST(MatMulNBitsFpAIntBLayout, Int4WeightsRoundTripExactly) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "fpA_intB layout test requires SM80+";
  ExpectExactDequantizedWeights(4);
}

TEST(MatMulNBitsFpAIntBLayout, Int2WeightsRoundTripExactly) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "fpA_intB layout test requires SM80+";
  ExpectExactDequantizedWeights(2);
}

TEST(MatMulNBitsFpAIntBLayout, DeterministicWeightsRoundTripExactly) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "fpA_intB layout test requires SM80+";
  for (int64_t bits : {2, 4, 8}) {
    ExpectExactDequantizedWeights(bits, true);
  }
}

// K = M = 512: eight 64-wide K tiles through the dequantize + cuBLAS path, and through the row-chunked CUTLASS path
// (threshold 0 turns the large-M path off). The threshold is set explicitly so that the path is also covered on
// architectures where it is off by default.
TEST(MatMulNBitsFpAIntBLayout, Int8LargeMDequantGemmRoundTripsExactly) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "fpA_intB layout test requires SM80+";
  ExpectExactDequantizedWeights(8, false, kLargeMK, "512");
  ExpectExactDequantizedWeights(8, false, kLargeMK, "0");
}

// The session option takes precedence over the environment, which is consulted only when no entry is set; an invalid
// value from either source fails kernel construction.
TEST(MatMulNBitsFpAIntBLayout, Int8LargeMDequantGemmSessionOption) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "fpA_intB layout test requires SM80+";
  constexpr const char* kEnv = "ORT_FPA_INTB_DEQUANT_GEMM_MIN_M";
  {
    ScopedEnvironmentVariables env{EnvVarMap{{kEnv, "invalid"}}};
    ExpectExactDequantizedWeights(8, false, kLargeMK, "512");
    ExpectExactDequantizedWeights(8, false, kLargeMK, "0");
    ExpectExactDequantizedWeights(8, false, kLargeMK, nullptr, kEnv);
  }
  for (const char* env_value : {"0", "512"}) {
    ScopedEnvironmentVariables env{EnvVarMap{{kEnv, env_value}}};
    ExpectExactDequantizedWeights(8, false, kLargeMK, nullptr);
  }
  for (const char* invalid : {"-1", "", "1x", "many", "2147483648"}) {
    SCOPED_TRACE(std::string("value='") + invalid + "'");
    ExpectExactDequantizedWeights(8, false, kLargeMK, invalid, kOrtSessionOptionsCudaFpAIntBDequantGemmMinM);
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif
