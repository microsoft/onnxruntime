// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the deterministic two-pass Split-K MatMul/Gemm path in the WebGPU EP.
//
// Whether Split-K actually fires is decided by `SplitKConfig`, which is gated on the adapter vendor.
// To exercise this path on an adapter whose vendor table leaves Split-K disabled, run with
// `ORT_WEBGPU_SPLIT_K=on` in the environment; `ORT_WEBGPU_SPLIT_K=off` forces it off. The environment
// is read when the WebGPU context is created, so it must be set before the test process starts, not
// from inside a test.
//
// These tests pass either way — they assert correctness of whatever path is taken — so they are a
// regression net rather than proof that Split-K ran. Proof that Split-K ran comes from the shader dump
// (`ORT_WEBGPU_EP_SHADER_DUMP_FILE`), which must contain `MatMul_Split_K_Reduce`.

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/framework/tensor.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

// Reference 2-D/batched matmul. `batch` slices of [M,K] x [K,N].
void ComputeExpectedMatMul(const std::vector<float>& a_vals, const std::vector<float>& b_vals,
                           int64_t batch, int64_t M, int64_t K, int64_t N,
                           bool b_is_batched, std::vector<float>& out_vals) {
  out_vals.assign(static_cast<size_t>(batch * M * N), 0.0f);
  for (int64_t bi = 0; bi < batch; ++bi) {
    const float* a = a_vals.data() + bi * M * K;
    const float* b = b_vals.data() + (b_is_batched ? bi * K * N : 0);
    float* out = out_vals.data() + bi * M * N;
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (int64_t k = 0; k < K; ++k) {
          sum += a[m * K + k] * b[k * N + n];
        }
        out[m * N + n] = sum;
      }
    }
  }
}

// Compares two buffers by raw bit pattern. Deliberately not an error metric: `-0.0` and `+0.0`, and
// NaNs with different payloads, must not compare equal here.
::testing::AssertionResult BytesIdentical(const std::vector<uint8_t>& lhs, const std::vector<uint8_t>& rhs) {
  if (lhs.size() != rhs.size()) {
    return ::testing::AssertionFailure() << "size mismatch: " << lhs.size() << " vs " << rhs.size();
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (lhs[i] != rhs[i]) {
      return ::testing::AssertionFailure()
             << "first differing byte at offset " << i << ": 0x" << std::hex << static_cast<int>(lhs[i])
             << " vs 0x" << static_cast<int>(rhs[i]);
    }
  }
  return ::testing::AssertionSuccess();
}

// Runs one MatMul on the WebGPU EP and returns the raw bytes of the output tensor. A fresh OpTester
// (and therefore a fresh session) is used per call so that a difference between two calls reflects the
// kernel, not retained state.
std::vector<uint8_t> RunMatMulCaptureBytes(const std::vector<int64_t>& a_dims,
                                           const std::vector<int64_t>& b_dims,
                                           const std::vector<int64_t>& y_dims,
                                           const std::vector<float>& a_vals,
                                           const std::vector<float>& b_vals) {
  std::vector<uint8_t> bytes;

  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    return bytes;
  }

  OpTester test("MatMul", 13);
  test.AddInput<float>("A", a_dims, a_vals);
  test.AddInput<float>("B", b_dims, b_vals);

  int64_t y_size = 1;
  for (int64_t d : y_dims) {
    y_size *= d;
  }
  // Placeholder: the custom verifier below replaces the default comparison.
  test.AddOutput<float>("Y", y_dims, std::vector<float>(static_cast<size_t>(y_size), 0.0f));

  test.SetCustomOutputVerifier([&bytes](const std::vector<OrtValue>& fetches, const std::string&) {
    ASSERT_EQ(fetches.size(), static_cast<size_t>(1));
    const Tensor& tensor = fetches[0].Get<Tensor>();
    const auto* data = static_cast<const uint8_t*>(tensor.DataRaw());
    bytes.assign(data, data + tensor.SizeInBytes());
  });

  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
  return bytes;
}

void RunMatMulAgainstReference(const std::vector<int64_t>& a_dims, const std::vector<int64_t>& b_dims,
                               const std::vector<int64_t>& y_dims,
                               int64_t batch, int64_t M, int64_t K, int64_t N, bool b_is_batched) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  RandomValueGenerator random{1234};
  std::vector<float> a_vals(random.Gaussian<float>(AsSpan(a_dims), 0.0f, 0.25f));
  std::vector<float> b_vals(random.Gaussian<float>(AsSpan(b_dims), 0.0f, 0.25f));

  std::vector<float> expected;
  ComputeExpectedMatMul(a_vals, b_vals, batch, M, K, N, b_is_batched, expected);

  OpTester test("MatMul", 13);
  test.AddInput<float>("A", a_dims, a_vals);
  test.AddInput<float>("B", b_dims, b_vals);
  test.AddOutput<float>("Y", y_dims, expected);
  // K here is up to a few thousand, so a fused-multiply-add on the GPU legitimately differs from the
  // sequential CPU reference by more than the default tolerance.
  test.SetOutputAbsErr("Y", 1e-3f);
  test.SetOutputRelErr("Y", 1e-3f);
  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

}  // namespace

// EfficientNet-B0's classifier shape: M = 1, N = 1000, K = 1280. `dim_inner / split_dim_inner` is
// exactly 5, so every split covers a full 256-wide slice.
TEST(MatMul_SplitK, GemvClassifierShape) {
  RunMatMulAgainstReference({1, 1280}, {1280, 1000}, {1, 1000},
                            /*batch*/ 1, /*M*/ 1, /*K*/ 1280, /*N*/ 1000, /*b_is_batched*/ false);
}

// Squeeze-and-excite shapes: M = 1 after a global average pool, small N.
TEST(MatMul_SplitK, GemvSqueezeExciteShapes) {
  RunMatMulAgainstReference({1, 512}, {512, 20}, {1, 20},
                            /*batch*/ 1, /*M*/ 1, /*K*/ 512, /*N*/ 20, /*b_is_batched*/ false);
  RunMatMulAgainstReference({1, 1024}, {1024, 64}, {1, 64},
                            /*batch*/ 1, /*M*/ 1, /*K*/ 1024, /*N*/ 64, /*b_is_batched*/ false);
}

// `dim_inner` not a multiple of `split_dim_inner` (256): the last split runs past `dim_inner` and its
// out-of-range reads must contribute exactly zero.
TEST(MatMul_SplitK, InnerDimNotDivisibleBySplit) {
  RunMatMulAgainstReference({1, 1000}, {1000, 64}, {1, 64},
                            /*batch*/ 1, /*M*/ 1, /*K*/ 1000, /*N*/ 64, /*b_is_batched*/ false);
  RunMatMulAgainstReference({4, 516}, {516, 32}, {4, 32},
                            /*batch*/ 1, /*M*/ 4, /*K*/ 516, /*N*/ 32, /*b_is_batched*/ false);
}

// Batched B, so `dispatch_z` encodes both the batch index and the split index and the scratch buffer
// is indexed across both. A non-batched B would be folded into M instead.
TEST(MatMul_SplitK, BatchedB) {
  RunMatMulAgainstReference({2, 4, 512}, {2, 512, 64}, {2, 4, 64},
                            /*batch*/ 2, /*M*/ 4, /*K*/ 512, /*N*/ 64, /*b_is_batched*/ true);
  RunMatMulAgainstReference({3, 2, 768}, {3, 768, 128}, {3, 2, 128},
                            /*batch*/ 3, /*M*/ 2, /*K*/ 768, /*N*/ 128, /*b_is_batched*/ true);
}

// The property the two-pass design exists to provide: identical input must produce identical output
// bytes, every run. Compared as raw bit patterns rather than with a tolerance, because the failure
// being guarded against is a reordered floating-point sum, which can differ in the last bit only.
//
// Note this is a weaker check than the one the benchmark harness runs, which compares whole-model
// outputs across separate processes. It catches the common case cheaply.
TEST(MatMul_SplitK, BitReproducibleAcrossRuns) {
  if (!DefaultWebGpuExecutionProvider()) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  const std::vector<int64_t> a_dims{1, 1280};
  const std::vector<int64_t> b_dims{1280, 1000};
  const std::vector<int64_t> y_dims{1, 1000};

  RandomValueGenerator random{4321};
  std::vector<float> a_vals(random.Gaussian<float>(AsSpan(a_dims), 0.0f, 0.25f));
  std::vector<float> b_vals(random.Gaussian<float>(AsSpan(b_dims), 0.0f, 0.25f));

  const std::vector<uint8_t> first = RunMatMulCaptureBytes(a_dims, b_dims, y_dims, a_vals, b_vals);
  ASSERT_FALSE(first.empty()) << "No output captured from the first run.";

  constexpr int kRepeats = 4;
  for (int i = 1; i <= kRepeats; ++i) {
    const std::vector<uint8_t> again = RunMatMulCaptureBytes(a_dims, b_dims, y_dims, a_vals, b_vals);
    ASSERT_FALSE(again.empty()) << "No output captured from run " << i << ".";
    EXPECT_TRUE(BytesIdentical(first, again)) << "Run " << i << " is not bit-identical to run 0.";
  }
}

// Sanity check on the comparison itself: it must reject a single flipped mantissa bit. Without this,
// a passing `BitReproducibleAcrossRuns` is ambiguous between "deterministic" and "the comparison
// cannot detect a difference".
TEST(MatMul_SplitK, BitComparisonRejectsSingleBitDifference) {
  std::vector<uint8_t> lhs(64, 0);
  std::vector<uint8_t> rhs(64, 0);
  EXPECT_TRUE(BytesIdentical(lhs, rhs));

  rhs[17] ^= 0x01;
  EXPECT_FALSE(BytesIdentical(lhs, rhs));

  // Positive and negative zero share every bit but the sign, and must not compare equal.
  const float pos_zero = 0.0f;
  const float neg_zero = -0.0f;
  std::vector<uint8_t> pos(sizeof(float));
  std::vector<uint8_t> neg(sizeof(float));
  std::memcpy(pos.data(), &pos_zero, sizeof(float));
  std::memcpy(neg.data(), &neg_zero, sizeof(float));
  EXPECT_FALSE(BytesIdentical(pos, neg));

  // NaNs with different payloads must not compare equal either.
  const uint32_t nan_a = 0x7FC00000u;
  const uint32_t nan_b = 0x7FC00001u;
  std::vector<uint8_t> a(sizeof(uint32_t));
  std::vector<uint8_t> b(sizeof(uint32_t));
  std::memcpy(a.data(), &nan_a, sizeof(uint32_t));
  std::memcpy(b.data(), &nan_b, sizeof(uint32_t));
  EXPECT_FALSE(BytesIdentical(a, b));
}

// Gemm reaches Split-K through `ApplyGemmPacked`, which uses a separate program with its own uniform
// set and no batch dimension. `beta * C` is applied by the reduction rather than by a pre-fill.
TEST(Gemm_SplitK, WithAndWithoutBias) {
  auto make_ep = []() { return DefaultWebGpuExecutionProvider(); };
  if (!make_ep()) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  constexpr int64_t M = 4;
  constexpr int64_t K = 1024;
  constexpr int64_t N = 64;

  const std::vector<int64_t> a_dims{M, K};
  const std::vector<int64_t> b_dims{K, N};
  const std::vector<int64_t> c_dims{N};
  const std::vector<int64_t> y_dims{M, N};

  RandomValueGenerator random{99};
  std::vector<float> a_vals(random.Gaussian<float>(AsSpan(a_dims), 0.0f, 0.25f));
  std::vector<float> b_vals(random.Gaussian<float>(AsSpan(b_dims), 0.0f, 0.25f));
  std::vector<float> c_vals(random.Gaussian<float>(AsSpan(c_dims), 0.0f, 0.25f));

  std::vector<float> matmul_only;
  ComputeExpectedMatMul(a_vals, b_vals, /*batch*/ 1, M, K, N, /*b_is_batched*/ false, matmul_only);

  // No bias.
  {
    OpTester test("Gemm", 13);
    test.AddInput<float>("A", a_dims, a_vals);
    test.AddInput<float>("B", b_dims, b_vals);
    test.AddOutput<float>("Y", y_dims, matmul_only);
    test.SetOutputAbsErr("Y", 1e-3f);
    test.SetOutputRelErr("Y", 1e-3f);
    test.ConfigEp(make_ep()).RunWithConfig();
  }

  // Bias broadcast along N, beta = 1.
  {
    std::vector<float> expected(matmul_only);
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        expected[static_cast<size_t>(m * N + n)] += c_vals[static_cast<size_t>(n)];
      }
    }

    OpTester test("Gemm", 13);
    test.AddInput<float>("A", a_dims, a_vals);
    test.AddInput<float>("B", b_dims, b_vals);
    test.AddInput<float>("C", c_dims, c_vals);
    test.AddOutput<float>("Y", y_dims, expected);
    test.SetOutputAbsErr("Y", 1e-3f);
    test.SetOutputRelErr("Y", 1e-3f);
    test.ConfigEp(make_ep()).RunWithConfig();
  }
}

}  // namespace test
}  // namespace onnxruntime
