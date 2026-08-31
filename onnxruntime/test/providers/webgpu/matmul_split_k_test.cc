// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the deterministic two-pass Split-K MatMul/Gemm path in the WebGPU EP.
//
// `SplitKConfig` gates Split-K on the adapter vendor. Set `ORT_WEBGPU_SPLIT_K=on` to exercise the
// path on an adapter that would otherwise leave it disabled, or `off` to force it off. The value is
// read when the WebGPU context is created, so it must be set before the process starts.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

#include "gtest/gtest.h"

#include "core/framework/tensor.h"
#include "core/graph/constants.h"
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

std::string ShapeToString(const std::vector<int64_t>& dims) {
  std::string s = "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i > 0) {
      s += ",";
    }
    s += std::to_string(dims[i]);
  }
  return s + "]";
}

// A is [M,K] or [batch,M,K]; B is [K,N] or [batch,K,N]. The logical dimensions are derived from the
// shapes rather than passed in, so the reference cannot be computed for a different shape than the one
// the EP is given.
template <typename T>
void RunMatMulAgainstReferenceTyped(const std::vector<int64_t>& a_dims, const std::vector<int64_t>& b_dims,
                                    const std::vector<int64_t>& y_dims) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, MLFloat16>, "unexpected type for T");

  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  SCOPED_TRACE("A=" + ShapeToString(a_dims) + " B=" + ShapeToString(b_dims));
  ASSERT_GE(a_dims.size(), static_cast<size_t>(2));
  ASSERT_GE(b_dims.size(), static_cast<size_t>(2));

  const int64_t K = a_dims.back();
  const int64_t M = a_dims[a_dims.size() - 2];
  const int64_t N = b_dims.back();
  const int64_t batch = a_dims.size() > 2 ? a_dims[0] : 1;
  const bool b_is_batched = b_dims.size() > 2;
  ASSERT_EQ(b_dims[b_dims.size() - 2], K) << "B's inner dimension must match A's.";

  RandomValueGenerator random{1234};
  std::vector<float> a_vals(random.Gaussian<float>(AsSpan(a_dims), 0.0f, 0.25f));
  std::vector<float> b_vals(random.Gaussian<float>(AsSpan(b_dims), 0.0f, 0.25f));

  std::vector<float> expected;
  ComputeExpectedMatMul(a_vals, b_vals, batch, M, K, N, b_is_batched, expected);

  OpTester test("MatMul", 13);
  if constexpr (std::is_same_v<T, float>) {
    test.AddInput<T>("A", a_dims, a_vals);
    test.AddInput<T>("B", b_dims, b_vals);
    test.AddOutput<T>("Y", y_dims, expected);
    // K here is up to a few thousand, so a fused-multiply-add on the GPU legitimately differs from the
    // sequential CPU reference by more than the default tolerance.
    test.SetOutputAbsErr("Y", 1e-3f);
    test.SetOutputRelErr("Y", 1e-3f);
  } else {
    test.AddInput<T>("A", a_dims, FloatsToMLFloat16s(a_vals));
    test.AddInput<T>("B", b_dims, FloatsToMLFloat16s(b_vals));
    test.AddOutput<T>("Y", y_dims, FloatsToMLFloat16s(expected));
    // Inputs and partials are f16 while the reference sums in f32, so the tolerance is set by the
    // input rounding rather than by the reduction.
    test.SetOutputAbsErr("Y", 0.055f);
    test.SetOutputRelErr("Y", 0.02f);
  }
  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

void RunMatMulAgainstReference(const std::vector<int64_t>& a_dims, const std::vector<int64_t>& b_dims,
                               const std::vector<int64_t>& y_dims) {
  RunMatMulAgainstReferenceTyped<float>(a_dims, b_dims, y_dims);
}

// A fused activation reaches ComputeMatMul only through Conv's 1x1 path, so the activation gate is
// exercised through FusedConv rather than a bare MatMul. Squeeze-and-excite shape: many channels in,
// few out, which is what the rate gate selects for.
void RunFusedConvAgainstReference(const char* activation_name,
                                  const std::vector<float>& activation_params,
                                  const std::function<float(float)>& apply) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  constexpr int64_t kIn = 1024;
  constexpr int64_t kOut = 16;
  const std::vector<int64_t> x_dims{1, kIn, 1, 1};
  const std::vector<int64_t> w_dims{kOut, kIn, 1, 1};

  RandomValueGenerator random{1234};
  std::vector<float> x_vals(random.Gaussian<float>(AsSpan(x_dims), 0.0f, 0.25f));
  std::vector<float> w_vals(random.Gaussian<float>(AsSpan(w_dims), 0.0f, 0.25f));

  std::vector<float> expected(kOut);
  for (int64_t o = 0; o < kOut; ++o) {
    float acc = 0.0f;
    for (int64_t c = 0; c < kIn; ++c) {
      acc += x_vals[static_cast<size_t>(c)] * w_vals[static_cast<size_t>(o * kIn + c)];
    }
    expected[static_cast<size_t>(o)] = apply(acc);
  }

  OpTester test("FusedConv", 1, kMSDomain);
  test.AddAttribute("activation", activation_name);
  test.AddAttribute("kernel_shape", std::vector<int64_t>{1, 1});
  if (!activation_params.empty()) {
    test.AddAttribute("activation_params", activation_params);
  }
  test.AddInput<float>("X", x_dims, x_vals);
  test.AddInput<float>("W", w_dims, w_vals);
  test.AddOutput<float>("Y", {1, kOut, 1, 1}, expected);
  test.SetOutputAbsErr("Y", 1e-3f);
  test.SetOutputRelErr("Y", 1e-3f);
  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

}  // namespace

// The activation must be applied to the completed sum, not per split. That is only distinguishable
// when some partial sums differ in sign from the total, which the Gaussian inputs provide.
TEST(MatMul_SplitK, FusedConvReluAppliedAfterReduction) {
  RunFusedConvAgainstReference("Relu", {}, [](float v) { return v > 0.0f ? v : 0.0f; });
}

// Alpha 1 is SiLU, the only QuickGelu the gate admits.
TEST(MatMul_SplitK, FusedConvSiluAppliedAfterReduction) {
  RunFusedConvAgainstReference("QuickGelu", {1.0f},
                               [](float v) { return v / (1.0f + std::exp(-v)); });
}

// EfficientNet-B0's classifier shape: M = 1, N = 1000, K = 1280. `dim_inner / split_dim_inner` is
// exactly 5, so every split covers a full 256-wide slice.
TEST(MatMul_SplitK, GemvClassifierShape) {
  RunMatMulAgainstReference({1, 1280}, {1280, 1000}, {1, 1000});
}

// Squeeze-and-excite shapes: M = 1 after a global average pool, small N.
TEST(MatMul_SplitK, GemvSqueezeExciteShapes) {
  RunMatMulAgainstReference({1, 512}, {512, 20}, {1, 20});
  RunMatMulAgainstReference({1, 1024}, {1024, 64}, {1, 64});
}

// `dim_inner` not a multiple of `split_dim_inner` (256): the last split runs past `dim_inner` and its
// out-of-range reads must contribute exactly zero.
TEST(MatMul_SplitK, InnerDimNotDivisibleBySplit) {
  RunMatMulAgainstReference({1, 1000}, {1000, 64}, {1, 64});
  RunMatMulAgainstReference({4, 516}, {516, 32}, {4, 32});
}

// Batched B, so `dispatch_z` encodes both the batch index and the split index and the scratch buffer
// is indexed across both. A non-batched B would be folded into M instead.
TEST(MatMul_SplitK, BatchedB) {
  RunMatMulAgainstReference({2, 4, 512}, {2, 512, 64}, {2, 4, 64});
  RunMatMulAgainstReference({3, 2, 768}, {3, 768, 128}, {3, 2, 128});
}

// Pass 1 stores `vec4<f16>` partials and pass 2 accumulates them in f32 before a single narrowing
// write, so f16 takes a different path through the reduction than f32.
TEST(MatMul_SplitK, GemvClassifierShapeFloat16) {
  RunMatMulAgainstReferenceTyped<MLFloat16>({1, 1280}, {1280, 1000}, {1, 1000});
}

// The reduction's `has_bias && !is_gemm` branch is only reachable through Conv, since ONNX MatMul has
// no bias input. Conv's 1x1 fast path forwards its bias into `ComputeMatMul`. With an activation the
// reduction must compute `act(sum + b)`; the two are distinguishable because Relu is not additive.
void RunConv1x1BiasAgainstReference(const char* activation_name,
                                    const std::function<float(float)>& apply) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  SCOPED_TRACE(activation_name == nullptr ? "no activation" : activation_name);

  constexpr int64_t kInChannels = 512;  // dim_inner
  constexpr int64_t kOutChannels = 64;  // dim_b_outer
  constexpr int64_t kHeight = 4;
  constexpr int64_t kWidth = 4;  // dim_a_outer = kHeight * kWidth
  constexpr int64_t kSpatial = kHeight * kWidth;

  const std::vector<int64_t> x_dims{1, kInChannels, kHeight, kWidth};
  const std::vector<int64_t> w_dims{kOutChannels, kInChannels, 1, 1};
  const std::vector<int64_t> b_dims{kOutChannels};
  const std::vector<int64_t> y_dims{1, kOutChannels, kHeight, kWidth};

  RandomValueGenerator random{7};
  std::vector<float> x_vals(random.Gaussian<float>(AsSpan(x_dims), 0.0f, 0.25f));
  std::vector<float> w_vals(random.Gaussian<float>(AsSpan(w_dims), 0.0f, 0.25f));
  std::vector<float> b_vals(random.Gaussian<float>(AsSpan(b_dims), 0.0f, 0.25f));

  // A 1x1 convolution is a matmul over the channel axis, plus the bias once per output channel.
  std::vector<float> expected(static_cast<size_t>(kOutChannels * kSpatial));
  for (int64_t m = 0; m < kOutChannels; ++m) {
    for (int64_t p = 0; p < kSpatial; ++p) {
      float sum = 0.0f;
      for (int64_t c = 0; c < kInChannels; ++c) {
        sum += w_vals[static_cast<size_t>(m * kInChannels + c)] * x_vals[static_cast<size_t>(c * kSpatial + p)];
      }
      expected[static_cast<size_t>(m * kSpatial + p)] = apply(sum + b_vals[static_cast<size_t>(m)]);
    }
  }

  const bool fused = activation_name != nullptr;
  OpTester test(fused ? "FusedConv" : "Conv", fused ? 1 : 11, fused ? kMSDomain : kOnnxDomain);
  if (fused) {
    test.AddAttribute("activation", activation_name);
  }
  test.AddAttribute("group", static_cast<int64_t>(1));
  test.AddAttribute("kernel_shape", std::vector<int64_t>{1, 1});
  test.AddAttribute("pads", std::vector<int64_t>{0, 0, 0, 0});
  test.AddAttribute("strides", std::vector<int64_t>{1, 1});
  test.AddInput<float>("X", x_dims, x_vals);
  test.AddInput<float>("W", w_dims, w_vals);
  test.AddInput<float>("B", b_dims, b_vals);
  test.AddOutput<float>("Y", y_dims, expected);
  test.SetOutputAbsErr("Y", 1e-3f);
  test.SetOutputRelErr("Y", 1e-3f);
  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

TEST(MatMul_SplitK, Conv1x1BiasAppliedOnceByReduction) {
  RunConv1x1BiasAgainstReference(nullptr, [](float v) { return v; });
}

// Bias and activation together. `act(sum + b)` and `act(sum) + b` differ, and only the first is
// correct, so this guards the order the reduction applies them in.
TEST(MatMul_SplitK, Conv1x1BiasThenActivation) {
  RunConv1x1BiasAgainstReference("Relu", [](float v) { return v > 0.0f ? v : 0.0f; });
}

// Identical input must produce identical output bytes. Compared as raw bit patterns rather than with
// a tolerance, because a reordered floating-point sum can differ in the last bit only.
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

// Guards the comparator itself: `BytesIdentical` must reject a flipped mantissa bit, and must treat
// `-0.0` and differing NaN payloads as distinct rather than equal.
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
// set and no batch dimension. `beta * C` is applied by the reduction.
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
  const std::vector<int64_t> y_dims{M, N};

  RandomValueGenerator random{99};
  std::vector<float> a_vals(random.Gaussian<float>(AsSpan(a_dims), 0.0f, 0.25f));
  std::vector<float> b_vals(random.Gaussian<float>(AsSpan(b_dims), 0.0f, 0.25f));

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

  // C's shape selects one of three branches in the reduction's write function: read per column for
  // [N], as a scalar for [1], and per row for [M,1]. `beta` is a uniform on the same path, so a case
  // with beta != 1 is what separates applying it from ignoring it.
  const auto run_with_bias = [&](const std::vector<int64_t>& c_dims, float beta) {
    SCOPED_TRACE("C=" + ShapeToString(c_dims) + " beta=" + std::to_string(beta));
    std::vector<float> c_vals(random.Gaussian<float>(AsSpan(c_dims), 0.0f, 0.25f));

    std::vector<float> expected(matmul_only);
    for (int64_t m = 0; m < M; ++m) {
      for (int64_t n = 0; n < N; ++n) {
        const float c = c_vals.size() == 1
                            ? c_vals[0]
                            : c_vals[static_cast<size_t>(c_dims.back() == 1 ? m : n)];
        expected[static_cast<size_t>(m * N + n)] += beta * c;
      }
    }

    OpTester test("Gemm", 13);
    test.AddAttribute("beta", beta);
    test.AddInput<float>("A", a_dims, a_vals);
    test.AddInput<float>("B", b_dims, b_vals);
    test.AddInput<float>("C", c_dims, c_vals);
    test.AddOutput<float>("Y", y_dims, expected);
    test.SetOutputAbsErr("Y", 1e-3f);
    test.SetOutputRelErr("Y", 1e-3f);
    test.ConfigEp(make_ep()).RunWithConfig();
  };

  run_with_bias({N}, 1.0f);
  run_with_bias({N}, 0.5f);
  run_with_bias({1}, 1.0f);
  run_with_bias({M, 1}, 1.0f);
}

}  // namespace test
}  // namespace onnxruntime
