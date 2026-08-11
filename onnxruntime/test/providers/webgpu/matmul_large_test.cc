// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/providers/cpu/math/matmul_helper.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {

// Reference matmul using MatMulComputeHelper for shape/offset computation.
// Supports arbitrary-rank batched matmul with broadcasting.
static void ComputeExpectedResult(const std::vector<float>& a_vals, const std::vector<float>& b_vals,
                                  std::vector<float>& out_vals,
                                  const MatMulComputeHelper& helper) {
  const auto M = helper.M();
  const auto K = helper.K();
  const auto N = helper.N();
  const auto& left_offsets = helper.LeftOffsets();
  const auto& right_offsets = helper.RightOffsets();
  const auto& output_offsets = helper.OutputOffsets();
  const size_t num_batches = output_offsets.size();

  for (size_t batch = 0; batch < num_batches; ++batch) {
    const float* a = a_vals.data() + left_offsets[batch];
    const float* b = b_vals.data() + right_offsets[batch];
    float* out = out_vals.data() + output_offsets[batch];
    for (ptrdiff_t m = 0; m < M; ++m) {
      for (ptrdiff_t n = 0; n < N; ++n) {
        float sum = 0.0f;
        for (ptrdiff_t k = 0; k < K; ++k) {
          sum += a[m * K + k] * b[k * N + n];
        }
        out[m * N + n] = sum;
      }
    }
  }
}

template <typename T, int version = 13>
void RunTestTyped(std::initializer_list<int64_t> a_dims, std::initializer_list<int64_t> b_dims) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, MLFloat16>, "unexpected type for T");

  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  TensorShape a_shape(a_dims);
  TensorShape b_shape(b_dims);
  MatMulComputeHelper helper;
  ASSERT_STATUS_OK(helper.Compute(a_shape, b_shape));
  const TensorShape& output_shape = helper.OutputShape();

  RandomValueGenerator random{1234};
  std::vector<float> a_vals(random.Gaussian<float>(AsSpan(a_dims), 0.0f, 0.25f));
  std::vector<float> b_vals(random.Gaussian<float>(AsSpan(b_dims), 0.0f, 0.25f));

  std::vector<float> expected_vals(output_shape.Size());
  ComputeExpectedResult(a_vals, b_vals, expected_vals, helper);

  std::vector<int64_t> output_dims(output_shape.NumDimensions());
  output_shape.CopyDims(output_dims.data(), output_shape.NumDimensions());

  OpTester test("MatMul", version);
  if constexpr (std::is_same_v<T, float>) {
    test.AddInput<T>("A", a_dims, a_vals);
    test.AddInput<T>("B", b_dims, b_vals);
    test.AddOutput<T>("Y", output_dims, expected_vals);
  } else {
    test.AddInput<T>("A", a_dims, FloatsToMLFloat16s(a_vals));
    test.AddInput<T>("B", b_dims, FloatsToMLFloat16s(b_vals));
    test.AddOutput<T>("Y", output_dims, FloatsToMLFloat16s(expected_vals));
    test.SetOutputAbsErr("Y", 0.055f);
    test.SetOutputRelErr("Y", 0.02f);
  }

  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

template <int version = 13>
void RunBothTypes(std::initializer_list<int64_t> a_dims, std::initializer_list<int64_t> b_dims) {
  RunTestTyped<float, version>(a_dims, b_dims);
  RunTestTyped<MLFloat16, version>(a_dims, b_dims);
}

// 2D aligned baseline shapes.
TEST(MatMul_Large, DISABLED_Aligned) {
  RunBothTypes({128, 64}, {64, 1024});
}

// 2D unaligned edge shapes.
TEST(MatMul_Large, DISABLED_Unaligned) {
  RunBothTypes({127, 64}, {64, 1024});
  RunBothTypes({127, 63}, {63, 1023});
  RunBothTypes({128, 36}, {36, 1024});
  RunBothTypes({128, 68}, {68, 1024});
}

// 3D broadcast and non-broadcast cases.
TEST(MatMul_Large, DISABLED_Broadcast3D) {
  RunBothTypes({2, 128, 64}, {64, 1024});
  RunBothTypes({2, 128, 64}, {2, 64, 1024});
  RunBothTypes({2, 128, 64}, {64, 1023});
  RunBothTypes({2, 128, 64}, {2, 64, 1023});
}

// 4D broadcast cases.
TEST(MatMul_Large, DISABLED_Broadcast4D) {
  RunBothTypes({2, 2, 128, 64}, {2, 64, 1024});
  RunBothTypes({2, 2, 128, 64}, {2, 64, 1023});
}

// Batched B (true bmm): A [..., M, K] x B [..., K, N] with matching batch. On the
// Intel subgroup path each (A, B) slice is dispatched on z. Covers small and
// larger batch counts with tile-aligned per-slice shapes.
TEST(MatMul_Large, DISABLED_BatchedB) {
  RunBothTypes({2, 128, 64}, {2, 64, 1024});
  RunBothTypes({4, 64, 128}, {4, 128, 256});
  RunBothTypes({8, 32, 64}, {8, 64, 64});
  RunBothTypes({16, 64, 64}, {16, 64, 128});
}

// Batched B with per-slice shapes that are not tile multiples. The odd-N slices
// (1023, 33) fall back to the generic path (the Intel f16 subgroup kernel needs
// an even B row stride); the even-N odd-M slice (130 x 65) still exercises the
// kernel's bounds-checked partial-M stores under z-dispatch. All must be correct.
TEST(MatMul_Large, DISABLED_BatchedB_Unaligned) {
  RunBothTypes({3, 127, 64}, {3, 64, 1023});
  RunBothTypes({5, 65, 96}, {5, 96, 130});
  RunBothTypes({2, 129, 80}, {2, 80, 33});
}

// Multi-dimensional batch: leading A/B dims collapse into the z grid.
TEST(MatMul_Large, DISABLED_BatchedB_4D) {
  RunBothTypes({2, 2, 64, 128}, {2, 2, 128, 256});
  RunBothTypes({2, 3, 32, 64}, {2, 3, 64, 96});
}

// Large batch with a small per-slice M x N grid: batch alone fills the machine,
// so the selector should retire split-K (ClampSplitKForBatch). Correctness must
// hold regardless of the chosen config.
TEST(MatMul_Large, DISABLED_BatchedB_LargeBatchSmallTile) {
  RunBothTypes({64, 16, 128}, {64, 128, 32});
  RunBothTypes({128, 8, 256}, {128, 256, 16});
}

// Broadcasted batch dims that are NOT identical but share the same batch
// *product* (A=[2,1,...], B=[1,2,...] -> [2,2,...]; A=[1,4,...], B=[4,1,...] ->
// [4,4,...]). A product-only batch check would wrongly route these onto the
// Intel subgroup path, which pairs slice i of A with slice i of B and copies A's
// shape to the output - producing the wrong output shape and mismatched pairing.
// N is even and every per-slice shape is tile-aligned, so only the
// identical-batch-dims guard (not the odd-N or partial-tile fallbacks) keeps them
// on the generic broadcasting MatMul. Results must match the broadcast reference.
TEST(MatMul_Large, DISABLED_BatchedB_BroadcastEqualProduct) {
  RunBothTypes({2, 1, 128, 64}, {1, 2, 64, 256});
  RunBothTypes({1, 4, 64, 128}, {4, 1, 128, 96});
}

}  // namespace test
}  // namespace onnxruntime
