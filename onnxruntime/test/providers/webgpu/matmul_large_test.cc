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

}  // namespace test
}  // namespace onnxruntime
