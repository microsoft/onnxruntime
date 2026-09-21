// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/math/matmul_algorithm.h"
#if !defined(ORT_USE_EP_API_ADAPTERS)
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/webgpu_context.h"
#endif
#include "core/providers/webgpu/webgpu_provider_options.h"
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
  if (K == 0) {
    return;
  }
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

#if defined(_WIN32) && defined(DAWN_ENABLE_VULKAN)
static std::optional<std::string> GetForcedAlgorithmUnsupportedReason(
    const IExecutionProvider& ep,
    webgpu::MatMulAlgorithm algorithm) {
#if defined(ORT_USE_EP_API_ADAPTERS)
  ORT_UNUSED_PARAMETER(ep);
  switch (algorithm) {
    case webgpu::MatMulAlgorithm::IntelSubgroup:
    case webgpu::MatMulAlgorithm::PackedSplitK:
    case webgpu::MatMulAlgorithm::SubgroupMatrix:
      return "hardware-specific forced MatMul tests require direct adapter capability inspection.";
    case webgpu::MatMulAlgorithm::Naive:
    case webgpu::MatMulAlgorithm::Packed:
      return std::nullopt;
  }
  return std::nullopt;
#else
  auto& context = webgpu::WebGpuContextFactory::GetContext(ep.GetDeviceId());

  switch (algorithm) {
    case webgpu::MatMulAlgorithm::IntelSubgroup:
      if (context.AdapterInfo().vendor != std::string_view{"intel"}) {
        return "intel_subgroup requires an Intel adapter.";
      }
      if (!context.DeviceHasFeature(wgpu::FeatureName::Subgroups)) {
        return "intel_subgroup requires the WebGPU Subgroups feature.";
      }
      break;
    case webgpu::MatMulAlgorithm::PackedSplitK:
      if (context.GetSplitKConfig().GetSplitDimInner() == 0) {
        return "packed_split_k is not configured for the selected adapter.";
      }
      break;
    case webgpu::MatMulAlgorithm::SubgroupMatrix: {
      if (!context.DeviceHasFeature(wgpu::FeatureName::ChromiumExperimentalSubgroupMatrix)) {
        return "subgroup_matrix requires the WebGPU subgroup-matrix feature.";
      }

      const auto& adapter_info = context.AdapterInfo();
      const auto& device_configs = context.SubgroupMatrixConfigs();
      bool has_required_config = false;
      for (const auto& required_config : webgpu::supported_subgroup_matrix_configs) {
        if (!required_config.Is(8, 16, 16) ||
            required_config.componentType != wgpu::SubgroupMatrixComponentType::F16 ||
            required_config.resultComponentType != wgpu::SubgroupMatrixComponentType::F16) {
          continue;
        }
        for (size_t i = 0; i < device_configs.configCount; ++i) {
          const auto& device_config = device_configs.configs[i];
          if (device_config.componentType == required_config.componentType &&
              device_config.resultComponentType == required_config.resultComponentType &&
              device_config.M == required_config.M &&
              device_config.N == required_config.N &&
              device_config.K == required_config.K &&
              webgpu::IsSubgroupSizeSupported(
                  adapter_info.subgroupMinSize, adapter_info.subgroupMaxSize,
                  required_config.subgroupSize,
                  context.DeviceHasFeature(wgpu::FeatureName::SubgroupSizeControl))) {
            has_required_config = true;
            break;
          }
        }
        if (has_required_config) {
          break;
        }
      }
      if (!has_required_config) {
        return "subgroup_matrix requires an 8x16x16 F16 configuration with subgroup size 32.";
      }
      break;
    }
    case webgpu::MatMulAlgorithm::Naive:
    case webgpu::MatMulAlgorithm::Packed:
      break;
  }

  return std::nullopt;
#endif
}
#endif

template <typename T, int version = 13>
void RunTestTyped(std::initializer_list<int64_t> a_dims, std::initializer_list<int64_t> b_dims,
                  bool b_is_constant = false,
                  std::optional<webgpu::MatMulAlgorithm> forced_algorithm = std::nullopt,
                  OpTester::ExpectResult expected_result = OpTester::ExpectResult::kExpectSuccess,
                  const char* expected_error = nullptr) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, MLFloat16>, "unexpected type for T");

  std::unique_ptr<IExecutionProvider> webgpu_ep;
  if (forced_algorithm.has_value()) {
    ConfigOptions config_options{};
    const std::string algorithm_name{webgpu::MatMulAlgorithmName(*forced_algorithm)};
    ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kDawnBackendType,
                                                   webgpu::options::kDawnBackendType_Vulkan));
    ASSERT_STATUS_OK(config_options.AddConfigEntry(
        webgpu::options::kForceMatMulAlgorithm,
        algorithm_name.c_str()));
    webgpu_ep = WebGpuExecutionProviderWithOptions(config_options);
  } else {
    webgpu_ep = DefaultWebGpuExecutionProvider();
  }
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }
#if defined(_WIN32) && defined(DAWN_ENABLE_VULKAN)
  if (forced_algorithm.has_value()) {
    if (const auto reason = GetForcedAlgorithmUnsupportedReason(*webgpu_ep, *forced_algorithm);
        reason.has_value()) {
      GTEST_SKIP() << *reason;
    }
  }
#endif

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
    test.AddInput<T>("B", b_dims, b_vals, b_is_constant);
    test.AddOutput<T>("Y", output_dims, expected_vals);
  } else {
    test.AddInput<T>("A", a_dims, FloatsToMLFloat16s(a_vals));
    test.AddInput<T>("B", b_dims, FloatsToMLFloat16s(b_vals), b_is_constant);
    test.AddOutput<T>("Y", output_dims, FloatsToMLFloat16s(expected_vals));
    test.SetOutputAbsErr("Y", 0.055f);
    test.SetOutputRelErr("Y", 0.02f);
  }

  test.ConfigEp(std::move(webgpu_ep))
      .Config(expected_result, expected_error == nullptr ? "" : expected_error)
      .RunWithConfig();
}

template <int version = 13>
void RunBothTypes(std::initializer_list<int64_t> a_dims, std::initializer_list<int64_t> b_dims) {
  RunTestTyped<float, version>(a_dims, b_dims);
  RunTestTyped<MLFloat16, version>(a_dims, b_dims);
}

TEST(MatMulNaiveProgramTest, Broadcast4DExecution) {
  RunTestTyped<float>({3, 1, 1, 2}, {2, 2, 2});
  RunTestTyped<float>({2, 2, 3, 2}, {2, 1});
  RunTestTyped<float>({2, 3, 2}, {3, 2, 2, 1});
}

TEST(MatMulNaiveProgramTest, VectorExecution) {
  RunTestTyped<float>({2}, {2, 3});
  RunTestTyped<float>({3}, {3});
}

TEST(MatMulProgramTest, VectorFallbackExecution) {
  RunTestTyped<float>({8}, {8, 3});
  RunTestTyped<float>({2, 8}, {8});
  RunTestTyped<float>({8}, {8});
  RunTestTyped<float>({8}, {2, 8, 3});
  RunTestTyped<float>({2, 2, 8}, {8});
}

#if defined(_WIN32)
TEST(WebGpuMatMulAlgorithmTest, RejectsUnknownForcedAlgorithm) {
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kForceMatMulAlgorithm, "unknown"));
  EXPECT_THROW(WebGpuExecutionProviderWithOptions(config_options), OnnxRuntimeException);
}

#if defined(DAWN_ENABLE_VULKAN)
TEST(WebGpuMatMulAlgorithmTest, ForcedNaive) {
  RunTestTyped<float>({8, 8}, {8, 8}, false, webgpu::MatMulAlgorithm::Naive);
}

TEST(WebGpuMatMulAlgorithmTest, ForcedPacked) {
  RunTestTyped<float>({2, 2}, {2, 2}, false, webgpu::MatMulAlgorithm::Packed);
}

TEST(WebGpuMatMulAlgorithmTest, ForcedPackedRejectsZeroContractionDimension) {
  RunTestTyped<float>({1, 0}, {0, 1}, false, webgpu::MatMulAlgorithm::Packed,
                      OpTester::ExpectResult::kExpectFailure,
                      "MatMul algorithm packed");
}

TEST(WebGpuMatMulAlgorithmTest, ForcedIntelSubgroup) {
  RunTestTyped<float>({8, 32}, {32, 64}, false, webgpu::MatMulAlgorithm::IntelSubgroup);
}

TEST(WebGpuMatMulAlgorithmTest, ForcedIntelSubgroupRejectsZeroContractionDimension) {
  RunTestTyped<float>({1, 0}, {0, 1}, false, webgpu::MatMulAlgorithm::IntelSubgroup,
                      OpTester::ExpectResult::kExpectFailure,
                      "MatMul algorithm intel_subgroup");
}

TEST(WebGpuMatMulAlgorithmTest, ForcedPackedSplitK) {
  RunTestTyped<float>({1, 1024}, {1024, 16}, false, webgpu::MatMulAlgorithm::PackedSplitK);
}

TEST(WebGpuMatMulAlgorithmTest, ForcedSubgroupMatrix) {
  RunTestTyped<MLFloat16>({32, 16}, {16, 32}, false, webgpu::MatMulAlgorithm::SubgroupMatrix);
}

TEST(WebGpuMatMulAlgorithmTest, ForcedSubgroupMatrixRejectsFloatInputs) {
  RunTestTyped<float>({32, 16}, {16, 32}, false, webgpu::MatMulAlgorithm::SubgroupMatrix,
                      OpTester::ExpectResult::kExpectFailure,
                      "MatMul algorithm subgroup_matrix");
}
#endif  // defined(DAWN_ENABLE_VULKAN)
#endif

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

// Constant f16 weight with odd N. The Intel f16 subgroup-matrix load needs an
// even B row stride, so a non-constant odd-N B falls back to the generic path. When
// B is a constant initializer, the first Compute lazily pads it to an even stride
// (N+1) and the subgroup kernel consumes the cached copy via the N_b uniform (output
// is still written at the real, odd N). Marking B constant here exercises that
// padded path for both shared 2D and batched weights across several odd N (1023,
// 33, 65), with even K and both aligned and partial M. Results must match the
// reference. f16 only: the subgroup kernel is f16, so a float B would take the
// generic path.
TEST(MatMul_Large, DISABLED_ConstantWeightOddN) {
  RunTestTyped<MLFloat16>({128, 64}, {64, 1023}, /*b_is_constant=*/true);
  RunTestTyped<MLFloat16>({127, 64}, {64, 1023}, /*b_is_constant=*/true);
  RunTestTyped<MLFloat16>({64, 96}, {96, 33}, /*b_is_constant=*/true);
  RunTestTyped<MLFloat16>({130, 80}, {80, 65}, /*b_is_constant=*/true);
  RunTestTyped<MLFloat16>({2, 127, 64}, {2, 64, 1023}, /*b_is_constant=*/true);
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
