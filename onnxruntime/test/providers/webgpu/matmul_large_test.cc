// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/graph/onnx_protobuf.h"
#include "core/providers/webgpu/math/matmul_algorithm.h"
#if !defined(ORT_USE_EP_API_ADAPTERS)
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/webgpu_context.h"
#endif
#include "core/providers/webgpu/webgpu_provider_options.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {

// Independent reference for ONNX MatMul, including vector promotion and
// broadcasting of leading batch dimensions.
static void ComputeExpectedResult(std::initializer_list<int64_t> a_dims,
                                  std::initializer_list<int64_t> b_dims,
                                  const std::vector<float>& a_vals, const std::vector<float>& b_vals,
                                  TensorShapeVector& output_dims, std::vector<float>& out_vals) {
  ASSERT_GT(a_dims.size(), 0u);
  ASSERT_GT(b_dims.size(), 0u);
  TensorShapeVector a_shape(a_dims), b_shape(b_dims);
  const bool a_is_vector = a_shape.size() == 1;
  const bool b_is_vector = b_shape.size() == 1;
  if (a_is_vector) a_shape.insert(a_shape.begin(), 1);
  if (b_is_vector) b_shape.push_back(1);
  const size_t rank = std::max(a_shape.size(), b_shape.size());
  a_shape.insert(a_shape.begin(), rank - a_shape.size(), 1);
  b_shape.insert(b_shape.begin(), rank - b_shape.size(), 1);
  const int64_t M = a_shape[rank - 2];
  const int64_t K = a_shape.back();
  const int64_t N = b_shape.back();
  ASSERT_EQ(K, b_shape[rank - 2]);
  output_dims.clear();
  for (size_t axis = 0; axis + 2 < rank; ++axis) {
    ASSERT_TRUE(a_shape[axis] == b_shape[axis] || a_shape[axis] == 1 || b_shape[axis] == 1);
    output_dims.push_back(a_shape[axis] == 1 ? b_shape[axis] : a_shape[axis]);
  }
  const int64_t num_batches = std::accumulate(output_dims.begin(), output_dims.end(), int64_t{1},
                                              std::multiplies<int64_t>());
  out_vals.assign(static_cast<size_t>(num_batches * M * N), 0.0f);
  for (int64_t batch = 0; batch < num_batches; ++batch) {
    int64_t remaining = batch;
    int64_t a_offset = 0, b_offset = 0;
    int64_t a_stride = M * K, b_stride = K * N;
    for (size_t i = output_dims.size(); i > 0; --i) {
      const size_t axis = i - 1;
      const int64_t coordinate = remaining % output_dims[axis];
      remaining /= output_dims[axis];
      if (a_shape[axis] != 1) a_offset += coordinate * a_stride;
      if (b_shape[axis] != 1) b_offset += coordinate * b_stride;
      a_stride *= a_shape[axis];
      b_stride *= b_shape[axis];
    }
    const float* a = a_vals.data() + a_offset;
    const float* b = b_vals.data() + b_offset;
    float* out = out_vals.data() + batch * M * N;
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
  if (!a_is_vector) output_dims.push_back(M);
  if (!b_is_vector) output_dims.push_back(N);
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
      constexpr auto kF16 = wgpu::SubgroupMatrixComponentType::F16;
      if (!webgpu::detail::SelectSubgroupMatrixConfigFromAdapterConfigs(
              {device_configs.configs, device_configs.configCount},
              adapter_info.subgroupMinSize, adapter_info.subgroupMaxSize,
              context.DeviceHasFeature(wgpu::FeatureName::SubgroupSizeControl),
              {{kF16, kF16, 8, 16, 16, 32, false}})) {
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

  RandomValueGenerator random{1234};
  std::vector<float> a_vals(random.Gaussian<float>(AsSpan(a_dims), 0.0f, 0.25f));
  std::vector<float> b_vals(random.Gaussian<float>(AsSpan(b_dims), 0.0f, 0.25f));

  TensorShapeVector output_dims;
  std::vector<float> expected_vals;
  ASSERT_NO_FATAL_FAILURE(ComputeExpectedResult(a_dims, b_dims, a_vals, b_vals, output_dims, expected_vals));

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

TEST(MatMulNaiveProgramTest, ZeroContractionDimensionExecution) {
  RunTestTyped<float>({1, 0}, {0, 8});
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
  ConfigOptions valid_config_options{};
  if (!WebGpuExecutionProviderWithOptions(valid_config_options)) {
    GTEST_SKIP() << "WebGPU execution provider is unavailable in this build.";
  }

  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kForceMatMulAlgorithm, "unknown"));
  EXPECT_THROW(WebGpuExecutionProviderWithOptions(config_options), OnnxRuntimeException);
}

#if defined(DAWN_ENABLE_VULKAN)
static std::string BuildDynamicMatMulModelBytes() {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  auto* opset = model.add_opset_import();
  opset->set_domain("");
  opset->set_version(13);

  auto* graph = model.mutable_graph();
  graph->set_name("dynamic_matmul");

  auto set_float_shape = [](ONNX_NAMESPACE::ValueInfoProto* value_info,
                            const char* first_dimension,
                            const char* second_dimension) {
    auto* tensor_type = value_info->mutable_type()->mutable_tensor_type();
    tensor_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    auto* shape = tensor_type->mutable_shape();
    shape->add_dim()->set_dim_param(first_dimension);
    if (second_dimension != nullptr) {
      shape->add_dim()->set_dim_param(second_dimension);
    } else {
      shape->add_dim()->set_dim_value(8);
    }
  };

  auto* a = graph->add_input();
  a->set_name("A");
  set_float_shape(a, "M", "K");

  auto* b = graph->add_input();
  b->set_name("B");
  set_float_shape(b, "K", nullptr);

  auto* y = graph->add_output();
  y->set_name("Y");
  set_float_shape(y, "M", nullptr);

  auto* node = graph->add_node();
  node->set_name("MatMul");
  node->set_op_type("MatMul");
  node->add_input("A");
  node->add_input("B");
  node->add_output("Y");

  std::string bytes;
  model.SerializeToString(&bytes);
  return bytes;
}

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

TEST(WebGpuMatMulAlgorithmTest, ReselectsAlgorithmForEachDynamicShape) {
  ConfigOptions config_options{};
  ASSERT_STATUS_OK(config_options.AddConfigEntry(webgpu::options::kDawnBackendType,
                                                 webgpu::options::kDawnBackendType_Vulkan));
  auto webgpu_ep = WebGpuExecutionProviderWithOptions(config_options);
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  SessionOptions session_options;
  session_options.session_logid = "WebGpuDynamicMatMulAlgorithmSelection";
  InferenceSessionWrapper session(session_options, GetEnvironment());
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(webgpu_ep)));

  const std::string model_bytes = BuildDynamicMatMulModelBytes();
  ASSERT_STATUS_OK(session.Load(model_bytes.data(), static_cast<int>(model_bytes.size())));
  ASSERT_STATUS_OK(session.Initialize());

  const std::vector<std::string> output_names{"Y"};
  auto run = [&](const TensorShape& a_shape, std::vector<float> a_data,
                 const TensorShape& b_shape, std::vector<float> b_data,
                 const TensorShape& expected_shape, float expected_value) {
    OrtValue a_value;
    OrtValue b_value;
    CreateMLValue<float>(a_shape.GetDims(), a_data.data(), OrtMemoryInfo(), &a_value);
    CreateMLValue<float>(b_shape.GetDims(), b_data.data(), OrtMemoryInfo(), &b_value);

    NameMLValMap feeds;
    feeds.emplace("A", std::move(a_value));
    feeds.emplace("B", std::move(b_value));
    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(session.Run(feeds, output_names, &fetches));
    ASSERT_EQ(fetches.size(), 1u);
    const Tensor& output = fetches[0].Get<Tensor>();
    ASSERT_EQ(output.Shape(), expected_shape);
    for (float value : output.DataAsSpan<float>()) {
      EXPECT_EQ(value, expected_value);
    }
  };

  // The first invocation uses Packed. The second has K=0 and must reselect Naive on the same
  // session and kernel; reusing the first execution plan would fail Packed's nonzero-K prerequisite.
  run(TensorShape({8, 8}), std::vector<float>(64, 1.0f),
      TensorShape({8, 8}), std::vector<float>(64, 1.0f),
      TensorShape({8, 8}), 8.0f);

  const float unused_storage = 0.0f;
  run(TensorShape({1, 0}), std::vector<float>{unused_storage},
      TensorShape({0, 8}), std::vector<float>{unused_storage},
      TensorShape({1, 8}), 0.0f);
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
