// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_WEBGPU) && !defined(DISABLE_CONTRIB_OPS) && !defined(ORT_USE_EP_API_ADAPTERS) && !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)

#include <algorithm>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "contrib_ops/webgpu/quantization/dp4a_matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits_tile.h"
#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"
#include "core/common/inlined_containers.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/graph/model.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {

using namespace webgpu;
namespace quantization = contrib::webgpu;

class PreparationTestKernel final : public OpKernel {
 public:
  explicit PreparationTestKernel(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext*) const override { return Status::OK(); }
};

class PreparationTestContext final : public OpKernelContext {
 public:
  explicit PreparationTestContext(AllocatorPtr allocator)
      : OpKernelContext(nullptr, DefaultLoggingManager().DefaultLogger(), nullptr), allocator_(std::move(allocator)) {}
  Status GetTempSpaceAllocator(AllocatorPtr* output) const override {
    *output = allocator_;
    return Status::OK();
  }

 private:
  AllocatorPtr allocator_;
};

class MatMulNBitsInputPreparationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ep_ = WebGpuProviderFactoryCreator::Create(options_)->CreateProvider();
    ep_->SetLogger(&DefaultLoggingManager().DefaultLogger());
    allocator_ = std::make_shared<GpuBufferAllocator>(
        0, [this]() -> const BufferManager& { return Ep().BufferManager(); },
        [this]() -> CommandRecordingState& { return Ep().Recording(); }, false);
  }

  WebGpuExecutionProvider& Ep() { return static_cast<WebGpuExecutionProvider&>(*ep_); }

  template <typename T, typename Container>
  Tensor Upload(const TensorShape& shape, const Container& values) {
    Tensor tensor(DataTypeImpl::GetType<T>(), shape, allocator_);
    Ep().BufferManager().Upload(Ep().Recording(), values.data(),
                                reinterpret_cast<WGPUBuffer>(tensor.MutableDataRaw()), tensor.SizeInBytes());
    return tensor;
  }

  template <typename T>
  std::vector<T> Download(Tensor& tensor) {
    std::vector<T> values(tensor.Shape().Size());
    Ep().BufferManager().Download(Ep().Recording(), reinterpret_cast<WGPUBuffer>(tensor.MutableDataRaw()),
                                  values.data(), tensor.SizeInBytes());
    return values;
  }

  template <typename T>
  void Check(bool subgroup_matrix) {
    auto& device = WebGpuContextFactory::GetContext(0);
    if constexpr (std::is_same_v<T, MLFloat16>) {
      if (!device.DeviceHasFeature(wgpu::FeatureName::ShaderF16)) {
        GTEST_SKIP() << "FP16 is unavailable.";
      }
    }
    Model model("input_preparation", false, DefaultLoggingManager().DefaultLogger());
    auto& node = model.MainGraph().AddNode("unused", "Identity", "", {}, {});
    auto kernel_def = KernelDefBuilder().SetName("Identity").Provider(kWebGpuExecutionProvider).SinceVersion(1).Build();
    const std::unordered_map<int, OrtValue> initializers;
    const OrtValueNameIdxMap values;
    const DataTransferManager transfers;
    const AllocatorMap allocators;
    OpKernelInfo info(node, *kernel_def, Ep(), initializers, values, transfers, allocators, options_);
    PreparationTestKernel kernel(info);
    PreparationTestContext kernel_context(allocator_);
    webgpu::ComputeContext context(device, Ep(), kernel, kernel_context);
    constexpr uint32_t M = 65, K = 256;
    std::optional<SubgroupMatrixConfig> config;
    if (subgroup_matrix) {
      if (!quantization::CanApplySubgroupMatrixMatMulNBits(context, 4, 32, 1, 64, K, 4,
                                                           std::is_same_v<T, MLFloat16>, config, M, false) ||
          !config->needsPrepack) {
        GTEST_SKIP() << "A compatible subgroup-matrix layout is unavailable.";
      }
    } else if (!quantization::CanApplyDP4AMatrixMatMulNBits(context, 4, 32, 64, K, 4)) {
      GTEST_SKIP() << "DP4A is unavailable.";
    }
    // Each invocation owns a new workspace; two projections reuse it within that invocation.
    for (int invocation = 0; invocation < 2; ++invocation) {
      std::vector<T> input(M * K);
      for (size_t i = 0; i < input.size(); ++i) {
        input[i] = T(static_cast<float>(static_cast<int>(i % 17) - 8 + invocation) * 0.125f);
      }
      auto a = Upload<T>({M, K}, input);
      quantization::DP4AQuantizedInput workspace;
      Tensor packed;
      if (subgroup_matrix) {
        // M=65 exercises padding; 256 rows cover the supported workgroup row tiles.
        ASSERT_STATUS_OK(quantization::PrepackMatMulNBitsInput(&a, M, K, 256, *config, context, packed));
      }
      const void* quantized_values = nullptr;
      const void* quantized_scales = nullptr;
      for (uint32_t projection = 0; projection < 2; ++projection) {
        const uint32_t N = subgroup_matrix ? 64 : (projection == 0 ? 16 : 64);
        const uint32_t bits = projection == 0 ? 8 : 4;
        auto b = Upload<uint8_t>({N, K / 32, 4 * bits},
                                 std::vector<uint8_t>(N * K * bits / 8, bits == 8 ? 131 : 0x99));
        auto scales = Upload<T>({N, K / 32}, std::vector<T>(N * K / 32, T(0.125f)));
        Tensor shared(DataTypeImpl::GetType<T>(), TensorShape{M, N}, allocator_);
        Tensor fresh(DataTypeImpl::GetType<T>(), TensorShape{M, N}, allocator_);
        if (subgroup_matrix) {
          ASSERT_STATUS_OK(quantization::ApplySubgroupMatrixMatMulNBits(
              &a, &b, &scales, nullptr, nullptr, M, N, K, bits, K / 32, *config, context, &shared,
              0, nullptr, &packed));
          ASSERT_STATUS_OK(quantization::ApplySubgroupMatrixMatMulNBits(
              &a, &b, &scales, nullptr, nullptr, M, N, K, bits, K / 32, *config, context, &fresh, 0));
        } else {
          ASSERT_STATUS_OK(quantization::ApplyDP4AMatrixMatMulNBits(
              &a, &b, &scales, nullptr, nullptr, 1, M, M, N, K, 32, K / 32, 4, bits,
              context, &shared, 0, nullptr, &workspace));
          ASSERT_TRUE(workspace.values.has_value());
          ASSERT_TRUE(workspace.scales.has_value());
          if (projection == 0) {
            quantized_values = workspace.values->DataRaw();
            quantized_scales = workspace.scales->DataRaw();
          } else {
            EXPECT_EQ(workspace.values->DataRaw(), quantized_values);
            EXPECT_EQ(workspace.scales->DataRaw(), quantized_scales);
          }
          ASSERT_STATUS_OK(quantization::ApplyDP4AMatrixMatMulNBits(
              &a, &b, &scales, nullptr, nullptr, 1, M, M, N, K, 32, K / 32, 4, bits, context, &fresh, 0));
        }
        EXPECT_EQ(Download<T>(shared), Download<T>(fresh));
      }
    }
  }

  void CheckTile(bool matrix) {
    auto& device = WebGpuContextFactory::GetContext(0);
    const auto& adapter = device.AdapterInfo();
    if (adapter.vendor != std::string_view{"apple"} || adapter.subgroupMinSize != 32 ||
        adapter.subgroupMaxSize != 32 || !device.DeviceHasFeature(wgpu::FeatureName::ShaderF16) ||
        !device.DeviceHasFeature(wgpu::FeatureName::Subgroups)) {
      GTEST_SKIP() << "Requires Apple FP16 and fixed 32-lane subgroups.";
    }
    Model model("matmul_tile", false, DefaultLoggingManager().DefaultLogger());
    auto& node = model.MainGraph().AddNode("unused", "Identity", "", {}, {});
    auto kernel_def = KernelDefBuilder().SetName("Identity").Provider(kWebGpuExecutionProvider).SinceVersion(1).Build();
    const std::unordered_map<int, OrtValue> initializers;
    const OrtValueNameIdxMap values;
    const DataTransferManager transfers;
    const AllocatorMap allocators;
    OpKernelInfo info(node, *kernel_def, Ep(), initializers, values, transfers, allocators, options_);
    PreparationTestKernel kernel(info);
    PreparationTestContext kernel_context(allocator_);
    webgpu::ComputeContext context(device, Ep(), kernel, kernel_context);
    constexpr auto f32 = wgpu::SubgroupMatrixComponentType::F32;
    if (matrix && !SelectSubgroupMatrixConfig(context, {{f32, f32, 8, 8, 8, 32, false}})) {
      GTEST_SKIP() << "Requires FP32 8x8x8 subgroup matrices.";
    }
    const uint32_t m = matrix ? 128 : 1024;
    constexpr uint32_t n = 4096, k = 4096;
    InlinedVector<MLFloat16> input(m * k);
    for (uint32_t row = 0; row < m; ++row) {
      const float value = (row % 2 == 0 ? 1.0f : -1.0f) * (matrix ? 127.0f : 0.125f);
      std::fill_n(input.begin() + row * k, k, MLFloat16(value));
    }
    InlinedVector<uint8_t> weights(n * k / 2);
    for (uint32_t col = 0; col < n; ++col) {
      for (uint32_t inner = 0; inner < k; ++inner) {
        // Matrix partial sums overflow FP16 before cancellation; the final result is finite.
        const int weight = matrix ? (inner == k - 1 ? 0 : (inner < k / 2 ? 1 : -1))
                                  : static_cast<int>(1u << (col % 3));
        weights[(col * k + inner) / 2] |= static_cast<uint8_t>((weight + 8) << ((inner % 2) * 4));
      }
    }
    auto a = Upload<MLFloat16>({m, k}, input);
    auto b = Upload<uint8_t>({n, k / 32, 16}, weights);
    auto scales = Upload<MLFloat16>({n, k / 32},
                                    InlinedVector<MLFloat16>(n * k / 32, MLFloat16(matrix ? 16.0f : 1.0f / 128)));
    Tensor output(DataTypeImpl::GetType<MLFloat16>(), TensorShape{m, n}, allocator_);
    bool handled = false;
    ASSERT_STATUS_OK(quantization::TryApplyMatMulNBitsTile(&a, &b, &scales, &output, m, n, k, context, handled));
    ASSERT_TRUE(handled);  // A fallback must not make this test pass.
    const auto actual = Download<MLFloat16>(output);
    for (uint32_t row = 0; row < m; ++row) {
      for (uint32_t col = 0; col < n; ++col) {
        const float expected = (row % 2 == 0 ? 1.0f : -1.0f) * (matrix ? 2032.0f : 4.0f * (1u << (col % 3)));
        ASSERT_EQ(actual[row * n + col], MLFloat16(expected)) << "row=" << row << ", col=" << col;
      }
    }
  }

 private:
  ConfigOptions options_;
  std::unique_ptr<IExecutionProvider> ep_;
  AllocatorPtr allocator_;
};

TEST_F(MatMulNBitsInputPreparationTest, RegisterTile) { CheckTile(false); }
TEST_F(MatMulNBitsInputPreparationTest, MatrixTileFp32Cancellation) { CheckTile(true); }

TEST(MatMulNBitsTileSelectionTest, FeaturePrecisionAndShapeBoundaries) {
  using quantization::MatMulNBitsTile;
  using quantization::SelectMatMulNBitsTile;
  EXPECT_EQ(SelectMatMulNBitsTile(128, 4096, 4096, true, false), MatMulNBitsTile::Matrix);
  EXPECT_EQ(SelectMatMulNBitsTile(128, 4096, 4096, false, false), MatMulNBitsTile::None);
  EXPECT_EQ(SelectMatMulNBitsTile(256, 4096, 4096, false, false), MatMulNBitsTile::Register);
  EXPECT_EQ(SelectMatMulNBitsTile(256, 4096, 4096, true, true), MatMulNBitsTile::Matrix);
  EXPECT_EQ(SelectMatMulNBitsTile(512, 4096, 4096, true, false), MatMulNBitsTile::Matrix);
  EXPECT_EQ(SelectMatMulNBitsTile(1024, 4096, 4096, true, true), MatMulNBitsTile::None);
  for (uint32_t m : {0u, 1u, 32u, 64u, 127u, 129u, 255u, 257u}) {
    EXPECT_EQ(SelectMatMulNBitsTile(m, 4096, 4096, true, false), MatMulNBitsTile::None);
  }
  EXPECT_EQ(SelectMatMulNBitsTile(256, 4095, 4096, true, false), MatMulNBitsTile::None);
  EXPECT_EQ(SelectMatMulNBitsTile(256, 4096, 4095, true, false), MatMulNBitsTile::None);
}

TEST_F(MatMulNBitsInputPreparationTest, SharedDP4AInputFloat32) { Check<float>(false); }
TEST_F(MatMulNBitsInputPreparationTest, SharedDP4AInputFloat16) { Check<MLFloat16>(false); }
TEST_F(MatMulNBitsInputPreparationTest, SharedSubgroupMatrixInputFloat16) { Check<MLFloat16>(true); }

}  // namespace
}  // namespace test
}  // namespace onnxruntime
#endif
