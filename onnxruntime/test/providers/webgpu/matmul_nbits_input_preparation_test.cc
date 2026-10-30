// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_WEBGPU) && !defined(DISABLE_CONTRIB_OPS) && !defined(ORT_USE_EP_API_ADAPTERS) && !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)

#include <memory>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "contrib_ops/webgpu/quantization/dp4a_matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"
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

  template <typename T>
  Tensor Upload(const TensorShape& shape, std::vector<T> values) {
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
      uint32_t previous_a_block_size = 0;
      const bool per_channel_available = context.AdapterInfo().vendor != std::string_view{"qualcomm"};
      const uint32_t num_projections = subgroup_matrix ? 2 : 4;
      for (uint32_t projection = 0; projection < num_projections; ++projection) {
        const uint32_t N = subgroup_matrix ? 64 : (projection == 0 ? 16 : 64);
        const uint32_t bits = projection == 1 ? 4 : 8;
        const uint32_t block_size = projection == 2 ? K : 32;
        const uint32_t blocks = K / block_size;
        auto b = Upload<uint8_t>({N, blocks, block_size * bits / 8},
                                 std::vector<uint8_t>(N * K * bits / 8, bits == 8 ? 131 : 0x99));
        auto scales = Upload<T>({N, blocks}, std::vector<T>(N * blocks, T(0.125f)));
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
              &a, &b, &scales, nullptr, nullptr, 1, M, M, N, K, block_size, blocks, 4, bits,
              context, &shared, 0, nullptr, &workspace));
          ASSERT_TRUE(workspace.values.has_value());
          ASSERT_TRUE(workspace.scales.has_value());
          const uint32_t expected_a_block_size = projection == 2 && per_channel_available ? K : 128;
          EXPECT_EQ(workspace.a_block_size, expected_a_block_size);
          const bool expect_reuse = projection != 0 && expected_a_block_size == previous_a_block_size;
          if (expect_reuse) {
            EXPECT_EQ(workspace.values->DataRaw(), quantized_values);
            EXPECT_EQ(workspace.scales->DataRaw(), quantized_scales);
          }
          quantized_values = workspace.values->DataRaw();
          quantized_scales = workspace.scales->DataRaw();
          previous_a_block_size = workspace.a_block_size;
          ASSERT_STATUS_OK(quantization::ApplyDP4AMatrixMatMulNBits(
              &a, &b, &scales, nullptr, nullptr, 1, M, M, N, K, block_size, blocks, 4, bits, context, &fresh, 0));
        }
        EXPECT_EQ(Download<T>(shared), Download<T>(fresh));
      }
    }
  }

 private:
  ConfigOptions options_;
  std::unique_ptr<IExecutionProvider> ep_;
  AllocatorPtr allocator_;
};

TEST_F(MatMulNBitsInputPreparationTest, SharedDP4AInputFloat32) { Check<float>(false); }
TEST_F(MatMulNBitsInputPreparationTest, SharedDP4AInputFloat16) { Check<MLFloat16>(false); }
TEST_F(MatMulNBitsInputPreparationTest, SharedSubgroupMatrixInputFloat16) { Check<MLFloat16>(true); }

}  // namespace
}  // namespace test
}  // namespace onnxruntime
#endif
