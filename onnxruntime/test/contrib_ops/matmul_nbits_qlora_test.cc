// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#if defined(USE_WEBGPU) && !defined(DISABLE_CONTRIB_OPS)
#if !defined(ORT_USE_EP_API_ADAPTERS)
#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"
#include "core/providers/webgpu/webgpu_context.h"
#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
#include "dawn/native/DawnNative.h"
#endif
#endif
#include "core/providers/webgpu/webgpu_provider_options.h"
#endif

namespace onnxruntime::test {
#if defined(USE_WEBGPU) && !defined(DISABLE_CONTRIB_OPS)

template <typename T>
void RunEmptyAndActiveAdapters() {
  for (int64_t m : {1, 37}) {
    for (auto ranks : {std::pair{0, 0}, {0, 1}, {32, 0}, {32, 1}}) {
      auto ep = DefaultWebGpuExecutionProvider();
      ASSERT_NE(ep, nullptr);
      OpTester test("MatMulNBitsQLoRA", 1, kMSDomain);
      test.AddAttribute<int64_t>("K", 64);
      test.AddAttribute<int64_t>("N", 128);
      test.AddInput<T>("X", {1, m, 64}, std::vector<T>(m * 64, T(0.5f)));
      test.AddInput<uint8_t>("W", {128, 2, 16}, std::vector<uint8_t>(4096, 0x99), true);
      test.AddInput<T>("S", {128, 2}, std::vector<T>(256, T(0.03125f)), true);
      test.AddInput<uint8_t>("AQ", {ranks.first, 2, 32}, std::vector<uint8_t>(ranks.first * 64, 129));
      test.AddInput<T>("AS", {32, 2}, std::vector<T>(64, T(0.03125f)));
      test.AddInput<uint8_t>("BQ", {128, ranks.second, 32}, std::vector<uint8_t>(128 * ranks.second * 32, 129));
      test.AddInput<T>("BS", {128, 1}, std::vector<T>(128, T(0.0625f)));
      const float expected = ranks.first && ranks.second ? 3.0f : 1.0f;
      test.AddOutput<T>("Y", {1, m, 128}, std::vector<T>(m * 128, T(expected)));
      std::vector<std::unique_ptr<IExecutionProvider>> eps;
      eps.push_back(std::move(ep));
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
    }
  }
}

TEST(MatMulNBitsQLoRA, EmptyAndActiveAdapters) { RunEmptyAndActiveAdapters<MLFloat16>(); }
TEST(MatMulNBitsQLoRA, EmptyAndActiveAdaptersFp32) { RunEmptyAndActiveAdapters<float>(); }

TEST(MatMulNBitsQLoRA, RejectsUnsupportedAdapterShapeAndBits) {
  for (bool wrong_bits : {false, true}) {
    auto ep = DefaultWebGpuExecutionProvider();
    ASSERT_NE(ep, nullptr);
    OpTester test("MatMulNBitsQLoRA", 1, kMSDomain);
    test.AddAttribute<int64_t>("K", 64);
    test.AddAttribute<int64_t>("N", 128);
    if (wrong_bits) test.AddAttribute<int64_t>("lora_bits", 4);
    test.AddInput<MLFloat16>("X", {1, 1, 64}, std::vector<MLFloat16>(64, MLFloat16(0.5f)));
    test.AddInput<uint8_t>("W", {128, 2, 16}, std::vector<uint8_t>(4096, 0x99), true);
    test.AddInput<MLFloat16>("S", {128, 2}, std::vector<MLFloat16>(256, MLFloat16(0.03125f)), true);
    const int64_t rank = wrong_bits ? 32 : 16;
    test.AddInput<uint8_t>("AQ", {rank, 2, 32}, std::vector<uint8_t>(rank * 64, 129));
    test.AddInput<MLFloat16>("AS", {32, 2}, std::vector<MLFloat16>(64, MLFloat16(0.03125f)));
    test.AddInput<uint8_t>("BQ", {128, 1, 32}, std::vector<uint8_t>(4096, 129));
    test.AddInput<MLFloat16>("BS", {128, 1}, std::vector<MLFloat16>(128, MLFloat16(0.0625f)));
    test.AddOutput<MLFloat16>("Y", {1, 1, 128}, std::vector<MLFloat16>(128, MLFloat16(3.0f)));
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(std::move(ep));
    test.Run(OpTester::ExpectResult::kExpectFailure,
             wrong_bits ? "Unsupported MatMulNBitsQLoRA attributes" : "Invalid quantized adapter dimensions", {},
             nullptr, &eps);
  }
}

template <typename T>
void RunQuantizedPrefillAndReusedInput() {
  for (int64_t m : {1, 2, 3, 4, 65}) {
    for (int64_t accuracy : {0, 4}) {
      OpTester test("MatMulNBitsQLoRA", 1, kMSDomain);
      constexpr int64_t k = 256;
      constexpr int64_t n = 128;
      constexpr int64_t batches = 2;
      test.AddAttribute<int64_t>("K", k);
      test.AddAttribute<int64_t>("N", n);
      test.AddAttribute<int64_t>("accuracy_level", accuracy);
      std::vector<T> x(batches * m * k);
      std::vector<T> expected(batches * m * n);
      for (int64_t row = 0; row < batches * m; ++row) {
        const float value = row % 2 == 0 ? 0.5f : -0.25f;
        std::fill_n(x.begin() + row * k, k, T(value));
        std::fill_n(expected.begin() + row * n, n, T(value * 24.0f));
      }
      test.AddInput<T>("X", {batches, m, k}, x);
      test.AddInput<uint8_t>("W", {n, k / 32, 16}, std::vector<uint8_t>(n * k / 2, 0x99), true);
      test.AddInput<T>("S", {n, k / 32}, std::vector<T>(n * k / 32, T(0.03125f)), true);
      test.AddInput<uint8_t>("AQ", {32, k / 32, 32}, std::vector<uint8_t>(32 * k, 129));
      test.AddInput<T>("AS", {32, k / 32}, std::vector<T>(k, T(0.03125f)));
      test.AddInput<uint8_t>("BQ", {n, 1, 32}, std::vector<uint8_t>(n * 32, 129));
      test.AddInput<T>("BS", {n, 1}, std::vector<T>(n, T(0.0625f)));
      test.AddOutput<T>("Y", {batches, m, n}, expected);
      std::vector<std::unique_ptr<IExecutionProvider>> eps;
      eps.push_back(DefaultWebGpuExecutionProvider());
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
    }
  }
}

TEST(MatMulNBitsQLoRA, QuantizedPrefillAndReusedInput) { RunQuantizedPrefillAndReusedInput<MLFloat16>(); }
TEST(MatMulNBitsQLoRA, QuantizedPrefillAndReusedInputFp32) { RunQuantizedPrefillAndReusedInput<float>(); }

void RunFp32Numerics(const ConfigOptions& provider_options = {}) {
  struct Shape {
    int64_t m, k, n, batch;
  };
  for (const auto shape : {Shape{1, 64, 128, 1}, Shape{1, 2048, 36, 1}, Shape{1, 2048, 64, 1}, Shape{1, 8192, 132, 1},
                           Shape{1, 2176, 64, 7}, Shape{1, 8192, 64, 1},
                           Shape{2, 2048, 36, 2}, Shape{2, 8192, 64, 1}, Shape{3, 3072, 64, 3},
                           Shape{37, 2048, 132, 2}, Shape{128, 3072, 64, 1}}) {
    const auto [m, k, n, batch] = shape;
    SCOPED_TRACE(MakeString("M=", m, ", K=", k, ", N=", n, ", batch=", batch));
    std::vector<float> x(batch * m * k), as(k), bs(n), expected(batch * m * n);
    std::vector<uint8_t> aq(32 * k), bq(n * 32);
    for (int64_t row = 0; row < batch * m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        // Exactly representable by DP4A quantization, including zero rows.
        const int value = col % 128 == 0 ? 127 : static_cast<int>((row * 19 + col * 11) % 255) - 127;
        x[row * k + col] = row % 7 == 6 ? 0.0f : value / 128.0f;
      }
    }
    for (int64_t r = 0; r < 32; ++r) {
      for (int64_t col = 0; col < k; ++col)
        aq[r * k + col] = static_cast<uint8_t>(96 + (r * 17 + col * 13) % 65);
      for (int64_t block = 0; block < k / 32; ++block)
        as[r * (k / 32) + block] = 0.001234567f * (1 + (r + block) % 4);
    }
    for (int64_t col = 0; col < n; ++col) {
      bs[col] = 0.002345678f * (1 + col % 3);
      for (int64_t r = 0; r < 32; ++r)
        bq[col * 32 + r] = static_cast<uint8_t>(112 + (col * 7 + r * 3) % 33);
    }
    // Independent dense FP64 reference, rounding weights and intermediate T
    // to FP32. Non-half-representable scales detect accidental FP16 arithmetic.
    for (int64_t row = 0; row < batch * m; ++row) {
      double base = 0;
      for (int64_t col = 0; col < k; ++col) base += x[row * k + col] * 0.03125;
      std::vector<float> t(32);
      for (int64_t r = 0; r < 32; ++r) {
        double sum = 0;
        for (int64_t col = 0; col < k; ++col) {
          const float weight = (static_cast<int>(aq[r * k + col]) - 128) * as[r * (k / 32) + col / 32];
          sum += static_cast<double>(x[row * k + col]) * weight;
        }
        t[r] = static_cast<float>(sum);
      }
      for (int64_t col = 0; col < n; ++col) {
        double delta = 0;
        for (int64_t r = 0; r < 32; ++r) {
          const float weight = (static_cast<int>(bq[col * 32 + r]) - 128) * bs[col];
          delta += static_cast<double>(t[r]) * weight;
        }
        expected[row * n + col] = static_cast<float>(base) + static_cast<float>(delta);
      }
    }
    for (int64_t accuracy : {0, 4}) {
      SCOPED_TRACE(accuracy);
      OpTester test("MatMulNBitsQLoRA", 1, kMSDomain);
      test.AddAttribute<int64_t>("K", k);
      test.AddAttribute<int64_t>("N", n);
      test.AddAttribute<int64_t>("accuracy_level", accuracy);
      test.AddInput<float>("X", {batch, m, k}, x);
      test.AddInput<uint8_t>("W", {n, k / 32, 16}, std::vector<uint8_t>(n * k / 2, 0x99), true);
      test.AddInput<float>("S", {n, k / 32}, std::vector<float>(n * k / 32, 0.03125f), true);
      test.AddInput<uint8_t>("AQ", {32, k / 32, 32}, aq);
      test.AddInput<float>("AS", {32, k / 32}, as);
      test.AddInput<uint8_t>("BQ", {n, 1, 32}, bq);
      test.AddInput<float>("BS", {n, 1}, bs);
      test.AddOutput<float>("Y", {batch, m, n}, expected);
      test.SetOutputAbsErr("Y", 0.00001f);
      test.SetOutputRelErr("Y", 0.00001f);
      std::vector<std::unique_ptr<IExecutionProvider>> eps;
      eps.push_back(WebGpuExecutionProviderWithOptions(provider_options));
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
    }
  }
}

TEST(MatMulNBitsQLoRA, Fp32Numerics) { RunFp32Numerics(); }

TEST(MatMulNBitsQLoRA, Fp32NumericsWithoutSubgroups) {
#if defined(ORT_USE_EP_API_ADAPTERS) || defined(__wasm__) || defined(USE_EXTERNAL_DAWN)
  GTEST_SKIP() << "Requires native Dawn to create a device without subgroups.";
#else
  // Keep the default context alive to initialize Dawn's proc table.
  auto initial_ep = DefaultWebGpuExecutionProvider();
  ASSERT_NE(initial_ep, nullptr);
  const auto instance_feature = wgpu::InstanceFeatureName::TimedWaitAny;
  wgpu::InstanceDescriptor instance_desc{};
  instance_desc.requiredFeatureCount = 1;
  instance_desc.requiredFeatures = &instance_feature;
  dawn::native::Instance instance{&instance_desc};
  wgpu::RequestAdapterOptions adapter_options{};
  adapter_options.backendType = static_cast<wgpu::BackendType>(webgpu::WebGpuContextConfig{}.backend_type);
  auto adapters = instance.EnumerateAdapters(&adapter_options);
  ASSERT_FALSE(adapters.empty());

  const auto device_feature = wgpu::FeatureName::ImplicitDeviceSynchronization;
  wgpu::DeviceDescriptor device_desc{};
  device_desc.requiredFeatureCount = 1;
  device_desc.requiredFeatures = &device_feature;
  auto device = wgpu::Device::Acquire(adapters.front().CreateDevice(&device_desc));
  ASSERT_NE(device, nullptr);
  ASSERT_FALSE(device.HasFeature(wgpu::FeatureName::Subgroups));

  ConfigOptions options;
  ASSERT_STATUS_OK(options.AddConfigEntry(webgpu::options::kDeviceId, "1"));
  ASSERT_STATUS_OK(options.AddConfigEntry(
      webgpu::options::kWebGpuInstance, std::to_string(reinterpret_cast<uintptr_t>(instance.Get())).c_str()));
  ASSERT_STATUS_OK(options.AddConfigEntry(
      webgpu::options::kWebGpuDevice, std::to_string(reinterpret_cast<uintptr_t>(device.Get())).c_str()));
  auto external_ep = WebGpuExecutionProviderWithOptions(options);
  ASSERT_NE(external_ep, nullptr);
  ASSERT_FALSE(webgpu::WebGpuContextFactory::GetContext(1).DeviceHasFeature(wgpu::FeatureName::Subgroups));
  // Exercise the joint decode workgroup-memory reduction, including K tails,
  // and the prefill fallback against the same dense FP64 reference.
  RunFp32Numerics(options);
#endif
}

void RunFp32DP4AQuantization(int64_t m) {
  SCOPED_TRACE(m);
#if defined(ORT_USE_EP_API_ADAPTERS)
  GTEST_SKIP() << "Hardware-specific specialization checks require direct adapter capability inspection.";
#else
  auto initial_ep = DefaultWebGpuExecutionProvider();
  ASSERT_NE(initial_ep, nullptr);
  if (!contrib::webgpu::HasDP4ADeviceSupport(initial_ep->GetDeviceId())) {
    GTEST_SKIP() << "Requires DP4A device support.";
  }
#endif
  constexpr int64_t k = 2176, n = 64, batch = 3;
  const int64_t rows = batch * m;
  std::vector<float> x(rows * k), quantized_x(rows * k), as(k), expected(rows * n);
  std::vector<uint8_t> aq(32 * k), bq(n * 32, 128);
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      const float scale = 1.0f / static_cast<float>(1 << (8 + (row + col / 128) % 3));
      const int value = col % 128 == 0 ? 127 : static_cast<int>((row * 19 + col * 11) % 255) - 127;
      quantized_x[row * k + col] = row == 1 ? 0.0f : value * scale;
      // Stay inside the rounding interval of the known quantization grid.
      const float offset = value == 127 || value == -127 ? 0.0f : 0.2f;
      x[row * k + col] = row == 1 ? 0.0f : (value + offset) * scale;
    }
  }
  for (int64_t r = 0; r < 32; ++r) {
    for (int64_t col = 0; col < k; ++col)
      aq[r * k + col] = static_cast<uint8_t>((r * 17 + col * 13) % 256);
    for (int64_t block = 0; block < k / 32; ++block)
      as[r * (k / 32) + block] = (1 + (r + block) % 4) / 2048.0f;
  }
  for (int64_t col = 0; col < n; ++col) bq[col * 32 + col % 32] = 129;
  // An identity up projection exposes every T value. The reference uses
  // dense FP64 products on the known grid, including a zero row and a K tail.
  for (int64_t row = 0; row < rows; ++row) {
    double base = 0;
    for (int64_t col = 0; col < k; ++col) base += quantized_x[row * k + col] * 0.03125;
    for (int64_t r = 0; r < 32; ++r) {
      double sum = 0;
      for (int64_t col = 0; col < k; ++col)
        sum += static_cast<double>(quantized_x[row * k + col]) *
               (static_cast<int>(aq[r * k + col]) - 128) * as[r * (k / 32) + col / 32];
      expected[row * n + r] = expected[row * n + r + 32] = static_cast<float>(base) + static_cast<float>(sum);
    }
  }
  OpTester test("MatMulNBitsQLoRA", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", k);
  test.AddAttribute<int64_t>("N", n);
  test.AddAttribute<int64_t>("accuracy_level", 4);
  test.AddInput<float>("X", {batch, m, k}, x);
  test.AddInput<uint8_t>("W", {n, k / 32, 16}, std::vector<uint8_t>(n * k / 2, 0x99), true);
  test.AddInput<float>("S", {n, k / 32}, std::vector<float>(n * k / 32, 0.03125f), true);
  test.AddInput<uint8_t>("AQ", {32, k / 32, 32}, aq);
  test.AddInput<float>("AS", {32, k / 32}, as);
  test.AddInput<uint8_t>("BQ", {n, 1, 32}, bq);
  test.AddInput<float>("BS", {n, 1}, std::vector<float>(n, 1.0f));
  test.AddOutput<float>("Y", {batch, m, n}, expected);
  test.SetOutputAbsErr("Y", 0.00001f);
  test.SetOutputRelErr("Y", 0.00001f);
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(DefaultWebGpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
}

TEST(MatMulNBitsQLoRA, DecodeJointDP4AQuantization) { RunFp32DP4AQuantization(1); }

TEST(MatMulNBitsQLoRA, SmallMPrefillDP4AQuantization) {
  for (int64_t m : {2, 3}) RunFp32DP4AQuantization(m);
}

TEST(MatMulNBitsQLoRA, NarrowKParallelReduction) {
#if defined(ORT_USE_EP_API_ADAPTERS)
  GTEST_SKIP() << "Hardware-specific specialization checks require direct adapter capability inspection.";
#else
  auto initial_ep = DefaultWebGpuExecutionProvider();
  ASSERT_NE(initial_ep, nullptr);
  if (!contrib::webgpu::HasDP4ADeviceSupport(initial_ep->GetDeviceId())) {
    GTEST_SKIP() << "Requires a device that selects DP4A for the rank-32 down projection.";
  }
#endif
  struct Shape {
    int64_t m;
    int64_t k;
    int64_t batch;
  };
  for (const auto shape : {Shape{4, 128, 1}, Shape{3, 1920, 1}, Shape{4, 1920, 1}, Shape{4, 2048, 1},
                           Shape{37, 2176, 2}, Shape{128, 3072, 1},
                           Shape{65, 8192, 1}, Shape{4097, 128, 1}}) {
    const auto [m, k, batch] = shape;
    SCOPED_TRACE(MakeString("M=", m, ", K=", k, ", batch=", batch));
    constexpr int64_t n = 64;
    std::vector<MLFloat16> x(batch * m * k);
    std::vector<uint8_t> aq(32 * k);
    std::vector<MLFloat16> as(k);
    std::vector<uint8_t> bq(n * 32, 128);
    std::vector<MLFloat16> bs(n, MLFloat16(1.0f));
    std::vector<MLFloat16> expected(batch * m * n);
    for (int64_t row = 0; row < batch * m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        // The maximum in every 128-element block is 127 * scale, so DP4A's
        // input quantization preserves these signed integer values exactly.
        const float scale = 1.0f / static_cast<float>(1 << (7 + (row + col / 128) % 4));
        const int value = col % 128 == 0 ? 127 : static_cast<int>((row * 19 + col * 11) % 255) - 127;
        x[row * k + col] = MLFloat16(row % 7 == 0 ? 0.0f : value * scale);
      }
    }
    for (int64_t r = 0; r < 32; ++r) {
      for (int64_t col = 0; col < k; ++col) {
        aq[r * k + col] = static_cast<uint8_t>((r * 17 + col * 13) % 256);
      }
      for (int64_t block = 0; block < k / 32; ++block) {
        as[r * (k / 32) + block] = MLFloat16(1.0f / static_cast<float>(1 << (8 + (r + block) % 4)));
      }
    }
    // A zero base and an identity up projection expose every element of T.
    // Compute the reference as a dense dequantized dot product in double.
    for (int64_t col = 0; col < n; ++col) {
      bq[col * 32 + col % 32] = 129;
    }
    for (int64_t row = 0; row < batch * m; ++row) {
      for (int64_t r = 0; r < 32; ++r) {
        double sum = 0;
        for (int64_t col = 0; col < k; ++col) {
          sum += static_cast<double>(x[row * k + col].ToFloat()) *
                 (static_cast<int>(aq[r * k + col]) - 128) * as[r * (k / 32) + col / 32].ToFloat();
        }
        expected[row * n + r] = expected[row * n + r + 32] = MLFloat16(static_cast<float>(sum));
      }
    }
    OpTester test("MatMulNBitsQLoRA", 1, kMSDomain);
    test.AddAttribute<int64_t>("K", k);
    test.AddAttribute<int64_t>("N", n);
    test.AddAttribute<int64_t>("accuracy_level", 4);
    test.AddInput<MLFloat16>("X", {batch, m, k}, x);
    test.AddInput<uint8_t>("W", {n, k / 32, 16}, std::vector<uint8_t>(n * k / 2, 0x88), true);
    test.AddInput<MLFloat16>("S", {n, k / 32}, std::vector<MLFloat16>(n * k / 32, MLFloat16(1.0f)), true);
    test.AddInput<uint8_t>("AQ", {32, k / 32, 32}, aq);
    test.AddInput<MLFloat16>("AS", {32, k / 32}, as);
    test.AddInput<uint8_t>("BQ", {n, 1, 32}, bq);
    test.AddInput<MLFloat16>("BS", {n, 1}, bs);
    test.AddOutput<MLFloat16>("Y", {batch, m, n}, expected);
    test.SetOutputAbsErr("Y", 0.002f);
    test.SetOutputRelErr("Y", 0.001f);
    ConfigOptions options;
    // Short-K fallbacks do not force FP32 accumulation like the narrow DP4A
    // kernel. Enable it explicitly for this tight dense-reference comparison.
    if (k < 2048) {
      ASSERT_STATUS_OK(options.AddConfigEntry(webgpu::options::kEnableMatmulFp32Accumulation,
                                              webgpu::options::kEnableMatmulFp32Accumulation_ON));
    }
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(WebGpuExecutionProviderWithOptions(options));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  }
}

TEST(MatMulNBitsQLoRA, DecodeSubgroupReduction) {
#if defined(ORT_USE_EP_API_ADAPTERS)
  GTEST_SKIP() << "Hardware-specific specialization checks require direct adapter capability inspection.";
#else
  auto initial_ep = DefaultWebGpuExecutionProvider();
  ASSERT_NE(initial_ep, nullptr);
  auto& gpu = webgpu::WebGpuContextFactory::GetContext(initial_ep->GetDeviceId());
  const auto& adapter = gpu.AdapterInfo();
  const bool subgroup_32 = (adapter.subgroupMinSize == 32 && adapter.subgroupMaxSize == 32) ||
                           (gpu.DeviceHasFeature(wgpu::FeatureName::SubgroupSizeControl) &&
                            adapter.subgroupMinSize <= 32 && adapter.subgroupMaxSize >= 32);
  if (!gpu.DeviceHasFeature(wgpu::FeatureName::Subgroups) || !subgroup_32) {
    GTEST_SKIP() << "The decode specialization requires fixed or selectable 32-lane subgroups.";
  }
#endif
  for (bool acc_f32 : {false, true}) {
    for (const auto& [k, n] : {std::pair<int64_t, int64_t>{2048, 36}, {4096, 132}, {8192, 64}}) {
      SCOPED_TRACE(MakeString("K=", k, ", N=", n, ", acc_f32=", acc_f32));
      std::vector<MLFloat16> x(k);
      std::vector<uint8_t> aq(32 * k);
      std::vector<MLFloat16> as(k);
      std::vector<uint8_t> bq(n * 32, 128);
      std::vector<MLFloat16> expected(n);
      double base = 0;
      for (int64_t col = 0; col < k; ++col) {
        x[col] = MLFloat16((static_cast<int>(col % 17) - 8) / 64.0f);
        base += x[col].ToFloat() * 0.03125;
      }
      std::vector<MLFloat16> t(32);
      for (int64_t rank = 0; rank < 32; ++rank) {
        for (int64_t block = 0; block < k / 32; ++block) {
          as[rank * (k / 32) + block] = MLFloat16(0.001f * (1 + (rank + block) % 4));
        }
        double sum = 0;
        for (int64_t col = 0; col < k; ++col) {
          const int q = rank == 0 ? 0 : static_cast<int>((rank * 19 + col * 13) % 65) - 32;
          aq[rank * k + col] = static_cast<uint8_t>(q + 128);
          const auto weight = MLFloat16(q * as[rank * (k / 32) + col / 32].ToFloat());
          sum += static_cast<double>(x[col].ToFloat()) * weight.ToFloat();
        }
        t[rank] = MLFloat16(static_cast<float>(sum));
      }
      // Identity rows expose all 32 T values; the nonzero base also exercises
      // ordinary workgroups in the same dispatch and partial output tiles.
      const auto rounded_base = MLFloat16(static_cast<float>(base));
      for (int64_t col = 0; col < n; ++col) {
        bq[col * 32 + col % 32] = 129;
        expected[col] = MLFloat16(rounded_base.ToFloat() + t[col % 32].ToFloat());
      }
      OpTester test("MatMulNBitsQLoRA", 1, kMSDomain);
      test.AddAttribute<int64_t>("K", k);
      test.AddAttribute<int64_t>("N", n);
      test.AddInput<MLFloat16>("X", {1, 1, k}, x);
      test.AddInput<uint8_t>("W", {n, k / 32, 16}, std::vector<uint8_t>(n * k / 2, 0x99), true);
      test.AddInput<MLFloat16>("S", {n, k / 32}, std::vector<MLFloat16>(n * k / 32, MLFloat16(0.03125f)), true);
      test.AddInput<uint8_t>("AQ", {32, k / 32, 32}, aq);
      test.AddInput<MLFloat16>("AS", {32, k / 32}, as);
      test.AddInput<uint8_t>("BQ", {n, 1, 32}, bq);
      test.AddInput<MLFloat16>("BS", {n, 1}, std::vector<MLFloat16>(n, MLFloat16(1.0f)));
      test.AddOutput<MLFloat16>("Y", {1, 1, n}, expected);
      // The long-K half-precision lane accumulation can drift before the
      // FP32 subgroup reduction (up to 0.00238 on RTX 4070 at K=8192).
      // Keep the tighter bound for shorter reductions and FP32 accumulation.
      test.SetOutputAbsErr("Y", !acc_f32 && k == 8192 ? 0.003f : 0.002f);
      test.SetOutputRelErr("Y", 0.001f);
      ConfigOptions options;
      if (acc_f32) {
        ASSERT_STATUS_OK(options.AddConfigEntry(webgpu::options::kEnableMatmulFp32Accumulation,
                                                webgpu::options::kEnableMatmulFp32Accumulation_ON));
      }
      std::vector<std::unique_ptr<IExecutionProvider>> eps;
      eps.push_back(WebGpuExecutionProviderWithOptions(options));
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
    }
  }
}

#endif
}  // namespace onnxruntime::test
