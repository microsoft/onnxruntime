// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#if defined(USE_WEBGPU) && !defined(DISABLE_CONTRIB_OPS)
#if !defined(ORT_USE_EP_API_ADAPTERS)
#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"
#include "core/providers/webgpu/webgpu_context.h"
#endif
#include "core/providers/webgpu/webgpu_provider_options.h"
#endif

namespace onnxruntime::test {
#if defined(USE_WEBGPU) && !defined(DISABLE_CONTRIB_OPS)

TEST(MatMulNBitsQLoRA, EmptyAndActiveAdapters) {
  for (const char* mode : {"auto", "fuse_down", "fuse_up"}) {
    for (int64_t m : {1, 37}) {
      for (auto ranks : {std::pair{0, 0}, {0, 1}, {32, 0}, {32, 1}}) {
        auto ep = DefaultWebGpuExecutionProvider();
        ASSERT_NE(ep, nullptr);
        OpTester test("MatMulNBitsQLoRA", 1, kMSDomain);
        test.AddAttribute<int64_t>("K", 64);
        test.AddAttribute<int64_t>("N", 128);
        test.AddInput<MLFloat16>("X", {1, m, 64}, std::vector<MLFloat16>(m * 64, MLFloat16(0.5f)));
        test.AddInput<uint8_t>("W", {128, 2, 16}, std::vector<uint8_t>(4096, 0x99), true);
        test.AddInput<MLFloat16>("S", {128, 2}, std::vector<MLFloat16>(256, MLFloat16(0.03125f)), true);
        test.AddInput<uint8_t>("AQ", {ranks.first, 2, 32}, std::vector<uint8_t>(ranks.first * 64, 129));
        test.AddInput<MLFloat16>("AS", {32, 2}, std::vector<MLFloat16>(64, MLFloat16(0.03125f)));
        test.AddInput<uint8_t>("BQ", {128, ranks.second, 32}, std::vector<uint8_t>(128 * ranks.second * 32, 129));
        test.AddInput<MLFloat16>("BS", {128, 1}, std::vector<MLFloat16>(128, MLFloat16(0.0625f)));
        const float expected = ranks.first && ranks.second ? 3.0f : 1.0f;
        test.AddOutput<MLFloat16>("Y", {1, m, 128}, std::vector<MLFloat16>(m * 128, MLFloat16(expected)));
        SessionOptions options;
        ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.webgpu.lora_fusion_mode", mode));
        test.Config(options);
        std::vector<std::unique_ptr<IExecutionProvider>> eps;
        eps.push_back(std::move(ep));
        test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
      }
    }
  }
}

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

TEST(MatMulNBitsQLoRA, QuantizedPrefillAndReusedInput) {
  for (int64_t m : {1, 3, 4, 65}) {
    for (int64_t accuracy : {0, 4}) {
      OpTester test("MatMulNBitsQLoRA", 1, kMSDomain);
      constexpr int64_t k = 256;
      constexpr int64_t n = 128;
      constexpr int64_t batches = 2;
      test.AddAttribute<int64_t>("K", k);
      test.AddAttribute<int64_t>("N", n);
      test.AddAttribute<int64_t>("accuracy_level", accuracy);
      std::vector<MLFloat16> x(batches * m * k);
      std::vector<MLFloat16> expected(batches * m * n);
      for (int64_t row = 0; row < batches * m; ++row) {
        const float value = row % 2 == 0 ? 0.5f : -0.25f;
        std::fill_n(x.begin() + row * k, k, MLFloat16(value));
        std::fill_n(expected.begin() + row * n, n, MLFloat16(value * 24.0f));
      }
      test.AddInput<MLFloat16>("X", {batches, m, k}, x);
      test.AddInput<uint8_t>("W", {n, k / 32, 16}, std::vector<uint8_t>(n * k / 2, 0x99), true);
      test.AddInput<MLFloat16>("S", {n, k / 32}, std::vector<MLFloat16>(n * k / 32, MLFloat16(0.03125f)), true);
      test.AddInput<uint8_t>("AQ", {32, k / 32, 32}, std::vector<uint8_t>(32 * k, 129));
      test.AddInput<MLFloat16>("AS", {32, k / 32}, std::vector<MLFloat16>(k, MLFloat16(0.03125f)));
      test.AddInput<uint8_t>("BQ", {n, 1, 32}, std::vector<uint8_t>(n * 32, 129));
      test.AddInput<MLFloat16>("BS", {n, 1}, std::vector<MLFloat16>(n, MLFloat16(0.0625f)));
      test.AddOutput<MLFloat16>("Y", {batches, m, n}, expected);
      std::vector<std::unique_ptr<IExecutionProvider>> eps;
      eps.push_back(DefaultWebGpuExecutionProvider());
      test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
    }
  }
}

TEST(MatMulNBitsQLoRA, NarrowSmallMReduction) {
#if defined(ORT_USE_EP_API_ADAPTERS)
  GTEST_SKIP() << "Hardware-specific specialization checks require direct adapter capability inspection.";
#else
  auto initial_ep = DefaultWebGpuExecutionProvider();
  ASSERT_NE(initial_ep, nullptr);
  auto& gpu = webgpu::WebGpuContextFactory::GetContext(initial_ep->GetDeviceId());
  if (!contrib::webgpu::HasDP4ADeviceSupport(initial_ep->GetDeviceId()) ||
      gpu.AdapterInfo().vendor != std::string_view{"nvidia"}) {
    GTEST_SKIP() << "The narrow small-M projection is selected on NVIDIA DP4A devices.";
  }
#endif
  struct Shape {
    int64_t m;
    int64_t k;
    int64_t batch;
  };
  for (const auto shape : {Shape{4, 2048, 1}, Shape{37, 2176, 2}, Shape{128, 3072, 1}, Shape{65, 8192, 1}}) {
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
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultWebGpuExecutionProvider());
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
  if (!gpu.DeviceHasFeature(wgpu::FeatureName::Subgroups) ||
      adapter.vendor != std::string_view{"nvidia"} ||
      adapter.subgroupMinSize != 32 || adapter.subgroupMaxSize != 32) {
    GTEST_SKIP() << "The decode specialization requires NVIDIA 32-lane subgroups.";
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
