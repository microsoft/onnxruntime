// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime::test {
#if defined(USE_WEBGPU) && !defined(DISABLE_CONTRIB_OPS)

template <typename T>
void RunEmptyAdapterSkipsProjection() {
  for (const char* mode : {"", "auto", "fuse_down", "fuse_up"}) {
    SCOPED_TRACE(mode[0] ? mode : "default");
    for (bool transposed : {false, true}) {
      for (int64_t m : {1, 37}) {
        for (auto ranks : {std::pair{0, 0}, {0, 32}, {32, 0}, {32, 32}}) {
          auto ep = DefaultWebGpuExecutionProvider();
          ASSERT_NE(ep, nullptr);
          OpTester test("MatMulNBitsLoRA", 1, kMSDomain);
          test.AddAttribute<int64_t>("K", 64);
          test.AddAttribute<int64_t>("N", 128);
          test.AddAttribute<int64_t>("bits", 4);
          test.AddAttribute<int64_t>("block_size", 32);
          test.AddAttribute<int64_t>("adapter_transposed", transposed);
          test.AddInput<T>("X", {1, m, 64}, std::vector<T>(m * 64, T(0.5f)));
          // Packed nibbles 9 => dequantized weight 1/32, so the base is exactly 1.
          test.AddInput<uint8_t>("W", {128, 2, 16}, std::vector<uint8_t>(128 * 2 * 16, 0x99), true);
          test.AddInput<T>("scales", {128, 2}, std::vector<T>(256, T(0.03125f)), true);
          test.AddInput<T>("lora_A", transposed ? std::vector<int64_t>{ranks.first, 64} : std::vector<int64_t>{64, ranks.first},
                           std::vector<T>(64 * ranks.first, T(0.0625f)));
          test.AddInput<T>("lora_B", transposed ? std::vector<int64_t>{128, ranks.second} : std::vector<int64_t>{ranks.second, 128},
                           std::vector<T>(128 * ranks.second, T(0.03125f)));
          const float expected = ranks.first && ranks.second ? 3.0f : 1.0f;
          test.AddOutput<T>("Y", {1, m, 128}, std::vector<T>(m * 128, T(expected)));
          std::vector<std::unique_ptr<IExecutionProvider>> eps;
          eps.push_back(std::move(ep));
          SessionOptions options;
          if (mode[0]) {
            ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.webgpu.lora_fusion_mode", mode));
          }
          test.Config(options);
          test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
        }
      }
    }
  }
}

TEST(MatMulNBitsLoRA, EmptyAdapterSkipsProjection) { RunEmptyAdapterSkipsProjection<MLFloat16>(); }
TEST(MatMulNBitsLoRA, EmptyAdapterSkipsProjectionFp32) { RunEmptyAdapterSkipsProjection<float>(); }

TEST(MatMulNBitsLoRA, Fp32Numerics) {
  struct Shape {
    int64_t m, k, n, batch;
  };
  for (const auto shape : {Shape{1, 64, 128, 1}, Shape{1, 2048, 36, 1}, Shape{1, 2048, 64, 1}, Shape{1, 8192, 132, 1},
                           Shape{2, 2048, 36, 2}, Shape{37, 2048, 132, 2}, Shape{128, 3072, 64, 1}}) {
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
    std::vector<float> at(32 * k), bt(n * 32);
    for (int64_t r = 0; r < 32; ++r)
      for (int64_t col = 0; col < k; ++col)
        at[r * k + col] = (static_cast<int>(aq[r * k + col]) - 128) * as[r * (k / 32) + col / 32];
    for (int64_t col = 0; col < n; ++col)
      for (int64_t r = 0; r < 32; ++r)
        bt[col * 32 + r] = (static_cast<int>(bq[col * 32 + r]) - 128) * bs[col];
    // Independent dense FP64 reference, rounding intermediate T
    // to FP32. Non-half-representable weights detect accidental FP16 arithmetic.
    for (int64_t row = 0; row < batch * m; ++row) {
      double base = 0;
      for (int64_t col = 0; col < k; ++col) base += x[row * k + col] * 0.03125;
      std::vector<float> t(32);
      for (int64_t r = 0; r < 32; ++r) {
        double sum = 0;
        for (int64_t col = 0; col < k; ++col) {
          const float weight = at[r * k + col];
          sum += static_cast<double>(x[row * k + col]) * weight;
        }
        t[r] = static_cast<float>(sum);
      }
      for (int64_t col = 0; col < n; ++col) {
        double delta = 0;
        for (int64_t r = 0; r < 32; ++r) {
          const float weight = bt[col * 32 + r];
          delta += static_cast<double>(t[r]) * weight;
        }
        expected[row * n + col] = static_cast<float>(base) + static_cast<float>(delta);
      }
    }
    for (const char* mode : {"auto", "fuse_down", "fuse_up"}) {
      for (int64_t accuracy : {0, 4}) {
        SCOPED_TRACE(MakeString(mode, ", accuracy=", accuracy));
        OpTester test("MatMulNBitsLoRA", 1, kMSDomain);
        test.AddAttribute<int64_t>("K", k);
        test.AddAttribute<int64_t>("N", n);
        test.AddAttribute<int64_t>("accuracy_level", accuracy);
        test.AddAttribute<int64_t>("adapter_transposed", 1);
        test.AddInput<float>("X", {batch, m, k}, x);
        test.AddInput<uint8_t>("W", {n, k / 32, 16}, std::vector<uint8_t>(n * k / 2, 0x99), true);
        test.AddInput<float>("S", {n, k / 32}, std::vector<float>(n * k / 32, 0.03125f), true);
        test.AddInput<float>("A", {32, k}, at);
        test.AddInput<float>("B", {n, 32}, bt);
        test.AddOutput<float>("Y", {batch, m, n}, expected);
        test.SetOutputAbsErr("Y", 0.00001f);
        test.SetOutputRelErr("Y", 0.00001f);
        SessionOptions options;
        ASSERT_STATUS_OK(options.config_options.AddConfigEntry("session.webgpu.lora_fusion_mode", mode));
        test.Config(options);
        std::vector<std::unique_ptr<IExecutionProvider>> eps;
        eps.push_back(DefaultWebGpuExecutionProvider());
        test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
      }
    }
  }
}

template <typename T>
void RunParallelDownNumerics() {
  struct Shape {
    int64_t batch, m, k, n;
  };
  for (const auto shape : {Shape{1, 1, 2048, 64}, Shape{2, 1, 8192, 64}, Shape{2, 33, 2176, 64},
                           Shape{1, 128, 4096, 64}, Shape{1, 1024, 2048, 16}}) {
    const auto [batch, m, k, n] = shape;
    SCOPED_TRACE(MakeString("batch=", batch, ", M=", m, ", K=", k, ", N=", n));
    std::vector<T> x(batch * m * k), a(32 * k), b(n * 32), expected(batch * m * n);
    for (int64_t row = 0; row < batch * m; ++row) {
      for (int64_t col = 0; col < k; ++col) {
        // Off-grid floating X detects accidental reuse of the base's INT8 X.
        x[row * k + col] = T(row % 7 == 6 ? 0.0f : (static_cast<int>((row * 19 + col * 11) % 255) - 127) * 0.00712345f);
      }
    }
    for (int64_t r = 0; r < 32; ++r) {
      for (int64_t col = 0; col < k; ++col)
        a[r * k + col] = T((static_cast<int>((r * 17 + col * 13) % 65) - 32) * 0.001234567f);
    }
    for (int64_t col = 0; col < n; ++col)
      for (int64_t r = 0; r < 32; ++r)
        b[col * 32 + r] = T((static_cast<int>((col * 7 + r * 3) % 33) - 16) * 0.002345678f);
    // Dense FP64 reference with exactly one cast at each MatMul boundary.
    // A zero base isolates adapter arithmetic from activation quantization.
    for (int64_t row = 0; row < batch * m; ++row) {
      std::vector<T> t(32);
      for (int64_t r = 0; r < 32; ++r) {
        double sum = 0;
        for (int64_t col = 0; col < k; ++col)
          sum += static_cast<double>(static_cast<float>(x[row * k + col])) * static_cast<float>(a[r * k + col]);
        t[r] = T(static_cast<float>(sum));
      }
      for (int64_t col = 0; col < n; ++col) {
        double sum = 0;
        for (int64_t r = 0; r < 32; ++r)
          sum += static_cast<double>(static_cast<float>(t[r])) * static_cast<float>(b[col * 32 + r]);
        expected[row * n + col] = T(static_cast<float>(sum));
      }
    }
    OpTester test("MatMulNBitsLoRA", 1, kMSDomain);
    test.AddAttribute<int64_t>("K", k);
    test.AddAttribute<int64_t>("N", n);
    test.AddAttribute<int64_t>("accuracy_level", 4);
    test.AddAttribute<int64_t>("adapter_transposed", 1);
    test.AddInput<T>("X", {batch, m, k}, x);
    test.AddInput<uint8_t>("W", {n, k / 32, 16}, std::vector<uint8_t>(n * k / 2, 0x88), true);
    test.AddInput<T>("S", {n, k / 32}, std::vector<T>(n * k / 32, T(0.03125f)), true);
    test.AddInput<T>("A", {32, k}, a);
    test.AddInput<T>("B", {n, 32}, b);
    test.AddOutput<T>("Y", {batch, m, n}, expected);
    test.SetOutputAbsErr("Y", std::is_same_v<T, float> ? 2e-6f : 1e-4f);
    test.SetOutputRelErr("Y", std::is_same_v<T, float> ? 2e-5f : 0.003f);
    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultWebGpuExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &eps);
  }
}

TEST(MatMulNBitsLoRA, ParallelDownNumericsFp16) { RunParallelDownNumerics<MLFloat16>(); }
TEST(MatMulNBitsLoRA, ParallelDownNumericsFp32) { RunParallelDownNumerics<float>(); }

TEST(MatMulNBitsLoRA, RejectInvalidAdapterShape) {
  auto ep = DefaultWebGpuExecutionProvider();
  ASSERT_NE(ep, nullptr);
  OpTester test("MatMulNBitsLoRA", 1, kMSDomain);
  test.AddAttribute<int64_t>("K", 64);
  test.AddAttribute<int64_t>("N", 128);
  test.AddInput<MLFloat16>("X", {1, 64}, std::vector<MLFloat16>(64));
  test.AddInput<uint8_t>("W", {128, 2, 16}, std::vector<uint8_t>(4096), true);
  test.AddInput<MLFloat16>("scales", {128, 2}, std::vector<MLFloat16>(256), true);
  test.AddInput<MLFloat16>("lora_A", {63, 32}, std::vector<MLFloat16>(63 * 32));
  test.AddInput<MLFloat16>("lora_B", {32, 128}, std::vector<MLFloat16>(32 * 128));
  test.AddOutput<MLFloat16>("Y", {1, 128}, std::vector<MLFloat16>(128));
  std::vector<std::unique_ptr<IExecutionProvider>> eps;
  eps.push_back(std::move(ep));
  test.Run(OpTester::ExpectResult::kExpectFailure, "Invalid adapter dimensions", {}, nullptr, &eps);
}

#endif
}  // namespace onnxruntime::test
