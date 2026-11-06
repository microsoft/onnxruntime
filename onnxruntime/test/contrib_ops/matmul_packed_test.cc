// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL)
namespace onnxruntime::test {
namespace {

void Configure(OpTester& test, int64_t k, int64_t n, const std::string& quant_type = "iq2_xxs") {
  test.AddAttribute("quant_type", quant_type);
  test.AddAttribute("K", k);
  test.AddAttribute("N", n);
}

void RunCuda(OpTester& test, OpTester::ExpectResult expected = OpTester::ExpectResult::kExpectSuccess,
             const std::string& error = "") {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(expected, error, {}, nullptr, &providers);
}

TEST(MatMulPackedTest, NativeIQ2XXSBlocks) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int k = 1280;
  constexpr std::array<std::array<float, 8>, 3> grids{{
      {8, 8, 8, 8, 8, 8, 8, 8},
      {43, 8, 8, 8, 8, 8, 8, 8},
      {25, 25, 8, 8, 8, 8, 8, 8},
  }};
  for (int rows : {1, 3, 8}) {
    for (int n : {19, 8193}) {
      SCOPED_TRACE(testing::Message() << "rows=" << rows << " N=" << n);
      std::vector<BFloat16> a(rows * k, BFloat16(0.f)), expected(rows * n);
      std::vector<uint8_t> b(n * (k / 256) * 66);
      for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < k / 256; ++block)
          a[row * k + block * 256 + row * 32 + row] = BFloat16(float(row + 1) / 8.f);
      }
      for (int col = 0; col < n; ++col) {
        std::array<float, 8> sums{};
        for (int block = 0; block < k / 256; ++block) {
          auto* bytes = b.data() + (col * (k / 256) + block) * 66;
          const MLFloat16 d(0.001003f * float(1 + col % 7 + block));
          bytes[0] = static_cast<uint8_t>(d.val);
          bytes[1] = static_cast<uint8_t>(d.val >> 8);
          for (int record = 0; record < 8; ++record) {
            const int local = (record + col) % 16;
            uint32_t packed = static_cast<uint32_t>(local) << 28;
            for (int group = 0; group < 4; ++group) {
              const int grid = (col + block + record + group) % 3;
              const uint32_t sign_index = (1 + col + block + record + group * 17) % 128;
              bytes[2 + record * 8 + group] = static_cast<uint8_t>(grid);
              packed |= sign_index << (7 * group);
              for (int row = 0; row < rows; ++row) {
                if (record != row || group != row / 8) continue;
                unsigned parity = 0;
                for (int bit = 0; bit < 7; ++bit) parity ^= (sign_index >> bit) & 1;
                const unsigned signs = sign_index | (parity << 7);
                const float sign = ((signs >> row) & 1) ? -1.f : 1.f;
                const float scale = d.ToFloat() * (0.5f + float(local)) * 0.25f;
                const BFloat16 weight(grids[grid][row] * sign * scale);
                sums[row] += weight.ToFloat() * float(row + 1) / 8.f;
              }
            }
            for (int byte = 0; byte < 4; ++byte)
              bytes[2 + record * 8 + 4 + byte] = static_cast<uint8_t>(packed >> (8 * byte));
          }
        }
        for (int row = 0; row < rows; ++row) expected[row * n + col] = BFloat16(sums[row]);
      }
      OpTester test("MatMulPacked", 1, kMSDomain);
      Configure(test, k, n);
      test.AddInput<BFloat16>("A", {1, rows, k}, a);
      test.AddInput<uint8_t>("B", {n, (k / 256) * 66}, b);
      test.AddOutput<BFloat16>("Y", {1, rows, n}, expected);
      test.SetOutputTolerance(0, 0);
      RunCuda(test);
    }
  }
}

TEST(MatMulPackedTest, NativeQ8Blocks) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr int k = 1280;
  for (int rows : {1, 3, 7, 8}) {
    for (int n : {19, 8193}) {
      SCOPED_TRACE(testing::Message() << "rows=" << rows << " N=" << n);
      std::vector<BFloat16> a(rows * k, BFloat16(0.f)), expected(rows * n);
      std::vector<uint8_t> b(n * (k / 32) * 34);
      for (int row = 0; row < rows; ++row)
        for (int block = 0; block < k / 32; ++block)
          a[row * k + block * 32 + row * 3] = BFloat16(float(row + 1) / 8.f);
      for (int col = 0; col < n; ++col) {
        std::array<float, 8> sums{};
        for (int block = 0; block < k / 32; ++block) {
          auto* bytes = b.data() + (col * (k / 32) + block) * 34;
          const MLFloat16 d(0.001003f * float(1 + col % 7 + block % 3));
          bytes[0] = static_cast<uint8_t>(d.val);
          bytes[1] = static_cast<uint8_t>(d.val >> 8);
          for (int element = 0; element < 32; ++element)
            bytes[2 + element] = static_cast<uint8_t>(block * 11 + col * 13 + element * 17);
          for (int row = 0; row < rows; ++row) {
            const int raw = bytes[2 + row * 3];
            const BFloat16 weight(float(raw < 128 ? raw : raw - 256) * d.ToFloat());
            sums[row] += weight.ToFloat() * float(row + 1) / 8.f;
          }
        }
        for (int row = 0; row < rows; ++row) expected[row * n + col] = BFloat16(sums[row]);
      }
      OpTester test("MatMulPacked", 1, kMSDomain);
      Configure(test, k, n, "q8_0");
      test.AddInput<BFloat16>("A", {1, rows, k}, a);
      test.AddInput<uint8_t>("B", {n, (k / 32) * 34}, b);
      test.AddOutput<BFloat16>("Y", {1, rows, n}, expected);
      test.SetOutputTolerance(0, 0);
      RunCuda(test);
    }
  }
}

TEST(MatMulPackedTest, NativeNVFP4Blocks) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  constexpr std::array<float, 16> values{
      0, 0.5f, 1, 1.5f, 2, 3, 4, 6, 0, -0.5f, -1, -1.5f, -2, -3, -4, -6};
  constexpr float global = 0.013379f;
  for (int k : {4096, 8192}) {
    for (int rows : {1, 3, 8}) {
      for (int n : {19, 8193}) {
        SCOPED_TRACE(testing::Message() << "K=" << k << " rows=" << rows << " N=" << n);
        std::vector<BFloat16> a(rows * k, BFloat16(0.f)), expected(rows * n);
        std::vector<uint8_t> b(n * (k / 64) * 36);
        for (int row = 0; row < rows; ++row) {
          const int position = row % 2 == 0 ? row * 17 : k - 1 - row * 19;
          a[row * k + position] = BFloat16(float(row + 1) / 8.f);
        }
        for (int col = 0; col < n; ++col) {
          for (int block = 0; block < k / 64; ++block) {
            auto* bytes = b.data() + (col * (k / 64) + block) * 36;
            for (int group = 0; group < 4; ++group) {
              bytes[group] = static_cast<uint8_t>((col + block * 13 + group * 7) % 128);
              for (int element = 0; element < 8; ++element) {
                const unsigned low = (col + block + group + element) % 16;
                const unsigned high = (col + block + group + element + 8) % 16;
                bytes[4 + group * 8 + element] = static_cast<uint8_t>(low | (high << 4));
              }
            }
          }
          for (int row = 0; row < rows; ++row) {
            const int position = row % 2 == 0 ? row * 17 : k - 1 - row * 19;
            const int block = position / 64, group = position % 64 / 16, element = position % 16;
            const unsigned scale = (col + block * 13 + group * 7) % 128;
            const float d = scale == 127 ? 0.f
                            : scale < 8  ? float(scale) / 512.f
                                         : std::ldexp(1.f + float(scale & 7) / 8.f, int(scale >> 3) - 7);
            const float weight = values[(col + block + group + element) % 16] * d;
            expected[row * n + col] = BFloat16(weight * float(row + 1) / 8.f * global);
          }
        }
        OpTester test("MatMulPacked", 1, kMSDomain);
        Configure(test, k, n, "nvfp4");
        test.AddInput<BFloat16>("A", {1, rows, k}, a);
        test.AddInput<uint8_t>("B", {n, (k / 64) * 36}, b);
        test.AddInput<float>("global_scale", {}, {global});
        test.AddOutput<BFloat16>("Y", {1, rows, n}, expected);
        test.SetOutputTolerance(0, 0);
        RunCuda(test);
      }
    }
  }
}

TEST(MatMulPackedTest, RejectsInvalidGlobalScale) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int variant : {0, 1, 2}) {
    const bool nvfp4 = variant != 1;
    const int k = nvfp4 ? 4096 : 64;
    OpTester test("MatMulPacked", 1, kMSDomain);
    Configure(test, k, 1, nvfp4 ? "nvfp4" : "q8_0");
    test.AddInput<BFloat16>("A", {1, k}, std::vector<BFloat16>(k, BFloat16(0.f)));
    test.AddInput<uint8_t>("B", {1, nvfp4 ? (k / 64) * 36 : (k / 32) * 34},
                           std::vector<uint8_t>(nvfp4 ? (k / 64) * 36 : (k / 32) * 34));
    if (variant == 2)
      test.AddInput<float>("global_scale", {2}, {1.f, 1.f});
    else if (variant == 1)
      test.AddInput<float>("global_scale", {}, {1.f});
    test.AddOutput<BFloat16>("Y", {1, 1}, {BFloat16(0.f)});
    RunCuda(test, OpTester::ExpectResult::kExpectFailure,
            variant == 2 ? "scalar or a one-element vector" : "required for NVFP4");
  }
}

TEST(MatMulPackedTest, RejectsUnsupportedNVFP4ContractionSize) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  OpTester test("MatMulPacked", 1, kMSDomain);
  Configure(test, 256, 1, "nvfp4");
  test.AddInput<BFloat16>("A", {1, 256}, std::vector<BFloat16>(256, BFloat16(0.f)));
  test.AddInput<uint8_t>("B", {1, 144}, std::vector<uint8_t>(144));
  test.AddInput<float>("global_scale", {}, {1.f});
  test.AddOutput<BFloat16>("Y", {1, 1}, {BFloat16(0.f)});
  RunCuda(test, OpTester::ExpectResult::kExpectFailure, "K=4096 or K=8192");
}

TEST(MatMulPackedTest, RejectsInvalidPayloadAndCodec) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  for (bool bad_codec : {false, true}) {
    OpTester test("MatMulPacked", 1, kMSDomain);
    Configure(test, 256, 1, bad_codec ? "unknown" : "iq2_xxs");
    test.AddInput<BFloat16>("A", {1, 256}, std::vector<BFloat16>(256, BFloat16(0.f)));
    test.AddInput<uint8_t>("B", {1, 65}, std::vector<uint8_t>(65));
    test.AddOutput<BFloat16>("Y", {1, 1}, {BFloat16(0.f)});
    RunCuda(test, OpTester::ExpectResult::kExpectFailure,
            bad_codec ? "quant_type='iq2_xxs'" : "B must have shape");
  }
}

TEST(MatMulPackedTest, RejectsUnsupportedQ8ContractionSize) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  OpTester test("MatMulPacked", 1, kMSDomain);
  Configure(test, 32, 1, "q8_0");
  test.AddInput<BFloat16>("A", {1, 32}, std::vector<BFloat16>(32, BFloat16(0.f)));
  test.AddInput<uint8_t>("B", {1, 34}, std::vector<uint8_t>(34));
  test.AddOutput<BFloat16>("Y", {1, 1}, {BFloat16(0.f)});
  RunCuda(test, OpTester::ExpectResult::kExpectFailure, "divisible by 256 (IQ2_XXS) or 64 (Q8_0)");
}

TEST(MatMulPackedTest, RejectsUnsupportedDynamicRowCount) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "Requires CUDA SM80+";
  OpTester test("MatMulPacked", 1, kMSDomain);
  Configure(test, 256, 1);
  test.AddInput<BFloat16>("A", {9, 256}, std::vector<BFloat16>(9 * 256, BFloat16(0.f)));
  test.AddInput<uint8_t>("B", {1, 66}, std::vector<uint8_t>(66));
  test.AddOutput<BFloat16>("Y", {9, 1}, std::vector<BFloat16>(9, BFloat16(0.f)));
  test.AddShapeToTensorData(false);
  RunCuda(test, OpTester::ExpectResult::kExpectFailure, "1 through 8 activation rows");
}

}  // namespace
}  // namespace onnxruntime::test
#endif
