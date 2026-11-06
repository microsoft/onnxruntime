// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <memory>
#include <vector>

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL)
namespace onnxruntime::test {
namespace {

void RunPacked(OpTester& test) {
  test.SetOutputTolerance(0, 0);
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

void ConfigureFixedQ8(OpTester& test, int k, int n) {
  test.AddAttribute("K", static_cast<int64_t>(k));
  test.AddAttribute("N", static_cast<int64_t>(n));
  test.AddAttribute("bits", int64_t{8});
  test.AddAttribute("block_size", int64_t{32});
  test.AddAttribute("gemv_kernel", std::string("fixed"));
}

void ConfigureFixedIQ2(OpTester& test, int k, int n) {
  test.AddAttribute("K", static_cast<int64_t>(k));
  test.AddAttribute("N", static_cast<int64_t>(n));
  test.AddAttribute("bits", int64_t{2});
  test.AddAttribute("block_size", int64_t{256});
  test.AddAttribute("gemv_kernel", std::string("fixed_iq2_xxs"));
}

template <typename S>
void AddPackedScales(OpTester& test, int n, int blocks, const std::vector<S>& scales) {
  test.AddOptionalInputEdge<BFloat16>();
  test.AddOptionalInputEdge<uint8_t>();
  test.AddOptionalInputEdge<int32_t>();
  test.AddOptionalInputEdge<BFloat16>();
  test.AddInput<S>("fixed_scales", {n, blocks}, scales);
}

template <typename S>
void CheckQ8(int rows, int n) {
  constexpr int k = 256;
  std::vector<BFloat16> a(rows * k, BFloat16(0.0f));
  std::vector<uint8_t> b(n * k);
  std::vector<S> scales(n * k / 32);
  std::vector<BFloat16> expected(rows * n);
  for (int col = 0; col < n; ++col) {
    for (int block = 0; block < k / 32; ++block) {
      scales[col * (k / 32) + block] = S(0.001003f * (1 + col % 5 + block));
    }
    for (int kk = 0; kk < k; ++kk) b[col * k + kk] = static_cast<uint8_t>(col * 31 + kk);
  }
  for (int r = 0; r < rows; ++r) {
    const int kk = r * 32 + 3;
    a[r * k + kk] = BFloat16(1.0f);
    for (int col = 0; col < n; ++col) {
      expected[r * n + col] =
          BFloat16(static_cast<float>(scales[col * (k / 32) + kk / 32]) * (int(b[col * k + kk]) - 128));
    }
  }
  OpTester test("MatMulNBits", 1, kMSDomain);
  ConfigureFixedQ8(test, k, n);
  // Non-matrix leading dimensions must survive flattening.
  test.AddInput<BFloat16>("A", {1, rows, k}, a);
  test.AddInput<uint8_t>("B", {n, k / 32, 32}, b);
  AddPackedScales(test, n, k / 32, scales);
  test.AddOutput<BFloat16>("Y", {1, rows, n}, expected);
  RunPacked(test);
}

void CheckIQ2(int rows, int n) {
  constexpr int k = 256;
  std::vector<BFloat16> a(rows * k, BFloat16(0.0f));
  std::vector<uint8_t> b(n * k / 4, 0);
  std::vector<MLFloat16> scales(n);
  std::vector<BFloat16> expected(rows * n);
  for (int col = 0; col < n; ++col) {
    const auto d = MLFloat16(0.001003f * (1 + col % 7));
    scales[col] = d;
    for (int block = 0; block < 8; ++block) {
      auto* record = b.data() + (col * 8 + block) * 8;
      // Grid entry 1: {43,8,8,8,8,8,8,8}; sign index 1 negates element zero.
      record[0] = record[1] = record[2] = record[3] = 1;
      record[4] = 1;
      record[7] = static_cast<uint8_t>((block + col % 3) << 4);
    }
    for (int r = 0; r < rows; ++r) {
      const float block_scale = d.ToFloat() * (0.5f + float(r + col % 3)) * 0.25f;
      expected[r * n + col] = BFloat16(-43.0f * block_scale);
    }
  }
  for (int r = 0; r < rows; ++r) a[r * k + r * 32] = BFloat16(1.0f);
  OpTester test("MatMulNBits", 1, kMSDomain);
  ConfigureFixedIQ2(test, k, n);
  const std::vector<int64_t> shape{1, rows, k};
  test.AddInput<BFloat16>("A", shape, a);
  test.AddInput<uint8_t>("B", {n, k / 256, 64}, b);
  AddPackedScales(test, n, k / 256, scales);
  auto output_shape = shape;
  output_shape.back() = n;
  test.AddOutput<BFloat16>("Y", output_shape, expected);
  RunPacked(test);
}

void CheckNVFP4(int rows, int n) {
  constexpr int k = 4096;
  constexpr std::array<float, 8> values{0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
  std::vector<BFloat16> a(rows * k, BFloat16(0.0f));
  std::vector<uint8_t> b(n * k / 2);
  std::vector<uint8_t> scales(n * k / 16);
  std::vector<BFloat16> expected(rows * n);
  for (int col = 0; col < n; ++col) {
    const uint8_t code = static_cast<uint8_t>(col % 16);
    std::fill_n(b.begin() + col * k / 2, k / 2, static_cast<uint8_t>(code * 17));
    std::fill_n(scales.begin() + col * k / 16, k / 16, uint8_t{0x3a});  // E4M3 1.25.
    const float weight = values[code & 7] * ((code & 8) ? -1.0f : 1.0f);
    for (int r = 0; r < rows; ++r) expected[r * n + col] = BFloat16(weight * 1.25f * 0.33333f);
  }
  for (int r = 0; r < rows; ++r) a[r * k + r * 16] = BFloat16(1.0f);
  OpTester test("MatMulBlockQuantizedFp4Weight", 1, kMSDomain);
  test.AddAttribute("gemv_kernel", std::string("fixed"));
  test.AddInput<BFloat16>("A", {rows, k}, a);
  test.AddInput<uint8_t>("B", {n, k / 2}, b);
  test.AddInput<uint8_t>("scales", {n, k / 16}, scales);
  test.AddInput<float>("global_scale", {1}, {0.33333f});
  test.AddOutput<BFloat16>("Y", {rows, n}, expected);
  RunPacked(test);
}

// Dyadic fixtures keep sums exact in FP32, so a scalar oracle is independent of
// the kernels' MMA tiling, warp reduction order, and GEMV iteration counts.
template <typename S>
void CheckQ8Chunks(int rows, int n, int k) {
  SCOPED_TRACE(testing::Message() << "rows=" << rows << " N=" << n << " K=" << k);
  std::vector<BFloat16> a(rows * k, BFloat16(0.0f));
  std::vector<uint8_t> b(n * k);
  std::vector<S> scales(n * k / 32);
  std::vector<BFloat16> expected(rows * n);
  for (int r = 0; r < rows; ++r) {
    for (int block = 0; block < k / 32; ++block) {
      for (int offset : {0, 7, 16, 31}) {
        a[r * k + block * 32 + offset] = BFloat16(float((r + 1) * (1 + block % 3)) / 8.0f);
      }
    }
  }
  for (int col = 0; col < n; ++col) {
    for (int block = 0; block < k / 32; ++block) {
      scales[col * (k / 32) + block] = S(float(1 + col % 3 + block % 5) / 256.0f);
    }
    for (int kk = 0; kk < k; ++kk)
      b[col * k + kk] = uint8_t(129 + (col + 3 * kk + kk / 32) % 7);
    for (int r = 0; r < rows; ++r) {
      float sum = 0.0f;
      for (int block = 0; block < k / 32; ++block) {
        for (int offset : {0, 7, 16, 31}) {
          const int kk = block * 32 + offset;
          const float weight = float(int(b[col * k + kk]) - 128) *
                               static_cast<float>(scales[col * (k / 32) + block]);
          sum += static_cast<float>(a[r * k + kk]) * weight;
        }
      }
      expected[r * n + col] = BFloat16(sum);
    }
  }
  OpTester test("MatMulNBits", 1, kMSDomain);
  ConfigureFixedQ8(test, k, n);
  test.AddInput<BFloat16>("A", {rows, k}, a);
  test.AddInput<uint8_t>("B", {n, k / 32, 32}, b);
  AddPackedScales(test, n, k / 32, scales);
  test.AddOutput<BFloat16>("Y", {rows, n}, expected);
  RunPacked(test);
}

void CheckIQ2Chunks(int rows, int n) {
  constexpr int k = 1280;  // 20 MMA chunks: multiple iterations and a final partial KW=16 batch.
  constexpr std::array<std::array<float, 8>, 3> grids{
      {{8, 8, 8, 8, 8, 8, 8, 8},
       {43, 8, 8, 8, 8, 8, 8, 8},
       {25, 25, 8, 8, 8, 8, 8, 8}}};
  SCOPED_TRACE(testing::Message() << "rows=" << rows << " N=" << n);
  std::vector<BFloat16> a(rows * k, BFloat16(0.0f));
  std::vector<uint8_t> b(n * k / 4, 0);
  std::vector<MLFloat16> scales(n * k / 256);
  std::vector<BFloat16> expected(rows * n);
  for (int r = 0; r < rows; ++r) {
    for (int block = 0; block < k / 32; ++block) {
      for (int offset : {0, 7, 8, 31})
        a[r * k + block * 32 + offset] = BFloat16(float(r + 1) / 8.0f);
    }
  }
  for (int col = 0; col < n; ++col) {
    std::array<float, 8> sums{};
    for (int block = 0; block < k / 32; ++block) {
      const auto d = MLFloat16(float(1 + col % 3 + (block / 8) % 2) / 1024.0f);
      scales[col * (k / 256) + block / 8] = d;
      auto* record = b.data() + (col * (k / 32) + block) * 8;
      uint32_t sign_and_scale = uint32_t(block % 16) << 28;
      for (int group = 0; group < 4; ++group) {
        record[group] = uint8_t((col + block + group) % 3);
        const uint32_t signs = uint32_t(1 + (col + block + group * 17) % 127);
        sign_and_scale |= signs << (7 * group);
        uint32_t parity = 0;
        for (int bit = 0; bit < 7; ++bit)
          parity ^= (signs >> bit) & 1;
        for (int element = 0; element < 8; ++element) {
          const int offset = group * 8 + element;
          if (offset != 0 && offset != 7 && offset != 8 && offset != 31)
            continue;
          const bool negative = element == 7 ? parity != 0 : ((signs >> element) & 1) != 0;
          const float magnitude = grids[record[group]][element];
          const float block_scale = d.ToFloat() * (0.5f + float(block % 16)) * 0.25f;
          const float weight = static_cast<float>(BFloat16((negative ? -magnitude : magnitude) * block_scale));
          for (int r = 0; r < rows; ++r)
            sums[r] += static_cast<float>(a[r * k + block * 32 + offset]) * weight;
        }
      }
      for (int byte = 0; byte < 4; ++byte)
        record[4 + byte] = uint8_t(sign_and_scale >> (8 * byte));
    }
    for (int r = 0; r < rows; ++r)
      expected[r * n + col] = BFloat16(sums[r]);
  }
  OpTester test("MatMulNBits", 1, kMSDomain);
  ConfigureFixedIQ2(test, k, n);
  test.AddInput<BFloat16>("A", {rows, k}, a);
  test.AddInput<uint8_t>("B", {n, k / 256, 64}, b);
  AddPackedScales(test, n, k / 256, scales);
  test.AddOutput<BFloat16>("Y", {rows, n}, expected);
  RunPacked(test);
}

void CheckNVFP4Chunks(int rows, int n, int k) {
  constexpr std::array<float, 8> values{0, 0.5f, 1, 1.5f, 2, 3, 4, 6};
  constexpr std::array<uint8_t, 4> scale_codes{0x30, 0x38, 0x3a, 0x40};
  constexpr std::array<float, 4> scale_values{0.5f, 1.0f, 1.25f, 2.0f};
  SCOPED_TRACE(testing::Message() << "rows=" << rows << " N=" << n << " K=" << k);
  std::vector<BFloat16> a(rows * k, BFloat16(0.0f));
  std::vector<uint8_t> b(n * k / 2);
  std::vector<uint8_t> scales(n * k / 16);
  std::vector<BFloat16> expected(rows * n);
  for (int r = 0; r < rows; ++r) {
    for (int block = 0; block < k / 16; ++block) {
      a[r * k + block * 16 + (block + r) % 16] = BFloat16(float(1 + (block + r) % 3) / 64.0f);
    }
  }
  for (int col = 0; col < n; ++col) {
    for (int block = 0; block < k / 16; ++block) {
      const int code = 1 + (col + block) % 7;
      const int scale_index = (col + block) % 4;
      std::fill_n(b.begin() + col * (k / 2) + block * 8, 8, uint8_t(code * 17));
      scales[col * (k / 16) + block] = scale_codes[scale_index];
    }
    for (int r = 0; r < rows; ++r) {
      float sum = 0.0f;
      for (int block = 0; block < k / 16; ++block) {
        const int kk = block * 16 + (block + r) % 16;
        sum += static_cast<float>(a[r * k + kk]) * values[1 + (col + block) % 7] *
               scale_values[(col + block) % 4];
      }
      expected[r * n + col] = BFloat16(sum * 0.33333f);
    }
  }
  OpTester test("MatMulBlockQuantizedFp4Weight", 1, kMSDomain);
  test.AddAttribute("gemv_kernel", std::string("fixed"));
  test.AddInput<BFloat16>("A", {rows, k}, a);
  test.AddInput<uint8_t>("B", {n, k / 2}, b);
  test.AddInput<uint8_t>("scales", {n, k / 16}, scales);
  test.AddInput<float>("global_scale", {1}, {0.33333f});
  test.AddOutput<BFloat16>("Y", {rows, n}, expected);
  RunPacked(test);
}

}  // namespace

TEST(MatMulPackedProjection, Q8SourceScalePrecisionAndRowTails) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  for (int rows : {1, 3, 8}) {
    CheckQ8<MLFloat16>(rows, 19);
    CheckQ8<BFloat16>(rows, 19);
  }
}

TEST(MatMulNBitsFixedBF16, IndependentFP16ScalePrecisionAndLeadingDimensions) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  for (int rows = 1; rows <= 8; ++rows) {
    for (int n : {1, 17, 33}) CheckQ8<MLFloat16>(rows, n);
  }
}

TEST(MatMulNBitsFixedBF16, MultipleChunksAndLargeNDispatch) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  CheckQ8Chunks<MLFloat16>(3, 19, 1088);
  CheckQ8Chunks<MLFloat16>(3, 8193, 1088);
  CheckQ8Chunks<MLFloat16>(1, 19, 4096);
  CheckQ8Chunks<MLFloat16>(8, 19, 4096);
}

TEST(MatMulNBitsFixedBF16, IndependentBF16ScalePrecisionAndLeadingDimensions) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  for (int rows = 1; rows <= 8; ++rows) {
    for (int n : {1, 17, 33}) CheckQ8<BFloat16>(rows, n);
  }
}

TEST(MatMulNBitsFixedBF16, BF16ScaleChunksAndRowOneBoundary) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  CheckQ8Chunks<BFloat16>(3, 19, 1088);
  CheckQ8Chunks<BFloat16>(3, 8193, 1088);
  CheckQ8Chunks<BFloat16>(1, 19, 4096);
  CheckQ8Chunks<BFloat16>(1, 19, 6144);
  CheckQ8Chunks<BFloat16>(3, 19, 6208);
  CheckQ8Chunks<MLFloat16>(1, 19, 6208);
}

TEST(MatMulNBitsFixedBF16, BF16ScaleRowOneKeepsBlockDotBeforeScale) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  constexpr int k = 64;
  for (int n : {19, 4097}) {
    std::vector<BFloat16> a(k, BFloat16(0.f));
    a[0] = BFloat16(1.f);
    a[1] = BFloat16(-1.f);
    std::vector<uint8_t> b(n * k, 128);
    for (int col = 0; col < n; ++col) {
      b[col * k] = 255;
      b[col * k + 1] = 254;
    }
    // (127 - 126) * scale is 1.0078125; rounding weights before the dot gives 1.
    const auto scale = BFloat16::FromBits(0x3f81);
    OpTester test("MatMulNBits", 1, kMSDomain);
    ConfigureFixedQ8(test, k, n);
    test.AddInput<BFloat16>("A", {1, k}, a);
    test.AddInput<uint8_t>("B", {n, k / 32, 32}, b);
    AddPackedScales(test, n, k / 32, std::vector<BFloat16>(n * k / 32, scale));
    test.AddOutput<BFloat16>("Y", {1, n}, std::vector<BFloat16>(n, scale));
    RunPacked(test);
  }
}

#ifndef ORT_NO_EXCEPTIONS
TEST(MatMulNBitsFixedBF16, RejectsBF16ScaleRowOneBeyondSupportedK) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  constexpr int k = 6208;
  OpTester test("MatMulNBits", 1, kMSDomain);
  ConfigureFixedQ8(test, k, 1);
  test.AddInput<BFloat16>("A", {1, k}, std::vector<BFloat16>(k, BFloat16(1.f)));
  test.AddInput<uint8_t>("B", {1, k / 32, 32}, std::vector<uint8_t>(k, 129));
  AddPackedScales(test, 1, k / 32, std::vector<BFloat16>(k / 32, BFloat16(1.f)));
  test.AddOutput<BFloat16>("Y", {1, 1}, {BFloat16(0.f)});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "BF16-scale row-one path requires K <= 6144",
           {}, nullptr, &providers);
}

TEST(MatMulNBitsFixedBF16, RejectsUnsupportedRowsAndKInsteadOfFallingBack) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  for (bool bad_k : {false, true}) {
    const int rows = bad_k ? 1 : 9, k = bad_k ? 32 : 64;
    OpTester test("MatMulNBits", 1, kMSDomain);
    ConfigureFixedQ8(test, k, 1);
    test.AddInput<BFloat16>("A", {rows, k}, std::vector<BFloat16>(rows * k, BFloat16(1.f)));
    test.AddInput<uint8_t>("B", {1, k / 32, 32}, std::vector<uint8_t>(k, 129));
    AddPackedScales(test, 1, k / 32, std::vector<MLFloat16>(k / 32, MLFloat16(1.f)));
    test.AddOutput<BFloat16>("Y", {rows, 1}, std::vector<BFloat16>(rows, BFloat16(0.f)));
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectFailure,
             bad_k ? "positive int32 K divisible by 64" : "flattened activation rows must be in [1,8]",
             {}, nullptr, &providers);
  }
}

TEST(MatMulNBitsFixedBF16, DefaultDispatchStillRequiresScales) {
  OpTester test("MatMulNBits", 1, kMSDomain);
  test.AddAttribute("K", int64_t{64});
  test.AddAttribute("N", int64_t{1});
  test.AddAttribute("bits", int64_t{8});
  test.AddAttribute("block_size", int64_t{32});
  test.AddInput<float>("A", {1, 64}, std::vector<float>(64, 1.f));
  test.AddInput<uint8_t>("B", {1, 2, 32}, std::vector<uint8_t>(64, 129));
  test.AddOutput<float>("Y", {1, 1}, {0.f});
  test.Run(OpTester::ExpectResult::kExpectFailure, "auto requires scales");
}
#endif

TEST(MatMulNBitsFixedIQ2XXS, NonlinearOperandsLiteralHalfScalesAndLeadingDimensions) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  for (int rows = 1; rows <= 8; ++rows) {
    for (int n : {1, 19, 33}) CheckIQ2(rows, n);
  }
}

TEST(MatMulNBitsFixedIQ2XXS, MultipleBlocksPartialChunksAndLargeNDispatch) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  for (int rows : {1, 3, 8}) {
    for (int n : {19, 8192, 8193}) CheckIQ2Chunks(rows, n);
  }
}

#ifndef ORT_NO_EXCEPTIONS
TEST(MatMulNBitsFixedIQ2XXS, RejectsBF16ScaleInterpretation) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  OpTester test("MatMulNBits", 1, kMSDomain);
  ConfigureFixedIQ2(test, 256, 1);
  test.AddInput<BFloat16>("A", {1, 256}, std::vector<BFloat16>(256, BFloat16(0.f)));
  test.AddInput<uint8_t>("B", {1, 1, 64}, std::vector<uint8_t>(64));
  AddPackedScales(test, 1, 1, std::vector<BFloat16>{BFloat16(1.f)});
  test.AddOutput<BFloat16>("Y", {1, 1}, {BFloat16(0.f)});
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectFailure, "requires literal FP16 fixed_scales",
           {}, nullptr, &providers);
}

TEST(MatMulNBitsFixedIQ2XXS, RejectsAffineProfilesAndMismatchedCodecMetadata) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  struct Case {
    int64_t bits, block;
    const char* kernel;
  };
  for (const Case spec : {Case{8, 256, "fixed_iq2_xxs"}, Case{2, 32, "fixed_iq2_xxs"},
                          Case{2, 256, "fixed"}}) {
    OpTester test("MatMulNBits", 1, kMSDomain);
    test.AddAttribute("K", int64_t{256});
    test.AddAttribute("N", int64_t{1});
    test.AddAttribute("bits", spec.bits);
    test.AddAttribute("block_size", spec.block);
    test.AddAttribute("gemv_kernel", std::string(spec.kernel));
    test.AddInput<BFloat16>("A", {1, 256}, std::vector<BFloat16>(256, BFloat16(0.f)));
    test.AddInput<uint8_t>("B", {1, 1, 64}, std::vector<uint8_t>(64));
    AddPackedScales(test, 1, 1, std::vector<MLFloat16>{MLFloat16(0.001003f)});
    test.AddOutput<BFloat16>("Y", {1, 1}, {BFloat16(0.f)});
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectFailure,
             std::string("MatMulNBits ") + spec.kernel + " requires", {}, nullptr, &providers);
  }
}

TEST(MatMulNBitsFixedIQ2XXS, RejectsUnsupportedRowsAndKInsteadOfFallingBack) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  for (bool bad_k : {false, true}) {
    const int rows = bad_k ? 1 : 9, k = bad_k ? 128 : 256;
    OpTester test("MatMulNBits", 1, kMSDomain);
    ConfigureFixedIQ2(test, k, 1);
    test.AddInput<BFloat16>("A", {rows, k}, std::vector<BFloat16>(rows * k, BFloat16(0.f)));
    test.AddInput<uint8_t>("B", {1, 1, 64}, std::vector<uint8_t>(64));
    AddPackedScales(test, 1, 1, std::vector<MLFloat16>{MLFloat16(0.001003f)});
    test.AddOutput<BFloat16>("Y", {rows, 1}, std::vector<BFloat16>(rows, BFloat16(0.f)));
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectFailure,
             bad_k ? "positive int32 K divisible by 256" : "flattened activation rows must be in [1,8]",
             {}, nullptr, &providers);
  }
}
#endif

TEST(MatMulPackedProjection, IQ2CompleteOperandRoundingAndLiteralHalfScales) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  for (int rows : {1, 3, 8}) {
    CheckIQ2(rows, 19);
  }
}

TEST(MatMulPackedProjection, Q8LargeNMmaAndColumnTails) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  for (int n : {19, 8192, 8193}) {
    CheckQ8Chunks<MLFloat16>(3, n, 1088);
    CheckQ8Chunks<BFloat16>(8, n, 1088);
  }
}

TEST(MatMulPackedProjection, Q8MmaReductionOrderCancellation) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  constexpr int rows = 3, k = 1088;
  std::vector<BFloat16> a(rows * k, BFloat16(0.f));
  a[0] = BFloat16(0x1p24f);
  a[64] = BFloat16(1.f);
  a[256] = BFloat16(-0x1p24f);
  a[k + 64] = BFloat16(2.f);
  a[2 * k + 256] = BFloat16(-2.f);
  for (int n : {19, 8192}) {
    for (bool fp16_scale : {false, true}) {
      SCOPED_TRACE(testing::Message() << "N=" << n << " fp16_scale=" << fp16_scale);
      std::vector<BFloat16> expected(rows * n);
      // KW=16 loses the unit before cancellation; KW=4 cancels the large terms first.
      std::fill_n(expected.begin(), n, BFloat16(n == 19 ? 0.f : 1.f));
      std::fill_n(expected.begin() + n, n, BFloat16(2.f));
      std::fill_n(expected.begin() + 2 * n, n, BFloat16(-2.f));
      OpTester test("MatMulNBits", 1, kMSDomain);
      ConfigureFixedQ8(test, k, n);
      test.AddInput<BFloat16>("A", {rows, k}, a);
      test.AddInput<uint8_t>("B", {n, k / 32, 32}, std::vector<uint8_t>(n * k, 129));
      if (fp16_scale)
        AddPackedScales(test, n, k / 32, std::vector<MLFloat16>(n * k / 32, MLFloat16(1.f)));
      else
        AddPackedScales(test, n, k / 32, std::vector<BFloat16>(n * k / 32, BFloat16(1.f)));
      test.AddOutput<BFloat16>("Y", {rows, n}, expected);
      RunPacked(test);
    }
  }
}

TEST(MatMulPackedProjection, Q8BFloatRowOneIterationAndKBoundaries) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  // K/32=34/66/130 selects IT=2/4/6 with guarded load tails; 192 fills IT=6.
  for (int k : {1088, 2112, 4160, 6144}) {
    CheckQ8Chunks<BFloat16>(1, 19, k);
    CheckQ8Chunks<BFloat16>(1, 8193, k);
  }
  CheckQ8Chunks<BFloat16>(1, 8192, 1088);
}

TEST(MatMulPackedProjection, IQ2MultipleChunksAndLargeNMma) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  for (int n : {19, 8192, 8193}) {
    CheckIQ2Chunks(3, n);
    CheckIQ2Chunks(8, n);
  }
}

TEST(MatMulPackedProjection, Q8BFloatScaleRowOneScalesAfterIntegerDot) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  std::vector<BFloat16> a(64, BFloat16(0.0f));
  a[0] = BFloat16(125.0f);
  a[1] = BFloat16(-127.0f);
  std::vector<uint8_t> b(64, uint8_t{128});
  b[0] = 255;
  b[1] = 253;
  OpTester test("MatMulNBits", 1, kMSDomain);
  ConfigureFixedQ8(test, 64, 1);
  test.AddInput<BFloat16>("A", {1, 64}, a);
  test.AddInput<uint8_t>("B", {1, 2, 32}, b);
  AddPackedScales(test, 1, 2, std::vector<BFloat16>{BFloat16(0.00314f), BFloat16(0.017f)});
  test.AddOutput<BFloat16>("Y", {1, 1}, {BFloat16(0.0f)});
  RunPacked(test);
}

TEST(MatMulPackedProjection, NVFP4HalfPairsAndMmaWithGlobalScale) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  for (int rows : {1, 3, 8}) CheckNVFP4(rows, 19);
}

TEST(MatMulPackedProjection, NVFP4MultipleChunksAndLargeNBoundaries) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  for (int n : {19, 8192, 8193}) {
    for (int k : {4096, 8192}) {
      CheckNVFP4Chunks(1, n, k);
      CheckNVFP4Chunks(3, n, k);
    }
  }
}

TEST(MatMulPackedProjection, NVFP4RowOneHalfPairCancellation) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  for (int k : {4096, 8192}) {
    for (int n : {1, 8193}) {
      SCOPED_TRACE(testing::Message() << "N=" << n << " K=" << k);
      std::vector<BFloat16> a(k, BFloat16(0.0f));
      a[0] = BFloat16(16384.0f);
      a[1] = BFloat16(1.0f);
      a[2] = BFloat16(-16384.0f);
      // k=0,1,2 enter the SAME low FP16 accumulator (paired with k=4,5,6).
      // After x/64, half(256 + 1/64) is 256, then -256 gives 0; FP32 gives 1.
      float fp32_sum = 0.0f;
      MLFloat16 half_sum(0.0f);
      for (int kk : {0, 1, 2}) {
        fp32_sum += static_cast<float>(a[kk]);
        half_sum = MLFloat16(half_sum.ToFloat() + static_cast<float>(a[kk]) / 64.0f);
      }
      ASSERT_EQ(fp32_sum, 1.0f);
      ASSERT_EQ(half_sum.ToFloat() * 64.0f, 0.0f);
      OpTester test("MatMulBlockQuantizedFp4Weight", 1, kMSDomain);
      test.AddAttribute("gemv_kernel", std::string("fixed"));
      test.AddInput<BFloat16>("A", {1, k}, a);
      // Unit E2M1 weights and unit E4M3 scales.
      test.AddInput<uint8_t>("B", {n, k / 2}, std::vector<uint8_t>(n * k / 2, uint8_t{0x22}));
      test.AddInput<uint8_t>("scales", {n, k / 16}, std::vector<uint8_t>(n * k / 16, uint8_t{0x38}));
      test.AddInput<float>("global_scale", {1}, {1.0f});
      test.AddOutput<BFloat16>("Y", {1, n}, std::vector<BFloat16>(n, BFloat16(0.0f)));
      RunPacked(test);
    }
  }
}

#ifndef ORT_NO_EXCEPTIONS
TEST(MatMulPackedProjection, RejectsUnsupportedPrefillRows) {
  if (!HasCudaEnvironment(800)) GTEST_SKIP() << "CUDA SM80 is required.";
  for (bool add_shapes : {true, false}) {
    SCOPED_TRACE(testing::Message() << "graph shapes=" << add_shapes);
    OpTester test("MatMulNBits", 1, kMSDomain);
    ConfigureFixedQ8(test, 64, 1);
    test.AddShapeToTensorData(add_shapes);
    test.AddInput<BFloat16>("A", {9, 64}, std::vector<BFloat16>(9 * 64));
    test.AddInput<uint8_t>("B", {1, 2, 32}, std::vector<uint8_t>(64));
    AddPackedScales(test, 1, 2, std::vector<MLFloat16>{MLFloat16(1.0f), MLFloat16(1.0f)});
    test.AddOutput<BFloat16>("Y", {9, 1}, std::vector<BFloat16>(9));
    std::vector<std::unique_ptr<IExecutionProvider>> providers;
    providers.push_back(DefaultCudaExecutionProvider());
    test.Run(OpTester::ExpectResult::kExpectFailure, "flattened activation rows must be in [1,8]", {}, nullptr,
             &providers);
  }
}
#endif

}  // namespace onnxruntime::test
#endif
