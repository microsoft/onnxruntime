// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "gtest/gtest.h"
#include "contrib_ops/cpu/moe/qmoe_shared_layout.h"
#include "core/framework/moe_offload_plan.h"

namespace onnxruntime::test {
namespace {

namespace shared = contrib::qmoe_shared;

TEST(QMoESharedLayoutTest, StorageRangeChecksOffsetEvenWithoutExternalLength) {
  constexpr auto max_offset = std::numeric_limits<uint64_t>::max();
  size_t bytes = 123;
  EXPECT_FALSE(shared::ValidateExpertStorageRange(2, 256, max_offset - 511, 0, &bytes));
  EXPECT_EQ(bytes, 123U);
  EXPECT_FALSE(shared::ValidateExpertStorageRange(2, 256, max_offset - 511, 512));
  EXPECT_TRUE(shared::ValidateExpertStorageRange(2, 256, max_offset - 512, 0, &bytes));
  EXPECT_EQ(bytes, 512U);
  EXPECT_TRUE(shared::ValidateExpertStorageRange(2, 256, max_offset - 512, 512));
}

TEST(QMoESharedLayoutTest, TensorExpertStoreRejectsEmptyOrNullBuffers) {
  std::vector<uint8_t> raw(512 + shared::kBankAlignment, 0);
  const uintptr_t addr =
      (reinterpret_cast<uintptr_t>(raw.data()) + shared::kBankAlignment - 1) & ~(shared::kBankAlignment - 1);
  auto* aligned_ptr = reinterpret_cast<uint8_t*>(addr);
  shared::TensorExpertStore store;
  EXPECT_FALSE(store.Initialize(nullptr, 512, 2, 256, 256, 0));
  EXPECT_FALSE(store.Initialize(nullptr, 512, 2, 256, 256, 42));
  EXPECT_FALSE(store.Initialize(aligned_ptr, 0, 2, 256, 256, 0));
  EXPECT_FALSE(store.Initialize(aligned_ptr, 0, 2, 256, 256, 42));
  EXPECT_FALSE(store.Initialize(aligned_ptr, 256, 2, 256, 256, 42));
  EXPECT_TRUE(store.Initialize(aligned_ptr, 512, 2, 256, 256, 0));
}

TEST(QMoESharedLayoutTest, LargeAlignmentRequirementsSupportedInTensorAndFileStores) {
  constexpr size_t kLargeAlign = 8192;
  constexpr size_t kNumExperts = 2;
  constexpr size_t kTotalBytes = kNumExperts * kLargeAlign;
  void* raw = std::malloc(kTotalBytes + kLargeAlign);
  ASSERT_NE(raw, nullptr);
  const uintptr_t addr = (reinterpret_cast<uintptr_t>(raw) + kLargeAlign - 1) & ~(kLargeAlign - 1);
  auto* aligned_ptr = reinterpret_cast<uint8_t*>(addr);
  std::fill_n(aligned_ptr, kLargeAlign, uint8_t{0x12});
  std::fill_n(aligned_ptr + kLargeAlign, kLargeAlign, uint8_t{0x34});

  EXPECT_GE(shared::ComputePointerAlignment(aligned_ptr), kLargeAlign);
  shared::ExpertBlockLease lease{};
  ASSERT_TRUE(shared::MakeExpertBlockLease(aligned_ptr, kLargeAlign, kLargeAlign, &lease));
  EXPECT_GE(lease.alignment, kLargeAlign);

  shared::TensorExpertStore tensor_store;
  ASSERT_TRUE(tensor_store.Initialize(aligned_ptr, kTotalBytes, kNumExperts, kLargeAlign, kLargeAlign));
  ASSERT_TRUE(tensor_store.AcquireReady(0, &lease));
  EXPECT_GE(lease.alignment, kLargeAlign);
  EXPECT_EQ(lease.data[0], 0x12);
  ASSERT_TRUE(tensor_store.AcquireReady(1, &lease));
  EXPECT_GE(lease.alignment, kLargeAlign);
  EXPECT_EQ(lease.data[0], 0x34);

  const auto tmp_path = std::filesystem::temp_directory_path() / "ort_qmoe_align8192_test.bin";
  {
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out.write(reinterpret_cast<const char*>(aligned_ptr), static_cast<std::streamsize>(kTotalBytes));
  }
  shared::FileExpertStore file_store;
  ASSERT_TRUE(file_store.Initialize(tmp_path.string().c_str(), 0, kNumExperts, kLargeAlign, 2, kLargeAlign));
  ASSERT_TRUE(file_store.AcquireReady(0, &lease));
  EXPECT_GE(lease.alignment, kLargeAlign);
  EXPECT_EQ(lease.data[0], 0x12);
  file_store.ReleaseLease(0);
  ASSERT_TRUE(file_store.TryAcquireCached(0, &lease));
  EXPECT_GE(lease.alignment, kLargeAlign);
  file_store.ReleaseLease(0);
  const int32_t group_id = 0;
  ASSERT_TRUE(file_store.TryAcquireGroupCached(&group_id, 1, &lease));
  EXPECT_GE(lease.alignment, kLargeAlign);
  file_store.ReleaseGroupLeases(&group_id, 1);
  file_store.Shutdown();
  std::filesystem::remove(tmp_path);
  std::free(raw);
}

TEST(QMoESharedLayoutTest, FileStoreRetainsOpenHandleAcrossPathReplacement) {
  constexpr size_t kStride = shared::kBankAlignment;
  constexpr int64_t kNumExperts = 2;
  const auto dir = std::filesystem::temp_directory_path();
  const auto orig_path = dir / "ort_qmoe_handle_orig.bin";
  const auto moved_path = dir / "ort_qmoe_handle_moved.bin";
  std::filesystem::remove(orig_path);
  std::filesystem::remove(moved_path);

  {
    std::vector<uint8_t> initial(kNumExperts * kStride);
    std::fill_n(initial.data(), kStride, uint8_t{0x11});
    std::fill_n(initial.data() + kStride, kStride, uint8_t{0x22});
    std::ofstream out(orig_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out.write(reinterpret_cast<const char*>(initial.data()), static_cast<std::streamsize>(initial.size()));
  }

  shared::FileExpertStore store;
  ASSERT_TRUE(store.Initialize(orig_path.string().c_str(), 0, kNumExperts, kStride, /*cache_slots=*/1));
  EXPECT_TRUE(store.IsBankFileBacked(0));
  EXPECT_TRUE(store.AreAllBanksFileBacked());

  shared::ExpertBlockLease lease{};
  ASSERT_TRUE(store.AcquireReady(0, &lease));
  EXPECT_EQ(lease.data[0], 0x11);
  store.ReleaseLease(0);

  // Rename the active file path and write a replacement file with different contents at orig_path.
  std::error_code ec;
  std::filesystem::rename(orig_path, moved_path, ec);
  ASSERT_FALSE(ec) << ec.message();
  {
    std::vector<uint8_t> replacement(kNumExperts * kStride, uint8_t{0x99});
    std::ofstream out(orig_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out.write(reinterpret_cast<const char*>(replacement.data()), static_cast<std::streamsize>(replacement.size()));
  }

  // Subsequent cache miss on expert 1 must read from the retained handle (0x22), never the replacement (0x99).
  ASSERT_TRUE(store.AcquireReady(1, &lease));
  EXPECT_EQ(lease.data[0], 0x22);
  store.ReleaseLease(1);
  store.Shutdown();

  std::filesystem::remove(orig_path, ec);
  std::filesystem::remove(moved_path, ec);
}

TEST(QMoESharedLayoutTest, PrefetchCoalescesDuplicatesRefreshesGenerationAndBoundsQueue) {
  constexpr size_t kStride = shared::kBankAlignment;
  constexpr int64_t kNumExperts = 6;
  const auto tmp_path = std::filesystem::temp_directory_path() / "ort_qmoe_prefetch_coalesce.bin";
  {
    std::vector<uint8_t> data(static_cast<size_t>(kNumExperts) * kStride);
    for (int64_t e = 0; e < kNumExperts; ++e) {
      std::fill_n(data.data() + static_cast<size_t>(e) * kStride, kStride, static_cast<uint8_t>(e + 10));
    }
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  }

  shared::FileExpertStore store;
  ASSERT_TRUE(store.Initialize(tmp_path.string().c_str(), 0, kNumExperts, kStride, /*cache_slots=*/2));
  const std::array<int32_t, 6> dup_batch{0, 0, 0, 1, 1, 0};
  ASSERT_TRUE(store.Prefetch(dup_batch.data(), dup_batch.size(), 1));
  const std::array<int32_t, 4> gen2_batch{2, 3, 2, 3};
  ASSERT_TRUE(store.Prefetch(gen2_batch.data(), gen2_batch.size(), 2));
  // Refresh expert 0 to generation 3 and enqueue expert 5 at generation 3.
  const std::array<int32_t, 4> gen3_batch{0, 5, 0, 5};
  ASSERT_TRUE(store.Prefetch(gen3_batch.data(), gen3_batch.size(), 3));
  store.WaitPrefetchIdle();

  shared::ExpertBlockLease lease{};
  ASSERT_TRUE(store.AcquireReady(0, &lease));
  EXPECT_EQ(lease.data[0], 10);
  store.ReleaseLease(0);
  ASSERT_TRUE(store.AcquireReady(5, &lease));
  EXPECT_EQ(lease.data[0], 15);
  store.ReleaseLease(5);

  store.Shutdown();
  std::filesystem::remove(tmp_path);
}

TEST(QMoESharedLayoutTest, PrefetchKeepsEarlierRequestsWhenQueueIsFullOfTheSameGeneration) {
  constexpr size_t kStride = shared::kBankAlignment;
  constexpr int64_t kNumExperts = 6;
  const auto tmp_path = std::filesystem::temp_directory_path() / "ort_qmoe_prefetch_same_generation.bin";
  {
    std::vector<uint8_t> data(static_cast<size_t>(kNumExperts) * kStride);
    for (int64_t e = 0; e < kNumExperts; ++e) {
      std::fill_n(data.data() + static_cast<size_t>(e) * kStride, kStride, static_cast<uint8_t>(e + 10));
    }
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  }

  shared::FileExpertStore store;
  constexpr size_t kCacheSlots = 2;
  ASSERT_TRUE(store.Initialize(tmp_path.string().c_str(), 0, kNumExperts, kStride, kCacheSlots));
  // Prefetch holds the store lock for the whole batch, so the worker only sees the final queue. Every request shares
  // generation 1, i.e. none is staler than another: once the queue is full the earlier requests (served first by the
  // worker) must be kept rather than displaced by later ones.
  const std::array<int32_t, 6> batch{0, 1, 2, 3, 4, 5};
  ASSERT_TRUE(store.Prefetch(batch.data(), batch.size(), 1));
  store.WaitPrefetchIdle();

  shared::ExpertBlockLease lease{};
  for (int32_t e = 0; e < kNumExperts; ++e) {
    const bool cached = store.TryAcquireCached(e, &lease);
    EXPECT_EQ(cached, static_cast<size_t>(e) < kCacheSlots) << "expert " << e;
    if (cached) {
      EXPECT_EQ(lease.data[0], static_cast<uint8_t>(e + 10));
      store.ReleaseLease(e);
    }
  }

  store.Shutdown();
  std::filesystem::remove(tmp_path);
}

TEST(QMoESharedLayoutTest, HalfConversionPreservesSignedZero) {
  const std::array<uint16_t, 2> bits{0x0000, 0x8000};
  for (bool scalar : {true, false}) {
    std::array<float, 2> output{};
    shared::HalfBitsToFloatRow(bits.data(), output.data(), bits.size(), scalar);
    EXPECT_EQ(output[0], 0.0f);
    EXPECT_FALSE(std::signbit(output[0]));
    EXPECT_EQ(output[1], 0.0f);
    EXPECT_TRUE(std::signbit(output[1]));
  }
}

void CheckPartialInt4Ranges(bool scalar) {
  constexpr int64_t hidden = 128;
  constexpr int64_t inter = 64;
  constexpr float sentinel = 12345.0f;
  shared::ExpertBlockLayout layout;
  ASSERT_TRUE(shared::MakeExpertBlockLayout(hidden, inter, &layout));
  std::vector<uint8_t> block(layout.stride);
  auto* weights = block.data() + layout.fc2_weights;
  auto* scales = reinterpret_cast<uint16_t*>(block.data() + layout.fc2_scales);
  auto* bias = reinterpret_cast<uint16_t*>(block.data() + layout.fc2_bias);
  for (int64_t i = 0; i < hidden * inter / 2; ++i) weights[i] = static_cast<uint8_t>(i * 67 + i / 13);
  for (int64_t i = 0; i < hidden * inter / 32; ++i) scales[i] = shared::FloatToHalf((i % 7 + 1) / 64.0f);
  for (int64_t i = 0; i < hidden; ++i) bias[i] = shared::FloatToHalf((i % 11 - 5) / 32.0f);

  const std::array<std::pair<int64_t, int64_t>, 8> ranges{
      {{0, 0}, {0, 1}, {1, 4}, {1, 5}, {4, 7}, {124, 127}, {126, 128}, {0, 128}}};
  for (int64_t rows = 1; rows <= shared::kMaxRowsPerCall; ++rows) {
    std::vector<float> input(static_cast<size_t>(rows * inter));
    for (size_t i = 0; i < input.size(); ++i) input[i] = (static_cast<int>(i % 17) - 8) / 64.0f;
    std::vector<float> expected(static_cast<size_t>(rows * hidden));
    shared::Fc2Range(layout, block.data(), input.data(), rows, expected.data(), 0, hidden, true);
    for (const auto& [begin, end] : ranges) {
      SCOPED_TRACE(testing::Message() << "rows=" << rows << " range=[" << begin << "," << end << ")");
      std::vector<float> output(static_cast<size_t>(rows * hidden + 2), sentinel);
      shared::Fc2Range(layout, block.data(), input.data(), rows, output.data() + 1, begin, end, scalar);
      EXPECT_EQ(output.front(), sentinel);
      EXPECT_EQ(output.back(), sentinel);
      for (int64_t row = 0; row < rows; ++row) {
        for (int64_t col = 0; col < hidden; ++col) {
          const size_t index = static_cast<size_t>(row * hidden + col);
          if (col >= begin && col < end) {
            EXPECT_NEAR(output[index + 1], expected[index], 1e-5f);
          } else {
            EXPECT_EQ(output[index + 1], sentinel);
          }
        }
      }
    }
  }
}

TEST(QMoESharedLayoutTest, ScalarPartialInt4RangesPreserveOtherColumns) {
  CheckPartialInt4Ranges(true);
}

TEST(QMoESharedLayoutTest, Avx512PartialInt4RangesPreserveOtherColumns) {
  if (!shared::CpuHasAvx512()) GTEST_SKIP() << "AVX-512 kernels are not available";
  CheckPartialInt4Ranges(false);
}

TEST(QMoESharedLayoutTest, Avx2PartialInt4RangesPreserveOtherColumns) {
  if (!shared::CpuHasAvx2FmaF16c()) GTEST_SKIP() << "AVX2+FMA+F16C kernels are not available";
  CheckPartialInt4Ranges(false);
}

TEST(QMoESharedLayoutTest, Avx2BFloat16AndHalfConversionMatchScalarIncludingNaNs) {
  if (!shared::CpuHasAvx2FmaF16c()) GTEST_SKIP() << "AVX2+FMA+F16C kernels are not available";
  const std::array<uint32_t, 18> patterns{
      0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x3F808000, 0x3F818000,
      0x00000001, 0x00008000, 0x00018000, 0x7F7FFFFF, 0x7F800000, 0xFF800000,
      0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF, 0x7FC00000, 0xFFC00000};
  std::array<float, 36> input{};
  for (size_t i = 0; i < input.size(); ++i) {
    const auto bits = patterns[i % patterns.size()];
    std::memcpy(&input[i], &bits, sizeof(bits));
  }
  std::array<uint16_t, 36> scalar_bf16{}, simd_bf16{}, scalar_fp16{}, simd_fp16{};
  shared::FloatToBFloat16BitsRow(input.data(), scalar_bf16.data(), input.size(), true);
  shared::FloatToBFloat16BitsRow(input.data(), simd_bf16.data(), input.size(), false);
  EXPECT_EQ(simd_bf16, scalar_bf16);
  EXPECT_EQ(simd_bf16[12], 0x7FC1);
  EXPECT_EQ(simd_bf16[13], 0x7FC1);

  std::array<float, 36> roundtrip_scalar_bf16{}, roundtrip_simd_bf16{};
  shared::BFloat16BitsToFloatRow(scalar_bf16.data(), roundtrip_scalar_bf16.data(), scalar_bf16.size(), true);
  shared::BFloat16BitsToFloatRow(simd_bf16.data(), roundtrip_simd_bf16.data(), simd_bf16.size(), false);
  for (size_t i = 0; i < 36; ++i) {
    if (std::isnan(roundtrip_scalar_bf16[i])) {
      EXPECT_TRUE(std::isnan(roundtrip_simd_bf16[i]));
    } else {
      EXPECT_EQ(roundtrip_simd_bf16[i], roundtrip_scalar_bf16[i]);
    }
  }

  shared::FloatToHalfBitsRow(input.data(), scalar_fp16.data(), 10, true);
  shared::FloatToHalfBitsRow(input.data(), simd_fp16.data(), 10, false);
  for (size_t i = 0; i < 10; ++i) {
    EXPECT_EQ(simd_fp16[i], scalar_fp16[i]);
  }
}

TEST(QMoESharedLayoutTest, Avx2Int4Fp16BoundariesMatchScalarForPartialRanges) {
  if (!shared::CpuHasAvx2FmaF16c()) GTEST_SKIP() << "AVX2+FMA+F16C kernels are not available";
  constexpr int64_t hidden = 32;
  constexpr int64_t inter = 96;
  constexpr int64_t n_total = 2 * inter;
  constexpr float sentinel = 12345.0f;
  const std::array<float, 10> projections{
      0.0f, -0.0f, 1.00048828125f, 1.00146484375f, -1.00048828125f,
      -1.00146484375f, 0.0000000298023223876953125f, -0.0000000298023223876953125f,
      5.501953125f, -5.501953125f};
  std::vector<uint8_t> weights(static_cast<size_t>(n_total * hidden / 2), 0x88);
  std::vector<uint16_t> scales(static_cast<size_t>(n_total), shared::FloatToHalf(1.0f));
  std::vector<float> bias(static_cast<size_t>(n_total));
  for (size_t i = 0; i < bias.size(); ++i) bias[i] = projections[i % projections.size()];

  shared::Int4ExpertView view{};
  view.fc1_weights = weights.data();
  view.fc1_scales = scales.data();
  view.fc1_bias = reinterpret_cast<const uint16_t*>(bias.data());
  view.weights_are_prepacked = false;
  // FP32 bias keeps the GEMM scalar, isolating the F16C boundary rounding path.
  view.bias_is_fp32 = true;
  const shared::SwiGluParams swiglu{1.702f, 1.0f, std::numeric_limits<float>::infinity()};
  const std::array<std::pair<int64_t, int64_t>, 8> ranges{
      {{0, 0}, {0, 2}, {0, 6}, {0, 8}, {0, 10}, {2, 136}, {128, n_total}, {0, n_total}}};
  for (int64_t rows : {int64_t{1}, int64_t{shared::kMaxRowsPerCall}}) {
    std::vector<float> input(static_cast<size_t>(rows * hidden), 0.0f);
    for (const auto& [begin, end] : ranges) {
      SCOPED_TRACE(testing::Message() << "rows=" << rows << " range=[" << begin << "," << end << ")");
      std::vector<float> expected(static_cast<size_t>(rows * inter + 2), sentinel);
      std::vector<float> actual(expected.size(), sentinel);
      shared::Int4Fc1SwiGluRange(hidden, inter, view, input.data(), rows, swiglu, expected.data() + 1,
                                 begin, end, /*force_scalar=*/true,
                                 /*bf16_boundaries=*/false, /*fp16_boundaries=*/true);
      shared::Int4Fc1SwiGluRange(hidden, inter, view, input.data(), rows, swiglu, actual.data() + 1,
                                 begin, end, /*force_scalar=*/false,
                                 /*bf16_boundaries=*/false, /*fp16_boundaries=*/true);
      EXPECT_EQ(actual, expected);
      EXPECT_EQ(actual.front(), sentinel);
      EXPECT_EQ(actual.back(), sentinel);
    }
  }
}

TEST(QMoESharedLayoutTest, Avx512BFloat16ConversionMatchesScalarIncludingNaNs) {
  if (!shared::CpuHasAvx512()) GTEST_SKIP() << "AVX-512 kernels are not available";
  const std::array<uint32_t, 18> patterns{
      0x00000000, 0x80000000, 0x3F800000, 0xBF800000, 0x3F808000, 0x3F818000,
      0x00000001, 0x00008000, 0x00018000, 0x7F7FFFFF, 0x7F800000, 0xFF800000,
      0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF, 0x7FC00000, 0xFFC00000};
  std::array<float, 36> input{};
  for (size_t i = 0; i < input.size(); ++i) {
    const auto bits = patterns[i % patterns.size()];
    std::memcpy(&input[i], &bits, sizeof(bits));
  }
  std::array<uint16_t, 36> scalar{}, simd{};
  shared::FloatToBFloat16BitsRow(input.data(), scalar.data(), input.size(), true);
  shared::FloatToBFloat16BitsRow(input.data(), simd.data(), input.size(), false);
  EXPECT_EQ(simd, scalar);
  EXPECT_EQ(simd[12], 0x7FC1);
  EXPECT_EQ(simd[13], 0x7FC1);
}

TEST(QMoESharedLayoutTest, Avx512Fp4ScalesMatchScalarIncludingNonfiniteValues) {
  if (!shared::CpuHasAvx512()) GTEST_SKIP() << "AVX-512 kernels are not available";
  constexpr int64_t k = 32;
  constexpr int64_t columns = 9;  // Exercise vector groups and the scalar tail.
  std::array<uint8_t, columns * k / 2> weights;
  weights.fill(0x22);  // E2M1 1.0 in both nibbles.
  std::array<uint8_t, columns * k / 16> scales{};
  std::array<float, shared::kMaxRowsPerCall * k> input;
  input.fill(1.0f);
  std::array<float, shared::kMaxRowsPerCall * columns> scalar{}, simd{};
  for (bool mxfp4 : {false, true}) {
    for (bool bf16 : {false, true}) {
      for (int code = 0; code < 256; ++code) {
        scales.fill(static_cast<uint8_t>(code));
        for (int64_t rows = 1; rows <= shared::kMaxRowsPerCall; ++rows) {
          SCOPED_TRACE(testing::Message() << "mxfp4=" << mxfp4 << " bf16=" << bf16
                                          << " scale=" << code << " rows=" << rows);
          const auto gemm = mxfp4 ? shared::Mxfp4GemmRange : shared::Nvfp4GemmRange;
          gemm(weights.data(), scales.data(), 1.0f, nullptr, bf16, columns, k, input.data(), rows,
               scalar.data(), columns, 0, columns, true, bf16);
          gemm(weights.data(), scales.data(), 1.0f, nullptr, bf16, columns, k, input.data(), rows,
               simd.data(), columns, 0, columns, false, bf16);
          for (int64_t i = 0; i < rows * columns; ++i) {
            if (std::isnan(scalar[i])) {
              EXPECT_TRUE(std::isnan(simd[i])) << i;
            } else {
              EXPECT_EQ(simd[i], scalar[i]) << i;
            }
          }
        }
      }
    }
  }
}

// The NVFP4/MXFP4 kernels read weights as [N, K/2] bytes in natural K order (the low nibble holds the even k) and
// activations permuted by ReorderNvfp4ActivationRow. The reference below dequantizes every weight in natural order
// without the reorder helpers or the library decode tables, so swapped nibble lanes, wrong block-scale selection or
// a wrong activation permutation change the result. Rounding points mirror the kernels: the scale is rounded after
// the global-scale multiply and each weight after the E2M1 multiply, both to BF16 or FP16.

class Fp4Rng {
 public:
  explicit Fp4Rng(uint64_t seed) : state_(seed) {}
  uint32_t Below(uint32_t bound) {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(state_ >> 33) % bound;
  }

 private:
  uint64_t state_;
};

// Forces the AVX-512 FP4 column tiling (0xF = 8-column tiles for every row tile) for the current scope.
class ScopedFp4Group8Mask {
 public:
  explicit ScopedFp4Group8Mask(int mask) { shared::SetFp4Group8MaskForTesting(mask); }
  ~ScopedFp4Group8Mask() { shared::SetFp4Group8MaskForTesting(-1); }
  ScopedFp4Group8Mask(const ScopedFp4Group8Mask&) = delete;
  ScopedFp4Group8Mask& operator=(const ScopedFp4Group8Mask&) = delete;
};

// Round-to-nearest-even onto a binary format with `mantissa_bits` fraction bits; magnitudes below
// 2^min_exponent land on the subnormal grid. Test inputs never reach the overflow range.
double RoundToFormat(double v, int mantissa_bits, int min_exponent) {
  if (v == 0.0 || !std::isfinite(v)) return v;
  int exponent = 0;
  std::frexp(v, &exponent);  // v = f * 2^exponent with 0.5 <= |f| < 1.
  const double quantum = std::ldexp(1.0, std::max(exponent - 1, min_exponent) - mantissa_bits);
  return std::nearbyint(v / quantum) * quantum;
}

double RoundBf16Reference(double v) { return RoundToFormat(v, 7, -126); }
double RoundFp16Reference(double v) { return RoundToFormat(v, 10, -14); }

constexpr double kFp4E2M1Reference[16] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                                          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0};

// Finite E4M3FN codes only (0x7F and 0xFF are NaN).
double DecodeE4M3FnReference(uint8_t code) {
  const int exponent = (code >> 3) & 0xF;
  const int mantissa = code & 0x7;
  const double magnitude = exponent == 0 ? std::ldexp(static_cast<double>(mantissa), -9)
                                         : std::ldexp(static_cast<double>(8 + mantissa), exponent - 10);
  return (code & 0x80) ? -magnitude : magnitude;
}

struct Fp4Matrix {
  int64_t rows = 0;
  int64_t cols = 0;  // K
  bool mxfp4 = false;
  std::vector<uint8_t> weights;  // [rows, cols / 2], natural K order.
  std::vector<uint8_t> scales;   // [rows, cols / block]: E4M3 per 16 (NVFP4) or E8M0 per 32 (MXFP4).

  int64_t Block() const { return mxfp4 ? 32 : 16; }
};

// Random nibbles and scale codes. `narrow_scales` limits the scale exponents so that, for activations on the k/8
// grid in [-1, 1] and K <= 256, every dot product (and partial sum) is a multiple of 2^-11 below 2^12: float
// accumulation is then exact in any order and an FC1 result equals the double reference bit for bit.
Fp4Matrix MakeFp4Matrix(Fp4Rng& rng, int64_t rows, int64_t cols, bool mxfp4, bool narrow_scales) {
  Fp4Matrix mat;
  mat.rows = rows;
  mat.cols = cols;
  mat.mxfp4 = mxfp4;
  mat.weights.resize(static_cast<size_t>(rows * cols / 2));
  for (auto& byte : mat.weights) byte = static_cast<uint8_t>(rng.Below(256));
  mat.scales.resize(static_cast<size_t>(rows * cols / mat.Block()));
  for (auto& scale : mat.scales) {
    if (mxfp4) {
      scale = static_cast<uint8_t>(narrow_scales ? 125 + rng.Below(3) : 110 + rng.Below(26));
    } else if (narrow_scales) {
      scale = static_cast<uint8_t>(((4 + rng.Below(3)) << 3) | rng.Below(8));
    } else {
      do {
        scale = static_cast<uint8_t>(rng.Below(256));
      } while ((scale & 0x7F) == 0x7F);
    }
  }
  return mat;
}

// Natural-order [rows, cols] dequantized weights, rounded where the kernels round them.
std::vector<double> DequantizeFp4(const Fp4Matrix& mat, float global_scale, bool round_bf16) {
  const auto round_fn = round_bf16 ? RoundBf16Reference : RoundFp16Reference;
  std::vector<double> dequantized(static_cast<size_t>(mat.rows * mat.cols));
  for (int64_t n = 0; n < mat.rows; ++n) {
    for (int64_t k = 0; k < mat.cols; ++k) {
      const uint8_t byte = mat.weights[static_cast<size_t>(n * (mat.cols / 2) + k / 2)];
      const uint8_t nibble = (k & 1) ? static_cast<uint8_t>(byte >> 4) : static_cast<uint8_t>(byte & 0x0F);
      const uint8_t code = mat.scales[static_cast<size_t>(n * (mat.cols / mat.Block()) + k / mat.Block())];
      const double decoded = mat.mxfp4 ? std::ldexp(1.0, code - 127) : DecodeE4M3FnReference(code);
      const double scale = round_fn(static_cast<double>(static_cast<float>(decoded) * global_scale));
      dequantized[static_cast<size_t>(n * mat.cols + k)] = round_fn(kFp4E2M1Reference[nibble] * scale);
    }
  }
  return dequantized;
}

struct Fp4Bias {
  std::vector<double> values;  // Multiples of 1/16 in [-2, 2], exact in BF16 and FP16.
  std::vector<uint16_t> bits;
};

Fp4Bias MakeFp4Bias(Fp4Rng& rng, int64_t count, bool bf16) {
  Fp4Bias bias;
  for (int64_t i = 0; i < count; ++i) {
    const float value = static_cast<float>(static_cast<int>(rng.Below(65)) - 32) / 16.0f;
    bias.values.push_back(static_cast<double>(value));
    bias.bits.push_back(bf16 ? shared::FloatToBFloat16(value) : shared::FloatToHalf(value));
  }
  return bias;
}

struct DotReference {
  double value = 0.0;
  double abs_sum = 0.0;  // Scale for the float accumulation error bound.
};

template <typename T>
DotReference ReferenceDot(const double* weights, const T* x, int64_t k, double bias) {
  DotReference dot;
  dot.value = bias;
  dot.abs_sum = std::fabs(bias);
  for (int64_t i = 0; i < k; ++i) {
    const double term = weights[i] * static_cast<double>(x[i]);
    dot.value += term;
    dot.abs_sum += std::fabs(term);
  }
  return dot;
}

constexpr double kFp4RelativeTolerance = 5e-5;  // Float accumulation of up to 256 terms stays well below this.
constexpr float kFp4Sentinel = 12345.0f;

TEST(QMoESharedLayoutTest, Fp4GemmMatchesNaturalOrderReference) {
  Fp4Rng rng(0x5eed0001);
  constexpr int64_t kPadding = 3;
  int combo = 0;
  for (bool mxfp4 : {false, true}) {
    for (bool round_bf16 : {true, false}) {
      for (int64_t k : {32, 64, 96, 160, 256}) {
        for (int64_t n_total : {1, 5, 8, 12, 17, 33}) {
          for (int64_t m : {1, 2, 3, 4, 7, 8}) {
            // Rotate the global scale and the bias kind (none, BF16, FP16) instead of crossing them.
            const float global_scale = (combo % 2 == 0) ? 0.3f : 1.0f;
            const int bias_kind = combo % 3;
            ++combo;
            const Fp4Matrix mat = MakeFp4Matrix(rng, n_total, k, mxfp4, /*narrow_scales=*/false);
            const std::vector<double> dequantized = DequantizeFp4(mat, global_scale, round_bf16);
            std::vector<float> x(static_cast<size_t>(m * k));
            for (auto& value : x) value = static_cast<float>(static_cast<int>(rng.Below(4001)) - 2000) / 1000.0f;
            std::vector<float> x_reordered(x.size());
            for (int64_t r = 0; r < m; ++r) {
              shared::ReorderNvfp4ActivationRow(x.data() + r * k, x_reordered.data() + r * k, k);
            }
            Fp4Bias bias;
            if (bias_kind != 0) bias = MakeFp4Bias(rng, n_total, bias_kind == 1);

            std::vector<std::pair<int64_t, int64_t>> ranges{{0, n_total}};
            if (n_total >= 3) ranges.emplace_back(1, n_total - 1);
            if (n_total >= 8) ranges.emplace_back(n_total / 2, n_total);
            for (const auto& [begin, end] : ranges) {
              // 0: scalar, 1: AVX-512 with 4-column tiles, 2: AVX-512 with 8-column tiles.
              for (int variant = 0; variant < 3; ++variant) {
                if (variant != 0 && !shared::CpuHasAvx512()) continue;
                SCOPED_TRACE(testing::Message()
                             << "mxfp4=" << mxfp4 << " bf16=" << round_bf16 << " k=" << k << " n_total=" << n_total
                             << " m=" << m << " range=[" << begin << "," << end << ") variant=" << variant
                             << " global_scale=" << global_scale << " bias_kind=" << bias_kind);
                ScopedFp4Group8Mask group8(variant == 2 ? 0xF : 0x0);
                const int64_t width = end - begin;
                const int64_t stride = width + kPadding;
                std::vector<float> out(static_cast<size_t>(m * stride), kFp4Sentinel);
                const auto gemm = mxfp4 ? shared::Mxfp4GemmRange : shared::Nvfp4GemmRange;
                gemm(mat.weights.data(), mat.scales.data(), global_scale,
                     bias.bits.empty() ? nullptr : bias.bits.data(), bias_kind != 2, n_total, k, x_reordered.data(),
                     m, out.data(), stride, begin, end, variant == 0, round_bf16);
                for (int64_t r = 0; r < m; ++r) {
                  for (int64_t c = 0; c < stride; ++c) {
                    const float actual = out[static_cast<size_t>(r * stride + c)];
                    if (c >= width) {
                      ASSERT_EQ(actual, kFp4Sentinel) << "write past the requested range, row=" << r;
                      continue;
                    }
                    const int64_t n = begin + c;
                    const DotReference expected =
                        ReferenceDot(&dequantized[static_cast<size_t>(n * k)], &x[static_cast<size_t>(r * k)], k,
                                     bias.values.empty() ? 0.0 : bias.values[static_cast<size_t>(n)]);
                    ASSERT_NEAR(actual, expected.value, kFp4RelativeTolerance * expected.abs_sum)
                        << "row=" << r << " column=" << n;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

enum class Fp4Boundary {
  kBf16,
  kFp16,
  kNone,
};

TEST(QMoESharedLayoutTest, Fp4ExpertForwardMatchesNaturalOrderReference) {
  Fp4Rng rng(0x5eed0002);
  const std::array<std::pair<int64_t, int64_t>, 5> shapes{{{32, 32}, {64, 32}, {32, 64}, {96, 64}, {128, 96}}};
  const std::array<shared::SwiGluParams, 2> activations{
      shared::SwiGluParams{1.702f, 1.0f, 7.0f},
      shared::SwiGluParams{1.0f, 0.0f, std::numeric_limits<float>::infinity()}};
  const std::array<float, 3> global_scales{0.5f, 1.0f, 2.0f};
  int combo = 0;
  for (bool mxfp4 : {false, true}) {
    for (Fp4Boundary boundary : {Fp4Boundary::kBf16, Fp4Boundary::kFp16, Fp4Boundary::kNone}) {
      for (const auto& [hidden, inter] : shapes) {
        for (int64_t m : {1, 3, 8}) {
          for (bool apply_fc2_bias : {false, true}) {
            for (const auto& swiglu : activations) {
              const bool act_bf16 = boundary == Fp4Boundary::kBf16;
              const bool act_fp16 = boundary == Fp4Boundary::kFp16;
              // Nvfp4ExpertForward rounds scales and weights to BF16 unless the FP16 boundary is selected.
              const bool weights_bf16 = !act_fp16;
              const float fc1_scale = global_scales[static_cast<size_t>(combo % 3)];
              const float fc2_scale = global_scales[static_cast<size_t>((combo / 3) % 3)];
              const bool fc1_has_bias = (combo % 2) == 0;
              ++combo;
              SCOPED_TRACE(testing::Message()
                           << "mxfp4=" << mxfp4 << " boundary=" << static_cast<int>(boundary) << " hidden=" << hidden
                           << " inter=" << inter << " m=" << m << " fc2_bias=" << apply_fc2_bias
                           << " alpha=" << swiglu.alpha << " limit=" << swiglu.limit
                           << " fc1_scale=" << fc1_scale << " fc2_scale=" << fc2_scale << " fc1_bias=" << fc1_has_bias);

              const Fp4Matrix fc1 = MakeFp4Matrix(rng, 2 * inter, hidden, mxfp4, /*narrow_scales=*/true);
              const Fp4Matrix fc2 = MakeFp4Matrix(rng, hidden, inter, mxfp4, /*narrow_scales=*/true);
              const Fp4Bias bias1 = MakeFp4Bias(rng, 2 * inter, weights_bf16);
              const Fp4Bias bias2 = MakeFp4Bias(rng, hidden, weights_bf16);
              std::vector<float> x(static_cast<size_t>(m * hidden));
              for (auto& value : x) value = static_cast<float>(static_cast<int>(rng.Below(17)) - 8) / 8.0f;

              shared::Nvfp4ExpertView view;
              view.fc1_weights = fc1.weights.data();
              view.fc1_scales = fc1.scales.data();
              view.fc1_global_scale = fc1_scale;
              view.fc1_bias = fc1_has_bias ? bias1.bits.data() : nullptr;
              view.fc2_weights = fc2.weights.data();
              view.fc2_scales = fc2.scales.data();
              view.fc2_global_scale = fc2_scale;
              view.fc2_bias = bias2.bits.data();
              view.bias_is_bf16 = weights_bf16;
              view.is_mxfp4 = mxfp4;

              // FC1 interleaves gate (even) and up (odd) columns; the SwiGLU result is the natural-order FC2 input.
              const std::vector<double> w1 = DequantizeFp4(fc1, fc1_scale, weights_bf16);
              const std::vector<double> w2 = DequantizeFp4(fc2, fc2_scale, weights_bf16);
              std::vector<double> act(static_cast<size_t>(m * inter));
              for (int64_t r = 0; r < m; ++r) {
                for (int64_t j = 0; j < inter; ++j) {
                  const size_t gate_row = static_cast<size_t>(2 * j * hidden);
                  const size_t up_row = static_cast<size_t>((2 * j + 1) * hidden);
                  const DotReference gate_dot = ReferenceDot(&w1[gate_row], &x[static_cast<size_t>(r * hidden)], hidden,
                                                             fc1_has_bias ? bias1.values[static_cast<size_t>(2 * j)] : 0.0);
                  const DotReference up_dot = ReferenceDot(&w1[up_row], &x[static_cast<size_t>(r * hidden)], hidden,
                                                           fc1_has_bias ? bias1.values[static_cast<size_t>(2 * j + 1)] : 0.0);
                  float gate = static_cast<float>(gate_dot.value);
                  float value = static_cast<float>(up_dot.value);
                  ASSERT_EQ(static_cast<double>(gate), gate_dot.value) << "FC1 reference must be exact in float";
                  ASSERT_EQ(static_cast<double>(value), up_dot.value) << "FC1 reference must be exact in float";
                  if (std::isfinite(swiglu.limit)) {
                    gate = std::min(gate, swiglu.limit);
                    value = std::clamp(value, -swiglu.limit, swiglu.limit);
                  }
                  const float activated = gate / (1.0f + std::exp(-swiglu.alpha * gate)) * (value + swiglu.beta);
                  act[static_cast<size_t>(r * inter + j)] = act_bf16   ? RoundBf16Reference(activated)
                                                            : act_fp16 ? RoundFp16Reference(activated)
                                                                       : static_cast<double>(activated);
                }
              }
              std::vector<DotReference> expected(static_cast<size_t>(m * hidden));
              for (int64_t r = 0; r < m; ++r) {
                for (int64_t c = 0; c < hidden; ++c) {
                  expected[static_cast<size_t>(r * hidden + c)] =
                      ReferenceDot(&w2[static_cast<size_t>(c * inter)], &act[static_cast<size_t>(r * inter)], inter,
                                   apply_fc2_bias ? bias2.values[static_cast<size_t>(c)] : 0.0);
                }
              }

              // 0: scalar, 1: AVX-512 with 4-column tiles, 2: AVX-512 with 8-column tiles.
              for (int variant = 0; variant < 3; ++variant) {
                if (variant != 0 && !shared::CpuHasAvx512()) continue;
                ScopedFp4Group8Mask group8(variant == 2 ? 0xF : 0x0);
                std::vector<float> out(static_cast<size_t>(m * hidden), kFp4Sentinel);
                ASSERT_TRUE(shared::Nvfp4ExpertForward(hidden, inter, view, x.data(), m, swiglu, out.data(),
                                                       variant == 0, apply_fc2_bias, act_bf16, act_fp16))
                    << "variant=" << variant;
                for (size_t i = 0; i < out.size(); ++i) {
                  ASSERT_NEAR(out[i], expected[i].value, kFp4RelativeTolerance * expected[i].abs_sum)
                      << "variant=" << variant << " row=" << i / static_cast<size_t>(hidden)
                      << " column=" << i % static_cast<size_t>(hidden);
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(QMoESharedLayoutTest, Fp4ExpertRangeSplitMatchesWholeForward) {
  // Worker-style splitting: FC1 over even column ranges and FC2 over arbitrary column ranges must agree with the
  // whole-expert call (within accumulation-order noise, since tile boundaries differ between the two).
  Fp4Rng rng(0x5eed0003);
  constexpr int64_t hidden = 96;
  constexpr int64_t inter = 96;
  constexpr int64_t m = 5;
  const shared::SwiGluParams swiglu{1.702f, 1.0f, 7.0f};
  for (bool mxfp4 : {false, true}) {
    SCOPED_TRACE(testing::Message() << "mxfp4=" << mxfp4);
    const Fp4Matrix fc1 = MakeFp4Matrix(rng, 2 * inter, hidden, mxfp4, /*narrow_scales=*/true);
    const Fp4Matrix fc2 = MakeFp4Matrix(rng, hidden, inter, mxfp4, /*narrow_scales=*/true);
    const Fp4Bias bias1 = MakeFp4Bias(rng, 2 * inter, true);
    const Fp4Bias bias2 = MakeFp4Bias(rng, hidden, true);
    shared::Nvfp4ExpertView view;
    view.fc1_weights = fc1.weights.data();
    view.fc1_scales = fc1.scales.data();
    view.fc1_global_scale = 1.0f;
    view.fc1_bias = bias1.bits.data();
    view.fc2_weights = fc2.weights.data();
    view.fc2_scales = fc2.scales.data();
    view.fc2_global_scale = 0.5f;
    view.fc2_bias = bias2.bits.data();
    view.is_mxfp4 = mxfp4;
    std::vector<float> x(static_cast<size_t>(m * hidden));
    for (auto& value : x) value = static_cast<float>(static_cast<int>(rng.Below(17)) - 8) / 8.0f;

    std::vector<float> whole(static_cast<size_t>(m * hidden), kFp4Sentinel);
    ASSERT_TRUE(shared::Nvfp4ExpertForward(hidden, inter, view, x.data(), m, swiglu, whole.data(),
                                           /*force_scalar=*/false, /*apply_fc2_bias=*/true));

    std::vector<float> x_reordered(x.size());
    for (int64_t r = 0; r < m; ++r) {
      shared::ReorderNvfp4ActivationRow(x.data() + r * hidden, x_reordered.data() + r * hidden, hidden);
    }
    std::vector<float> inter_reordered(static_cast<size_t>(m * inter), kFp4Sentinel);
    const std::array<int64_t, 5> fc1_splits{0, 50, 66, 128, 2 * inter};
    for (size_t i = 0; i + 1 < fc1_splits.size(); ++i) {
      shared::Nvfp4Fc1SwiGluRange(hidden, inter, view, x_reordered.data(), m, swiglu, inter_reordered.data(),
                                  fc1_splits[i], fc1_splits[i + 1], false, true, false);
    }
    std::vector<float> split(static_cast<size_t>(m * hidden), kFp4Sentinel);
    const std::array<int64_t, 5> fc2_splits{0, 7, 40, 41, hidden};
    for (size_t i = 0; i + 1 < fc2_splits.size(); ++i) {
      shared::Nvfp4Fc2Range(hidden, inter, view, inter_reordered.data(), m, split.data(), fc2_splits[i],
                            fc2_splits[i + 1], false, true, true);
    }
    for (size_t i = 0; i < split.size(); ++i) {
      ASSERT_NEAR(split[i], whole[i], 1e-5 * (std::fabs(whole[i]) + 1.0)) << "element " << i;
    }
  }
}

TEST(QMoESharedLayoutTest, Fp4ExpertForwardRejectsUnsupportedShapes) {
  Fp4Rng rng(0x5eed0004);
  const Fp4Matrix fc1 = MakeFp4Matrix(rng, 64, 32, false, true);
  const Fp4Matrix fc2 = MakeFp4Matrix(rng, 32, 32, false, true);
  shared::Nvfp4ExpertView view;
  view.fc1_weights = fc1.weights.data();
  view.fc1_scales = fc1.scales.data();
  view.fc2_weights = fc2.weights.data();
  view.fc2_scales = fc2.scales.data();
  const shared::SwiGluParams swiglu{1.0f, 0.0f, std::numeric_limits<float>::infinity()};
  std::vector<float> x(static_cast<size_t>(shared::kMaxRowsPerCall + 1) * 48, 0.5f);
  std::vector<float> out(static_cast<size_t>(shared::kMaxRowsPerCall + 1) * 48, 0.0f);
  EXPECT_TRUE(shared::Nvfp4ExpertForward(32, 32, view, x.data(), 1, swiglu, out.data()));
  EXPECT_TRUE(shared::Nvfp4ExpertForward(32, 32, view, x.data(), shared::kMaxRowsPerCall, swiglu, out.data()));
  EXPECT_FALSE(shared::Nvfp4ExpertForward(32, 32, view, x.data(), 0, swiglu, out.data()));
  EXPECT_FALSE(shared::Nvfp4ExpertForward(32, 32, view, x.data(), shared::kMaxRowsPerCall + 1, swiglu, out.data()));
  EXPECT_FALSE(shared::Nvfp4ExpertForward(48, 32, view, x.data(), 1, swiglu, out.data()));
  EXPECT_FALSE(shared::Nvfp4ExpertForward(32, 48, view, x.data(), 1, swiglu, out.data()));
  EXPECT_FALSE(shared::Nvfp4ExpertForward(0, 32, view, x.data(), 1, swiglu, out.data()));
  shared::Nvfp4ExpertView missing_fc2 = view;
  missing_fc2.fc2_weights = nullptr;
  EXPECT_FALSE(shared::Nvfp4ExpertForward(32, 32, missing_fc2, x.data(), 1, swiglu, out.data()));
}

#if defined(__linux__)
TEST(QMoESharedLayoutTest, MappedStoreHonorsPinsAndRollsBackFailedGroups) {
  constexpr size_t stride = shared::kBankAlignment;
  alignas(shared::kBankAlignment) std::array<uint8_t, 3 * stride> backing{};
  for (size_t i = 0; i < backing.size(); ++i) backing[i] = static_cast<uint8_t>(i / stride + 1);
  shared::FileExpertStore::MappedBank bank{backing.data(), stride, 0};
  shared::FileExpertStore store;
  ASSERT_TRUE(store.InitializeMapped(&bank, 1, 3, stride, 2));
  shared::ExpertBlockLease lease{};
  ASSERT_TRUE(store.AcquireReady(0, &lease));
  EXPECT_EQ(lease.data[0], 1);
  EXPECT_FALSE(store.Evict(0));
  const std::array<int32_t, 2> group{1, 3};
  std::array<shared::ExpertBlockLease, 2> group_leases{};
  EXPECT_FALSE(store.AcquireGroupReady(group.data(), group.size(), group_leases.data()));
  EXPECT_TRUE(store.Evict(1));
  store.ReleaseLease(0);
  const int32_t prefetch = 2;
  ASSERT_TRUE(store.Prefetch(&prefetch, 1, 1));
  store.WaitPrefetchIdle();
  ASSERT_TRUE(store.TryAcquireCached(2, &lease));
  EXPECT_EQ(lease.data[0], 3);
  store.ReleaseLease(2);
  auto* replacement = store.BeginReplace(2, 1);
  ASSERT_NE(replacement, nullptr);
  EXPECT_FALSE(store.TryAcquireCached(1, &lease));
  std::fill_n(replacement, stride, uint8_t{42});
  store.FinishReplace(1, true);
  ASSERT_TRUE(store.TryAcquireCached(1, &lease));
  EXPECT_EQ(lease.data[0], 42);
  store.ReleaseLease(1);
  ASSERT_NE(store.BeginReplace(1, 2), nullptr);
  store.FinishReplace(2, false);
  EXPECT_FALSE(store.TryAcquireCached(2, &lease));
  ASSERT_TRUE(store.AcquireReady(2, &lease));
  EXPECT_EQ(lease.data[0], 3);
  store.ReleaseLease(2);
}

TEST(QMoESharedLayoutTest, ModifiedPrivateReadOnlyMappingExcludedFromReloadAndEviction) {
  const long page_raw = sysconf(_SC_PAGESIZE);
  const size_t page_size = (page_raw > 0) ? static_cast<size_t>(page_raw) : 4096u;
  const size_t expert_bytes = page_size * 2;
  constexpr int64_t num_experts = 2;
  const size_t total_bytes = static_cast<size_t>(num_experts) * expert_bytes;

  const auto tmp_path = std::filesystem::temp_directory_path() / "ort_qmoe_private_map_test.bin";
  {
    std::vector<uint8_t> file_bytes(total_bytes, uint8_t{0x11});
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out.write(reinterpret_cast<const char*>(file_bytes.data()), static_cast<std::streamsize>(file_bytes.size()));
  }

  const int fd = open(tmp_path.c_str(), O_RDONLY | O_CLOEXEC);
  ASSERT_GE(fd, 0);

  void* priv_map = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  ASSERT_NE(priv_map, MAP_FAILED);
  auto* priv_bytes = static_cast<uint8_t*>(priv_map);
  std::fill_n(priv_bytes, expert_bytes, uint8_t{0x22});
  std::fill_n(priv_bytes + expert_bytes, expert_bytes, uint8_t{0x33});
  ASSERT_EQ(mprotect(priv_map, total_bytes, PROT_READ), 0);

  // A modified MAP_PRIVATE mapping frozen to PROT_READ (r--p) must not be treated as reloadable from file.
  EXPECT_FALSE(shared::IsFileBackedReadOnlyMapping(priv_map, total_bytes));
  EXPECT_EQ(shared::EvictFileBackedHostPages(priv_map, total_bytes), 0U);
  EXPECT_EQ(priv_bytes[0], 0x22);
  EXPECT_EQ(priv_bytes[expert_bytes], 0x33);

  shared::FileExpertStore store;
  shared::FileExpertStore::MappedBank bank{priv_map, expert_bytes, 0, nullptr, 0};
  ASSERT_TRUE(store.InitializeMapped(&bank, 1, num_experts, expert_bytes, /*cache_slots=*/1));
  EXPECT_FALSE(store.IsBankFileBacked(0));
  EXPECT_EQ(store.ResidentBackingBytes(), total_bytes);

  shared::ExpertBlockLease lease{};
  ASSERT_TRUE(store.AcquireReady(0, &lease));
  EXPECT_EQ(lease.data[0], 0x22);
  store.ReleaseLease(0);
  ASSERT_TRUE(store.AcquireReady(1, &lease));
  EXPECT_EQ(lease.data[0], 0x33);
  store.ReleaseLease(1);
  store.Shutdown();

  // Verify that a genuine shared read-only mapping (r--s) is accepted, including via /proc/self/maps auto-discovery.
  void* shared_map = mmap(nullptr, total_bytes, PROT_READ, MAP_SHARED, fd, 0);
  ASSERT_NE(shared_map, MAP_FAILED);
  EXPECT_TRUE(shared::IsFileBackedReadOnlyMapping(shared_map, total_bytes));
  EXPECT_EQ(shared::EvictFileBackedHostPages(shared_map, total_bytes), total_bytes);

  shared::FileExpertStore shared_store;
  shared::FileExpertStore::MappedBank shared_bank{shared_map, expert_bytes, 0, nullptr, 0};
  ASSERT_TRUE(shared_store.InitializeMapped(&shared_bank, 1, num_experts, expert_bytes, /*cache_slots=*/1));
  EXPECT_TRUE(shared_store.IsBankFileBacked(0));
  ASSERT_TRUE(shared_store.AcquireReady(0, &lease));
  EXPECT_EQ(lease.data[0], 0x11);
  shared_store.ReleaseLease(0);
  shared_store.Shutdown();

  munmap(shared_map, total_bytes);
  munmap(priv_map, total_bytes);
  close(fd);
  std::filesystem::remove(tmp_path);
}

TEST(QMoESharedLayoutTest, PrivateReadOnlyMappingIsReloadableOnlyWhileEveryPageMatchesFile) {
  const long page_raw = sysconf(_SC_PAGESIZE);
  const size_t page_size = (page_raw > 0) ? static_cast<size_t>(page_raw) : 4096u;
  const size_t total_bytes = page_size * 3;
  constexpr uint8_t kFileByte = 0x11;
  constexpr uint8_t kPrivateByte = 0x22;
  // Without pagemap access, private mappings cannot be verified and are conservatively rejected.
  const bool pagemap_readable = access("/proc/self/pagemap", R_OK) == 0;

  const auto tmp_path = std::filesystem::temp_directory_path() /
                        ("ort_qmoe_reloadable_mapping_test_" + std::to_string(getpid()) + ".bin");
  {
    std::vector<uint8_t> file_bytes(total_bytes, kFileByte);
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out.write(reinterpret_cast<const char*>(file_bytes.data()), static_cast<std::streamsize>(file_bytes.size()));
  }
  const int fd = open(tmp_path.c_str(), O_RDONLY | O_CLOEXEC);
  ASSERT_GE(fd, 0);

  // Same discard the controller applies to uploaded experts, but only for mappings proven reloadable.
  const auto discard_if_reloadable = [](void* data, size_t bytes) {
    if (moe_offload::IsReloadableFileMapping(data, bytes)) {
      EXPECT_EQ(madvise(data, bytes, MADV_DONTNEED), 0);
    }
  };

  // Privately modify one byte of the middle page, then freeze the mapping to PROT_READ (r--p).
  void* modified_map = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  ASSERT_NE(modified_map, MAP_FAILED);
  auto* modified = static_cast<uint8_t*>(modified_map);
  const size_t modified_offset = page_size + 7;
  modified[modified_offset] = kPrivateByte;
  ASSERT_EQ(mprotect(modified_map, total_bytes, PROT_READ), 0);

  EXPECT_FALSE(moe_offload::IsReloadableFileMapping(modified_map, total_bytes));
  EXPECT_FALSE(moe_offload::IsReloadableFileMapping(modified + page_size, page_size));
  discard_if_reloadable(modified_map, total_bytes);
  EXPECT_EQ(modified[modified_offset], kPrivateByte);
  EXPECT_EQ(modified[0], kFileByte);

  // Only the pages that were never modified are still reloadable from the file.
  if (pagemap_readable) {
    EXPECT_TRUE(moe_offload::IsReloadableFileMapping(modified_map, page_size));
    EXPECT_TRUE(moe_offload::IsReloadableFileMapping(modified + 2 * page_size, page_size));
  }

  // A private mapping that was only read stays identical to the file, so it is reloadable and survives a discard.
  void* clean_map = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  ASSERT_NE(clean_map, MAP_FAILED);
  auto* clean = static_cast<const uint8_t*>(clean_map);
  uint8_t checksum = 0;
  for (size_t offset = 0; offset < total_bytes; offset += page_size) {
    checksum ^= clean[offset];
  }
  EXPECT_EQ(checksum, kFileByte);
  ASSERT_EQ(mprotect(clean_map, total_bytes, PROT_READ), 0);
  if (pagemap_readable) {
    EXPECT_TRUE(moe_offload::IsReloadableFileMapping(clean_map, total_bytes));
  }
  discard_if_reloadable(clean_map, total_bytes);
  EXPECT_EQ(clean[modified_offset], kFileByte);

  // A mapping that was never writable cannot hold private modifications.
  void* read_only_map = mmap(nullptr, total_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
  ASSERT_NE(read_only_map, MAP_FAILED);
  if (pagemap_readable) {
    EXPECT_TRUE(moe_offload::IsReloadableFileMapping(read_only_map, total_bytes));
  }

  // A writable mapping is never reloadable, whatever its contents.
  void* writable_map = mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  ASSERT_NE(writable_map, MAP_FAILED);
  EXPECT_FALSE(moe_offload::IsReloadableFileMapping(writable_map, total_bytes));

  // A shared read-only mapping always reflects the file.
  void* shared_map = mmap(nullptr, total_bytes, PROT_READ, MAP_SHARED, fd, 0);
  ASSERT_NE(shared_map, MAP_FAILED);
  EXPECT_TRUE(moe_offload::IsReloadableFileMapping(shared_map, total_bytes));

  munmap(shared_map, total_bytes);
  munmap(writable_map, total_bytes);
  munmap(read_only_map, total_bytes);
  munmap(clean_map, total_bytes);
  munmap(modified_map, total_bytes);
  close(fd);
  std::filesystem::remove(tmp_path);
}

TEST(QMoESharedLayoutTest, PromoteCleanPrivateMappingAndParallelChunkedLoadPreservesBankBytes) {
  const long page_raw = sysconf(_SC_PAGESIZE);
  const size_t page_size = (page_raw > 0) ? static_cast<size_t>(page_raw) : 4096u;
  constexpr int64_t num_experts = 3;
  constexpr int32_t s_rows = 20;
  constexpr int32_t s_cols = 18;
  const size_t w0_bytes = page_size * 2;
  const size_t w1_bytes = page_size;
  const size_t s_bytes = static_cast<size_t>(s_rows) * static_cast<size_t>(s_cols) * sizeof(uint16_t);
  const size_t stride = ((w0_bytes + w1_bytes + s_bytes + shared::kBankAlignment - 1) /
                         shared::kBankAlignment) *
                        shared::kBankAlignment;

  const auto tmp_path = std::filesystem::temp_directory_path() / "ort_qmoe_promote_chunk_load_test.bin";
  const size_t file_bytes = static_cast<size_t>(num_experts) * w0_bytes;
  {
    std::vector<uint8_t> raw(file_bytes);
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = static_cast<uint8_t>((i * 37 + 11) & 0xFF);
    std::ofstream out(tmp_path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out.write(reinterpret_cast<const char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
  }

  const int fd = open(tmp_path.c_str(), O_RDONLY | O_CLOEXEC);
  ASSERT_GE(fd, 0);

  // Dirty rw-p mapping must not be promoted to shared read-only (both first-page and non-first-page COW).
  void* dirty_map = mmap(nullptr, file_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  ASSERT_NE(dirty_map, MAP_FAILED);
  static_cast<uint8_t*>(dirty_map)[0] ^= 0xFF;
  EXPECT_FALSE(moe_offload::PromoteFileMappingToSharedReadOnly(dirty_map, file_bytes));
  EXPECT_FALSE(shared::IsFileBackedReadOnlyMapping(dirty_map, file_bytes));

  void* dirty_nonfirst_map = mmap(nullptr, file_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  ASSERT_NE(dirty_nonfirst_map, MAP_FAILED);
  static_cast<uint8_t*>(dirty_nonfirst_map)[page_size * 2] ^= 0xFF;
  EXPECT_FALSE(moe_offload::PromoteFileMappingToSharedReadOnly(dirty_nonfirst_map, file_bytes));
  EXPECT_FALSE(shared::IsFileBackedReadOnlyMapping(dirty_nonfirst_map, file_bytes));
  EXPECT_EQ(munmap(dirty_nonfirst_map, file_bytes), 0);

  // Clean rw-p mapping (matching PosixRandomAccessFile::Map) promotes to r--s in-place.
  void* clean_map = mmap(nullptr, file_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
  ASSERT_NE(clean_map, file_bytes ? MAP_FAILED : nullptr);
  EXPECT_FALSE(shared::IsFileBackedReadOnlyMapping(clean_map, file_bytes));
  EXPECT_TRUE(moe_offload::PromoteFileMappingToSharedReadOnly(clean_map, file_bytes));
  EXPECT_TRUE(shared::IsFileBackedReadOnlyMapping(clean_map, file_bytes));
  EXPECT_TRUE(moe_offload::IsReloadableFileMapping(clean_map, file_bytes));

  std::vector<uint8_t> w1_backing(static_cast<size_t>(num_experts) * w1_bytes);
  for (size_t i = 0; i < w1_backing.size(); ++i) w1_backing[i] = static_cast<uint8_t>((i * 53 + 3) & 0xFF);

  std::vector<uint16_t> s_backing(static_cast<size_t>(num_experts) * s_rows * s_cols);
  for (size_t i = 0; i < s_backing.size(); ++i) s_backing[i] = static_cast<uint16_t>(i + 1);

  const shared::FileExpertStore::MappedBank banks[]{
      {clean_map, w0_bytes, 0, nullptr, 0},
      {w1_backing.data(), w1_bytes, w0_bytes, nullptr, 0},
      {s_backing.data(), s_bytes, w0_bytes + w1_bytes, nullptr, 0}};

  shared::FileExpertStore store;
  ASSERT_TRUE(store.InitializeMapped(banks, std::size(banks), num_experts, stride, /*cache_slots=*/2));
  EXPECT_TRUE(store.IsBankFileBacked(0));

  uint8_t* slot_dst = nullptr;
  shared::ExpertBlockLease lease{};
  ASSERT_EQ(store.BeginLoadExpert(1, &slot_dst, &lease), 1);
  ASSERT_NE(slot_dst, nullptr);
  for (int chunk = 0; chunk < shared::FileExpertStore::kLoadChunksPerExpert; ++chunk) {
    EXPECT_TRUE(store.LoadExpertChunk(1, slot_dst, chunk));
  }
  ASSERT_TRUE(store.FinishLoadExpert(1, true, &lease));
  ASSERT_NE(lease.data, nullptr);

  // Verify chunked w0 + w1 + scales preserve exact source bytes without transposition.
  EXPECT_EQ(std::memcmp(lease.data + w0_bytes, w1_backing.data() + w1_bytes, w1_bytes), 0);
  EXPECT_EQ(std::memcmp(lease.data + w0_bytes + w1_bytes,
                        s_backing.data() + static_cast<size_t>(1) * s_rows * s_cols, s_bytes),
            0);

  // Calling BeginLoadExpert on an already-ready expert returns 0 and pins a second lease.
  uint8_t* slot_dst2 = nullptr;
  shared::ExpertBlockLease lease2{};
  EXPECT_EQ(store.BeginLoadExpert(1, &slot_dst2, &lease2), 0);
  EXPECT_EQ(lease2.data, lease.data);
  store.ReleaseLease(1);
  store.ReleaseLease(1);

  // Aborting a chunk load with ok=false rolls back the slot cleanly.
  ASSERT_EQ(store.BeginLoadExpert(2, &slot_dst, &lease), 1);
  EXPECT_FALSE(store.FinishLoadExpert(2, false, &lease));
  EXPECT_FALSE(store.TryAcquireCached(2, &lease));

  store.Shutdown();
  munmap(clean_map, file_bytes);
  munmap(dirty_map, file_bytes);
  close(fd);
  std::filesystem::remove(tmp_path);
}
#endif

TEST(QMoESharedLayoutTest, CanonicalInt4ExpertViewMatchesReferenceAcrossPrepackedRawFp16AndBf16) {
  constexpr int64_t hidden = 128;
  constexpr int64_t inter = 64;
  constexpr int64_t n1 = 2 * inter;
  std::vector<uint8_t> fc1_raw(static_cast<size_t>(n1 * (hidden / 2)));
  std::vector<uint8_t> fc2_raw(static_cast<size_t>(hidden * (inter / 2)));
  for (size_t i = 0; i < fc1_raw.size(); ++i) fc1_raw[i] = static_cast<uint8_t>((i * 73 + 19) & 0xFF);
  for (size_t i = 0; i < fc2_raw.size(); ++i) fc2_raw[i] = static_cast<uint8_t>((i * 41 + 7) & 0xFF);

  std::vector<uint8_t> fc1_sm80(fc1_raw.size());
  std::vector<uint8_t> fc2_sm80(fc2_raw.size());
  shared::PackSm80Int4Weights(fc1_raw.data(), fc1_sm80.data(), n1, hidden);
  shared::PackSm80Int4Weights(fc2_raw.data(), fc2_sm80.data(), hidden, inter);

  std::vector<uint8_t> fc1_unpacked(fc1_raw.size());
  shared::UnpackSm80Int4Weights(fc1_sm80.data(), fc1_unpacked.data(), n1, hidden);
  EXPECT_EQ(fc1_unpacked, fc1_raw);
  for (int64_t n = 0; n < n1; ++n) {
    for (int64_t k = 0; k < hidden; ++k) {
      const uint8_t raw_byte = fc1_raw[static_cast<size_t>(n * (hidden / 2) + (k / 2))];
      const uint8_t expected_code = (k & 1) ? (raw_byte >> 4) : (raw_byte & 0x0F);
      EXPECT_EQ(shared::WeightCodeAt(fc1_sm80.data(), n1, hidden, n, k), expected_code);
    }
  }

  std::vector<uint16_t> fc1_s_fp16(static_cast<size_t>(n1 * (hidden / 32)));
  std::vector<uint16_t> fc2_s_fp16(static_cast<size_t>(hidden * (inter / 32)));
  std::vector<uint16_t> fc1_s_bf16(fc1_s_fp16.size());
  std::vector<uint16_t> fc2_s_bf16(fc2_s_fp16.size());
  for (size_t i = 0; i < fc1_s_fp16.size(); ++i) {
    const float v = static_cast<float>((i % 7) + 1) / 64.0f;
    fc1_s_fp16[i] = shared::FloatToHalf(v);
    fc1_s_bf16[i] = shared::FloatToBFloat16(v);
  }
  for (size_t i = 0; i < fc2_s_fp16.size(); ++i) {
    const float v = static_cast<float>((i % 5) + 1) / 64.0f;
    fc2_s_fp16[i] = shared::FloatToHalf(v);
    fc2_s_bf16[i] = shared::FloatToBFloat16(v);
  }
  std::vector<uint16_t> fc1_b_fp16(static_cast<size_t>(n1));
  std::vector<uint16_t> fc2_b_fp16(static_cast<size_t>(hidden));
  std::vector<uint16_t> fc1_b_bf16(static_cast<size_t>(n1));
  std::vector<uint16_t> fc2_b_bf16(static_cast<size_t>(hidden));
  for (int64_t i = 0; i < n1; ++i) {
    const float v = static_cast<float>((i % 9) - 4) / 32.0f;
    fc1_b_fp16[static_cast<size_t>(i)] = shared::FloatToHalf(v);
    fc1_b_bf16[static_cast<size_t>(i)] = shared::FloatToBFloat16(v);
  }
  for (int64_t i = 0; i < hidden; ++i) {
    const float v = static_cast<float>((i % 11) - 5) / 32.0f;
    fc2_b_fp16[static_cast<size_t>(i)] = shared::FloatToHalf(v);
    fc2_b_bf16[static_cast<size_t>(i)] = shared::FloatToBFloat16(v);
  }

  // Build legacy SharedExpertBlockV1 block to cross-check FP16 prepacked parity.
  shared::ExpertBlockLayout blk_layout;
  ASSERT_TRUE(shared::MakeExpertBlockLayout(hidden, inter, &blk_layout));
  std::vector<uint8_t> blk(blk_layout.stride, 0);
  std::memcpy(blk.data() + blk_layout.fc1_weights, fc1_sm80.data(), fc1_sm80.size());
  shared::TransposeInt4Scales(fc1_s_fp16.data(), reinterpret_cast<uint16_t*>(blk.data() + blk_layout.fc1_scales),
                              n1, hidden / 32);
  std::memcpy(blk.data() + blk_layout.fc1_bias, fc1_b_fp16.data(), fc1_b_fp16.size() * sizeof(uint16_t));
  std::memcpy(blk.data() + blk_layout.fc2_weights, fc2_sm80.data(), fc2_sm80.size());
  shared::TransposeInt4Scales(fc2_s_fp16.data(), reinterpret_cast<uint16_t*>(blk.data() + blk_layout.fc2_scales),
                              hidden, inter / 32);
  std::memcpy(blk.data() + blk_layout.fc2_bias, fc2_b_fp16.data(), fc2_b_fp16.size() * sizeof(uint16_t));

  constexpr int64_t rows = 5;
  std::vector<float> x(static_cast<size_t>(rows * hidden));
  for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(static_cast<int>(i % 19) - 9) / 32.0f;
  const shared::SwiGluParams swiglu{1.702f, 1.0f, 7.0f};

  std::vector<float> ref_blk_out(static_cast<size_t>(rows * hidden));
  ASSERT_TRUE(shared::ExpertForward(blk_layout, blk.data(), x.data(), rows, swiglu, ref_blk_out.data(),
                                    /*force_scalar=*/true, /*apply_fc2_bias=*/true, /*fp16_boundaries=*/true));

  for (bool prepacked : {true, false}) {
    for (bool is_bf16 : {false, true}) {
      shared::Int4ExpertView view{};
      view.fc1_weights = prepacked ? fc1_sm80.data() : fc1_raw.data();
      view.fc1_scales = is_bf16 ? fc1_s_bf16.data() : fc1_s_fp16.data();
      view.fc1_bias = is_bf16 ? fc1_b_bf16.data() : fc1_b_fp16.data();
      view.fc2_weights = prepacked ? fc2_sm80.data() : fc2_raw.data();
      view.fc2_scales = is_bf16 ? fc2_s_bf16.data() : fc2_s_fp16.data();
      view.fc2_bias = is_bf16 ? fc2_b_bf16.data() : fc2_b_fp16.data();
      view.scales_are_block_major = false;
      view.weights_are_prepacked = prepacked;
      view.scale_is_bf16 = is_bf16;
      view.bias_is_bf16 = is_bf16;

      std::vector<float> out_scalar(static_cast<size_t>(rows * hidden));
      std::vector<float> out_simd(static_cast<size_t>(rows * hidden));
      ASSERT_TRUE(shared::Int4ExpertForward(hidden, inter, view, x.data(), rows, swiglu, out_scalar.data(),
                                            /*force_scalar=*/true, /*apply_fc2_bias=*/true,
                                            /*bf16_boundaries=*/is_bf16, /*fp16_boundaries=*/!is_bf16));
      ASSERT_TRUE(shared::Int4ExpertForward(hidden, inter, view, x.data(), rows, swiglu, out_simd.data(),
                                            /*force_scalar=*/false, /*apply_fc2_bias=*/true,
                                            /*bf16_boundaries=*/is_bf16, /*fp16_boundaries=*/!is_bf16));
      for (size_t i = 0; i < out_scalar.size(); ++i) {
        EXPECT_NEAR(out_simd[i], out_scalar[i], 1e-5f) << "prepacked=" << prepacked << " bf16=" << is_bf16 << " i=" << i;
        if (!is_bf16) {
          EXPECT_NEAR(out_scalar[i], ref_blk_out[i], 1e-5f) << "prepacked=" << prepacked << " i=" << i;
        }
      }
    }
  }
}

TEST(QMoESharedLayoutTest, MakeInt3ExpertBlockLayoutValidatesDimensionsAndAlignment) {
  shared::Nvfp4ExpertBlockLayout layout{};
  EXPECT_FALSE(shared::MakeInt3ExpertBlockLayout(0, 64, 32, &layout));
  EXPECT_FALSE(shared::MakeInt3ExpertBlockLayout(128, 64, 16, &layout));
  EXPECT_FALSE(shared::MakeInt3ExpertBlockLayout(128, 32, 64, &layout));

  ASSERT_TRUE(shared::MakeInt3ExpertBlockLayout(256, 128, 32, &layout));
  EXPECT_EQ(layout.fc1_weights % shared::kBankAlignment, 0U);
  EXPECT_EQ(layout.fc1_scales % shared::kBankAlignment, 0U);
  EXPECT_EQ(layout.fc2_weights % shared::kBankAlignment, 0U);
  EXPECT_EQ(layout.fc2_scales % shared::kBankAlignment, 0U);
  EXPECT_EQ(layout.stride, 43520U);

  ASSERT_TRUE(shared::MakeInt3ExpertBlockLayout(256, 128, 64, &layout));
  EXPECT_EQ(layout.stride, 40448U);
}

TEST(QMoESharedLayoutTest, Avx512Int3GemmAndExpertRangesMatchScalarForBlock32And64) {
  if (!shared::CpuHasAvx512()) GTEST_SKIP() << "AVX-512 kernels are not available";
  constexpr int64_t hidden = 128;
  constexpr int64_t inter = 64;
  constexpr int64_t n1 = 2 * inter;
  std::vector<uint8_t> fc1_w(static_cast<size_t>(n1 * (hidden * 3) / 8));
  std::vector<uint8_t> fc2_w(static_cast<size_t>(hidden * (inter * 3) / 8));
  for (size_t i = 0; i < fc1_w.size(); ++i) fc1_w[i] = static_cast<uint8_t>(i * 73 + i / 11);
  for (size_t i = 0; i < fc2_w.size(); ++i) fc2_w[i] = static_cast<uint8_t>(i * 59 + i / 7);

  for (int64_t block_size : {int64_t{32}, int64_t{64}}) {
    for (bool scale_is_bf16 : {false, true}) {
      std::vector<uint16_t> fc1_s(static_cast<size_t>(n1 * (hidden / block_size)));
      std::vector<uint16_t> fc2_s(static_cast<size_t>(hidden * (inter / block_size)));
      std::vector<uint16_t> fc1_b(static_cast<size_t>(n1));
      std::vector<uint16_t> fc2_b(static_cast<size_t>(hidden));
      auto encode = [&](float v) -> uint16_t {
        return scale_is_bf16 ? shared::FloatToBFloat16(v) : shared::FloatToHalf(v);
      };
      for (size_t i = 0; i < fc1_s.size(); ++i) fc1_s[i] = encode((static_cast<int>(i % 7) + 1) / 64.0f);
      for (size_t i = 0; i < fc2_s.size(); ++i) fc2_s[i] = encode((static_cast<int>(i % 5) + 1) / 64.0f);
      for (size_t i = 0; i < fc1_b.size(); ++i) fc1_b[i] = encode((static_cast<int>(i % 9) - 4) / 32.0f);
      for (size_t i = 0; i < fc2_b.size(); ++i) fc2_b[i] = encode((static_cast<int>(i % 11) - 5) / 32.0f);

      shared::Int3ExpertView view{};
      view.fc1_weights = fc1_w.data();
      view.fc1_scales = fc1_s.data();
      view.fc1_bias = fc1_b.data();
      view.fc2_weights = fc2_w.data();
      view.fc2_scales = fc2_s.data();
      view.fc2_bias = fc2_b.data();
      view.block_size = block_size;
      view.scale_is_bf16 = scale_is_bf16;
      view.bias_is_bf16 = scale_is_bf16;

      const shared::SwiGluParams swiglu{1.702f, 1.0f, 7.0f};
      for (int64_t rows = 1; rows <= shared::kMaxRowsPerCall; ++rows) {
        std::vector<float> x(static_cast<size_t>(rows * hidden));
        for (size_t i = 0; i < x.size(); ++i) x[i] = (static_cast<int>(i % 19) - 9) / 32.0f;

        std::vector<float> inter_scalar(static_cast<size_t>(rows * inter), 0.0f);
        std::vector<float> inter_simd(static_cast<size_t>(rows * inter), 0.0f);
        shared::Int3Fc1SwiGluRange(hidden, inter, view, x.data(), rows, swiglu, inter_scalar.data(),
                                   0, n1, /*force_scalar=*/true, scale_is_bf16, !scale_is_bf16);
        shared::Int3Fc1SwiGluRange(hidden, inter, view, x.data(), rows, swiglu, inter_simd.data(),
                                   0, n1, /*force_scalar=*/false, scale_is_bf16, !scale_is_bf16);
        for (size_t i = 0; i < inter_scalar.size(); ++i) {
          EXPECT_NEAR(inter_simd[i], inter_scalar[i], 1e-5f);
        }

        const std::array<std::pair<int64_t, int64_t>, 4> col_ranges{{{0, 13}, {2, 19}, {16, 127}, {0, hidden}}};
        for (const auto& [begin, end] : col_ranges) {
          std::vector<float> out_scalar(static_cast<size_t>(rows * hidden), -999.0f);
          std::vector<float> out_simd(static_cast<size_t>(rows * hidden), -999.0f);
          shared::Int3Fc2Range(hidden, inter, view, inter_scalar.data(), rows, out_scalar.data(),
                               begin, end, /*force_scalar=*/true, /*apply_bias=*/true);
          shared::Int3Fc2Range(hidden, inter, view, inter_simd.data(), rows, out_simd.data(),
                               begin, end, /*force_scalar=*/false, /*apply_bias=*/true);
          for (size_t i = 0; i < out_scalar.size(); ++i) {
            EXPECT_NEAR(out_simd[i], out_scalar[i], 1e-5f);
          }
        }
      }
    }
  }
}

TEST(QMoESharedLayoutTest, CanonicalInt3ExpertViewMatchesReferenceAcrossBlockSizesFp16AndBf16) {
  constexpr int64_t hidden = 256;
  constexpr int64_t inter = 128;
  constexpr int64_t rows = 5;
  const shared::SwiGluParams swiglu{1.702f, 1.0f, 7.0f};

  std::vector<float> x(static_cast<size_t>(rows * hidden));
  for (size_t i = 0; i < x.size(); ++i) {
    x[i] = static_cast<float>(static_cast<int>(i % 23) - 11) * 0.0625f;
  }

  const size_t fc1_w_bytes = static_cast<size_t>(2 * inter) * ((hidden * 3) / 8);
  const size_t fc2_w_bytes = static_cast<size_t>(hidden) * ((inter * 3) / 8);
  std::vector<uint8_t> fc1_w(fc1_w_bytes), fc2_w(fc2_w_bytes);
  for (size_t i = 0; i < fc1_w.size(); ++i) fc1_w[i] = static_cast<uint8_t>((i * 41 + 7) & 0xFF);
  for (size_t i = 0; i < fc2_w.size(); ++i) fc2_w[i] = static_cast<uint8_t>((i * 67 + 13) & 0xFF);

  std::vector<uint16_t> fc1_b_fp16(static_cast<size_t>(2 * inter)), fc1_b_bf16(static_cast<size_t>(2 * inter));
  std::vector<uint16_t> fc2_b_fp16(static_cast<size_t>(hidden)), fc2_b_bf16(static_cast<size_t>(hidden));
  for (size_t i = 0; i < fc1_b_fp16.size(); ++i) {
    const float v = static_cast<float>(static_cast<int>(i % 13) - 6) * 0.03125f;
    fc1_b_fp16[i] = shared::FloatToHalf(v);
    fc1_b_bf16[i] = shared::FloatToBFloat16(v);
  }
  for (size_t i = 0; i < fc2_b_fp16.size(); ++i) {
    const float v = static_cast<float>(static_cast<int>(i % 11) - 5) * 0.03125f;
    fc2_b_fp16[i] = shared::FloatToHalf(v);
    fc2_b_bf16[i] = shared::FloatToBFloat16(v);
  }

  for (int64_t block_size : {int64_t{32}, int64_t{64}, int64_t{128}}) {
    const size_t fc1_s_count = static_cast<size_t>(2 * inter) * static_cast<size_t>(hidden / block_size);
    const size_t fc2_s_count = static_cast<size_t>(hidden) * static_cast<size_t>(inter / block_size);
    std::vector<uint16_t> fc1_s_fp16(fc1_s_count), fc1_s_bf16(fc1_s_count);
    std::vector<uint16_t> fc2_s_fp16(fc2_s_count), fc2_s_bf16(fc2_s_count);
    for (size_t i = 0; i < fc1_s_count; ++i) {
      const float v = 0.03125f * static_cast<float>((i % 7) + 1);
      fc1_s_fp16[i] = shared::FloatToHalf(v);
      fc1_s_bf16[i] = shared::FloatToBFloat16(v);
    }
    for (size_t i = 0; i < fc2_s_count; ++i) {
      const float v = 0.03125f * static_cast<float>((i % 5) + 1);
      fc2_s_fp16[i] = shared::FloatToHalf(v);
      fc2_s_bf16[i] = shared::FloatToBFloat16(v);
    }

    for (bool is_bf16 : {false, true}) {
      for (bool apply_fc2_bias : {true, false}) {
        for (int64_t m_rows : {int64_t{1}, int64_t{2}, int64_t{3}, rows}) {
          shared::Int3ExpertView view{};
          view.fc1_weights = fc1_w.data();
          view.fc1_scales = is_bf16 ? fc1_s_bf16.data() : fc1_s_fp16.data();
          view.fc1_bias = is_bf16 ? fc1_b_bf16.data() : fc1_b_fp16.data();
          view.fc2_weights = fc2_w.data();
          view.fc2_scales = is_bf16 ? fc2_s_bf16.data() : fc2_s_fp16.data();
          view.fc2_bias = is_bf16 ? fc2_b_bf16.data() : fc2_b_fp16.data();
          view.block_size = block_size;
          view.scale_is_bf16 = is_bf16;
          view.bias_is_bf16 = is_bf16;

          std::vector<float> out_scalar(static_cast<size_t>(m_rows * hidden));
          std::vector<float> out_simd(static_cast<size_t>(m_rows * hidden));
          ASSERT_TRUE(shared::Int3ExpertForward(hidden, inter, view, x.data(), m_rows, swiglu, out_scalar.data(),
                                                /*force_scalar=*/true, /*apply_fc2_bias=*/apply_fc2_bias,
                                                /*bf16_boundaries=*/is_bf16, /*fp16_boundaries=*/!is_bf16));
          ASSERT_TRUE(shared::Int3ExpertForward(hidden, inter, view, x.data(), m_rows, swiglu, out_simd.data(),
                                                /*force_scalar=*/false, /*apply_fc2_bias=*/apply_fc2_bias,
                                                /*bf16_boundaries=*/is_bf16, /*fp16_boundaries=*/!is_bf16));
          for (size_t i = 0; i < out_scalar.size(); ++i) {
            EXPECT_NEAR(out_simd[i], out_scalar[i], 1e-5f)
                << "block_size=" << block_size << " bf16=" << is_bf16 << " bias=" << apply_fc2_bias
                << " m=" << m_rows << " i=" << i;
          }
        }
      }
    }
  }
}

TEST(QMoESharedLayoutTest, Int3GuardPageAndReadOnlyBanks) {
#if defined(__linux__)
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  auto* mapping = static_cast<uint8_t*>(mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE,
                                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(mapping, MAP_FAILED);
  ASSERT_EQ(mprotect(mapping + page, page, PROT_NONE), 0);
  uint8_t* weights = mapping + page - 24;
  std::memset(weights, 0x55, 24);
  ASSERT_EQ(mprotect(mapping, page, PROT_READ), 0);
  const uint16_t scale = shared::FloatToBFloat16(0.1001f);
  std::array<float, 64> a;
  a.fill(0.25f);
  float scalar = 0.0f;
  float dispatched = 0.0f;
  shared::Int3GemmRange(weights, &scale, nullptr, 64, /*scale_is_bf16=*/true, /*bias_is_bf16=*/true,
                        1, 64, a.data(), 1, &scalar, 1, 0, 1, /*force_scalar=*/true);
  shared::Int3GemmRange(weights, &scale, nullptr, 64, /*scale_is_bf16=*/true, /*bias_is_bf16=*/true,
                        1, 64, a.data(), 1, &dispatched, 1, 0, 1, /*force_scalar=*/false);
  EXPECT_NEAR(scalar, dispatched, 1e-6f);
  EXPECT_EQ(munmap(mapping, 2 * page), 0);
#else
  GTEST_SKIP() << "Read-only guard-page test requires mmap.";
#endif
}

TEST(QMoESharedLayoutTest, IntegerDeferredScaleModesPreservePartialRanges) {
  constexpr int64_t k = 128, n = 9;
  for (int bits : {3, 4}) {
    const int64_t row_bytes = k * bits / 8;
    std::vector<uint8_t> weights(static_cast<size_t>(n * row_bytes));
    for (int64_t column = 0; column < n; ++column) {
      uint8_t* row = weights.data() + column * row_bytes;
      if (bits == 4) {
        std::fill(row, row + row_bytes, 0x88);
        row[0] = 0x9b;  // Signed codes [3, 1, 0, ...].
      } else {
        for (int64_t i = 0; i < row_bytes; i += 3) {
          row[i] = 0x24;
          row[i + 1] = 0x49;
          row[i + 2] = 0x92;
        }
        row[0] = 0x2f;
      }
    }
    for (bool bf16 : {false, true}) {
      const uint16_t scale = bf16 ? uint16_t{0x3dcd} : shared::FloatToHalf(0.1f);
      const float typed_expected = bf16 ? 0.5f : -0.125f;
      for (int64_t block : {int64_t{32}, int64_t{64}, int64_t{128}}) {
        if (bits == 4 && block != 32) continue;  // The direct INT4 GEMM interface has block size 32.
        std::vector<uint16_t> scales(static_cast<size_t>(n * k / block), scale);
        for (int64_t rows : {int64_t{1}, int64_t{3}, int64_t{8}}) {
          std::vector<float> x(static_cast<size_t>(rows * k), 0.0f), reordered(x.size());
          for (int64_t r = 0; r < rows; ++r) {
            x[r * k] = 1024;
            x[r * k + 1] = -3072;
            shared::ReorderRawInt4ActivationRow(x.data() + r * k, reordered.data() + r * k, k);
          }
          for (bool deferred : {false, true}) {
            for (bool scalar : {false, true}) {
              for (auto [begin, end] : {std::pair<int64_t, int64_t>{0, n}, {1, 8}}) {
                std::vector<float> out(static_cast<size_t>(rows * n), -999.0f);
                if (bits == 3) {
                  shared::Int3GemmRange(weights.data(), scales.data(), nullptr, block, bf16, bf16, n, k,
                                        x.data(), rows, out.data() + begin, n, begin, end, scalar, deferred);
                } else {
                  shared::Int4GemmRange(weights.data(), scales.data(), nullptr, false, false, bf16, bf16, n, k,
                                        reordered.data(), rows, out.data() + begin, n, begin, end, scalar, deferred);
                }
                for (int64_t r = 0; r < rows; ++r) {
                  for (int64_t c = 0; c < n; ++c) {
                    const float expected = c < begin || c >= end ? -999.0f : (deferred ? 0.0f : typed_expected);
                    EXPECT_EQ(out[r * n + c], expected)
                        << "bits=" << bits << " bf16=" << bf16 << " block=" << block
                        << " rows=" << rows << " deferred=" << deferred << " scalar=" << scalar << " c=" << c;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(QMoESharedLayoutTest, Int2ExpertBlockLayoutGeometry) {
  shared::Nvfp4ExpertBlockLayout layout{};
  EXPECT_FALSE(shared::MakeInt2ExpertBlockLayout(128, 64, 16, &layout));
  EXPECT_FALSE(shared::MakeInt2ExpertBlockLayout(96, 64, 64, &layout));
  EXPECT_FALSE(shared::MakeInt2ExpertBlockLayout(128, 64, 64, nullptr));

  ASSERT_TRUE(shared::MakeInt2ExpertBlockLayout(2560, 640, 64, &layout));
  EXPECT_EQ(layout.fc1_weights, 0U);
  EXPECT_EQ(layout.fc1_scales, 819200U);
  EXPECT_EQ(layout.fc1_global_scale, 921600U);
  EXPECT_EQ(layout.fc2_weights, 921856U);
  EXPECT_EQ(layout.fc2_scales, 1331456U);
  EXPECT_EQ(layout.fc2_global_scale, 1382656U);
  EXPECT_EQ(layout.stride, 1382912U);
  EXPECT_EQ(layout.fc2_weights % shared::kBankAlignment, 0U);
  EXPECT_EQ(layout.fc1_scales % shared::kBankAlignment, 0U);
  EXPECT_EQ(layout.fc2_scales % shared::kBankAlignment, 0U);

  ASSERT_TRUE(shared::MakeInt2ExpertBlockLayout(256, 128, 32, &layout));
  EXPECT_EQ(layout.stride % shared::kBankAlignment, 0U);
  ASSERT_TRUE(shared::MakeInt2ExpertBlockLayout(256, 128, 128, &layout));
  EXPECT_EQ(layout.stride % shared::kBankAlignment, 0U);
}

TEST(QMoESharedLayoutTest, Int2ReorderedPositionAndGemmMatchIndependentReference) {
  constexpr int64_t k = 128;
  constexpr int64_t columns = 24;
  for (int64_t b = 0; b < 16; ++b) {
    for (int64_t s = 0; s < 4; ++s) {
      const int64_t logical_k = 32 * (b & 1) + 8 * s + (b >> 1);
      EXPECT_EQ(shared::Int2ReorderedPosition(static_cast<int>(logical_k)), 16 * s + b);
      const int64_t raw_k = 4 * b + s;
      EXPECT_EQ(shared::RawInt2ReorderedPosition(static_cast<int>(raw_k)), 16 * s + b);
    }
  }

  std::vector<float> a_natural(static_cast<size_t>(k));
  for (int64_t i = 0; i < k; ++i) {
    a_natural[static_cast<size_t>(i)] = static_cast<float>((i % 17) - 8) * 0.125f;
  }
  std::vector<float> a_prepacked(static_cast<size_t>(k));
  std::vector<float> a_raw(static_cast<size_t>(k));
  shared::ReorderInt2ActivationRow(a_natural.data(), a_prepacked.data(), k);
  shared::ReorderRawInt2ActivationRow(a_natural.data(), a_raw.data(), k);
  for (int64_t tile = 0; tile < k / 64; ++tile) {
    for (int64_t off = 0; off < 64; ++off) {
      EXPECT_EQ(a_prepacked[static_cast<size_t>(tile * 64 + shared::Int2ReorderedPosition(static_cast<int>(off)))],
                a_natural[static_cast<size_t>(tile * 64 + off)]);
      EXPECT_EQ(a_raw[static_cast<size_t>(tile * 64 + shared::RawInt2ReorderedPosition(static_cast<int>(off)))],
                a_natural[static_cast<size_t>(tile * 64 + off)]);
    }
  }

  std::vector<uint8_t> weights(static_cast<size_t>(columns * (k / 4)));
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] = static_cast<uint8_t>((i * 73 + 19) & 0xFF);
  }

  std::vector<shared::Int2GemmBackend> backends{shared::Int2GemmBackend::Auto};
  if (shared::CompilerHasAvx2() && shared::CpuHasAvx2FmaF16c()) {
    backends.push_back(shared::Int2GemmBackend::Avx2);
  }
  for (int64_t block_size : {32, 64, 128}) {
    const int64_t k_blocks = k / block_size;
    for (bool bf16 : {false, true}) {
      std::vector<uint16_t> scales(static_cast<size_t>(columns * k_blocks));
      std::vector<uint16_t> block_scales(scales.size());
      std::vector<uint16_t> biases(static_cast<size_t>(columns));
      for (int64_t n = 0; n < columns; ++n) {
        const float bias = 0.03125f * static_cast<float>(n % 7 - 3);
        biases[static_cast<size_t>(n)] = bf16 ? shared::FloatToBFloat16(bias) : shared::FloatToHalf(bias);
        for (int64_t b = 0; b < k_blocks; ++b) {
          const float scale = 0.0625f * static_cast<float>((n * k_blocks + b) % 5 + 1);
          const auto bits = bf16 ? shared::FloatToBFloat16(scale) : shared::FloatToHalf(scale);
          scales[static_cast<size_t>(n * k_blocks + b)] = bits;
          block_scales[static_cast<size_t>(b * columns + n)] = bits;
        }
      }
      for (bool prepacked : {true, false}) {
        std::vector<float> ref(static_cast<size_t>(columns), 0.0f);
        for (int64_t n = 0; n < columns; ++n) {
          double col_acc = 0.0;
          for (int64_t i = 0; i < k; ++i) {
            const int64_t offset = prepacked
                                       ? (n / 8 * 2 * k + i / 64 * 128 + n % 8 * 16 +
                                          2 * (i % 8) + (i % 64) / 32)
                                       : (n * k / 4 + i / 4);
            const int shift = static_cast<int>(prepacked ? 2 * ((i % 32) / 8) : 2 * (i % 4));
            const int code = (weights[static_cast<size_t>(offset)] >> shift) & 3;
            const auto bits = scales[static_cast<size_t>(n * k_blocks + i / block_size)];
            const double scale = bf16 ? shared::BFloat16ToFloat(bits) : shared::HalfToFloat(bits);
            col_acc += a_natural[static_cast<size_t>(i)] * (code - 2) * scale;
          }
          ref[static_cast<size_t>(n)] = static_cast<float>(col_acc);
        }
        for (int64_t rows : {1, 2, 3, 5, shared::kMaxRowsPerCall}) {
          std::vector<float> activations(static_cast<size_t>(rows * k));
          const auto& row = prepacked ? a_prepacked : a_raw;
          for (int64_t r = 0; r < rows; ++r) {
            for (int64_t i = 0; i < k; ++i) {
              activations[static_cast<size_t>(r * k + i)] = row[static_cast<size_t>(i)] * static_cast<float>(r + 1);
            }
          }
          for (bool block_major : {false, true}) {
            for (auto backend : backends) {
              for (bool scalar : {false, true}) {
                for (const auto& [begin, end] : {std::pair<int64_t, int64_t>{0, columns}, {1, 23}, {4, 12}, {3, 4}}) {
                  SCOPED_TRACE(testing::Message() << "block=" << block_size << " bf16=" << bf16
                                                  << " prepacked=" << prepacked << " block_major=" << block_major << " rows=" << rows
                                                  << " backend=" << static_cast<int>(backend) << " scalar=" << scalar
                                                  << " range=[" << begin << "," << end << ")");
                  std::vector<float> out(static_cast<size_t>(rows * columns), -999.0f);
                  shared::Int2GemmRange(weights.data(), block_major ? block_scales.data() : scales.data(),
                                        biases.data(), block_size, block_major, prepacked, bf16, bf16, columns, k,
                                        activations.data(), rows, out.data() + begin, columns, begin, end, scalar, backend);
                  for (int64_t r = 0; r < rows; ++r) {
                    for (int64_t n = 0; n < columns; ++n) {
                      const auto bias_bits = biases[static_cast<size_t>(n)];
                      const float bias = bf16 ? shared::BFloat16ToFloat(bias_bits) : shared::HalfToFloat(bias_bits);
                      EXPECT_NEAR(out[static_cast<size_t>(r * columns + n)],
                                  n >= begin && n < end ? ref[static_cast<size_t>(n)] * static_cast<float>(r + 1) + bias : -999.0f,
                                  1e-5f);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(QMoESharedLayoutTest, Int2GemmAndExpertRangesMatchScalarAcrossBlockSizesFp16AndBf16) {
  constexpr int64_t hidden = 128;
  constexpr int64_t inter = 128;
  constexpr int64_t n1 = 2 * inter;
  std::vector<uint8_t> fc1_w(static_cast<size_t>(n1 * (hidden / 4)));
  std::vector<uint8_t> fc2_w(static_cast<size_t>(hidden * (inter / 4)));
  for (size_t i = 0; i < fc1_w.size(); ++i) fc1_w[i] = static_cast<uint8_t>(i * 73 + i / 11);
  for (size_t i = 0; i < fc2_w.size(); ++i) fc2_w[i] = static_cast<uint8_t>(i * 59 + i / 7);

  for (int64_t block_size : {int64_t{32}, int64_t{64}, int64_t{128}}) {
    for (bool scale_is_bf16 : {false, true}) {
      for (bool prepacked : {true, false}) {
        for (bool block_major : {false, true}) {
          std::vector<uint16_t> fc1_s(static_cast<size_t>(n1 * (hidden / block_size)));
          std::vector<uint16_t> fc2_s(static_cast<size_t>(hidden * (inter / block_size)));
          std::vector<uint16_t> fc1_b(static_cast<size_t>(n1));
          std::vector<uint16_t> fc2_b(static_cast<size_t>(hidden));
          auto encode = [&](float v) -> uint16_t {
            return scale_is_bf16 ? shared::FloatToBFloat16(v) : shared::FloatToHalf(v);
          };
          for (size_t i = 0; i < fc1_s.size(); ++i) fc1_s[i] = encode((static_cast<int>(i % 7) + 1) / 64.0f);
          for (size_t i = 0; i < fc2_s.size(); ++i) fc2_s[i] = encode((static_cast<int>(i % 5) + 1) / 64.0f);
          for (size_t i = 0; i < fc1_b.size(); ++i) fc1_b[i] = encode((static_cast<int>(i % 9) - 4) / 32.0f);
          for (size_t i = 0; i < fc2_b.size(); ++i) fc2_b[i] = encode((static_cast<int>(i % 11) - 5) / 32.0f);

          if (block_major) {
            auto transpose = [](std::vector<uint16_t>& scales, int64_t n, int64_t blocks) {
              const auto rows = scales;
              for (int64_t c = 0; c < n; ++c) {
                for (int64_t b = 0; b < blocks; ++b) {
                  scales[static_cast<size_t>(b * n + c)] = rows[static_cast<size_t>(c * blocks + b)];
                }
              }
            };
            transpose(fc1_s, n1, hidden / block_size);
            transpose(fc2_s, hidden, inter / block_size);
          }
          shared::Int2ExpertView view{};
          view.fc1_weights = fc1_w.data();
          view.fc1_scales = fc1_s.data();
          view.fc1_bias = fc1_b.data();
          view.fc2_weights = fc2_w.data();
          view.fc2_scales = fc2_s.data();
          view.fc2_bias = fc2_b.data();
          view.block_size = block_size;
          view.scales_are_block_major = block_major;
          view.scale_is_bf16 = scale_is_bf16;
          view.bias_is_bf16 = scale_is_bf16;
          view.weights_are_prepacked = prepacked;

          const shared::SwiGluParams swiglu{1.702f, 1.0f, 7.0f};
          for (int64_t rows : {int64_t{1}, int64_t{2}, int64_t{5}, int64_t{shared::kMaxRowsPerCall}}) {
            std::vector<float> x(static_cast<size_t>(rows * hidden));
            for (size_t i = 0; i < x.size(); ++i) x[i] = (static_cast<int>(i % 19) - 9) / 32.0f;

            std::vector<float> inter_scalar(static_cast<size_t>(rows * inter), 0.0f);
            std::vector<float> inter_simd(static_cast<size_t>(rows * inter), 0.0f);
            shared::Int2Fc1SwiGluRange(hidden, inter, view, x.data(), rows, swiglu, inter_scalar.data(),
                                       0, n1, /*force_scalar=*/true, scale_is_bf16, !scale_is_bf16);
            shared::Int2Fc1SwiGluRange(hidden, inter, view, x.data(), rows, swiglu, inter_simd.data(),
                                       0, n1, /*force_scalar=*/false, scale_is_bf16, !scale_is_bf16);
            for (const auto& [begin, end] : {std::pair<int64_t, int64_t>{0, 2}, {2, 130}, {126, n1}}) {
              std::vector<float> partial(static_cast<size_t>(rows * inter), -999.0f);
              ASSERT_TRUE(shared::Int2Fc1SwiGluRange(hidden, inter, view, x.data(), rows, swiglu, partial.data(),
                                                     begin, end, false, scale_is_bf16, !scale_is_bf16));
              for (int64_t r = 0; r < rows; ++r) {
                for (int64_t j = begin / 2; j < end / 2; ++j) {
                  const int pos = prepacked ? shared::Int2ReorderedPosition(static_cast<int>(j % 64))
                                            : shared::RawInt2ReorderedPosition(static_cast<int>(j % 64));
                  const auto index = static_cast<size_t>(r * inter + (j / 64) * 64 + pos);
                  EXPECT_EQ(partial[index], inter_simd[index]);
                }
              }
            }
            for (size_t i = 0; i < inter_scalar.size(); ++i) {
              EXPECT_NEAR(inter_simd[i], inter_scalar[i], 1e-5f);
            }

            const std::array<std::pair<int64_t, int64_t>, 4> col_ranges{{{0, 13}, {2, 19}, {16, 127}, {0, hidden}}};
            for (const auto& [begin, end] : col_ranges) {
              std::vector<float> out_scalar(static_cast<size_t>(rows * hidden), -999.0f);
              std::vector<float> out_simd(static_cast<size_t>(rows * hidden), -999.0f);
              shared::Int2Fc2Range(hidden, inter, view, inter_scalar.data(), rows, out_scalar.data(),
                                   begin, end, /*force_scalar=*/true, /*apply_bias=*/true);
              shared::Int2Fc2Range(hidden, inter, view, inter_simd.data(), rows, out_simd.data(),
                                   begin, end, /*force_scalar=*/false, /*apply_bias=*/true);
              for (size_t i = 0; i < out_scalar.size(); ++i) {
                EXPECT_NEAR(out_simd[i], out_scalar[i], 1e-5f);
              }
            }

            std::vector<float> fwd_scalar(static_cast<size_t>(rows * hidden), 0.0f);
            std::vector<float> fwd_simd(static_cast<size_t>(rows * hidden), 0.0f);
            ASSERT_TRUE(shared::Int2ExpertForward(hidden, inter, view, x.data(), rows, swiglu, fwd_scalar.data(),
                                                  /*force_scalar=*/true, /*apply_fc2_bias=*/true,
                                                  /*bf16_boundaries=*/scale_is_bf16, /*fp16_boundaries=*/!scale_is_bf16));
            ASSERT_TRUE(shared::Int2ExpertForward(hidden, inter, view, x.data(), rows, swiglu, fwd_simd.data(),
                                                  /*force_scalar=*/false, /*apply_fc2_bias=*/true,
                                                  /*bf16_boundaries=*/scale_is_bf16, /*fp16_boundaries=*/!scale_is_bf16));
            for (size_t i = 0; i < fwd_scalar.size(); ++i) {
              EXPECT_NEAR(fwd_simd[i], fwd_scalar[i], 1e-5f);
            }
          }
        }
      }
    }
  }
}

TEST(QMoESharedLayoutTest, Int2ScalesBeforeLargeActivationProducts) {
  constexpr int64_t k = 128;
  constexpr int64_t columns = 16;
  constexpr int64_t rows = 3;
  const std::vector<uint8_t> weights(static_cast<size_t>(columns * k / 4), 0);  // All weights are -2.
  for (int64_t block : {32, 64, 128}) {
    for (bool bf16 : {false, true}) {
      const int64_t blocks = k / block;
      std::vector<uint16_t> scales(static_cast<size_t>(columns * blocks));
      for (size_t i = 0; i < scales.size(); ++i) {
        const float s = 1e-4f * static_cast<float>(1 + i % 3);
        scales[i] = bf16 ? shared::FloatToBFloat16(s) : shared::FloatToHalf(s);
      }
      for (bool prepacked : {false, true}) {
        std::vector<float> natural(static_cast<size_t>(rows * k), 0.0f);
        natural[33] = -2e38f;  // One unscaled product overflows.
        natural[static_cast<size_t>(k + 97)] = -2e38f;
        std::fill(natural.begin() + 2 * k, natural.end(), -1e37f);  // The unscaled sum overflows.
        std::vector<float> reordered(natural.size());
        for (int64_t r = 0; r < rows; ++r) {
          if (prepacked) {
            shared::ReorderInt2ActivationRow(natural.data() + r * k, reordered.data() + r * k, k);
          } else {
            shared::ReorderRawInt2ActivationRow(natural.data() + r * k, reordered.data() + r * k, k);
          }
        }
        // Full groups and four-column subranges exercise both vector tile implementations.
        for (const auto& [begin, end] : {std::pair<int64_t, int64_t>{0, columns}, {4, 12}}) {
          for (bool scalar : {false, true}) {
            std::vector<float> output(static_cast<size_t>(rows * columns));
            shared::Int2GemmRange(weights.data(), scales.data(), nullptr, block, false, prepacked,
                                  bf16, false, columns, k, reordered.data(), rows, output.data(), columns,
                                  begin, end, scalar);
            for (int64_t r = 0; r < rows; ++r) {
              for (int64_t n = begin; n < end; ++n) {
                double expected = 0;
                for (int64_t i = 0; i < k; ++i) {
                  const auto bits = scales[static_cast<size_t>(n * blocks + i / block)];
                  const double scale = bf16 ? shared::BFloat16ToFloat(bits) : shared::HalfToFloat(bits);
                  expected += -2.0 * scale * natural[static_cast<size_t>(r * k + i)];
                }
                const float actual = output[static_cast<size_t>(r * columns + n - begin)];
                ASSERT_TRUE(std::isfinite(actual));
                EXPECT_NEAR(actual / expected, 1.0, 2e-6)
                    << "block=" << block << " bf16=" << bf16 << " prepacked=" << prepacked;
              }
            }
          }
        }
      }
    }
  }
}

TEST(QMoESharedLayoutTest, Int2SwiGluRoundsProjectionsBeforeBiasAndActivation) {
  constexpr int64_t hidden = 128;
  constexpr int64_t inter = 128;
  constexpr int64_t columns = 2 * inter;
  for (int64_t block : {32, 64, 128}) {
    for (bool prepacked : {false, true}) {
      std::vector<uint8_t> weights(static_cast<size_t>(columns * hidden / 4));
      std::vector<uint16_t> scales(static_cast<size_t>(columns * hidden / block));
      for (int64_t n = 0; n < columns; ++n) {
        // Gate weight -2, up weight +1; only the first activation is nonzero.
        for (int64_t tile = 0; tile < hidden / 64; ++tile) {
          const int64_t offset = prepacked ? (n / 8 * 2 * hidden + tile * 128 + n % 8 * 16)
                                           : (n * hidden / 4 + tile * 16);
          std::fill_n(weights.data() + offset, 16, (n % 2 == 0) ? 0 : 0xFF);
        }
        std::fill_n(scales.data() + n * (hidden / block), hidden / block,
                    shared::FloatToHalf((n % 2 == 0) ? 1.0f : 1e-6f));
      }
      shared::Int2ExpertView view{};
      view.fc1_weights = weights.data();
      view.fc1_scales = scales.data();
      view.block_size = block;
      view.scale_is_bf16 = false;
      view.weights_are_prepacked = prepacked;
      for (bool scalar : {false, true}) {
        for (float activation : {-40000.0f, 40000.0f}) {
          std::vector<float> natural(static_cast<size_t>(hidden), 0.0f), reordered(natural.size());
          natural[0] = activation;
          if (prepacked) {
            shared::ReorderInt2ActivationRow(natural.data(), reordered.data(), hidden);
          } else {
            shared::ReorderRawInt2ActivationRow(natural.data(), reordered.data(), hidden);
          }
          std::vector<float> output(static_cast<size_t>(inter));
          shared::Int2Fc1SwiGluRange(hidden, inter, view, reordered.data(), 1, {1.0f, 0.0f, 100000.0f}, output.data(),
                                     0, columns, scalar, false, true);
          const double gate = std::min(100000.0, static_cast<double>(shared::HalfToFloat(shared::FloatToHalf(-2.0f * activation))));
          const double up = shared::HalfToFloat(shared::FloatToHalf(
              shared::HalfToFloat(shared::FloatToHalf(1e-6f)) * activation));
          const float expected = shared::HalfToFloat(shared::FloatToHalf(
              static_cast<float>((gate < 0 ? 0.0 : gate / (1.0 + std::exp(-gate))) * up)));
          for (float value : output) {
            ASSERT_TRUE(std::isfinite(value));
            EXPECT_EQ(value, expected);
          }
        }
      }
    }
  }
}

TEST(QMoESharedLayoutTest, Int2ExtremeBf16ScalesUseWiderIntermediates) {
  constexpr int64_t k = 128, columns = 16, rows = 3;
  const std::vector<uint8_t> weights(static_cast<size_t>(columns * k / 4), 0);
  std::vector<shared::Int2GemmBackend> backends{shared::Int2GemmBackend::Auto};
  if (shared::CompilerHasAvx2() && shared::CpuHasAvx2FmaF16c()) {
    backends.push_back(shared::Int2GemmBackend::Avx2);
  }
  for (int64_t block : {32, 64, 128}) {
    const int64_t blocks = k / block;
    for (bool block_major : {false, true}) {
      std::vector<uint16_t> scales(static_cast<size_t>(columns * blocks));
      for (int64_t n = 0; n < columns; ++n) {
        for (int64_t b = 0; b < blocks; ++b) {
          scales[static_cast<size_t>(block_major ? b * columns + n : n * blocks + b)] =
              shared::FloatToBFloat16((n + b) % 3 == 0 ? -2e38f : 2e38f);
        }
      }
      for (bool prepacked : {false, true}) {
        std::vector<float> input(static_cast<size_t>(rows * k)), reordered(input.size());
        for (int64_t r = 0; r < rows; ++r) {
          std::fill_n(input.data() + r * k, k, static_cast<float>(r + 1) * 1e-30f);
          if (prepacked) {
            shared::ReorderInt2ActivationRow(input.data() + r * k, reordered.data() + r * k, k);
          } else {
            shared::ReorderRawInt2ActivationRow(input.data() + r * k, reordered.data() + r * k, k);
          }
        }
        for (auto backend : backends) {
          for (bool scalar : {false, true}) {
            for (const auto& [begin, end] : {std::pair<int64_t, int64_t>{0, columns}, {1, 15}, {4, 12}}) {
              std::vector<float> out(static_cast<size_t>(rows * columns));
              shared::Int2GemmRange(weights.data(), scales.data(), nullptr, block, block_major, prepacked,
                                    true, true, columns, k, reordered.data(), rows, out.data(), columns, begin, end, scalar, backend);
              for (int64_t r = 0; r < rows; ++r) {
                for (int64_t n = begin; n < end; ++n) {
                  double expected = 0;
                  for (int64_t i = 0; i < k; ++i) {
                    const auto bits = scales[static_cast<size_t>(block_major ? i / block * columns + n : n * blocks + i / block)];
                    expected += -2.0 * shared::BFloat16ToFloat(bits) * input[static_cast<size_t>(r * k + i)];
                  }
                  const float actual = out[static_cast<size_t>(r * columns + n - begin)];
                  ASSERT_TRUE(std::isfinite(actual));
                  EXPECT_NEAR(actual, expected, std::abs(expected) * 2e-6);
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(QMoESharedLayoutTest, Int2Fc1RejectsIncompletePairsWithoutWritingOutput) {
  std::array<float, 128> output;
  output.fill(-999.0f);
  const shared::Int2ExpertView view{};
  for (const auto& [begin, end] : {std::pair<int64_t, int64_t>{0, 1}, {1, 2}, {1, 3}, {-2, 2}, {4, 2}, {0, 258}}) {
    EXPECT_FALSE(shared::Int2Fc1SwiGluRange(128, 128, view, nullptr, 1, {}, output.data(), begin, end));
    for (float value : output) EXPECT_EQ(value, -999.0f);
  }
  EXPECT_FALSE(shared::Int2Fc1SwiGluRange(128, 128, view, nullptr, shared::kMaxRowsPerCall + 1,
                                          {}, output.data(), 0, 2));
  EXPECT_TRUE(shared::Int2Fc1SwiGluRange(128, 128, view, nullptr, 1, {}, output.data(), 2, 2));
}

TEST(QMoESharedLayoutTest, Int2GuardPageAndReadOnlyBanks) {
#if defined(__linux__)
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  auto* mapping = static_cast<uint8_t*>(mmap(nullptr, 2 * page, PROT_READ | PROT_WRITE,
                                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(mapping, MAP_FAILED);
  ASSERT_EQ(mprotect(mapping + page, page, PROT_NONE), 0);
  uint8_t* weights = mapping + page - 128;
  std::memset(weights, 0x9B, 128);
  ASSERT_EQ(mprotect(mapping, page, PROT_READ), 0);
  std::array<uint16_t, 8> scales;
  scales.fill(shared::FloatToBFloat16(0.125f));
  std::array<float, 64> a;
  a.fill(0.25f);
  std::array<float, 8> scalar{}, dispatched{};
  shared::Int2GemmRange(weights, scales.data(), nullptr, 64,
                        /*scales_are_block_major=*/false, /*weights_are_prepacked=*/true,
                        /*scale_is_bf16=*/true, /*bias_is_bf16=*/true,
                        8, 64, a.data(), 1, scalar.data(), 8, 0, 8, /*force_scalar=*/true);
  shared::Int2GemmRange(weights, scales.data(), nullptr, 64,
                        /*scales_are_block_major=*/false, /*weights_are_prepacked=*/true,
                        /*scale_is_bf16=*/true, /*bias_is_bf16=*/true,
                        8, 64, a.data(), 1, dispatched.data(), 8, 0, 8, /*force_scalar=*/false);
  for (size_t n = 0; n < scalar.size(); ++n) EXPECT_NEAR(scalar[n], dispatched[n], 1e-6f);
  EXPECT_EQ(munmap(mapping, 2 * page), 0);
#else
  GTEST_SKIP() << "Read-only guard-page test requires mmap.";
#endif
}

// A leaked pin removes a slot from a bounded cache for good, so one failed copy would starve every later expert.
class ScopedExpertLeaseTest : public ::testing::Test {
 protected:
  static constexpr size_t kStride = shared::kBankAlignment;
  static constexpr int64_t kNumExperts = 3;

  void SetUp() override {
    path_ = std::filesystem::temp_directory_path() / "ort_qmoe_scoped_lease.bin";
    std::vector<uint8_t> data(static_cast<size_t>(kNumExperts) * kStride);
    for (int64_t e = 0; e < kNumExperts; ++e) {
      std::fill_n(data.data() + static_cast<size_t>(e) * kStride, kStride, static_cast<uint8_t>(e + 1));
    }
    std::ofstream out(path_, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(out.is_open());
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    out.close();
    ASSERT_TRUE(store_.Initialize(path_.string().c_str(), 0, kNumExperts, kStride, /*cache_slots=*/1));
  }

  void TearDown() override {
    store_.Shutdown();
    std::error_code ec;
    std::filesystem::remove(path_, ec);
  }

  // True when the one-slot cache can still serve `expert_id`; the temporary pin is dropped again.
  bool CanServe(int32_t expert_id) {
    shared::ExpertBlockLease lease{};
    if (!store_.AcquireReady(expert_id, &lease)) return false;
    store_.ReleaseLease(expert_id);
    return true;
  }

  std::filesystem::path path_;
  shared::FileExpertStore store_;
};

TEST_F(ScopedExpertLeaseTest, UnguardedPinStarvesOneSlotCache) {
  shared::ExpertBlockLease lease{};
  ASSERT_TRUE(store_.AcquireReady(0, &lease));
  EXPECT_FALSE(CanServe(1));
  store_.ReleaseLease(0);
  EXPECT_TRUE(CanServe(1));
}

TEST_F(ScopedExpertLeaseTest, ReleasesPinOnEarlyReturn) {
  const auto failed_copy = [&]() {
    shared::ExpertBlockLease lease{};
    shared::ScopedExpertLease guard;
    ASSERT_TRUE(guard.AcquireReady(store_, 0, &lease));
    EXPECT_EQ(lease.data[0], 1);
    EXPECT_FALSE(CanServe(1));
    // Returning here mimics a CUDA error return that skips every explicit release.
  };
  failed_copy();
  EXPECT_TRUE(CanServe(1));
  EXPECT_TRUE(CanServe(2));
}

// Throwing is not available in builds compiled without exceptions; ReleasesPinOnEarlyReturn covers the scope guard there.
#ifndef ORT_NO_EXCEPTIONS
TEST_F(ScopedExpertLeaseTest, ReleasesPinWhenAnExceptionUnwinds) {
  try {
    shared::ExpertBlockLease lease{};
    shared::ScopedExpertLease guard;
    ASSERT_TRUE(guard.AcquireReady(store_, 0, &lease));
    throw std::runtime_error("injected copy failure");
  } catch (const std::runtime_error&) {
  }
  EXPECT_TRUE(CanServe(1));
}
#endif  // !ORT_NO_EXCEPTIONS

TEST_F(ScopedExpertLeaseTest, TryAcquireCachedPinIsAlsoGuarded) {
  ASSERT_TRUE(CanServe(0));  // make expert 0 resident
  {
    shared::ExpertBlockLease lease{};
    shared::ScopedExpertLease guard;
    ASSERT_TRUE(guard.TryAcquireCached(store_, 0, &lease));
    EXPECT_FALSE(CanServe(1));
  }
  EXPECT_TRUE(CanServe(1));
}

struct BarrierProbe {
  bool succeed = true;
  int calls = 0;
  bool pin_held_when_called = false;
  shared::FileExpertStore* store = nullptr;
};

bool ProbeBarrier(void* context) noexcept {
  auto* probe = static_cast<BarrierProbe*>(context);
  ++probe->calls;
  // The slot must still be pinned while queued device reads are being completed.
  shared::ExpertBlockLease other{};
  probe->pin_held_when_called = !probe->store->AcquireReady(1, &other);
  if (!probe->pin_held_when_called) probe->store->ReleaseLease(1);
  return probe->succeed;
}

TEST_F(ScopedExpertLeaseTest, FinishesQueuedDeviceReadsBeforeDroppingPin) {
  BarrierProbe probe;
  probe.store = &store_;
  {
    shared::ExpertBlockLease lease{};
    shared::ScopedExpertLease guard;
    ASSERT_TRUE(guard.AcquireReady(store_, 0, &lease));
    guard.SetDeviceReadBarrier(&ProbeBarrier, &probe);
    guard.MarkDeviceReadsQueued();
  }
  EXPECT_EQ(probe.calls, 1);
  EXPECT_TRUE(probe.pin_held_when_called);
  EXPECT_TRUE(CanServe(1));
}

TEST_F(ScopedExpertLeaseTest, SkipsBarrierWhenNoDeviceReadWasQueued) {
  BarrierProbe probe;
  probe.store = &store_;
  {
    shared::ExpertBlockLease lease{};
    shared::ScopedExpertLease guard;
    ASSERT_TRUE(guard.AcquireReady(store_, 0, &lease));
    guard.SetDeviceReadBarrier(&ProbeBarrier, &probe);
  }
  EXPECT_EQ(probe.calls, 0);
  EXPECT_TRUE(CanServe(1));
}

TEST_F(ScopedExpertLeaseTest, QuarantinesPinWhenQueuedReadsCannotBeConfirmed) {
  BarrierProbe probe;
  probe.store = &store_;
  probe.succeed = false;
  shared::ExpertBlockLease lease{};
  shared::ScopedExpertLease guard;
  ASSERT_TRUE(guard.AcquireReady(store_, 0, &lease));
  guard.SetDeviceReadBarrier(&ProbeBarrier, &probe);
  guard.MarkDeviceReadsQueued();

  EXPECT_FALSE(guard.Release());
  EXPECT_TRUE(guard.held());
  EXPECT_FALSE(CanServe(1));  // the slot is never recycled under a read that may still be running

  probe.succeed = true;
  EXPECT_TRUE(guard.Release());
  EXPECT_FALSE(guard.held());
  EXPECT_TRUE(CanServe(1));
}

TEST_F(ScopedExpertLeaseTest, ExplicitCompletionReleasesWithoutRunningBarrier) {
  BarrierProbe probe;
  probe.store = &store_;
  shared::ExpertBlockLease lease{};
  shared::ScopedExpertLease guard;
  ASSERT_TRUE(guard.AcquireReady(store_, 0, &lease));
  guard.SetDeviceReadBarrier(&ProbeBarrier, &probe);
  guard.MarkDeviceReadsQueued();
  guard.MarkDeviceReadsComplete();

  EXPECT_TRUE(guard.Release());
  EXPECT_EQ(probe.calls, 0);
  EXPECT_TRUE(guard.Release());  // idempotent
  EXPECT_TRUE(CanServe(1));
}

}  // namespace
}  // namespace onnxruntime::test
