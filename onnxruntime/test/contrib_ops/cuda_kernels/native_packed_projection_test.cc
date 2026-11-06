// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>

#include "contrib_ops/cuda/quantization/matmul_packed_projection.cuh"
#include "core/common/float16.h"
#include "core/common/inlined_containers.h"
#include "gtest/gtest.h"

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS)
namespace onnxruntime::test {
namespace {

using DeviceBuffer = std::unique_ptr<void, decltype(&cudaFree)>;

DeviceBuffer AllocateNative(size_t bytes) {
  void* ptr = nullptr;
  EXPECT_EQ(cudaMalloc(&ptr, bytes), cudaSuccess);
  return DeviceBuffer(ptr, cudaFree);
}

void CompareNativeAndSplitLayouts(bool q8, bool bf16_scales = false) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count == 0) GTEST_SKIP() << "Requires CUDA";
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major < 8) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 3, 7, 8}) {
    if (bf16_scales && rows == 1) continue;  // One-row BF16 source uses integer-dot-then-scale.
    for (int k : {256, 1280, 4096}) {
      for (int n : {19, 8193}) {
        SCOPED_TRACE(testing::Message() << "rows=" << rows << " K=" << k << " N=" << n);
        const int blocks = k / (q8 ? 32 : 256);
        const int record_bytes = q8 ? 32 : 64;
        const int block_bytes = record_bytes + 2;
        InlinedVector<uint16_t> a(rows * k), scales(n * blocks), expected(rows * n), actual(rows * n + 2);
        InlinedVector<uint8_t> native(n * blocks * block_bytes), records(n * blocks * record_bytes);
        for (int i = 0; i < rows * k; ++i) a[i] = BFloat16(float((i * 3) % 17 - 8) / 32.f).val;
        uint32_t random = 17;
        for (int block = 0; block < n * blocks; ++block) {
          const MLFloat16 source_scale(0.0007f * float(1 + block % 7));
          uint16_t scale = source_scale.val;
          if (bf16_scales) {
            const BFloat16 value(source_scale.ToFloat());
            scales[block] = value.val;
            const MLFloat16 converted(value.ToFloat());
            ASSERT_EQ(converted.ToFloat(), value.ToFloat());
            scale = converted.val;
          } else {
            scales[block] = scale;
          }
          native[block * block_bytes] = static_cast<uint8_t>(scale);
          native[block * block_bytes + 1] = static_cast<uint8_t>(scale >> 8);
          for (int byte = 0; byte < record_bytes; ++byte) {
            random = random * 1664525u + 1013904223u;
            const auto value = static_cast<uint8_t>(random >> 24);
            native[block * block_bytes + 2 + byte] = value;
            records[block * record_bytes + byte] = q8 ? value ^ 128u : value;
          }
        }
        auto da = AllocateNative((a.size() + 1) * sizeof(uint16_t));
        auto dn = AllocateNative(native.size() + 4), dr = AllocateNative(records.size());
        auto ds = AllocateNative(scales.size() * sizeof(uint16_t));
        auto dy = AllocateNative(actual.size() * sizeof(uint16_t));
        auto de = AllocateNative(expected.size() * sizeof(uint16_t));
        for (const auto* ptr : {da.get(), dn.get(), dr.get(), ds.get(), dy.get(), de.get()})
          ASSERT_NE(ptr, nullptr);
        ASSERT_EQ(cudaMemcpy(dr.get(), records.data(), records.size(), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(ds.get(), scales.data(), scales.size() * sizeof(uint16_t), cudaMemcpyHostToDevice),
                  cudaSuccess);
        for (int offsets = 0; offsets < 10; ++offsets) {
          SCOPED_TRACE(testing::Message() << "offset_mask=" << offsets);
          auto* input = static_cast<uint16_t*>(da.get()) + (offsets & 1);
          auto* weights = static_cast<uint8_t*>(dn.get()) + (offsets >> 1);
          auto* output = static_cast<uint16_t*>(dy.get()) + 1;
          ASSERT_EQ(cudaMemcpy(input, a.data(), a.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), cudaSuccess);
          ASSERT_EQ(cudaMemcpy(weights, native.data(), native.size(), cudaMemcpyHostToDevice), cudaSuccess);
          ASSERT_EQ(cudaMemset(dy.get(), 0x55, actual.size() * sizeof(uint16_t)), cudaSuccess);
          ASSERT_EQ(contrib::cuda::LaunchPackedProjection(
                        q8 ? contrib::cuda::PackedProjectionFormat::Q8 : contrib::cuda::PackedProjectionFormat::IQ2XXS, input,
                        static_cast<const uint8_t*>(dr.get()), ds.get(), nullptr,
                        !bf16_scales, rows, n, k, de.get(), nullptr),
                    cudaSuccess);
          ASSERT_EQ(contrib::cuda::LaunchPackedProjection(
                        q8 ? contrib::cuda::PackedProjectionFormat::Q8GGUF : contrib::cuda::PackedProjectionFormat::IQ2XXSGGUF, input, weights,
                        nullptr, nullptr, false, rows, n, k, output, nullptr),
                    cudaSuccess);
          ASSERT_EQ(cudaMemcpy(expected.data(), de.get(), expected.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
                    cudaSuccess);
          ASSERT_EQ(cudaMemcpy(actual.data(), dy.get(), actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
                    cudaSuccess);
          EXPECT_EQ(actual.front(), 0x5555);
          EXPECT_EQ(actual.back(), 0x5555);
          for (int i = 0; i < rows * n; ++i) ASSERT_EQ(actual[i + 1], expected[i]) << i;
        }
      }
    }
  }
}

TEST(NativePackedProjectionKernelTest, IQ2NativeAndSplitLayoutsWithOffsets) {
  CompareNativeAndSplitLayouts(false);
}

TEST(NativePackedProjectionKernelTest, Q8NativeAndSplitLayoutsWithOffsets) {
  CompareNativeAndSplitLayouts(true);
}

TEST(NativePackedProjectionKernelTest, Q8LosslessBF16ScaleMultirowLayouts) {
  CompareNativeAndSplitLayouts(true, true);
}

TEST(NativePackedProjectionKernelTest, NVFP4NativeAndSplitLayoutsWithOffsets) {
  int count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&count), cudaSuccess);
  if (count == 0) GTEST_SKIP() << "Requires CUDA";
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, 0), cudaSuccess);
  if (prop.major < 8) GTEST_SKIP() << "Requires CUDA SM80+";
  for (int rows : {1, 3, 7, 8}) {
    for (int k : {4096, 8192}) {
      for (int n : {19, 8193}) {
        SCOPED_TRACE(testing::Message() << "rows=" << rows << " K=" << k << " N=" << n);
        InlinedVector<uint16_t> a(rows * k), expected(rows * n), actual(rows * n + 2);
        InlinedVector<uint8_t> native(n * (k / 64) * 36), packed(n * (k / 2)), scales(n * (k / 16));
        constexpr float global = 0.013379f;
        for (int i = 0; i < rows * k; ++i) a[i] = BFloat16(float((i * 3) % 17 - 8) / 32.f).val;
        uint32_t random = 17;
        for (int col = 0; col < n; ++col) {
          for (int block = 0; block < k / 64; ++block) {
            auto* bytes = native.data() + (col * (k / 64) + block) * 36;
            for (int group = 0; group < 4; ++group) {
              random = random * 1664525u + 1013904223u;
              bytes[group] = scales[col * (k / 16) + block * 4 + group] =
                  static_cast<uint8_t>((random >> 24) % 127);
              uint8_t elements[16];
              for (int j = 0; j < 8; ++j) {
                random = random * 1664525u + 1013904223u;
                const auto value = static_cast<uint8_t>(random >> 24);
                bytes[4 + group * 8 + j] = value;
                elements[j] = value & 15;
                elements[j + 8] = value >> 4;
              }
              for (int j = 0; j < 8; ++j)
                packed[col * (k / 2) + block * 32 + group * 8 + j] =
                    elements[2 * j] | (elements[2 * j + 1] << 4);
            }
          }
        }
        auto da = AllocateNative((a.size() + 1) * sizeof(uint16_t));
        auto dn = AllocateNative(native.size() + 4), dp = AllocateNative(packed.size());
        auto ds = AllocateNative(scales.size()), dg = AllocateNative(sizeof(float));
        auto dy = AllocateNative(actual.size() * sizeof(uint16_t));
        auto de = AllocateNative(expected.size() * sizeof(uint16_t));
        for (const auto* ptr : {da.get(), dn.get(), dp.get(), ds.get(), dg.get(), dy.get(), de.get()})
          ASSERT_NE(ptr, nullptr);
        ASSERT_EQ(cudaMemcpy(dp.get(), packed.data(), packed.size(), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(ds.get(), scales.data(), scales.size(), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(dg.get(), &global, sizeof(float), cudaMemcpyHostToDevice), cudaSuccess);
        for (int offsets = 0; offsets < 10; ++offsets) {
          SCOPED_TRACE(testing::Message() << "offset_mask=" << offsets);
          auto* input = static_cast<uint16_t*>(da.get()) + (offsets & 1);
          auto* weights = static_cast<uint8_t*>(dn.get()) + (offsets >> 1);
          auto* output = static_cast<uint16_t*>(dy.get()) + 1;
          ASSERT_EQ(cudaMemcpy(input, a.data(), a.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), cudaSuccess);
          ASSERT_EQ(cudaMemcpy(weights, native.data(), native.size(), cudaMemcpyHostToDevice), cudaSuccess);
          ASSERT_EQ(cudaMemset(dy.get(), 0x55, actual.size() * sizeof(uint16_t)), cudaSuccess);
          ASSERT_EQ(contrib::cuda::LaunchPackedProjection(
                        contrib::cuda::PackedProjectionFormat::NVFP4, input,
                        static_cast<const uint8_t*>(dp.get()), ds.get(), static_cast<const float*>(dg.get()),
                        false, rows, n, k, de.get(), nullptr),
                    cudaSuccess);
          ASSERT_EQ(contrib::cuda::LaunchPackedProjection(
                        contrib::cuda::PackedProjectionFormat::NVFP4GGUF, input, weights,
                        nullptr, static_cast<const float*>(dg.get()), false, rows, n, k, output, nullptr),
                    cudaSuccess);
          ASSERT_EQ(cudaMemcpy(expected.data(), de.get(), expected.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
                    cudaSuccess);
          ASSERT_EQ(cudaMemcpy(actual.data(), dy.get(), actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
                    cudaSuccess);
          EXPECT_EQ(actual.front(), 0x5555);
          EXPECT_EQ(actual.back(), 0x5555);
          for (int i = 0; i < rows * n; ++i) ASSERT_EQ(actual[i + 1], expected[i]) << i;
        }
      }
    }
  }
}

}  // namespace
}  // namespace onnxruntime::test
#endif
