// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace test {
namespace {

namespace qmoe = contrib::cuda;

struct CudaBuffer {
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudaBuffer);

  explicit CudaBuffer(size_t size) : bytes(size) {
    CUDA_CALL_THROW(cudaMalloc(&data, bytes));
  }

  ~CudaBuffer() noexcept {
    const cudaError_t status = cudaFree(data);
    if (status != cudaSuccess) {
      std::fprintf(stderr, "cudaFree failed during QMoE test cleanup: %s\n", cudaGetErrorString(status));
    }
  }

  template <typename T>
  T* As() { return static_cast<T*>(data); }

  void Upload(const void* source) {
    CUDA_CALL_THROW(cudaMemcpy(data, source, bytes, cudaMemcpyHostToDevice));
  }

  void Download(void* destination) {
    CUDA_CALL_THROW(cudaMemcpy(destination, data, bytes, cudaMemcpyDeviceToHost));
  }

  void* data = nullptr;
  size_t bytes;
};

TEST(CUDA_EP_Unittest, QMoEFp8CompactExpertsPreservesRoutesAndClearsUnusedSlots) {
  constexpr int num_experts = 512;
  constexpr int capacity = 6;
  CudaBuffer indices(capacity * sizeof(int));
  CudaBuffer compact_indices(capacity * sizeof(int));
  CudaBuffer expert_map(num_experts * sizeof(int));
  CudaBuffer slot_map(capacity * sizeof(int));

  for (const auto& routes : {std::array<int, capacity>{511, 7, 511, 7, 2, 7},
                             std::array<int, capacity>{0, 0, 0, 0, 0, 0}}) {
    indices.Upload(routes.data());
    qmoe::LaunchQMoECompactExperts(indices.As<int>(), compact_indices.As<int>(),
                                   expert_map.As<int>(), slot_map.As<int>(),
                                   num_experts, capacity, capacity, nullptr);
    std::array<int, capacity> original{};
    std::array<int, capacity> compact{};
    std::array<int, capacity> slots{};
    std::array<int, num_experts> reverse{};
    indices.Download(original.data());
    compact_indices.Download(compact.data());
    slot_map.Download(slots.data());
    expert_map.Download(reverse.data());
    EXPECT_EQ(original, routes);
    int used = 0;
    for (int expert = 0; expert < num_experts; ++expert) {
      if (reverse[expert] >= 0) {
        EXPECT_EQ(reverse[expert], used);
        EXPECT_EQ(slots[used++], expert);
      }
    }
    EXPECT_EQ(used, routes[0] == 511 ? 3 : 1);
    for (int slot = used; slot < capacity; ++slot) {
      EXPECT_EQ(slots[slot], -1);
    }
    for (int route = 0; route < capacity; ++route) {
      ASSERT_GE(compact[route], 0);
      ASSERT_LT(compact[route], used);
      EXPECT_EQ(slots[compact[route]], routes[route]);
    }
  }
}

template <typename T>
void TestSelectedDequantization(int layout) {
  constexpr int source_experts = 5;
  constexpr int capacity = 3;
  constexpr int n = 6;
  constexpr int k = 5;
  constexpr int block_size = 4;
  const bool split = layout == 1;
  const int output_n = split ? 2 * n : n;
  const std::array<int, capacity> experts{4, 1, -1};
  std::vector<uint8_t> weights(source_experts * n * k, 0x7f);
  std::vector<float> scales(source_experts * 4);
  std::vector<T> bias(source_experts * n);
  for (int expert = 0; expert < source_experts; ++expert) {
    for (int i = 0; i < n * k; ++i) {
      if (expert == 4 || expert == 1) {
        weights[expert * n * k + i] = 0x38;  // E4M3 1.0; unselected experts contain NaNs.
      }
    }
    for (int i = 0; i < 4; ++i) {
      scales[expert * 4 + i] = expert + 1.0f + i * 0.25f;
    }
    for (int row = 0; row < n; ++row) {
      bias[expert * n + row] = T(static_cast<float>(expert * n + row));
    }
  }
  std::vector<T> output(capacity * output_n * k, T(-123.0f));
  std::vector<T> output_bias(capacity * output_n, T(-123.0f));
  CudaBuffer device_weights(weights.size());
  CudaBuffer device_scales(scales.size() * sizeof(float));
  CudaBuffer device_bias(bias.size() * sizeof(T));
  CudaBuffer device_output(output.size() * sizeof(T));
  CudaBuffer device_output_bias(output_bias.size() * sizeof(T));
  CudaBuffer device_experts(experts.size() * sizeof(int));
  device_weights.Upload(weights.data());
  device_scales.Upload(scales.data());
  device_bias.Upload(bias.data());
  device_output.Upload(output.data());
  device_output_bias.Upload(output_bias.data());
  device_experts.Upload(experts.data());

  for (int projection = 0; projection < (split ? 2 : 1); ++projection) {
    qmoe::LaunchQMoEDequantizeFp8Weights(
        device_weights.As<uint8_t>(), device_scales.As<float>(), device_output.As<T>() + projection * k,
        capacity, n, k, nullptr, block_size, split, layout == 2, device_experts.As<int>(),
        device_bias.As<T>(), device_output_bias.As<T>() + projection);
  }
  device_output.Download(output.data());
  device_output_bias.Download(output_bias.data());
  for (int slot = 0; slot < capacity; ++slot) {
    for (int output_row = 0; output_row < output_n; ++output_row) {
      const int row = split ? output_row / 2
                            : (layout == 2 ? output_row / 2 + (output_row % 2) * (n / 2) : output_row);
      const int expert = experts[slot];
      const float expected_bias = expert < 0 ? -123.0f : static_cast<float>(bias[expert * n + row]);
      EXPECT_EQ(static_cast<float>(output_bias[slot * output_n + output_row]), expected_bias);
      for (int col = 0; col < k; ++col) {
        const float expected = expert < 0 ? -123.0f
                                          : scales[expert * 4 + (row / block_size) * 2 + col / block_size];
        EXPECT_EQ(static_cast<float>(output[(slot * output_n + output_row) * k + col]), expected);
      }
    }
  }
}

TEST(CUDA_EP_Unittest, QMoEFp8DequantizesOnlySelectedExpertsAndGathersBias) {
  for (int layout : {0, 1, 2}) {
    SCOPED_TRACE(layout);
    TestSelectedDequantization<half>(layout);
    TestSelectedDequantization<__nv_bfloat16>(layout);
  }
}

template <typename T>
void TestAllFp8Codes() {
  std::array<uint8_t, 256> codes{};
  for (int i = 0; i < 256; ++i) {
    codes[i] = static_cast<uint8_t>(i);
  }
  const float scale = 1.0f;
  CudaBuffer weights(codes.size());
  CudaBuffer scales(sizeof(scale));
  CudaBuffer output(codes.size() * sizeof(T));
  weights.Upload(codes.data());
  scales.Upload(&scale);
  qmoe::LaunchQMoEDequantizeFp8Weights(weights.As<uint8_t>(), scales.As<float>(), output.As<T>(),
                                       1, 1, 256, nullptr);
  std::array<T, 256> actual{};
  output.Download(actual.data());
  for (int code = 0; code < 256; ++code) {
    SCOPED_TRACE(code);
    const float value = static_cast<float>(actual[code]);
    if ((code & 127) == 127) {
      EXPECT_TRUE(std::isnan(value));
      continue;
    }
    const int exponent = (code >> 3) & 15;
    const int mantissa = code & 7;
    const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -9)
                                          : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
    const float expected = code & 128 ? -magnitude : magnitude;
    EXPECT_EQ(value, expected);
    EXPECT_EQ(std::signbit(value), std::signbit(expected));
  }
}

TEST(CUDA_EP_Unittest, QMoEFp8AllCodesPreserveFiniteValuesSignedZerosAndNaNs) {
  TestAllFp8Codes<half>();
  TestAllFp8Codes<__nv_bfloat16>();
}

template <typename T>
void TestNvfp4RowMajorDequantization(int k, bool compact, bool with_bias) {
  constexpr int source_experts = 5;
  constexpr int n = 3;
  const std::array<int, 3> selected{4, 1, -1};
  const int capacity = compact ? static_cast<int>(selected.size()) : source_experts;
  constexpr std::array<float, 8> magnitudes{0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  constexpr std::array<uint8_t, 8> scale_codes{0x00, 0x01, 0x20, 0x38, 0x41, 0x7e, 0x80, 0xb8};
  constexpr std::array<float, 8> scale_values{0.0f, 1.0f / 512.0f, 0.125f, 1.0f, 2.25f, 448.0f, -0.0f, -1.0f};
  std::vector<uint8_t> weights(source_experts * n * k / 2);
  std::vector<uint8_t> scales(source_experts * n * k / 16);
  std::vector<float> globals(source_experts);
  std::vector<T> bias(source_experts * n);
  for (int expert = 0; expert < source_experts; ++expert) {
    globals[expert] = 0.37f * (expert + 1);
    for (int row = 0; row < n; ++row) {
      const int source_row = expert * n + row;
      bias[source_row] = T(static_cast<float>(source_row));
      for (int col = 0; col < k; col += 2) {
        const int low = (col + row + expert) % 16;
        const int high = (low + 1) % 16;
        weights[source_row * (k / 2) + col / 2] = static_cast<uint8_t>(low | (high << 4));
      }
      for (int group = 0; group < k / 16; ++group) {
        scales[source_row * (k / 16) + group] = scale_codes[(group + source_row) % scale_codes.size()];
      }
    }
  }
  std::vector<T> output(capacity * n * k, T(-123.0f));
  std::vector<T> output_bias(capacity * n, T(-123.0f));
  CudaBuffer device_weights(weights.size());
  CudaBuffer device_scales(scales.size());
  CudaBuffer device_globals(globals.size() * sizeof(float));
  CudaBuffer device_bias(bias.size() * sizeof(T));
  CudaBuffer device_output(output.size() * sizeof(T));
  CudaBuffer device_output_bias(output_bias.size() * sizeof(T));
  CudaBuffer device_selected(selected.size() * sizeof(int));
  device_weights.Upload(weights.data());
  device_scales.Upload(scales.data());
  device_globals.Upload(globals.data());
  device_bias.Upload(bias.data());
  device_output.Upload(output.data());
  device_output_bias.Upload(output_bias.data());
  device_selected.Upload(selected.data());
  qmoe::LaunchQMoEDequantizeNvfp4Weights(
      device_weights.As<uint8_t>(), device_scales.As<uint8_t>(), device_globals.As<float>(),
      device_output.As<T>(), capacity, n, k, nullptr,
      compact ? device_selected.As<int>() : nullptr,
      with_bias ? device_bias.As<T>() : nullptr,
      with_bias ? device_output_bias.As<T>() : nullptr, true);
  device_output.Download(output.data());
  device_output_bias.Download(output_bias.data());
  for (int slot = 0; slot < capacity; ++slot) {
    const int expert = compact ? selected[slot] : slot;
    for (int row = 0; row < n; ++row) {
      const T expected_bias = with_bias && expert >= 0 ? bias[expert * n + row] : T(-123.0f);
      EXPECT_EQ(std::memcmp(&output_bias[slot * n + row], &expected_bias, sizeof(T)), 0);
      for (int col = 0; col < k; ++col) {
        T expected(-123.0f);
        if (expert >= 0) {
          const int code = (col + row + expert) % 16;
          const float value = code < 8 ? magnitudes[code] : -magnitudes[code - 8];
          expected = T(value * scale_values[(col / 16 + expert * n + row) % scale_values.size()] * globals[expert]);
        }
        ASSERT_EQ(std::memcmp(&output[(slot * n + row) * k + col], &expected, sizeof(T)), 0)
            << "slot=" << slot << " row=" << row << " col=" << col;
      }
    }
  }
}

TEST(CUDA_EP_Unittest, QMoENvfp4RowMajorVectorDequantizationPreservesValuesAndBias) {
  for (int k : {16, 48, 1280, 2560}) {
    for (bool compact : {false, true}) {
      for (bool with_bias : {false, true}) {
        SCOPED_TRACE(::testing::Message() << "k=" << k << " compact=" << compact << " bias=" << with_bias);
        TestNvfp4RowMajorDequantization<half>(k, compact, with_bias);
        TestNvfp4RowMajorDequantization<__nv_bfloat16>(k, compact, with_bias);
      }
    }
  }
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
