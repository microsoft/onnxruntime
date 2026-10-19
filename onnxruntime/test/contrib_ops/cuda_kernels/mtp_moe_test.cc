// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "contrib_ops/cuda/llm/mtp_moe.h"

namespace onnxruntime {
namespace test {
namespace {

namespace mtp_moe = onnxruntime::contrib::cuda::mtp_moe;

constexpr float kGateUpGlobalScale = 0.0005173456156626344f;
constexpr float kDownGlobalScale = 0.00031534829759038985f;

void CheckCuda(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t size) {
    CheckCuda(cudaMalloc(&data_, size));
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) {
      cudaFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  template <typename T>
  T* As() const {
    return reinterpret_cast<T*>(data_);
  }
  void Upload(const void* source, size_t size) {
    CheckCuda(cudaMemcpy(data_, source, size, cudaMemcpyHostToDevice));
  }
  void Download(void* destination, size_t size) const {
    CheckCuda(cudaMemcpy(destination, data_, size, cudaMemcpyDeviceToHost));
  }
  void UploadAt(size_t offset, const void* source, size_t size) {
    CheckCuda(cudaMemcpy(static_cast<uint8_t*>(data_) + offset, source, size, cudaMemcpyHostToDevice));
  }

 private:
  void* data_ = nullptr;
};

float RoundBf16(float value) {
  return __bfloat162float(__float2bfloat16(value));
}

float DecodeFp4(uint32_t code) {
  constexpr std::array<float, 8> kMagnitude = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float magnitude = kMagnitude[code & 7u];
  return (code & 8u) != 0 ? -magnitude : magnitude;
}

float DecodeE4M3Scale(uint8_t encoded) {
  // The test's synthetic block scales use only exactly representable 1.0 and 0.5 codes.
  EXPECT_TRUE(encoded == 0x38 || encoded == 0x30);
  return encoded == 0x38 ? 1.0f : 0.5f;
}

float DecodePackedWeight(const uint8_t* row, int k, int column) {
  const uint8_t packed = row[column >> 1];
  const uint32_t nibble = (column & 1) == 0 ? (packed & 0xfu) : (packed >> 4);
  const int scale_index = k / 2 + column / 16;
  return DecodeFp4(nibble) * DecodeE4M3Scale(row[scale_index]);
}

float WarpReduceXor(std::array<float, 32> values) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    const auto previous = values;
    for (int lane = 0; lane < 32; ++lane) {
      values[lane] = previous[lane] + previous[lane ^ offset];
    }
  }
  return values[0];
}

std::vector<uint8_t> MakeNvfp4Weights(int experts, int rows, int k) {
  const int row_bytes = k * 9 / 16;
  const int packed_bytes = k / 2;
  const int scale_bytes = k / 16;
  std::vector<uint8_t> weights(static_cast<size_t>(experts) * rows * row_bytes);
  for (int expert = 0; expert < experts; ++expert) {
    for (int row = 0; row < rows; ++row) {
      uint8_t* weight_row = weights.data() +
                           (static_cast<size_t>(expert) * rows + row) * row_bytes;
      for (int column = 0; column < k; ++column) {
        const uint8_t code = static_cast<uint8_t>((expert * 13 + row * 7 + column * 3 + column / 17) & 0xf);
        uint8_t& packed = weight_row[column >> 1];
        packed = (column & 1) == 0
                     ? static_cast<uint8_t>((packed & 0xf0u) | code)
                     : static_cast<uint8_t>((packed & 0x0fu) | (code << 4));
      }
      for (int group = 0; group < scale_bytes; ++group) {
        weight_row[packed_bytes + group] = ((expert + row + group) % 3 == 0) ? 0x30 : 0x38;
      }
    }
  }
  return weights;
}

float ReferenceNvfp4Dot(const uint8_t* weight_row, const __nv_bfloat16* input, int k) {
  const int chunks = k / 8;
  std::array<float, 32> lane_sum{};
  for (int lane = 0; lane < 32; ++lane) {
    float accumulator = 0.0f;
    for (int chunk = lane; chunk < chunks; chunk += 32) {
      for (int element = 0; element < 8; ++element) {
        const int column = chunk * 8 + element;
        const float weight = DecodePackedWeight(weight_row, k, column);
        const float activation = __bfloat162float(input[column]);
        accumulator = std::fma(weight, activation, accumulator);
      }
    }
    lane_sum[lane] = accumulator;
  }
  return WarpReduceXor(lane_sum);
}

struct ReferenceResult {
  std::vector<__nv_bfloat16> output;
  std::vector<__nv_bfloat16> hidden;
  std::array<int, mtp_moe::kTopK> experts{};
  std::array<float, mtp_moe::kTopK> routing_weights{};
};

ReferenceResult ReferenceMtpMoe(const std::vector<__nv_bfloat16>& input,
                                const std::vector<__nv_bfloat16>& router,
                                const std::vector<uint8_t>& gate_up,
                                const std::vector<uint8_t>& down) {
  constexpr int hidden = mtp_moe::kHiddenSize;
  constexpr int ffn = mtp_moe::kFfnSize;
  constexpr int gate_up_rows = 2 * ffn;
  constexpr int gate_row_bytes = hidden * 9 / 16;
  constexpr int down_row_bytes = ffn * 9 / 16;

  ReferenceResult result;
  std::array<float, mtp_moe::kExpertCount> probabilities{};
  float maximum = -std::numeric_limits<float>::infinity();
  for (int expert = 0; expert < mtp_moe::kExpertCount; ++expert) {
    probabilities[expert] = __bfloat162float(router[expert]);
    maximum = std::max(maximum, probabilities[expert]);
  }
  float denominator = 0.0f;
  for (float& probability : probabilities) {
    probability = std::exp(probability - maximum);
    denominator += probability;
  }
  for (float& probability : probabilities) {
    probability /= denominator;
  }
  std::array<bool, mtp_moe::kExpertCount> selected{};
  for (int slot = 0; slot < mtp_moe::kTopK; ++slot) {
    int best = -1;
    for (int expert = 0; expert < mtp_moe::kExpertCount; ++expert) {
      if (!selected[expert] &&
          (best < 0 || probabilities[expert] > probabilities[best] ||
           (probabilities[expert] == probabilities[best] && expert < best))) {
        best = expert;
      }
    }
    selected[best] = true;
    result.experts[slot] = best;
  }
  float top_sum = 0.0f;
  for (int slot = 0; slot < mtp_moe::kTopK; ++slot) {
    top_sum += probabilities[result.experts[slot]];
  }
  for (int slot = 0; slot < mtp_moe::kTopK; ++slot) {
    result.routing_weights[slot] = RoundBf16(probabilities[result.experts[slot]] / top_sum);
  }

  result.hidden.resize(mtp_moe::kTopK * ffn);
  for (int slot = 0; slot < mtp_moe::kTopK; ++slot) {
    const int expert = result.experts[slot];
    const size_t expert_offset = static_cast<size_t>(expert) * gate_up_rows * gate_row_bytes;
    for (int output_row = 0; output_row < ffn; ++output_row) {
      const uint8_t* gate_row = gate_up.data() + expert_offset + output_row * gate_row_bytes;
      const uint8_t* up_row = gate_up.data() + expert_offset + (ffn + output_row) * gate_row_bytes;
      const float gate = RoundBf16(ReferenceNvfp4Dot(gate_row, input.data(), hidden) * kGateUpGlobalScale);
      const float up = RoundBf16(ReferenceNvfp4Dot(up_row, input.data(), hidden) * kGateUpGlobalScale);
      const float silu = RoundBf16(gate / (1.0f + std::exp(-gate)));
      result.hidden[slot * ffn + output_row] = __float2bfloat16(silu * up);
    }
  }

  result.output.resize(hidden);
  for (int output_column = 0; output_column < hidden; ++output_column) {
    float total = 0.0f;
    for (int slot = 0; slot < mtp_moe::kTopK; ++slot) {
      const int expert = result.experts[slot];
      const size_t expert_offset = static_cast<size_t>(expert) * hidden * down_row_bytes;
      const uint8_t* weight_row = down.data() + expert_offset + output_column * down_row_bytes;
      const float down_sum = RoundBf16(ReferenceNvfp4Dot(
          weight_row, result.hidden.data() + slot * ffn, ffn) * kDownGlobalScale);
      const float weighted = RoundBf16(down_sum * result.routing_weights[slot]);
      total += weighted;
    }
    result.output[output_column] = __float2bfloat16(total);
  }
  return result;
}

TEST(CUDA_EP_Unittest, MtpMoeRouterSelectsStableTopTenAndBf16Normalizes) {
  constexpr int rows = 1;
  std::vector<__nv_bfloat16> logits(mtp_moe::kExpertCount, __float2bfloat16(-8.0f));
  for (int expert = 0; expert < mtp_moe::kTopK; ++expert) {
    logits[expert] = __float2bfloat16(0.0f);
  }

  DeviceBuffer device_logits(logits.size() * sizeof(__nv_bfloat16));
  DeviceBuffer device_scratch(rows * mtp_moe::kScratchStride * sizeof(float));
  device_logits.Upload(logits.data(), logits.size() * sizeof(__nv_bfloat16));

  cudaStream_t stream = nullptr;
  CheckCuda(cudaStreamCreate(&stream));
  const Status status = mtp_moe::LaunchRouter(device_logits.As<__nv_bfloat16>(),
                                               device_scratch.As<float>(), rows, stream);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  CheckCuda(cudaStreamSynchronize(stream));
  CheckCuda(cudaStreamDestroy(stream));

  std::array<float, mtp_moe::kScratchStride> scratch{};
  device_scratch.Download(scratch.data(), sizeof(scratch));
  float weight_sum = 0.0f;
  for (int slot = 0; slot < mtp_moe::kTopK; ++slot) {
    int expert = -1;
    std::memcpy(&expert, &scratch[slot], sizeof(expert));
    EXPECT_EQ(expert, slot);
    EXPECT_NEAR(scratch[mtp_moe::kTopK + slot],
                __bfloat162float(__float2bfloat16(0.1f)), 0.0f);
    weight_sum += scratch[mtp_moe::kTopK + slot];
  }
  EXPECT_NEAR(weight_sum, 1.0f, 0.004f);
}

TEST(CUDA_EP_Unittest, MtpMoeNvfp4MatchesReferenceArithmeticEndToEnd) {
  constexpr int rows = 1;
  constexpr int experts = mtp_moe::kTopK;
  constexpr int hidden = mtp_moe::kHiddenSize;
  constexpr int ffn = mtp_moe::kFfnSize;
  constexpr int gate_row_bytes = hidden * 9 / 16;
  constexpr int down_row_bytes = ffn * 9 / 16;
  const size_t full_gate_bytes = static_cast<size_t>(mtp_moe::kExpertCount) * 2 * ffn * gate_row_bytes;
  const size_t full_down_bytes = static_cast<size_t>(mtp_moe::kExpertCount) * hidden * down_row_bytes;

  size_t free_bytes = 0;
  size_t total_bytes = 0;
  CheckCuda(cudaMemGetInfo(&free_bytes, &total_bytes));
  constexpr size_t kMemoryReserve = size_t{128} << 20;
  if (free_bytes < full_gate_bytes + full_down_bytes + kMemoryReserve) {
    GTEST_SKIP() << "Insufficient CUDA memory for exact-shape MtpMoe weights";
  }

  std::vector<__nv_bfloat16> input(hidden);
  for (int column = 0; column < hidden; ++column) {
    input[column] = __float2bfloat16(static_cast<float>((column % 23) - 11) * 6.0f);
  }
  std::vector<__nv_bfloat16> router(mtp_moe::kExpertCount, __float2bfloat16(-8.0f));
  for (int expert = 0; expert < experts; ++expert) {
    router[expert] = __float2bfloat16(0.0f);
  }
  const std::vector<uint8_t> gate_up = MakeNvfp4Weights(experts, 2 * ffn, hidden);
  const std::vector<uint8_t> down = MakeNvfp4Weights(experts, hidden, ffn);
  const ReferenceResult expected = ReferenceMtpMoe(input, router, gate_up, down);

  DeviceBuffer device_input(input.size() * sizeof(__nv_bfloat16));
  DeviceBuffer device_router(router.size() * sizeof(__nv_bfloat16));
  DeviceBuffer device_gate_up(full_gate_bytes);
  DeviceBuffer device_down(full_down_bytes);
  DeviceBuffer device_output(hidden * sizeof(__nv_bfloat16));
  DeviceBuffer device_scratch(mtp_moe::kScratchStride * sizeof(float));
  device_input.Upload(input.data(), input.size() * sizeof(__nv_bfloat16));
  device_router.Upload(router.data(), router.size() * sizeof(__nv_bfloat16));
  for (int expert = 0; expert < experts; ++expert) {
    const size_t gate_expert_bytes = static_cast<size_t>(2 * ffn) * gate_row_bytes;
    const size_t down_expert_bytes = static_cast<size_t>(hidden) * down_row_bytes;
    device_gate_up.UploadAt(static_cast<size_t>(expert) * gate_expert_bytes,
                            gate_up.data() + static_cast<size_t>(expert) * gate_expert_bytes,
                            gate_expert_bytes);
    device_down.UploadAt(static_cast<size_t>(expert) * down_expert_bytes,
                         down.data() + static_cast<size_t>(expert) * down_expert_bytes,
                         down_expert_bytes);
  }

  const Status status = mtp_moe::LaunchNvfp4MtpMoe(
      device_input.As<__nv_bfloat16>(), device_router.As<__nv_bfloat16>(),
      device_gate_up.As<uint8_t>(), device_down.As<uint8_t>(),
      kGateUpGlobalScale, kDownGlobalScale, device_scratch.As<float>(),
      device_output.As<__nv_bfloat16>(), rows, nullptr);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  std::vector<__nv_bfloat16> actual_output(hidden);
  std::vector<uint8_t> actual_scratch(mtp_moe::kScratchStride * sizeof(float));
  device_output.Download(actual_output.data(), actual_output.size() * sizeof(__nv_bfloat16));
  device_scratch.Download(actual_scratch.data(), actual_scratch.size());

  for (int slot = 0; slot < mtp_moe::kTopK; ++slot) {
    int actual_expert = -1;
    std::memcpy(&actual_expert, actual_scratch.data() + slot * sizeof(float), sizeof(actual_expert));
    EXPECT_EQ(actual_expert, expected.experts[slot]) << "routing slot " << slot;
    float actual_weight = 0.0f;
    std::memcpy(&actual_weight,
                actual_scratch.data() + (mtp_moe::kTopK + slot) * sizeof(float),
                sizeof(actual_weight));
    EXPECT_EQ(actual_weight, expected.routing_weights[slot]) << "routing weight " << slot;

    const auto* actual_hidden = reinterpret_cast<const __nv_bfloat16*>(
        actual_scratch.data() + 32 * sizeof(float));
    for (int column = 0; column < ffn; ++column) {
      EXPECT_EQ(actual_hidden[slot * ffn + column], expected.hidden[slot * ffn + column])
          << "hidden intermediate slot=" << slot << " column=" << column;
    }
  }
  for (int column = 0; column < hidden; ++column) {
    const float actual = __bfloat162float(actual_output[column]);
    const float reference = __bfloat162float(expected.output[column]);
    EXPECT_NEAR(actual, reference, 0.01f + 0.01f * std::abs(reference)) << "output column " << column;
  }
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
