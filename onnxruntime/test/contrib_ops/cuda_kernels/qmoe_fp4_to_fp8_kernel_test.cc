#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "core/providers/cuda/cuda_common.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace onnxruntime {
namespace test {
namespace {

struct CudaBuffer {
  void* data = nullptr;
  size_t bytes = 0;

  explicit CudaBuffer(size_t size_in_bytes) : bytes(size_in_bytes) {
    CUDA_CALL_THROW(cudaMalloc(&data, bytes));
  }

  ~CudaBuffer() {
    if (data != nullptr) {
      cudaFree(data);
    }
  }

  template <typename T>
  T* As() {
    return reinterpret_cast<T*>(data);
  }

  void CopyFromHost(const void* src) {
    CUDA_CALL_THROW(cudaMemcpy(data, src, bytes, cudaMemcpyHostToDevice));
  }

  void CopyToHost(void* dst) const {
    CUDA_CALL_THROW(cudaMemcpy(dst, data, bytes, cudaMemcpyDeviceToHost));
  }
};

struct ConversionResult {
  float scale;
  int inexact;
};

ConversionResult ConvertSingleBlock(uint8_t default_scale, float global_scale,
                                    uint8_t exceptional_scale = 0) {
  constexpr int num_experts = 1;
  constexpr int n = 128;
  constexpr int k = 128;

  std::vector<uint8_t> packed_weights(static_cast<size_t>(k) * n / 2, 0x77);
  std::vector<uint8_t> block_scales(static_cast<size_t>(n) * k / 32, default_scale);
  if (exceptional_scale != 0) {
    block_scales[0] = exceptional_scale;
  }

  CudaBuffer device_weights(packed_weights.size());
  CudaBuffer device_block_scales(block_scales.size());
  CudaBuffer device_global_scale(sizeof(float));
  CudaBuffer device_output(static_cast<size_t>(n) * k);
  CudaBuffer device_output_scale(sizeof(float));
  CudaBuffer device_inexact(sizeof(int));
  device_weights.CopyFromHost(packed_weights.data());
  device_block_scales.CopyFromHost(block_scales.data());
  device_global_scale.CopyFromHost(&global_scale);
  int inexact = 0;
  device_inexact.CopyFromHost(&inexact);

  cudaStream_t stream = nullptr;
  CUDA_CALL_THROW(cudaStreamCreate(&stream));
  onnxruntime::contrib::cuda::LaunchQMoEQuantizeFp4WeightsToFp8(
      device_weights.As<uint8_t>(), device_block_scales.As<uint8_t>(), device_global_scale.As<float>(),
      device_output.As<uint8_t>(), device_output_scale.As<float>(), device_inexact.As<int>(),
      num_experts, n, k, stream);
  CUDA_CALL_THROW(cudaStreamSynchronize(stream));
  CUDA_CALL_THROW(cudaStreamDestroy(stream));

  ConversionResult result{};
  device_output_scale.CopyToHost(&result.scale);
  device_inexact.CopyToHost(&result.inexact);
  return result;
}

TEST(CUDA_EP_Unittest, QMoEFp4ToFp8PreservesExactPowerOfTwoScale) {
  const ConversionResult result = ConvertSingleBlock(127, 448.0f / 6.0f);
  const float expected_scale = 448.0f / (6.0f * 64.0f);
  uint32_t result_scale_bits = 0;
  uint32_t expected_scale_bits = 0;
  std::memcpy(&result_scale_bits, &result.scale, sizeof(result_scale_bits));
  std::memcpy(&expected_scale_bits, &expected_scale, sizeof(expected_scale_bits));
  EXPECT_EQ(result_scale_bits, expected_scale_bits);
  EXPECT_EQ(result.inexact, 0);
}

TEST(CUDA_EP_Unittest, QMoEFp4ToFp8RoundsScaleUp) {
  const ConversionResult result = ConvertSingleBlock(127, 300.0f / 6.0f);
  EXPECT_FLOAT_EQ(result.scale, 300.0f / (6.0f * 64.0f));
  EXPECT_EQ(result.inexact, 0);
}

TEST(CUDA_EP_Unittest, QMoEFp4ToFp8ReportsWideExponentSpread) {
  constexpr float largest_group_scale = 1048576.0f;
  const ConversionResult result = ConvertSingleBlock(127, 448.0f / (6.0f * largest_group_scale), 147);
  EXPECT_FLOAT_EQ(result.scale, 448.0f / (6.0f * 64.0f));
  EXPECT_EQ(result.inexact, 1);
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime