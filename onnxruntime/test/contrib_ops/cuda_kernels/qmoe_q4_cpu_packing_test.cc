// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdio>
#include <vector>

#include <gtest/gtest.h>

#include "contrib_ops/cuda/llm/fpA_intB_gemm_preprocessors.h"
#include "contrib_ops/cuda/moe/moe_kernels.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::test {
namespace {

struct Q4DeviceBuffer {
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Q4DeviceBuffer);

  explicit Q4DeviceBuffer(size_t bytes) {
    CUDA_CALL_THROW(cudaMalloc(&data, bytes));
  }

  ~Q4DeviceBuffer() {
    const cudaError_t status = cudaFree(data);
    if (status != cudaSuccess) {
      std::fprintf(stderr, "cudaFree failed during Q4 packing test cleanup: %s\n", cudaGetErrorString(status));
    }
  }

  void* data{};
};

TEST(CUDA_EP_Unittest, QMoEQ4CpuPackingMatchesMlas) {
#if defined(MLAS_TARGET_AMD64_IX86)
  namespace weight_only = onnxruntime::llm::kernels::weight_only;
  constexpr size_t k = 512;
  for (size_t block : {16, 32, 64, 128, 256}) {
    if (!MlasIsQNBitGemmAvailable(4, block, SQNBIT_CompFp32)) {
      GTEST_SKIP() << "x86 MLAS Q4 FP32 kernel is unavailable.";
    }
    for (size_t n : {8, 24, 64}) {
      SCOPED_TRACE(MakeString("block=", block, ", N=", n));
      const size_t bytes = n * k / 2;
      std::vector<uint8_t> canonical(bytes);
      std::vector<int8_t> row_major(bytes, 0);
      uint32_t state = 1234567;
      for (size_t column = 0; column < n; ++column) {
        for (size_t row = 0; row < k; ++row) {
          state = state * 1664525 + 1013904223;
          const uint8_t code = (state >> 24) & 15;
          canonical[column * (k / 2) + row / 2] |= static_cast<uint8_t>(code << ((row % 2) * 4));
          const size_t offset = row * n + column;
          auto* raw = reinterpret_cast<uint8_t*>(row_major.data());
          raw[offset / 2] |= static_cast<uint8_t>(((code + 8) & 15) << ((offset % 2) * 4));
        }
      }
      const size_t packed_size = MlasQNBitGemmPackQuantBDataSize(
          n, k, 4, block, false, SQNBIT_CompFp32, nullptr);
      ASSERT_EQ(packed_size, bytes);
      std::vector<std::byte> reference(bytes);
      MlasQNBitGemmPackQuantBData(n, k, 4, block, SQNBIT_CompFp32, canonical.data(),
                                  reference.data(), nullptr, false, nullptr, nullptr, nullptr);

      Q4DeviceBuffer input(bytes);
      Q4DeviceBuffer cutlass(bytes);
      Q4DeviceBuffer packed(bytes);
      Q4DeviceBuffer permutation(32 * sizeof(int));
      CUDA_CALL_THROW(cudaMemcpy(input.data, row_major.data(), bytes, cudaMemcpyHostToDevice));
      weight_only::preprocess_weights_for_mixed_gemm_cuda(
          nullptr, 80, static_cast<int8_t*>(cutlass.data), static_cast<int8_t*>(input.data),
          static_cast<int32_t*>(permutation.data), {k, n}, weight_only::QuantType::W4_A16, true);
      const size_t sub_block = block == 16 ? 16 : (block == 32 ? 32 : 64);
      std::vector<std::byte> actual(bytes);
      for (bool cutlass_layout : {false, true}) {
        SCOPED_TRACE(cutlass_layout ? "CUTLASS input" : "canonical input");
        if (!cutlass_layout) {
          CUDA_CALL_THROW(cudaMemcpy(input.data, canonical.data(), bytes, cudaMemcpyHostToDevice));
        }
        contrib::cuda::LaunchPackMlasQ4Weights(
            static_cast<const uint8_t*>(cutlass_layout ? cutlass.data : input.data),
            static_cast<uint8_t*>(packed.data), n, k, block, sub_block, cutlass_layout, nullptr);
        CUDA_CALL_THROW(cudaMemcpy(actual.data(), packed.data, bytes, cudaMemcpyDeviceToHost));
        EXPECT_EQ(reference, actual);
      }
    }
  }
#else
  GTEST_SKIP() << "GPU CPU-packing implements the x86 MLAS Q4 FP32 layout.";
#endif
}

}  // namespace
}  // namespace onnxruntime::test
