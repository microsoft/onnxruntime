// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA) && !defined(USE_CUDA_MINIMAL)

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <vector>

#include "contrib_ops/cuda/quantization/matmul_packed.h"

namespace onnxruntime::test {
namespace {

class PackedProjectionBuffer {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PackedProjectionBuffer);

  explicit PackedProjectionBuffer(size_t bytes) {
    CUDA_CALL_THROW(cudaMalloc(&data_, bytes));
  }

  ~PackedProjectionBuffer() { ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaFree(data_))); }

  void* Data() const { return data_; }

 private:
  void* data_{nullptr};
};

TEST(CUDA_EP_Unittest, MatMulPackedNvfp4WideContiguous128KCancellation) {
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  cudaDeviceProp properties{};
  ASSERT_EQ(cudaGetDeviceProperties(&properties, device), cudaSuccess);
  if (properties.major < 8) GTEST_SKIP() << "Requires CUDA SM80+";
  bool supported = false;
  const auto status = contrib::cuda::GetMatMulPackedBf16KernelSupport(device, supported);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  if (!supported) GTEST_SKIP() << "Requires kernels compiled with BF16 MMA support";
  constexpr int k = 256, row_bytes = (k / 64) * 36;
  // N=8192 selects the staged kernel on SM90; N=35 uses the non-staged wide kernel.
  for (int n : {35, 8192}) {
    std::vector<uint8_t> packed(n * row_bytes, 0x22);
    for (int col = 0; col < n; ++col) {
      for (int block = 0; block < k / 64; ++block) {
        for (int group = 0; group < 4; ++group) packed[col * row_bytes + block * 36 + group] = 0x38;
      }
    }
    PackedProjectionBuffer b(packed.size());
    ASSERT_EQ(cudaMemcpy(b.Data(), packed.data(), packed.size(), cudaMemcpyHostToDevice), cudaSuccess);
    for (int rows : {1, 8}) {
      SCOPED_TRACE(testing::Message() << "n=" << n << " rows=" << rows);
      std::vector<__nv_bfloat16> a(rows * k, __float2bfloat16(0.f)), output(rows * n);
      for (int row = 0; row < rows; ++row) {
        // The small terms share one 128-K warp partial, and the large terms
        // cancel in another. Four 64-K partials instead reduce to zero.
        a[row * k] = __float2bfloat16(1.f);
        a[row * k + 64] = __float2bfloat16(1.f);
        a[row * k + 128] = __float2bfloat16(0x1p25f);
        a[row * k + 192] = __float2bfloat16(-0x1p25f);
      }
      PackedProjectionBuffer input(a.size() * sizeof(__nv_bfloat16));
      PackedProjectionBuffer y(output.size() * sizeof(__nv_bfloat16));
      ASSERT_EQ(cudaMemcpy(input.Data(), a.data(), a.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), cudaSuccess);
      contrib::cuda::MatMulPackedParams params;
      params.b_data = static_cast<const uint8_t*>(b.Data());
      params.k = k;
      params.n = n;
      params.row_bytes = row_bytes;
      params.ggml_type = 40;
      params.bf16_tc_supported = supported;
      const auto launch = contrib::cuda::LaunchMatMulPackedCuda(
          nullptr, nullptr, true, input.Data(), params, nullptr, y.Data(), rows);
      ASSERT_TRUE(launch.IsOK()) << launch.ErrorMessage();
      ASSERT_EQ(cudaMemcpy(output.data(), y.Data(), output.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), cudaSuccess);
      for (size_t index = 0; index < output.size(); ++index) {
        ASSERT_EQ(__bfloat162float(output[index]), 2.f) << "index=" << index;
      }
    }
  }
}

}  // namespace
}  // namespace onnxruntime::test

#endif
