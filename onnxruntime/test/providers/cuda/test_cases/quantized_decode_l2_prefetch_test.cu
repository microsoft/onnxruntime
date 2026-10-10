// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "contrib_ops/cuda/quantization/matmul_2bits_m1_impl.cuh"
#include "contrib_ops/cuda/quantization/matmul_4bits_m1_impl.cuh"
#include "contrib_ops/cuda/quantization/matmul_8bits_m1_impl.cuh"
#include "gtest/gtest.h"

namespace onnxruntime::test {
namespace {

template <int Bits, typename T>
void CheckDecodePrefetchParity() {
  using namespace contrib::cuda;
  constexpr int n = 16;
  constexpr int block_size = 32;
  constexpr int elements_per_byte = 8 / Bits;
  constexpr int code = (1 << (Bits - 1)) + 1;
  constexpr uint8_t packed_code = Bits == 2 ? 0xff : Bits == 4 ? 0x99
                                                               : 0x81;

  for (int k : {32, 128, 256, 512, 544, 1024, 4096, 4128}) {
    SCOPED_TRACE(k);
    const int blocks_per_k = k / block_size;
    std::vector<T> activations(k);
    std::vector<T> scales(n * blocks_per_k);
    for (int i = 0; i < k; ++i) {
      activations[i] = static_cast<T>(i % 2 == 0 ? 1.0f : -0.5f);
    }
    for (int col = 0; col < n; ++col) {
      for (int block = 0; block < blocks_per_k; ++block) {
        scales[col * blocks_per_k + block] = static_cast<T>(0.125f * (1 << (col % 4)));
      }
    }
    thrust::device_vector<T> device_a(activations);
    thrust::device_vector<T> device_scales(scales);
    thrust::device_vector<uint8_t> device_b(n * k / elements_per_byte, packed_code);
    thrust::device_vector<T> device_output(n);
    std::vector<T> baseline(n), prefetched(n);
    const dim3 grid(n / 4);
    const dim3 threads(32, 4);
    const size_t shared_bytes = sizeof(T) * 4 * blocks_per_k;

    // Pass the kernel flag directly: the environment switch is cached per process.
    auto launch = [&](bool enabled) {
      T* output = thrust::raw_pointer_cast(device_output.data());
      const T* a = thrust::raw_pointer_cast(device_a.data());
      const T* scale = thrust::raw_pointer_cast(device_scales.data());
      const uint8_t* b = thrust::raw_pointer_cast(device_b.data());
      if constexpr (Bits == 2) {
        MatMulFloat2bKernelM1<T, block_size, false><<<grid, threads, shared_bytes>>>(
            output, a, b, scale, nullptr, n, k, blocks_per_k, enabled);
      } else if constexpr (Bits == 4) {
        MatMulFloat4BitsKernelM1<T, block_size, false><<<grid, threads, shared_bytes>>>(
            output, a, b, scale, nullptr, 1, n, k, blocks_per_k, enabled);
      } else {
        MatMulFloat8bKernelM1<T, block_size, false><<<grid, threads, shared_bytes>>>(
            output, a, b, scale, nullptr, n, k, blocks_per_k, enabled);
      }
      CUDA_CALL_THROW(cudaGetLastError());
      CUDA_CALL_THROW(cudaDeviceSynchronize());
    };
    launch(false);
    thrust::copy(device_output.begin(), device_output.end(), baseline.begin());
    launch(true);
    thrust::copy(device_output.begin(), device_output.end(), prefetched.begin());
    for (int col = 0; col < n; ++col) {
      const float expected = (code - (1 << (Bits - 1))) * (k / 4.0f) *
                             static_cast<float>(scales[col * blocks_per_k]);
      EXPECT_EQ(static_cast<float>(baseline[col]), expected);
      EXPECT_EQ(static_cast<float>(prefetched[col]), static_cast<float>(baseline[col]));
    }
  }
}

TEST(CUDA_EP_Unittest, QuantizedDecodeL2PrefetchParity) {
  int device, major, minor;
  CUDA_CALL_THROW(cudaGetDevice(&device));
  CUDA_CALL_THROW(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  CUDA_CALL_THROW(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  if (!contrib::cuda::ShouldPrefetchQuantizedDecodeL2(true, major, minor)) {
    GTEST_SKIP() << "L2 decode prefetch is enabled only on SM121";
  }
  CheckDecodePrefetchParity<2, float>();
  CheckDecodePrefetchParity<4, float>();
  CheckDecodePrefetchParity<8, float>();
  CheckDecodePrefetchParity<2, half>();
  CheckDecodePrefetchParity<4, half>();
  CheckDecodePrefetchParity<8, half>();
}

}  // namespace
}  // namespace onnxruntime::test
