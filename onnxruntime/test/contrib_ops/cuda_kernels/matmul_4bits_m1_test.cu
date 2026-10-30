// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "contrib_ops/cuda/quantization/matmul_4bits_m1_impl.cuh"

namespace onnxruntime {
namespace cuda {
namespace test {
namespace {

template <int MaxUnroll>
std::string CompareDirectScales(int k, int n, int pattern) {
  constexpr int group_size = 32;
  const int groups = (k + group_size - 1) / group_size;
  std::mt19937 random(1234 + pattern);
  std::uniform_int_distribution<int> activation(-8, 8);
  std::uniform_int_distribution<int> scale(1, 8);
  std::uniform_int_distribution<int> packed(0, 255);
  std::vector<half> a(k), scales(n * groups);
  std::vector<uint8_t> weights(n * groups * group_size / 2);
  for (int i = 0; i < k; ++i) {
    const float value = pattern == 0 ? activation(random) / 16.0f
                                     : (i % 2 == 0 ? 1.0f : -1.0f) * (i % 7 + 1) / 8.0f;
    a[i] = __float2half_rn(value);
  }
  for (auto& value : scales) {
    value = __float2half_rn(scale(random) / 32.0f);
  }
  for (auto& value : weights) {
    value = static_cast<uint8_t>(packed(random));
  }

  thrust::device_vector<half> device_a(a), device_scales(scales);
  thrust::device_vector<uint8_t> device_weights(weights);
  thrust::device_vector<half> staged(n), direct(n);
  auto* staged_ptr = thrust::raw_pointer_cast(staged.data());
  auto* direct_ptr = thrust::raw_pointer_cast(direct.data());
  if (const auto error = cudaMemset(staged_ptr, 0xff, n * sizeof(half)); error != cudaSuccess) {
    return cudaGetErrorString(error);
  }
  if (const auto error = cudaMemset(direct_ptr, 0xff, n * sizeof(half)); error != cudaSuccess) {
    return cudaGetErrorString(error);
  }

  const dim3 grid(n / 8);
  const dim3 block(32, 8);
  const auto* a_ptr = thrust::raw_pointer_cast(device_a.data());
  const auto* weights_ptr = thrust::raw_pointer_cast(device_weights.data());
  const auto* scales_ptr = thrust::raw_pointer_cast(device_scales.data());
  contrib::cuda::MatMulFloat4BitsKernelM1<half, group_size, false>
      <<<grid, block, 8 * groups * sizeof(half), nullptr>>>(
          staged_ptr, a_ptr, weights_ptr, scales_ptr, nullptr, 1, n, k, groups);
  if (const auto error = cudaGetLastError(); error != cudaSuccess) {
    return cudaGetErrorString(error);
  }
  // Explicit specialization makes coverage independent of the process-cached experiment mode.
  contrib::cuda::MatMulFloat4BitsKernelM1<half, group_size, false, false, true, MaxUnroll>
      <<<grid, block, 0, nullptr>>>(
          direct_ptr, a_ptr, weights_ptr, scales_ptr, nullptr, 1, n, k, groups);
  if (const auto error = cudaGetLastError(); error != cudaSuccess) {
    return cudaGetErrorString(error);
  }
  if (const auto error = cudaDeviceSynchronize(); error != cudaSuccess) {
    return cudaGetErrorString(error);
  }

  const thrust::host_vector<half> expected(staged), actual(direct);
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(__half2float(expected[i])) || !std::isfinite(__half2float(actual[i]))) {
      return "Non-finite output at index " + std::to_string(i);
    }
  }
  return std::memcmp(expected.data(), actual.data(), n * sizeof(half)) == 0 ? "" : "Bitwise output mismatch";
}

}  // namespace

std::string CompareMatMul4BitsM1DirectScales(int max_unroll, int k, int n, int pattern) {
  if (max_unroll == 4) {
    return CompareDirectScales<4>(k, n, pattern);
  }
  if (max_unroll == 16) {
    return CompareDirectScales<16>(k, n, pattern);
  }
  return "Unsupported unroll factor";
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
