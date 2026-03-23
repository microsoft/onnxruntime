// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// GPU transpose kernel for the Scan control flow helper.
// Handles arbitrary N-D permutations by computing output coordinates
// from linear indices.

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

#include "core/common/status.h"

namespace onnxruntime {
namespace cuda {
namespace plugin {

namespace {

Status CudaStatus(cudaError_t cuda_status, const char* operation) {
  if (cuda_status == cudaSuccess) {
    return Status::OK();
  }

  return common::Status(common::ONNXRUNTIME, common::FAIL,
                        std::string("Scan Transpose: ") + operation + " failed: " + cudaGetErrorString(cuda_status));
}

struct DeviceArraySet {
  int64_t* input_strides = nullptr;
  int64_t* output_strides = nullptr;
  int* perm = nullptr;

  ~DeviceArraySet() {
    if (perm != nullptr) {
      cudaFree(perm);
    }
    if (output_strides != nullptr) {
      cudaFree(output_strides);
    }
    if (input_strides != nullptr) {
      cudaFree(input_strides);
    }
  }
};

}  // namespace

// Maximum number of dimensions supported by the transpose kernel.
// Most real-world tensors have <= 8 dimensions.
constexpr int kMaxTransposeDims = 8;

// Kernel: each thread handles one element, computing its output position
// from the input position via the permutation.
__global__ void TransposeNDKernel(const char* __restrict__ input,
                                  char* __restrict__ output,
                                  const int64_t* __restrict__ input_strides,
                                  const int64_t* __restrict__ output_strides,
                                  const int* __restrict__ perm,
                                  int num_dims,
                                  size_t element_size,
                                  size_t total_elements) {
  size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total_elements) return;

  // Decompose linear index into input coordinates
  int64_t coords[kMaxTransposeDims];
  size_t remaining = idx;
  for (int d = 0; d < num_dims; d++) {
    coords[d] = static_cast<int64_t>(remaining / static_cast<size_t>(input_strides[d]));
    remaining %= static_cast<size_t>(input_strides[d]);
  }

  // Compute output linear index via permutation
  size_t out_idx = 0;
  for (int d = 0; d < num_dims; d++) {
    out_idx += static_cast<size_t>(coords[perm[d]]) * static_cast<size_t>(output_strides[d]);
  }

  // Copy element bytes
  const char* src = input + idx * element_size;
  char* dst = output + out_idx * element_size;
  // Use memcpy for arbitrary element sizes (compiler optimizes for common sizes)
  memcpy(dst, src, element_size);
}

Status LaunchTransposeKernel(const void* input, void* output,
                             const int64_t* input_shape, const size_t* permutation,
                             size_t num_dims, size_t element_size, size_t total_elements,
                             cudaStream_t stream) {
  if (total_elements == 0 || num_dims == 0) {
    return Status::OK();
  }

  if (num_dims > static_cast<size_t>(kMaxTransposeDims)) {
    return common::Status(common::ONNXRUNTIME, common::FAIL,
                          "Scan Transpose: rank " + std::to_string(num_dims) +
                              " exceeds the supported maximum rank of " + std::to_string(kMaxTransposeDims));
  }

  // Compute input strides (row-major)
  int64_t input_strides[kMaxTransposeDims];
  input_strides[num_dims - 1] = 1;
  for (int d = static_cast<int>(num_dims) - 2; d >= 0; d--) {
    input_strides[d] = input_strides[d + 1] * input_shape[d + 1];
  }

  // Compute output shape and strides from permutation
  int64_t output_shape[kMaxTransposeDims];
  int64_t output_strides[kMaxTransposeDims];
  int perm_int[kMaxTransposeDims];
  for (size_t d = 0; d < num_dims; d++) {
    output_shape[d] = input_shape[permutation[d]];
    perm_int[d] = static_cast<int>(permutation[d]);
  }
  output_strides[num_dims - 1] = 1;
  for (int d = static_cast<int>(num_dims) - 2; d >= 0; d--) {
    output_strides[d] = output_strides[d + 1] * output_shape[d + 1];
  }

  // Copy arrays to device
  DeviceArraySet device_arrays;
  auto status = CudaStatus(cudaMalloc(&device_arrays.input_strides, num_dims * sizeof(int64_t)), "cudaMalloc(input_strides)");
  if (!status.IsOK()) {
    return status;
  }
  status = CudaStatus(cudaMalloc(&device_arrays.output_strides, num_dims * sizeof(int64_t)), "cudaMalloc(output_strides)");
  if (!status.IsOK()) {
    return status;
  }
  status = CudaStatus(cudaMalloc(&device_arrays.perm, num_dims * sizeof(int)), "cudaMalloc(perm)");
  if (!status.IsOK()) {
    return status;
  }

  status = CudaStatus(cudaMemcpyAsync(device_arrays.input_strides, input_strides, num_dims * sizeof(int64_t), cudaMemcpyHostToDevice, stream),
                      "cudaMemcpyAsync(input_strides)");
  if (!status.IsOK()) {
    return status;
  }
  status = CudaStatus(cudaMemcpyAsync(device_arrays.output_strides, output_strides, num_dims * sizeof(int64_t), cudaMemcpyHostToDevice, stream),
                      "cudaMemcpyAsync(output_strides)");
  if (!status.IsOK()) {
    return status;
  }
  status = CudaStatus(cudaMemcpyAsync(device_arrays.perm, perm_int, num_dims * sizeof(int), cudaMemcpyHostToDevice, stream),
                      "cudaMemcpyAsync(perm)");
  if (!status.IsOK()) {
    return status;
  }

  constexpr int kBlockSize = 256;
  int num_blocks = static_cast<int>((total_elements + kBlockSize - 1) / kBlockSize);

  TransposeNDKernel<<<num_blocks, kBlockSize, 0, stream>>>(
      static_cast<const char*>(input),
      static_cast<char*>(output),
      device_arrays.input_strides,
      device_arrays.output_strides,
      device_arrays.perm,
      static_cast<int>(num_dims),
      element_size,
      total_elements);

  status = CudaStatus(cudaGetLastError(), "TransposeNDKernel launch");
  if (!status.IsOK()) {
    return status;
  }
  return Status::OK();
}

}  // namespace plugin
}  // namespace cuda
}  // namespace onnxruntime
