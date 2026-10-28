// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>

#include "unit_lower_triangular_solve_impl.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

namespace onnxruntime::contrib::cuda {
namespace {

__global__ void ToColumn(const float* input, float* output, size_t count, int rows, int columns) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count;
       i += size_t(blockDim.x) * gridDim.x) {
    const size_t matrix_size = size_t(rows) * columns;
    const size_t matrix = i / matrix_size;
    const size_t local = i % matrix_size;
    output[matrix * matrix_size + (local % columns) * rows + local / columns] = input[i];
  }
}

__global__ void FromColumn(const float* input, float* output, size_t count, int rows, int columns) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < count;
       i += size_t(blockDim.x) * gridDim.x) {
    const size_t matrix_size = size_t(rows) * columns;
    const size_t matrix = i / matrix_size;
    const size_t local = i % matrix_size;
    output[i] = input[matrix * matrix_size + (local % columns) * rows + local / columns];
  }
}

__global__ void Pointers(float* a, float* b, const float** a_pointers, float** b_pointers,
                         int batch, int rows, int columns) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < size_t(batch);
       i += size_t(blockDim.x) * gridDim.x) {
    a_pointers[i] = a + i * size_t(rows) * rows;
    b_pointers[i] = b + i * size_t(rows) * columns;
  }
}

int Blocks(size_t count) {
  return static_cast<int>(std::min(size_t{65535}, (count - 1) / 256 + 1));
}

}  // namespace

Status UnitLowerTriangularSolveToColumn(cudaStream_t stream, const float* input, float* output,
                                        size_t count, int rows, int columns) {
  ToColumn<<<Blocks(count), 256, 0, stream>>>(input, output, count, rows, columns);
  return CUDA_CALL(cudaGetLastError());
}

Status UnitLowerTriangularSolveFromColumn(cudaStream_t stream, const float* input, float* output,
                                          size_t count, int rows, int columns) {
  FromColumn<<<Blocks(count), 256, 0, stream>>>(input, output, count, rows, columns);
  return CUDA_CALL(cudaGetLastError());
}

Status UnitLowerTriangularSolvePointers(cudaStream_t stream, float* a, float* b,
                                        const float** a_pointers, float** b_pointers,
                                        int batch, int rows, int columns) {
  Pointers<<<Blocks(static_cast<size_t>(batch)), 256, 0, stream>>>(
      a, b, a_pointers, b_pointers, batch, rows, columns);
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace onnxruntime::contrib::cuda
