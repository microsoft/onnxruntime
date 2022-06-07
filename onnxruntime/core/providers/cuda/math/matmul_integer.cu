// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_integer.cuh"

#include <cub/cub.cuh>
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

template <int TPB>
__global__ void ReduceRowSumOnMatrixAKernel(const int8_t* matrix, int32_t* row_sum, const int8_t offset, int32_t K) {
  int32_t thread_data = 0;
  const int8_t* row_ptr = matrix + blockIdx.x * K;
  for (int i = threadIdx.x; i < K; i += TPB) {
    thread_data += *(row_ptr + i);
  }

  using BlockReduce = cub::BlockReduce<int32_t, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int32_t sum = BlockReduce(temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) {
    row_sum[blockIdx.x] = offset * sum;
  }
}

Status ReduceRowSumOnMatrixA(cudaStream_t stream, const int8_t* matrix, int32_t* row_sum, const int8_t offset, const MatMulComputeHelper& helper) {
  for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
    ReduceRowSumOnMatrixAKernel<static_cast<int>(GridDim::maxThreadsPerBlock)><<<static_cast<int>(helper.M()), GridDim::maxThreadsPerBlock, 0, stream>>>(matrix + helper.LeftOffsets()[batch],
                                                                                                                                                 row_sum + batch * helper.M(),
                                                                                                                                                 offset,
                                                                                                                                                 static_cast<int>(helper.K()));
  }

  return CUDA_CALL(cudaPeekAtLastError()) ? Status::OK() : Status(common::ONNXRUNTIME, common::FAIL);
}

template <int TPB>
__global__ void ReduceColSumOnMatrixBKernel(const int8_t* matrix, int32_t* col_sum, const int8_t offset, int32_t row, int32_t col) {
  int32_t thread_data = 0;
  const int8_t* col_ptr = matrix + blockIdx.x;
  for (int i = threadIdx.x; i < row; i += TPB) {
    thread_data += *(col_ptr + i * col);
  }

  using BlockReduce = cub::BlockReduce<int32_t, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int32_t sum = BlockReduce(temp_storage).Sum(thread_data);

  if (threadIdx.x == 0) {
    col_sum[blockIdx.x] = offset * sum;
  }
}

Status ReduceColSumOnMatrixB(cudaStream_t stream, const int8_t* matrix, int32_t* col_sum, const int8_t offset, const MatMulComputeHelper& helper) {
  for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
    ReduceColSumOnMatrixBKernel<static_cast<int>(GridDim::maxThreadsPerBlock)><<<static_cast<int>(helper.N()), GridDim::maxThreadsPerBlock, 0, stream>>>(matrix + helper.RightOffsets()[batch],
                                                                                                                                                 col_sum + batch * helper.N(),
                                                                                                                                                 offset,
                                                                                                                                                 static_cast<int32_t>(helper.K()),
                                                                                                                                                 static_cast<int32_t>(helper.N()));
  }

  return CUDA_CALL(cudaPeekAtLastError()) ? Status::OK() : Status(common::ONNXRUNTIME, common::FAIL);
}

__global__ void ComputeOffsetOfMatrixAB(const int32_t* row_sum,
                                        const int32_t* col_sum,
                                        int32_t* output,
                                        int32_t K_A_B,
                                        int32_t N) {
  for (int32_t i = threadIdx.x; i < N; i += blockDim.x) {
    *(output + blockIdx.x * N + i) = K_A_B - row_sum[blockIdx.x] - col_sum[i];
  }
}

__global__ void ComputeOffsetOfMatrixA(const int32_t* col_sum,
                                       int32_t* output,
                                       int32_t N) {
  for (int32_t i = threadIdx.x; i < N; i += blockDim.x) {
    *(output + blockIdx.x * N + i) = -col_sum[i];
  }
}

__global__ void ComputeOffsetOfMatrixB(const int32_t* row_sum,
                                       int32_t* output,
                                       int32_t N) {
  for (int32_t i = threadIdx.x; i < N; i += blockDim.x) {
    *(output + blockIdx.x * N + i) = -row_sum[blockIdx.x];
  }
}

Status OffsetOutput(cudaStream_t stream,
                    const int32_t* row_sum,
                    const int32_t* col_sum,
                    int32_t* output,
                    const int8_t a_offset,
                    const int8_t b_offset,
                    const MatMulComputeHelper& helper) {
  if (a_offset && b_offset) {
    for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
      ComputeOffsetOfMatrixAB<<<static_cast<int>(helper.M()), GridDim::maxThreadsPerBlock, 0, stream>>>(
          row_sum + batch * helper.M(),
          col_sum + batch * helper.N(),
          output + helper.OutputOffsets()[batch],
          static_cast<int32_t>(helper.K()) * a_offset * b_offset,
          static_cast<int32_t>(helper.N()));
    }
  } else if (a_offset) {
    for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
      ComputeOffsetOfMatrixA<<<static_cast<int>(helper.M()), GridDim::maxThreadsPerBlock, 0, stream>>>(
          col_sum + batch * helper.N(),
          output + helper.OutputOffsets()[batch],
          static_cast<int32_t>(helper.N()));
    }
  } else if (b_offset) {
    for (size_t batch = 0; batch < helper.OutputOffsets().size(); batch++) {
      ComputeOffsetOfMatrixB<<<static_cast<int>(helper.M()), GridDim::maxThreadsPerBlock, 0, stream>>>(
          row_sum + batch * helper.M(),
          output + helper.OutputOffsets()[batch],
          static_cast<int32_t>(helper.N()));
    }
  }

  return CUDA_CALL(cudaPeekAtLastError()) ? Status::OK() : Status(common::ONNXRUNTIME, common::FAIL);
}

}  // namespace cuda
}  // namespace onnxruntime
