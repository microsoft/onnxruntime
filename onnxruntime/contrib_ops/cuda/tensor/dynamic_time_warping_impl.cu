// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/tensor/dynamic_time_warping_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/common/common.h"
#include <core/common/safeint.h>
#include <cfloat>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

__global__ void DynamicTimeWarpingInitCost(float* cost_buffer, int8_t* trace_buffer, size_t cols_plus_1) {
    int r = blockIdx.x;
    cost_buffer += cols_plus_1 * r;
    for (size_t i = threadIdx.x; i < cols_plus_1; i += blockDim.x) {
        cost_buffer[i] = FLT_MAX;
    }
    if (r == 0) {
      for (size_t i = threadIdx.x; i < cols_plus_1; i += blockDim.x) {
        trace_buffer[i] = 2;
      }
    }
    if (threadIdx.x == 0) trace_buffer[cols_plus_1 * r] = 1;
    if (threadIdx.x == 0 && r == 0) *cost_buffer = 0.0f;
}

__global__ void DynamicTimeWarpingKernel(
    size_t rows,
    size_t cols,
    size_t max_index_len,
    const float* input,
    float* cost_buffer,
    int8_t* trace_buffer,
    int32_t* result_buffer,
    size_t* result_len_device
) {
  const int diag_max = static_cast<int>(rows + cols);
  for (int d = 1; d <= diag_max; d++) {
    for (int c = threadIdx.x + 1; c <= cols; c += blockDim.x) {
        int r = d - c;
        if (r >= 1 && r <= rows) {
            int cost_idx = ((r - 1) * (cols + 1) + (c - 1)); //[r - 1, c - 1]
            const float c0 = cost_buffer[cost_idx];
            const float c1 = cost_buffer[cost_idx + 1]; // [r - 1, c]
            const float c2 = cost_buffer[cost_idx + cols + 1]; // [r, c - 1]

            float cost;
            int8_t t;
            if (c0 < c1 && c0 < c2) {
                cost = c0;
                t = 0;
            } else if (c1 < c0 && c1 < c2) {
                cost = c1;
                t = 1;
            } else {
                cost = c2;
                t = 2;
            }
            cost_idx += ((cols + 1) + 1);
            cost_buffer[cost_idx] = cost + input[(r - 1) * cols + (c - 1)];
            trace_buffer[cost_idx] = t;
        }
    }
    __syncthreads();
  }

  //back tracing, reverse append to result buffer
  if (threadIdx.x == 0) {
    int r = rows - 1;
    int c = cols - 1;
    int pos = static_cast<int>(max_index_len); // reverse put
    while (r >= 0 && c >= 0) {
        --pos;
        result_buffer[pos] = r;
        result_buffer[max_index_len + pos] = c;
        const int trace_index = (r + 1) * (cols + 1) + (c + 1);
        int8_t t = trace_buffer[trace_index];
        switch (t) {
        case 0: r -= 1; c -= 1; break;
        case 1: r -= 1; break;
        default: c -= 1; break;
        }
    }
    *result_len_device = max_index_len - static_cast<size_t>(pos);
  }
}

size_t GetDynamicTimeWarpingBufferSize(size_t batch, size_t rows, size_t cols, size_t& max_index_len) {
  max_index_len = rows + cols + 1;
  size_t cost_buffer_size = ((rows + 1) * (cols + 1));
  return batch * max_index_len * 2 * sizeof(int32_t) + // two index arrays
         sizeof(int64_t) + // final index array length
         batch* cost_buffer_size * sizeof(float) + // cost buffer
         batch* cost_buffer_size * sizeof(int8_t); // trace buffer
}

Status LaunchDynamicTimeWarping(
    cudaStream_t stream,
    const cudaDeviceProp& device_prop,
    size_t batch,
    size_t rows,
    size_t cols,
    const float* input,
    void* buffer,
    size_t& result_len
) {
  ORT_ENFORCE(batch == 1);
  size_t max_index_len = rows + cols + 1;
  int32_t* result_buffer = (int32_t*)buffer;
  size_t* result_len_device_buf = (size_t*)(result_buffer + (batch * max_index_len * 2));
  float* cost_buffer = (float*)(result_len_device_buf + 1);
  int8_t* trace_buffer = (int8_t*)(cost_buffer + ((rows + 1) * (cols + 1)));

  dim3 block(device_prop.maxThreadsPerBlock);
  dim3 grid_init((unsigned)SafeInt<unsigned>(rows + 1), (unsigned)SafeInt<unsigned>(batch));
  DynamicTimeWarpingInitCost<<<grid_init, block, 0, stream>>>(cost_buffer, trace_buffer, cols+1);
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));

  dim3 grid(1, (unsigned)SafeInt<unsigned>(batch));
  DynamicTimeWarpingKernel<<<grid, block, 0, stream>>>(
    rows,
    cols,
    max_index_len,
    input,
    cost_buffer,
    trace_buffer,
    result_buffer,
    result_len_device_buf);
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));

  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaMemcpyAsync(&result_len, result_len_device_buf, sizeof(size_t), cudaMemcpyDeviceToHost, stream)));
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));
  return CUDA_CALL(cudaStreamSynchronize(stream));
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
