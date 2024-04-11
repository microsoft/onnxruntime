// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_attention_impl.h"
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_common.cuh"

#include <cub/cub.cuh>

namespace onnxruntime {
namespace contrib {
namespace cuda {

__global__ void
BuildTableForSoftmaxPowerOfKernel(const double base, float* table) {
  int g = threadIdx.x - 255;
  table[255 + g] = __double2float_rn(pow(base, static_cast<double>(g)));
}

Status BuildTableForSoftmaxPowerOf(cudaStream_t stream, const double base, float* table) {
  BuildTableForSoftmaxPowerOfKernel<<<1, 256, 0, stream>>>(base, table);
  return CUDA_CALL(cudaGetLastError());
}

template <int TPB>
__global__ void
QOrderMaskedSoftmaxKernel(const int8_t* src, const float* lookup_table, const int32_t* mask_index,
                          int8_t* dst, const float scale_dst, const unsigned sequence_len) {
  using BlockReduceInt32 = cub::BlockReduce<int32_t, TPB>;
  using BlockReduceFP32 = cub::BlockReduce<float, TPB>;

  __shared__ union {
    typename BlockReduceInt32::TempStorage i32;
    typename BlockReduceFP32::TempStorage f32;
  } unioned_tmp_storage;
  __shared__ float sum_reverse_block;
  __shared__ int32_t max_in_block;

  const int block_offset = (blockIdx.y * gridDim.x + blockIdx.x) * sequence_len; /* 4 bytes per thread */
  src += block_offset;
  dst += block_offset;
  mask_index += (blockIdx.y * sequence_len);
  int offset = threadIdx.x * 4;

  char4 ch4 = make_char4(-128, -128, -128, -128);
  int4 four_masks = make_int4(0, 0, 0, 0);
  if (offset < sequence_len) {
    four_masks = *(const int4*)(mask_index + offset);
    ch4 = *(const char4*)(src + offset);
  }
  int32_t max_of_4 = max(max(static_cast<int>(ch4.x), static_cast<int>(ch4.y)),
                         max(static_cast<int>(ch4.z), static_cast<int>(ch4.w)));
  const int32_t max_all = BlockReduceInt32(unioned_tmp_storage.i32).Reduce(max_of_4, cub::Max());
  if (threadIdx.x == 0) {
    max_in_block = max_all;
  }
  __syncthreads();

  float4 epow_of_4 = {
      four_masks.x ? lookup_table[255 - max_in_block + ch4.x] : 0.0f,
      four_masks.y ? lookup_table[255 - max_in_block + ch4.y] : 0.0f,
      four_masks.z ? lookup_table[255 - max_in_block + ch4.z] : 0.0f,
      four_masks.w ? lookup_table[255 - max_in_block + ch4.w] : 0.0f};
  float sum_of_4 = epow_of_4.x + epow_of_4.y + epow_of_4.z + epow_of_4.w;
  const float sum_all = BlockReduceFP32(unioned_tmp_storage.f32).Reduce(sum_of_4, cub::Sum());
  if (threadIdx.x == 0) {
    sum_reverse_block = (float)(1.0 / ((double)sum_all * scale_dst));
  }
  __syncthreads();

  if (offset < sequence_len) {
    ch4.x = QuantizeFloatS8(epow_of_4.x, sum_reverse_block);
    ch4.y = QuantizeFloatS8(epow_of_4.y, sum_reverse_block);
    ch4.z = QuantizeFloatS8(epow_of_4.z, sum_reverse_block);
    ch4.w = QuantizeFloatS8(epow_of_4.w, sum_reverse_block);
    *(char4*)(dst + offset) = ch4;
  }
}

Status QOrderMaskedSoftmax(
    cudaStream_t stream, const cudaDeviceProp& /*device_prop*/,
    const int8_t* src, const float* lookup_table,
    const int32_t* mask_index,
    int8_t* dst, const float scale_dst,
    const unsigned batch, const unsigned num_heads, const unsigned sequence_len) {
  int tpb = (sequence_len + 3) / 4;
  if (tpb <= 32) {
    constexpr int TPB = 32;
    dim3 threads(TPB, 1, 1);
    dim3 blocks(sequence_len * num_heads, batch, 1);
    QOrderMaskedSoftmaxKernel<TPB><<<blocks, threads, 0, stream>>>(src, lookup_table, mask_index, dst, scale_dst, sequence_len);
  } else if (tpb <= 128) {
    constexpr int TPB = 128;
    dim3 threads(TPB, 1, 1);
    dim3 blocks(sequence_len * num_heads, batch, 1);
    QOrderMaskedSoftmaxKernel<TPB><<<blocks, threads, 0, stream>>>(src, lookup_table, mask_index, dst, scale_dst, sequence_len);
  } else if (tpb <= 256) {
    constexpr int TPB = 256;
    dim3 threads(TPB, 1, 1);
    dim3 blocks(sequence_len * num_heads, batch, 1);
    QOrderMaskedSoftmaxKernel<TPB><<<blocks, threads, 0, stream>>>(src, lookup_table, mask_index, dst, scale_dst, sequence_len);
  } else if (tpb <= 512) {
    constexpr int TPB = 512;
    dim3 threads(TPB, 1, 1);
    dim3 blocks(sequence_len * num_heads, batch, 1);
    QOrderMaskedSoftmaxKernel<TPB><<<blocks, threads, 0, stream>>>(src, lookup_table, mask_index, dst, scale_dst, sequence_len);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Sequence length too long (> 2048) currently not supported!");
  }
  return CUDA_CALL(cudaGetLastError());
}

constexpr int S8TransposeWidth = 16;

__global__ void
QOrderBatchInt8MatrixTransposeKernel(const int8_t* src, const int8_t* dst, const int rows, const int cols) {
  __shared__ char4 shm[S8TransposeWidth * 4][S8TransposeWidth + 1];
  const int64_t batch_offset = int64_t(rows) * cols * blockIdx.z;
  src += batch_offset;
  dst += batch_offset;

  const int src_col = blockIdx.x * (S8TransposeWidth << 2);
  const int src_row = blockIdx.y * (S8TransposeWidth << 2);
  const int c = threadIdx.x << 2;
  const int r = threadIdx.y << 2;

  int col = src_col + c;
  int row = src_row + r;
  if (row < rows && col < cols) {
    src += row * cols + col;
    char4 ch4_0 = *(const char4*)(src);
    char4 ch4_1 = *(const char4*)(src += cols);
    char4 ch4_2 = *(const char4*)(src += cols);
    char4 ch4_3 = *(const char4*)(src += cols);

    shm[c + 0][threadIdx.y] = {ch4_0.x, ch4_1.x, ch4_2.x, ch4_3.x};
    shm[c + 1][threadIdx.y] = {ch4_0.y, ch4_1.y, ch4_2.y, ch4_3.y};
    shm[c + 2][threadIdx.y] = {ch4_0.z, ch4_1.z, ch4_2.z, ch4_3.z};
    shm[c + 3][threadIdx.y] = {ch4_0.w, ch4_1.w, ch4_2.w, ch4_3.w};
  }
  __syncthreads();

  int tcol = src_row + c;
  int trow = src_col + r;
  if (trow < cols && tcol < rows) {
    dst += trow * rows + tcol;
    *(char4*)(dst) = shm[r + 0][threadIdx.x];
    *(char4*)(dst += rows) = shm[r + 1][threadIdx.x];
    *(char4*)(dst += rows) = shm[r + 2][threadIdx.x];
    *(char4*)(dst += rows) = shm[r + 3][threadIdx.x];
  }
}

Status QOrderBatchTransposeInt8Matrix(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/,
                                      const int batch_size, const int rows, const int cols,
                                      const int8_t* input, int8_t* output) {
  ORT_ENFORCE(rows % 4 == 0 && cols % 4 == 0, "Matrix rows and cols must be divisible by 4!");
  ORT_ENFORCE(rows > 0 && cols > 0 && batch_size > 0, "batch_size, rows, cols should be positive");
  dim3 block(S8TransposeWidth, S8TransposeWidth);
  dim3 grid((cols / 4 + S8TransposeWidth - 1) / S8TransposeWidth, (rows / 4 + S8TransposeWidth - 1) / S8TransposeWidth, batch_size);
  QOrderBatchInt8MatrixTransposeKernel<<<grid, block, 0, stream>>>(input, output, rows, cols);
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
