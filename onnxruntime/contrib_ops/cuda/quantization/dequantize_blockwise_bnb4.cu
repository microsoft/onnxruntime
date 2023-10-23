// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cpu/quantization/blockwise_quant_block_bnb4.h"
#include "dequantize_blockwise_bnb4.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <class T>
Status SetBnbQuantMap(int quant_type, T* quant_map_buffer, cudaStream_t stream) {
  ORT_ENFORCE(
      quant_type == FP4 || quant_type == NF4,
      "Invalid quant_type, only 0 (FP4) and 1 (NF4) are supported.");

  T host_quant_map[16];
  switch (quant_type) {
    case FP4:
      for (int i = 0; i < 16; i++) host_quant_map[i] = static_cast<T>(fp4_qaunt_map[i]);
      break;
    case NF4:
      for (int i = 0; i < 16; i++) host_quant_map[i] = static_cast<T>(nf4_qaunt_map[i]);
      break;
  }
  CUDA_CALL_THROW(cudaMemcpyAsync(quant_map_buffer, host_quant_map, sizeof(T) * 16, cudaMemcpyHostToDevice, stream));

  return Status::OK();
}

template Status SetBnbQuantMap<float>(int quant_type, float* quant_map_buffer, cudaStream_t stream);

template Status SetBnbQuantMap<half>(int quant_type, half* quant_map_buffer, cudaStream_t stream);

template <typename T, int TILE_SIZE, int THREADS, int NUM_PER_TH>
__global__ void kDequantizeBlockwise(
    const T* quant_map,
    T* output,
    const uint8_t* quant_data,
    const T* absmax,
    const int block_size,
    const int n) {
  const int n_load = (gridDim.x * TILE_SIZE);
  int valid_items_load = 0;
  int valid_items_store = 0;
  const int base_idx = (blockIdx.x * TILE_SIZE);

  T vals[NUM_PER_TH * 2];
  uint8_t qvals[NUM_PER_TH];
  T local_abs_max = T(0.0f);

  typedef cub::BlockLoad<uint8_t, THREADS, NUM_PER_TH, cub::BLOCK_LOAD_WARP_TRANSPOSE> LoadChar;
  typedef cub::BlockStore<T, THREADS, NUM_PER_TH * 2, cub::BLOCK_STORE_WARP_TRANSPOSE> StoreT;

  __shared__ typename LoadChar::TempStorage loadchar;
  __shared__ typename StoreT::TempStorage storet;

  for (unsigned int i = base_idx; i < n_load; i += gridDim.x * TILE_SIZE) {
    valid_items_load = (n + 1) / 2 - i > TILE_SIZE ? TILE_SIZE : (n + 1) / 2 - i;
    valid_items_store = n - i * 2 > TILE_SIZE * 2 ? TILE_SIZE * 2 : n - i * 2;

    local_abs_max = __ldg(&absmax[(i + threadIdx.x * NUM_PER_TH) / (block_size)]);

    __syncthreads();
    LoadChar(loadchar).Load(&(quant_data[i]), qvals, valid_items_load, 128);

    #pragma unroll NUM_PER_TH
    for (int j = 0; j < NUM_PER_TH; j++) {
      vals[j * 2] = quant_map[qvals[j] >> 4] * local_abs_max;
      vals[j * 2 + 1] = quant_map[qvals[j] & 0x0F] * local_abs_max;
    }

    __syncthreads();
    StoreT(storet).Store(&(output[i * 2]), vals, valid_items_store);
  }
}

template <class T>
Status DequantizeBnb4(
    const T* quant_map,
    T* output,
    const uint8_t* quant_data,
    const T* absmax,
    int block_size,
    int numel,
    cudaStream_t stream) {
  int tile_size = 1024;
  kDequantizeBlockwise<T, 512, 64, 8><<<(numel + tile_size - 1) / tile_size, 64, 0, stream>>>(
      quant_map, output, quant_data, absmax, block_size / 2, numel);

  return Status::OK();
}

template Status DequantizeBnb4<float>(
    const float* quant_map,
    float* output,
    const uint8_t* quant_data,
    const float* absmax,
    int block_size,
    int numel,
    cudaStream_t stream);

template Status DequantizeBnb4<half>(
    const half* quant_map,
    half* output,
    const uint8_t* quant_data,
    const half *absmax,
    int block_size,
    int numel,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
