// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "transpose_impl.h"

namespace onnxruntime {
namespace cuda {

constexpr int kTileDim = 32;
constexpr int kBlockRows = 8;

template <typename T>
__global__ void _TransposeKernel(int32_t shape_rank, const TArray<int64_t> input_strides,
                                 const T* input_data, const TArray<fast_divmod> output_strides, T* output_data, CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;
  CUDA_LONG output_index = id;

  #pragma unroll
  for (auto dim = 0; dim < input_strides.GetCapacity(); ++dim) {
    if (dim >= shape_rank) {
      break;
    }
    int out_coord, r;
    output_strides[dim].divmod(output_index, out_coord, r);
    output_index = r;
    input_index += input_strides[dim] * out_coord;
  }
  output_data[id] = input_data[input_index];
}

Status TransposeImpl(size_t element_size, int32_t shape_rank, const TArray<int64_t>& input_strides,
                     const void* input_data, const TArray<fast_divmod>& fdm_output_strides, void* output_data, int64_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  switch (element_size) {
    case sizeof(int8_t):
      _TransposeKernel<int8_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          shape_rank, input_strides,
          reinterpret_cast<const ToCudaType<int8_t>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<ToCudaType<int8_t>::MappedType*>(output_data),
          N);
      break;
    case sizeof(int16_t):
      _TransposeKernel<int16_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          shape_rank, input_strides,
          reinterpret_cast<const ToCudaType<int16_t>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<ToCudaType<int16_t>::MappedType*>(output_data),
          N);
      break;
    case sizeof(int32_t):
      _TransposeKernel<int32_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          shape_rank, input_strides,
          reinterpret_cast<const ToCudaType<int32_t>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<ToCudaType<int32_t>::MappedType*>(output_data),
          N);
      break;
    case sizeof(int64_t):
      _TransposeKernel<int64_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          shape_rank, input_strides,
          reinterpret_cast<const ToCudaType<int64_t>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<ToCudaType<int64_t>::MappedType*>(output_data),
          N);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for transpose on CUDA. Element size was ",
                             element_size);
  }

  return Status::OK();
}

template <typename T>
__global__ void _BatchTranspose2DKernel(
    int64_t N,
    int64_t H,
    int64_t W,
    int64_t dh,
    int64_t dw,
    const T* X,
    T* Y) {
  __shared__ T tile[kTileDim][kTileDim + 1];
  const int64_t n = blockIdx.x / (dh * dw);
  const int64_t k = blockIdx.x % (dh * dw);
  const int64_t r = k / dw;
  const int64_t c = k % dw;
  const int64_t offset = n * H * W;
  int x = c * kTileDim + threadIdx.x;
  int y = r * kTileDim + threadIdx.y;
  if (x < W) {
    for (int i = 0; i < kTileDim && y + i < H; i += kBlockRows) {
#if __CUDA_ARCH__ >= 350
      tile[threadIdx.y + i][threadIdx.x] = __ldg(X + offset + (y + i) * W + x);
#else
      tile[threadIdx.y + i][threadIdx.x] = X[offset + (y + i) * W + x];
#endif
    }
  }
  __syncthreads();
  x = r * kTileDim + threadIdx.x;
  y = c * kTileDim + threadIdx.y;
  if (x < H) {
    for (int i = 0; i < kTileDim && y + i < W; i += kBlockRows) {
      Y[offset + (y + i) * H + x] = tile[threadIdx.x][threadIdx.y + i];
    }
  }
}


template <typename T>
void BatchTranspose2DImpl(
    int64_t N,
    int64_t H,
    int64_t W,
    const T* X,
    T* Y) {
  int64_t dh = (H + kTileDim - 1) / kTileDim;
  int64_t dw = (W + kTileDim - 1) / kTileDim;
  _BatchTranspose2DKernel<T><<<N * dh * dw, dim3(kTileDim, kBlockRows), 0>>>(N, H, W, dh, dw, X, Y);
}

constexpr int kCubeDim = 16;
constexpr int kCubeBlockRows = 4;

//Perform transpose 3d with perm (0, 2, 1, 3)
template <typename T>
__global__ void _BatchTranspose3DKernel(
    int64_t N,
    int64_t M,
    int64_t L,
    int64_t K,
    int64_t dm,
    int64_t dl,
    int64_t dk,
    const T* X,
    T* Y) {
  __shared__ T tile[kCubeDim][kCubeDim][kCubeDim + 1];
  const int64_t n = blockIdx.x / (dm * dl * dk);
  const int64_t n_left = blockIdx.x % (dm * dl * dk);
  const int64_t m = n_left / (dl * dk);
  const int64_t m_left = n_left % (dl * dk);
  const int64_t l = m_left / dk;
  const int64_t k = m_left % dk;

  const int64_t offset = n * M * L * K;
  int x = k * kCubeDim + threadIdx.x;
  int y = l * kCubeDim + threadIdx.y;
  int z = m * kCubeDim + threadIdx.z;
  if (x < K && y < L) {
    for (int i = 0; i < kCubeDim && z + i < M; i += kCubeBlockRows) {
#if __CUDA_ARCH__ >= 350
      tile[threadIdx.z + i][threadIdx.y][threadIdx.x] = __ldg(X + offset + (z + i) * L * K + y * K + x);
#else
      tile[threadIdx.z + i][threadIdx.y][threadIdx.x] = X[offset + (z + i) * L * K + y * K + x];
#endif
    }
  }
  __syncthreads();

  x = k * kCubeDim + threadIdx.x;
  y = m * kCubeDim + threadIdx.y;
  z = l * kCubeDim + threadIdx.z;
  if (x < K && y < M) {
    for (int i = 0; i < kCubeDim && z + i < L; i += kCubeBlockRows) {
      Y[offset + (z + i) * M * K + y * K + x] = tile[threadIdx.y][threadIdx.z + i][threadIdx.x];
    }
  }
}


//Perform transpose3d with perm(0, 2, 3, 1)
template <typename T>
__global__ void _BatchTranspose3DKernel2(
    int64_t N,
    int64_t M,
    int64_t L,
    int64_t K,
    int64_t dm,
    int64_t dl,
    int64_t dk,
    const T* X,
    T* Y) {
  __shared__ T tile[kCubeDim][kCubeDim][kCubeDim + 1];
  const int64_t n = blockIdx.x / (dm * dl * dk);
  const int64_t n_left = blockIdx.x % (dm * dl * dk);
  const int64_t m = n_left / (dl * dk);
  const int64_t m_left = n_left % (dl * dk);
  const int64_t l = m_left / dk;
  const int64_t k = m_left % dk;

  const int64_t offset = n * M * L * K;
  int x = k * kCubeDim + threadIdx.x;
  int y = l * kCubeDim + threadIdx.y;
  int z = m * kCubeDim + threadIdx.z;
  if (x < K && y < L) {
    for (int i = 0; i < kCubeDim && z + i < M; i += kCubeBlockRows) {
#if __CUDA_ARCH__ >= 350
      tile[threadIdx.z + i][threadIdx.y][threadIdx.x] = __ldg(X + offset + (z + i) * L * K + y * K + x);
#else
      tile[threadIdx.z + i][threadIdx.y][threadIdx.x] = X[offset + (z + i) * L * K + y * K + x];
#endif
    }
  }
  __syncthreads();

  x = m * kCubeDim + threadIdx.x;
  y = k * kCubeDim + threadIdx.y;
  z = l * kCubeDim + threadIdx.z;
  if (x < M && y < K) {
    for (int i = 0; i < kCubeDim && z + i < L; i += kCubeBlockRows) {
      Y[offset + (z + i) * M * K + y * M + x] = tile[threadIdx.x][threadIdx.z + i][threadIdx.y];
    }
  }
}

template <typename T>
void BatchTranspose3DImpl(
    int64_t N,
    int64_t M,
    int64_t L,
    int64_t K,
    const T* X,
    T* Y,
    bool perm_1) {
  int64_t dm = (M + kCubeDim - 1) / kCubeDim;
  int64_t dl = (L + kCubeDim - 1) / kCubeDim;
  int64_t dk = (K + kCubeDim - 1) / kCubeDim;
  if (perm_1)
    _BatchTranspose3DKernel<T><<<N * dm * dl * dk, dim3(kCubeDim, kCubeDim, kCubeBlockRows), 0>>>(N, M, L, K, dm, dl, dk, X, Y);
  else
    _BatchTranspose3DKernel2<T><<<N * dm * dl * dk, dim3(kCubeDim, kCubeDim, kCubeBlockRows), 0>>>(N, M, L, K, dm, dl, dk, X, Y);
}

#define SPECIALIZED_IMPL(T)                                                                                  \
  template void BatchTranspose2DImpl<T>(int64_t N, int64_t H, int64_t W, const T* X, T* Y);                  \
  template void BatchTranspose3DImpl<T>(int64_t N, int64_t M, int64_t L, int64_t K, const T* X, T* Y, bool perm_1);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)
}  // namespace cuda
}  // namespace onnxruntime
