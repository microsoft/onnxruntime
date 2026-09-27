// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Thrust code needs to be compiled with nvcc
#include <algorithm>
#include <memory>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "cudnn_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _Fill(
    T* output_data,
    T val,
    CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * blockDim.x * blockIdx.x + threadIdx.x;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = val;
      id += blockDim.x;
    }
  }
}

template <typename T>
void Fill(cudaStream_t stream, T* output, T value, int64_t count) {
  int blocksPerGrid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _Fill<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(output, value, N);
}

template <typename T, typename Index>
__global__ void BroadcastBiasKernel(const T* bias, T* output, int64_t count,
                                    DivMod<Index> cols, int bias_row_stride, int bias_col_stride, T scale) {
  // Keep offsets and the loop increment wide even when fast 32-bit division is sufficient.
  for (int64_t id = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       id < count; id += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    Index row, col;
    cols.divmod(static_cast<Index>(id), row, col);
    const T value = bias[static_cast<int64_t>(row) * bias_row_stride + col * bias_col_stride];
    output[id] = scale == T(1) ? value : scale * value;
  }
}

template <typename T>
void BroadcastBias(cudaStream_t stream, const T* bias, T* output, int rows, int cols,
                   int bias_rows, int bias_cols, T scale) {
  ORT_ENFORCE(rows >= 0 && cols >= 0 &&
                  (bias_rows == 1 || bias_rows == rows) && (bias_cols == 1 || bias_cols == cols),
              "Invalid bias broadcast dimensions");
  const int64_t count = static_cast<int64_t>(rows) * cols;
  if (count == 0) {
    return;
  }

  constexpr int threads = GridDim::maxThreadsPerBlock;
  const int blocks = static_cast<int>(std::min<int64_t>(CeilDiv(count, threads), 65535));
  const int row_stride = bias_rows == 1 ? 0 : bias_cols;
  const int col_stride = bias_cols == 1 ? 0 : 1;
  if (count <= std::numeric_limits<int>::max()) {
    BroadcastBiasKernel<<<blocks, threads, 0, stream>>>(
        bias, output, count, DivMod<int>(cols), row_stride, col_stride, scale);
  } else {
    BroadcastBiasKernel<<<blocks, threads, 0, stream>>>(
        bias, output, count, DivMod<int64_t>(cols), row_stride, col_stride, scale);
  }
}

#define SPECIALIZED_BROADCAST_BIAS(T) \
  template void BroadcastBias<T>(cudaStream_t, const T*, T*, int, int, int, int, T);

SPECIALIZED_BROADCAST_BIAS(float)
SPECIALIZED_BROADCAST_BIAS(double)
SPECIALIZED_BROADCAST_BIAS(half)
SPECIALIZED_BROADCAST_BIAS(BFloat16)

#undef SPECIALIZED_BROADCAST_BIAS

template <typename T>
class ConstantBufferImpl : public IConstantBuffer<T> {
 public:
  ConstantBufferImpl(T val) : val_(val), buffer_(nullptr), count_(0) {
  }
  ~ConstantBufferImpl() {
    if (buffer_)
      cudaFree(buffer_);
  }

  virtual const T* GetBuffer(cudaStream_t stream, size_t count) {
    if (count > count_) {
      if (buffer_) {
        cudaFree(buffer_);
        buffer_ = nullptr;
      }
      CUDA_CALL_THROW(cudaMalloc(&buffer_, count * sizeof(T)));
      count_ = count;

      Fill(stream, buffer_, val_, count);
    }
    return buffer_;
  }

 private:
  T* buffer_;
  size_t count_;
  T val_;
};

template <typename T>
std::unique_ptr<IConstantBuffer<T>> CreateConstantOnes() {
  return std::make_unique<ConstantBufferImpl<T>>(Consts<T>::One);
}

template std::unique_ptr<IConstantBuffer<float>> CreateConstantOnes<float>();
template std::unique_ptr<IConstantBuffer<double>> CreateConstantOnes<double>();
template std::unique_ptr<IConstantBuffer<half>> CreateConstantOnes<half>();
template std::unique_ptr<IConstantBuffer<BFloat16>> CreateConstantOnes<BFloat16>();
#if !defined(DISABLE_FLOAT8_TYPES)
template std::unique_ptr<IConstantBuffer<Float8E4M3FN>> CreateConstantOnes<Float8E4M3FN>();
template std::unique_ptr<IConstantBuffer<Float8E5M2>> CreateConstantOnes<Float8E5M2>();
#endif

#define SPECIALIZED_FILL(T) \
  template void Fill<T>(cudaStream_t stream, T * output, T value, int64_t count);

SPECIALIZED_FILL(int8_t)
SPECIALIZED_FILL(uint8_t)
SPECIALIZED_FILL(bool)
SPECIALIZED_FILL(int16_t)
SPECIALIZED_FILL(int32_t)
SPECIALIZED_FILL(int64_t)
SPECIALIZED_FILL(float)
SPECIALIZED_FILL(double)
SPECIALIZED_FILL(__half)
SPECIALIZED_FILL(BFloat16)
#if !defined(DISABLE_FLOAT8_TYPES)
SPECIALIZED_FILL(Float8E4M3FN)
SPECIALIZED_FILL(Float8E5M2)
#endif

}  // namespace cuda
}  // namespace onnxruntime
