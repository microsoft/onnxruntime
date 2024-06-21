// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Thrust code needs to be compiled with nvcc
#include <memory>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "cudnn_common.h"
#include "core/providers/cpu/tensor/utils.h"
namespace onnxruntime {
namespace cuda {

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void StridedTensorCopy(
    const int rank,
    const int N,
    const T* input_data,
    T* output_data,
    const TArray<fast_divmod> output_strides,
    const TArray<int64_t> input_strides) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[NumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG index = 0;
      CUDA_LONG offset = id;
#pragma unroll
      for (auto dim = 0; dim < output_strides.Capacity(); dim++) {
        if (dim >= rank) {
          break;
        }

        int q, r;
        output_strides[dim].divmod(offset, q, r);
        index += static_cast<int>(input_strides[dim]) * q;
        offset = r;
      }

      value[i] = input_data[index];
      id += NumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = value[i];
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T>
void LaunchStridedTensorCopyKernel(const void* input_data, void* output_data,
                                   int blocksPerGrid, int rank, int N_output,
                                   const TArray<fast_divmod>& output_strides,
                                   const TArray<int64_t>& input_strides, cudaStream_t stream) {
  StridedTensorCopy<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          rank, N_output, reinterpret_cast<const T*>(input_data), reinterpret_cast<T*>(output_data),
          output_strides, input_strides);
}

Status StridedTensorCopyImpl(
    cudaStream_t stream,
    const size_t element_byte_size,
    const size_t total_element_count,
    const void* input_data,
    void* output_data,
    const std::vector<int64_t>& output_dims,
    const std::vector<int64_t>& input_strides) {
  const int rank = static_cast<int>(output_dims.size());
  TensorPitches dst_pitches(output_dims);
  TArray<fast_divmod> output_strides_fdms;
  output_strides_fdms.SetSize(rank);
  for (auto i = 0; i < rank; ++i) {
    output_strides_fdms[i] = fast_divmod(gsl::narrow_cast<int>(dst_pitches[i]));
  }

  TArray<int64_t> input_strides_ta(input_strides);

  int blocksPerGrid = gsl::narrow_cast<int>(CeilDiv(total_element_count,
                                                    GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));

  switch (element_byte_size) {
    case sizeof(uint8_t):
      LaunchStridedTensorCopyKernel<uint8_t>(input_data, output_data, blocksPerGrid, rank, total_element_count,
                                             output_strides_fdms, input_strides_ta, stream);
      break;
    case sizeof(uint16_t):
      LaunchStridedTensorCopyKernel<uint16_t>(input_data, output_data, blocksPerGrid, rank, total_element_count,
                                              output_strides_fdms, input_strides_ta, stream);
      break;
    case sizeof(uint32_t):
      LaunchStridedTensorCopyKernel<uint32_t>(input_data, output_data, blocksPerGrid, rank, total_element_count,
                                              output_strides_fdms, input_strides_ta, stream);
      break;
    case sizeof(uint64_t):
      LaunchStridedTensorCopyKernel<uint64_t>(input_data, output_data, blocksPerGrid, rank, total_element_count,
                                              output_strides_fdms, input_strides_ta, stream);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for strided tensor copy");
  }
  return Status::OK();
}

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
