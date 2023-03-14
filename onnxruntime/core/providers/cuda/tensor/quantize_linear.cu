// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quantize_linear.cuh"

#include <limits>

#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename InT, typename OutT>
struct Round;

template <>
struct Round<float, int8_t> {
  __device__ __forceinline__ int8_t operator()(float v, float scale, int8_t zero_point) const {
    int value = __float2int_rn(v / scale) + zero_point;
    return static_cast<int8_t>(max(std::numeric_limits<int8_t>::min(), min(std::numeric_limits<int8_t>::max(), value)));
  }
};

template <>
struct Round<float, uint8_t> {
  __device__ __forceinline__ uint8_t operator()(float v, float scale, uint8_t zero_point) const {
    int value = __float2int_rn(v / scale) + zero_point;
    return static_cast<uint8_t>(max(std::numeric_limits<uint8_t>::min(), min(std::numeric_limits<uint8_t>::max(), value)));
  }
};

template <>
struct Round<float, Float8E4M3FN> {
  __device__ __forceinline__ Float8E4M3FN operator()(float v, float scale, Float8E4M3FN zero_point) const {
    return Float8E4M3FN(v / scale);
  }
};

template <>
struct Round<float, Float8E4M3FNUZ> {
  __device__ __forceinline__ Float8E4M3FNUZ operator()(float v, float scale, Float8E4M3FNUZ zero_point) const {
    return Float8E4M3FNUZ(v / scale);
  }
};

template <>
struct Round<float, Float8E5M2> {
  __device__ __forceinline__ Float8E5M2 operator()(float v, float scale, Float8E5M2 zero_point) const {
    return Float8E5M2(v / scale);
  }
};

template <>
struct Round<float, Float8E5M2FNUZ> {
  __device__ __forceinline__ Float8E5M2FNUZ operator()(float v, float scale, Float8E5M2FNUZ zero_point) const {
    return Float8E5M2FNUZ(v / scale);
  }
};

template <>
struct Round<half, int8_t> {
  __device__ __forceinline__ int8_t operator()(half v, half scale, int8_t zero_point) const {
    int value = __half2int_rn(v / scale) + zero_point;
    return static_cast<int8_t>(max(std::numeric_limits<int8_t>::min(), min(std::numeric_limits<int8_t>::max(), value)));
  }
};

template <>
struct Round<half, uint8_t> {
  __device__ __forceinline__ int8_t operator()(half v, half scale, uint8_t zero_point) const {
    int value = __half2int_rn(v / scale) + zero_point;
    return static_cast<uint8_t>(max(std::numeric_limits<uint8_t>::min(), min(std::numeric_limits<uint8_t>::max(), value)));
  }
};

template <int NumThreadsPerBlock, int NumElementsPerThread, typename OutT, typename InT>
__global__ void QuantizeLinearKernel(const InT* input, OutT* output, const InT* scale_ptr, const OutT* zero_point_ptr, CUDA_LONG N, Round<InT, OutT> round) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

  InT scale = *scale_ptr;
  OutT zero_point = zero_point_ptr != nullptr ? *zero_point_ptr : static_cast<OutT>(0);
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output[id] = round(input[id], scale, zero_point);
      id += NumThreadsPerBlock;
    }
  }
}

template <class OutT, class InT>
Status CudaQuantizeLinear(cudaStream_t stream, const InT* input, OutT* output, const InT* scale, const OutT* zero_point, size_t num_of_element) {
  if (num_of_element <= 0)
    return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  QuantizeLinearKernel<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      scale,
      zero_point,
      static_cast<int>(num_of_element),
      Round<InT, OutT>());
  return Status::OK();
}

template <class InT, class OutT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void DequantizeLinearKernel(const InT* input, OutT* output, const OutT* scale_ptr, const InT* zero_point_ptr, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

  OutT scale = *scale_ptr;
  InT zero_point = zero_point_ptr != nullptr ? *zero_point_ptr : static_cast<InT>(0);
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output[id] = static_cast<OutT>(input[id] - zero_point) * scale;
      id += NumThreadsPerBlock;
    }
  }
}

template <class InT, class OutT>
Status CudaDequantizeLinear(cudaStream_t stream, const InT* input, OutT* output, const OutT* scale, const InT* zero_point, size_t num_of_element) {
  if (num_of_element <= 0)
    return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  DequantizeLinearKernel<InT, OutT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      scale,
      zero_point,
      static_cast<int>(num_of_element));
  return Status::OK();
}

template Status CudaQuantizeLinear<int8_t, float>(cudaStream_t stream, const float* input, int8_t* output, const float* scale, const int8_t* zero_point, size_t num_of_element);
template Status CudaQuantizeLinear<uint8_t, float>(cudaStream_t stream, const float* input, uint8_t* output, const float* scale, const uint8_t* zero_point, size_t num_of_element);
template Status CudaQuantizeLinear<int8_t, half>(cudaStream_t stream, const half* input, int8_t* output, const half* scale, const int8_t* zero_point, size_t num_of_element);
template Status CudaQuantizeLinear<uint8_t, half>(cudaStream_t stream, const half* input, uint8_t* output, const half* scale, const uint8_t* zero_point, size_t num_of_element);
template Status CudaQuantizeLinear<Float8E4M3FN, float>(cudaStream_t stream, const float* input, Float8E4M3FN* output, const float* scale, const Float8E4M3FN* zero_point, size_t num_of_element);
template Status CudaQuantizeLinear<Float8E4M3FNUZ, float>(cudaStream_t stream, const float* input, Float8E4M3FNUZ* output, const float* scale, const Float8E4M3FNUZ* zero_point, size_t num_of_element);
template Status CudaQuantizeLinear<Float8E5M2, float>(cudaStream_t stream, const float* input, Float8E5M2* output, const float* scale, const Float8E5M2* zero_point, size_t num_of_element);
template Status CudaQuantizeLinear<Float8E5M2FNUZ, float>(cudaStream_t stream, const float* input, Float8E5M2FNUZ* output, const float* scale, const Float8E5M2FNUZ* zero_point, size_t num_of_element);

template Status CudaDequantizeLinear<int8_t, float>(cudaStream_t stream, const int8_t* input, float* output, const float* scale, const int8_t* zero_point, size_t num_of_element);
template Status CudaDequantizeLinear<uint8_t, float>(cudaStream_t stream, const uint8_t* input, float* output, const float* scale, const uint8_t* zero_point, size_t num_of_element);
template Status CudaDequantizeLinear<int8_t, half>(cudaStream_t stream, const int8_t* input, half* output, const half* scale, const int8_t* zero_point, size_t num_of_element);
template Status CudaDequantizeLinear<uint8_t, half>(cudaStream_t stream, const uint8_t* input, half* output, const half* scale, const uint8_t* zero_point, size_t num_of_element);
template Status CudaDequantizeLinear<Float8E4M3FN, float>(cudaStream_t stream, const Float8E4M3FN* input, float* output, const float* scale, const Float8E4M3FN* zero_point, size_t num_of_element);
template Status CudaDequantizeLinear<Float8E4M3FNUZ, float>(cudaStream_t stream, const Float8E4M3FNUZ* input, float* output, const float* scale, const Float8E4M3FNUZ* zero_point, size_t num_of_element);
template Status CudaDequantizeLinear<Float8E5M2, float>(cudaStream_t stream, const Float8E5M2* input, float* output, const float* scale, const Float8E5M2* zero_point, size_t num_of_element);
template Status CudaDequantizeLinear<Float8E5M2FNUZ, float>(cudaStream_t stream, const Float8E5M2FNUZ* input, float* output, const float* scale, const Float8E5M2FNUZ* zero_point, size_t num_of_element);

}  // namespace cuda
}  // namespace onnxruntime
