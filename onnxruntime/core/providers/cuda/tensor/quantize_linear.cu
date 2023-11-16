// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quantize_linear.cuh"

#include <limits>

#include "core/providers/cuda/cu_inc/common.cuh"

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include "cuda_fp8.h"
#endif

namespace onnxruntime {
namespace cuda {

template <typename InT, typename OutT>
struct RoundStd;

template <typename InT, typename OutT>
struct RoundSat;

template <>
struct RoundStd<float, int8_t> {
  __device__ __forceinline__ int8_t operator()(float v, float scale, int8_t zero_point) const {
    int value = __float2int_rn(v / scale) + zero_point;
    return static_cast<int8_t>(max(std::numeric_limits<int8_t>::min(), min(std::numeric_limits<int8_t>::max(), value)));
  }
};

template <>
struct RoundStd<float, uint8_t> {
  __device__ __forceinline__ uint8_t operator()(float v, float scale, uint8_t zero_point) const {
    int value = __float2int_rn(v / scale) + zero_point;
    return static_cast<uint8_t>(max(std::numeric_limits<uint8_t>::min(), min(std::numeric_limits<uint8_t>::max(), value)));
  }
};

#if !defined(DISABLE_FLOAT8_TYPES)

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080

// Conversion from float 8 to float or float16 does not need zero_point argument as defined by onnx standard.

template <>
struct RoundSat<float, Float8E4M3FN> {
  __device__ __forceinline__ Float8E4M3FN operator()(float v, float scale, Float8E4M3FN /* zero_point */, bool saturate) const {
    return Float8E4M3FN(static_cast<unsigned char>(__nv_cvt_float_to_fp8(v / scale, saturate ? __NV_SATFINITE : __NV_NOSAT, __NV_E4M3)), Float8E4M3FN::FromBits());
  }
};

template <>
struct RoundSat<half, Float8E4M3FN> {
  __device__ __forceinline__ Float8E4M3FN operator()(half v, half scale, Float8E4M3FN /* zero_point */, bool saturate) const {
    return Float8E4M3FN(static_cast<unsigned char>(__nv_cvt_halfraw_to_fp8(v / scale, saturate ? __NV_SATFINITE : __NV_NOSAT, __NV_E4M3)), Float8E4M3FN::FromBits());
  }
};

template <>
struct RoundSat<float, Float8E5M2> {
  __device__ __forceinline__ Float8E5M2 operator()(float v, float scale, Float8E5M2 /* zero_point */, bool saturate) const {
    return Float8E5M2(static_cast<unsigned char>(__nv_cvt_float_to_fp8(v / scale, saturate ? __NV_SATFINITE : __NV_NOSAT, __NV_E5M2)), Float8E5M2::FromBits());
  }
};

template <>
struct RoundSat<half, Float8E5M2> {
  __device__ __forceinline__ Float8E5M2 operator()(half v, half scale, Float8E5M2 /* zero_point */, bool saturate) const {
    return Float8E5M2(static_cast<unsigned char>(__nv_cvt_halfraw_to_fp8(v / scale, saturate ? __NV_SATFINITE : __NV_NOSAT, __NV_E5M2)), Float8E5M2::FromBits());
  }
};

#else

// Conversion from float 8 to float or float16 does not need zero_point argument as defined by onnx standard.

template <>
struct RoundSat<float, Float8E4M3FN> {
  __device__ __forceinline__ Float8E4M3FN operator()(float v, float scale, Float8E4M3FN /* zero_point */, bool saturate) const {
    return Float8E4M3FN(v / scale, saturate);
  }
};

template <>
struct RoundSat<half, Float8E4M3FN> {
  __device__ __forceinline__ Float8E4M3FN operator()(half v, half scale, Float8E4M3FN /* zero_point */, bool saturate) const {
    return Float8E4M3FN(__half2float(v / scale), true);
  }
};

template <>
struct RoundSat<float, Float8E5M2> {
  __device__ __forceinline__ Float8E5M2 operator()(float v, float scale, Float8E5M2 /* zero_point */, bool saturate) const {
    return Float8E5M2(v / scale, saturate);
  }
};

template <>
struct RoundSat<half, Float8E5M2> {
  __device__ __forceinline__ Float8E5M2 operator()(half v, half scale, Float8E5M2 /* zero_point */, bool saturate) const {
    return Float8E5M2(__half2float(v / scale), saturate);
  }
};

#endif

#endif  // DISABLE_FLOAT8_TYPES 

template <>
struct RoundStd<half, int8_t> {
  __device__ __forceinline__ int8_t operator()(half v, half scale, int8_t zero_point) const {
    int value = __half2int_rn(v / scale) + zero_point;
    return static_cast<int8_t>(max(std::numeric_limits<int8_t>::min(), min(std::numeric_limits<int8_t>::max(), value)));
  }
};

template <>
struct RoundStd<half, uint8_t> {
  __device__ __forceinline__ int8_t operator()(half v, half scale, uint8_t zero_point) const {
    int value = __half2int_rn(v / scale) + zero_point;
    return static_cast<uint8_t>(max(std::numeric_limits<uint8_t>::min(), min(std::numeric_limits<uint8_t>::max(), value)));
  }
};

template <int NumThreadsPerBlock, int NumElementsPerThread, typename OutT, typename InT>
__global__ void QuantizeLinearKernelStd(const InT* input, OutT* output, const InT* scale_ptr, const OutT* zero_point_ptr, CUDA_LONG N, RoundStd<InT, OutT> round) {
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

template <int NumThreadsPerBlock, int NumElementsPerThread, typename OutT, typename InT>
__global__ void QuantizeLinearKernelAxisStd(const InT* input, OutT* output, const InT* scale_ptr, const OutT* zero_point_ptr, CUDA_LONG N, size_t batch_size, size_t n_scales, RoundStd<InT, OutT> round) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  // The scale needs to change every n_same_scale.
  CUDA_LONG n_same_scale = N / (batch_size * n_scales);
  int scale_id;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      scale_id = (id / n_same_scale) % n_scales;
      output[id] = round(input[id], scale_ptr[scale_id], zero_point_ptr == nullptr ? static_cast<OutT>(0) : zero_point_ptr[scale_id]);
      id += NumThreadsPerBlock;
    }
  }
}

#if !defined(DISABLE_FLOAT8_TYPES)

template <int NumThreadsPerBlock, int NumElementsPerThread, typename OutT, typename InT>
__global__ void QuantizeLinearKernelSat(const InT* input, OutT* output, const InT* scale_ptr, const OutT* zero_point_ptr, CUDA_LONG N, RoundSat<InT, OutT> round, bool saturate) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

  InT scale = *scale_ptr;
  OutT zero_point = zero_point_ptr != nullptr ? *zero_point_ptr : OutT(0, true);
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output[id] = round(input[id], scale, zero_point, saturate);
      id += NumThreadsPerBlock;
    }
  }
}

template <int NumThreadsPerBlock, int NumElementsPerThread, typename OutT, typename InT>
__global__ void QuantizeLinearKernelAxisSat(const InT* input, OutT* output, const InT* scale_ptr, const OutT* zero_point_ptr, CUDA_LONG N,
                                            size_t batch_size, size_t n_scales, RoundSat<InT, OutT> round, bool saturate) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  // The scale needs to change every n_same_scale.
  CUDA_LONG n_same_scale = N / (batch_size * n_scales);
  int scale_id;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      scale_id = (id / n_same_scale) % n_scales;
      output[id] = round(input[id], scale_ptr[scale_id], zero_point_ptr == nullptr ? OutT(0, true) : zero_point_ptr[scale_id], saturate);
      id += NumThreadsPerBlock;
    }
  }
}

#endif  // DISABLE_FLOAT8_TYPES

template <class OutT, class InT>
Status CudaQuantizeLinearStd(cudaStream_t stream, const InT* input, OutT* output, const InT* scale, const OutT* zero_point, size_t num_of_element) {
  if (num_of_element <= 0)
    return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  QuantizeLinearKernelStd<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      scale,
      zero_point,
      static_cast<int>(num_of_element),
      RoundStd<InT, OutT>());
  return Status::OK();
}

template <class OutT, class InT>
Status CudaQuantizeLinearAxisStd(cudaStream_t stream, const InT* input, OutT* output, const InT* scale, const OutT* zero_point, size_t num_of_element,
                                 size_t batch_size, size_t n_scales) {
  if (num_of_element <= 0)
    return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  QuantizeLinearKernelAxisStd<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      scale,
      zero_point,
      static_cast<int>(num_of_element),
      batch_size,
      n_scales,
      RoundStd<InT, OutT>());
  return Status::OK();
}

#if !defined(DISABLE_FLOAT8_TYPES)

template <class OutT, class InT>
Status CudaQuantizeLinearSat(cudaStream_t stream, const InT* input, OutT* output, const InT* scale, const OutT* zero_point, size_t num_of_element, bool saturate) {
  if (num_of_element <= 0)
    return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  QuantizeLinearKernelSat<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      scale,
      zero_point,
      static_cast<int>(num_of_element),
      RoundSat<InT, OutT>(),
      saturate);
  return Status::OK();
}

template <class OutT, class InT>
Status CudaQuantizeLinearAxisSat(cudaStream_t stream, const InT* input, OutT* output, const InT* scale, const OutT* zero_point, size_t num_of_element,
                                 size_t batch_size, size_t n_scales, bool saturate) {
  if (num_of_element <= 0)
    return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  QuantizeLinearKernelAxisSat<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      scale,
      zero_point,
      static_cast<int>(num_of_element),
      batch_size,
      n_scales,
      RoundSat<InT, OutT>(),
      saturate);
  return Status::OK();
}

#endif

template <class InT, class OutT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void DequantizeLinearKernelStd(const InT* input, OutT* output, const OutT* scale_ptr, const InT* zero_point_ptr, CUDA_LONG N) {
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

template <class InT, class OutT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void DequantizeLinearKernelAxisStd(const InT* input, OutT* output, const OutT* scale_ptr, const InT* zero_point_ptr, CUDA_LONG N,
                                              size_t batch_size, size_t n_scales) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  // The scale needs to change every n_same_scale.
  CUDA_LONG n_same_scale = N / (batch_size * n_scales);
  int scale_id;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      scale_id = (id / n_same_scale) % n_scales;
      output[id] = (zero_point_ptr == nullptr ? static_cast<OutT>(input[id]) : static_cast<OutT>(input[id] - zero_point_ptr[scale_id])) * scale_ptr[scale_id];
      id += NumThreadsPerBlock;
    }
  }
}

template <typename InT, typename OutT>
struct DQFloat8;

#if !defined(DISABLE_FLOAT8_TYPES)

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080

template <>
struct DQFloat8<Float8E4M3FN, half> {
  __device__ __forceinline__ half operator()(Float8E4M3FN v, half scale) const {
    return __nv_cvt_fp8_to_halfraw(v.val, __NV_E4M3) * scale;
  }
};

template <>
struct DQFloat8<Float8E5M2, half> {
  __device__ __forceinline__ half operator()(Float8E5M2 v, half scale) const {
    return __nv_cvt_fp8_to_halfraw(v.val, __NV_E5M2) * scale;
  }
};

template <>
struct DQFloat8<Float8E4M3FN, float> {
  __device__ __forceinline__ float operator()(Float8E4M3FN v, float scale) const {
    return __half2float(__nv_cvt_fp8_to_halfraw(v.val, __NV_E4M3)) * scale;
  }
};

template <>
struct DQFloat8<Float8E5M2, float> {
  __device__ __forceinline__ float operator()(Float8E5M2 v, float scale) const {
    return __half2float(__nv_cvt_fp8_to_halfraw(v.val, __NV_E5M2)) * scale;
  }
};

#else

template <>
struct DQFloat8<Float8E4M3FN, half> {
  __device__ __forceinline__ half operator()(Float8E4M3FN v, half scale) const {
    return __float2half(v.ToFloat()) * scale;
  }
};

template <>
struct DQFloat8<Float8E5M2, half> {
  __device__ __forceinline__ half operator()(Float8E5M2 v, half scale) const {
    return __float2half(v.ToFloat()) * scale;
  }
};

template <>
struct DQFloat8<Float8E4M3FN, float> {
  __device__ __forceinline__ float operator()(Float8E4M3FN v, float scale) const {
    return v.ToFloat() * scale;
  }
};

template <>
struct DQFloat8<Float8E5M2, float> {
  __device__ __forceinline__ float operator()(Float8E5M2 v, float scale) const {
    return v.ToFloat() * scale;
  }
};

#endif

#endif

template <class InT, class OutT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void DequantizeLinearKernelSat(const InT* input, OutT* output, const OutT* scale_ptr, const InT* zero_point_ptr, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;

  OutT scale = *scale_ptr;
  // zero_point is unused.
  // InT zero_point = zero_point_ptr != nullptr ? *zero_point_ptr : InT(0, true);
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output[id] = DQFloat8<InT, OutT>()(input[id], scale);
      id += NumThreadsPerBlock;
    }
  }
}

#if !defined(DISABLE_FLOAT8_TYPES)

template <class InT, class OutT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void DequantizeLinearKernelAxisSat(const InT* input, OutT* output, const OutT* scale_ptr, const InT* zero_point_ptr, CUDA_LONG N,
                                              size_t batch_size, size_t n_scales) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  // The scale needs to change every n_same_scale.
  CUDA_LONG n_same_scale = N / (batch_size * n_scales);
  int scale_id;

#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      scale_id = (id / n_same_scale) % n_scales;
      output[id] = DQFloat8<InT, OutT>()(input[id], scale_ptr[scale_id]);
      id += NumThreadsPerBlock;
    }
  }
}

#endif

template <class InT, class OutT>
Status CudaDequantizeLinearStd(cudaStream_t stream, const InT* input, OutT* output, const OutT* scale, const InT* zero_point, size_t num_of_element) {
  if (num_of_element <= 0)
    return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  DequantizeLinearKernelStd<InT, OutT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      scale,
      zero_point,
      static_cast<int>(num_of_element));
  return Status::OK();
}

template <class InT, class OutT>
Status CudaDequantizeLinearAxisStd(cudaStream_t stream, const InT* input, OutT* output, const OutT* scale, const InT* zero_point, size_t num_of_element,
                                   size_t batch_size, size_t n_scales) {
  if (num_of_element <= 0)
    return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  DequantizeLinearKernelAxisStd<InT, OutT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      scale,
      zero_point,
      static_cast<int>(num_of_element),
      batch_size,
      n_scales);
  return Status::OK();
}

#if !defined(DISABLE_FLOAT8_TYPES)

template <class InT, class OutT>
Status CudaDequantizeLinearSat(cudaStream_t stream, const InT* input, OutT* output, const OutT* scale, const InT* zero_point, size_t num_of_element) {
  if (num_of_element <= 0)
    return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  DequantizeLinearKernelSat<InT, OutT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      scale,
      zero_point,
      static_cast<int>(num_of_element));
  return Status::OK();
}

template <class InT, class OutT>
Status CudaDequantizeLinearAxisSat(cudaStream_t stream, const InT* input, OutT* output, const OutT* scale, const InT* zero_point, size_t num_of_element,
                                   size_t batch_size, size_t n_scales) {
  if (num_of_element <= 0)
    return Status::OK();

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  DequantizeLinearKernelAxisSat<InT, OutT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      scale,
      zero_point,
      static_cast<int>(num_of_element),
      batch_size,
      n_scales);
  return Status::OK();
}

#endif

template Status CudaQuantizeLinearStd<int8_t, float>(cudaStream_t stream, const float* input, int8_t* output, const float* scale, const int8_t* zero_point, size_t num_of_element);
template Status CudaQuantizeLinearStd<uint8_t, float>(cudaStream_t stream, const float* input, uint8_t* output, const float* scale, const uint8_t* zero_point, size_t num_of_element);
template Status CudaQuantizeLinearStd<int8_t, half>(cudaStream_t stream, const half* input, int8_t* output, const half* scale, const int8_t* zero_point, size_t num_of_element);
template Status CudaQuantizeLinearStd<uint8_t, half>(cudaStream_t stream, const half* input, uint8_t* output, const half* scale, const uint8_t* zero_point, size_t num_of_element);

template Status CudaQuantizeLinearAxisStd<int8_t, float>(cudaStream_t stream, const float* input, int8_t* output, const float* scale, const int8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaQuantizeLinearAxisStd<uint8_t, float>(cudaStream_t stream, const float* input, uint8_t* output, const float* scale, const uint8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaQuantizeLinearAxisStd<int8_t, half>(cudaStream_t stream, const half* input, int8_t* output, const half* scale, const int8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaQuantizeLinearAxisStd<uint8_t, half>(cudaStream_t stream, const half* input, uint8_t* output, const half* scale, const uint8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);

#if !defined(DISABLE_FLOAT8_TYPES)

template Status CudaQuantizeLinearSat<Float8E4M3FN, float>(cudaStream_t stream, const float* input, Float8E4M3FN* output, const float* scale, const Float8E4M3FN* zero_point, size_t num_of_element, bool saturate);
template Status CudaQuantizeLinearSat<Float8E5M2, float>(cudaStream_t stream, const float* input, Float8E5M2* output, const float* scale, const Float8E5M2* zero_point, size_t num_of_element, bool saturate);
template Status CudaQuantizeLinearSat<Float8E4M3FN, half>(cudaStream_t stream, const half* input, Float8E4M3FN* output, const half* scale, const Float8E4M3FN* zero_point, size_t num_of_element, bool saturate);
template Status CudaQuantizeLinearSat<Float8E5M2, half>(cudaStream_t stream, const half* input, Float8E5M2* output, const half* scale, const Float8E5M2* zero_point, size_t num_of_element, bool saturate);

template Status CudaQuantizeLinearAxisSat<Float8E4M3FN, float>(cudaStream_t stream, const float* input, Float8E4M3FN* output, const float* scale, const Float8E4M3FN* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales, bool saturate);
template Status CudaQuantizeLinearAxisSat<Float8E5M2, float>(cudaStream_t stream, const float* input, Float8E5M2* output, const float* scale, const Float8E5M2* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales, bool saturate);
template Status CudaQuantizeLinearAxisSat<Float8E4M3FN, half>(cudaStream_t stream, const half* input, Float8E4M3FN* output, const half* scale, const Float8E4M3FN* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales, bool saturate);
template Status CudaQuantizeLinearAxisSat<Float8E5M2, half>(cudaStream_t stream, const half* input, Float8E5M2* output, const half* scale, const Float8E5M2* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales, bool saturate);

#endif

template Status CudaDequantizeLinearStd<int8_t, float>(cudaStream_t stream, const int8_t* input, float* output, const float* scale, const int8_t* zero_point, size_t num_of_element);
template Status CudaDequantizeLinearStd<uint8_t, float>(cudaStream_t stream, const uint8_t* input, float* output, const float* scale, const uint8_t* zero_point, size_t num_of_element);
template Status CudaDequantizeLinearStd<int8_t, half>(cudaStream_t stream, const int8_t* input, half* output, const half* scale, const int8_t* zero_point, size_t num_of_element);
template Status CudaDequantizeLinearStd<uint8_t, half>(cudaStream_t stream, const uint8_t* input, half* output, const half* scale, const uint8_t* zero_point, size_t num_of_element);

template Status CudaDequantizeLinearAxisStd<int8_t, float>(cudaStream_t stream, const int8_t* input, float* output, const float* scale, const int8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaDequantizeLinearAxisStd<uint8_t, float>(cudaStream_t stream, const uint8_t* input, float* output, const float* scale, const uint8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaDequantizeLinearAxisStd<int8_t, half>(cudaStream_t stream, const int8_t* input, half* output, const half* scale, const int8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaDequantizeLinearAxisStd<uint8_t, half>(cudaStream_t stream, const uint8_t* input, half* output, const half* scale, const uint8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);

#if !defined(DISABLE_FLOAT8_TYPES)

template Status CudaDequantizeLinearSat<Float8E4M3FN, float>(cudaStream_t stream, const Float8E4M3FN* input, float* output, const float* scale, const Float8E4M3FN* zero_point, size_t num_of_element);
template Status CudaDequantizeLinearSat<Float8E5M2, float>(cudaStream_t stream, const Float8E5M2* input, float* output, const float* scale, const Float8E5M2* zero_point, size_t num_of_element);
template Status CudaDequantizeLinearSat<Float8E4M3FN, half>(cudaStream_t stream, const Float8E4M3FN* input, half* output, const half* scale, const Float8E4M3FN* zero_point, size_t num_of_element);
template Status CudaDequantizeLinearSat<Float8E5M2, half>(cudaStream_t stream, const Float8E5M2* input, half* output, const half* scale, const Float8E5M2* zero_point, size_t num_of_element);

template Status CudaDequantizeLinearAxisSat<Float8E4M3FN, float>(cudaStream_t stream, const Float8E4M3FN* input, float* output, const float* scale, const Float8E4M3FN* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaDequantizeLinearAxisSat<Float8E5M2, float>(cudaStream_t stream, const Float8E5M2* input, float* output, const float* scale, const Float8E5M2* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaDequantizeLinearAxisSat<Float8E4M3FN, half>(cudaStream_t stream, const Float8E4M3FN* input, half* output, const half* scale, const Float8E4M3FN* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaDequantizeLinearAxisSat<Float8E5M2, half>(cudaStream_t stream, const Float8E5M2* input, half* output, const half* scale, const Float8E5M2* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);

#endif

}  // namespace cuda
}  // namespace onnxruntime
