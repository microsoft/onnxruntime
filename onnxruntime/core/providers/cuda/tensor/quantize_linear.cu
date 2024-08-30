// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "quantize_linear.cuh"

#include <limits>

#include "core/providers/cuda/cu_inc/common.cuh"

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11080
#include "cuda_fp8.h"
#include "cuda_fp16.h"
#endif

namespace onnxruntime {
namespace cuda {

template <typename InT, typename OutT>
struct RoundStd;

template <typename InT, typename OutT>
struct RoundStdInt4;

template <typename InT, typename OutT>
struct RoundSat;

template <typename T>
__device__ __forceinline__ int ExtractInt4FromByte(T byte, int index) {
  return static_cast<int>((byte >> (index << 2)) & 0x0f);
}

template <>
__device__ __forceinline__ int ExtractInt4FromByte<int8_t>(int8_t byte, int index) {
  constexpr auto shift = (sizeof(int) << 3) - 4;
  return (static_cast<int>(((byte >> (index << 2)) & 0x0f)) << shift) >> shift;
}

template <>
struct RoundStd<float, int8_t> {
  __device__ __forceinline__ int8_t operator()(float v, float scale, int8_t zero_point) const {
    int value = __float2int_rn(v / scale) + zero_point;
    return static_cast<int8_t>(max(std::numeric_limits<int8_t>::min(), min(std::numeric_limits<int8_t>::max(), value)));
  }
};

template <>
struct RoundStdInt4<float, int8_t> {
  __device__ __forceinline__ int8_t operator()(float v0,
                                               float v1,
                                               float scale0,
                                               float scale1,
                                               int zp0,
                                               int zp1) const {
    int value0 = __float2int_rn(v0 / scale0) + zp0;
    int value1 = __float2int_rn(v1 / scale1) + zp1;
    int value0_clip = max(-8, min(7, value0));
    int value1_clip = max(-8, min(7, value1));
    return static_cast<int8_t>((value0_clip & 0x0f) | ((value1_clip & 0x0f) << 4));
  }
};

template <>
struct RoundStd<float, uint8_t> {
  __device__ __forceinline__ uint8_t operator()(float v, float scale, uint8_t zero_point) const {
    int value = __float2int_rn(v / scale) + zero_point;
    return static_cast<uint8_t>(max(std::numeric_limits<uint8_t>::min(), min(std::numeric_limits<uint8_t>::max(), value)));
  }
};

template <>
struct RoundStdInt4<float, uint8_t> {
  __device__ __forceinline__ uint8_t operator()(float v0,
                                                float v1,
                                                float scale0,
                                                float scale1,
                                                int zp0,
                                                int zp1) const {
    int value0 = __float2int_rn(v0 / scale0) + zp0;
    int value1 = __float2int_rn(v1 / scale1) + zp1;
    int value0_clip = max(0, min(15, value0));
    int value1_clip = max(0, min(15, value1));
    return static_cast<uint8_t>((value0_clip & 0x0f) | ((value1_clip & 0x0f) << 4));
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
struct RoundStdInt4<half, int8_t> {
  __device__ __forceinline__ int8_t operator()(half v0,
                                               half v1,
                                               half scale0,
                                               half scale1,
                                               int zp0,
                                               int zp1) const {
    half2 v = __halves2half2(v0, v1);
    half2 scale = __halves2half2(scale0, scale1);
    half2 scaled_v = v / scale;

    int value0 = __half2int_rn(__low2half(scaled_v)) + zp0;
    int value1 = __half2int_rn(__high2half(scaled_v)) + zp1;
    int value0_clip = max(-8, min(7, value0));
    int value1_clip = max(-8, min(7, value1));
    return static_cast<int8_t>((value0_clip & 0x0f) | ((value1_clip & 0x0f) << 4));
  }
};

template <>
struct RoundStd<half, uint8_t> {
  __device__ __forceinline__ int8_t operator()(half v, half scale, uint8_t zero_point) const {
    int value = __half2int_rn(v / scale) + zero_point;
    return static_cast<uint8_t>(max(std::numeric_limits<uint8_t>::min(), min(std::numeric_limits<uint8_t>::max(), value)));
  }
};

template <>
struct RoundStdInt4<half, uint8_t> {
  __device__ __forceinline__ uint8_t operator()(half v0,
                                                half v1,
                                                half scale0,
                                                half scale1,
                                                int zp0,
                                                int zp1) const {
    half2 v = __halves2half2(v0, v1);
    half2 scale = __halves2half2(scale0, scale1);
    half2 scaled_v = v / scale;

    int value0 = __half2int_rn(__low2half(scaled_v)) + zp0;
    int value1 = __half2int_rn(__high2half(scaled_v)) + zp1;
    int value0_clip = max(0, min(15, value0));
    int value1_clip = max(0, min(15, value1));
    return static_cast<uint8_t>((value0_clip & 0x0f) | ((value1_clip & 0x0f) << 4));
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

// cuda kernel for int4 per tensor quantization with standard rounding
// OutT is int8_t for Int4x2 and uint8_t for UInt4x2
// NumElementsPerThread must be multiple of 2.
template <int NumThreadsPerBlock, int NumElementsPerThread, typename OutT, typename InT>
__global__ void QuantizeLinearKernelStdInt4(const InT* input, OutT* output, const InT* scale_ptr,
                                            const OutT* zero_point_ptr, CUDA_LONG N,
                                            RoundStdInt4<InT, OutT> round) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + (threadIdx.x << 1);
  InT scale = *scale_ptr;
  int zero_point = zero_point_ptr ? ExtractInt4FromByte(*zero_point_ptr, 0) : 0;
  int i = 0;
  constexpr int step = NumThreadsPerBlock << 1;

#pragma unroll
  for (; i + 1 < NumElementsPerThread && id + 1 < N; i += 2, id += step) {
    output[id >> 1] = round(input[id], input[id + 1], scale, scale, zero_point, zero_point);
  }

  if (i < NumElementsPerThread && id < N) {
    output[id >> 1] = round(input[id], 0.0, scale, 1.0, zero_point, 0);
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

// cuda kernel for int4 per axis quantization with standard rounding
// OutT is int8_t for Int4x2 and uint8_t for UInt4x2
// NumElementsPerThread must be multiple of 2.
template <int NumThreadsPerBlock, int NumElementsPerThread, typename OutT, typename InT>
__global__ void QuantizeLinearKernelAxisStdInt4(const InT* input, OutT* output, const InT* scale_ptr,
                                                const OutT* zero_point_ptr, CUDA_LONG num_element,
                                                size_t batch_size, size_t n_scales,
                                                RoundStdInt4<InT, OutT> round) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + (threadIdx.x << 1);
  // Process continuous NumElementsPerThread int4 per thread.
  int i = 0;
  // The scale needs to change every n_same_scale.
  CUDA_LONG n_same_scale = num_element / (batch_size * n_scales);
  constexpr int step = NumThreadsPerBlock << 1;

#pragma unroll
  for (; i + 1 < NumElementsPerThread && id + 1 < num_element; i += 2, id += step) {
    int scale_id0 = (id / n_same_scale) % n_scales;
    int scale_id1 = ((id + 1) / n_same_scale) % n_scales;
    int zp0 = zero_point_ptr == nullptr ? 0 : ExtractInt4FromByte(zero_point_ptr[scale_id0 >> 1], scale_id0 & 1);
    int zp1 = zero_point_ptr == nullptr ? 0 : ExtractInt4FromByte(zero_point_ptr[scale_id1 >> 1], scale_id1 & 1);
    output[id >> 1] = round(input[id],
                            input[id + 1],
                            scale_ptr[scale_id0],
                            scale_ptr[scale_id1],
                            zp0,
                            zp1);
  }

  if (i < NumElementsPerThread && id < num_element) {
    int scale_id0 = (id / n_same_scale) % n_scales;
    int zp0 = zero_point_ptr == nullptr ? 0 : ExtractInt4FromByte(zero_point_ptr[scale_id0 >> 1], scale_id0 & 1);
    output[id >> 1] = round(input[id],
                            0.0,
                            scale_ptr[scale_id0],
                            1.0,
                            zp0,
                            0);
  }
}

// cuda kernel for int4 block-wise quantization with standard rounding
// OutT is int8_t for Int4x2 and uint8_t for UInt4x2
// NumElementsPerThread must be multiple of 2.
template <int NumThreadsPerBlock, int NumElementsPerThread, typename OutT, typename InT>
__global__ void QuantizeLinearKernelBlockStdInt4(const InT* input, OutT* output, const InT* scale_ptr,
                                                 const OutT* zero_point_ptr, CUDA_LONG num_element, size_t KN,
                                                 size_t N, size_t scale_KN, size_t block_size,
                                                 RoundStdInt4<InT, OutT> round) {
  // Process continuous NumElementsPerThread int4 per thread.
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + (threadIdx.x << 1);
  int i = 0;
  constexpr int step = NumThreadsPerBlock << 1;

#pragma unroll
  // Process two elements which belong to one byte at a time.
  for (; i + 1 < NumElementsPerThread && id + 1 < num_element; i += 2, id += step) {
    int x0 = id / KN, x1 = (id + 1) / KN;
    int y0 = id % KN / N, y1 = (id + 1) % KN / N;
    int z0 = id % N, z1 = (id + 1) % N;
    int scale_id0 = x0 * scale_KN + y0 / block_size * N + z0;
    int scale_id1 = x1 * scale_KN + y1 / block_size * N + z1;
    output[id >> 1] = round(input[id],
                            input[id + 1],
                            scale_ptr[scale_id0],
                            scale_ptr[scale_id1],
                            zero_point_ptr == nullptr
                                ? 0
                                : ExtractInt4FromByte(zero_point_ptr[scale_id0 >> 1], scale_id0 & 1),
                            zero_point_ptr == nullptr
                                ? 0
                                : ExtractInt4FromByte(zero_point_ptr[scale_id1 >> 1], scale_id1 & 1));
  }

  // last non-paired element
  if (i < NumElementsPerThread && id < num_element) {
    int x0 = id / KN;
    int y0 = id % KN / N;
    int z0 = id % N;
    int scale_id0 = x0 * scale_KN + y0 / block_size * N + z0;
    output[id >> 1] = round(input[id],
                            0.0,
                            scale_ptr[scale_id0],
                            1.0,
                            zero_point_ptr == nullptr
                                ? 0
                                : ExtractInt4FromByte(zero_point_ptr[scale_id0 >> 1], scale_id0 & 1),
                            0);
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
Status CudaQuantizeLinearStdInt4(cudaStream_t stream, const InT* input, OutT* output, const InT* scale,
                                 const OutT* zero_point, size_t num_of_element) {
  if (num_of_element <= 0)
    return Status::OK();

  static_assert((GridDim::maxElementsPerThread & 1) == 0);

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element,
                                               GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  QuantizeLinearKernelStdInt4<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
  <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      scale,
      zero_point,
      static_cast<int>(num_of_element),
      RoundStdInt4<InT, OutT>());
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

template <class OutT, class InT>
Status CudaQuantizeLinearAxisStdInt4(cudaStream_t stream, const InT* input, OutT* output, const InT* scale,
                                     const OutT* zero_point, size_t num_of_element,
                                     size_t batch_size, size_t n_scales) {
  if (num_of_element <= 0)
    return Status::OK();

  static_assert((GridDim::maxElementsPerThread & 1) == 0);

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element,
                                               GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  QuantizeLinearKernelAxisStdInt4<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          input,
          output,
          scale,
          zero_point,
          static_cast<int>(num_of_element),
          batch_size,
          n_scales,
          RoundStdInt4<InT, OutT>());
  return Status::OK();
}

template <class OutT, class InT>
Status CudaQuantizeLinearBlockStdInt4(cudaStream_t stream, const InT* input, OutT* output, const InT* scale,
                                      const OutT* zero_point, size_t num_of_element, size_t K, size_t N,
                                      size_t block_size) {
  if (num_of_element <= 0)
    return Status::OK();

  static_assert((GridDim::maxElementsPerThread & 1) == 0);

  size_t KN = K * N;
  size_t num_block = (K + block_size - 1) / block_size;
  size_t scale_KN = num_block * N;
  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element,
                                               GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  QuantizeLinearKernelBlockStdInt4<GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          input,
          output,
          scale,
          zero_point,
          static_cast<int>(num_of_element),
          KN,
          N,
          scale_KN,
          block_size,
          RoundStdInt4<InT, OutT>());
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
__global__ void DequantizeLinearKernelStdInt4(const InT* input, OutT* output, const OutT* scale_ptr,
                                              const InT* zero_point_ptr, CUDA_LONG num_element) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + (threadIdx.x << 1);

  OutT scale = *scale_ptr;
  int zero_point = zero_point_ptr ? ExtractInt4FromByte(*zero_point_ptr, 0) : 0;
  int i = 0, v0, v1;
  constexpr int step = NumThreadsPerBlock << 1;
#pragma unroll
  for (; i + 1 < NumElementsPerThread && id + 1 < num_element; i += 2, id += step) {
    v0 = ExtractInt4FromByte(input[id >> 1], 0);
    v1 = ExtractInt4FromByte(input[id >> 1], 1);
    output[id] = static_cast<OutT>(v0 - zero_point) * scale;
    output[id + 1] = static_cast<OutT>(v1 - zero_point) * scale;
  }

  if (i < NumElementsPerThread && id < num_element) {
    v0 = ExtractInt4FromByte(input[id >> 1], 0);
    output[id] = static_cast<OutT>(v0 - zero_point) * scale;
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

template <class InT, class OutT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void DequantizeLinearKernelAxisStdInt4(const InT* input, OutT* output, const OutT* scale_ptr,
                                                  const InT* zero_point_ptr, CUDA_LONG num_element,
                                                  size_t batch_size, size_t n_scales) {
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + (threadIdx.x << 1);
  // The scale needs to change every n_same_scale.
  CUDA_LONG n_same_scale = num_element / (batch_size * n_scales);
  int i = 0;
  int scale_id0, scale_id1, zp0, zp1, v0, v1;
  constexpr int step = NumThreadsPerBlock << 1;

#pragma unroll
  for (; i + 1 < NumElementsPerThread && id + 1 < num_element; i += 2, id += step) {
      scale_id0 = (id / n_same_scale) % n_scales;
      scale_id1 = ((id + 1) / n_same_scale) % n_scales;

      v0 = ExtractInt4FromByte(input[id >> 1], 0);
      v1 = ExtractInt4FromByte(input[id >> 1], 1);
      zp0 = zero_point_ptr == nullptr ? 0 : ExtractInt4FromByte(zero_point_ptr[scale_id0 >> 1], scale_id0 & 1);
      zp1 = zero_point_ptr == nullptr ? 0 : ExtractInt4FromByte(zero_point_ptr[scale_id1 >> 1], scale_id1 & 1);
      output[id] = static_cast<OutT>(v0 - zp0) * scale_ptr[scale_id0];
      output[id + 1] = static_cast<OutT>(v1 - zp1) * scale_ptr[scale_id1];
  }

  if (i < NumElementsPerThread && id < num_element) {
    scale_id0 = (id / n_same_scale) % n_scales;
    v0 = ExtractInt4FromByte(input[id >> 1], 0);
    zp0 = zero_point_ptr == nullptr ? 0 : ExtractInt4FromByte(zero_point_ptr[scale_id0 >> 1], scale_id0 & 1);
    output[id] = static_cast<OutT>(v0 - zp0) * scale_ptr[scale_id0];
  }
}

// cuda kernel for int4 block-wise dequantization with standard rounding
// IntT is int8_t for Int4x2 and uint8_t for UInt4x2
// NumElementsPerThread must be multiple of 2.
template <class InT, class OutT, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void DequantizeLinearKernelBlockStdInt4(const InT* input, OutT* output, const OutT* scale_ptr,
                                                   const InT* zero_point_ptr, CUDA_LONG num_element,
                                                   size_t KN, size_t N, size_t scale_KN, size_t block_size) {
  // Process continuous NumElementsPerThread int4 per thread.
  CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + (threadIdx.x << 1);
  int i = 0;
  constexpr int step = NumThreadsPerBlock << 1;

#pragma unroll
  // Process two elements which belong to one byte at a time.
  for (; i + 1 < NumElementsPerThread && id + 1 < num_element; i += 2, id += step) {
    int x0 = id / KN, x1 = (id + 1) / KN;
    int y0 = id % KN / N, y1 = (id + 1) % KN / N;
    int z0 = id % N, z1 = (id + 1) % N;
    int scale_id0 = x0 * scale_KN + y0 / block_size * N + z0;
    int scale_id1 = x1 * scale_KN + y1 / block_size * N + z1;

    int v0 = ExtractInt4FromByte(input[id >> 1], 0);
    int v1 = ExtractInt4FromByte(input[id >> 1], 1);
    int zp0 = zero_point_ptr == nullptr ? 0 : ExtractInt4FromByte(zero_point_ptr[scale_id0 >> 1], scale_id0 & 1);
    int zp1 = zero_point_ptr == nullptr ? 0 : ExtractInt4FromByte(zero_point_ptr[scale_id1 >> 1], scale_id1 & 1);
    output[id] = static_cast<OutT>(v0 - zp0) * scale_ptr[scale_id0];
    output[id + 1] = static_cast<OutT>(v1 - zp1) * scale_ptr[scale_id1];
  }

  // last non-paired element
  if (i < NumElementsPerThread && id < num_element) {
    int x0 = id / KN;
    int y0 = id % KN / N;
    int z0 = id % N;
    int scale_id0 = x0 * scale_KN + y0 / block_size * N + z0;

    int v0 = ExtractInt4FromByte(input[id >> 1], 0);
    int zp0 = zero_point_ptr == nullptr ? 0 : ExtractInt4FromByte(zero_point_ptr[scale_id0 >> 1], scale_id0 & 1);
    output[id] = static_cast<OutT>(v0 - zp0) * scale_ptr[scale_id0];
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
Status CudaDequantizeLinearStdInt4(cudaStream_t stream, const InT* input, OutT* output, const OutT* scale,
                                   const InT* zero_point, size_t num_of_element) {
  if (num_of_element <= 0)
    return Status::OK();

  static_assert((GridDim::maxElementsPerThread & 1) == 0);

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element,
                                               GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  DequantizeLinearKernelStdInt4<InT, OutT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
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

template <class InT, class OutT>
Status CudaDequantizeLinearAxisStdInt4(cudaStream_t stream, const InT* input, OutT* output, const OutT* scale,
                                       const InT* zero_point, size_t num_of_element,
                                       size_t batch_size, size_t n_scales) {
  if (num_of_element <= 0)
    return Status::OK();

  static_assert((GridDim::maxElementsPerThread & 1) == 0);

  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element,
                                               GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  DequantizeLinearKernelAxisStdInt4<InT, OutT, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          input,
          output,
          scale,
          zero_point,
          static_cast<int>(num_of_element),
          batch_size,
          n_scales);
  return Status::OK();
}

template <class T, class U>
Status CudaDequantizeLinearBlockStdInt4(cudaStream_t stream, const T* input, U* output, const U* scale,
                                        const T* zero_point, size_t num_of_element, size_t K, size_t N,
                                        size_t block_size) {
  if (num_of_element <= 0)
    return Status::OK();

  static_assert((GridDim::maxElementsPerThread & 1) == 0);

  size_t KN = K * N;
  size_t num_block = (K + block_size - 1) / block_size;
  size_t scale_KN = num_block * N;
  int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element,
                                               GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  DequantizeLinearKernelBlockStdInt4<T, U, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          input,
          output,
          scale,
          zero_point,
          static_cast<CUDA_LONG>(num_of_element),
          KN,
          N,
          scale_KN,
          block_size);
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
template Status CudaQuantizeLinearStdInt4<int8_t, float>(cudaStream_t stream, const float* input, int8_t* output, const float* scale, const int8_t* zero_point, size_t num_of_element);
template Status CudaQuantizeLinearStdInt4<uint8_t, float>(cudaStream_t stream, const float* input, uint8_t* output, const float* scale, const uint8_t* zero_point, size_t num_of_element);
template Status CudaQuantizeLinearStdInt4<int8_t, half>(cudaStream_t stream, const half* input, int8_t* output, const half* scale, const int8_t* zero_point, size_t num_of_element);
template Status CudaQuantizeLinearStdInt4<uint8_t, half>(cudaStream_t stream, const half* input, uint8_t* output, const half* scale, const uint8_t* zero_point, size_t num_of_element);

template Status CudaQuantizeLinearAxisStd<int8_t, float>(cudaStream_t stream, const float* input, int8_t* output, const float* scale, const int8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaQuantizeLinearAxisStd<uint8_t, float>(cudaStream_t stream, const float* input, uint8_t* output, const float* scale, const uint8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaQuantizeLinearAxisStd<int8_t, half>(cudaStream_t stream, const half* input, int8_t* output, const half* scale, const int8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaQuantizeLinearAxisStd<uint8_t, half>(cudaStream_t stream, const half* input, uint8_t* output, const half* scale, const uint8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaQuantizeLinearAxisStdInt4<int8_t, float>(cudaStream_t stream, const float* input, int8_t* output, const float* scale, const int8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaQuantizeLinearAxisStdInt4<uint8_t, float>(cudaStream_t stream, const float* input, uint8_t* output, const float* scale, const uint8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaQuantizeLinearAxisStdInt4<int8_t, half>(cudaStream_t stream, const half* input, int8_t* output, const half* scale, const int8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaQuantizeLinearAxisStdInt4<uint8_t, half>(cudaStream_t stream, const half* input, uint8_t* output, const half* scale, const uint8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);

template Status CudaQuantizeLinearBlockStdInt4<int8_t, float>(cudaStream_t stream, const float* input, int8_t* output, const float* scale, const int8_t* zero_point, size_t num_of_element, size_t K, size_t N, size_t block_size);
template Status CudaQuantizeLinearBlockStdInt4<uint8_t, float>(cudaStream_t stream, const float* input, uint8_t* output, const float* scale, const uint8_t* zero_point, size_t num_of_element, size_t K, size_t N, size_t block_size);
template Status CudaQuantizeLinearBlockStdInt4<int8_t, half>(cudaStream_t stream, const half* input, int8_t* output, const half* scale, const int8_t* zero_point, size_t num_of_element, size_t K, size_t N, size_t block_size);
template Status CudaQuantizeLinearBlockStdInt4<uint8_t, half>(cudaStream_t stream, const half* input, uint8_t* output, const half* scale, const uint8_t* zero_point, size_t num_of_element, size_t K, size_t N, size_t block_size);

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
template Status CudaDequantizeLinearStdInt4<int8_t, float>(cudaStream_t stream, const int8_t* input, float* output, const float* scale, const int8_t* zero_point, size_t num_of_element);
template Status CudaDequantizeLinearStdInt4<uint8_t, float>(cudaStream_t stream, const uint8_t* input, float* output, const float* scale, const uint8_t* zero_point, size_t num_of_element);
template Status CudaDequantizeLinearStdInt4<int8_t, half>(cudaStream_t stream, const int8_t* input, half* output, const half* scale, const int8_t* zero_point, size_t num_of_element);
template Status CudaDequantizeLinearStdInt4<uint8_t, half>(cudaStream_t stream, const uint8_t* input, half* output, const half* scale, const uint8_t* zero_point, size_t num_of_element);

template Status CudaDequantizeLinearAxisStd<int8_t, float>(cudaStream_t stream, const int8_t* input, float* output, const float* scale, const int8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaDequantizeLinearAxisStd<uint8_t, float>(cudaStream_t stream, const uint8_t* input, float* output, const float* scale, const uint8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaDequantizeLinearAxisStd<int8_t, half>(cudaStream_t stream, const int8_t* input, half* output, const half* scale, const int8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaDequantizeLinearAxisStd<uint8_t, half>(cudaStream_t stream, const uint8_t* input, half* output, const half* scale, const uint8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaDequantizeLinearAxisStdInt4<int8_t, float>(cudaStream_t stream, const int8_t* input, float* output, const float* scale, const int8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaDequantizeLinearAxisStdInt4<uint8_t, float>(cudaStream_t stream, const uint8_t* input, float* output, const float* scale, const uint8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaDequantizeLinearAxisStdInt4<int8_t, half>(cudaStream_t stream, const int8_t* input, half* output, const half* scale, const int8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);
template Status CudaDequantizeLinearAxisStdInt4<uint8_t, half>(cudaStream_t stream, const uint8_t* input, half* output, const half* scale, const uint8_t* zero_point, size_t num_of_element, size_t batch_size, size_t n_scales);

template Status CudaDequantizeLinearBlockStdInt4<int8_t, float>(cudaStream_t stream, const int8_t* input, float* output, const float* scale, const int8_t* zero_point, size_t num_of_element, size_t K, size_t N, size_t block_size);
template Status CudaDequantizeLinearBlockStdInt4<uint8_t, float>(cudaStream_t stream, const uint8_t* input, float* output, const float* scale, const uint8_t* zero_point, size_t num_of_element, size_t K, size_t N, size_t block_size);
template Status CudaDequantizeLinearBlockStdInt4<int8_t, half>(cudaStream_t stream, const int8_t* input, half* output, const half* scale, const int8_t* zero_point, size_t num_of_element, size_t K, size_t N, size_t block_size);
template Status CudaDequantizeLinearBlockStdInt4<uint8_t, half>(cudaStream_t stream, const uint8_t* input, half* output, const half* scale, const uint8_t* zero_point, size_t num_of_element, size_t K, size_t N, size_t block_size);

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
