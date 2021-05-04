// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "gist_impl.h"
#include "gist.h"
#include <cuda_runtime.h>
using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _GistBinarizeEncoderKernel(
    const T* input_data,
    bool* output_data,
    const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output_data[id] = (input_data[id] > (T)0);
}

template <typename T>
__global__ void _GistBinarizeDecoderKernel(
    const bool* input_data,
    T* output_data,
    const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output_data[id] = (input_data[id] ? (T)1 : (T)0);
}

template <typename T>
__global__ void _GistPack1EncoderKernel(
    const T* input_data,
    uint8_t* output_data,
    const size_t factor,
    const CUDA_LONG N) {
 
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N); // id of Y (compressed tensor)
  uint8_t out = 0x0;
  uint8_t bit_out = 0x0;
  size_t begin = id * factor;
  size_t end = id * factor + factor;
  for(size_t idx = begin; idx < end; idx++){
    bool bit = (input_data[idx] > (T)0);
    int nidxshift = idx % factor;
    bit_out = bit ? (0x80 >> nidxshift) : 0;
    out |= bit_out;
  }
  output_data[id] = out;
}
template <typename T>
__global__ void _GistPack1DecoderKernel(
    const uint8_t* input_data,
    T* output_data,
    const size_t factor,
    const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N); // id of Y (uncompressed tensor)
  int nidx = id / factor;
  int nidxshift = id % factor;
  uint8_t mask = 0x80 >> nidxshift;
  uint8_t in = input_data[nidx] & mask;
  output_data[id] = (in > 0) ? (T)1 : (T)0;
}

template <typename T>
__global__ void _GistPack8EncoderKernel(
    const T* input_data,
    uint8_t* output_data,
    const CUDA_LONG N) {
 
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  
  T X = input_data[id];

  if (X == (T)0) {
    output_data[id] = (uint8_t)(0);
    return;
  }
  uint32_t i = (uint32_t)__float_as_uint(X);
  uint32_t e_mask = 0x7f800000;
  uint32_t m_residual_mask = 0x00080000;
  uint32_t m_mask = 0x007fffff;
  uint32_t m_size = 23;
  uint32_t e_size = 8;
  uint32_t pack_e_size = 5;
  uint32_t pack_m_size = 2;
  uint8_t bias = 127;
  switch(sizeof(T)){
    case 4:
      m_size = 23;
      e_size = 8;
      e_mask = 0x7f800000;
      m_mask = 0x007fffff;
      m_residual_mask = 0x00080000;
      bias = 127;
      break;
    case 2:
      m_size = 10;
      e_size = 5;
      e_mask = 0x0f800000;
      m_mask = 0x000003ff;
      m_residual_mask = 0x00000007;
      bias = 15;
      break;
  }
  uint32_t pack_e_shift = e_size - pack_e_size;
  uint32_t pack_m_shift = m_size - pack_m_size;

  uint32_t s = i >> (m_size + e_size);
  uint32_t e = i & e_mask;
  e >>= (m_size);
  e -= bias;
  uint32_t m = i & m_mask;

  uint32_t pack_e = e >> pack_e_shift;
  uint32_t pack_m = m >> pack_m_shift;
  uint32_t m_residual = m & m_residual_mask;
  if(m_residual > 0){ // round up
    if(pack_m == 0x3){
      pack_e +=1; // increase exponent
      pack_m = 0;
    }
    else{
      pack_m +=1; // increase mantissa
    }
  }
  if (pack_e >= 0x1f) { //NaN values
    pack_e = 0;
  }
  output_data[id] = (s << (pack_e_size + pack_m_size)) | (pack_e << pack_m_size) | pack_m;  
}

template <typename T>
__global__ void _GistPack8DecoderKernel(
    const uint8_t* input_data,
    T* output_data,
    const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  uint8_t i = input_data[id];
  if (i == 0) {
    output_data[id] = (T)0;
    return;
  }
  uint32_t pack_e_size = 5;
  uint32_t pack_m_size = 2;
  uint32_t pack_e_mask = 0x0000007c;
  uint32_t pack_m_mask = 0x00000003;
  uint32_t m_size = 23;
  uint32_t e_size = 8;
  uint32_t bias = 127;

  switch(sizeof(T)){
    case 4:
      m_size = 23;
      e_size = 8;
      bias = 127;
      break;
    case 2:
      m_size = 10;
      e_size = 5;
      bias = 15;
      break;
  }
  uint32_t pack_e_shift = e_size - pack_e_size;
  uint32_t s = i >> (pack_e_size+ pack_m_size);
  uint32_t pack_e = i & pack_e_mask;
  pack_e >>= pack_m_size;
  uint32_t pack_m = i & pack_m_mask;
  uint32_t unpack_e = pack_e << (pack_e_shift + m_size);
  unpack_e += bias;
  uint32_t unpack_m = pack_m << (m_size -pack_m_size);
  uint32_t unpack = (s << (m_size+e_size)) | unpack_e | unpack_m;

  output_data[id] = (T)__uint_as_float((unsigned int)unpack);
}

template <typename T>
__global__ void _GistPack16EncoderKernel(
    const T* input_data,
    half* output_data,
    const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  
  T X = input_data[id];
  output_data[id] = __float2half(X);
}

template <typename T>
__global__ void _GistPack16DecoderKernel(
    const half* input_data,
    T* output_data,
    const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  half X = input_data[id];
  output_data[id] = (T)__half2float(X);
}

template <typename T>
__global__ void _GistPackMsfp15EncoderKernel(
    const T* input_data,
    uint8_t* output_data,
    const CUDA_LONG num_threads,
    const CUDA_LONG pre_axis_size,
    const CUDA_LONG axis_size,
    const CUDA_LONG num_tiles,
    const CUDA_LONG tile_size) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, num_threads);

  // Quantization parameters
  const int bits = 7;
  // mantissa bits, remove sign
  const int m_bits = bits - 1;

  // float32 parameters
  const uint32_t s_mask = 0x80000000;
  const int s_shift = 31;
  const int pack_s_shift = 6;
  const uint32_t e_mask = 0x7f800000;
  const int e_shift = 23;
  const int pack_e_shift = 7;
  const uint32_t m_mask = 0x007fffff;

  const int tile_i = id % num_tiles;
  const int pre_axis_i = id / num_tiles;

  // Loop over bounding box to find shared exponent
  uint32_t shared_exp = 0;
  for (size_t i = 0; i < tile_size; i++) {
    // Get input
    size_t in_i = pre_axis_i * axis_size +
                  tile_i * tile_size +
                  i;
    T X = input_data[in_i];
    uint32_t X_i = (uint32_t)__float_as_uint(X);
    // Get exponent
    uint32_t exp = (X_i & e_mask) >> e_shift;
    // Shared exponent is max of exponents
    if (exp > shared_exp) {
      shared_exp = exp;
    }
  }


  // If inf/nan is found, zero out values
  if (shared_exp >= 0xff) {
    for (size_t i = 0; i < tile_size; i++) {
      size_t in_i = pre_axis_i * axis_size +
                    tile_i * tile_size +
                    i;
      output_data[in_i] = 0;
    }

    return;
  }


  // Copy of shared exponent for packing
  uint32_t pack_shared_exp = shared_exp;

  // Loop over bounding box to quantize
  for (size_t i = 0; i < tile_size; i++) {
    size_t in_i = pre_axis_i * axis_size +
                  tile_i * tile_size +
                  i;
    T X = input_data[in_i];
    uint32_t X_i = (uint32_t)__float_as_uint(X);

    // Get biased exponent
    uint32_t exp = (X_i & e_mask) >> e_shift;
    uint32_t sign;
    uint32_t mantissa;
    if (exp == 0) {
      // Flush denorm to 0
      sign = 0;
      mantissa = 0;
    } else {
      // Decode float
      sign = X_i & s_mask;
      mantissa = X_i & m_mask;

      // Difference in exponents
      uint32_t exp_diff = shared_exp - exp;

      // Implied 1
      mantissa = mantissa + (1 << 23);
      // Adjust for shared exponent
      mantissa = mantissa >> exp_diff;
      // Shift down to target bit width + 1
      mantissa = mantissa >> (24 - m_bits - 1);
      // Rounding (with overflow check)
      if (mantissa != ((1 << (m_bits + 1)) - 1)) {
        mantissa += 1;
      }
      // Shift away last bit
      mantissa = mantissa >> 1;
    }
    // Store {exponent bit, mantissa} in output
    uint8_t exp_bit = (pack_shared_exp % 2) << pack_e_shift;
    pack_shared_exp = pack_shared_exp >> 1;
    output_data[in_i] = (uint8_t) (exp_bit | (sign >> (s_shift - pack_s_shift)) | mantissa);
  }
}

template <typename T>
__global__ void _GistPackMsfp15DecoderKernel(
  const uint8_t* input_data,
  T* output_data,
  const CUDA_LONG num_threads,
  const CUDA_LONG pre_axis_size,
  const CUDA_LONG axis_size,
  const CUDA_LONG num_tiles,
  const CUDA_LONG tile_size) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, num_threads);

  // Quantization parameters
  const int bits = 7;
  // mantissa bits, remove sign
  const int mbits = bits - 1;

  const int s_shift = 31;
  const int pack_s_shift = 6;
  const uint8_t pack_s_mask = 0x40;
  const int e_shift = 23;
  const int pack_e_shift = 7;
  const uint8_t pack_m_mask = 0x3f;

  const int tile_i = id % num_tiles;
  const int pre_axis_i = id / num_tiles;

  // Extract exponent 
  uint32_t shared_exp = 0;
  for (int i = 7; i >= 0; i--) {
    size_t in_i = pre_axis_i * axis_size +
                  tile_i * tile_size +
                  i;
    shared_exp = shared_exp << 1;
    shared_exp += (input_data[in_i] >> pack_e_shift);
  }

  // De-quantize values
  for (size_t i = 0; i < tile_size; i++) {
    size_t in_i = pre_axis_i * axis_size +
                  tile_i * tile_size +
                  i;
    uint8_t X = input_data[in_i];
    // Get sign bit
    uint32_t sign = X & pack_s_mask;
    // Get mantissa
    uint32_t mantissa = (uint32_t) (X & pack_m_mask);

    if (mantissa == 0) {
      output_data[in_i] = 0.0;
    } else {
      // Find leading 1
      uint8_t leading_bit_pos = floorf(log2f(mantissa));
      // Difference from shared exponent of this value
      int exp_diff = 5 - leading_bit_pos;
      // Adjust exponent
      uint32_t exp = shared_exp - exp_diff;

      // Shift back to restore mantissa
      mantissa = mantissa << (24 - mbits + exp_diff);
      // Remove implied 1
      mantissa = mantissa & ((1 << 23) - 1);

      // Reconstruct float number
      uint32_t output =  (sign << (s_shift - pack_s_shift)) | (exp << e_shift) | mantissa;
      output_data[in_i] = (float)__uint_as_float(output);
    }
  }
}

template <typename T>
void GistBinarizeEncoderImpl(
    cudaStream_t stream,
    const T* input_data,
    bool* output_data,
    const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _GistBinarizeEncoderKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(input_data, output_data, (CUDA_LONG)N);
}

template <typename T>
void GistBinarizeDecoderImpl(
    cudaStream_t stream,
    const bool* input_data,
    T* output_data,
    const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _GistBinarizeDecoderKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(input_data, output_data, (CUDA_LONG)N);
}

template <typename T>
void GistPack1EncoderImpl(
    cudaStream_t stream,
    const T* input_data,
    uint8_t* output_data,
    const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  cudaMemset(output_data, 0, N);
  _GistPack1EncoderKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(input_data, output_data, GIST_PACK1_FACTOR, (CUDA_LONG)N);
}

template <typename T>
void GistPack1DecoderImpl(
    cudaStream_t stream,
    const uint8_t* input_data,
    T* output_data,
    const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  _GistPack1DecoderKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(input_data, output_data, GIST_PACK1_FACTOR, (CUDA_LONG)N);
}

template <typename T>
void GistPack8EncoderImpl(
    cudaStream_t stream,
    const T* input_data,
    uint8_t* output_data,
    const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  
  _GistPack8EncoderKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(input_data, output_data, (CUDA_LONG)N);
}

template <typename T>
void GistPack8DecoderImpl(
    cudaStream_t stream,
    const uint8_t* input_data,
    T* output_data,
    const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  _GistPack8DecoderKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(input_data, output_data, (CUDA_LONG)N);
}

template <typename T>
void GistPack16EncoderImpl(
    cudaStream_t stream,
    const T* input_data,
    half* output_data,
    const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  _GistPack16EncoderKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(input_data, output_data, (CUDA_LONG)N);
}

template <typename T>
void GistPack16DecoderImpl(
    cudaStream_t stream,
    const half* input_data,
    T* output_data,
    const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));

  _GistPack16DecoderKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(input_data, output_data, (CUDA_LONG)N);
}

template <typename T>
void GistPackMsfp15EncoderImpl(
    cudaStream_t stream,
    const T* input_data,
    uint8_t* output_data,
    const size_t pre_axis_size,
    const size_t axis_size,
    const size_t tile_size) {

  assert(axis_size % tile_size == 0);
  const int num_tiles = axis_size / tile_size;

  const int threads = pre_axis_size * num_tiles;

  int blocksPerGrid = (int)(ceil(static_cast<float>(threads) / GridDim::maxThreadsPerBlock));
  _GistPackMsfp15EncoderKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
    input_data, 
    output_data, 
    (CUDA_LONG)threads,
    (CUDA_LONG)pre_axis_size,
    (CUDA_LONG)axis_size,
    (CUDA_LONG)num_tiles,
    (CUDA_LONG)tile_size
  );
}

template <typename T>
void GistPackMsfp15DecoderImpl(
  cudaStream_t stream,
  const uint8_t* input_data,
  T* output_data,
  const size_t pre_axis_size,
  const size_t axis_size,
  const size_t tile_size) {

  assert(axis_size % tile_size == 0);
  const int num_tiles = axis_size / tile_size;

  const int threads = pre_axis_size * num_tiles;

  int blocksPerGrid = (int)(ceil(static_cast<float>(threads) / GridDim::maxThreadsPerBlock));
  _GistPackMsfp15DecoderKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
    input_data,
    output_data,
    (CUDA_LONG)threads,
    (CUDA_LONG)pre_axis_size,
    (CUDA_LONG)axis_size,
    (CUDA_LONG)num_tiles,
    (CUDA_LONG)tile_size
  );
}

#define SPECIALIZED_IMPL_BIN_ENC(T) \
  template void GistBinarizeEncoderImpl<T>(cudaStream_t stream, const T* input_data, bool* output_data, const size_t N);
#define SPECIALIZED_IMPL_BIN_DEC(T) \
  template void GistBinarizeDecoderImpl<T>(cudaStream_t stream, const bool* input_data, T* output_data, const size_t N);
#define SPECIALIZED_IMPL_PACK1_ENC(T) \
  template void GistPack1EncoderImpl<T>(cudaStream_t stream, const T* input_data, uint8_t* output_data, const size_t N);
#define SPECIALIZED_IMPL_PACK1_DEC(T) \
  template void GistPack1DecoderImpl<T>(cudaStream_t stream, const uint8_t* input_data, T* output_data, const size_t N);
#define SPECIALIZED_IMPL_PACK8_ENC(T) \
  template void GistPack8EncoderImpl<T>(cudaStream_t stream, const T* input_data, uint8_t* output_data, const size_t N);
#define SPECIALIZED_IMPL_PACK8_DEC(T) \
  template void GistPack8DecoderImpl<T>(cudaStream_t stream, const uint8_t* input_data, T* output_data, const size_t N);
#define SPECIALIZED_IMPL_PACK16_ENC(T) \
  template void GistPack16EncoderImpl<T>(cudaStream_t stream, const T* input_data, half* output_data, const size_t N);
#define SPECIALIZED_IMPL_PACK16_DEC(T) \
  template void GistPack16DecoderImpl<T>(cudaStream_t stream, const half* input_data, T* output_data, const size_t N);
#define SPECIALIZED_IMPL_PACKMSFP15_ENC(T) \
  template void GistPackMsfp15EncoderImpl<T>(cudaStream_t stream, const T* input_data, uint8_t* output_data, const size_t pre_axis_size, const size_t axis_size, const size_t tile_size);
#define SPECIALIZED_IMPL_PACKMSFP15_DEC(T) \
  template void GistPackMsfp15DecoderImpl<T>(cudaStream_t stream, const uint8_t* input_data, T* output_data, const size_t pre_axis_size, const size_t axis_size, const size_t tile_size);

SPECIALIZED_IMPL_BIN_ENC(float)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
SPECIALIZED_IMPL_BIN_ENC(half)
#endif
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
SPECIALIZED_IMPL_BIN_ENC(double)
#endif

SPECIALIZED_IMPL_BIN_DEC(float)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
SPECIALIZED_IMPL_BIN_DEC(half)
#endif
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
SPECIALIZED_IMPL_BIN_DEC(double)
#endif

SPECIALIZED_IMPL_PACK1_ENC(bool)
SPECIALIZED_IMPL_PACK1_ENC(float)

SPECIALIZED_IMPL_PACK1_DEC(bool)
SPECIALIZED_IMPL_PACK1_DEC(float)

SPECIALIZED_IMPL_PACK8_ENC(float)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
SPECIALIZED_IMPL_PACK8_ENC(half)
#endif

SPECIALIZED_IMPL_PACK8_DEC(float)
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
SPECIALIZED_IMPL_PACK8_DEC(half)
#endif

SPECIALIZED_IMPL_PACK16_ENC(float)

SPECIALIZED_IMPL_PACK16_DEC(float)

SPECIALIZED_IMPL_PACKMSFP15_ENC(float)

SPECIALIZED_IMPL_PACKMSFP15_DEC(float)

}  // namespace cuda
}  // namespace onnxruntime
