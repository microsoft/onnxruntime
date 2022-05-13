#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

#include "qorder_binary_op.cuh"
#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"

#include "core/providers/cuda/cu_inc/common.cuh"

using onnxruntime::cuda::BinaryElementWiseImpl;

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

#define QORDERED_BINARY_THREE_SCALE_IMPL(name)                               \
  QORDERED_BINARY_THREE_SCALE_DECLARATION(name) {                            \
    BinaryElementWiseImpl(stream,                                            \
                          output_rank_or_simple_broadcast,                   \
                          lhs_padded_strides,                                \
                          lhs_data,                                          \
                          rhs_padded_strides,                                \
                          rhs_data,                                          \
                          fdm_output_strides,                                \
                          fdm_H,                                             \
                          fdm_C,                                             \
                          output_data,                                       \
                          QORDERED_OP_##name(lhs_scale, rhs_scale, y_scale), \
                          count);                                            \
  }

#define QORDERED_BINARY_TWO_SCALE_IMPL(name)                        \
  QORDERED_BINARY_TWO_SCALE_DECLARATION(name) {                     \
    BinaryElementWiseImpl(stream,                                   \
                          output_rank_or_simple_broadcast,          \
                          lhs_padded_strides,                       \
                          lhs_data,                                 \
                          rhs_padded_strides,                       \
                          rhs_data,                                 \
                          fdm_output_strides,                       \
                          fdm_H,                                    \
                          fdm_C,                                    \
                          output_data,                              \
                          QORDERED_OP_##name(lhs_scale, rhs_scale), \
                          count);                                   \
  }

struct QORDERED_OP_Add {
  float scaleA_;
  float scaleB_;
  QORDERED_OP_Add(float scaleA, float scaleB)
      : scaleA_(scaleA), scaleB_(scaleB) {
  }
  __device__ __inline__ int8_t operator()(int8_t a, int8_t b) const {
    float v = scaleA_ * a + scaleB_ * b;
    v = fmaxf(fminf(v, 127.0f), -128.0f);
    return static_cast<int8_t>(lrintf(v));
  }
};

// QORDERED_BINARY_TWO_SCALE_IMPL(Add);

// constants for approximating the normal cdf
constexpr float A = 0.5f;
constexpr float B = 0.7978845608028654f;    // sqrt(2.0/M_PI)
constexpr float C = 0.035677408136300125f;  // 0.044715 * sqrt(2.0/M_PI)

struct QORDERED_OP_BiasGelu {
  float scaleA_;
  float scaleB_;
  float scaleY_A_;

  QORDERED_OP_BiasGelu(float scaleA, float scaleB, float scaleY)
      : scaleA_(scaleA), scaleB_(scaleB), scaleY_A_(scaleY * A) {
  }

  __device__ __inline__ int8_t operator()(int8_t a, int8_t b) const {
    float x = scaleA_ * a + scaleB_ * b;
    float y = x * (scaleY_A_ + scaleY_A_ * tanhf(x * (C * x * x + B)));
    y = fmaxf(fminf(y, 127.0f), -128.0f);
    return static_cast<int8_t>(lrintf(y));
  }
};

// QORDERED_BINARY_THREE_SCALE_IMPL(BiasGelu);

constexpr int kNumElementsPerThread = GridDim::maxElementsPerThread;
constexpr int kNumThreadsPerBlock = GridDim::maxThreadsPerBlock;
constexpr int kNumElementsPerBlockLine = sizeof(char4) * kNumThreadsPerBlock;
constexpr int kNumElementsPerBlock = sizeof(char4) * kNumElementsPerThread * kNumThreadsPerBlock;

// Assumes that `input_data`, `output_data`, and `bias_data` are all in COL32 data format
__global__ void _QOrdered_Col32OrderKernel_BiasGelu(
    const int8_t* input_data,
    half2 half2_input_scale,
    const int8_t* bias_data,
    half2 half2_bias_scale,
    int8_t* output_data,
    half2 half2_inverse_output_scale,
    const int64_t batches,
    const int64_t rows,
    const int64_t cols) {
  union U1S2 {
    unsigned u1;
    short2 s2;
    char4 c4;
  } u1s2;

  char4 i4;
  char4 b4;

  const int row_offset = (blockIdx.y * rows * cols) + (blockIdx.x << 5);
  unsigned col_stride = (blockDim.x << 2);
  unsigned c = (threadIdx.x << 2);

  while (c < cols) {
    auto id = row_offset + (c >> 5) * rows * 32 + (c & 31);

    // Process 4 quantized int8_t elements by processing them as 4 halfs (2 half2 s)
    i4 = *(const char4*)(input_data + id);
    b4 = *(const char4*)(bias_data + c);

    // Convert float constants to half2's for computation
    half2 half2_A = __float2half2_rn(A);
    half2 half2_B = __float2half2_rn(B);
    half2 half2_C = __float2half2_rn(C);

    // Process 1st half2
    half2 low2_data = __halves2half2(__short2half_rn((short)i4.x), __short2half_rn((short)i4.y));
    half2 low2_bias = __halves2half2(__short2half_rn((short)b4.x), __short2half_rn((short)b4.y));
    low2_data = low2_data * half2_input_scale + low2_bias * half2_bias_scale;
    low2_data = low2_data * (half2_A + half2_A * _Tanh(low2_data * (half2_C * low2_data * low2_data + half2_B))) * half2_inverse_output_scale;

    // Process 2nd half2
    half2 high2_data = __halves2half2(__short2half_rn((short)i4.z), __short2half_rn((short)i4.w));
    half2 high2_bias = __halves2half2(__short2half_rn((short)b4.z), __short2half_rn((short)b4.w));
    high2_data = high2_data * half2_input_scale + high2_bias * half2_bias_scale;
    high2_data = high2_data * (half2_A + half2_A * _Tanh(high2_data * (half2_C * high2_data * high2_data + half2_B))) * half2_inverse_output_scale;

    // Clamp 1st half2 results
    u1s2.s2.x = __half2short_rn(low2_data.x);
    u1s2.s2.y = __half2short_rn(low2_data.y);
    u1s2.u1 = __vmaxs2(__vmins2(u1s2.u1, 0x007F007F), 0xFF80FF80);
    i4.x = u1s2.c4.x;
    i4.y = u1s2.c4.z;

    // Clamp 2nd half2 results
    u1s2.s2.x = __half2short_rn(high2_data.x);
    u1s2.s2.y = __half2short_rn(high2_data.y);
    u1s2.u1 = __vmaxs2(__vmins2(u1s2.u1, 0x007F007F), 0xFF80FF80);
    i4.z = u1s2.c4.x;
    i4.w = u1s2.c4.z;

    // Write results to output indices
    *(char4*)(output_data + id) = i4;

    // Update c
    c += col_stride;
  }
}

void QOrdered_Col32OrderImpl_BiasGelu(
    cudaStream_t stream,
    const int8_t* input_tensor,
    float input_scale,
    const int8_t* bias_tensor,
    float bias_scale,
    int8_t* output_tensor,
    float output_scale,
    const int64_t batches,
    const int64_t rows,
    const int64_t cols) {
  constexpr unsigned tpb = 128;

  const dim3 blocks((unsigned)rows, (unsigned)batches, 1);
  const dim3 threads((unsigned)tpb, 1, 1);

  half2 half2_input_scale = __float2half2_rn(input_scale);
  half2 half2_bias_scale = __float2half2_rn(bias_scale);
  half2 half2_inverse_output_scale = __float2half2_rn(1.0f / output_scale);

  _QOrdered_Col32OrderKernel_BiasGelu<<<blocks, threads, 0, stream>>>(
      input_tensor, half2_input_scale, bias_tensor, half2_bias_scale, output_tensor,
      half2_inverse_output_scale, batches, rows, cols);
}

// Assumes that `input_data`, `output_data`, and `bias_data` are all in COL32 data format
__global__ void _QOrdered_Col32OrderKernel_Add(
    const int8_t* input_data,
    half2 half2_input_scale,
    const int8_t* bias_data,
    half2 half2_bias_scale,
    int8_t* output_data,
    half2 half2_inverse_output_scale,
    const fast_divmod batch_size,
    const fast_divmod rows_times_thirty_two,
    const CUDA_LONG count) {
  CUDA_LONG id = kNumElementsPerBlock * blockIdx.x + threadIdx.x * (CUDA_LONG)sizeof(char4);

  union U1S2 {
    unsigned u1;
    short2 s2;
    char4 c4;
  } u1s2;

  char4 i4;
  char4 b4;

#pragma unroll
  for (int line = 0; line < kNumElementsPerThread; line++) {
    if (id < count) {
      // Calculate the start bias index to process 4 inputs starting from `id`
      // As the kernel name indicates, assumes the input data is in COL32 format
      auto bias_id = batch_size.mod(id);
      int q, r;
      rows_times_thirty_two.divmod(bias_id, q, r);
      bias_id = (q << 5) + (r & 31);

      // Process 4 quantized int8_t elements by processing them as 4 halfs (2 half2 s)
      i4 = *(const char4*)(input_data + id);
      b4 = *(const char4*)(bias_data + bias_id);

      // Process 1st half2
      half2 low2_data = __halves2half2(__short2half_rn((short)i4.x), __short2half_rn((short)i4.y));
      half2 low2_bias = __halves2half2(__short2half_rn((short)b4.x), __short2half_rn((short)b4.y));
      low2_data = low2_data * half2_input_scale + low2_bias * half2_bias_scale;

      // Process 2nd half2
      half2 high2_data = __halves2half2(__short2half_rn((short)i4.z), __short2half_rn((short)i4.w));
      half2 high2_bias = __halves2half2(__short2half_rn((short)b4.z), __short2half_rn((short)b4.w));
      high2_data = high2_data * half2_input_scale + high2_bias * half2_bias_scale;

      // Clamp 1st half2 results
      u1s2.s2.x = __half2short_rn(low2_data.x);
      u1s2.s2.y = __half2short_rn(low2_data.y);
      u1s2.u1 = __vmaxs2(__vmins2(u1s2.u1, 0x007F007F), 0xFF80FF80);
      i4.x = u1s2.c4.x;
      i4.y = u1s2.c4.z;

      // Clamp 2nd half2 results
      u1s2.s2.x = __half2short_rn(high2_data.x);
      u1s2.s2.y = __half2short_rn(high2_data.y);
      u1s2.u1 = __vmaxs2(__vmins2(u1s2.u1, 0x007F007F), 0xFF80FF80);
      i4.z = u1s2.c4.x;
      i4.w = u1s2.c4.z;

      // Write results to output indices
      *(char4*)(output_data + id) = i4;

      // Update input index for next iteration
      id += kNumElementsPerBlockLine;
    }
  }
}

void QOrdered_Col32OrderImpl_Add(
    cudaStream_t stream,
    const int8_t* input_tensor,
    float input_scale,
    const int8_t* bias_tensor,
    float bias_scale,
    int8_t* output_tensor,
    float output_scale,
    const fast_divmod& batch_size,
    const fast_divmod& rows_times_thirty_two,
    size_t count) {
  if (count == 0) {
    return;
  }

  int blocksPerGrid = static_cast<int>(CeilDiv(count, kNumElementsPerBlock));

  half2 half2_input_scale = __float2half2_rn(input_scale);
  half2 half2_bias_scale = __float2half2_rn(bias_scale);
  half2 half2_inverse_output_scale = __float2half2_rn(1.0f / output_scale);

  _QOrdered_Col32OrderKernel_Add<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input_tensor, half2_input_scale, bias_tensor, half2_bias_scale, output_tensor,
      half2_inverse_output_scale, batch_size, rows_times_thirty_two, (CUDA_LONG)count);
}
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
