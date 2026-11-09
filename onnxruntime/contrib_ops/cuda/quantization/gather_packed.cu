// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/gather_packed.h"
#include "contrib_ops/cuda/quantization/gguf_quant_cuda.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

template <typename CudaT, typename Tind>
__global__ __launch_bounds__(256) void GatherPackedKernel(
    const uint8_t* __restrict__ data,
    const Tind* __restrict__ indices,
    CudaT* __restrict__ output,
    int64_t num_rows,
    int64_t vocab_size,
    int k,
    int64_t row_bytes,
    int ggml_type) {
  const int64_t r = static_cast<int64_t>(blockIdx.x);
  if (r >= num_rows) return;
  const int64_t raw_id = static_cast<int64_t>(indices[r]);
  const int64_t tok = (raw_id >= 0 && raw_id < vocab_size) ? raw_id : 0;
  const uint8_t* row_ptr = data + tok * row_bytes;
  CudaT* dst_row = output + r * static_cast<int64_t>(k);

  // Unquantized rows can end partway through a sub-block.
  if ((k % 32) != 0) {
    for (int col = threadIdx.x; col < k; col += blockDim.x) {
      float value;
      if (ggml_type == gguf_quant::kGgmlTypeF32) {
        value = __uint_as_float(gguf_cuda::ReadU32Dev(row_ptr + static_cast<int64_t>(col) * 4));
      } else {
        const uint16_t bits = gguf_cuda::ReadU16Dev(row_ptr + static_cast<int64_t>(col) * 2);
        value = ggml_type == gguf_quant::kGgmlTypeF16 ? gguf_cuda::Fp16ToFloatDev(bits)
                                                      : gguf_cuda::Bf16ToFloatDev(bits);
      }
      dst_row[col] = gguf_cuda::FloatToCudaType<CudaT>(value);
    }
    return;
  }

  const int total_quarters = k / 8;
  for (int q_idx = threadIdx.x; q_idx < total_quarters; q_idx += blockDim.x) {
    float vals[8];
    gguf_cuda::DecodeGgufSubBlock8(row_ptr, ggml_type, q_idx >> 2, q_idx & 3, vals);
    CudaT* dst8 = dst_row + q_idx * 8;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      dst8[j] = gguf_cuda::FloatToCudaType<CudaT>(vals[j]);
    }
  }
}

template <typename CudaT>
Status DispatchGatherPackedByIndex(
    cudaStream_t stream,
    const uint8_t* data,
    const void* indices,
    bool indices_are_int64,
    CudaT* output,
    int64_t num_rows,
    int64_t vocab_size,
    int64_t k,
    int64_t row_bytes,
    int ggml_type) {
  ORT_RETURN_IF_NOT(k > 0 && k <= kGatherPackedMaxK, "GatherPacked requires K in 1..", kGatherPackedMaxK);
  ORT_RETURN_IF_NOT(num_rows >= 0 && num_rows <= kGatherPackedMaxRows,
                    "GatherPacked index count exceeds the CUDA grid limit");
  if (num_rows == 0) return Status::OK();
  ORT_RETURN_IF_NOT(vocab_size > 0 && row_bytes > 0, "GatherPacked requires nonempty packed rows");
  const dim3 grid(static_cast<unsigned int>(num_rows));
  const dim3 block(256);
  if (indices_are_int64) {
    GatherPackedKernel<CudaT, int64_t><<<grid, block, 0, stream>>>(
        data, static_cast<const int64_t*>(indices), output, num_rows, vocab_size, static_cast<int>(k), row_bytes,
        ggml_type);
  } else {
    GatherPackedKernel<CudaT, int32_t><<<grid, block, 0, stream>>>(
        data, static_cast<const int32_t*>(indices), output, num_rows, vocab_size, static_cast<int>(k), row_bytes,
        ggml_type);
  }
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace

Status LaunchGatherPackedCuda(
    cudaStream_t stream,
    const uint8_t* data,
    const void* indices,
    bool indices_are_int64,
    void* output,
    int output_dtype,
    int64_t num_rows,
    int64_t vocab_size,
    int64_t k,
    int64_t row_bytes,
    int ggml_type) {
  if (output_dtype == 16) {
    return DispatchGatherPackedByIndex<__nv_bfloat16>(
        stream, data, indices, indices_are_int64, static_cast<__nv_bfloat16*>(output), num_rows, vocab_size, k,
        row_bytes, ggml_type);
  } else if (output_dtype == 10) {
    return DispatchGatherPackedByIndex<half>(
        stream, data, indices, indices_are_int64, static_cast<half*>(output), num_rows, vocab_size, k, row_bytes,
        ggml_type);
  } else if (output_dtype == 1) {
    return DispatchGatherPackedByIndex<float>(
        stream, data, indices, indices_are_int64, static_cast<float*>(output), num_rows, vocab_size, k, row_bytes,
        ggml_type);
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported GatherPacked output_dtype=", output_dtype);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
