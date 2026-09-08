// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

void LaunchSoftmaxTopK(
    const float* logits,
    float* topk_scales,
    int* topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales,
    cudaStream_t stream);

void LaunchSoftmaxTopK(
    const half* logits,
    float* topk_scales,
    int* topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales,
    cudaStream_t stream);

void LaunchSoftmaxTopK(
    const __nv_bfloat16* logits,
    float* topk_scales,
    int* topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales,
    cudaStream_t stream);

void LaunchSparseMixerTop2(
    const float* input,
    float* output,
    int* indices,
    int* source_rows,
    int num_rows,
    int num_experts,
    cudaStream_t stream);

void LaunchSparseMixerTop2(
    const half* input,
    float* output,
    int* indices,
    int* source_rows,
    int num_rows,
    int num_experts,
    cudaStream_t stream);

void LaunchSparseMixerTop2(
    const __nv_bfloat16* input,
    float* output,
    int* indices,
    int* source_rows,
    int num_rows,
    int num_experts,
    cudaStream_t stream);

void LaunchQMoEPrePackZP(
    const uint8_t* zp,
    const float* scales,
    float* output,
    int num_elements,
    cudaStream_t stream);

void LaunchQMoEPrePackZP(
    const uint8_t* zp,
    const half* scales,
    half* output,
    int num_elements,
    cudaStream_t stream);

void LaunchQMoEPrePackZP(
    const uint8_t* zp,
    const __nv_bfloat16* scales,
    __nv_bfloat16* output,
    int num_elements,
    cudaStream_t stream);

void LaunchQMoEPrePackPacked4BitZPKernel(
    const uint8_t* packed_zp,
    const float* scales,
    float* output,
    int num_elements,
    int N,
    cudaStream_t stream);

void LaunchQMoEPrePackPacked4BitZPKernel(
    const uint8_t* packed_zp,
    const half* scales,
    half* output,
    int num_elements,
    int N,
    cudaStream_t stream);

void LaunchQMoEPrePackPacked4BitZPKernel(
    const uint8_t* packed_zp,
    const __nv_bfloat16* scales,
    __nv_bfloat16* output,
    int num_elements,
    int N,
    cudaStream_t stream);

void LaunchQMoEPrePackOffsetBias(
    const uint8_t* zp,
    const float* scales,
    float* output,
    int num_elements,
    float offset,
    cudaStream_t stream);

void LaunchQMoEPrePackOffsetBias(
    const uint8_t* zp,
    const half* scales,
    half* output,
    int num_elements,
    float offset,
    cudaStream_t stream);

void LaunchQMoEPrePackOffsetBias(
    const uint8_t* zp,
    const __nv_bfloat16* scales,
    __nv_bfloat16* output,
    int num_elements,
    float offset,
    cudaStream_t stream);

// Batched 4-bit packed zero-point scaled bias computation.
// ZP layout: [experts, n, packed_k_blocks] (packed_k_blocks = (k_blocks+1)/2)
// Scale/Output layout: [experts, k_blocks, n]
// Computes: output[e][row][col] = scale[e][row][col] * (-unpack4(zp[e][col][row/2]) + default_zp)
void LaunchQMoEScaledZP4BitBatched(
    const uint8_t* packed_zp,
    const half* transposed_scale,
    half* scaled_zero_point,
    int experts, int n, int k_blocks,
    float default_zero_point,
    cudaStream_t stream);

void LaunchQMoEScaledZP4BitBatched(
    const uint8_t* packed_zp,
    const __nv_bfloat16* transposed_scale,
    __nv_bfloat16* scaled_zero_point,
    int experts, int n, int k_blocks,
    float default_zero_point,
    cudaStream_t stream);

void LaunchQMoEShiftWeights(
    const uint8_t* input,
    uint8_t* output,
    int num_elements,
    cudaStream_t stream);

void LaunchQMoETranspose2D(
    const float* input,
    float* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream);

void LaunchQMoETranspose2D(
    const half* input,
    half* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream);

void LaunchQMoETranspose2D(
    const __nv_bfloat16* input,
    __nv_bfloat16* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream);

void LaunchQMoETranspose2D(
    const uint8_t* input,
    uint8_t* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream);

void LaunchQMoEBlockScaleInterleave(
    const uint8_t* input,
    uint8_t* output,
    int batch_size,
    int rows,
    int cols,
    int rows_padded,
    int cols_padded,
    int multi_processor_count,
    cudaStream_t stream);

void LaunchQMoEDequantizeFp4Weights(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    half* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream);

void LaunchQMoEDequantizeFp4Weights(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    __nv_bfloat16* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream);

// Packs MXFP4 e8m0 block scales from [experts, n, k_blocks] into the SM90 TMA WS
// WFP4A16 layout. The currently dispatched native WFP4A16 K tile is 256, so one
// TMA scale element contains 8 adjacent k_blocks for one output row.
void LaunchQMoEPackFp4ScalesForTmaWs(
    const uint8_t* input,
    uint8_t* output,
    int experts,
    int n,
    int k_blocks,
    cudaStream_t stream);

void LaunchQMoEDequantizeFp8Weights(
    const uint8_t* weights,
    const float* global_scales,
    half* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream);

void LaunchQMoEDequantizeFp8Weights(
    const uint8_t* weights,
    const float* global_scales,
    __nv_bfloat16* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream);

// NVFP4 weight dequantization: E2M1 4-bit weights with Float8E4M3FN block scales
// (block size 16) and per-expert float32 global scales. Weight layout [E, K, N/2],
// block-scale layout [E, N, K/16]. Mirrors LaunchQMoEDequantizeFp4Weights (MXFP4).
void LaunchQMoEDequantizeNvfp4Weights(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    half* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream);

void LaunchQMoEDequantizeNvfp4Weights(
    const uint8_t* packed_weights,
    const uint8_t* block_scales,
    const float* global_scales,
    __nv_bfloat16* output,
    int num_experts,
    int n,
    int k,
    cudaStream_t stream);

// Repack column-major FP4 packed weights to row-major layout on GPU.
// Input shape interpretation: [experts, k, n/2] (col-major packed),
// output: [experts, n, k/2] (row-major packed).
// Each byte holds two 4-bit values. k and n must be even.
void LaunchQMoERepackFP4ColToRow(
    const uint8_t* input,
    uint8_t* output,
    int experts,
    int64_t k,
    int64_t n,
    cudaStream_t stream);

void LaunchQMoECombineFp4ScalesForGemv(
    const uint8_t* block_scales,
    const float* global_scales,
    half* output,
    int experts,
    int n,
    int k_blocks,
    cudaStream_t stream);

void LaunchQMoECombineFp4ScalesForGemv(
    const uint8_t* block_scales,
    const float* global_scales,
    __nv_bfloat16* output,
    int experts,
    int n,
    int k_blocks,
    cudaStream_t stream);

void LaunchQMoECombineNvfp4ScalesForGemv(
    const uint8_t* block_scales,
    const float* global_scales,
    half* output,
    int experts,
    int n,
    int k_blocks,
    cudaStream_t stream);

void LaunchQMoECombineNvfp4ScalesForGemv(
    const uint8_t* block_scales,
    const float* global_scales,
    __nv_bfloat16* output,
    int experts,
    int n,
    int k_blocks,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

namespace onnxruntime::llm::kernels {
void LaunchBatchedTranspose(cudaStream_t stream, const void* input, void* output, int batch, int rows, int cols, int element_size);
}
