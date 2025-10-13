/*++

// TODO: finish filling this out

module includes kernel functions for generating LUT for T-MAC GEMM optimization strategy.
*/

#include "qlutgemm.h"
#include <cstddef>
#include <cstdio>
#include <cassert>

bool MLASCALL MlasIsTMACAvailable(
    size_t /*BlkBitWidth*/,
    size_t /*BlkLen*/
)
{
    const auto* Dispatch = GetMlasPlatform().LUTGenKernel;
    return Dispatch != nullptr;
    // return Dispatch != nullptr && BlkLen == 4; // only support group sizes of 4 for now
}

size_t CalculateLUTSize(int k, int m, size_t group_size) {
    size_t lut_scales_size = k / group_size;
    size_t wsize = k * m * 4 * sizeof(int8_t) + lut_scales_size * m * 2 * sizeof(float);
    // if (sizeof(tmac_float_type) == 2) {
        // Need fp32 to fp16 conversion
        // TODO: do we...?
        // wsize += std::max(k, n) * m * sizeof(_Float16);
    // }
    wsize = ((wsize - 1) / 64 + 1) * 64;
    return wsize;
}

void MLASCALL MlasTmac(
    const void* A,
    size_t BlkLen,
    const void* QuantBData,     // Quantized weights (B matrix)
    const float* QuantBScale,   // scale(s) for quantized weights
    void* C,
    int K,
    int M,                // batch size (number of rows in activation)
    int N,
    MLAS_THREADPOOL* threadpool
) {
    // adapted from ggml_backend_tmac_mul_mat
    const auto* Dispatch = GetMlasPlatform().LUTGenKernel;
    if (!Dispatch || !Dispatch->GenerateLUT) {
        ORT_THROW("TMAC not supported in this configuration.");
    }

    size_t lut_size = CalculateLUTSize(K, M, BlkLen);
    auto lut_buffer = std::make_unique<uint8_t[]>(lut_size);

    const size_t lut_meta_size = (K / BlkLen) * M;
    auto biases_float = std::make_unique<float[]>(lut_meta_size);
    auto scales_float = std::make_unique<float[]>(lut_meta_size);

    const auto* a_float = reinterpret_cast<const float*>(A);  // Activation data

    const int num_groups = static_cast<int>(K / BlkLen);

    // Parallelize over M (batch dimension)
    // Each iteration processes one row of the activation matrix
    MlasTrySimpleParallel(
        threadpool, 
        static_cast<size_t>(M),
        [&](ptrdiff_t ine11) {
            const size_t row_offset = static_cast<size_t>(ine11) * K;
            const size_t lut_offset = static_cast<size_t>(ine11) * K * 4;  // 4 bytes per K element for 2-bit LUT
            const size_t scale_bias_offset = static_cast<size_t>(ine11) * num_groups;

            // Call the dispatch function for this row
            Dispatch->GenerateLUT(
                static_cast<int32_t>(BlkLen), 
                reinterpret_cast<int8_t*>(lut_buffer.get()) + lut_offset,  // Output LUT for this row
                a_float + row_offset,                     // Input activation for this row
                scales_float.get() + scale_bias_offset,   // Scales for this row
                biases_float.get() + scale_bias_offset,   // Biases for this row
                K
            );
        }
    );

    // all relevant LUT's have been generated
    // equivalent of lut_mul_mat's ggml_backend_tmac_mul_mat function ggml_barrier line
    const size_t bits = 2;  // TODO: parameterize if needed
    const size_t bm = 64; // TODO: hardcoding for now

    // TODO: fix the below 4
    // Matrix multiplication: Output[N×M] = QuantBData[N×K] × Weights[K×M]
    const size_t OutputRows = M;    // Number of output features
    const size_t OutputCols = N;    // Batch size
    const size_t NumTiles = M * bits / bm;

    const size_t ChunkSize0 = M / NumTiles;
    const size_t ChunkSize1 = 8; // process one batch item at a time

// In llama.cpp terminology (note the swap!):
// ne0 = M (output features, called "n" in llama.cpp)
// ne1 = N (batch size, called "m" in llama.cpp)

    // Calculate number of chunks in each dimension
    const size_t nchunk0 = (OutputRows + ChunkSize0 - 1) / ChunkSize0;  // Should equal NumTiles
    const size_t nchunk1 = (OutputCols + ChunkSize1 - 1) / ChunkSize1;
    const size_t total_chunks = nchunk0 * nchunk1;

    // Pre-calculate sizes for offset calculations
    const size_t w_size = OutputRows * K * bits / 8;
    const size_t w_chunk_size = w_size / NumTiles;

    // Determine weight-scale layout. These should be provided by the caller or inferred from the packed weights.
    // For now we default to per-group symmetric quantization (no zero-point, not one-scale).
    bool one_scale = false;            // TODO: expose this as a function parameter if needed
    bool has_zero_point = false;       // TODO: expose this as a function parameter if needed

    // Total number of scale (float) entries for the whole weight matrix:
    // - if one_scale: single global scale (1)
    // - otherwise: number of quantization groups = (M * K / BlkLen)
    //   and if zero-points are present each group stores (scale, zero_point) -> *2
    const size_t groups_total = static_cast<size_t>(M) * static_cast<size_t>(K) / BlkLen;
    const size_t scales_size_total = one_scale ? 1 : (groups_total * (has_zero_point ? 2 : 1));

    // n_tile_num == NumTiles (number of M tiles)
    const size_t n_tile_num = NumTiles;

    // Per-tile scales size = total scales size divided evenly across tiles.
    // If one_scale is true we do not advance the scales pointer per tile, so set per tile size to 0
    size_t scales_size_per_tile = 0;
    if (!one_scale) {
        if (scales_size_total % n_tile_num != 0) {
            // Sanity: scales should partition evenly across tiles. If they don't, choose floor division
            // and document that callers must layout scales accordingly.
            // Prefer to error loudly in debug builds.
            fprintf(stderr, "Warning: scales_size_total=%zu is not divisible by n_tile_num=%zu; using floor division.\n", scales_size_total, n_tile_num);
        }
        scales_size_per_tile = scales_size_total / n_tile_num;
    }

    // Note: when one_scale == true, callers should pass a pointer to a single scale value (scales_offset=0 will be used)

    // Cast to appropriate types
    const auto* packed_weights = reinterpret_cast<const uint8_t*>(QuantBData);
    const int8_t* lut_i8 = reinterpret_cast<const int8_t*>(lut_buffer.get());

    // lut_scales_size is the number of scale values per batch item (= K / BlkLen)
    const size_t lut_scales_size = static_cast<size_t>(K) / BlkLen;

    // Parallelize over the 2D chunk grid
    MlasTrySimpleParallel(
        threadpool,
        total_chunks,
        [&](ptrdiff_t current_chunk) {
            // Decompose linear chunk index into 2D coordinates
            const size_t ith0 = current_chunk % nchunk0;  // Chunk in dimension 0 (output rows)
            const size_t ith1 = current_chunk / nchunk0;  // Chunk in dimension 1 (batch)

            // Calculate ranges for this chunk
            const size_t ir0_start = ChunkSize0 * ith0;
            const size_t ir0_end = std::min(ir0_start + ChunkSize0, OutputRows);

            const size_t ir1_start = ChunkSize1 * ith1;
            const size_t ir1_end = std::min(ir1_start + ChunkSize1, OutputCols);

            // Process all tiles in dimension 0 for this chunk
            for (size_t ichunk0 = ir0_start / ChunkSize0; ichunk0 < ir0_end / ChunkSize0; ichunk0++) {
                // Calculate weight offsets
                const size_t w_offset = ichunk0 * w_chunk_size;
                const size_t scales_offset = ichunk0 * scales_size_per_tile;

                // Process all batch items in this chunk
                for (size_t ine11 = ir1_start; ine11 < ir1_end; ine11++) {
                    // Calculate LUT offsets for this batch item
                    const size_t qlut_offset = K * ine11 * 4;
                    const size_t lut_scales_offset = lut_scales_size * ine11;
                    
                    // Calculate output offset
                    const size_t dst_offset = OutputRows * ine11 + ichunk0 * ChunkSize0;

                    // Call the dispatch function to compute this tile
                    Dispatch->ComputeGemm(
                        const_cast<void*>(reinterpret_cast<const void*>(packed_weights + w_offset)),  // Weight tile
                        QuantBScale + scales_offset,                    // Weight scales for this tile
                        const_cast<void*>(reinterpret_cast<const void*>(lut_i8 + qlut_offset)),      // LUT for this batch row
                        scales_float.get() + lut_scales_offset,         // LUT scales
                        biases_float.get() + lut_scales_offset,         // LUT biases
                        reinterpret_cast<uint8_t*>(C) + dst_offset,     // Output location
                        static_cast<int>(bm),                           // bm
                        static_cast<int>(K),                            // K dimension
                        static_cast<int>(M),                            // K dimension
                        static_cast<int>(N),                            // N dimension (batch size)
                        BlkLen                                          // Weight quantization group size
                    );
                }
            }
        }
    );
}