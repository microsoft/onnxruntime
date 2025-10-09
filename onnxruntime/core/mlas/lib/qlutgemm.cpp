/*++

// TODO: finish filling this out

module includes kernel functions for generating LUT for T-MAC GEMM optimization strategy.
*/

#include "qlutgemm.h"
#include <cstddef>

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

bool MLASCALL MlasTmac(
    void* A,
    size_t BlkLen,
    void* QuantBData,     // B in float (activation data)
    float* QuantBScale,   // scale(s) in float
    int K,
    int M,                // batch size (number of rows in activation)
    int N,
    void* lut_biases,     // Add biases output
    MLAS_THREADPOOL* threadpool
) {
    // adapted from ggml_backend_tmac_mul_mat
    const auto* Dispatch = GetMlasPlatform().LUTGenKernel;
    if (!Dispatch || !Dispatch->GenerateLUT) return false;

    size_t lut_size = CalculateLUTSize(K, M, BlkLen);
    auto lut_buffer = std::make_unique<std::byte[]>(lut_size);

    auto* b_float = reinterpret_cast<float*>(QuantBData);
    auto* biases_float = reinterpret_cast<float*>(lut_biases);

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
                lut_buffer + lut_offset,                      // Output LUT for this row
                b_float + row_offset,                     // Input activation for this row
                QuantBScale + scale_bias_offset,          // Scales for this row
                biases_float + scale_bias_offset,         // Biases for this row
                K
            );
        }
    );=
    

    // all relevant LUT's have been generated
    // equivalent of lut_mul_mat's ggml_backend_tmac_mul_mat function ggml_barrier line

    // TODO: fix the below 4
    // Matrix multiplication: Output[N×M] = QuantBData[N×K] × Weights[K×M]
    const size_t OutputRows = M;    // Number of output features
    const size_t OutputCols = N;    // Batch size

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
    const size_t bits = 2;  // TODO: parameterize if needed
    const size_t w_size = OutputRows * K * bits / 8;
    const size_t w_chunk_size = w_size / NumTiles;
    const size_t scales_size_per_tile = /* calculate based on your config */;
    const size_t lut_scales_size = K / BlkLen;  // Assuming BlkLen is act_group_size

    // Cast to appropriate types
    const auto* packed_weights = reinterpret_cast<const uint8_t*>(PackedWeights);
    const auto* lut_i8 = reinterpret_cast<const int8_t*>(qlut_buffer);

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
                    Dispatch->ComputeGEMM(
                        packed_weights + w_offset,           // Weight tile
                        WeightScales + scales_offset,        // Weight scales for this tile
                        lut_i8 + qlut_offset,               // LUT for this batch row
                        lut_scales + lut_scales_offset,     // LUT scales
                        lut_biases + lut_scales_offset,     // LUT biases
                        Output + dst_offset,                // Output location // n, k, m, bits
                        // bm, k, m, kernel config
                        64, // hardcoding bm as 64 for now...
                        static_cast<int32_t>(K),            // K dimension
                        static_cast<int32_t>(N),            // N dimension
                        BlkLen
                    );
                }
            }
        }
    );
    return true;
}