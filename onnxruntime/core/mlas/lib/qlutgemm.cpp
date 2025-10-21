/*++

// TODO: finish filling this out

module includes kernel functions for generating LUT for T-MAC GEMM optimization strategy.
*/
#include <string>
#include <thread>
#include <unordered_map>

#include "qlutgemm.h"

/** T-MAC GEMM kernel Config */
static std::unordered_map<std::string, struct MlasTMACKernelParams> tmac_kernel_configs;




const MlasTMACKernelParams& GetTMACKernelParams(size_t M, size_t N, size_t nbits, size_t block_size) {
    std::string key = std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(nbits);
    if (tmac_kernel_configs.count(key)) {
        return tmac_kernel_configs[key];
    }

    MlasTMACKernelParams params;
    params.g = 4;
    params.ngroups_per_elem = 8 / params.g;
    params.simd_n_in = 16;
    params.simd_n_out = 8;
    params.chunk_n = 8;

    params.bits = nbits;
    params.q_group_size = block_size;

    if (block_size % 64 == 0) {
        params.act_group_size = 64;
    } else if (block_size % 32 == 0) {
        params.act_group_size = 32;
    } else {
        // throw error
        ORT_THROW("Unsupported activation group size: ", block_size);;
    }
    params.actk = params.act_group_size / params.g;

    //search space
    std::vector<size_t> bms;
    if (nbits == 1 || nbits == 2 || nbits == 4) {
        bms = {256, 512, 1024, 2048, 320, 640, 1280};
    } else if (nbits == 3) {
        bms = {192, 384, 576, 758};
    }

    std::vector<size_t> bns = {8, 16, 32, 64};
    std::vector<size_t> kfactors = {8, 16};

    // TODO: add profile based policy
    int threads = std::thread::hardware_concurrency();

    float smallest_penalty = 1e9;
    for (int bm: bms) {
        if (M % (bm/nbits) != 0 || bm % nbits != 0) {
            continue;
        }
        size_t num_tiles = M/ (bm/nbits);
        size_t num_groups = (num_tiles + threads - 1) / threads;
        float penalty = 0.1 * num_groups + (num_groups - 1.0 * num_tiles / threads) / num_groups;
        if (penalty < smallest_penalty) {
            smallest_penalty = penalty;
            params.bm = bm;
        }
    }

    size_t largest_kfactor = 0;
    for (size_t kfactor: kfactors) {
        if ((kfactor < params.actk) || (kfactor * params.g > params.q_group_size)) {
            continue;
        }
        if (kfactor > largest_kfactor) {
            largest_kfactor = kfactor;
            params.kfactor = kfactor;
        }
    }

    tmac_kernel_configs[key] = params;
    return tmac_kernel_configs[key];
}

void MlasTMACPackScalesAndZeroPoints(
    size_t N,
    size_t K,
    size_t BitWidth,
    size_t BlkLen,
    bool HasZeroPoint,
    float* PackedQuantBZPBegin,
    const float* QuantBScale,
    const uint8_t* QuantBZeroPoint
)
{
    const MlasTMACKernelParams& tmac_params = GetTMACKernelParams(N, K, 2, BlkLen);
    const size_t bits = tmac_params.bits;
    const size_t simd_n_out = tmac_params.simd_n_out;
    const size_t bm = tmac_params.bm;
    const size_t num_elem_per_byte = 8 / bits;


    for (size_t im = 0; im < N ; im += 1) {
        for (size_t ik = 0; ik < K; ik += BlkLen) {
            size_t idx = (im * K + ik) / BlkLen;
            float scale = QuantBScale[idx];
            float zp;
            if (HasZeroPoint) {
                // zp are two bit packed
                size_t elem_idx = idx % num_elem_per_byte;
                uint8_t v = QuantBZeroPoint[idx / num_elem_per_byte] >> (elem_idx * bits) & (1 << bits) - 1;
                zp = static_cast<float>(v);

                // Note: TMAC does this during model conversion. Since, we follow ORT format, we need to do it here.
                // This seems gptq quantization specific.
                // We should either use different op than matmul_nbits or add attribute to matmul_nbits to indicate this.
                zp = zp - (1 << (bits - 1)) - 1;  // make it signed
                zp = zp * scale;              // store scale * zp
            }

            size_t nb1 = K / BlkLen;
            size_t nb0 = bm / BitWidth * nb1;
            size_t new_im = idx / nb0;
            size_t new_ibm = (idx % nb0) / nb1;
            size_t new_ik = (idx % nb1);

            if (HasZeroPoint) {
                size_t new_isimd = new_ibm % simd_n_out;
                size_t new_idx_outer = new_im * bm / bits * K / BlkLen / simd_n_out + new_ik * bm / bits / simd_n_out + new_ibm / simd_n_out;
                size_t new_idx_scale = new_idx_outer * (simd_n_out * 2) + new_isimd;
                size_t new_idx_zero = new_idx_outer * (simd_n_out * 2) + simd_n_out + new_isimd;

                PackedQuantBZPBegin[new_idx_scale] = scale;
                PackedQuantBZPBegin[new_idx_zero] = zp;
            } else {
                size_t new_idx = new_im * bm / bits * K / BlkLen + new_ik * bm / bits + new_ibm;
                PackedQuantBZPBegin[new_idx] = scale;
            }
        }
    }
}


bool MLASCALL MlasIsTMACAvailable(
    size_t /*BlkBitWidth*/,
    size_t /*BlkLen*/
) // TODO: fix the below to use smthg besides the gen kernel
{
    const auto* Dispatch = GetMlasPlatform().LUTGenKernel;
    return Dispatch != nullptr;
    // return Dispatch != nullptr && BlkLen == 4; // only support group sizes of 4 for now
}

size_t CalculateLUTSize(int k, int m, size_t group_size) {
    return k * m * group_size;
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

    const MlasTMACKernelParams& tmac_params = GetTMACKernelParams(N, K, 2, BlkLen);
    size_t lut_size = CalculateLUTSize(K, M, tmac_params.g);
    auto lut_buffer = std::make_unique<uint8_t[]>(lut_size);

    const size_t lut_scales_size_meta = 64;
    const size_t lut_meta_size = 64 * M * tmac_params.g; // TODO: 64 should be stored as lut_scales_size
    auto biases_float = std::make_unique<float[]>(lut_meta_size);
    auto scales_float = std::make_unique<float[]>(lut_meta_size);

    const auto* a_float = reinterpret_cast<const float*>(A);  // Activation data

    // const int num_groups = static_cast<int>(K / BlkLen);

    // Parallelize over M (batch dimension)
    // Each iteration processes one row of the activation matrix
    MlasTrySimpleParallel(
        threadpool, 
        static_cast<size_t>(M),
        [&](ptrdiff_t ine11) {
            const size_t row_offset = static_cast<size_t>(ine11) * K;
            const size_t lut_offset = static_cast<size_t>(ine11) * K * 4;  // 4 bytes per K element for 2-bit LUT
            const size_t scale_bias_offset = static_cast<size_t>(ine11) * lut_scales_size_meta;

            // Call the dispatch function for this row
            Dispatch->GenerateLUT(
                static_cast<int32_t>(BlkLen), 
                reinterpret_cast<int8_t*>(lut_buffer.get()) + lut_offset,  // Output LUT for this row
                const_cast<float*>(a_float + row_offset),                     // Input activation for this row
                scales_float.get() + scale_bias_offset,   // Scales for this row
                biases_float.get() + scale_bias_offset,   // Biases for this row
                K
            );
        }
    );

    // all relevant LUT's have been generated
    // equivalent of lut_mul_mat's ggml_backend_tmac_mul_mat function ggml_barrier line
    const size_t bm = tmac_params.bm; // TODO: hardcoding for now
    const size_t bits = tmac_params.bits; 

    // TODO: fix the below 4
    // Matrix multiplication: Output[N×M] = QuantBData[N×K] × Weights[K×M]
    const size_t OutputRows = N;    // Number of output features
    const size_t OutputCols = M;    // Batch size
    const size_t NumTiles = 8; // hardcoding -- TODO: should be moved to tmac kernel config

    const size_t ChunkSize0 = N / NumTiles;
    const size_t ChunkSize1 = tmac_params.chunk_n; // process one batch item at a time

// In llama.cpp terminology (note the swap!):
// ne0 = M (output features, called "n" in llama.cpp)
// ne1 = N (batch size, called "m" in llama.cpp)

    // Calculate number of chunks in each dimension
    const size_t nchunk0 = (OutputRows + ChunkSize0 - 1) / ChunkSize0;  // Should equal NumTiles
    const size_t nchunk1 = (OutputCols + ChunkSize1 - 1) / ChunkSize1;
    const size_t total_chunks = nchunk0 * nchunk1;

    // Pre-calculate sizes for offset calculations
    const size_t w_size = N * K * bits / 8;
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
                        1,
                        BlkLen                                          // Weight quantization group size
                    );
                }
            }
        }
    );
}