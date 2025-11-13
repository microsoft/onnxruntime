/*++

// TODO: finish filling this out

module includes kernel functions for generating LUT for T-MAC GEMM optimization strategy.
*/
#include <string>
#include <thread>
#include <unordered_map>

#include "qlutgemm.h"

#include <cassert>

/** T-MAC GEMM kernel Config */
static std::unordered_map<std::string, struct MlasTMACKernelParams> tmac_kernel_configs;

const MlasTMACKernelParams& MlasGetLUTGemmKernelParams(size_t M, size_t N, size_t nbits) {
    std::string key = std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(nbits);
    if (tmac_kernel_configs.count(key)) {
        return tmac_kernel_configs[key];
    } else {
        ORT_THROW("T-MAC kernel parameters not initialized for M=", M, ", N=", N, ", nbits=", nbits);
    }
}

void MlasInitLUTGemmKernelConfig(size_t M, size_t N, size_t nbits, size_t block_size, bool has_zp_point) {
    std::string key = std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(nbits);
    if (tmac_kernel_configs.count(key)) {
        return;
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

    // TODO(vraspar): add profile based policy
    int threads = std::thread::hardware_concurrency();

    float smallest_penalty = 1e9;
    params.bm = bms[0];
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
    params.kfactor = kfactors[0];
    for (size_t kfactor: kfactors) {
        if ((kfactor < params.actk) || (kfactor * params.g > params.q_group_size)) {
            continue;
        }
        if (kfactor > largest_kfactor) {
            largest_kfactor = kfactor;
            params.kfactor = kfactor;
        }
    }

    params.n_tiles_num = M * params.bits / params.bm;
    params.has_scale = true; // TODO(vraspar): TMAC supports only scale for now
    params.has_zero_point = has_zp_point;
    params.one_scale = false; //TODO(vraspar): support one scale case for bitnet

    tmac_kernel_configs[key] = params;
    return;
}


size_t MlasLUTGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    bool HasZeroPoint,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE ComputeType
)
{
    MLAS_UNREFERENCED_PARAMETER(ComputeType);
    const MlasTMACKernelParams& tmac_params = MlasGetLUTGemmKernelParams(N, K, BlkBitWidth);
    const size_t PackedQuantBDataSize = (N * BlkBitWidth) * (K / tmac_params.g / tmac_params.ngroups_per_elem);
    return PackedQuantBDataSize;
}


void
MlasLUTGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool
)
{
    // decompose W into w1,... w_bits create temp buffer buf2 of size N * bits * (K/g)
    const MlasTMACKernelParams& tmac_params = MlasGetLUTGemmKernelParams(N, K, BlkBitWidth);
    const size_t bits = tmac_params.bits;
    const size_t g = tmac_params.g;
    const size_t ngroups_per_elem = tmac_params.ngroups_per_elem;
    const size_t simd_n_in = tmac_params.simd_n_in;
    const size_t simd_n_out = tmac_params.simd_n_out;
    const size_t bm = tmac_params.bm;
    const size_t kfactor = tmac_params.kfactor;

    assert(BlkLen % g == 0);
    assert((BlkLen / g) % kfactor == 0);

    const int mgroup = ngroups_per_elem * simd_n_in;  // 32
    assert(bm % mgroup == 0);
    assert(bm % bits == 0);

    uint8_t* buf = new uint8_t[N * bits * (K / g)];
    memset(buf, 0, N * bits * (K / g));

    const size_t Iterations = N;  // we parallelize over N, TODO:: tune if needed

    MlasTrySimpleParallel(
        ThreadPool, Iterations,
        [&](ptrdiff_t tid) {
            size_t im = static_cast<size_t>(tid);
            for (size_t ik = 0; ik < K; ++ik) {
                size_t idx = (im * K + ik);
                size_t num_elem_per_byte = 8 / bits;
                size_t elem_idx = idx % num_elem_per_byte;

                uint8_t v = ((const uint8_t*)QuantBDataBegin)[idx / num_elem_per_byte] >> (elem_idx * bits);

                for (size_t ib = 0; ib < bits; ++ib) {
                    size_t new_ik = ik / g;
                    size_t shft_left = ik % g;
                    buf[im * bits * K / g + ib * K / g + new_ik] += ((v >> ib) & 1) << shft_left;
                }
            }
        }
    );

    // Now buf contains the bit planes grouped by g along K
    // Next, we need to do a multi-reshape/transpose into the final layout

    const size_t c0_fac2 = K / g;
    const size_t c0_fac1 = simd_n_out * c0_fac2;
    const size_t c0_fac0 = bits * c0_fac1;

    const size_t c1_nb2 = K / g;
    const size_t c1_nb1 = simd_n_in * c1_nb2;
    const size_t c1_nb0 = ngroups_per_elem * c1_nb1;
    const size_t c1_fac2 = K / g;
    const size_t c1_fac1 = ngroups_per_elem * c1_fac2;
    const size_t c1_fac0 = simd_n_in * c1_fac1;

    const size_t c2_nb4 = kfactor;
    const size_t c2_nb3 = K / g / kfactor * c2_nb4;
    const size_t c2_nb2 = ngroups_per_elem * c2_nb3;
    const size_t c2_nb1 = simd_n_in * c2_nb2;
    const size_t c2_nb0 = bm / mgroup * c2_nb1;
    const size_t c2_fac3 = simd_n_in * ngroups_per_elem;
    const size_t c2_fac2 = kfactor * c2_fac3;
    const size_t c2_fac1 = bm / mgroup * c2_fac2;
    const size_t c2_fac0 = K / g / kfactor * c2_fac1;

    const size_t PackedQuantBDataSize = (N * bits) * (K / g / ngroups_per_elem);
    memset(PackedQuantBDataBegin, 0, PackedQuantBDataSize);  // TODO: is this needed?

    MlasTrySimpleParallel(
        ThreadPool, Iterations,
        [&](ptrdiff_t tid) {
            size_t im = static_cast<size_t>(tid);
            for (size_t ib = 0; ib < bits; ib++) {
                for (size_t ik = 0; ik < K / g; ik++) {
                    // w = w.reshape(M // bits // simd_n_out, simd_n_out, bits, K // g).transpose(0, 2, 1, 3)
                    size_t new_im = im / simd_n_out;
                    size_t new_isno = im % simd_n_out;
                    size_t new_ib = ib;
                    size_t new_ik = ik;
                    size_t new_idx = new_im * c0_fac0 + new_ib * c0_fac1 + new_isno * c0_fac2 + new_ik;

                    // w = w.reshape(M // mgroup, ngroups_per_elem, simd_n_in, K // g).transpose(0, 2, 1, 3)
                    new_im = new_idx / c1_nb0;
                    size_t new_ing = (new_idx % c1_nb0) / c1_nb1;
                    size_t new_isni = (new_idx % c1_nb1) / c1_nb2;
                    new_ik = (new_idx % c1_nb2);
                    new_idx = new_im * c1_fac0 + new_isni * c1_fac1 + new_ing * c1_fac2 + new_ik;

                    // #             0        1             2             3                 4                  5
                    // w = w.reshape(M // bm, bm // mgroup, simd_n_in, ngroups_per_elem, K // g // kfactor, kfactor).transpose(0, 4, 1, 5, 2, 3)
                    new_im = new_idx / c2_nb0;
                    size_t new_ibm = (new_idx % c2_nb0) / c2_nb1;
                    new_isni = (new_idx % c2_nb1) / c2_nb2;
                    new_ing = (new_idx % c2_nb2) / c2_nb3;
                    new_ik = (new_idx % c2_nb3) / c2_nb4;
                    size_t new_ikf = (new_idx % c2_nb4);
                    new_idx = new_im * c2_fac0 +
                              new_ik * c2_fac1 +
                              new_ibm * c2_fac2 +
                              new_ikf * c2_fac3 +
                              new_isni * ngroups_per_elem +
                              new_ing;
                    new_idx = new_idx / ngroups_per_elem;
                    size_t buf_idx = im * bits * K / g + ib * K / g + ik;
                    uint8_t buf_val = buf[buf_idx];

                    // w = sum([(w[:, :, :, :, :, ng] << (ng * g)) for ng in range(ngroups_per_elem)])
                    PackedQuantBDataBegin[new_idx] = static_cast<std::byte>(
                        static_cast<unsigned>(PackedQuantBDataBegin[new_idx]) +
                        (buf_val << (new_ing * g))
                    );
                }
            }
        }
    );
    delete[] buf;

}


size_t MLASCALL
MlasLUTPackScalesAndZeroPointsSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    bool HasZeroPoint
)
{
    // TODO(vraspar): support one scale case
    if (HasZeroPoint) {
        return N * K / BlkLen * 2;
    } else {
        return N * K / BlkLen;
    }
}


void MlasLUTPackScalesAndZeroPoints(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    bool HasZeroPoint,
    float* PackedQuantBZPBegin,
    const float* QuantBScale,
    const uint8_t* QuantBZeroPoint
)
{
    const MlasTMACKernelParams& tmac_params = MlasGetLUTGemmKernelParams(N, K, BlkBitWidth);
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
                // TODO(vraspar): logically correct but not readable
                uint8_t v = QuantBZeroPoint[idx / num_elem_per_byte] >> (elem_idx * bits) & (1 << bits) - 1;
                zp = static_cast<float>(v);

                // Note: TMAC does this during model conversion. Since, we follow ORT format, we need to do it here.
                // This seems gptq quantization specific.
                // We should either use different op than matmul_nbits or add attribute to matmul_nbits to indicate this.
                zp = zp - (1 << (bits - 1)) - 1;  // make it signed
                zp = zp * scale;              // store scale * zp
            }

            // TODO(vraspar): fix when k < BlkLen and nb1 is 0
            size_t nb1 = K / BlkLen;
            size_t nb0 = bm / bits * nb1;

            size_t new_im, new_ibm, new_ik;
            if (nb1 == 0) {
                new_im = 0;
                new_ibm = 0;
                new_ik = 0;

            } else {
                new_im = idx / nb0;
                new_ibm = (idx % nb0) / nb1;
                new_ik = (idx % nb1);
            }

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


bool MLASCALL MlasIsLUTGemmAvailable(
    size_t /*BlkBitWidth*/,
    size_t /*BlkLen*/
) // TODO(Vraspar): fix the below to use smthg besides the gen kernel, add ComputeGemm
{
    const auto* Dispatch = GetMlasPlatform().LUTGenKernel;
    return Dispatch != nullptr;
    // return Dispatch != nullptr && BlkLen == 4; // only support group sizes of 4 for now
}


size_t
CalculateLUTBufferSize(size_t n, size_t k, size_t m, const MlasTMACKernelParams& tmac_params) {
    constexpr size_t kAllockAligment = 64;
    const size_t lut_scales_size = k / tmac_params.act_group_size;


    size_t wsize = k * m * 4 * sizeof(int8_t);  // 4 bytes per k element for 2-bit LUT
    wsize += lut_scales_size * m * 2 * sizeof(float);  // scales + biases

    wsize = ((wsize - 1) / kAllockAligment + 1) * kAllockAligment;

    // TODO(vrapar): add temp buffer for FP16
    return wsize;
}

void MLASCALL MlasLUTGemm(
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



    /** TODO(vraspar): The biases_float and scales float values don't make sense
     * FP 16
     * QLUT K(ne10) x M(ne11) x 4 bytes
     * Scales: lut_scales_size * M * 2 bytes
     * Biases: lut_scales_size * M * 2 bytes
     * Needs FP 16 conversion Buffer: max(K, N) * M * 2 bytes
     *
     * FP 32
     * QLUT K x M x 4 bytes
     * Scales: lut_scales_size * M * 4 bytes
     * Biases: lut_scales_size * M * 4 bytes
     *
     * Currently, we only support FP32, add FP16 support later which requires conversion buffer
     *
     * LUT Buffer for FP32 : K * M * 4 * sizeof(uint8_t) bytes + lut_scale_size * m * 2 * sizeof(float) bytes  + allignment
     *
     */

    // n_tiles_num = m * bits / bm;

    // TODO(vraspar): support other bitwidths
    const MlasTMACKernelParams& tmac_params = MlasGetLUTGemmKernelParams(N, K, 2);
    const size_t lut_scales_size = K / tmac_params.act_group_size;
    size_t lut_buffer_size = CalculateLUTBufferSize(N, K, M, tmac_params);

    // make buffer of lut_buffer_size bytes
    // TODO(vraspar): other way to do it
    auto lut_buffer = std::make_unique<int8_t[]>(lut_buffer_size);

    int8_t* qlut = reinterpret_cast<int8_t*>(lut_buffer.get());
    float* lut_scales = reinterpret_cast<float*>(qlut + K * M * 4);  // after lut
    float* lut_biases = reinterpret_cast<float*>(lut_scales + lut_scales_size * M);  // after scales


    const auto* a_float = reinterpret_cast<const float*>(A);  // Activation data

    // const int num_groups = static_cast<int>(K / BlkLen);

    // Parallelize over M (batch dimension)
    // Each iteration processes one row of the activation matrix
    // TODO(vraspar): Ideally we have to do block parallelism here

    MlasTrySimpleParallel(
        threadpool,
        static_cast<size_t>(M),
        [&](ptrdiff_t ine11) {
            const size_t row_offset = static_cast<size_t>(ine11) * K;
            const size_t lut_offset = static_cast<size_t>(ine11) * K * 4;  // 4 bytes per K element for 2-bit LUT
            const size_t scale_bias_offset = static_cast<size_t>(ine11) * lut_scales_size;

            // Call the dispatch function for this row
            // ggml_tmac_mul_mat_task_init
            Dispatch->GenerateLUT(
                const_cast<float*>(a_float + row_offset),  // Input activation for this row
                qlut + lut_offset,                         // Output LUT for this row
                lut_scales + scale_bias_offset,            // Scales for this row
                lut_biases + scale_bias_offset,            // Biases for this row
                M,
                K,
                N,
                tmac_params.act_group_size
            );
        }
    );

    // all relevant LUT's have been generated
    // equivalent of lut_mul_mat's ggml_backend_tmac_mul_mat function ggml_barrier line

    const size_t n_tiles_num = tmac_params.n_tiles_num;
    assert(N % n_tiles_num == 0);

    const size_t bm = tmac_params.bm; // TODO: hardcoding for now
    const size_t bits = tmac_params.bits;

    // Pre-calculate sizes for offset calculations
    const size_t w_size = N * K * bits / 8;
    const size_t w_chunk_size = w_size / n_tiles_num;

    // TODO: fix the below 4
    // Matrix multiplication: Output[N×M] = QuantBData[N×K] × Weights[K×M]
    const size_t OutputRows = N;    // Number of output features
    const size_t OutputCols = M;    // Batch size

    const size_t ChunkSize0 = N / n_tiles_num;
    const size_t ChunkSize1 = tmac_params.chunk_n; // process one batch item at a time

// In llama.cpp terminology (note the swap!):
// ne0 = M (output features, called "n" in llama.cpp)
// ne1 = N (batch size, called "m" in llama.cpp)

    // Calculate number of chunks in each dimension
    const size_t nchunk0 = (OutputRows + ChunkSize0 - 1) / ChunkSize0;  // Should equal NumTiles
    const size_t nchunk1 = (OutputCols + ChunkSize1 - 1) / ChunkSize1;
    const size_t total_chunks = nchunk0 * nchunk1;


    // TODO(vraspar): support one_scale case
    // Determine weight-scale layout. These should be provided by the caller or inferred from the packed weights.
    // For now we default to per-group symmetric quantization (no zero-point, not one-scale).

    // Total number of scale (float) entries for the whole weight matrix:
    // - if one_scale: single global scale (1)
    // - otherwise: number of quantization groups = (M * K / BlkLen)
    //   and if zero-points are present each group stores (scale, zero_point) -> *2
    const size_t groups_total = static_cast<size_t>(M) * static_cast<size_t>(K) / BlkLen;
    const size_t scales_size_total = MlasLUTPackScalesAndZeroPointsSize(
        static_cast<size_t>(N),
        static_cast<size_t>(K),
        BlkLen,
        tmac_params.has_zero_point
    );


    // Per-tile scales size = total scales size divided evenly across tiles.
    // If one_scale is true we do not advance the scales pointer per tile, so set per tile size to 0
    size_t scales_size_per_tile = 0;

    if (scales_size_total % n_tiles_num != 0) {
        // Sanity: scales should partition evenly across tiles. If they don't, choose floor division
        // and document that callers must layout scales accordingly.
        // Prefer to error loudly in debug builds.
        fprintf(stderr, "Warning: scales_size_total=%zu is not divisible by n_tiles_num=%zu; using floor division.\n", scales_size_total, n_tiles_num);
    }
    scales_size_per_tile = scales_size_total / n_tiles_num;


    // Note: when one_scale == true, callers should pass a pointer to a single scale value (scales_offset=0 will be used)

    // Cast to appropriate types
    const auto* packed_weights = reinterpret_cast<const uint8_t*>(QuantBData);

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
                    // Note M and N are swapped in TMAC terminology
                    // TODO(vrapsar): fix this M and N swapp mess
                    Dispatch->ComputeGemm(
                        packed_weights + w_offset,                      // Weight tile
                        QuantBScale + scales_offset,                    // Weight scales for this tile
                        qlut + qlut_offset,                             // LUT for this batch row
                        lut_scales + lut_scales_offset,                 // LUT scales
                        lut_biases + lut_scales_offset,                 // LUT biases
                        reinterpret_cast<float*>(C) + dst_offset,     // Output location
                        static_cast<int>(K),                            // K dimension
                        static_cast<int>(N),                            // K dimension
                        static_cast<int>(1),
                        BlkLen                                          // Weight quantization group size
                    );
                }
            }
        }
    );
}
