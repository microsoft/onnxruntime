/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qlutgemm.cpp

Abstract:

    This module implements kernel functions for generating lookup tables (LUT)
    and computing matrix multiplication for the T-MAC GEMM optimization strategy.

    It provides functionality to pack quantized weight data, compute LUT scales
    and biases, and perform efficient quantized GEMM operations using lookup
    table based computation.

--*/
#include "qlutgemm.h"

#include <cassert>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <thread>
#include <mutex>
#include <unordered_map>

/** T-MAC GEMM kernel config key - struct-based for type safety and performance */
struct TMACConfigKey {
    size_t M;
    size_t N;
    size_t nbits;
    size_t block_size;
    bool has_zero_point;

    bool operator==(const TMACConfigKey& other) const {
        return M == other.M && N == other.N && nbits == other.nbits &&
               block_size == other.block_size && has_zero_point == other.has_zero_point;
    }
};

struct TMACConfigKeyHash {
    size_t operator()(const TMACConfigKey& k) const {
        // Combine hash values using a simple mixing function
        size_t h = std::hash<size_t>{}(k.M);
        h ^= std::hash<size_t>{}(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>{}(k.nbits) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<size_t>{}(k.block_size) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<bool>{}(k.has_zero_point) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

/**
 * Global cache for T-MAC kernel parameters, indexed by configuration.
 * This map and its associated mutex ensure thread-safe parameter management
 * across concurrent MLAS calls.
 */
static std::unordered_map<TMACConfigKey, MlasTMACKernelParams, TMACConfigKeyHash> tmac_kernel_configs;
static std::mutex tmac_kernel_configs_mutex;

static std::string
GetTmacKey(size_t M, size_t N, size_t nbits, size_t block_size, bool has_zero_point)
{
    // Generate a unique cache key based on the GEMM and quantization configuration.
    return std::to_string(M) + "_" + std::to_string(N) + "_" + std::to_string(nbits) + "_" +
           std::to_string(block_size) + "_" + (has_zero_point ? "1" : "0");
}

MlasTMACKernelParams
MlasGetLutGemmKernelParams(size_t M, size_t N, size_t nbits, size_t block_size, bool has_zero_point)
{
    TMACConfigKey key{M, N, nbits, block_size, has_zero_point};
    std::lock_guard<std::mutex> lock(tmac_kernel_configs_mutex);
    auto it = tmac_kernel_configs.find(key);
    if (it != tmac_kernel_configs.end()) {
        return it->second;
    }
    MLAS_THROW_EX(std::runtime_error, "T-MAC kernel parameters not initialized for key: " + GetTmacKey(M, N, nbits, block_size, has_zero_point));
}

void MLASCALL
MlasClearLutGemmKernelConfig()
{
    std::lock_guard<std::mutex> lock(tmac_kernel_configs_mutex);
    tmac_kernel_configs.clear();
}

void MLASCALL
MlasInitLutGemmKernelConfig(size_t M, size_t N, size_t nbits, size_t block_size, bool has_zero_point)
{
    TMACConfigKey key{M, N, nbits, block_size, has_zero_point};
    {
        std::lock_guard<std::mutex> lock(tmac_kernel_configs_mutex);
        if (tmac_kernel_configs.find(key) != tmac_kernel_configs.end()) {
            return;
        }
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
        MLAS_THROW_EX(std::runtime_error, "Unsupported activation group size");
    }
    params.actk = params.act_group_size / params.g;

    // search space
    std::vector<size_t> bms;
    if (nbits == 1 || nbits == 2 || nbits == 4) {
        bms = {256, 512, 1024, 2048, 320, 640, 1280};
    } else if (nbits == 3) {
        bms = {192, 384, 576, 758};
    }

    std::vector<size_t> kfactors = {8, 16};

    // TODO(vraspar): add profile based policy
    size_t threads = static_cast<size_t>(std::thread::hardware_concurrency());

    float smallest_penalty = 1e9f;
    params.bm = bms[0];
    for (size_t bm : bms) {
        if (M % (bm / nbits) != 0 || bm % nbits != 0) {
            continue;
        }
        size_t num_tiles = M / (bm / nbits);
        size_t num_groups = (num_tiles + threads - 1) / threads;
        float penalty = 0.1f * static_cast<float>(num_groups) +
                        (static_cast<float>(num_groups) - 1.0f * static_cast<float>(num_tiles) / static_cast<float>(threads)) /
                            static_cast<float>(num_groups);
        if (penalty < smallest_penalty) {
            smallest_penalty = penalty;
            params.bm = bm;
        }
    }

    size_t largest_kfactor = 0;
    params.kfactor = kfactors[0];
    for (size_t kfactor : kfactors) {
        if ((kfactor < params.actk) || (kfactor * params.g > params.q_group_size)) {
            continue;
        }
        if (kfactor > largest_kfactor) {
            largest_kfactor = kfactor;
            params.kfactor = kfactor;
        }
    }

    params.n_tiles_num = M * params.bits / params.bm;
    params.has_scale = true;  // TODO(vraspar): TMAC supports only scale for now
    params.has_zero_point = has_zero_point;
    params.one_scale = false;  // TODO(vraspar): support one scale case for bitnet

    {
        std::lock_guard<std::mutex> lock(tmac_kernel_configs_mutex);
        tmac_kernel_configs[key] = params;
    }
    return;
}

// Internal helper: calculates packed quantized B data size
static size_t
LutGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    bool HasZeroPoint
)
{
    const MlasTMACKernelParams& tmac_params = MlasGetLutGemmKernelParams(N, K, BlkBitWidth, BlkLen, HasZeroPoint);
    const size_t PackedQuantBDataSize = (N * BlkBitWidth) * (K / tmac_params.g / tmac_params.ngroups_per_elem);
    return PackedQuantBDataSize;
}

// Internal helper: packs quantized B data
static void
LutGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    bool HasZeroPoint,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* ThreadPool
)
{
    // decompose W into w1,... w_bits create temp buffer buf2 of size N * bits * (K/g)
    const MlasTMACKernelParams& tmac_params = MlasGetLutGemmKernelParams(N, K, BlkBitWidth, BlkLen, HasZeroPoint);
    const size_t bits = tmac_params.bits;
    const size_t g = tmac_params.g;
    const size_t ngroups_per_elem = tmac_params.ngroups_per_elem;
    const size_t simd_n_in = tmac_params.simd_n_in;
    const size_t simd_n_out = tmac_params.simd_n_out;
    const size_t bm = tmac_params.bm;
    const size_t kfactor = tmac_params.kfactor;

    // LUT GEMM requires a valid LUT dispatch implementation, so dispatch must be available
    const auto* Dispatch = GetMlasPlatform().LutGenKernel;
    if (Dispatch == nullptr || Dispatch->PackQuantBData == nullptr) {
        MLAS_THROW_EX(std::runtime_error, "PackQuantBData requires LUT GEMM dispatch support");
    }

    Dispatch->PackQuantBData(
        N, K, bits, g, ngroups_per_elem,
        simd_n_in, simd_n_out, bm, kfactor,
        QuantBDataBegin, PackedQuantBDataBegin, ThreadPool
    );
}

// Internal helper: calculates packed scales and zero points size in floats
static size_t
LutPackScalesAndZeroPointsSize(
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

// Internal helper: packs scales and zero points
static void
LutPackScalesAndZeroPoints(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    bool HasZeroPoint,
    float* PackedQuantBZPBegin,
    const float* QuantBScale,
    const uint8_t* QuantBZeroPoint,
    MLAS_THREADPOOL* ThreadPool
)
{
    const MlasTMACKernelParams& tmac_params = MlasGetLutGemmKernelParams(N, K, BlkBitWidth, BlkLen, HasZeroPoint);
    const size_t bits = tmac_params.bits;
    const size_t simd_n_out = tmac_params.simd_n_out;
    const size_t bm = tmac_params.bm;

    // LUT GEMM is only available for AVX2, so dispatch must be available
    const auto* Dispatch = GetMlasPlatform().LutGenKernel;
    if (Dispatch == nullptr || Dispatch->PackScalesAndZeroPoints == nullptr) {
        MLAS_THROW_EX(std::runtime_error, "PackScalesAndZeroPoints requires AVX2 dispatch");
    }

    Dispatch->PackScalesAndZeroPoints(
        N, K, bits, BlkLen, simd_n_out, bm, HasZeroPoint,
        PackedQuantBZPBegin, QuantBScale, QuantBZeroPoint, ThreadPool
    );
}

// Internal helper: calculates the offset to scales in the packed buffer
static size_t
LutGemmPackedScalesOffset(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    bool HasZeroPoint
)
{
    constexpr size_t kAlignment = 64;  // Cache line alignment
    size_t packed_b_size = LutGemmPackQuantBDataSize(N, K, BlkBitWidth, BlkLen, HasZeroPoint);
    return ((packed_b_size + kAlignment - 1) / kAlignment) * kAlignment;
}

size_t MLASCALL
MlasLutGemmPackedSize(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    bool HasZeroPoint
)
{
    // Get packed B size (aligned)
    size_t aligned_b_size = LutGemmPackedScalesOffset(N, K, BlkBitWidth, BlkLen, HasZeroPoint);

    // Get packed scales/zp size (in floats, convert to bytes)
    size_t packed_scales_count = LutPackScalesAndZeroPointsSize(N, K, BlkLen, HasZeroPoint);
    size_t packed_scales_bytes = packed_scales_count * sizeof(float);

    return aligned_b_size + packed_scales_bytes;
}

void MLASCALL
MlasLutGemmPack(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen,
    bool HasZeroPoint,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const uint8_t* QuantBZeroPoint,
    std::byte* PackedBuf,
    MLAS_THREADPOOL* ThreadPool
)
{
    // Pack B data if provided
    if (QuantBData != nullptr) {
        LutGemmPackQuantBData(N, K, BlkBitWidth, BlkLen, HasZeroPoint, QuantBData, PackedBuf, ThreadPool);
    }

    // Pack scales/zero points if scales are provided
    if (QuantBScale != nullptr) {
        size_t scales_offset = LutGemmPackedScalesOffset(N, K, BlkBitWidth, BlkLen, HasZeroPoint);
        float* scales_dest = reinterpret_cast<float*>(PackedBuf + scales_offset);
        LutPackScalesAndZeroPoints(N, K, BlkBitWidth, BlkLen, HasZeroPoint, scales_dest, QuantBScale, QuantBZeroPoint, ThreadPool);
    }
}

bool MLASCALL
MlasIsLutGemmAvailable(
    size_t N,
    size_t K,
    size_t BlkBitWidth,
    size_t BlkLen
)
{
    const auto* lut_kernel = GetMlasPlatform().LutGenKernel;
    if (lut_kernel == nullptr ||
        lut_kernel->GenerateLUT == nullptr ||
        lut_kernel->ComputeGemm == nullptr ||
        lut_kernel->PackQuantBData == nullptr ||
        lut_kernel->PackScalesAndZeroPoints == nullptr) {
        return false;
    }

    // currently only 2-bit is supported
    if (BlkBitWidth != 2 || BlkLen == 0 || (BlkLen % 32) != 0) {
        return false;
    }

    if (K % 32 != 0) {
        return false;
    }

    size_t n_div = 0;
    switch (BlkBitWidth) {
        case 1:
            n_div = 256;
            break;
        case 2:
            n_div = 128;
            break;
        case 3:
            n_div = 64;
            break;
        case 4:
            n_div = 32;
            break;
        default:
            return false;
    }

    if (N % n_div != 0) {
        return false;
    }
    return true;
}

size_t
CalculateLutBufferSize(size_t n, size_t k, size_t m, const MlasTMACKernelParams& tmac_params)
{
    MLAS_UNREFERENCED_PARAMETER(n);
    const size_t lut_scales_size = k / tmac_params.act_group_size;

    // The AVX2 kernel (g=4) expects 16 entries (16 bytes) per group of 4 activations.
    // This effectively requires 4 bytes per activation in the K dimension.
    size_t lut_size_bytes = m * k * 4;
    size_t scales_size_bytes = m * lut_scales_size * sizeof(float);
    size_t biases_size_bytes = m * lut_scales_size * sizeof(float);

    return lut_size_bytes + scales_size_bytes + biases_size_bytes + 256;  // + alignment/safety padding
}

void MLASCALL
MlasLutGemm(
    const void* A,
    size_t BlkLen,
    const void* PackedBuf,  // Packed buffer containing weights followed by scales/zp
    void* C,
    size_t K,
    size_t M,  // batch size (number of rows in activation)
    size_t N,
    bool HasZeroPoint,
    MLAS_THREADPOOL* threadpool
)
{
    // adapted from ggml_backend_tmac_mul_mat
    const auto* Dispatch = GetMlasPlatform().LutGenKernel;
    // This should be ensured by calling MlasIsLutGemmAvailable() before MlasLutGemm()
    if (Dispatch == nullptr || Dispatch->GenerateLUT == nullptr || Dispatch->ComputeGemm == nullptr) {
        MLAS_THROW_EX(std::runtime_error, "TMAC not supported in this configuration");
    }

    // Calculate scales offset from packed buffer
    // TODO(vraspar): support other bitwidths
    constexpr size_t BlkBitWidth = 2;
    size_t scales_offset = LutGemmPackedScalesOffset(N, K, BlkBitWidth, BlkLen, HasZeroPoint);
    const auto* QuantBData = PackedBuf;
    const auto* QuantBScale = reinterpret_cast<const float*>(
        static_cast<const std::byte*>(PackedBuf) + scales_offset
    );

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
    // For T-MAC, kernel properties (bm, n_tiles_num) are primarily driven by the number of output features (N).
    // Initialization during packing (LutGemmPackQuantBDataSize) uses N as the major dimension,
    // so we must match that here to ensure consistent weight tiling.
    MlasInitLutGemmKernelConfig(N, K, 2, BlkLen, HasZeroPoint);
    const MlasTMACKernelParams& tmac_params = MlasGetLutGemmKernelParams(N, K, 2, BlkLen, HasZeroPoint);
    const size_t lut_scales_size = K / tmac_params.act_group_size;
    const size_t lut_size_bytes = static_cast<size_t>(M) * static_cast<size_t>(K) * 4;
    size_t lut_buffer_size = CalculateLutBufferSize(N, K, M, tmac_params);

    // make buffer of lut_buffer_size bytes
    // TODO(vraspar): other way to do it
    auto lut_buffer = std::make_unique<int8_t[]>(lut_buffer_size);
    memset(lut_buffer.get(), 0, lut_buffer_size);

    int8_t* qlut = reinterpret_cast<int8_t*>(lut_buffer.get());
    float* lut_scales = reinterpret_cast<float*>(qlut + lut_size_bytes);                  // after lut
    float* lut_biases = reinterpret_cast<float*>(lut_scales + lut_scales_size * M);       // after scales

    const auto* a_float = reinterpret_cast<const float*>(A);  // Activation data

    // const int num_groups = static_cast<int>(K / BlkLen);

    // Iterate over M (batch dimension)
    // Each iteration processes one row of the activation matrix.
    // NOTE: This loop is intentionally serialized. Previous attempts to parallelize
    // using MlasTrySimpleParallel caused flaky test failures (race conditions)
    // when M > 1 (e.g., Batch32 case). Since GenerateLUT is lightweight,
    // serial execution ensures correctness with negligible performance impact.
    // TODO(vraspar): Ideally we have to do block parallelism here

    for (size_t ine11 = 0; ine11 < static_cast<size_t>(M); ine11++) {
        const size_t row_offset = ine11 * K;
        // Call the LUT generation kernel for this activation row.
        // We use a 4-byte stride (per activation) for the LUT entries to satisfy
        // the memory layout requirements of the computation kernel.
        const size_t lut_offset = ine11 * K * 4;
        const size_t scale_bias_offset = ine11 * lut_scales_size;

        Dispatch->GenerateLUT(
            const_cast<float*>(a_float + row_offset),  // Input activation for this row
            qlut + lut_offset,                         // Output LUT for this row
            lut_scales + scale_bias_offset,            // Scales for this row
            lut_biases + scale_bias_offset,            // Biases for this row
            M,
            K,
            N,
            tmac_params.act_group_size,
            tmac_params.act_group_size * 4
        );
    }

    // all relevant LUT's have been generated
    // equivalent of lut_mul_mat's ggml_backend_tmac_mul_mat function ggml_barrier line

    const size_t n_tiles_num = tmac_params.n_tiles_num;
    assert(N % n_tiles_num == 0);

    const size_t bits = tmac_params.bits;

    // Pre-calculate sizes for offset calculations
    const size_t w_size = N * K * bits / 8;
    const size_t w_chunk_size = w_size / n_tiles_num;

    // TODO: fix the below 4
    // Matrix multiplication: Output[N×M] = QuantBData[N×K] × Weights[K×M]
    const size_t OutputRows = N;  // Number of output features
    const size_t OutputCols = M;  // Batch size

    const size_t ChunkSize0 = N / n_tiles_num;
    const size_t ChunkSize1 = tmac_params.chunk_n;  // process one batch item at a time

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

    const size_t scales_size_total = LutPackScalesAndZeroPointsSize(
        static_cast<size_t>(N),
        static_cast<size_t>(K),
        BlkLen,
        tmac_params.has_zero_point
    );

    // Per-tile scales size = total scales size divided evenly across tiles.
    // If one_scale is true we do not advance the scales pointer per tile, so set per tile size to 0
    size_t scales_size_per_tile = 0;

    if (scales_size_total % n_tiles_num != 0) {
        // Scales must partition evenly across tiles. Callers must ensure proper layout.
        MLAS_THROW_EX(std::runtime_error, "scales_size_total must be divisible by n_tiles_num");
    }
    scales_size_per_tile = scales_size_total / n_tiles_num;

    // Note: when one_scale == true, callers should pass a pointer to a single scale value (scales_offset=0 will be used)

    // Cast to appropriate types
    const auto* packed_weights = reinterpret_cast<const uint8_t*>(QuantBData);
    float* act_output = reinterpret_cast<float*>(C);

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
                    // Calculate LUT offsets with 4-byte stride (per activation) for consistent access.
                    const size_t qlut_offset = K * ine11 * 4;
                    const size_t lut_scales_offset = lut_scales_size * ine11;

                    // Calculate output offset
                    const size_t dst_offset = OutputRows * ine11 + ichunk0 * ChunkSize0;

                    // Call the dispatch function to compute this tile.
                    // We pass one batch item at a time (M=1) and ChunkSize0 output features.
                    // TotalN is passed specifically to allow the kernel to find the correct
                    // parameters (bm, tiles) used during weight packing.
                    Dispatch->ComputeGemm(
                        packed_weights + w_offset,       // Weight tile
                        QuantBScale + scales_offset,     // Weight scales for this tile
                        qlut + qlut_offset,              // LUT for this batch row
                        lut_scales + lut_scales_offset,  // LUT scales
                        lut_biases + lut_scales_offset,  // LUT biases
                        act_output + dst_offset,         // Output location
                        static_cast<int>(K),             // K dimension
                        static_cast<int>(1),             // M dimension (batch size = 1)
                        static_cast<int>(ir0_end - ir0_start), // N dimension (output features in chunk)
                        static_cast<int>(N),             // TotalN (total output features in weights)
                        BlkLen,                          // Weight quantization group size
                        HasZeroPoint                     // Whether zero points are used
                    );
                }
            }
        }
    );
}
