/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    halfgemm_kernel_power10.cpp

Abstract:

    This module implements the half-precision (FP16) matrix/matrix multiply
    operation (HalfGEMM) using POWER10 MMA instructions.

    Architecture & Cache Design:
    - 8 Hardware Accumulator Quads (acc0..acc7) utilized for an 8 rows x 16 cols tile.
    - PackNeeded = true: CopyPackB interleaves K-pairs into 4 contiguous vectors per step,
      eliminating all vec_perm from the inner compute loop.
    - B panel (K=512 x N=128 x 2 bytes) = 131 KB, fits in 512 KB L2 slice.
    - A panel (M=8 x K=512 x 2 bytes) = 8 KB, hot in L1 across all N-column tiles.
    - Inner compute loop: 4 streaming vector loads + 8 xvf16ger2pp per K-pair.
    - All K accumulated inside MMA accumulators; C written exactly once per output tile.

--*/

#include "mlasi.h"
#include "halfgemm.h"
#include <altivec.h>
#undef vector
#undef pixel
#undef bool

#include <cstring>
#include <algorithm>
#if defined(__linux__)
#include <alloca.h>
#endif

typedef __vector unsigned char vec_t;
typedef __vector unsigned short vec_h;
typedef __vector unsigned int vec_w;

//
// Kernel traits structure for POWER10 MMA FP16 GEMM.
//
// PackNeeded = true routes through MlasHalfGemmOperation (the packing driver):
// - CopyPackB pre-packs B into 16-column panels with K-pairs interleaved as 4 contiguous
//   vectors per step, removing all vec_perm from the MMA inner loop.
// - The driver manages K-tiling (Strides.K=512), accumulating partial sums via ZeroMode.
//   Each K-tile's MMA accumulators are fully drained to C once, then the next tile
//   accumulates on top -- zero intermediate FP16 round-trips within a tile.
// - A panel (M x K slice) is packed once and reused across all N-column iterations.
//
struct MLAS_HALF_GEMM_KERNEL_POWER10 {
    static constexpr bool PackNeeded = true;
    static constexpr size_t KernelMaxM = 8;
    // xvf16ger2pp consumes 2 K-elements per call; packing aligns K to 2.
    static constexpr size_t PackedK = 2;
    // Strides{M, N, K}: controls the panel sizes in the packing driver.
    // K=2048: accommodates large linear layers (e.g., ResNet-50 classification K=2048) in a single pass.
    // N=128: B panel in N-direction = 128 columns, 8 N-tiles for N=1024.
    // M=8:   One full MMA tile per M step.
    static constexpr MLAS_HALF_GEMM_STRIDES Strides{8, 128, 2048};
};

//
// Pre-packed B offset calculation for POWER10 MMA layout.
// B is packed in 16-column panels, each spanning AlignedK rows.
//
template<>
MLAS_FORCEINLINE
const _mlas_fp16_*
MlasHalfGemmPackedBOffset<MLAS_HALF_GEMM_KERNEL_POWER10>(
    const _mlas_fp16_* PackedB,
    size_t DimN,
    size_t DimK,
    size_t StartN,
    size_t StartK)
{
    MLAS_UNREFERENCED_PARAMETER(DimN);
    const size_t AlignedK = (DimK + 1) & ~1;
    return PackedB + (StartN / 16) * AlignedK * 16 + StartK * 16;
}

template<>
MLAS_FORCEINLINE
size_t
MlasHalfGemmPackedBLeadingDim<MLAS_HALF_GEMM_KERNEL_POWER10>(
    size_t DimN,
    size_t DimK)
{
    MLAS_UNREFERENCED_PARAMETER(DimN);
    MLAS_UNREFERENCED_PARAMETER(DimK);
    return 16;
}

//
// PackA: convert fp32 A rows to fp16, storing row-major with K padded to PackedK.
//
template<>
void
MlasHalfGemmConvertPackA<MLAS_HALF_GEMM_KERNEL_POWER10>(
    _mlas_fp16_* D,
    const float* A,
    size_t lda,
    size_t CountM,
    size_t CountK
    )
{
    const size_t alignedK = (CountK + MLAS_HALF_GEMM_KERNEL_POWER10::PackedK - 1) &
                            ~(MLAS_HALF_GEMM_KERNEL_POWER10::PackedK - 1);

    for (size_t m = 0; m < CountM; m++) {
        _mlas_fp16_* D_row = D + m * alignedK;
        for (size_t k = 0; k < CountK; k++) {
            D_row[k] = MLAS_Float2Half(A[m * lda + k]);
        }
        for (size_t kp = CountK; kp < alignedK; kp++) {
            D_row[kp] = 0;
        }
    }
}

//
// PackB (FP32 -> FP16): convert fp32 B to fp16, packing into 16-column panels
// with K-pairs interleaved as 4 contiguous vectors for direct vector loads in the MMA inner loop.
//
template<>
void
MlasHalfGemmConvertPackB<MLAS_HALF_GEMM_KERNEL_POWER10>(
    _mlas_fp16_* D,
    const float* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
    )
{
    uint16_t* dest = reinterpret_cast<uint16_t*>(D);
    const size_t AlignedK = (CountK + 1) & ~1;

    for (size_t n = 0; n < CountN; n += 16) {
        size_t cols = std::min(CountN - n, size_t(16));
        const float* src_col = B + n;

        for (size_t k = 0; k < AlignedK; k += 2) {
            const float* row0 = src_col + k * ldb;
            const float* row1 = row0 + ldb;

            for (size_t q = 0; q < 4; ++q) {
                size_t c_base = q * 4;
                for (size_t c = 0; c < 4; ++c) {
                    size_t col_idx = c_base + c;
                    uint16_t v0 = 0;
                    uint16_t v1 = 0;
                    if (col_idx < cols) {
                        v0 = (k < CountK) ? MLAS_Float2Half(row0[col_idx]) : 0u;
                        v1 = (k + 1 < CountK) ? MLAS_Float2Half(row1[col_idx]) : 0u;
                    }
                    *dest++ = v0;
                    *dest++ = v1;
                }
            }
        }
    }
}

//
// CopyPackB (FP16 -> FP16): pack B into 16-column panels with K-pairs
// interleaved as 4 contiguous vectors. This is the hot path for fp16 B input.
// After packing, the MMA inner loop becomes: 4 vec loads + 8 xvf16ger2pp per K-pair,
// with zero vec_perm in the compute loop.
//
template<>
void
MlasHalfGemmCopyPackB<MLAS_HALF_GEMM_KERNEL_POWER10>(
    _mlas_fp16_* D,
    const _mlas_fp16_* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
    )
{
    typedef __vector unsigned char vec_t;
    typedef __vector unsigned short vec_h;
    const vec_t perm_lo = {0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23};
    const vec_t perm_hi = {8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31};

    uint16_t* dest = reinterpret_cast<uint16_t*>(D);
    const size_t AlignedK = (CountK + 1) & ~1;

    for (size_t n = 0; n < CountN; n += 16) {
        size_t cols = std::min(CountN - n, size_t(16));
        const _mlas_fp16_* src_col = B + n;

        if (cols >= 16) {
            vec_t* dest_vec = reinterpret_cast<vec_t*>(dest);
            for (size_t k = 0; k < AlignedK; k += 2) {
                const _mlas_fp16_* row0 = src_col + k * ldb;
                const _mlas_fp16_* row1 = row0 + ldb;

                vec_h r0_lo = vec_xl(0, reinterpret_cast<const unsigned short*>(row0));
                vec_h r0_hi = vec_xl(16, reinterpret_cast<const unsigned short*>(row0));
                vec_h r1_lo = (k + 1 < CountK) ? vec_xl(0, reinterpret_cast<const unsigned short*>(row1)) : vec_splats((unsigned short)0);
                vec_h r1_hi = (k + 1 < CountK) ? vec_xl(16, reinterpret_cast<const unsigned short*>(row1)) : vec_splats((unsigned short)0);

                dest_vec[0] = vec_perm((vec_t)r0_lo, (vec_t)r1_lo, perm_lo);
                dest_vec[1] = vec_perm((vec_t)r0_lo, (vec_t)r1_lo, perm_hi);
                dest_vec[2] = vec_perm((vec_t)r0_hi, (vec_t)r1_hi, perm_lo);
                dest_vec[3] = vec_perm((vec_t)r0_hi, (vec_t)r1_hi, perm_hi);
                dest_vec += 4;
            }
            dest = reinterpret_cast<uint16_t*>(dest_vec);
        } else {
            for (size_t k = 0; k < AlignedK; k += 2) {
                const _mlas_fp16_* row0 = src_col + k * ldb;
                const _mlas_fp16_* row1 = row0 + ldb;

                for (size_t q = 0; q < 4; ++q) {
                    size_t c_base = q * 4;
                    for (size_t c = 0; c < 4; ++c) {
                        size_t col_idx = c_base + c;
                        uint16_t v0 = 0;
                        uint16_t v1 = 0;
                        if (col_idx < cols) {
                            v0 = (k < CountK) ? row0[col_idx] : 0u;
                            v1 = (k + 1 < CountK) ? row1[col_idx] : 0u;
                        }
                        *dest++ = v0;
                        *dest++ = v1;
                    }
                }
            }
        }
    }
}

//
// CopyPackB_Transposed: pack transposed B (shape [N, K], row-major with leading dimension ldb)
// into 16-column panels with K-pairs interleaved as 4 contiguous vectors.
// Element at (row=n+c, col=k) in B is at B[(n+c)*ldb + k].
//
template<>
void
MlasHalfGemmCopyPackB_Transposed<MLAS_HALF_GEMM_KERNEL_POWER10>(
    _mlas_fp16_* D,
    const _mlas_fp16_* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
    )
{
    uint16_t* dest = reinterpret_cast<uint16_t*>(D);
    const size_t AlignedK = (CountK + 1) & ~1;

    for (size_t n = 0; n < CountN; n += 16) {
        size_t cols = std::min(CountN - n, size_t(16));

        if (cols >= 16) {
            uint32_t* dest32 = reinterpret_cast<uint32_t*>(dest);
            for (size_t k = 0; k < AlignedK; k += 2) {
                if (k + 1 < CountK) {
                    for (size_t c = 0; c < 16; ++c) {
                        *dest32++ = *reinterpret_cast<const uint32_t*>(B + (n + c) * ldb + k);
                    }
                } else {
                    for (size_t c = 0; c < 16; ++c) {
                        uint16_t v0 = (k < CountK) ? B[(n + c) * ldb + k] : 0u;
                        *dest32++ = static_cast<uint32_t>(v0);
                    }
                }
            }
            dest = reinterpret_cast<uint16_t*>(dest32);
        } else {
            for (size_t k = 0; k < AlignedK; k += 2) {
                for (size_t q = 0; q < 4; ++q) {
                    size_t c_base = q * 4;
                    for (size_t c = 0; c < 4; ++c) {
                        size_t col_idx = c_base + c;
                        uint16_t v0 = 0;
                        uint16_t v1 = 0;
                        if (col_idx < cols) {
                            const _mlas_fp16_* row = B + (n + col_idx) * ldb;
                            v0 = (k < CountK) ? row[k] : 0u;
                            v1 = (k + 1 < CountK) ? row[k + 1] : 0u;
                        }
                        *dest++ = v0;
                        *dest++ = v1;
                    }
                }
            }
        }
    }
}

//
// ConvertPackB_Transposed: convert transposed fp32 B (shape [N, K], row-major with leading dimension ldb)
// to fp16, packing into 16-column panels with K-pairs interleaved as 4 contiguous vectors.
//
template<>
void
MlasHalfGemmConvertPackB_Transposed<MLAS_HALF_GEMM_KERNEL_POWER10>(
    _mlas_fp16_* D,
    const float* B,
    size_t ldb,
    size_t CountN,
    size_t CountK
    )
{
    uint16_t* dest = reinterpret_cast<uint16_t*>(D);
    const size_t AlignedK = (CountK + 1) & ~1;

    for (size_t n = 0; n < CountN; n += 16) {
        size_t cols = std::min(CountN - n, size_t(16));

        for (size_t k = 0; k < AlignedK; k += 2) {
            for (size_t q = 0; q < 4; ++q) {
                size_t c_base = q * 4;
                for (size_t c = 0; c < 4; ++c) {
                    size_t col_idx = c_base + c;
                    uint16_t v0 = 0;
                    uint16_t v1 = 0;
                    if (col_idx < cols) {
                        const float* row = B + (n + col_idx) * ldb;
                        v0 = (k < CountK) ? MLAS_Float2Half(row[k]) : 0u;
                        v1 = (k + 1 < CountK) ? MLAS_Float2Half(row[k + 1]) : 0u;
                    }
                    *dest++ = v0;
                    *dest++ = v1;
                }
            }
        }
    }
}


//
// MlasHalfGemmKernel specialization: POWER10 MMA FP16 GEMM.
//
// B is pre-packed by CopyPackB into 16-col panels: for each K-pair, 4 consecutive
// vec_t vectors hold the interleaved [row0_lo, row0_hi, row1_lo, row1_hi] data.
// The inner compute loop is then: 4 sequential vec loads + 8 MMA ops -- identical
// in structure to the SGEMM kernel.
//
// The driver (MlasHalfGemmOperation) manages K-tiling via ZeroMode:
//   ZeroMode=true  on the first K-tile: accumulators start from zero.
//   ZeroMode=false on subsequent K-tiles: output C is loaded and added.
// Within each tile, all CountK pairs are fully accumulated before any store.
//
template<>
void
MlasHalfGemmKernel<MLAS_HALF_GEMM_KERNEL_POWER10>(
    const size_t CountM,
    const size_t CountN,
    const size_t CountK,
    _mlas_fp16_* C,
    size_t ldc,
    const _mlas_fp16_* Bias,
    const _mlas_fp16_* A,
    const size_t lda,
    const _mlas_fp16_* B,
    const size_t ldb,
    const bool ZeroMode
    )
{
    MLAS_UNREFERENCED_PARAMETER(ldb);
    const size_t Rows = std::min(CountM, MLAS_HALF_GEMM_KERNEL_POWER10::KernelMaxM);
    if (Rows == 0 || CountN == 0 || CountK == 0) {
        return;
    }

#define PREFETCH_ADDR(addr) \
    asm volatile("dcbt 0, %0" ::"r"(addr) : "memory");

#define STORE_ROW_16_FP16(row_ptr, r0, r1, r2, r3) \
    do { \
        __vector float _v0 = (r0); \
        __vector float _v1 = (r1); \
        __vector float _v2 = (r2); \
        __vector float _v3 = (r3); \
        if (Bias != nullptr && ZeroMode) { \
            _v0 = vec_add(_v0, bias_f0); \
            _v1 = vec_add(_v1, bias_f1); \
            _v2 = vec_add(_v2, bias_f2); \
            _v3 = vec_add(_v3, bias_f3); \
        } \
        alignas(16) float _res[16]; \
        *reinterpret_cast<__vector float*>(&_res[0]) = _v0; \
        *reinterpret_cast<__vector float*>(&_res[4]) = _v1; \
        *reinterpret_cast<__vector float*>(&_res[8]) = _v2; \
        *reinterpret_cast<__vector float*>(&_res[12]) = _v3; \
        _mlas_fp16_* _dst = (row_ptr); \
        if (!ZeroMode) { \
            for (size_t _c = 0; _c < 16; ++_c) { \
                _dst[_c] = MLAS_Float2Half(_res[_c] + MLAS_Half2Float(_dst[_c])); \
            } \
        } else { \
            for (size_t _c = 0; _c < 16; ++_c) { \
                _dst[_c] = MLAS_Float2Half(_res[_c]); \
            } \
        } \
    } while (0)

    //
    // Pack A rows into a contiguous vector buffer aligned for MMA.
    // Each K-pair (k, k+1) is stored as a vec_w with one element per row:
    //   [row0_fp16_pair | row1_fp16_pair | row2_fp16_pair | row3_fp16_pair]
    //
    const size_t AlignedK = (CountK + MLAS_HALF_GEMM_KERNEL_POWER10::PackedK - 1) &
                            ~(MLAS_HALF_GEMM_KERNEL_POWER10::PackedK - 1);
    const size_t K_pairs = AlignedK / 2;

    vec_t* PackA0 = reinterpret_cast<vec_t*>(
        (reinterpret_cast<uintptr_t>(alloca(sizeof(vec_t) * K_pairs + 15)) + 15) & ~15);
    vec_t* PackA1 = reinterpret_cast<vec_t*>(
        (reinterpret_cast<uintptr_t>(alloca(sizeof(vec_t) * K_pairs + 15)) + 15) & ~15);

    for (size_t k2 = 0; k2 < K_pairs; ++k2) {
        const size_t k = k2 * 2;
        if (k + 1 < CountK) {
            uint32_t* p0 = reinterpret_cast<uint32_t*>(&PackA0[k2]);
            p0[0] = *reinterpret_cast<const uint32_t*>(A + 0 * lda + k);
            p0[1] = (Rows > 1) ? *reinterpret_cast<const uint32_t*>(A + 1 * lda + k) : 0;
            p0[2] = (Rows > 2) ? *reinterpret_cast<const uint32_t*>(A + 2 * lda + k) : 0;
            p0[3] = (Rows > 3) ? *reinterpret_cast<const uint32_t*>(A + 3 * lda + k) : 0;

            if (Rows > 4) {
                uint32_t* p1 = reinterpret_cast<uint32_t*>(&PackA1[k2]);
                p1[0] = *reinterpret_cast<const uint32_t*>(A + 4 * lda + k);
                p1[1] = (Rows > 5) ? *reinterpret_cast<const uint32_t*>(A + 5 * lda + k) : 0;
                p1[2] = (Rows > 6) ? *reinterpret_cast<const uint32_t*>(A + 6 * lda + k) : 0;
                p1[3] = (Rows > 7) ? *reinterpret_cast<const uint32_t*>(A + 7 * lda + k) : 0;
            }
        } else {
            uint16_t* p0 = reinterpret_cast<uint16_t*>(&PackA0[k2]);
            p0[0] = (k < CountK) ? A[0 * lda + k] : 0;
            p0[1] = 0;
            p0[2] = (Rows > 1 && k < CountK) ? A[1 * lda + k] : 0;
            p0[3] = 0;
            p0[4] = (Rows > 2 && k < CountK) ? A[2 * lda + k] : 0;
            p0[5] = 0;
            p0[6] = (Rows > 3 && k < CountK) ? A[3 * lda + k] : 0;
            p0[7] = 0;

            if (Rows > 4) {
                uint16_t* p1 = reinterpret_cast<uint16_t*>(&PackA1[k2]);
                p1[0] = (k < CountK) ? A[4 * lda + k] : 0;
                p1[1] = 0;
                p1[2] = (Rows > 5 && k < CountK) ? A[5 * lda + k] : 0;
                p1[3] = 0;
                p1[4] = (Rows > 6 && k < CountK) ? A[6 * lda + k] : 0;
                p1[5] = 0;
                p1[6] = (Rows > 7 && k < CountK) ? A[7 * lda + k] : 0;
                p1[7] = 0;
            }
        }
    }

    //
    // Loop over columns of B in panels of 16.
    // B is packed: for each 16-col panel, K-pairs are stored as 4 consecutive
    // vec_t vectors: [k_pair_vb0, k_pair_vb1, k_pair_vb2, k_pair_vb3].
    //

    for (size_t i = 0; i < K_pairs; i += 8) {
        PREFETCH_ADDR(PackA0 + i);
    }

    for (size_t nCol = 0; nCol < CountN; nCol += 16) {
        __vector_quad acc[8];
        __builtin_mma_xxsetaccz(&acc[0]);
        __builtin_mma_xxsetaccz(&acc[1]);
        __builtin_mma_xxsetaccz(&acc[2]);
        __builtin_mma_xxsetaccz(&acc[3]);
        if (Rows > 4) {
            __builtin_mma_xxsetaccz(&acc[4]);
            __builtin_mma_xxsetaccz(&acc[5]);
            __builtin_mma_xxsetaccz(&acc[6]);
            __builtin_mma_xxsetaccz(&acc[7]);
        }

        const size_t cols = std::min(CountN - nCol, size_t(16));

        //
        // For packed B: each K-pair occupies 4 vec_t = 64 bytes.
        // Advance pb by 4 per K-pair.
        //
        // Unrolled 8x loop: processes 8 K-pairs (16 K elements) per iteration.
        //
        if (cols >= 16) {
            const vec_t* pb_base = reinterpret_cast<const vec_t*>(B) + (nCol / 16) * K_pairs * 4;
            const vec_t* pb_k = pb_base;

            size_t k2 = 0;

            // 8-way unrolled main loop
            for (; k2 + 8 <= K_pairs; k2 += 8) {
                PREFETCH_ADDR(pb_k + 64);
                PREFETCH_ADDR(pb_k + 128);
                const vec_t* pa0_ptr = PackA0 + k2;
                const vec_t* pa1_ptr = (Rows > 4) ? (PackA1 + k2) : nullptr;

                // K-pair 0
                vec_t vb0 = pb_k[0]; vec_t vb1 = pb_k[1]; vec_t vb2 = pb_k[2]; vec_t vb3 = pb_k[3]; pb_k += 4;
                vec_t va0 = pa0_ptr[0];
                __builtin_mma_xvf16ger2pp(&acc[0], va0, vb0);
                __builtin_mma_xvf16ger2pp(&acc[1], va0, vb1);
                __builtin_mma_xvf16ger2pp(&acc[2], va0, vb2);
                __builtin_mma_xvf16ger2pp(&acc[3], va0, vb3);
                if (Rows > 4) {
                    vec_t va1 = pa1_ptr[0];
                    __builtin_mma_xvf16ger2pp(&acc[4], va1, vb0);
                    __builtin_mma_xvf16ger2pp(&acc[5], va1, vb1);
                    __builtin_mma_xvf16ger2pp(&acc[6], va1, vb2);
                    __builtin_mma_xvf16ger2pp(&acc[7], va1, vb3);
                }
                // K-pair 1
                vb0 = pb_k[0]; vb1 = pb_k[1]; vb2 = pb_k[2]; vb3 = pb_k[3]; pb_k += 4;
                va0 = pa0_ptr[1];
                __builtin_mma_xvf16ger2pp(&acc[0], va0, vb0);
                __builtin_mma_xvf16ger2pp(&acc[1], va0, vb1);
                __builtin_mma_xvf16ger2pp(&acc[2], va0, vb2);
                __builtin_mma_xvf16ger2pp(&acc[3], va0, vb3);
                if (Rows > 4) {
                    vec_t va1 = pa1_ptr[1];
                    __builtin_mma_xvf16ger2pp(&acc[4], va1, vb0);
                    __builtin_mma_xvf16ger2pp(&acc[5], va1, vb1);
                    __builtin_mma_xvf16ger2pp(&acc[6], va1, vb2);
                    __builtin_mma_xvf16ger2pp(&acc[7], va1, vb3);
                }
                // K-pair 2
                vb0 = pb_k[0]; vb1 = pb_k[1]; vb2 = pb_k[2]; vb3 = pb_k[3]; pb_k += 4;
                va0 = pa0_ptr[2];
                __builtin_mma_xvf16ger2pp(&acc[0], va0, vb0);
                __builtin_mma_xvf16ger2pp(&acc[1], va0, vb1);
                __builtin_mma_xvf16ger2pp(&acc[2], va0, vb2);
                __builtin_mma_xvf16ger2pp(&acc[3], va0, vb3);
                if (Rows > 4) {
                    vec_t va1 = pa1_ptr[2];
                    __builtin_mma_xvf16ger2pp(&acc[4], va1, vb0);
                    __builtin_mma_xvf16ger2pp(&acc[5], va1, vb1);
                    __builtin_mma_xvf16ger2pp(&acc[6], va1, vb2);
                    __builtin_mma_xvf16ger2pp(&acc[7], va1, vb3);
                }
                // K-pair 3
                vb0 = pb_k[0]; vb1 = pb_k[1]; vb2 = pb_k[2]; vb3 = pb_k[3]; pb_k += 4;
                va0 = pa0_ptr[3];
                __builtin_mma_xvf16ger2pp(&acc[0], va0, vb0);
                __builtin_mma_xvf16ger2pp(&acc[1], va0, vb1);
                __builtin_mma_xvf16ger2pp(&acc[2], va0, vb2);
                __builtin_mma_xvf16ger2pp(&acc[3], va0, vb3);
                if (Rows > 4) {
                    vec_t va1 = pa1_ptr[3];
                    __builtin_mma_xvf16ger2pp(&acc[4], va1, vb0);
                    __builtin_mma_xvf16ger2pp(&acc[5], va1, vb1);
                    __builtin_mma_xvf16ger2pp(&acc[6], va1, vb2);
                    __builtin_mma_xvf16ger2pp(&acc[7], va1, vb3);
                }
                // K-pair 4
                vb0 = pb_k[0]; vb1 = pb_k[1]; vb2 = pb_k[2]; vb3 = pb_k[3]; pb_k += 4;
                va0 = pa0_ptr[4];
                __builtin_mma_xvf16ger2pp(&acc[0], va0, vb0);
                __builtin_mma_xvf16ger2pp(&acc[1], va0, vb1);
                __builtin_mma_xvf16ger2pp(&acc[2], va0, vb2);
                __builtin_mma_xvf16ger2pp(&acc[3], va0, vb3);
                if (Rows > 4) {
                    vec_t va1 = pa1_ptr[4];
                    __builtin_mma_xvf16ger2pp(&acc[4], va1, vb0);
                    __builtin_mma_xvf16ger2pp(&acc[5], va1, vb1);
                    __builtin_mma_xvf16ger2pp(&acc[6], va1, vb2);
                    __builtin_mma_xvf16ger2pp(&acc[7], va1, vb3);
                }
                // K-pair 5
                vb0 = pb_k[0]; vb1 = pb_k[1]; vb2 = pb_k[2]; vb3 = pb_k[3]; pb_k += 4;
                va0 = pa0_ptr[5];
                __builtin_mma_xvf16ger2pp(&acc[0], va0, vb0);
                __builtin_mma_xvf16ger2pp(&acc[1], va0, vb1);
                __builtin_mma_xvf16ger2pp(&acc[2], va0, vb2);
                __builtin_mma_xvf16ger2pp(&acc[3], va0, vb3);
                if (Rows > 4) {
                    vec_t va1 = pa1_ptr[5];
                    __builtin_mma_xvf16ger2pp(&acc[4], va1, vb0);
                    __builtin_mma_xvf16ger2pp(&acc[5], va1, vb1);
                    __builtin_mma_xvf16ger2pp(&acc[6], va1, vb2);
                    __builtin_mma_xvf16ger2pp(&acc[7], va1, vb3);
                }
                // K-pair 6
                vb0 = pb_k[0]; vb1 = pb_k[1]; vb2 = pb_k[2]; vb3 = pb_k[3]; pb_k += 4;
                va0 = pa0_ptr[6];
                __builtin_mma_xvf16ger2pp(&acc[0], va0, vb0);
                __builtin_mma_xvf16ger2pp(&acc[1], va0, vb1);
                __builtin_mma_xvf16ger2pp(&acc[2], va0, vb2);
                __builtin_mma_xvf16ger2pp(&acc[3], va0, vb3);
                if (Rows > 4) {
                    vec_t va1 = pa1_ptr[6];
                    __builtin_mma_xvf16ger2pp(&acc[4], va1, vb0);
                    __builtin_mma_xvf16ger2pp(&acc[5], va1, vb1);
                    __builtin_mma_xvf16ger2pp(&acc[6], va1, vb2);
                    __builtin_mma_xvf16ger2pp(&acc[7], va1, vb3);
                }
                // K-pair 7
                vb0 = pb_k[0]; vb1 = pb_k[1]; vb2 = pb_k[2]; vb3 = pb_k[3]; pb_k += 4;
                va0 = pa0_ptr[7];
                __builtin_mma_xvf16ger2pp(&acc[0], va0, vb0);
                __builtin_mma_xvf16ger2pp(&acc[1], va0, vb1);
                __builtin_mma_xvf16ger2pp(&acc[2], va0, vb2);
                __builtin_mma_xvf16ger2pp(&acc[3], va0, vb3);
                if (Rows > 4) {
                    vec_t va1 = pa1_ptr[7];
                    __builtin_mma_xvf16ger2pp(&acc[4], va1, vb0);
                    __builtin_mma_xvf16ger2pp(&acc[5], va1, vb1);
                    __builtin_mma_xvf16ger2pp(&acc[6], va1, vb2);
                    __builtin_mma_xvf16ger2pp(&acc[7], va1, vb3);
                }
            }

            // Scalar remainder for any trailing K-pairs
            for (; k2 < K_pairs; ++k2) {
                vec_t vb0 = pb_k[0]; vec_t vb1 = pb_k[1]; vec_t vb2 = pb_k[2]; vec_t vb3 = pb_k[3]; pb_k += 4;
                vec_t va0 = PackA0[k2];
                __builtin_mma_xvf16ger2pp(&acc[0], va0, vb0);
                __builtin_mma_xvf16ger2pp(&acc[1], va0, vb1);
                __builtin_mma_xvf16ger2pp(&acc[2], va0, vb2);
                __builtin_mma_xvf16ger2pp(&acc[3], va0, vb3);
                if (Rows > 4) {
                    vec_t va1 = PackA1[k2];
                    __builtin_mma_xvf16ger2pp(&acc[4], va1, vb0);
                    __builtin_mma_xvf16ger2pp(&acc[5], va1, vb1);
                    __builtin_mma_xvf16ger2pp(&acc[6], va1, vb2);
                    __builtin_mma_xvf16ger2pp(&acc[7], va1, vb3);
                }
            }



            __vector float bias_f0 = vec_splats(0.0f);
            __vector float bias_f1 = vec_splats(0.0f);
            __vector float bias_f2 = vec_splats(0.0f);
            __vector float bias_f3 = vec_splats(0.0f);
            if (Bias != nullptr && ZeroMode) {
                alignas(16) float _b[16];
                for (size_t _c = 0; _c < 16; ++_c) {
                    _b[_c] = MLAS_Half2Float(Bias[nCol + _c]);
                }
                bias_f0 = *reinterpret_cast<__vector float*>(&_b[0]);
                bias_f1 = *reinterpret_cast<__vector float*>(&_b[4]);
                bias_f2 = *reinterpret_cast<__vector float*>(&_b[8]);
                bias_f3 = *reinterpret_cast<__vector float*>(&_b[12]);
            }

            __vector float res0[4], res1[4], res2[4], res3[4];
            __builtin_mma_disassemble_acc(res0, &acc[0]);
            __builtin_mma_disassemble_acc(res1, &acc[1]);
            __builtin_mma_disassemble_acc(res2, &acc[2]);
            __builtin_mma_disassemble_acc(res3, &acc[3]);

            size_t rows_top = std::min(Rows, size_t(4));
            for (size_t r = 0; r < rows_top; ++r) {
                STORE_ROW_16_FP16(C + r * ldc + nCol, res0[r], res1[r], res2[r], res3[r]);
            }

            if (Rows > 4) {
                __vector float res4[4], res5[4], res6[4], res7[4];
                __builtin_mma_disassemble_acc(res4, &acc[4]);
                __builtin_mma_disassemble_acc(res5, &acc[5]);
                __builtin_mma_disassemble_acc(res6, &acc[6]);
                __builtin_mma_disassemble_acc(res7, &acc[7]);

                for (size_t r = 4; r < Rows; ++r) {
                    STORE_ROW_16_FP16(C + r * ldc + nCol, res4[r - 4], res5[r - 4], res6[r - 4], res7[r - 4]);
                }
            }
        } else {
            //
            // Partial column tail (cols < 16): use packed B, scalar store.
            //
            const vec_t* pb_base = reinterpret_cast<const vec_t*>(B) + (nCol / 16) * K_pairs * 4;
            const vec_t* pb_k = pb_base;

            for (size_t k2 = 0; k2 < K_pairs; ++k2) {
                vec_t vb0 = pb_k[0]; vec_t vb1 = pb_k[1]; vec_t vb2 = pb_k[2]; vec_t vb3 = pb_k[3]; pb_k += 4;
                vec_t va0 = PackA0[k2];
                __builtin_mma_xvf16ger2pp(&acc[0], va0, vb0);
                __builtin_mma_xvf16ger2pp(&acc[1], va0, vb1);
                __builtin_mma_xvf16ger2pp(&acc[2], va0, vb2);
                __builtin_mma_xvf16ger2pp(&acc[3], va0, vb3);
                if (Rows > 4) {
                    vec_t va1 = PackA1[k2];
                    __builtin_mma_xvf16ger2pp(&acc[4], va1, vb0);
                    __builtin_mma_xvf16ger2pp(&acc[5], va1, vb1);
                    __builtin_mma_xvf16ger2pp(&acc[6], va1, vb2);
                    __builtin_mma_xvf16ger2pp(&acc[7], va1, vb3);
                }
            }

            for (int q = 0; q < 4; ++q) {
                size_t c_offset = static_cast<size_t>(q) * 4;
                if (nCol + c_offset >= CountN) {
                    continue;
                }
                size_t valid_cols = std::min(CountN - (nCol + c_offset), size_t(4));

                __vector float res_top[4];
                __builtin_mma_disassemble_acc(res_top, &acc[q]);

                size_t rows_top = std::min(Rows, size_t(4));
                for (size_t r = 0; r < rows_top; ++r) {
                    const float* res_floats = (const float*)&res_top[r];
                    _mlas_fp16_* c_dest = C + r * ldc + nCol + c_offset;
                    for (size_t c = 0; c < valid_cols; ++c) {
                        float val = res_floats[c];
                        if (ZeroMode && Bias) { val += MLAS_Half2Float(Bias[nCol + c_offset + c]); }
                        if (!ZeroMode) { val += MLAS_Half2Float(c_dest[c]); }
                        c_dest[c] = MLAS_Float2Half(val);
                    }
                }

                if (Rows > 4) {
                    __vector float res_bot[4];
                    __builtin_mma_disassemble_acc(res_bot, &acc[4 + q]);
                    for (size_t r = 4; r < Rows; ++r) {
                        const float* res_floats = (const float*)&res_bot[r - 4];
                        _mlas_fp16_* c_dest = C + r * ldc + nCol + c_offset;
                        for (size_t c = 0; c < valid_cols; ++c) {
                            float val = res_floats[c];
                            if (ZeroMode && Bias) { val += MLAS_Half2Float(Bias[nCol + c_offset + c]); }
                            if (!ZeroMode) { val += MLAS_Half2Float(c_dest[c]); }
                            c_dest[c] = MLAS_Float2Half(val);
                        }
                    }
                }
            }
        }
    }
}

#undef PREFETCH_ADDR
#undef STORE_ROW_16_FP16

//
// Dispatch table for POWER10 HalfGEMM.
//
extern const MLAS_HALFGEMM_DISPATCH MlasHalfGemmDispatchPOWER10 = {
    MlasHalfGemmOperation<MLAS_HALF_GEMM_KERNEL_POWER10>,
    MlasHalfGemmCopyPackB<MLAS_HALF_GEMM_KERNEL_POWER10>,
    MlasHalfGemmConvertPackB<MLAS_HALF_GEMM_KERNEL_POWER10>,
    MLAS_HALF_GEMM_KERNEL_POWER10::PackedK,
    64, // StrideM = 64 (balanced ThreadPool task partitioning)
    0,
    MlasHalfGemmCopyPackB_Transposed<MLAS_HALF_GEMM_KERNEL_POWER10>,
    MlasHalfGemmConvertPackB_Transposed<MLAS_HALF_GEMM_KERNEL_POWER10>
};
