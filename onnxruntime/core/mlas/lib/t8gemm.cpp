// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/mlas/inc/mlas_t8.h"
#include "mlasi.h"
#include "t8gemm_kernel.h"

#include <cstring>

#if defined(MLAS_TARGET_ARM64)
#include <arm_neon.h>
#endif

namespace
{

#if defined(MLAS_TARGET_ARM64)
using Bits = uint32x4_t;

MLAS_FORCEINLINE Bits
Nt(Bits a, Bits b, Bits c)
{
    return vandq_u32(veorq_u32(a, b), veorq_u32(b, c));
}
MLAS_FORCEINLINE Bits
Pt(Bits a, Bits b, Bits c, Bits nt)
{
    return vbicq_u32(veorq_u32(veorq_u32(a, b), c), nt);
}
MLAS_FORCEINLINE Bits
Bq(Bits a, Bits b, Bits c)
{
    return vandq_u32(veorq_u32(a, b), vorrq_u32(a, c));
}
MLAS_FORCEINLINE Bits
Neg(Bits a, Bits b, Bits c)
{
    return veorq_u32(b, vandq_u32(veorq_u32(a, b), c));
}
MLAS_FORCEINLINE Bits
Pos(Bits a, Bits b, Bits c)
{
    return vbicq_u32(veorq_u32(a, c), b);
}
MLAS_FORCEINLINE Bits
Zero(Bits a, Bits b, Bits c)
{
    return vmvnq_u32(vorrq_u32(a, veorq_u32(b, c)));
}
MLAS_FORCEINLINE Bits
Carry(Bits a, Bits b, Bits c)
{
    return vorrq_u32(a, vandq_u32(b, c));
}

// The decode3 gate network used by the AVX512 reference and the MLX T8 implementation.
MLAS_FORCEINLINE void
Decode(Bits b0, Bits b1, Bits b2, Bits b3, Bits b4, Bits b5, Bits b6, Bits b7, Bits* z, Bits* p)
{
    Bits nt = Nt(b7, b6, b5), pt = Pt(b7, b6, b5, nt);
    b5 = Bq(b7, b6, b5);
    b6 = vandq_u32(b6, b7);
    Bits n = Neg(nt, pt, b4), q = Pos(nt, pt, b4);
    b4 = Carry(nt, pt, b4);
    nt = Neg(n, q, b3);
    pt = Pos(n, q, b3);
    b3 = Carry(n, q, b3);
    n = Neg(nt, pt, b2);
    q = Pos(nt, pt, b2);
    b2 = Carry(nt, pt, b2);
    nt = Neg(n, q, b1);
    pt = Pos(n, q, b1);
    b1 = Carry(n, q, b1);
    z[0] = Zero(nt, pt, b0);
    p[0] = Pos(nt, pt, b0);
    b0 = Carry(nt, pt, b0);

    nt = Nt(b6, b5, b4);
    pt = Pt(b6, b5, b4, nt);
    b4 = Bq(b6, b5, b4);
    b5 = vandq_u32(b5, b6);
    n = Neg(nt, pt, b3);
    q = Pos(nt, pt, b3);
    b3 = Carry(nt, pt, b3);
    nt = Neg(n, q, b2);
    pt = Pos(n, q, b2);
    b2 = Carry(n, q, b2);
    n = Neg(nt, pt, b1);
    q = Pos(nt, pt, b1);
    b1 = Carry(nt, pt, b1);
    z[1] = Zero(n, q, b0);
    p[1] = Pos(n, q, b0);
    b0 = Carry(n, q, b0);

    nt = Nt(b4, b3, b2);
    pt = Pt(b4, b3, b2, nt);
    b2 = Bq(b4, b3, b2);
    b3 = vandq_u32(b3, b4);
    n = Neg(nt, pt, b1);
    q = Pos(nt, pt, b1);
    b1 = Carry(nt, pt, b1);
    z[2] = Zero(n, q, b0);
    p[2] = Pos(n, q, b0);
    b0 = Carry(n, q, b0);

    nt = Nt(b3, b2, b1);
    pt = Pt(b3, b2, b1, nt);
    Bits n4 = Bq(b3, b2, b1);
    z[3] = Zero(nt, pt, b0);
    p[3] = Pos(nt, pt, b0);
    Bits p4 = Carry(nt, pt, b0);
    z[4] = vmvnq_u32(vorrq_u32(n4, p4));
    p[4] = p4;
}

void
PrepareA(size_t M, size_t K, const float* A, float* Workspace)
{
    for (size_t m = 0; m < M; ++m) {
        for (size_t g = 0; g < K; g += 128) {
            for (size_t i = 0; i < 32; i += 4) {
                float32x4_t a = vld1q_f32(A + m * K + g + i);
                float32x4_t b = vld1q_f32(A + m * K + g + 32 + i);
                float32x4_t c = vld1q_f32(A + m * K + g + 64 + i);
                float32x4_t d = vld1q_f32(A + m * K + g + 96 + i);
                float32x4_t ab0 = vzip1q_f32(a, b), ab1 = vzip2q_f32(a, b);
                float32x4_t cd0 = vzip1q_f32(c, d), cd1 = vzip2q_f32(c, d);
                float* dst = Workspace + m * K + g + i * 4;
                vst1q_f32(dst, vcombine_f32(vget_low_f32(ab0), vget_low_f32(cd0)));
                vst1q_f32(dst + 4, vcombine_f32(vget_high_f32(ab0), vget_high_f32(cd0)));
                vst1q_f32(dst + 8, vcombine_f32(vget_low_f32(ab1), vget_low_f32(cd1)));
                vst1q_f32(dst + 12, vcombine_f32(vget_high_f32(ab1), vget_high_f32(cd1)));
            }
        }
    }
}

void
ComputeUnit(size_t N, size_t K, size_t unit, const float* A, const uint32_t* Planes, const MLAS_FP16* Scales, float* C)
{
    const size_t sets = K / 32;
    const uint32_t* planes = Planes + unit * 8 * sets;
    const MLAS_FP16* scales = Scales + unit * (K / 128) * 5;
    float32x4_t total[5]{};
    const Bits sign_bit = vdupq_n_u32(0x80000000u);
    for (size_t s = 0; s < sets; s += 4) {
        Bits z[5], p[5], n[5];
        Decode(vld1q_u32(planes + s), vld1q_u32(planes + sets + s), vld1q_u32(planes + 2 * sets + s), vld1q_u32(planes + 3 * sets + s), vld1q_u32(planes + 4 * sets + s), vld1q_u32(planes + 5 * sets + s), vld1q_u32(planes + 6 * sets + s), vld1q_u32(planes + 7 * sets + s), z, p);
        for (size_t j = 0; j < 5; ++j) n[j] = vmvnq_u32(vorrq_u32(z[j], p[j]));
        float32x4_t sum[5]{};
        for (int i = 0; i < 32; ++i) {
            Bits x = vreinterpretq_u32_f32(vld1q_f32(A + s * 32 + i * 4));
            int32x4_t shift = vdupq_n_s32(31 - i);
            for (size_t j = 0; j < 5; ++j) {
                Bits nonzero = vmvnq_u32(vreinterpretq_u32_s32(
                    vshrq_n_s32(vreinterpretq_s32_u32(vshlq_u32(z[j], shift)), 31)
                ));
                Bits sign = vandq_u32(vshlq_u32(n[j], shift), sign_bit);
                sum[j] = vaddq_f32(sum[j], vreinterpretq_f32_u32(vandq_u32(veorq_u32(x, sign), nonzero)));
            }
        }
        for (size_t j = 0; j < 5; ++j) {
            total[j] = vfmaq_n_f32(total[j], sum[j], scales[(s / 4) * 5 + j].ToFloat());
        }
    }
    for (size_t j = 0; j < std::min(size_t{5}, N - unit * 5); ++j) {
        C[unit * 5 + j] = vaddvq_f32(total[j]);
    }
}
#else
struct T8OpsScalar {
    using V = uint32_t;

    static MLAS_FORCEINLINE V Nt(V a, V b, V c) { return (a ^ b) & (b ^ c); }
    static MLAS_FORCEINLINE V Pt(V a, V b, V c, V nt) { return (a ^ b ^ c) & ~nt; }
    static MLAS_FORCEINLINE V Bq(V a, V b, V c) { return (a ^ b) & (a | c); }
    static MLAS_FORCEINLINE V Neg(V a, V b, V c) { return b ^ ((a ^ b) & c); }
    static MLAS_FORCEINLINE V Pos(V a, V b, V c) { return (a ^ c) & ~b; }
    static MLAS_FORCEINLINE V Zero(V a, V b, V c) { return ~(a | (b ^ c)); }
    static MLAS_FORCEINLINE V Carry(V a, V b, V c) { return a | (b & c); }
    static MLAS_FORCEINLINE V And(V a, V b) { return a & b; }
    static MLAS_FORCEINLINE V NotOr(V a, V b) { return ~(a | b); }
};

// Decodes the four words of one group, whose eight planes are Stride words
// apart, into nonzero masks Nz[j * 8 + w] and negative masks Nz[j * 8 + 4 + w]
// (the MlasT8Pack masks layout).
void
DecodeGroup(const uint32_t* Planes, size_t Stride, uint32_t* Nz)
{
    for (size_t w = 0; w < 4; ++w) {
        uint32_t z[5], p[5];
        T8Decode<T8OpsScalar>(Planes[w], Planes[Stride + w], Planes[2 * Stride + w], Planes[3 * Stride + w], Planes[4 * Stride + w], Planes[5 * Stride + w], Planes[6 * Stride + w], Planes[7 * Stride + w], z, p);
        for (size_t j = 0; j < 5; ++j) {
            Nz[j * 8 + w] = ~z[j];
            Nz[j * 8 + 4 + w] = ~(z[j] | p[j]);
        }
    }
}

// Adds the signed inputs of one 128-input group selected by the masks
// (masks layout) into Sum, one input at a time in input order.
void
AccumulateGroup(const float* A, const uint32_t* Masks, float* Sum)
{
    for (size_t i = 0; i < 128; ++i) {
        uint32_t x;
        std::memcpy(&x, A + i, sizeof(x));
        for (size_t j = 0; j < 5; ++j) {
            const uint32_t nonzero = (Masks[j * 8 + i / 32] >> (i % 32)) & 1u;
            const uint32_t negative = (Masks[j * 8 + 4 + i / 32] >> (i % 32)) & 1u;
            const uint32_t bits = (x ^ (negative << 31)) & (0u - nonzero);
            float value;
            std::memcpy(&value, &bits, sizeof(value));
            Sum[j] += value;
        }
    }
}

void
ComputeUnit(size_t N, size_t K, size_t unit, const float* A, const uint32_t* Planes, const MLAS_FP16* Scales, float* C)
{
    const size_t sets = K / 32;
    const uint32_t* planes = Planes + unit * 8 * sets;
    float total[5]{};
    for (size_t g = 0; g < K / 128; ++g) {
        uint32_t masks[40];
        DecodeGroup(planes + g * 4, sets, masks);
        float sum[5]{};
        AccumulateGroup(A + g * 128, masks, sum);
        for (size_t j = 0; j < 5; ++j) {
            total[j] += sum[j] * Scales[(unit * (K / 128) + g) * 5 + j].ToFloat();
        }
    }
    for (size_t j = 0; j < std::min(size_t{5}, N - unit * 5); ++j) C[unit * 5 + j] = total[j];
}
#endif

template <size_t RowTile>
void
ComputePackedUnit(size_t M, size_t N, size_t K, size_t unit, const float* A, const uint32_t* Packed, const MLAS_FP16* Scales, float* C, MLAS_T8_PACKED_LAYOUT Layout)
{
    const size_t groups = K / 128;
    const size_t words = Layout == MLAS_T8_PACKED_LAYOUT::Tiled ? 32 : 40;
    const uint32_t* packed = Packed + unit * groups * words;
    const MLAS_FP16* scales = Scales + unit * groups * 5;
#if defined(MLAS_TARGET_ARM64)
    float32x4_t total[RowTile][5]{};
    const Bits sign_bit = vdupq_n_u32(0x80000000u);
    for (size_t g = 0; g < groups; ++g) {
        const uint32_t* block = packed + g * words;
        Bits nz[5], neg[5];
        if (Layout == MLAS_T8_PACKED_LAYOUT::Tiled) {
            Bits z[5], p[5];
            Decode(vld1q_u32(block), vld1q_u32(block + 4), vld1q_u32(block + 8), vld1q_u32(block + 12), vld1q_u32(block + 16), vld1q_u32(block + 20), vld1q_u32(block + 24), vld1q_u32(block + 28), z, p);
            for (size_t j = 0; j < 5; ++j) {
                nz[j] = vmvnq_u32(z[j]);
                neg[j] = vmvnq_u32(vorrq_u32(z[j], p[j]));
            }
        } else {
            for (size_t j = 0; j < 5; ++j) {
                nz[j] = vld1q_u32(block + j * 8);
                neg[j] = vld1q_u32(block + j * 8 + 4);
            }
        }
        float32x4_t sum[RowTile][5]{};
        for (int i = 0; i < 32; ++i) {
            Bits x[RowTile];
            for (size_t r = 0; r < RowTile; ++r) {
                if (r < M) x[r] = vreinterpretq_u32_f32(vld1q_f32(A + r * K + g * 128 + i * 4));
            }
            const int32x4_t shift = vdupq_n_s32(31 - i);
            for (size_t j = 0; j < 5; ++j) {
                const Bits mask = vreinterpretq_u32_s32(
                    vshrq_n_s32(vreinterpretq_s32_u32(vshlq_u32(nz[j], shift)), 31)
                );
                const Bits sign = vandq_u32(vshlq_u32(neg[j], shift), sign_bit);
                for (size_t r = 0; r < RowTile; ++r) {
                    if (r < M) sum[r][j] = vaddq_f32(sum[r][j], vreinterpretq_f32_u32(vandq_u32(veorq_u32(x[r], sign), mask)));
                }
            }
        }
        for (size_t r = 0; r < RowTile; ++r) {
            if (r < M) {
                for (size_t j = 0; j < 5; ++j) {
                    total[r][j] = vfmaq_n_f32(total[r][j], sum[r][j], scales[g * 5 + j].ToFloat());
                }
            }
        }
    }
    for (size_t r = 0; r < RowTile; ++r) {
        if (r < M) {
            for (size_t j = 0; j < std::min(size_t{5}, N - unit * 5); ++j) {
                C[r * N + unit * 5 + j] = vaddvq_f32(total[r][j]);
            }
        }
    }
#else
    const size_t rows = std::min(RowTile, M);
    float total[RowTile][5]{};
    for (size_t g = 0; g < groups; ++g) {
        const uint32_t* block = packed + g * words;
        uint32_t decoded[40];
        if (Layout == MLAS_T8_PACKED_LAYOUT::Tiled) {
            DecodeGroup(block, 4, decoded);
            block = decoded;
        }
        for (size_t r = 0; r < rows; ++r) {
            float sum[5]{};
            AccumulateGroup(A + r * K + g * 128, block, sum);
            for (size_t j = 0; j < 5; ++j) total[r][j] += sum[j] * scales[g * 5 + j].ToFloat();
        }
    }
    for (size_t r = 0; r < rows; ++r) {
        for (size_t j = 0; j < std::min(size_t{5}, N - unit * 5); ++j) C[r * N + unit * 5 + j] = total[r][j];
    }
#endif
}

#if !defined(MLAS_TARGET_ARM64)
// Runs the platform T8 kernel over row tiles of RowTile rows and tiles of 16 units.
void
T8GemmWithKernel(size_t M, size_t N, size_t K, const float* A, const uint32_t* B, const MLAS_FP16* Scales, float* C, MLAS_THREADPOOL* ThreadPool, MLAS_T8_WEIGHT_LAYOUT Layout, size_t RowTile)
{
    MLAS_T8_GEMM_KERNEL* kernel = GetMlasPlatform().T8GemmKernel;
    const size_t units = N / 5 + (N % 5 != 0);
    const size_t tiles = units / 16 + (units % 16 != 0);
    const size_t row_tiles = M / RowTile + (M % RowTile != 0);
    MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(row_tiles * tiles), [&](ptrdiff_t task) {
        const size_t m = (static_cast<size_t>(task) / tiles) * RowTile;
        const size_t begin = (static_cast<size_t>(task) % tiles) * 16;
        kernel(std::min(RowTile, M - m), N, K, begin, std::min(begin + 16, units), A + m * K, B, Scales, C + m * N, Layout);
    });
}
#endif

}  // namespace

bool MLASCALL
MlasT8Validate(size_t N, size_t K, const uint32_t* Planes)
{
    const size_t units = N / 5 + (N % 5 != 0), sets = K / 32;
    for (size_t q = 0; q < units; ++q) {
        const uint32_t* p = Planes + q * 8 * sets;
        for (size_t s = 0; s < sets; ++s) {
            // 243..255 have the high nibble set and a low nibble >= 3.
            if ((p[7 * sets + s] & p[6 * sets + s] & p[5 * sets + s] & p[4 * sets + s] &
                 (p[3 * sets + s] | p[2 * sets + s] | (p[sets + s] & p[s]))) != 0) return false;
        }
    }
    return true;
}

void MLASCALL
MlasT8Gemm(size_t M, size_t N, size_t K, const float* A, const uint32_t* Planes, const MLAS_FP16* Scales, float* C, float* Workspace, MLAS_THREADPOOL* ThreadPool)
{
    if (M == 0) return;
#if defined(MLAS_TARGET_ARM64)
    PrepareA(M, K, A, Workspace);
    A = Workspace;
#else
    MLAS_UNREFERENCED_PARAMETER(Workspace);
    if (GetMlasPlatform().T8GemmKernel != nullptr) {
        T8GemmWithKernel(M, N, K, A, Planes, Scales, C, ThreadPool, MLAS_T8_WEIGHT_LAYOUT::Planes, 1);
        return;
    }
#endif
    const size_t units = N / 5 + (N % 5 != 0);
    const size_t tiles = units / 16 + (units % 16 != 0);
    MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(M * tiles), [&](ptrdiff_t task) {
        const size_t m = static_cast<size_t>(task) / tiles;
        const size_t begin = (static_cast<size_t>(task) % tiles) * 16;
        for (size_t q = begin; q < std::min(begin + 16, units); ++q) {
            ComputeUnit(N, K, q, A + m * K, Planes, Scales, C + m * N);
        }
    });
}

void MLASCALL
MlasT8Pack(size_t N, size_t K, const uint32_t* Planes, uint32_t* Packed, MLAS_T8_PACKED_LAYOUT Layout)
{
    const size_t groups = K / 128, sets = K / 32;
    const size_t words = Layout == MLAS_T8_PACKED_LAYOUT::Tiled ? 32 : 40;
    for (size_t q = 0; q < N / 5 + (N % 5 != 0); ++q) {
        for (size_t g = 0; g < groups; ++g) {
            const uint32_t* src = Planes + q * 8 * sets + g * 4;
            uint32_t* dst = Packed + (q * groups + g) * words;
            if (Layout == MLAS_T8_PACKED_LAYOUT::Tiled) {
                for (size_t b = 0; b < 8; ++b) std::copy_n(src + b * sets, 4, dst + b * 4);
            } else {
#if defined(MLAS_TARGET_ARM64)
                Bits z[5], p[5];
                Decode(vld1q_u32(src), vld1q_u32(src + sets), vld1q_u32(src + 2 * sets), vld1q_u32(src + 3 * sets), vld1q_u32(src + 4 * sets), vld1q_u32(src + 5 * sets), vld1q_u32(src + 6 * sets), vld1q_u32(src + 7 * sets), z, p);
                for (size_t j = 0; j < 5; ++j) {
                    vst1q_u32(dst + j * 8, vmvnq_u32(z[j]));
                    vst1q_u32(dst + j * 8 + 4, vmvnq_u32(vorrq_u32(z[j], p[j])));
                }
#else
                DecodeGroup(src, sets, dst);
#endif
            }
        }
    }
}

void MLASCALL
MlasT8GemmPacked(size_t M, size_t N, size_t K, const float* A, const uint32_t* Packed, const MLAS_FP16* Scales, float* C, float* Workspace, MLAS_THREADPOOL* ThreadPool, MLAS_T8_PACKED_LAYOUT Layout, size_t RowTile)
{
    if (M == 0) return;
#if defined(MLAS_TARGET_ARM64)
    PrepareA(M, K, A, Workspace);
    A = Workspace;
#else
    MLAS_UNREFERENCED_PARAMETER(Workspace);
    if (GetMlasPlatform().T8GemmKernel != nullptr) {
        const auto layout = Layout == MLAS_T8_PACKED_LAYOUT::Masks ? MLAS_T8_WEIGHT_LAYOUT::Masks : MLAS_T8_WEIGHT_LAYOUT::Tiled;
        T8GemmWithKernel(M, N, K, A, Packed, Scales, C, ThreadPool, layout, RowTile);
        return;
    }
#endif
    const size_t units = N / 5 + (N % 5 != 0);
    const size_t tiles = units / 16 + (units % 16 != 0);
    const size_t row_tiles = M / RowTile + (M % RowTile != 0);
    MlasTrySimpleParallel(ThreadPool, static_cast<ptrdiff_t>(row_tiles * tiles), [&](ptrdiff_t task) {
        const size_t m = (static_cast<size_t>(task) / tiles) * RowTile;
        const size_t begin = (static_cast<size_t>(task) % tiles) * 16;
        for (size_t q = begin; q < std::min(begin + 16, units); ++q) {
            if (RowTile == 2) {
                ComputePackedUnit<2>(M - m, N, K, q, A + m * K, Packed, Scales, C + m * N, Layout);
            } else {
                ComputePackedUnit<1>(M - m, N, K, q, A + m * K, Packed, Scales, C + m * N, Layout);
            }
        }
    });
}
