/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_zvector.cpp

Abstract:

    This module implements QGEMM kernel for S390X.

--*/

#include "mlasi.h"
#include "qgemm.h"
#include <inttypes.h>

struct MLAS_GEMM_QUANT_KERNEL_ZVECTOR
{
    typedef int8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef int8_t OffsetAType;
    typedef uint8_t OffsetBType;
    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{ 16, 256, 384 };
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{ 16, 128, 128 };
};

constexpr size_t MLAS_GEMM_QUANT_KERNEL_ZVECTOR::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_QUANT_KERNEL_ZVECTOR::Strides;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_QUANT_KERNEL_ZVECTOR::PackedStrides;

using vector_int = __attribute__((vector_size(16))) int;

template <typename Vtype>
MLAS_FORCEINLINE
static
vector_int vec_sum4s_impl(Vtype value)
{
    const __vector unsigned char mask_interleave = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

    __vector signed char signed_value = (__vector signed char) vec_perm(value, value, mask_interleave);

    auto tmp1 = vec_unpackh(vec_unpackh(signed_value));
    auto tmp2 = vec_unpackl(vec_unpackh(signed_value));
    auto tmp3 = vec_unpackh(vec_unpackl(signed_value));
    auto tmp4 = vec_unpackl(vec_unpackl(signed_value));

    return (__vector int) (tmp1 + tmp2 + tmp3 + tmp4);
}

#define INC_BUFFER(cnt) \
   ColumnSumBuffer += cnt; \
   if (ZeroPointB != nullptr) { \
       ZeroPointB += cnt; \
   }
template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointA<MLAS_GEMM_QUANT_KERNEL_ZVECTOR>(
    int32_t ZeroPointA,
    bool AIsSigned
    )
{
    if (!AIsSigned) {
        ZeroPointA = MLAS_GEMM_QUANT_KERNEL_ZVECTOR::OffsetAType(ZeroPointA ^ 0x80);
    }
    return ZeroPointA;
}

template<>
MLAS_FORCEINLINE
int32_t
MlasGemmQuantFixupZeroPointB<MLAS_GEMM_QUANT_KERNEL_ZVECTOR>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (BIsSigned) {
        ZeroPointB = MLAS_GEMM_QUANT_KERNEL_ZVECTOR::OffsetBType(ZeroPointB ^ 0x80);
    }
    return ZeroPointB;

}

template<typename Vtype, bool AIsSigned>
void
MlasGemmQuantCopyPackA8x8(
    MLAS_GEMM_QUANT_KERNEL_ZVECTOR::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    )
{
    constexpr uint8_t Flip = (AIsSigned ? 0 : 0x80);
    Vtype vmask = reinterpret_cast<Vtype>(vec_splats(Flip));

    const __vector unsigned char mask0 = { 0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23 };
    const __vector unsigned char mask3 = { 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31 };
    const __vector unsigned char mask_even = { 0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27 };
    const __vector unsigned char mask_odd = { 4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31 };

    // Process eight rows of matrix A in a loop.
    //
    // The buffer is packed as a series of 4x4 byte vectors to help
    // in getting into MMA loop.
    //
    // Unsigned buffers are converted to signed buffers in order to
    // share a common kernel.
    // This pattern is repeated (CountK / 4) times.
    //
    // If CountK is not aligned to a multiple of four, then the vector is padded
    // with zeroes.
    //
    while (CountM >= 8) {
        const uint8_t *a = A;
        __vector int vsum = { 0 };
        __vector int vsum2 = { 0 };
        size_t y = CountK;
        while (y >= 16) {
            Vtype a1 = *reinterpret_cast<const Vtype *>(&a[0]);
            Vtype a2 = *reinterpret_cast<const Vtype *>(&a[lda]);
            Vtype a3 = *reinterpret_cast<const Vtype *>(&a[lda * 2]);
            Vtype a4 = *reinterpret_cast<const Vtype *>(&a[lda * 3]);
            Vtype vx =
               reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a1),
                           reinterpret_cast<__vector int>(a2),
                           mask_even));
            Vtype vx1 =
               reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a3),
                           reinterpret_cast<__vector int>(a4),
                           mask_even));
            Vtype vx2 =
               reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a1),
                           reinterpret_cast<__vector int>(a2),
                           mask_odd));
            Vtype vx3 =
               reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a3),
                           reinterpret_cast<__vector int>(a4),
                           mask_odd));
            Vtype vx4 = vec_perm(vx, vx1, mask0);
            Vtype vx5 = vec_perm(vx2, vx3, mask0);
            Vtype vx6 = vec_perm(vx, vx1, mask3);
            Vtype vx7 = vec_perm(vx2, vx3, mask3);
            a1 = *reinterpret_cast<const Vtype *>(&a[lda*4]);
            a2 = *reinterpret_cast<const Vtype *>(&a[lda*5]);
            a3 = *reinterpret_cast<const Vtype *>(&a[lda*6]);
            a4 = *reinterpret_cast<const Vtype *>(&a[lda*7]);
            vx =
              reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2),
                               mask_even));
            vx1 =
              reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4),
                               mask_even));
            vx2 =
              reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2),
                               mask_odd));
            vx3 =
              reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4),
                               mask_odd));
            Vtype vx8 = vec_perm(vx, vx1, mask0);
            Vtype vx9 = vec_perm(vx2, vx3, mask0);
            Vtype vx10 = vec_perm(vx, vx1, mask3);
            Vtype vx11 = vec_perm(vx2, vx3, mask3);
            Vtype vxx = AIsSigned ? vx4 : vx4 - vmask;
            vsum += vec_sum4s_impl(vxx);
            *reinterpret_cast<Vtype *>(&D[0]) = vxx;
            vxx = AIsSigned ? vx5 : vx5 - vmask;
            vsum += vec_sum4s_impl(vxx);
            *reinterpret_cast<Vtype *>(&D[16]) = vxx;
            vxx = AIsSigned ? vx6 : vx6 - vmask;
            vsum += vec_sum4s_impl(vxx);
            *reinterpret_cast<Vtype *>(&D[32]) = vxx;
            vxx = AIsSigned ? vx7 : vx7 - vmask;
            vsum += vec_sum4s_impl(vxx);
            *reinterpret_cast<Vtype *>(&D[48]) = vxx;
            vxx = AIsSigned ? vx8 : vx8 - vmask;
            *reinterpret_cast<Vtype *>(&D[64]) = vxx;
            vsum2 += vec_sum4s_impl(vxx);
            vxx = AIsSigned ? vx9 : vx9 - vmask;
            *reinterpret_cast<Vtype *>(&D[80]) = vxx;
            vsum2 += vec_sum4s_impl(vxx);
            vxx = AIsSigned ? vx10 : vx10 - vmask;
            *reinterpret_cast<Vtype *>(&D[96]) = vxx;
            vsum2 += vec_sum4s_impl(vxx);
            vxx = AIsSigned ? vx11 : vx11 - vmask;
            *reinterpret_cast<Vtype *>(&D[112]) = vxx;
            vsum2 += vec_sum4s_impl(vxx);
            D += 16 * 8;
            a += 16;
            y -= 16;
        }
        size_t yval = y;
        while (y >= 4)
        {
            int a1 = *reinterpret_cast<const int *>(&a[0]);
            int a2 = *reinterpret_cast<const int *>(&a[lda]);
            int a3 = *reinterpret_cast<const int *>(&a[lda*2]);
            int a4 = *reinterpret_cast<const int *>(&a[lda*3]);
            __vector int vx1 = { a1, a2, a3, a4};
            Vtype vx = AIsSigned ? reinterpret_cast<Vtype>(vx1) : reinterpret_cast<Vtype>(vx1) - vmask;
            vsum += vec_sum4s_impl(vx);
            *reinterpret_cast<Vtype *>(&D[0]) = vx;
            a1 = *reinterpret_cast<const int *>(&a[lda*4]);
            a2 = *reinterpret_cast<const int *>(&a[lda*5]);
            a3 = *reinterpret_cast<const int *>(&a[lda*6]);
            a4 = *reinterpret_cast<const int *>(&a[lda*7]);
            __vector int vx2 = { a1, a2, a3, a4};
            vx = AIsSigned ? reinterpret_cast<Vtype>(vx2) : reinterpret_cast<Vtype>(vx2) - vmask;
            vsum2 += vec_sum4s_impl(vx);
            if (CountK & 3) {
                if (yval >= 12) {
                     *reinterpret_cast<Vtype *>(&D[64]) = vx;
                } else if (yval >= 8) {
                     *reinterpret_cast<Vtype *>(&D[48]) = vx;
                } else {
                     *reinterpret_cast<Vtype *>(&D[32]) = vx;
                }
            } else {
                if (yval >= 12) {
                     *reinterpret_cast<Vtype *>(&D[48]) = vx;
                } else if (yval >= 8) {
                     *reinterpret_cast<Vtype *>(&D[32]) = vx;
                } else {
                     *reinterpret_cast<Vtype *>(&D[16]) = vx;
                }
            }
            D += 16;
            a += 4;
            y -= 4;
        }
        if (yval >= 12) {
           if (!(CountK & 3)) {
               D += 48;
           }
        } else if (yval >= 8) {
           if (!(CountK & 3)) {
               D += 32;
            }
        } else if (yval >= 4) {
            if (!(CountK & 3)) {
               D += 16;
            }
        }
        if (y >= 1)
        {
            Vtype a1 = vmask;
            Vtype a2 = vmask;
            Vtype a3 = vmask;
            Vtype a4 = vmask;
            a1[0] = a[0];
            a2[0] = a[lda];
            a3[0] = a[lda * 2];
            a4[0] = a[lda * 3];
            if (y >= 2) {
                a1[1] = a[1];
                a2[1] = a[lda + 1];
                a3[1] = a[lda * 2 + 1];
                a4[1] = a[lda * 3 + 1];
            }
            if (y >= 3) {
                a1[2] = a[2];
                a2[2] = a[lda + 2];
                a3[2] = a[lda * 2 + 2];
                a4[2] = a[lda * 3 + 2];
            }
            Vtype vx = reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2),
                               mask_even));
            Vtype vx1 =
              reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4),
                               mask_even));
            Vtype vx2 = vec_perm(vx, vx1, mask0);
            Vtype vx3 = AIsSigned ? vx2 : vx2 - vmask;
            vsum += vec_sum4s_impl(vx3);

            *reinterpret_cast<Vtype *>(&D[0]) = vx3;
            a1 = vmask;
            a2 = vmask;
            a3 = vmask;
            a4 = vmask;
            a1[0] = a[lda * 4];
            a2[0] = a[lda * 5];
            a3[0] = a[lda * 6];
            a4[0] = a[lda * 7];
            if (y >= 2) {
                a1[1] = a[lda * 4 + 1];
                a2[1] = a[lda * 5 + 1];
                a3[1] = a[lda * 6 + 1];
                a4[1] = a[lda * 7 + 1];
            }
            if (y >= 3) {
                a1[2] = a[lda * 4 + 2];
                a2[2] = a[lda * 5 + 2];
                a3[2] = a[lda * 6 + 2];
                a4[2] = a[lda * 7 + 2];
            }
            vx =
              reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2),
                               mask_even));
            vx1 =
              reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4),
                               mask_even));
            vx2 = vec_perm(vx, vx1, mask0);
            vx3 = AIsSigned ? vx2 : vx2 - vmask;
            vsum2 += vec_sum4s_impl(vx3);
            if (CountK % 16 >= 12) {
                *reinterpret_cast<Vtype *>(&D[64]) = vx3;
                D += 80;
            } else if (CountK % 16 >= 8) {
                *reinterpret_cast<Vtype *>(&D[48]) = vx3;
                D += 64;
            } else if (CountK % 16 >= 4) {
                *reinterpret_cast<Vtype *>(&D[32]) = vx3;
                D += 48;
            } else {
                *reinterpret_cast<Vtype *>(&D[16]) = vx3;
                D += 16 * 2;
            }
            a += 16;
        }
        A += lda * 8;

        vec_xst(vsum, 0, &(RowSumBuffer[0]));
        vec_xst(vsum2, 16, &(RowSumBuffer[0]));

        RowSumBuffer += 8;
        CountM -= 8;
    }

    // Process four rows of matrix A in a loop.
    //
    if (CountM >= 4)
    {
        const uint8_t *a = A;
        __vector int vsum = { 0 };
        size_t y = CountK;

        while (y >= 16)
        {
            Vtype a1 = *reinterpret_cast<const Vtype *>(&a[0]);
            Vtype a2 = *reinterpret_cast<const Vtype *>(&a[lda]);
            Vtype a3 = *reinterpret_cast<const Vtype *>(&a[lda * 2]);
            Vtype a4 = *reinterpret_cast<const Vtype *>(&a[lda * 3]);
            Vtype vx =
              reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2),
                               mask_even));
            Vtype vx1 =
              reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4),
                               mask_even));
            Vtype vx2 =
              reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2),
                               mask_odd));
            Vtype vx3 =
              reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4),
                               mask_odd));
            Vtype vx4 = vec_perm(vx, vx1, mask0);
            Vtype vx5 = vec_perm(vx2, vx3, mask0);
            Vtype vx6 = vec_perm(vx, vx1, mask3);
            Vtype vx7 = vec_perm(vx2, vx3, mask3);
            Vtype vx0 = AIsSigned ? vx4 : vx4 - vmask;
            *reinterpret_cast<Vtype *>(&D[0]) = vx0;
            vsum += vec_sum4s_impl(vx0);
            vx0 = AIsSigned ? vx5 : vx5 - vmask;
            *reinterpret_cast<Vtype *>(&D[16]) = vx0;
            vsum += vec_sum4s_impl(vx0);
            vx0 = AIsSigned ? vx6 : vx6 - vmask;
            *reinterpret_cast<Vtype *>(&D[32]) = vx0;
            vsum += vec_sum4s_impl(vx0);
            vx0 = AIsSigned ? vx7 : vx7 - vmask;
            *reinterpret_cast<Vtype *>(&D[48]) = vx0;
            vsum += vec_sum4s_impl(vx0);
            D += 16 * 4;
            a += 16;
            y -= 16;
        }
        while (y >= 4)
        {
            int a1 = *reinterpret_cast<const int *>(&a[0]);
            int a2 = *reinterpret_cast<const int *>(&a[lda]);
            int a3 = *reinterpret_cast<const int *>(&a[lda*2]);
            int a4 = *reinterpret_cast<const int *>(&a[lda*3]);
            __vector int vx1 = { a1, a2, a3, a4};
            Vtype vx = AIsSigned ? reinterpret_cast<Vtype>(vx1) : reinterpret_cast<Vtype>(vx1) - vmask;
            *reinterpret_cast<Vtype *>(&D[0]) = vx;
            vsum += vec_sum4s_impl(vx);
            D += 16;
            a += 4;
            y -= 4;
        }
        if (y >= 1)
        {
            Vtype vx = vmask;
            vx[0] = a[0];
            vx[4] = a[lda];
            vx[8] = a[lda * 2];
            vx[12] = a[lda * 3];
            if (y >= 2) {
                vx[1] = a[1];
                vx[5] = a[lda + 1];
                vx[9] = a[lda * 2 + 1];
                vx[13] = a[lda * 3 + 1];
            }
            if (y >= 3) {
                vx[2] = a[2];
                vx[6] = a[lda + 2];
                vx[10] = a[lda * 2 + 2];
                vx[14] = a[lda * 3 + 2];
            }
            Vtype vx1 = AIsSigned ? vx : vx - vmask;
            *reinterpret_cast<Vtype *>(&D[0]) = vx1;
            vsum += vec_sum4s_impl(vx1);
            D += 16;
            a += 16;
        }
        A += lda * 4;

        vec_xst(vsum, 0, &(RowSumBuffer[0]));

        RowSumBuffer += 4;
        CountM -= 4;
    }

    // Process remaining rows of matrix A in a loop.
    //
    if (CountM <= 3 && CountM > 0) {
        const uint8_t *a = A;
        size_t y = CountK;
        __vector int vsum = { 0 };

        while (y >= 16) {
            Vtype a4 = vmask;
            Vtype a2 = vmask;
            Vtype a3 = vmask;
            Vtype a1 = *reinterpret_cast<const Vtype *>(&a[0]);
            if (CountM == 3) {
                a3 = *reinterpret_cast<const Vtype *>(&a[lda * 2]);
            }
            if (CountM >= 2) {
                a2 = *reinterpret_cast<const Vtype *>(&a[lda]);
            }
            Vtype vx =
              reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2),
                               mask_even));
            Vtype vx1 =
              reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4),
                               mask_even));
            Vtype vx2 =
              reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2),
                               mask_odd));
            Vtype vx3 =
              reinterpret_cast<Vtype>(vec_perm(reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4),
                               mask_odd));
            Vtype vx4 = vec_perm(vx, vx1, mask0);
            Vtype vx5 = vec_perm(vx2, vx3, mask0);
            Vtype vx6 = vec_perm(vx, vx1, mask3);
            Vtype vx7 = vec_perm(vx2, vx3, mask3);
            Vtype vx0 = AIsSigned ? vx4 : vx4 - vmask;
            *reinterpret_cast<Vtype *>(&D[0]) = vx0;
            vsum += vec_sum4s_impl(vx0);
            vx0 = AIsSigned ? vx5 : vx5 - vmask;
            *reinterpret_cast<Vtype *>(&D[16]) = vx0;
            vsum += vec_sum4s_impl(vx0);
            vx0 = AIsSigned ? vx6 : vx6 - vmask;
            *reinterpret_cast<Vtype *>(&D[32]) = vx0;
            vsum += vec_sum4s_impl(vx0);
            vx0 = AIsSigned ? vx7 : vx7 - vmask;
            *reinterpret_cast<Vtype *>(&D[48]) = vx0;
            vsum += vec_sum4s_impl(vx0);
            D += 16 * 4;
            a += 16;
            y -= 16;
        }
        while (y >= 4)
        {
            Vtype vb = vmask;
            __vector int vx1 = reinterpret_cast<__vector int>(vb);
            vx1[0] = *reinterpret_cast<const int *>(&a[0]);
            if (CountM >= 2) {
                vx1[1] = *reinterpret_cast<const int *>(&a[lda]);
            }
            if (CountM >= 3) {
                vx1[2] = *reinterpret_cast<const int *>(&a[lda*2]);
            }
            Vtype vx = AIsSigned ? reinterpret_cast<Vtype>(vx1) : reinterpret_cast<Vtype>(vx1) - vmask;
            *reinterpret_cast<Vtype *>(&D[0]) = vx;
            vsum += vec_sum4s_impl(vx);
            D += 16;
            a += 4;
            y -= 4;
        }
        if (y >= 1)
        {
            Vtype vx = (Vtype) vec_splats(0);
            vx[0] = a[0] ^ Flip;
            if (y >= 2) {
                vx[1] = a[1] ^ Flip;
            }
            if (y >= 3) {
                vx[2] = a[2] ^ Flip;
            }
            if (CountM >= 2) {
                vx[4] = a[lda] ^ Flip;
                if (y >= 2) {
                   vx[5] = a[lda + 1] ^ Flip;
                }
                if (y >= 3) {
                   vx[6] = a[lda + 2] ^ Flip;
                }
            }
            if (CountM == 3) {
                vx[8] = a[lda * 2] ^ Flip;
                if (y >= 2) {
                    vx[9] = a[lda * 2 + 1] ^ Flip;
                }
                if (y >= 3) {
                    vx[10] = a[lda * 2 + 2] ^ Flip;
                }
            }
            *reinterpret_cast<Vtype *>(&D[0]) = vx;
            vsum += vec_sum4s_impl(vx);
            D += 16;
        }
        *RowSumBuffer++ = vsum[0];
        if (CountM >= 2) {
            *RowSumBuffer++ = vsum[1];
        }
        if (CountM >= 3) {
            *RowSumBuffer++ = vsum[2];
        }
    }
}

template<typename Vtype, bool BIsSigned>
void
MlasGemmQuantCopyPackB8x8(
    MLAS_GEMM_QUANT_KERNEL_ZVECTOR::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer
    )
{
    [[maybe_unused]] constexpr uint8_t BitFlipValue = (BIsSigned ? 0x80 : 0);
    typedef __vector unsigned char vec_t;
    Vtype vmask = reinterpret_cast<Vtype>(vec_splats(BitFlipValue));
    vec_t mask = {0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15};

    const __vector unsigned char vec_zero = { 0 };

    // Copy columns from matrix B to the packed buffer. Signed buffers are
    // converted to unsigned buffers in order to share a common kernel.
    //
    // If CountK is not aligned to a multiple of four, then the packed buffer
    // is padded with zero vectors.

    // Process 16 columns of matrix B in a loop.
    //
    size_t PackedK = ((CountK + 4 - 1) / 4) * 16;
    size_t k2 = PackedK;
    size_t k3 = PackedK*2;
    size_t k4 = PackedK*3;

    while (CountN >= 16) {
        const uint8_t* b = B;
        __vector unsigned int vsum = {0};
        __vector unsigned int vsum2 = {0};
        __vector unsigned int vsum3 = {0};
        __vector unsigned int vsum4 = {0};
        size_t y = CountK;
        if (y >= 4) {
            do {
                Vtype b1 = *reinterpret_cast<const Vtype *>(&b[0]);
                Vtype b2 = *reinterpret_cast<const Vtype *>(&b[ldb]);
                Vtype b3 = *reinterpret_cast<const Vtype *>(&b[ldb*2]);
                Vtype b4 = *reinterpret_cast<const Vtype *>(&b[ldb*3]);
                Vtype t1 = vec_mergeh(b1, b3);
                Vtype t2 = vec_mergel(b1, b3);
                Vtype t3 = vec_mergeh(b2, b4);
                Vtype t4 = vec_mergel(b2, b4);
                b1 = vec_mergeh(t1, t3);
                b2 = vec_mergel(t1, t3);
                b3 = vec_mergeh(t2, t4);
                b4 = vec_mergel(t2, t4);
                vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(b1 + vmask) :
                                        reinterpret_cast<vec_t>(b1);
                vec_t vx2 = BIsSigned ? reinterpret_cast<vec_t>(b2 + vmask) :
                                        reinterpret_cast<vec_t>(b2);
                vec_t vx3 = BIsSigned ? reinterpret_cast<vec_t>(b3 + vmask) :
                                        reinterpret_cast<vec_t>(b3);
                vec_t vx4 = BIsSigned ? reinterpret_cast<vec_t>(b4 + vmask) :
                                        reinterpret_cast<vec_t>(b4);
                *reinterpret_cast<vec_t *>(&D[0]) = vx1;
                *reinterpret_cast<vec_t *>(&D[k2]) = vx2;
                *reinterpret_cast<vec_t *>(&D[k3]) = vx3;
                *reinterpret_cast<vec_t *>(&D[k4]) = vx4;
                vsum  += vec_sum4(vx1, vec_zero);
                vsum2 += vec_sum4(vx2, vec_zero);
                vsum3 += vec_sum4(vx3, vec_zero);
                vsum4 += vec_sum4(vx4, vec_zero);
                D += 16;
                b += ldb*4;
                y -= 4;
            } while (y >= 4);
        }
        if (y >= 1) {
            Vtype b1 = *reinterpret_cast<const Vtype *>(&b[0]);
            Vtype b2 = (y >= 2) ? *reinterpret_cast<const Vtype *>(&b[ldb]) : vmask;
            Vtype b3 = (y >= 3) ? *reinterpret_cast<const Vtype *>(&b[ldb*2]) : vmask;
            Vtype b4 = vmask;
            Vtype t1 = vec_mergeh(b1, b3);
            Vtype t2 = vec_mergel(b1, b3);
            Vtype t3 = vec_mergeh(b2, b4);
            Vtype t4 = vec_mergel(b2, b4);
            b1 = vec_mergeh(t1, t3);
            b2 = vec_mergel(t1, t3);
            b3 = vec_mergeh(t2, t4);
            b4 = vec_mergel(t2, t4);
            vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(b1 + vmask) :
                                    reinterpret_cast<vec_t>(b1);
            vec_t vx2 = BIsSigned ? reinterpret_cast<vec_t>(b2 + vmask) :
                                    reinterpret_cast<vec_t>(b2);
            vec_t vx3 = BIsSigned ? reinterpret_cast<vec_t>(b3 + vmask) :
                                    reinterpret_cast<vec_t>(b3);
            vec_t vx4 = BIsSigned ? reinterpret_cast<vec_t>(b4 + vmask) :
                                    reinterpret_cast<vec_t>(b4);
            *reinterpret_cast<vec_t *>(&D[0]) = vx1;
            *reinterpret_cast<vec_t *>(&D[k2]) = vx2;
            *reinterpret_cast<vec_t *>(&D[k3]) = vx3;
            *reinterpret_cast<vec_t *>(&D[k4]) = vx4;
            vsum  += vec_sum4(vx1, vec_zero);
            vsum2 += vec_sum4(vx2, vec_zero);
            vsum3 += vec_sum4(vx3, vec_zero);
            vsum4 += vec_sum4(vx4, vec_zero);
            D += 16;
        }

        vec_xst(vsum,   0, (unsigned int*) ColumnSumBuffer);
        vec_xst(vsum2, 16, (unsigned int*) ColumnSumBuffer);
        vec_xst(vsum3, 32, (unsigned int*) ColumnSumBuffer);
        vec_xst(vsum4, 48, (unsigned int*) ColumnSumBuffer);

        ColumnSumBuffer += 16;
        B += 16;
        CountN -= 16;
        D += k4;
    }

    // Process four columns of matrix B in a loop.
    //
    while (CountN >= 4) {
        const uint8_t* b = B;
        __vector unsigned int vsum = {0};
        size_t y = CountK;
        if (y >= 4) {
            do {
                int b1 = *reinterpret_cast<const int *>(&b[0]);
                int b2 = *reinterpret_cast<const int *>(&b[ldb]);
                int b3 = *reinterpret_cast<const int *>(&b[ldb*2]);
                int b4 = *reinterpret_cast<const int *>(&b[ldb*3]);
                __vector int vb = {b1, b2, b3, b4};
                Vtype vx = vec_perm(reinterpret_cast<Vtype>(vb), reinterpret_cast<Vtype>(vb), mask);
                vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vx + vmask) :
                                        reinterpret_cast<vec_t>(vx);
                *reinterpret_cast<vec_t *>(&D[0]) = vx1;
                vsum += vec_sum4(vx1, vec_zero);
                D += 16;
                b += ldb*4;
                y -= 4;
            } while (y >= 4);
        }
        if (y >= 1) {
            Vtype vb = vmask;
            __vector int vb1 = reinterpret_cast<__vector int>(vb);
            vb1[0] = *reinterpret_cast<const int *>(&b[0]);
            if (y >= 2) {
                vb1[1] = *reinterpret_cast<const int *>(&b[ldb]);
            }
            if (y >= 3) {
                vb1[2] = *reinterpret_cast<const int *>(&b[ldb*2]);
            }
            Vtype vx = vec_perm(reinterpret_cast<Vtype>(vb1), reinterpret_cast<Vtype>(vb1), mask);
            vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vx + vmask) :
                                    reinterpret_cast<vec_t>(vx);
            *reinterpret_cast<vec_t *>(&D[0]) = vx1;
            vsum += vec_sum4(vx1, vec_zero);
            D += 16;
        }

        vec_xst(vsum,   0, (unsigned int*) ColumnSumBuffer);

        ColumnSumBuffer += 4;
        B += 4;
        CountN -= 4;
    }

    //
    // Process the remaining columns of matrix B.
    //
    if (CountN > 0) {
        __vector unsigned int vsum = {0};
        const uint8_t* b = B;
        size_t y = CountK;
        if (y >= 4) {
            do {
                Vtype vb = vmask;
                if (CountN == 1) {
                    vb[0] = b[0];
                    vb[4] = b[ldb];
                    vb[8] = b[ldb*2];
                    vb[12] = b[ldb*3];
                }
                if (CountN == 2) {
                    vb[0] = b[0];
                    vb[1] = b[1];
                    vb[4] = b[ldb];
                    vb[5] = b[ldb+1];
                    vb[8] = b[ldb*2];
                    vb[9] = b[ldb*2+1];
                    vb[12] = b[ldb*3];
                    vb[13] = b[ldb*3+1];
                }
                if (CountN == 3) {
                    vb[0] = b[0];
                    vb[1] = b[1];
                    vb[2] = b[2];
                    vb[4] = b[ldb];
                    vb[5] = b[ldb+1];
                    vb[6] = b[ldb+2];
                    vb[8] = b[ldb*2];
                    vb[9] = b[ldb*2+1];
                    vb[10] = b[ldb*2+2];
                    vb[12] = b[ldb*3];
                    vb[13] = b[ldb*3+1];
                    vb[14] = b[ldb*3+2];
                }
                Vtype vx = vec_perm(reinterpret_cast<Vtype>(vb), reinterpret_cast<Vtype>(vb), mask);
                vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vx + vmask) :
                                        reinterpret_cast<vec_t>(vx);
                *reinterpret_cast<vec_t *>(&D[0]) = vx1;
                vsum += vec_sum4(vx1, vec_zero);
                D += 16;
                b += ldb*4;
                y -= 4;
            } while (y >= 4);
        }
        if (y >= 1) {
            Vtype vb = vmask;
            if (CountN == 1) {
                vb[0]= b[0];
                if (y >= 2) {
                    vb[4] = b[ldb];
                }
                if (y >= 3) {
                    vb[8] = b[ldb*2];
                }
            }
            if (CountN == 2) {
                vb[0] = b[0];
                vb[1] = b[1];
                if (y >= 2) {
                    vb[4] = b[ldb];
                    vb[5] = b[ldb+1];
                }
                if (y >= 3) {
                    vb[8] = b[ldb*2];
                    vb[9] = b[ldb*2+1];
                }
            }
            if (CountN == 3) {
                vb[0] = b[0];
                vb[1] = b[1];
                vb[2] = b[2];
                if (y >= 2) {
                    vb[4] = b[ldb];
                    vb[5] = b[ldb+1];
                    vb[6] = b[ldb+2];
                }
                if (y >= 3) {
                    vb[8] = b[ldb*2];
                    vb[9] = b[ldb*2+1];
                    vb[10] = b[ldb*2+2];
                }
            }
            Vtype vx = vec_perm(reinterpret_cast<Vtype>(vb), reinterpret_cast<Vtype>(vb), mask);
            vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vx + vmask) :
                                    reinterpret_cast<vec_t>(vx);
            *reinterpret_cast<vec_t *>(&D[0]) = vx1;
            vsum += vec_sum4(vx1, vec_zero);
            D += 16;
        }
        *ColumnSumBuffer++ = vsum[0];
        if (CountN >= 2) {
            *ColumnSumBuffer++ = vsum[1];
        }
        if (CountN >= 3) {
            *ColumnSumBuffer++ = vsum[2];
        }
    }
}

template<>
void
MlasGemmQuantCopyPackA<MLAS_GEMM_QUANT_KERNEL_ZVECTOR>(
    MLAS_GEMM_QUANT_KERNEL_ZVECTOR::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned
    )
{
    if (AIsSigned) {
        MlasGemmQuantCopyPackA8x8<__vector   signed char, true> (D, A, lda, CountM, CountK, RowSumBuffer);
    } else {
        MlasGemmQuantCopyPackA8x8<__vector unsigned char, false>(D, A, lda, CountM, CountK, RowSumBuffer);
    }
}
template<>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_QUANT_KERNEL_ZVECTOR>(
    MLAS_GEMM_QUANT_KERNEL_ZVECTOR::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    if (BIsSigned) {
        MlasGemmQuantCopyPackB8x8<__vector signed char, true>(D, B, ldb, CountN, CountK, ColumnSumBuffer);
    } else {
        MlasGemmQuantCopyPackB8x8< __vector unsigned char, false>(D, B, ldb, CountN, CountK, ColumnSumBuffer);
    }
}

template<size_t VectorCount>
MLAS_FORCEINLINE
void
MlasQgemmStoreVectorZVECTOR
    (
    MLAS_INT32X4 result[4],
    int32_t* C,
    size_t ldc,
    size_t row,
    bool ZeroMode,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    int pos
    )
{
    size_t RowCount;
    __vector signed int vsum0, vsum1, vsum2, vsum3;
    __vector signed int columnsum = *reinterpret_cast<const __vector int32_t *>(&ColumnSumBuffer[pos]);
    C += VectorCount;
    if (ZeroPointB != nullptr) {
        __vector signed int zeropoint = *reinterpret_cast<const __vector int32_t *>(&ZeroPointB[pos]);
        if (ZeroMode) {
            for (RowCount = 0; RowCount + 4 <= row; RowCount += 4, C += ldc*4) {
                vsum0 = vec_splats(RowSumBuffer[RowCount + 0]) * zeropoint + columnsum;
                vsum1 = vec_splats(RowSumBuffer[RowCount + 1]) * zeropoint + columnsum;
                vsum2 = vec_splats(RowSumBuffer[RowCount + 2]) * zeropoint + columnsum;
                vsum3 = vec_splats(RowSumBuffer[RowCount + 3]) * zeropoint + columnsum;
                *reinterpret_cast<__vector int *>(&C[0]) =
                    *reinterpret_cast<__vector int *>(&result[RowCount + 0]) + vsum0;
                *reinterpret_cast<__vector int *>(&C[ldc]) =
                    *reinterpret_cast<__vector int *>(&result[RowCount + 1]) + vsum1;
                *reinterpret_cast<__vector int *>(&C[ldc*2]) =
                    *reinterpret_cast<__vector int *>(&result[RowCount + 2]) + vsum2;
                *reinterpret_cast<__vector int *>(&C[ldc*3]) =
                    *reinterpret_cast<__vector int *>(&result[RowCount + 3]) + vsum3;
            }
            for (; RowCount < row; RowCount++, C += ldc) {
                vsum0 = vec_splats(RowSumBuffer[RowCount]) * zeropoint + columnsum;
                *reinterpret_cast<__vector int *>(&C[0]) =
                    *reinterpret_cast<__vector int *>(&result[RowCount + 0]) + vsum0;
            }
        } else {
            for (RowCount = 0; RowCount + 4 <= row; RowCount += 4, C += ldc*4) {
                vsum0 = vec_splats(RowSumBuffer[RowCount + 0]) * zeropoint + columnsum;
                vsum1 = vec_splats(RowSumBuffer[RowCount + 1]) * zeropoint + columnsum;
                vsum2 = vec_splats(RowSumBuffer[RowCount + 2]) * zeropoint + columnsum;
                vsum3 = vec_splats(RowSumBuffer[RowCount + 3]) * zeropoint + columnsum;
                *reinterpret_cast<__vector int *>(&C[0]) +=
                    *reinterpret_cast<__vector int *>(&result[RowCount + 0]) + vsum0;
                *reinterpret_cast<__vector int *>(&C[ldc]) +=
                    *reinterpret_cast<__vector int *>(&result[RowCount + 1]) + vsum1;
                *reinterpret_cast<__vector int *>(&C[ldc*2]) +=
                    *reinterpret_cast<__vector int *>(&result[RowCount + 2]) + vsum2;
                *reinterpret_cast<__vector int *>(&C[ldc*3]) +=
                    *reinterpret_cast<__vector int *>(&result[RowCount + 3]) + vsum3;
            }
            for (; RowCount < row; RowCount++, C += ldc) {
                vsum0 = vec_splats(RowSumBuffer[RowCount]) * zeropoint + columnsum;
                *reinterpret_cast<__vector int *>(&C[0]) +=
                    *reinterpret_cast<__vector int *>(&result[RowCount + 0]) + vsum0;
            }
        }
    } else {
        if (ZeroMode) {
            for (RowCount = 0; RowCount + 4 <= row; RowCount += 4, C += ldc*4) {
                vsum0 = vec_splats(RowSumBuffer[RowCount + 0]) + columnsum;
                vsum1 = vec_splats(RowSumBuffer[RowCount + 1]) + columnsum;
                vsum2 = vec_splats(RowSumBuffer[RowCount + 2]) + columnsum;
                vsum3 = vec_splats(RowSumBuffer[RowCount + 3]) + columnsum;
                *reinterpret_cast<__vector int *>(&C[0]) =
                    *reinterpret_cast<__vector int *>(&result[RowCount + 0]) + vsum0;
                *reinterpret_cast<__vector int *>(&C[ldc]) =
                    *reinterpret_cast<__vector int *>(&result[RowCount + 1]) + vsum1;
                *reinterpret_cast<__vector int *>(&C[ldc*2]) =
                    *reinterpret_cast<__vector int *>(&result[RowCount + 2]) + vsum2;
                *reinterpret_cast<__vector int *>(&C[ldc*3]) =
                    *reinterpret_cast<__vector int *>(&result[RowCount + 3]) + vsum3;
            }
            for (; RowCount < row; RowCount++, C += ldc) {
                vsum0 = vec_splats(RowSumBuffer[RowCount]) + columnsum;
                *reinterpret_cast<__vector int *>(&C[0]) =
                    *reinterpret_cast<__vector int *>(&result[RowCount + 0]) + vsum0;
            }
        } else {
            for (RowCount = 0; RowCount + 4 <= row; RowCount += 4, C += ldc*4) {
                vsum0 = vec_splats(RowSumBuffer[RowCount + 0]) + columnsum;
                vsum1 = vec_splats(RowSumBuffer[RowCount + 1]) + columnsum;
                vsum2 = vec_splats(RowSumBuffer[RowCount + 2]) + columnsum;
                vsum3 = vec_splats(RowSumBuffer[RowCount + 3]) + columnsum;
                *reinterpret_cast<__vector int *>(&C[0]) +=
                    *reinterpret_cast<__vector int *>(&result[RowCount + 0]) + vsum0;
                *reinterpret_cast<__vector int *>(&C[ldc]) +=
                    *reinterpret_cast<__vector int *>(&result[RowCount + 1]) + vsum1;
                *reinterpret_cast<__vector int *>(&C[ldc*2]) +=
                    *reinterpret_cast<__vector int *>(&result[RowCount + 2]) + vsum2;
                *reinterpret_cast<__vector int *>(&C[ldc*3]) +=
                    *reinterpret_cast<__vector int *>(&result[RowCount + 3]) + vsum3;
            }
            for (; RowCount < row; RowCount++, C += ldc) {
                vsum0 = vec_splats(RowSumBuffer[RowCount]) + columnsum;
                *reinterpret_cast<__vector int *>(&C[0]) +=
                    *reinterpret_cast<__vector int *>(&result[RowCount + 0]) + vsum0;
            }
        }
    }
};
template<size_t Lane>
MLAS_FORCEINLINE
void
MlasQgemmStoreScalarZVECTOR(
    MLAS_INT32X4 result[4],
    int32_t* C,
    size_t ldc,
    size_t row,
    bool ZeroMode,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB
    )
{
    if (ZeroPointB != nullptr) {
        if (ZeroMode) {
            for (size_t RowCount = 0;RowCount < row; RowCount++){
                int sum = RowSumBuffer[RowCount];
                sum *= ZeroPointB[0];
                sum += ColumnSumBuffer[0];
                C[RowCount*ldc+Lane] = result[RowCount][Lane] + sum;
            }
        } else {
            for (size_t RowCount = 0;RowCount < row; RowCount++){
                int sum = RowSumBuffer[RowCount];
                sum *= ZeroPointB[0];
                sum += ColumnSumBuffer[0];
                C[RowCount*ldc+Lane] += result[RowCount][Lane] + sum;
            }
        }
    } else {
        if (ZeroMode) {
            for (size_t RowCount = 0;RowCount < row; RowCount++){
                int sum = RowSumBuffer[RowCount] + ColumnSumBuffer[0];
                C[RowCount*ldc+Lane] = result[RowCount][Lane] + sum;
            }
        } else {
            for (size_t RowCount = 0;RowCount < row; RowCount++){
                int sum = RowSumBuffer[RowCount] + ColumnSumBuffer[0];
                C[RowCount*ldc+Lane] += result[RowCount][Lane] + sum;
            }
        }
    }
};

MLAS_FORCEINLINE
void
xvi8ger4pp_impl(
    MLAS_INT32X4 acc[4],
    __vector unsigned char va,
    __vector unsigned char vb
    )
{
    const __vector unsigned char maska[4] = {
        {  0,  1,  2,  3,  0,  1,  2,  3,  0,  1,  2  ,3,  0,  1,  2,  3 },
        {  4,  5,  6,  7,  4,  5,  6,  7,  4,  5,  6,  7,  4,  5,  6,  7 },
        {  8,  9, 10, 11,  8,  9, 10, 11,  8,  9, 10, 11,  8,  9, 10, 11 },
        { 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15, 12, 13, 14, 15 }
    };

    const __vector unsigned char maskb = { 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15 };

    __vector int va_interim[4];
    __vector unsigned char vb_interm = vec_perm(vb, vb, maskb);

    __vector int va_prep;
    __vector int vb_prep[4];

    va_interim[0] = (__vector int) vec_unpackh(vec_unpackh((__vector signed char) va));
    va_interim[1] = (__vector int) vec_unpackl(vec_unpackh((__vector signed char) va));
    va_interim[2] = (__vector int) vec_unpackh(vec_unpackl((__vector signed char) va));
    va_interim[3] = (__vector int) vec_unpackl(vec_unpackl((__vector signed char) va));

    vb_prep[0] = (__vector int) vec_unpackh(vec_unpackh(vb_interm));
    vb_prep[1] = (__vector int) vec_unpackl(vec_unpackh(vb_interm));
    vb_prep[2] = (__vector int) vec_unpackh(vec_unpackl(vb_interm));
    vb_prep[3] = (__vector int) vec_unpackl(vec_unpackl(vb_interm));

    for (size_t i = 0; i < 4; ++i)
    {
        for (size_t k = 0; k < 4; ++k)
        {
            va_prep = vec_perm(va_interim[i], va_interim[i], maska[k]);

            acc[i] += va_prep * vb_prep[k];
        }
    }
}

template<bool CountM, size_t CountK>
MLAS_FORCEINLINE
void
MlasQgemmComputeZVECTOR(
    MLAS_INT32X4 acc0[4],
    MLAS_INT32X4 acc1[4],
    __vector unsigned char *va,
    __vector unsigned char *vb
    )
{
    if (CountK == 16) {
        xvi8ger4pp_impl(acc0, va[0], vb[0]);
        xvi8ger4pp_impl(acc0, va[1], vb[1]);
        xvi8ger4pp_impl(acc0, va[2], vb[2]);
        xvi8ger4pp_impl(acc0, va[3], vb[3]);
        if (CountM) {
            xvi8ger4pp_impl(acc1, va[4], vb[0]);
            xvi8ger4pp_impl(acc1, va[5], vb[1]);
            xvi8ger4pp_impl(acc1, va[6], vb[2]);
            xvi8ger4pp_impl(acc1, va[7], vb[3]);
        }
    } else if (CountK == 12) {
        xvi8ger4pp_impl(acc0, va[0], vb[0]);
        xvi8ger4pp_impl(acc0, va[1], vb[1]);
        xvi8ger4pp_impl(acc0, va[2], vb[2]);
        if (CountM) {
            xvi8ger4pp_impl(acc1, va[3], vb[0]);
            xvi8ger4pp_impl(acc1, va[4], vb[1]);
            xvi8ger4pp_impl(acc1, va[5], vb[2]);
        }
    } else if (CountK == 8) {
        xvi8ger4pp_impl(acc0, va[0], vb[0]);
        xvi8ger4pp_impl(acc0, va[1], vb[1]);
        if (CountM) {
            xvi8ger4pp_impl(acc1, va[2], vb[0]);
            xvi8ger4pp_impl(acc1, va[3], vb[1]);
        }
    } else {
        xvi8ger4pp_impl(acc0, va[0], vb[0]);
        if (CountM) {
            xvi8ger4pp_impl(acc1, va[1], vb[0]);
        }
    }
};
template<>
size_t
MlasGemmQuantKernel<MLAS_GEMM_QUANT_KERNEL_ZVECTOR>(
    const MLAS_GEMM_QUANT_KERNEL_ZVECTOR::PackedAType* A,
    const MLAS_GEMM_QUANT_KERNEL_ZVECTOR::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountM,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    if (CountM < 8 && CountM >= 4) {
        CountM = 4;
    }
    size_t Mval = CountM;
    if (Mval >= 8) {
        Mval = 4;
    }
    while (CountN > 0) {
        const int8_t *a = A;
        typedef __vector unsigned char vec_t;
        const uint8_t *b = B;
        int32_t *C1;
        MLAS_INT32X4 acc0[4] = {0};
        MLAS_INT32X4 acc1[4] = {0};
        MLAS_INT32X4 acc2[4] = {0};
        MLAS_INT32X4 acc3[4] = {0};
        MLAS_INT32X4 acc4[4] = {0};
        MLAS_INT32X4 acc5[4] = {0};
        MLAS_INT32X4 acc6[4] = {0};
        MLAS_INT32X4 acc7[4] = {0};
        MLAS_INT32X4 result[4] = {0};
        MLAS_INT32X4 result1[4] = {0};
        size_t k = PackedCountK * MLAS_GEMM_QUANT_KERNEL_ZVECTOR::PackedK;
        size_t k1 = PackedCountK;
        //
        // Compute the output block using POWER10 MMA builtins.
        //
        while (k >= 16) {
            vec_t *va = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(a));
            vec_t *vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(b));
            if (CountM >= 8) {
                MlasQgemmComputeZVECTOR<true, 16>(acc0, acc4, va, vb);
            } else {
                MlasQgemmComputeZVECTOR<false, 16>(acc0, acc4, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*16]));
            if (CountM >= 8) {
                MlasQgemmComputeZVECTOR<true, 16>(acc1, acc5, va, vb);
            } else {
                MlasQgemmComputeZVECTOR<false, 16>(acc1, acc5, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*32]));
            if (CountM >= 8) {
                MlasQgemmComputeZVECTOR<true, 16>(acc2, acc6, va, vb);
            } else {
                MlasQgemmComputeZVECTOR<false, 16>(acc2, acc6, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*48]));
            if (CountM >= 8) {
                MlasQgemmComputeZVECTOR<true, 16>(acc3, acc7, va, vb);
            } else {
                MlasQgemmComputeZVECTOR<false, 16>(acc3, acc7, va, vb);
            }
            b += 64;
            if (CountM >= 8) {
                a += 128;
            } else {
                a += 64;
            }
            k -= 16;
        }
        if (k >= 12) {
            vec_t *va = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(a));
            vec_t *vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(b));
            if (CountM >= 8) {
                MlasQgemmComputeZVECTOR<true, 12>(acc0, acc4, va, vb);
            } else {
                MlasQgemmComputeZVECTOR<false, 12>(acc0, acc4, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*16]));
            if (CountM >= 8) {
                MlasQgemmComputeZVECTOR<true, 12>(acc1, acc5, va, vb);
            } else {
                MlasQgemmComputeZVECTOR<false, 12>(acc1, acc5, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*32]));
            if (CountM >= 8) {
                MlasQgemmComputeZVECTOR<true, 12>(acc2, acc6, va, vb);
            } else {
                MlasQgemmComputeZVECTOR<false, 12>(acc2, acc6, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*48]));
            if (CountM >= 8) {
                MlasQgemmComputeZVECTOR<true, 12>(acc3, acc7, va, vb);
            } else {
                MlasQgemmComputeZVECTOR<false, 12>(acc3, acc7, va, vb);
            }
            if (CountM >= 8) {
                a += 96;
            } else {
                a += 48;
            }
            b += 48;
            k -= 12;
        }
        if (k >= 8) {
            vec_t *va = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(a));
            vec_t *vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(b));
            if (CountM >= 8) {
                MlasQgemmComputeZVECTOR<true, 8>(acc0, acc4, va, vb);
            } else {
                MlasQgemmComputeZVECTOR<false, 8>(acc0, acc4, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*16]));
            if (CountM >= 8) {
                MlasQgemmComputeZVECTOR<true, 8>(acc1, acc5, va, vb);
            } else {
                MlasQgemmComputeZVECTOR<false, 8>(acc1, acc5, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*32]));
            if (CountM >= 8) {
                MlasQgemmComputeZVECTOR<true, 8>(acc2, acc6, va, vb);
            } else {
                MlasQgemmComputeZVECTOR<false, 8>(acc2, acc6, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*48]));
            if (CountM >= 8) {
                MlasQgemmComputeZVECTOR<true, 8>(acc3, acc7, va, vb);
            } else {
                MlasQgemmComputeZVECTOR<false, 8>(acc3, acc7, va, vb);
            }
            if (CountM >= 8) {
                a += 64;
            } else {
                a += 32;
            }
            b += 32;
            k -= 8;
        }
        if (k >= 4) {
            vec_t *va = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(a));
            vec_t *vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(b));
            if (CountM >= 8) {
                MlasQgemmComputeZVECTOR<true, 4>(acc0, acc4, va, vb);
            } else {
                MlasQgemmComputeZVECTOR<false, 4>(acc0, acc4, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*16]));
            if (CountM >= 8) {
                MlasQgemmComputeZVECTOR<true, 4>(acc1, acc5, va, vb);
            } else {
                MlasQgemmComputeZVECTOR<false, 4>(acc1, acc5, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*32]));
            if (CountM >= 8) {
                MlasQgemmComputeZVECTOR<true, 4>(acc2, acc6, va, vb);
            } else {
                MlasQgemmComputeZVECTOR<false, 4>(acc2, acc6, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*48]));
            if (CountM >= 8) {
                MlasQgemmComputeZVECTOR<true, 4>(acc3, acc7, va, vb);
            } else {
                MlasQgemmComputeZVECTOR<false, 4>(acc3, acc7, va, vb);
            }
        }
        // Store matrix C with accumulator result.
        if (CountN >= 16) {
            MlasQgemmStoreVectorZVECTOR<0>(acc0, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 0);
            MlasQgemmStoreVectorZVECTOR<4>(acc1, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 4);
            MlasQgemmStoreVectorZVECTOR<8>(acc2, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 8);
            MlasQgemmStoreVectorZVECTOR<12>(acc3, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 12);

            if (CountM >= 8) {
                C1 = C+ldc*4;

                MlasQgemmStoreVectorZVECTOR<0>(acc4, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 0);
                MlasQgemmStoreVectorZVECTOR<4>(acc5, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 4);
                MlasQgemmStoreVectorZVECTOR<8>(acc6, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 8);
                MlasQgemmStoreVectorZVECTOR<12>(acc7, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 12);
            }
            INC_BUFFER(16);
            CountN -= 16;
            B += 16 * 4 *PackedCountK;
            C += 16;
        } else {
            if (CountN >=12 ) {
                MlasQgemmStoreVectorZVECTOR<0>(acc0, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 0);
                MlasQgemmStoreVectorZVECTOR<4>(acc1, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 4);
                MlasQgemmStoreVectorZVECTOR<8>(acc2, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 8);
                if (CountM >= 8) {
                    C1 = C+ldc*4;

                    MlasQgemmStoreVectorZVECTOR<0>(acc4, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 0);
                    MlasQgemmStoreVectorZVECTOR<4>(acc5, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 4);
                    MlasQgemmStoreVectorZVECTOR<8>(acc6, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 8);
                }
                INC_BUFFER(12);
                if (CountN - 12 > 0) {
                    for (size_t i = 0; i < 4; ++i) {
                        result[i] = acc3[i];
                    }
                    if (CountM >= 8) {
                        for (size_t i = 0; i < 4; ++i) {
                            result1[i] = acc7[i];
                        }
                    }
                }
                CountN -= 12;
                C += 12;
            } else if (CountN >= 8) {
                MlasQgemmStoreVectorZVECTOR<0>(acc0, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 0);
                MlasQgemmStoreVectorZVECTOR<4>(acc1, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 4);
                if (CountM >= 8) {
                    C1 = C+ldc*4;

                    MlasQgemmStoreVectorZVECTOR<0>(acc4, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 0);
                    MlasQgemmStoreVectorZVECTOR<4>(acc5, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 4);
                }
                INC_BUFFER(8);
                if (CountN - 8 > 0) {
                    for (size_t i = 0; i < 4; ++i) {
                        result[i] = acc2[i];
                    }
                    if (CountM >= 8) {
                        for (size_t i = 0; i < 4; ++i) {
                            result1[i] = acc6[i];
                        }
                    }
                }
                CountN -= 8;
                C += 8;
            } else if (CountN >= 4) {
                MlasQgemmStoreVectorZVECTOR<0>(acc0, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 0);
                if (CountM >= 8) {
                    C1 = C+ldc*4;

                    MlasQgemmStoreVectorZVECTOR<0>(acc4, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 0);
                }
                INC_BUFFER(4);
                if (CountN - 4 > 0) {
                    for (size_t i = 0; i < 4; ++i) {
                        result[i] = acc1[i];
                    }
                    if (CountM >= 8) {
                        for (size_t i = 0; i < 4; ++i) {
                            result1[i] = acc5[i];
                        }
                    }
                }
                CountN -= 4;
                C += 4;
            } else {
                for (size_t i = 0; i < 4; ++i) {
                    result[i] = acc0[i];
                }
                if (CountM >= 8) {
                    for (size_t i = 0; i < 4; ++i) {
                        result1[i] = acc4[i];
                    }
                }
            }
            CountN &= 3;
            //
            // Output the remaining partial output block.
            //
            if (CountN > 0) {
                MlasQgemmStoreScalarZVECTOR<0>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB);
                if (CountM >= 8) {
                    MlasQgemmStoreScalarZVECTOR<0>(result1, C + (ldc*4), ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB);
                }
                INC_BUFFER(1);
                if (CountN >= 2) {
                     MlasQgemmStoreScalarZVECTOR<1>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB);
                     if (CountM >= 8) {
                         MlasQgemmStoreScalarZVECTOR<1>(result1, C + (ldc*4), ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB);
                     }
                     INC_BUFFER(1);
                }
                if (CountN >= 3) {
                     MlasQgemmStoreScalarZVECTOR<2>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB);
                     if (CountM >= 8) {
                         MlasQgemmStoreScalarZVECTOR<2>(result1, C + (ldc*4), ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB);
                     }
                     INC_BUFFER(1);
                }
            }
            CountN = 0;
        }
    }
    if (CountM >= 8) {
       return 8;
    }
    return CountM;
}

const MLAS_GEMM_QUANT_DISPATCH MlasGemm8X8DispatchZVECTOR = {
    MlasGemmQuantOperation<MLAS_GEMM_QUANT_KERNEL_ZVECTOR>,
    MlasGemmQuantPackedOperation<MLAS_GEMM_QUANT_KERNEL_ZVECTOR>,
    MlasGemmQuantCopyPackB<MLAS_GEMM_QUANT_KERNEL_ZVECTOR>,
    MLAS_GEMM_QUANT_KERNEL_ZVECTOR::PackedK,
    MLAS_GEMM_QUANT_KERNEL_ZVECTOR::PackedStrides.K,
    8 // Kernel M stride
};
