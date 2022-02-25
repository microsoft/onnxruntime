/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    qgemm_kernel_power10.cpp

Abstract:

    This module implements QGEMM kernel for POWER10.

--*/

#include "mlasi.h"
#include "qgemm.h"
#include <inttypes.h>

struct MLAS_GEMM_QUANT_KERNEL_POWER10
{
    typedef int8_t PackedAType;
    typedef uint8_t PackedBType;
    typedef int8_t OffsetAType;
    typedef uint8_t OffsetBType;
    static constexpr size_t PackedK = 4;
    static constexpr MLAS_GEMM_QUANT_STRIDES Strides{ 16, 256, 384 };
    static constexpr MLAS_GEMM_QUANT_STRIDES PackedStrides{ 16, 128, 128 };
};

constexpr size_t MLAS_GEMM_QUANT_KERNEL_POWER10::PackedK;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_QUANT_KERNEL_POWER10::Strides;
constexpr MLAS_GEMM_QUANT_STRIDES MLAS_GEMM_QUANT_KERNEL_POWER10::PackedStrides;

#define INC_BUFFER(cnt) \
   ColumnSumBuffer += cnt; \
   if (ZeroPointB != nullptr) { \
       ZeroPointB += cnt; \
   }
template<>
MLAS_FORCEINLINE constexpr
int32_t
MlasGemmQuantFixupZeroPointA<MLAS_GEMM_QUANT_KERNEL_POWER10>(
    int32_t ZeroPointA,
    bool AIsSigned
    )
{
    if (!AIsSigned) {
        ZeroPointA = MLAS_GEMM_QUANT_KERNEL_POWER10::OffsetAType(ZeroPointA ^ 0x80);
    }
    return ZeroPointA;
}

template<>
MLAS_FORCEINLINE
int32_t
MlasGemmQuantFixupZeroPointB<MLAS_GEMM_QUANT_KERNEL_POWER10>(
    int32_t ZeroPointB,
    bool BIsSigned
    )
{
    if (BIsSigned) {
        ZeroPointB = MLAS_GEMM_QUANT_KERNEL_POWER10::OffsetBType(ZeroPointB ^ 0x80);
    }
    return ZeroPointB;

}

template<typename Vtype>
void
MlasGemmQuantCopyPackA8x8(
    MLAS_GEMM_QUANT_KERNEL_POWER10::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned
    )
{
    const uint8_t Flip = (AIsSigned ? 0 : 0x80);
    Vtype vmask = reinterpret_cast<Vtype>(vec_splats(Flip));
    typedef __vector signed char vec_t;

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
        __vector signed int vsum = { 0 };
        __vector signed int vsum2 = { 0 };
        size_t y = CountK;
        while (y >= 16) {
            Vtype a1 = *reinterpret_cast<const Vtype *>(&a[0]);
            Vtype a2 = *reinterpret_cast<const Vtype *>(&a[lda]);
            Vtype a3 = *reinterpret_cast<const Vtype *>(&a[lda * 2]);
            Vtype a4 = *reinterpret_cast<const Vtype *>(&a[lda * 3]);
            Vtype vx =
               reinterpret_cast<Vtype>(vec_mergee (reinterpret_cast<__vector int>(a1),
                           reinterpret_cast<__vector int>(a2)));
            Vtype vx1 =
               reinterpret_cast<Vtype>(vec_mergee (reinterpret_cast<__vector int>(a3),
                           reinterpret_cast<__vector int>(a4)));
            Vtype vx2 =
               reinterpret_cast<Vtype>(vec_mergeo (reinterpret_cast<__vector int>(a1),
                           reinterpret_cast<__vector int>(a2)));
            Vtype vx3 =
               reinterpret_cast<Vtype>(vec_mergeo (reinterpret_cast<__vector int>(a3),
                           reinterpret_cast<__vector int>(a4)));
            Vtype vx4 = vec_xxpermdi (vx, vx1, 0);
            Vtype vx5 = vec_xxpermdi (vx2, vx3, 0);
            Vtype vx6 = vec_xxpermdi (vx, vx1, 3);
            Vtype vx7 = vec_xxpermdi (vx2, vx3, 3);
            a1 = *reinterpret_cast<const Vtype *>(&a[lda*4]);
            a2 = *reinterpret_cast<const Vtype *>(&a[lda*5]);
            a3 = *reinterpret_cast<const Vtype *>(&a[lda*6]);
            a4 = *reinterpret_cast<const Vtype *>(&a[lda*7]);
            vx =
              reinterpret_cast<Vtype>(vec_mergee (reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2)));
            vx1 =
              reinterpret_cast<Vtype>(vec_mergee (reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4)));
            vx2 =
              reinterpret_cast<Vtype>(vec_mergeo (reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2)));
            vx3 =
              reinterpret_cast<Vtype>(vec_mergeo (reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4)));
            Vtype vx8 = vec_xxpermdi (vx, vx1, 0);
            Vtype vx9 = vec_xxpermdi (vx2, vx3, 0);
            Vtype vx10 = vec_xxpermdi (vx, vx1, 3);
            Vtype vx11 = vec_xxpermdi (vx2, vx3, 3);
            vec_t vxx =
              reinterpret_cast<vec_t>(vec_sub (vx4, vmask));
            vsum = vec_sum4s (vxx, vsum);
            *reinterpret_cast<vec_t *>(&D[0]) = vxx;
            vxx = reinterpret_cast<vec_t>(vec_sub (vx5, vmask));
            vsum = vec_sum4s (vxx, vsum);
            *reinterpret_cast<vec_t *>(&D[16]) = vxx;
            vxx = reinterpret_cast<vec_t>(vec_sub (vx6, vmask));
            vsum = vec_sum4s (vxx, vsum);
            *reinterpret_cast<vec_t *>(&D[32]) = vxx;
            vxx = reinterpret_cast<vec_t>(vec_sub (vx7, vmask));
            vsum = vec_sum4s (vxx, vsum);
            *reinterpret_cast<vec_t *>(&D[48]) = vxx;
            vxx = reinterpret_cast<vec_t>(vec_sub (vx8, vmask));
            *reinterpret_cast<vec_t *>(&D[64]) = vxx;
            vsum2 = vec_sum4s (vxx, vsum2);
            vxx = reinterpret_cast<vec_t>(vec_sub (vx9, vmask));
            *reinterpret_cast<vec_t *>(&D[80]) = vxx;
            vsum2 = vec_sum4s (vxx, vsum2);
            vxx = reinterpret_cast<vec_t>(vec_sub (vx10, vmask));
            *reinterpret_cast<vec_t *>(&D[96]) = vxx;
            vsum2 = vec_sum4s (vxx, vsum2);
            vxx = reinterpret_cast<vec_t>(vec_sub (vx11, vmask));
            *reinterpret_cast<vec_t *>(&D[112]) = vxx;
            vsum2 = vec_sum4s (vxx, vsum2);
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
            vec_t vx =
              reinterpret_cast<vec_t>(vec_sub (reinterpret_cast<Vtype>(vx1), vmask));
            vsum = vec_sum4s (vx, vsum);
            *reinterpret_cast<vec_t *>(&D[0]) = vx;
            a1 = *reinterpret_cast<const int *>(&a[lda*4]);
            a2 = *reinterpret_cast<const int *>(&a[lda*5]);
            a3 = *reinterpret_cast<const int *>(&a[lda*6]);
            a4 = *reinterpret_cast<const int *>(&a[lda*7]);
            __vector int vx2 = { a1, a2, a3, a4};
            vx = reinterpret_cast<vec_t>(vec_sub (reinterpret_cast<Vtype>(vx2), vmask));
            vsum2 = vec_sum4s (vx, vsum2);
            if (CountK & 3) {
                if (yval >= 12) {
                     *reinterpret_cast<vec_t *>(&D[64]) = vx;
                } else if (yval >= 8) {
                     *reinterpret_cast<vec_t *>(&D[48]) = vx;
                } else {
                     *reinterpret_cast<vec_t *>(&D[32]) = vx;
                }
            } else {
                if (yval >= 12) {
                     *reinterpret_cast<vec_t *>(&D[48]) = vx;
                } else if (yval >= 8) {
                     *reinterpret_cast<vec_t *>(&D[32]) = vx;
                } else {
                     *reinterpret_cast<vec_t *>(&D[16]) = vx;
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
            Vtype a1 = reinterpret_cast<Vtype>(vec_splats(Flip));
            Vtype a2 = reinterpret_cast<Vtype>(vec_splats(Flip));
            Vtype a3 = reinterpret_cast<Vtype>(vec_splats(Flip));
            Vtype a4 = reinterpret_cast<Vtype>(vec_splats(Flip));
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
            Vtype vx =
              reinterpret_cast<Vtype>(vec_mergee (reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2)));
            Vtype vx1 =
              reinterpret_cast<Vtype>(vec_mergee (reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4)));
            Vtype vx2 = vec_xxpermdi (vx, vx1, 0);
            vec_t vx3 =
              reinterpret_cast<vec_t>(vec_sub (vx2, vmask));
            vsum = vec_sum4s (vx3, vsum);
            *reinterpret_cast<vec_t *>(&D[0]) = vx3;
            a1 = reinterpret_cast<Vtype>(vec_splats(Flip));
            a2 = reinterpret_cast<Vtype>(vec_splats(Flip));
            a3 = reinterpret_cast<Vtype>(vec_splats(Flip));
            a4 = reinterpret_cast<Vtype>(vec_splats(Flip));
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
              reinterpret_cast<Vtype>(vec_mergee (reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2)));
            vx1 =
              reinterpret_cast<Vtype>(vec_mergee (reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4)));
            vx2 = vec_xxpermdi (vx, vx1, 0);
            vx3 = reinterpret_cast<vec_t>(vec_sub (vx2, vmask));
            vsum2 = vec_sum4s (vx3, vsum2);
            if (CountK % 16 >= 12) {
                *reinterpret_cast<vec_t *>(&D[64]) = vx3;
                D += 80;
            } else if (CountK % 16 >= 8) {
                *reinterpret_cast<vec_t *>(&D[48]) = vx3;
                D += 64;
            } else if (CountK % 16 >= 4) {
                *reinterpret_cast<vec_t *>(&D[32]) = vx3;
                D += 48;
            } else {
                *reinterpret_cast<vec_t *>(&D[16]) = vx3;
                D += 16 * 2;
            }
            a += 16;
        }
        A += lda * 8;
        *RowSumBuffer++ = vsum[0];
        *RowSumBuffer++ = vsum[1];
        *RowSumBuffer++ = vsum[2];
        *RowSumBuffer++ = vsum[3];
        *RowSumBuffer++ = vsum2[0];
        *RowSumBuffer++ = vsum2[1];
        *RowSumBuffer++ = vsum2[2];
        *RowSumBuffer++ = vsum2[3];
        CountM -= 8;
    }

    // Process four rows of matrix A in a loop.
    //
    if (CountM >= 4)
    {
        const uint8_t *a = A;
        __vector signed int vsum = { 0 };
        size_t y = CountK;

        while (y >= 16)
        {
            Vtype a1 = *reinterpret_cast<const Vtype *>(&a[0]);
            Vtype a2 = *reinterpret_cast<const Vtype *>(&a[lda]);
            Vtype a3 = *reinterpret_cast<const Vtype *>(&a[lda * 2]);
            Vtype a4 = *reinterpret_cast<const Vtype *>(&a[lda * 3]);
            Vtype vx =
              reinterpret_cast<Vtype>(vec_mergee (reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2)));
            Vtype vx1 =
              reinterpret_cast<Vtype>(vec_mergee (reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4)));
            Vtype vx2 =
              reinterpret_cast<Vtype>(vec_mergeo (reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2)));
            Vtype vx3 =
              reinterpret_cast<Vtype>(vec_mergeo (reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4)));
            Vtype vx4 = vec_xxpermdi (vx, vx1, 0);
            Vtype vx5 = vec_xxpermdi (vx2, vx3, 0);
            Vtype vx6 = vec_xxpermdi (vx, vx1, 3);
            Vtype vx7 = vec_xxpermdi (vx2, vx3, 3);
            vec_t vx0 =
              reinterpret_cast<vec_t>(vec_sub (vx4, vmask));
            *reinterpret_cast<vec_t *>(&D[0]) = vx0;
            vsum = vec_sum4s (vx0, vsum);
            vx0 = reinterpret_cast<vec_t>(vec_sub (vx5, vmask));
            *reinterpret_cast<vec_t *>(&D[16]) = vx0;
            vsum = vec_sum4s (vx0, vsum);
            vx0 = reinterpret_cast<vec_t>(vec_sub (vx6, vmask));
            *reinterpret_cast<vec_t *>(&D[32]) = vx0;
            vsum = vec_sum4s (vx0, vsum);
            vx0 = reinterpret_cast<vec_t>(vec_sub (vx7, vmask));
            *reinterpret_cast<vec_t *>(&D[48]) = vx0;
            vsum = vec_sum4s (vx0, vsum);
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
            vec_t vx =
              reinterpret_cast<vec_t>(vec_sub (reinterpret_cast<Vtype>(vx1), vmask));
            *reinterpret_cast<vec_t *>(&D[0]) = vx;
            vsum = vec_sum4s (vx, vsum);
            D += 16;
            a += 4;
            y -= 4;
        }
        if (y >= 1)
        {
            Vtype vx = reinterpret_cast<Vtype>(vec_splats(Flip));
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
            vec_t vx1 =
               reinterpret_cast<vec_t>(vec_sub (vx, vmask));
            *reinterpret_cast<vec_t *>(&D[0]) = vx1;
            vsum = vec_sum4s (vx1, vsum);
            D += 16;
            a += 16;
        }
        A += lda * 4;
        *RowSumBuffer++ = vsum[0];
        *RowSumBuffer++ = vsum[1];
        *RowSumBuffer++ = vsum[2];
        *RowSumBuffer++ = vsum[3];
        CountM -= 4;
    }

    // Process remaining rows of matrix A in a loop.
    //
    if (CountM <= 3 && CountM > 0) {
        const uint8_t *a = A;
        size_t y = CountK;
        __vector signed int vsum = { 0 };

        while (y >= 16) {
            Vtype a4 = reinterpret_cast<Vtype>(vec_splats(Flip));
            Vtype a2 = reinterpret_cast<Vtype>(vec_splats(Flip));
            Vtype a3 = reinterpret_cast<Vtype>(vec_splats(Flip));
            Vtype a1 = *reinterpret_cast<const Vtype *>(&a[0]);
            if (CountM == 3) {
                a3 = *reinterpret_cast<const Vtype *>(&a[lda * 2]);
            }
            if (CountM >= 2) {
                a2 = *reinterpret_cast<const Vtype *>(&a[lda]);
            }
            Vtype vx =
              reinterpret_cast<Vtype>(vec_mergee (reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2)));
            Vtype vx1 =
              reinterpret_cast<Vtype>(vec_mergee (reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4)));
            Vtype vx2 =
              reinterpret_cast<Vtype>(vec_mergeo (reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2)));
            Vtype vx3 =
              reinterpret_cast<Vtype>(vec_mergeo (reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4)));
            Vtype vx4 = vec_xxpermdi (vx, vx1, 0);
            Vtype vx5 = vec_xxpermdi (vx2, vx3, 0);
            Vtype vx6 = vec_xxpermdi (vx, vx1, 3);
            Vtype vx7 = vec_xxpermdi (vx2, vx3, 3);
            vec_t vx0 =
              reinterpret_cast<vec_t>(vec_sub (vx4, vmask));
            *reinterpret_cast<vec_t *>(&D[0]) = vx0;
            vsum = vec_sum4s (vx0, vsum);
            vx0 = reinterpret_cast<vec_t>(vec_sub (vx5, vmask));
            *reinterpret_cast<vec_t *>(&D[16]) = vx0;
            vsum = vec_sum4s (vx0, vsum);
            vx0 = reinterpret_cast<vec_t>(vec_sub (vx6, vmask));
            *reinterpret_cast<vec_t *>(&D[32]) = vx0;
            vsum = vec_sum4s (vx0, vsum);
            vx0 = reinterpret_cast<vec_t>(vec_sub (vx7, vmask));
            *reinterpret_cast<vec_t *>(&D[48]) = vx0;
            vsum = vec_sum4s (vx0, vsum);
            D += 16 * 4;
            a += 16;
            y -= 16;
        }
        while (y >= 4)
        {
            Vtype vb = reinterpret_cast<Vtype>(vec_splats(Flip));
            __vector int vx1 = reinterpret_cast<__vector int>(vb);
            vx1[0] = *reinterpret_cast<const int *>(&a[0]);
            if(CountM >= 2) {
                vx1[1] = *reinterpret_cast<const int *>(&a[lda]);
            }
            if(CountM >= 3) {
                vx1[2] = *reinterpret_cast<const int *>(&a[lda*2]);
            }
            vec_t vx =
              reinterpret_cast<vec_t>(vec_sub (reinterpret_cast<Vtype>(vx1), vmask));
            *reinterpret_cast<vec_t *>(&D[0]) = vx;
            vsum = vec_sum4s (vx, vsum);
            D += 16;
            a += 4;
            y -= 4;
        }
        if (y >= 1)
        {
            int8_t vz = 0;
            vec_t vx = vec_splats(vz);
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
            *reinterpret_cast<vec_t *>(&D[0]) = vx;
            vsum = vec_sum4s (vx, vsum);
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

template<typename Vtype>
void
MlasGemmQuantCopyPackB8x8(
    MLAS_GEMM_QUANT_KERNEL_POWER10::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    const uint8_t BitFlipValue = (BIsSigned ? 0x80 : 0);
    typedef __vector unsigned char vec_t;
    Vtype vmask = reinterpret_cast<Vtype>(vec_splats(BitFlipValue));
    vec_t mask = {0,4,8,12,1,5,9,13,2,6,10,14,3,7,11,15};
    const int8_t Flip = (BIsSigned ? -128 : 0);

    // Process 4 columns of matrix B in a loop.
    //
    // Copy columns from matrix B to the packed buffer. Signed buffers are
    // converted to unsigned buffers in order to share a common kernel.
    //
    // If CountK is not aligned to a multiple of four, then the packed buffer
    // is padded with zero vectors.
    while (CountN >= 4) {

        const uint8_t* b = B;
        __vector unsigned int vsum = {0};
        size_t y = CountK;
        if(y >= 4) {
            do {
                int b1 = *reinterpret_cast<const int *>(&b[0]);
                int b2 = *reinterpret_cast<const int *>(&b[ldb]);
                int b3 = *reinterpret_cast<const int *>(&b[ldb*2]);
                int b4 = *reinterpret_cast<const int *>(&b[ldb*3]);
                __vector int vb = {b1, b2, b3, b4};
                Vtype vx = vec_perm(reinterpret_cast<Vtype>(vb), reinterpret_cast<Vtype>(vb), mask);
                vec_t vx1 = reinterpret_cast<vec_t>(vec_add (vx, vmask));
                *reinterpret_cast<vec_t *>(&D[0]) = vx1;
                vsum = vec_sum4s (vx1, vsum);
                D += 16;
                b += ldb*4;
                y -= 4;
            } while (y >= 4);
        }
        if (y >= 1) {
            Vtype vb = reinterpret_cast<Vtype>(vec_splats(Flip));
            __vector int vb1 = reinterpret_cast<__vector int>(vb);
            vb1[0] = *reinterpret_cast<const int *>(&b[0]);
            if( y >= 2) {
                vb1[1] = *reinterpret_cast<const int *>(&b[ldb]);
            }
            if( y >= 3) {
                vb1[2] = *reinterpret_cast<const int *>(&b[ldb*2]);
            }
            Vtype vx = vec_perm(reinterpret_cast<Vtype>(vb1), reinterpret_cast<Vtype>(vb1), mask);
            vec_t vx1 = reinterpret_cast<vec_t>(vec_add (vx, vmask));
            *reinterpret_cast<vec_t *>(&D[0]) = vx1;
            vsum = vec_sum4s (vx1, vsum);
            D += 16;
        }
        *ColumnSumBuffer++ = vsum[0];
        *ColumnSumBuffer++ = vsum[1];
        *ColumnSumBuffer++ = vsum[2];
        *ColumnSumBuffer++ = vsum[3];
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
                Vtype vb = reinterpret_cast<Vtype>(vec_splats(Flip));
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
                vec_t vx1 = reinterpret_cast<vec_t>(vec_add (vx, vmask));
                *reinterpret_cast<vec_t *>(&D[0]) = vx1;
                vsum = vec_sum4s (vx1, vsum);
                D += 16;
                b += ldb*4;
                y -= 4;
            } while (y >= 4);
        }
        if (y >= 1) {
            Vtype vb = reinterpret_cast<Vtype>(vec_splats(Flip));
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
            vec_t vx1 = reinterpret_cast<vec_t>(vec_add (vx, vmask));
            *reinterpret_cast<vec_t *>(&D[0]) = vx1;
            vsum = vec_sum4s (vx1, vsum);
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
MlasGemmQuantCopyPackA<MLAS_GEMM_QUANT_KERNEL_POWER10>(
    MLAS_GEMM_QUANT_KERNEL_POWER10::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer,
    bool AIsSigned
    )
{
    if (AIsSigned) {
        MlasGemmQuantCopyPackA8x8<__vector signed char>(D, A, lda, CountM, CountK, RowSumBuffer, AIsSigned);
    } else {
        MlasGemmQuantCopyPackA8x8<__vector unsigned char>(D, A, lda, CountM, CountK, RowSumBuffer, AIsSigned);
    }
}
template<>
void
MlasGemmQuantCopyPackB<MLAS_GEMM_QUANT_KERNEL_POWER10>(
    MLAS_GEMM_QUANT_KERNEL_POWER10::PackedBType* D,
    const uint8_t* B,
    size_t ldb,
    size_t CountN,
    size_t CountK,
    int32_t* ColumnSumBuffer,
    bool BIsSigned
    )
{
    if (BIsSigned) {
        MlasGemmQuantCopyPackB8x8<__vector signed char>(D, B, ldb, CountN, CountK, ColumnSumBuffer, BIsSigned);
    } else {
        MlasGemmQuantCopyPackB8x8< __vector unsigned char>(D, B, ldb, CountN, CountK, ColumnSumBuffer, BIsSigned);
    }
}

template<size_t VectorCount>
MLAS_FORCEINLINE
void
MlasQgemmStoreVectorMMA
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
    __vector int *rowC;
    __vector signed int vsum = {0};
    if (ZeroPointB != nullptr) {
        if (ZeroMode) {
            for (size_t RowCount = 0;RowCount < row; RowCount++){
                vsum[0] = RowSumBuffer[RowCount] * ZeroPointB[pos] + ColumnSumBuffer[pos];
                vsum[1] = RowSumBuffer[RowCount] * ZeroPointB[pos+1] + ColumnSumBuffer[pos+1];
                vsum[2] = RowSumBuffer[RowCount] * ZeroPointB[pos+2] + ColumnSumBuffer[pos+2];
                vsum[3] = RowSumBuffer[RowCount] * ZeroPointB[pos+3] + ColumnSumBuffer[pos+3];
                rowC = reinterpret_cast<__vector int *>(&C[ldc * RowCount + VectorCount]);
                rowC[0] = *reinterpret_cast<__vector int *>(&result[RowCount]) + vsum;
            }
        } else {
            for (size_t RowCount = 0;RowCount < row; RowCount++){
                vsum[0] = RowSumBuffer[RowCount] * ZeroPointB[pos] + ColumnSumBuffer[pos];
                vsum[1] = RowSumBuffer[RowCount] * ZeroPointB[pos+1] + ColumnSumBuffer[pos+1];
                vsum[2] = RowSumBuffer[RowCount] * ZeroPointB[pos+2] + ColumnSumBuffer[pos+2];
                vsum[3] = RowSumBuffer[RowCount] * ZeroPointB[pos+3] + ColumnSumBuffer[pos+3];
                rowC = reinterpret_cast<__vector int *>(&C[ldc * RowCount + VectorCount]);
                rowC[0] += *reinterpret_cast<__vector int *>(&result[RowCount]) + vsum;
            }
        }
    } else {
        if (ZeroMode) {
            for (size_t RowCount = 0;RowCount < row; RowCount++){
                vsum[0] = RowSumBuffer[RowCount] + ColumnSumBuffer[pos];
                vsum[1] = RowSumBuffer[RowCount] + ColumnSumBuffer[pos+1];
                vsum[2] = RowSumBuffer[RowCount] + ColumnSumBuffer[pos+2];
                vsum[3] = RowSumBuffer[RowCount] + ColumnSumBuffer[pos+3];
                rowC = reinterpret_cast<__vector int *>(&C[ldc * RowCount + VectorCount]);
                rowC[0] = *reinterpret_cast<__vector int *>(&result[RowCount]) + vsum;
            }
        } else {
            for (size_t RowCount = 0;RowCount < row; RowCount++){
                vsum[0] = RowSumBuffer[RowCount] + ColumnSumBuffer[pos];
                vsum[1] = RowSumBuffer[RowCount] + ColumnSumBuffer[pos+1];
                vsum[2] = RowSumBuffer[RowCount] + ColumnSumBuffer[pos+2];
                vsum[3] = RowSumBuffer[RowCount] + ColumnSumBuffer[pos+3];
                rowC = reinterpret_cast<__vector int *>(&C[ldc * RowCount + VectorCount]);
                rowC[0] += *reinterpret_cast<__vector int *>(&result[RowCount]) + vsum;
            }
        }
    }
};
template<size_t Lane>
MLAS_FORCEINLINE
void
MlasQgemmStoreScalarMMA(
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
template<bool CountM, size_t CountK>
MLAS_FORCEINLINE
void
MlasQgemmComputeMMA(
    __vector_quad *acc0,
    __vector_quad *acc1,
    __vector unsigned char *va,
    __vector unsigned char *vb
    )
{
    if (CountK == 16) {
        __builtin_mma_xvi8ger4pp (acc0, va[0], vb[0]);
        __builtin_mma_xvi8ger4pp (acc0, va[1], vb[1]);
        __builtin_mma_xvi8ger4pp (acc0, va[2], vb[2]);
        __builtin_mma_xvi8ger4pp (acc0, va[3], vb[3]);
        if (CountM) {
            __builtin_mma_xvi8ger4pp (acc1, va[4], vb[0]);
            __builtin_mma_xvi8ger4pp (acc1, va[5], vb[1]);
            __builtin_mma_xvi8ger4pp (acc1, va[6], vb[2]);
            __builtin_mma_xvi8ger4pp (acc1, va[7], vb[3]);
        }
    } else if (CountK == 12) {
        __builtin_mma_xvi8ger4pp (acc0, va[0], vb[0]);
        __builtin_mma_xvi8ger4pp (acc0, va[1], vb[1]);
        __builtin_mma_xvi8ger4pp (acc0, va[2], vb[2]);
        if (CountM) {
            __builtin_mma_xvi8ger4pp (acc1, va[3], vb[0]);
            __builtin_mma_xvi8ger4pp (acc1, va[4], vb[1]);
            __builtin_mma_xvi8ger4pp (acc1, va[5], vb[2]);
        }
    } else if (CountK == 8) {
        __builtin_mma_xvi8ger4pp (acc0, va[0], vb[0]);
        __builtin_mma_xvi8ger4pp (acc0, va[1], vb[1]);
        if (CountM) {
            __builtin_mma_xvi8ger4pp (acc1, va[2], vb[0]);
            __builtin_mma_xvi8ger4pp (acc1, va[3], vb[1]);
        }
    } else {
        __builtin_mma_xvi8ger4pp (acc0, va[0], vb[0]);
        if (CountM) {
            __builtin_mma_xvi8ger4pp (acc1, va[1], vb[0]);
        }
    }
};
template<>
size_t
MlasGemmQuantKernel<MLAS_GEMM_QUANT_KERNEL_POWER10>(
    const MLAS_GEMM_QUANT_KERNEL_POWER10::PackedAType* A,
    const MLAS_GEMM_QUANT_KERNEL_POWER10::PackedBType* B,
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
    while(CountN > 0) {
        const int8_t *a = A;
        typedef __vector unsigned char vec_t;
        const uint8_t *b = B;
        int32_t *C1;
        __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
        //
        // Initialize the accumulators with zero.
        //
        __builtin_mma_xxsetaccz(&acc0);
        __builtin_mma_xxsetaccz(&acc1);
        __builtin_mma_xxsetaccz(&acc2);
        __builtin_mma_xxsetaccz(&acc3);
        __builtin_mma_xxsetaccz(&acc4);
        __builtin_mma_xxsetaccz(&acc5);
        __builtin_mma_xxsetaccz(&acc6);
        __builtin_mma_xxsetaccz(&acc7);
        MLAS_INT32X4 result[4] = {0};
        MLAS_INT32X4 result1[4] = {0};
        size_t k = PackedCountK * MLAS_GEMM_QUANT_KERNEL_POWER10::PackedK;
        size_t k1 = PackedCountK;
        //
        // Compute the output block using POWER10 MMA builtins.
        //
        while (k >= 16) {
            vec_t *va = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(a));
            vec_t *vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(b));
            if (CountM >= 8) {
                MlasQgemmComputeMMA<true, 16>(&acc0, &acc4, va, vb);
            } else {
                MlasQgemmComputeMMA<false, 16>(&acc0, &acc4, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*16]));
            if (CountM >= 8) {
                MlasQgemmComputeMMA<true, 16>(&acc1, &acc5, va, vb);
            } else {
                MlasQgemmComputeMMA<false, 16>(&acc1, &acc5, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*32]));
            if (CountM >= 8) {
                MlasQgemmComputeMMA<true, 16>(&acc2, &acc6, va, vb);
            } else {
                MlasQgemmComputeMMA<false, 16>(&acc2, &acc6, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*48]));
            if (CountM >= 8) {
                MlasQgemmComputeMMA<true, 16>(&acc3, &acc7, va, vb);
            } else {
                MlasQgemmComputeMMA<false, 16>(&acc3, &acc7, va, vb);
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
                MlasQgemmComputeMMA<true, 12>(&acc0, &acc4, va, vb);
            } else {
                MlasQgemmComputeMMA<false, 12>(&acc0, &acc4, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*16]));
            if (CountM >= 8) {
                MlasQgemmComputeMMA<true, 12>(&acc1, &acc5, va, vb);
            } else {
                MlasQgemmComputeMMA<false, 12>(&acc1, &acc5, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*32]));
            if (CountM >= 8) {
                MlasQgemmComputeMMA<true, 12>(&acc2, &acc6, va, vb);
            } else {
                MlasQgemmComputeMMA<false, 12>(&acc2, &acc6, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*48]));
            if (CountM >= 8) {
                MlasQgemmComputeMMA<true, 12>(&acc3, &acc7, va, vb);
            } else {
                MlasQgemmComputeMMA<false, 12>(&acc3, &acc7, va, vb);
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
                MlasQgemmComputeMMA<true, 8>(&acc0, &acc4, va, vb);
            } else {
                MlasQgemmComputeMMA<false, 8>(&acc0, &acc4, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*16]));
            if (CountM >= 8) {
                MlasQgemmComputeMMA<true, 8>(&acc1, &acc5, va, vb);
            } else {
                MlasQgemmComputeMMA<false, 8>(&acc1, &acc5, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*32]));
            if (CountM >= 8) {
                MlasQgemmComputeMMA<true, 8>(&acc2, &acc6, va, vb);
            } else {
                MlasQgemmComputeMMA<false, 8>(&acc2, &acc6, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*48]));
            if (CountM >= 8) {
                MlasQgemmComputeMMA<true, 8>(&acc3, &acc7, va, vb);
            } else {
                MlasQgemmComputeMMA<false, 8>(&acc3, &acc7, va, vb);
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
                MlasQgemmComputeMMA<true, 4>(&acc0, &acc4, va, vb);
            } else {
                MlasQgemmComputeMMA<false, 4>(&acc0, &acc4, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*16]));
            if (CountM >= 8) {
                MlasQgemmComputeMMA<true, 4>(&acc1, &acc5, va, vb);
            } else {
                MlasQgemmComputeMMA<false, 4>(&acc1, &acc5, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*32]));
            if (CountM >= 8) {
                MlasQgemmComputeMMA<true, 4>(&acc2, &acc6, va, vb);
            } else {
                MlasQgemmComputeMMA<false, 4>(&acc2, &acc6, va, vb);
            }
            vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*48]));
            if (CountM >= 8) {
                MlasQgemmComputeMMA<true, 4>(&acc3, &acc7, va, vb);
            } else {
                MlasQgemmComputeMMA<false, 4>(&acc3, &acc7, va, vb);
            }
        }
        // Store matrix C with accumulator result.
        if (CountN >=16) {
            __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc0);
            MlasQgemmStoreVectorMMA<0>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 0);
            __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc1);
            MlasQgemmStoreVectorMMA<4>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 4);
            __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc2);
            MlasQgemmStoreVectorMMA<8>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 8);
            __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc3);
            MlasQgemmStoreVectorMMA<12>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 12);
            if (CountM >= 8) {
                C1 = C+ldc*4;
                __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc4);
                MlasQgemmStoreVectorMMA<0>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 0);
                __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc5);
                MlasQgemmStoreVectorMMA<4>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 4);
                __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc6);
                MlasQgemmStoreVectorMMA<8>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 8);
                __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc7);
                MlasQgemmStoreVectorMMA<12>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 12);
            }
            INC_BUFFER(16);
            CountN -= 16;
            B += 16 * 4 *PackedCountK;
            C += 16;
        } else {
            if (CountN >=12 ) {
                __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc0);
                MlasQgemmStoreVectorMMA<0>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 0);
                __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc1);
                MlasQgemmStoreVectorMMA<4>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 4);
                __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc2);
                MlasQgemmStoreVectorMMA<8>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 8);
                if (CountM >= 8) {
                    C1 = C+ldc*4;
                    __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc4);
                    MlasQgemmStoreVectorMMA<0>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 0);
                    __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc5);
                    MlasQgemmStoreVectorMMA<4>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 4);
                    __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc6);
                    MlasQgemmStoreVectorMMA<8>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 8);
                }
                INC_BUFFER(12);
                if (CountN - 12 > 0) {
                    __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc3);
                    if (CountM >= 8) {
                        __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result1), &acc7);
                    }
                }
                CountN -= 12;
                C += 12;
            } else if (CountN >= 8) {
                __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc0);
                MlasQgemmStoreVectorMMA<0>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 0);
                __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc1);
                MlasQgemmStoreVectorMMA<4>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 4);
                if (CountM >= 8) {
                    C1 = C+ldc*4;
                    __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc4);
                    MlasQgemmStoreVectorMMA<0>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 0);
                    __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc5);
                    MlasQgemmStoreVectorMMA<4>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 4);
                }
                INC_BUFFER(8);
                if (CountN - 8 > 0) {
                    __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc2);
                    if (CountM >= 8) {
                        __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result1), &acc6);
                    }
                }
                CountN -= 8;
                C += 8;
            } else if (CountN >= 4) {
                __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc0);
                MlasQgemmStoreVectorMMA<0>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 0);
                if (CountM >= 8) {
                    C1 = C+ldc*4;
                    __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc4);
                    MlasQgemmStoreVectorMMA<0>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 0);
                    if (CountN - 4 > 0) {
                        __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result1), &acc5);
                    }
                }
                INC_BUFFER(4);
                if (CountN - 4 > 0) {
                     __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc1);
                }
                CountN -= 4;
                C += 4;
            } else {
                __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result), &acc0);
                if (CountM >= 8) {
                    __builtin_mma_disassemble_acc (reinterpret_cast<void*>(result1), &acc4);
                }
            }
            CountN &= 3;
            //
            // Output the remaining partial output block.
            //
            if (CountN > 0) {
                MlasQgemmStoreScalarMMA<0>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB);
                if (CountM >= 8) {
                    MlasQgemmStoreScalarMMA<0>(result1, C + (ldc*4), ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB);
                }
                INC_BUFFER(1);
                if (CountN >= 2) {
                     MlasQgemmStoreScalarMMA<1>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB);
                     if (CountM >= 8) {
                         MlasQgemmStoreScalarMMA<1>(result1, C + (ldc*4), ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB);
                     }
                     INC_BUFFER(1);
                }
                if (CountN >= 3) {
                     MlasQgemmStoreScalarMMA<2>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB);
                     if (CountM >= 8) {
                         MlasQgemmStoreScalarMMA<2>(result1, C + (ldc*4), ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB);
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

const MLAS_GEMM_QUANT_DISPATCH MlasGemm8X8DispatchPOWER10 = {
    MlasGemmQuantOperation<MLAS_GEMM_QUANT_KERNEL_POWER10>,
    MlasGemmQuantPackedOperation<MLAS_GEMM_QUANT_KERNEL_POWER10>,
    MlasGemmQuantCopyPackB<MLAS_GEMM_QUANT_KERNEL_POWER10>,
    MLAS_GEMM_QUANT_KERNEL_POWER10::PackedK,
    MLAS_GEMM_QUANT_KERNEL_POWER10::PackedStrides.K,
};
