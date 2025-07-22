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

template<typename Vtype, bool AIsSigned>
void
MlasGemmQuantCopyPackA8x8(
    MLAS_GEMM_QUANT_KERNEL_POWER10::PackedAType* D,
    const uint8_t* A,
    size_t lda,
    size_t CountM,
    size_t CountK,
    int32_t* RowSumBuffer
    )
{
    constexpr uint8_t Flip = (AIsSigned ? 0 : 0x80);
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
               reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a1),
                           reinterpret_cast<__vector int>(a2)));
            Vtype vx1 =
               reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a3),
                           reinterpret_cast<__vector int>(a4)));
            Vtype vx2 =
               reinterpret_cast<Vtype>(vec_mergeo(reinterpret_cast<__vector int>(a1),
                           reinterpret_cast<__vector int>(a2)));
            Vtype vx3 =
               reinterpret_cast<Vtype>(vec_mergeo(reinterpret_cast<__vector int>(a3),
                           reinterpret_cast<__vector int>(a4)));
            Vtype vx4 = vec_xxpermdi(vx, vx1, 0);
            Vtype vx5 = vec_xxpermdi(vx2, vx3, 0);
            Vtype vx6 = vec_xxpermdi(vx, vx1, 3);
            Vtype vx7 = vec_xxpermdi(vx2, vx3, 3);
            a1 = *reinterpret_cast<const Vtype *>(&a[lda*4]);
            a2 = *reinterpret_cast<const Vtype *>(&a[lda*5]);
            a3 = *reinterpret_cast<const Vtype *>(&a[lda*6]);
            a4 = *reinterpret_cast<const Vtype *>(&a[lda*7]);
            vx =
              reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2)));
            vx1 =
              reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4)));
            vx2 =
              reinterpret_cast<Vtype>(vec_mergeo(reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2)));
            vx3 =
              reinterpret_cast<Vtype>(vec_mergeo(reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4)));
            Vtype vx8 = vec_xxpermdi(vx, vx1, 0);
            Vtype vx9 = vec_xxpermdi(vx2, vx3, 0);
            Vtype vx10 = vec_xxpermdi(vx, vx1, 3);
            Vtype vx11 = vec_xxpermdi(vx2, vx3, 3);
            vec_t vxx =
              AIsSigned ? reinterpret_cast<vec_t>(vx4) :
                          reinterpret_cast<vec_t>(vec_sub(vx4, vmask));
            vsum = vec_sum4s(vxx, vsum);
            *reinterpret_cast<vec_t *>(&D[0]) = vxx;
            vxx = AIsSigned ? reinterpret_cast<vec_t>(vx5) :
                              reinterpret_cast<vec_t>(vec_sub(vx5, vmask));
            vsum = vec_sum4s(vxx, vsum);
            *reinterpret_cast<vec_t *>(&D[16]) = vxx;
            vxx = AIsSigned ? reinterpret_cast<vec_t>(vx6) :
                              reinterpret_cast<vec_t>(vec_sub(vx6, vmask));
            vsum = vec_sum4s(vxx, vsum);
            *reinterpret_cast<vec_t *>(&D[32]) = vxx;
            vxx = AIsSigned ? reinterpret_cast<vec_t>(vx7) :
                              reinterpret_cast<vec_t>(vec_sub(vx7, vmask));
            vsum = vec_sum4s(vxx, vsum);
            *reinterpret_cast<vec_t *>(&D[48]) = vxx;
            vxx = AIsSigned ? reinterpret_cast<vec_t>(vx8) :
                              reinterpret_cast<vec_t>(vec_sub(vx8, vmask));
            *reinterpret_cast<vec_t *>(&D[64]) = vxx;
            vsum2 = vec_sum4s(vxx, vsum2);
            vxx = AIsSigned ? reinterpret_cast<vec_t>(vx9) :
                              reinterpret_cast<vec_t>(vec_sub(vx9, vmask));
            *reinterpret_cast<vec_t *>(&D[80]) = vxx;
            vsum2 = vec_sum4s(vxx, vsum2);
            vxx = AIsSigned ? reinterpret_cast<vec_t>(vx10) :
                              reinterpret_cast<vec_t>(vec_sub(vx10, vmask));
            *reinterpret_cast<vec_t *>(&D[96]) = vxx;
            vsum2 = vec_sum4s(vxx, vsum2);
            vxx = AIsSigned ? reinterpret_cast<vec_t>(vx11) :
                              reinterpret_cast<vec_t>(vec_sub(vx11, vmask));
            *reinterpret_cast<vec_t *>(&D[112]) = vxx;
            vsum2 = vec_sum4s(vxx, vsum2);
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
              AIsSigned ? reinterpret_cast<vec_t>(vx1) :
                          reinterpret_cast<vec_t>(vec_sub(reinterpret_cast<Vtype>(vx1), vmask));
            vsum = vec_sum4s(vx, vsum);
            *reinterpret_cast<vec_t *>(&D[0]) = vx;
            a1 = *reinterpret_cast<const int *>(&a[lda*4]);
            a2 = *reinterpret_cast<const int *>(&a[lda*5]);
            a3 = *reinterpret_cast<const int *>(&a[lda*6]);
            a4 = *reinterpret_cast<const int *>(&a[lda*7]);
            __vector int vx2 = { a1, a2, a3, a4};
            vx = AIsSigned ? reinterpret_cast<vec_t>(vx2) :
                             reinterpret_cast<vec_t>(vec_sub(reinterpret_cast<Vtype>(vx2), vmask));
            vsum2 = vec_sum4s(vx, vsum2);
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
            Vtype vx =
              reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2)));
            Vtype vx1 =
              reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4)));
            Vtype vx2 = vec_xxpermdi(vx, vx1, 0);
            vec_t vx3 =
              AIsSigned ? reinterpret_cast<vec_t>(vx2) :
                          reinterpret_cast<vec_t>(vec_sub(vx2, vmask));
            vsum = vec_sum4s(vx3, vsum);
            *reinterpret_cast<vec_t *>(&D[0]) = vx3;
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
              reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2)));
            vx1 =
              reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4)));
            vx2 = vec_xxpermdi(vx, vx1, 0);
            vx3 = AIsSigned ? reinterpret_cast<vec_t>(vx2) :
                              reinterpret_cast<vec_t>(vec_sub(vx2, vmask));
            vsum2 = vec_sum4s(vx3, vsum2);
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
              reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2)));
            Vtype vx1 =
              reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4)));
            Vtype vx2 =
              reinterpret_cast<Vtype>(vec_mergeo(reinterpret_cast<__vector int>(a1),
                               reinterpret_cast<__vector int>(a2)));
            Vtype vx3 =
              reinterpret_cast<Vtype>(vec_mergeo(reinterpret_cast<__vector int>(a3),
                               reinterpret_cast<__vector int>(a4)));
            Vtype vx4 = vec_xxpermdi(vx, vx1, 0);
            Vtype vx5 = vec_xxpermdi(vx2, vx3, 0);
            Vtype vx6 = vec_xxpermdi(vx, vx1, 3);
            Vtype vx7 = vec_xxpermdi(vx2, vx3, 3);
            vec_t vx0 =
              AIsSigned ? reinterpret_cast<vec_t>(vx4) :
                          reinterpret_cast<vec_t>(vec_sub(vx4, vmask));
            *reinterpret_cast<vec_t *>(&D[0]) = vx0;
            vsum = vec_sum4s(vx0, vsum);
            vx0 = AIsSigned ? reinterpret_cast<vec_t>(vx5) :
                              reinterpret_cast<vec_t>(vec_sub(vx5, vmask));
            *reinterpret_cast<vec_t *>(&D[16]) = vx0;
            vsum = vec_sum4s(vx0, vsum);
            vx0 = AIsSigned ? reinterpret_cast<vec_t>(vx6) :
                              reinterpret_cast<vec_t>(vec_sub(vx6, vmask));
            *reinterpret_cast<vec_t *>(&D[32]) = vx0;
            vsum = vec_sum4s(vx0, vsum);
            vx0 = AIsSigned ? reinterpret_cast<vec_t>(vx7) :
                              reinterpret_cast<vec_t>(vec_sub(vx7, vmask));
            *reinterpret_cast<vec_t *>(&D[48]) = vx0;
            vsum = vec_sum4s(vx0, vsum);
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
              AIsSigned ? reinterpret_cast<vec_t>(vx1) :
                          reinterpret_cast<vec_t>(vec_sub(reinterpret_cast<Vtype>(vx1), vmask));
            *reinterpret_cast<vec_t *>(&D[0]) = vx;
            vsum = vec_sum4s(vx, vsum);
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
            vec_t vx1 =
               AIsSigned ? reinterpret_cast<vec_t>(vx) :
                           reinterpret_cast<vec_t>(vec_sub(vx, vmask));
            *reinterpret_cast<vec_t *>(&D[0]) = vx1;
            vsum = vec_sum4s(vx1, vsum);
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
            Vtype a4 = vmask;
            Vtype a2 = vmask;
            Vtype a3 = vmask;
            Vtype a1 = *reinterpret_cast<const Vtype *>(&a[0]);
	    if (CountM == 1) {
		    vec_t va1 = AIsSigned ? reinterpret_cast<vec_t>(a1) :
			    reinterpret_cast<vec_t>(vec_sub(a1, vmask));
		    *reinterpret_cast<vec_t *>(&D[0])= (vec_t)va1;
		    vsum = vec_sum4s(va1,vsum);
	    }else {
		    a2 = *reinterpret_cast<const Vtype *>(&a[lda]);
		    if (CountM == 3)
			    a3 = *reinterpret_cast<const Vtype *>(&a[lda * 2]);
		    Vtype vx =
			    reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a1),
						    reinterpret_cast<__vector int>(a2)));
		    Vtype vx1 =
			    reinterpret_cast<Vtype>(vec_mergee(reinterpret_cast<__vector int>(a3),
						    reinterpret_cast<__vector int>(a4)));
		    Vtype vx2 =
			    reinterpret_cast<Vtype>(vec_mergeo(reinterpret_cast<__vector int>(a1),
						    reinterpret_cast<__vector int>(a2)));
		    Vtype vx3 =
			    reinterpret_cast<Vtype>(vec_mergeo(reinterpret_cast<__vector int>(a3),
						    reinterpret_cast<__vector int>(a4)));
		    Vtype vx4 = vec_xxpermdi(vx, vx1, 0);
		    Vtype vx5 = vec_xxpermdi(vx2, vx3, 0);
		    Vtype vx6 = vec_xxpermdi(vx, vx1, 3);
		    Vtype vx7 = vec_xxpermdi(vx2, vx3, 3);
		    vec_t vx0 = AIsSigned ? reinterpret_cast<vec_t>(vx4) :
			    reinterpret_cast<vec_t>(vec_sub(vx4, vmask));
		    *reinterpret_cast<vec_t *>(&D[0]) = vx0;
		    vsum = vec_sum4s(vx0, vsum);
		    vx0 = AIsSigned ? reinterpret_cast<vec_t>(vx5) :
			    reinterpret_cast<vec_t>(vec_sub(vx5, vmask));
		    *reinterpret_cast<vec_t *>(&D[16]) = vx0;
		    vsum = vec_sum4s(vx0, vsum);
		    vx0 = AIsSigned ? reinterpret_cast<vec_t>(vx6) :
			    reinterpret_cast<vec_t>(vec_sub(vx6, vmask));
		    *reinterpret_cast<vec_t *>(&D[32]) = vx0;
		    vsum = vec_sum4s(vx0, vsum);
		    vx0 = AIsSigned ? reinterpret_cast<vec_t>(vx7) :
			    reinterpret_cast<vec_t>(vec_sub(vx7, vmask));
		    *reinterpret_cast<vec_t *>(&D[48]) = vx0;
		    vsum = vec_sum4s(vx0, vsum);
	    }
            if (CountM == 1)
		    D += 16;
            else
		    D += 16 * 4;
            a += 16;
            y -= 16;
        }
        if (CountM == 1)
		vsum[0] += (vsum[1] + vsum[2] + vsum[3]);
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
            vec_t vx =
              AIsSigned ? reinterpret_cast<vec_t>(vx1) :
                          reinterpret_cast<vec_t>(vec_sub(reinterpret_cast<Vtype>(vx1), vmask));
            *reinterpret_cast<vec_t *>(&D[0]) = vx;
            vsum = vec_sum4s(vx, vsum);
	    if (CountM == 1)
		    D += 4;
	    else
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
            vsum = vec_sum4s(vx, vsum);
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
    MLAS_GEMM_QUANT_KERNEL_POWER10::PackedBType* D,
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
                vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b1, vmask)) :
                                        reinterpret_cast<vec_t>(b1);
                vec_t vx2 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b2, vmask)) :
                                        reinterpret_cast<vec_t>(b2);
                vec_t vx3 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b3, vmask)) :
                                        reinterpret_cast<vec_t>(b3);
                vec_t vx4 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b4, vmask)) :
                                        reinterpret_cast<vec_t>(b4);
                *reinterpret_cast<vec_t *>(&D[0]) = vx1;
                *reinterpret_cast<vec_t *>(&D[k2]) = vx2;
                *reinterpret_cast<vec_t *>(&D[k3]) = vx3;
                *reinterpret_cast<vec_t *>(&D[k4]) = vx4;
                vsum = vec_sum4s(vx1, vsum);
                vsum2 = vec_sum4s(vx2, vsum2);
                vsum3 = vec_sum4s(vx3, vsum3);
                vsum4 = vec_sum4s(vx4, vsum4);
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
            vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b1, vmask)) :
                                    reinterpret_cast<vec_t>(b1);
            vec_t vx2 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b2, vmask)) :
                                    reinterpret_cast<vec_t>(b2);
            vec_t vx3 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b3, vmask)) :
                                    reinterpret_cast<vec_t>(b3);
            vec_t vx4 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(b4, vmask)) :
                                    reinterpret_cast<vec_t>(b4);
            *reinterpret_cast<vec_t *>(&D[0]) = vx1;
            *reinterpret_cast<vec_t *>(&D[k2]) = vx2;
            *reinterpret_cast<vec_t *>(&D[k3]) = vx3;
            *reinterpret_cast<vec_t *>(&D[k4]) = vx4;
            vsum = vec_sum4s(vx1, vsum);
            vsum2 = vec_sum4s(vx2, vsum2);
            vsum3 = vec_sum4s(vx3, vsum3);
            vsum4 = vec_sum4s(vx4, vsum4);
            D += 16;
        }
        *ColumnSumBuffer++ = vsum[0];
        *ColumnSumBuffer++ = vsum[1];
        *ColumnSumBuffer++ = vsum[2];
        *ColumnSumBuffer++ = vsum[3];
        *ColumnSumBuffer++ = vsum2[0];
        *ColumnSumBuffer++ = vsum2[1];
        *ColumnSumBuffer++ = vsum2[2];
        *ColumnSumBuffer++ = vsum2[3];
        *ColumnSumBuffer++ = vsum3[0];
        *ColumnSumBuffer++ = vsum3[1];
        *ColumnSumBuffer++ = vsum3[2];
        *ColumnSumBuffer++ = vsum3[3];
        *ColumnSumBuffer++ = vsum4[0];
        *ColumnSumBuffer++ = vsum4[1];
        *ColumnSumBuffer++ = vsum4[2];
        *ColumnSumBuffer++ = vsum4[3];
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
                vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(vx, vmask)) :
                                        reinterpret_cast<vec_t>(vx);
                *reinterpret_cast<vec_t *>(&D[0]) = vx1;
                vsum = vec_sum4s(vx1, vsum);
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
            vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(vx, vmask)) :
                                    reinterpret_cast<vec_t>(vx);
            *reinterpret_cast<vec_t *>(&D[0]) = vx1;
            vsum = vec_sum4s(vx1, vsum);
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
                vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(vx, vmask)) :
                                        reinterpret_cast<vec_t>(vx);
                *reinterpret_cast<vec_t *>(&D[0]) = vx1;
                vsum = vec_sum4s(vx1, vsum);
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
            vec_t vx1 = BIsSigned ? reinterpret_cast<vec_t>(vec_add(vx, vmask)) :
                                    reinterpret_cast<vec_t>(vx);
            *reinterpret_cast<vec_t *>(&D[0]) = vx1;
            vsum = vec_sum4s(vx1, vsum);
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
        MlasGemmQuantCopyPackA8x8<__vector signed char, true>(D, A, lda, CountM, CountK, RowSumBuffer);
    } else {
        MlasGemmQuantCopyPackA8x8<__vector unsigned char, false>(D, A, lda, CountM, CountK, RowSumBuffer);
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
        MlasGemmQuantCopyPackB8x8<__vector signed char, true>(D, B, ldb, CountN, CountK, ColumnSumBuffer);
    } else {
        MlasGemmQuantCopyPackB8x8< __vector unsigned char, false>(D, B, ldb, CountN, CountK, ColumnSumBuffer);
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
    size_t RowCount;
    __vector signed int vsum0, vsum1, vsum2, vsum3;
#if defined(_AIX) && defined(__clang__)
    __vector signed int columnsum = *reinterpret_cast<const __vector int *>(&ColumnSumBuffer[pos]);
#else
    __vector signed int columnsum = *reinterpret_cast<const __vector int32_t *>(&ColumnSumBuffer[pos]);
#endif
    C += VectorCount;
    if (ZeroPointB != nullptr) {
#if defined(_AIX) && defined(__clang__)
        __vector signed int zeropoint = *reinterpret_cast<const __vector int *>(&ZeroPointB[pos]);
#else
        __vector signed int zeropoint = *reinterpret_cast<const __vector int32_t *>(&ZeroPointB[pos]);
#endif
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
        __builtin_mma_xvi8ger4pp(acc0, va[0], vb[0]);
        __builtin_mma_xvi8ger4pp(acc0, va[1], vb[1]);
        __builtin_mma_xvi8ger4pp(acc0, va[2], vb[2]);
        __builtin_mma_xvi8ger4pp(acc0, va[3], vb[3]);
        if (CountM) {
            __builtin_mma_xvi8ger4pp(acc1, va[4], vb[0]);
            __builtin_mma_xvi8ger4pp(acc1, va[5], vb[1]);
            __builtin_mma_xvi8ger4pp(acc1, va[6], vb[2]);
            __builtin_mma_xvi8ger4pp(acc1, va[7], vb[3]);
        }
    } else if (CountK == 12) {
        __builtin_mma_xvi8ger4pp(acc0, va[0], vb[0]);
        __builtin_mma_xvi8ger4pp(acc0, va[1], vb[1]);
        __builtin_mma_xvi8ger4pp(acc0, va[2], vb[2]);
        if (CountM) {
            __builtin_mma_xvi8ger4pp(acc1, va[3], vb[0]);
            __builtin_mma_xvi8ger4pp(acc1, va[4], vb[1]);
            __builtin_mma_xvi8ger4pp(acc1, va[5], vb[2]);
        }
    } else if (CountK == 8) {
        __builtin_mma_xvi8ger4pp(acc0, va[0], vb[0]);
        __builtin_mma_xvi8ger4pp(acc0, va[1], vb[1]);
        if (CountM) {
            __builtin_mma_xvi8ger4pp(acc1, va[2], vb[0]);
            __builtin_mma_xvi8ger4pp(acc1, va[3], vb[1]);
        }
    } else {
        __builtin_mma_xvi8ger4pp(acc0, va[0], vb[0]);
        if (CountM) {
            __builtin_mma_xvi8ger4pp(acc1, va[1], vb[0]);
        }
    }

};

MLAS_FORCEINLINE
void
MlasGemmQuantKernel_M1(
    const MLAS_GEMM_QUANT_KERNEL_POWER10::PackedAType* A,
    const MLAS_GEMM_QUANT_KERNEL_POWER10::PackedBType* B,
    int32_t* C,
    size_t PackedCountK,
    size_t CountN,
    size_t ldc,
    const int32_t* RowSumBuffer,
    const int32_t* ColumnSumBuffer,
    const int32_t* ZeroPointB,
    bool ZeroMode
    )
{
    size_t Mval = 1;
    while (CountN > 0) {
	    const int8_t *a = A;
	    typedef __vector unsigned char vec_t;
	    typedef __vector signed char svec_t;
	    const uint8_t *b = B;
	    MLAS_INT32X4 result = {0};
	    __vector signed int  VecC = {0,0, 0, 0};
	    __vector signed int  VecC2 = {0,0, 0, 0};
	    __vector signed int  VecC3 = {0,0, 0, 0};
	    __vector signed int  VecC4 = {0,0, 0, 0};
	    size_t k = PackedCountK * MLAS_GEMM_QUANT_KERNEL_POWER10::PackedK;
	    size_t k1 = PackedCountK;
	    __vector unsigned char va[4];
	    __vector unsigned char pat = {0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3};
	    __vector unsigned char pat2 = {4,5,6,7,4,5,6,7,4,5,6,7,4,5,6,7};
	    __vector unsigned char pat3 = {8,9,10,11,8,9,10,11,8,9,10,11,8,9,10,11};
	    __vector unsigned char pat4 = {12,13,14,15,12,13,14,15,12,13,14,15,12,13,14,15};
	    while (k >= 16) {
		    vec_t *vecA = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(a));
		    vec_t *vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(b));
		    va[0] = vec_perm(vecA[0],vecA[0],pat);
		    va[1] = vec_perm(vecA[0],vecA[0],pat2);
		    va[2] = vec_perm(vecA[0],vecA[0],pat3);
		    va[3] = vec_perm(vecA[0],vecA[0],pat4);
		    VecC = vec_msum((svec_t)va[0], (vec_t)vb[0], VecC);
		    VecC = vec_msum((svec_t)va[1], (vec_t)vb[1], VecC);
		    VecC = vec_msum((svec_t)va[2], (vec_t)vb[2], VecC);
		    VecC = vec_msum((svec_t)va[3], (vec_t)vb[3], VecC);
		    vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*16]));
		    VecC2 = vec_msum((svec_t)va[0], (vec_t)vb[0], VecC2);
		    VecC2 = vec_msum((svec_t)va[1], (vec_t)vb[1], VecC2);
		    VecC2 = vec_msum((svec_t)va[2], (vec_t)vb[2], VecC2);
		    VecC2 = vec_msum((svec_t)va[3], (vec_t)vb[3], VecC2);
		    vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*32]));
		    VecC3 = vec_msum((svec_t)va[0], (vec_t)vb[0], VecC3);
		    VecC3 = vec_msum((svec_t)va[1], (vec_t)vb[1], VecC3);
		    VecC3 = vec_msum((svec_t)va[2], (vec_t)vb[2], VecC3);
		    VecC3 = vec_msum((svec_t)va[3], (vec_t)vb[3], VecC3);
		    vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*48]));
		    VecC4 = vec_msum((svec_t)va[0], (vec_t)vb[0], VecC4);
		    VecC4 = vec_msum((svec_t)va[1], (vec_t)vb[1], VecC4);
		    VecC4 = vec_msum((svec_t)va[2], (vec_t)vb[2], VecC4);
		    VecC4 = vec_msum((svec_t)va[3], (vec_t)vb[3], VecC4);
		    b += 64;
		    a += 16;
		    k -= 16;
	    }
	    if (k >= 12) {
		    vec_t *vecA = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(a));
		    vec_t *vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(b));
		    va[0]=vec_perm(vecA[0],vecA[0],pat);
		    va[1] = vec_perm(vecA[0],vecA[0],pat2);
		    va[2] = vec_perm(vecA[0],vecA[0],pat3);
		    VecC = vec_msum((svec_t)va[0], (vec_t)vb[0], VecC);
		    VecC = vec_msum((svec_t)va[1], (vec_t)vb[1], VecC);
		    VecC = vec_msum((svec_t)va[2], (vec_t)vb[2], VecC);
		    vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*16]));
		    VecC2 = vec_msum((svec_t)va[0], (vec_t)vb[0], VecC2);
		    VecC2 = vec_msum((svec_t)va[1], (vec_t)vb[1], VecC2);
		    VecC2 = vec_msum((svec_t)va[2], (vec_t)vb[2], VecC2);
		    vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*32]));
		    VecC3 = vec_msum((svec_t)va[0], (vec_t)vb[0], VecC3);
		    VecC3 = vec_msum((svec_t)va[1], (vec_t)vb[1], VecC3);
		    VecC3 = vec_msum((svec_t)va[2], (vec_t)vb[2], VecC3);
		    vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*48]));
		    VecC4 = vec_msum((svec_t)va[0], (vec_t)vb[0], VecC4);
		    VecC4 = vec_msum((svec_t)va[1], (vec_t)vb[1], VecC4);
		    VecC4 = vec_msum((svec_t)va[2], (vec_t)vb[2], VecC4);
		    a += 12;
		    b += 48;
		    k -= 12;
	    }
            if (k >= 8) {
		    vec_t *vecA = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(a));
		    vec_t *vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(b));
		    va[0]=vec_perm(vecA[0],vecA[0],pat);
		    va[1] = vec_perm(vecA[0],vecA[0],pat2);
		    VecC = vec_msum((svec_t)va[0], (vec_t)vb[0], VecC);
		    VecC = vec_msum((svec_t)va[1], (vec_t)vb[1], VecC);
		    vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*16]));
		    VecC2 = vec_msum((svec_t)va[0], (vec_t)vb[0], VecC2);
		    VecC2 = vec_msum((svec_t)va[1], (vec_t)vb[1], VecC2);
		    vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*32]));
		    VecC3 = vec_msum((svec_t)va[0], (vec_t)vb[0], VecC3);
		    VecC3 = vec_msum((svec_t)va[1], (vec_t)vb[1], VecC3);
		    vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*48]));
		    VecC4 = vec_msum((svec_t)va[0], (vec_t)vb[0], VecC4);
		    VecC4 = vec_msum((svec_t)va[1], (vec_t)vb[1], VecC4);
		    a += 8;
		    b += 32;
		    k -= 8;
	    }
	    if (k >= 4) {
		    vec_t *vecA = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(a));
		    vec_t *vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(b));
		    va[0] = vec_perm(vecA[0],vecA[0],pat);
		    VecC = vec_msum((svec_t)va[0], (vec_t)vb[0], VecC);
		    vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*16]));
		    VecC2 = vec_msum((svec_t)va[0], (vec_t)vb[0], VecC2);
		    vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*32]));
		    VecC3 = vec_msum((svec_t)va[0], (vec_t)vb[0], VecC3);
		    vb = const_cast<vec_t *>(reinterpret_cast<const vec_t *>(&b[k1*48]));
		    VecC4 = vec_msum((svec_t)va[0], (vec_t)vb[0], VecC4);
		    a += 4;
		    b += 16;
		    k -= 4;
	    }
	    if (CountN >=16) {
		    MlasQgemmStoreVectorMMA<0>(&VecC, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 0);
		    MlasQgemmStoreVectorMMA<4>(&VecC2, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 4);
		    MlasQgemmStoreVectorMMA<8>(&VecC3, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 8);
		    MlasQgemmStoreVectorMMA<12>(&VecC4, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 12);
		    INC_BUFFER(16);
		    CountN -= 16;
		    B += 16 * 4 *PackedCountK;
		    C += 16;
	    }else {
		    if (CountN >=12) {
			    MlasQgemmStoreVectorMMA<0>(&VecC, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 0);
			    MlasQgemmStoreVectorMMA<4>(&VecC2, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 4);
			    MlasQgemmStoreVectorMMA<8>(&VecC3, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 8);
			    INC_BUFFER(12);
			    if (CountN - 12 > 0)
				    result = VecC4;
			    CountN -= 12;
			    C += 12;
		    } else if (CountN >=8) {
			    MlasQgemmStoreVectorMMA<0>(&VecC, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 0);
			    MlasQgemmStoreVectorMMA<4>(&VecC2, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 4);
			    INC_BUFFER(8);
			    if (CountN - 8 > 0)
				    result=VecC3;
			    CountN -= 8;
			    C += 8;
		    }else if (CountN >=4) {
			    MlasQgemmStoreVectorMMA<0>(&VecC, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 0);
			    INC_BUFFER(4);
			    if (CountN - 4 > 0)
				    result=VecC2;
			    CountN -= 4;
			    C += 4;
		    }else
			    result=VecC;
		    CountN &= 3;

		    // Output the remaining partial output block.
		    if (CountN > 0) {
			    MlasQgemmStoreScalarMMA<0>(&result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB);
			    INC_BUFFER(1);
		    }
		    if (CountN >= 2) {
			    MlasQgemmStoreScalarMMA<1>(&result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB);
			    INC_BUFFER(1);
		    }
		    if (CountN >= 3) {
			    MlasQgemmStoreScalarMMA<2>(&result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB);
			    INC_BUFFER(1);
		    }
		    CountN=0;
	    }
    }
}

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
    if (CountM == 1) {
	    MlasGemmQuantKernel_M1(A,B,C,PackedCountK,CountN,ldc,RowSumBuffer,ColumnSumBuffer,ZeroPointB,ZeroMode);
	    return 1;
    }
    else {
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
            __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc0);
            MlasQgemmStoreVectorMMA<0>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 0);
            __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc1);
            MlasQgemmStoreVectorMMA<4>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 4);
            __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc2);
            MlasQgemmStoreVectorMMA<8>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 8);
            __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc3);
            MlasQgemmStoreVectorMMA<12>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 12);
            if (CountM >= 8) {
                C1 = C+ldc*4;
                __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc4);
                MlasQgemmStoreVectorMMA<0>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 0);
                __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc5);
                MlasQgemmStoreVectorMMA<4>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 4);
                __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc6);
                MlasQgemmStoreVectorMMA<8>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 8);
                __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc7);
                MlasQgemmStoreVectorMMA<12>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 12);
            }
            INC_BUFFER(16);
            CountN -= 16;
            B += 16 * 4 *PackedCountK;
            C += 16;
        } else {
            if (CountN >=12 ) {
                __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc0);
                MlasQgemmStoreVectorMMA<0>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 0);
                __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc1);
                MlasQgemmStoreVectorMMA<4>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 4);
                __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc2);
                MlasQgemmStoreVectorMMA<8>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 8);
                if (CountM >= 8) {
                    C1 = C+ldc*4;
                    __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc4);
                    MlasQgemmStoreVectorMMA<0>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 0);
                    __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc5);
                    MlasQgemmStoreVectorMMA<4>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 4);
                    __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc6);
                    MlasQgemmStoreVectorMMA<8>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 8);
                }
                INC_BUFFER(12);
                if (CountN - 12 > 0) {
                    __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc3);
                    if (CountM >= 8) {
                        __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result1), &acc7);
                    }
                }
                CountN -= 12;
                C += 12;
            } else if (CountN >= 8) {
                __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc0);
                MlasQgemmStoreVectorMMA<0>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 0);
                __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc1);
                MlasQgemmStoreVectorMMA<4>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 4);
                if (CountM >= 8) {
                    C1 = C+ldc*4;
                    __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc4);
                    MlasQgemmStoreVectorMMA<0>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 0);
                    __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc5);
                    MlasQgemmStoreVectorMMA<4>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 4);
                }
                INC_BUFFER(8);
                if (CountN - 8 > 0) {
                    __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc2);
                    if (CountM >= 8) {
                        __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result1), &acc6);
                    }
                }
                CountN -= 8;
                C += 8;
            } else if (CountN >= 4) {
                __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc0);
                MlasQgemmStoreVectorMMA<0>(result, C, ldc, Mval, ZeroMode, RowSumBuffer, ColumnSumBuffer, ZeroPointB, 0);
                if (CountM >= 8) {
                    C1 = C+ldc*4;
                    __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc4);
                    MlasQgemmStoreVectorMMA<0>(result, C1, ldc, 4, ZeroMode, RowSumBuffer+4, ColumnSumBuffer, ZeroPointB, 0);
                    if (CountN - 4 > 0) {
                        __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result1), &acc5);
                    }
                }
                INC_BUFFER(4);
                if (CountN - 4 > 0) {
                     __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc1);
                }
                CountN -= 4;
                C += 4;
            } else {
                __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result), &acc0);
                if (CountM >= 8) {
                    __builtin_mma_disassemble_acc(reinterpret_cast<void*>(result1), &acc4);
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
}

const MLAS_GEMM_QUANT_DISPATCH MlasGemm8X8DispatchPOWER10 = {
    MlasGemmQuantOperation<MLAS_GEMM_QUANT_KERNEL_POWER10>,
    MlasGemmQuantPackedOperation<MLAS_GEMM_QUANT_KERNEL_POWER10>,
    MlasGemmQuantCopyPackB<MLAS_GEMM_QUANT_KERNEL_POWER10>,
    MLAS_GEMM_QUANT_KERNEL_POWER10::PackedK,
    MLAS_GEMM_QUANT_KERNEL_POWER10::PackedStrides.K,
    8 // Kernel M stride
};
