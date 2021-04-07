/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    SgemvKernelWasmSimd.cpp

Abstract:

    This module implements the kernels for the single precision matrix/vector
    multiply operation (SGEMV).

--*/

#include "mlasi.h"

size_t
MLASCALL 
MlasGemvFloatKernel(
    const float* A,
    const float* B,
    float* C,
    size_t CountK,
    size_t CountN,
    size_t ldb,
    bool ZeroMode
    )
/*++

Routine Description:

    This routine is an inner kernel to compute matrix multiplication for a
    set of rows. This handles the special case of M=1.

    The elements in matrix B are not transposed.

Arguments:

    A - Supplies the address of matrix A.

    B - Supplies the address of matrix B.

    C - Supplies the address of matrix C.

    CountK - Supplies the number of columns from matrix A and the number
        of rows from matrix B to iterate over.

    CountN - Supplies the number of columns from matrix B and matrix C to
        iterate over.

    ldb - Supplies the first dimension of matrix B.

    ZeroMode - Supplies true if the output matrix must be zero initialized,
        else false if the output matrix is accumulated into.

Return Value:

    None.

--*/
{
    if (ZeroMode && CountK > 0) {
        float* c = C;
        const float* b = B;
        const MLAS_FLOAT32X4 A0 = MlasBroadcastFloat32x4(A);
        auto N = CountN;
        for (; N >= 4; N -= 4) {
            MlasStoreFloat32x4(c, MlasMultiplyFloat32x4(A0, MlasLoadFloat32x4(b)));
            b += 4;
            c += 4;
        }
        for (; N > 0; N--) {
            c[0] = A[0] * b[0];
            c++;
            b++;
        }
        CountK--;
        B += ldb;
        A++;
    }

    for (; CountK >= 4; CountK -= 4) {
        float* c = C;
        const float* b = B;
        const float* b2 = b + ldb * 2;

        const MLAS_FLOAT32X4 A0 = MlasBroadcastFloat32x4(A);
        const MLAS_FLOAT32X4 A1 = MlasBroadcastFloat32x4(A + 1);
        const MLAS_FLOAT32X4 A2 = MlasBroadcastFloat32x4(A + 2);
        const MLAS_FLOAT32X4 A3 = MlasBroadcastFloat32x4(A + 3);

        auto N = CountN;
        constexpr size_t kWide = 8;
        for(; N >= kWide; N -= kWide) {
            MLAS_FLOAT32X4 vec_c0 = MlasMultiplyAddFloat32x4(A0, MlasLoadFloat32x4(b), MlasLoadFloat32x4(c));
            MLAS_FLOAT32X4 vec_c1 = MlasMultiplyAddFloat32x4(A0, MlasLoadFloat32x4(b + 4), MlasLoadFloat32x4(c + 4));

            vec_c0 = MlasMultiplyAddFloat32x4(A1, MlasLoadFloat32x4(b + ldb), vec_c0);
            vec_c1 = MlasMultiplyAddFloat32x4(A1, MlasLoadFloat32x4(b + ldb + 4), vec_c1);

            vec_c0 = MlasMultiplyAddFloat32x4(A2, MlasLoadFloat32x4(b2), vec_c0);
            vec_c1 = MlasMultiplyAddFloat32x4(A2, MlasLoadFloat32x4(b2 + 4), vec_c1);

            vec_c0 = MlasMultiplyAddFloat32x4(A3, MlasLoadFloat32x4(b2 + ldb), vec_c0);
            vec_c1 = MlasMultiplyAddFloat32x4(A3, MlasLoadFloat32x4(b2 + ldb + 4), vec_c1);

            MlasStoreFloat32x4(c, vec_c0);
            MlasStoreFloat32x4(c + 4, vec_c1);

            b += kWide;
            b2 += kWide;
            c += kWide;
        }

        for (; N >= 4; N -= 4) {
            MLAS_FLOAT32X4 vec_c0 = MlasMultiplyAddFloat32x4(MlasLoadFloat32x4(b), A0, MlasLoadFloat32x4(c));
            vec_c0 = MlasMultiplyAddFloat32x4(MlasLoadFloat32x4(b + ldb), A1, vec_c0);
            vec_c0 = MlasMultiplyAddFloat32x4(MlasLoadFloat32x4(b2), A2, vec_c0);
            vec_c0 = MlasMultiplyAddFloat32x4(MlasLoadFloat32x4(b2 + ldb), A3, vec_c0);
            MlasStoreFloat32x4(c, vec_c0);
            b += 4;
            b2 += 4;
            c += 4;
        }

        for (; N > 0; N--) {
            c[0] += A[0] * b[0] + A[1] * b[ldb] + A[2] * b2[0] + A[3] * b2[ldb];
            b++;
            b2++;
            c++;
        }

        B += 4 * ldb;
        A += 4;
    }

    for (; CountK > 0; CountK--) {
        float* c = C;
        const float* b = B;
        const MLAS_FLOAT32X4 A0 = MlasBroadcastFloat32x4(A);
        auto N = CountN;
        for (; N >= 4; N -= 4) {
            MlasStoreFloat32x4(c, MlasMultiplyAddFloat32x4(MlasLoadFloat32x4(b), A0, MlasLoadFloat32x4(c)));
            b += 4;
            c += 4;
        }
        for (; N > 0; N--) {
            c[0] += A[0] * b[0];
            c++;
            b++;
        }
        B += ldb;
        A++;
    }

    return 0;
}
