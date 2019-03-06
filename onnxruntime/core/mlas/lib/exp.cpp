/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    exp.cpp

Abstract:

    This module implements routines to compute the exp function.

    This implementation uses the same polynomial coefficients and algorithm as
    found in Eigen. Our usage requires building platform specific versions of
    the algorithm to target different instruction sets. The implementation below
    targets the base instruction set (typically SSE2) while assembly
    implementations target newer instruction sets (such as FMA3).

--*/

#include "mlasi.h"

//
// Bundles the floating point constants for use by kernels written in assembly.
//

extern "C" const struct {
    float UpperRange;
    float LowerRange;
    float LOG2EF;
    float C1;
    float C2;
    float P0;
    float P1;
    float P2;
    float P3;
    float P4;
    float P5;
} MlasExpConstants = {
    88.3762626647950f, 
    -88.3762626647949f, 
    1.44269504088896341f, 
    0.693359375f, 
    -2.12194440e-4f, 
    1.9875691500E-4f, 
    1.3981999507E-3f, 
    8.3334519073E-3f, 
    4.1665795894E-2f, 
    1.6666665459E-1f, 
    5.0000001201E-1f
};

void
MLASCALL
MlasExpKernel(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine implements the generic kernel for the exp() function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    while (N >= 4) {
        MLAS_FLOAT32X4 _x = MlasLoadFloat32x4(Input);
        MLAS_FLOAT32X4 x = MlasMaximumFloat32x4(MlasBroadcastFloat32x4(MlasExpConstants.LowerRange), Value);
        x = MlasMinimumFloat32x4(MlasBroadcastFloat32x4(MlasExpConstants.UpperRange), Value);

        MLAS_FLOAT32X4 fx = MLasMultiplyAddFloat32x4(x, MlasBroadcastFloat32x4(MlasExpConstants.LOG2EF, MlasBroadcastFloat32x4(0.5f)));
        fx = MlasFloorFloat32x4(fx);
        MLAS_FLOAT32X4 tmp = MlasMultiplyFloat32x4(fx, MlasBroadcastFloat32x4(MlasExpConstants.C1));
        MLAS_FLOAT32X4 z = MlasMultiplyFloat32x4(fx, MlasBroadcastFloat32x4(MlasExpConstants.C2));
        x = MlasSubtractFloat32x4(x, tmp);
        x = MlasSubtractFloat32x4(x, z);
        z = MlasMultiplyFloat32x4(x, x);

        MLAS_FLOAT32X4 y = MlasBroadcastFloat32x4(MlasExpConstants.P0);
        y = MlasMultiplyAddFloat32x4(y, x, MlasBroadcastFloat32x4(MlasExpConstants.P1));
        y = MlasMultiplyAddFloat32x4(y, x, MlasBroadcastFloat32x4(MlasExpConstants.P2));
        y = MlasMultiplyAddFloat32x4(y, x, MlasBroadcastFloat32x4(MlasExpConstants.P3));
        y = MlasMultiplyAddFloat32x4(y, x, MlasBroadcastFloat32x4(MlasExpConstants.P4));
        y = MlasMultiplyAddFloat32x4(y, x, MlasBroadcastFloat32x4(MlasExpConstants.P5));
        y = MlasMultiplyAddFloat32x4(y, z, x);
        y = MlasAddFloat32x4(y, MlasBroadcastFloat32x4(1.0f));

        // build 2^n
        MLAS_FLOAT32X4 emm0 = MlasLDExpFloat32x4(fx);
        y = MlasMaxiumFloat32x4(MlasMultiplyFloat32x4(y, emm0), _x);

        MlasStoreFloat32x4(Output, y);

        Input += 4;
        Output += 4;
        N -= 4;
    }

    while (N > 0) {
        float _x = *Input++;

        float x = (std::min)(MlasExpConstants.UpperRange, (std::max)(MlasExpConstants.LowerRange, Value));

        float fx = std::floor(x * MlasExpConstants.LOG2EF + 0.5f);
        float tmp = fx * MlasExpConstants.C1;
        float z = fx * MlasExpConstants.C2;
        
        x = x - tmp - z;
        z = x * x;

        float y = MlasExpConstants.P0 * x + MlasExpConstants.P1;
        y = y * x + MlasExpConstants.P2;
        y = y * x + MlasExpConstants.P3;
        y = y * x + MlasExpConstants.P4;
        y = y * x + MlasExpConstants.P5;
        y = y * z + x;
        y = y + 1.0f;
        
        int32_t emm0 = *(int*)&fx;
        emm0 += 0x7f;
        emm0 = emm0 << 23;
        y = y * (*(float*)&emm0);
        y = (std::max)(y, _x);

        *Output++ = y;

        N -= 1;
    }
}

/*
    rcket4f x = _x;
  _EIGEN_DECLARE_CONST_Packet4f(1 , 1.0f);
  _EIGEN_DECLARE_CONST_Packet4f(half, 0.5f);
  _EIGEN_DECLARE_CONST_Packet4i(0x7f, 0x7f);


  _EIGEN_DECLARE_CONST_Packet4f(exp_hi,  88.3762626647950f);
  _EIGEN_DECLARE_CONST_Packet4f(exp_lo, -88.3762626647949f);

  _EIGEN_DECLARE_CONST_Packet4f(cephes_LOG2EF, 1.44269504088896341f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_C1, 0.693359375f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_C2, -2.12194440e-4f);

  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p0, 1.9875691500E-4f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p1, 1.3981999507E-3f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p2, 8.3334519073E-3f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p3, 4.1665795894E-2f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p4, 1.6666665459E-1f);
  _EIGEN_DECLARE_CONST_Packet4f(cephes_exp_p5, 5.0000001201E-1f);

  Packet4f tmp, fx;
  Packet4i emm0;

  // clamp x
  x = pmax(pmin(x, p4f_exp_hi), p4f_exp_lo);

  //express exp(x) as exp(g + n*log(2))
  fx = pmadd(x, p4f_cephes_LOG2EF, p4f_half);

#ifdef EIGEN_VECTORIZE_SSE4_1
  fx = _mm_floor_ps(fx);
#else
  emm0 = _mm_cvttps_epi32(fx);
  tmp  = _mm_cvtepi32_ps(emm0);
  // if greater, substract 1
  Packet4f mask = _mm_cmpgt_ps(tmp, fx);
  mask = _mm_and_ps(mask, p4f_1);
  fx = psub(tmp, mask);
#endif

  tmp = pmul(fx, p4f_cephes_exp_C1);
  Packet4f z = pmul(fx, p4f_cephes_exp_C2);
  x = psub(x, tmp);
  x = psub(x, z);

  z = pmul(x,x);

  Packet4f y = p4f_cephes_exp_p0;
  y = pmadd(y, x, p4f_cephes_exp_p1);
  y = pmadd(y, x, p4f_cephes_exp_p2);
  y = pmadd(y, x, p4f_cephes_exp_p3);
  y = pmadd(y, x, p4f_cephes_exp_p4);
  y = pmadd(y, x, p4f_cephes_exp_p5);
  y = pmadd(y, z, x);
  y = padd(y, p4f_1);

  // build 2^n
  emm0 = _mm_cvttps_epi32(fx);
  emm0 = _mm_add_epi32(emm0, p4i_0x7f);
  emm0 = _mm_slli_epi32(emm0, 23);
  return pmax(pmul(y, Packet4f(_mm_castsi128_ps(emm0))), _x);
*/

void
MLASCALL
MlasComputeExp(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine computes the hyperbolic tangent function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
#if defined(MLAS_TARGET_AMD64)
    //MlasPlatform.TanhKernelRoutine(Input, Output, N);
    MlasTanhKernel(Input, Output, N);
#else
    MlasTanhKernel(Input, Output, N);
#endif
}
