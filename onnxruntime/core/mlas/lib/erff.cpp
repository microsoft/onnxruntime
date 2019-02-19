/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    erff.cpp

Abstract:

    This module implements routines to compute the error function.

    This implementation's computation logic is based on s_erff.S in glibc.
    As below:
    ==============================================================
    There are 8 paths:
    1. x = +/-0.0
    Return erff(x) = +/-0.0

    2. 0.0 < |x| < 0.125
    Return erff(x) = x *Pol3(x^2),
    where Pol3(x^2) = C3*x^6 + C2*x^4 + C1*x^2 + C0

    3. 0.125 <= |x| < 4.0
    Return erff(x) = sign(x)*PolD(x)*PolC(|x|) + sign(x)*PolA(|x|),
    where sign(x)*PolD(x) = sign(x)*(|x|^7 + D2*x^6 + D1*|x|^5 + D0*x^4),
            PolC(|x|) = B0*x^4 + C3*|x|^3 + C2*|x|^2 + C1*|x| + C0,
            PolA(|x|) = A3|x|^3 + A2*x^2 + A1*|x| + A0

    Actually range 0.125<=|x|< 4.0 is splitted to 5 subranges.
    For each subrange there is particular set of coefficients.
    Below is the list of subranges:
    3.1 0.125 <= |x| < 0.25
    3.2 0.25 <= |x| < 0.5
    3.3 0.5 <= |x| < 1.0
    3.4 1.0 <= |x| < 2.0
    3.5 2.0 <= |x| < 4.0

    4. 4.0 <= |x| < +INF
    Return erff(x) = sign(x)*(1.0d - 2^(-52))

    5. |x| = INF
    Return erff(x) = sign(x) * 1.0

    6. x = [S,Q]NaN
    Return erff(x) = QNaN

    7. x is positive denormal
    Return erff(x) = C0*x - x^2,
    where C0 = 2.0/sqrt(Pi)

    8. x is negative denormal
    Return erff(x) = C0*x + x^2,
    where C0 = 2.0/sqrt(Pi)
    ==============================================================

    (TODO: port the license after this description.)

    (TODO: Handle denormal and NaN.)
--*/

#include "mlasi.h"

// TODO: endian dependent const -- change to independent.
// Polynomial coefficients for |X| in range:              [0, 0.125)          [0.125, 0.25)       [0.25, 0.5)         [0.5, 1.0)          [1.0, 2.0)          [2.0, 4.0)
MLAS_DECLSPEC_ALIGN(static const uint64_t CC0[8], 64) = { 0x3FF0000000000000, 0xBE4218BB56B49E66, 0x3F90849356383F58, 0x3F85F7D419A13DE3, 0xBF49E07E3584C3AE, 0xBF849855D67E9407};
MLAS_DECLSPEC_ALIGN(static const uint64_t CC1[8], 64) = { 0x0000000000000000, 0x3F7AFB8315DA322B, 0x3F830BD5BA240F09, 0x3F791A13FF66D45A, 0x3F3166621131445C, 0x3F5ECA5FEC01C70C};
MLAS_DECLSPEC_ALIGN(static const uint64_t CC2[8], 64) = { 0x0000000000000000, 0x3F615D6EBEE0CA32, 0xBF3FA4970E2BCE23, 0x3F46B17B16B5929F, 0xBF65B7FC1EAC2099, 0xBF483110C30FABA4};
MLAS_DECLSPEC_ALIGN(static const uint64_t CC3[8], 64) = { 0x0000000000000000, 0xBF468D71CF4F0918, 0xBF6061798E58D0FD, 0xBF5124947A8BF45E, 0x3F508C6BD211D736, 0x3F1618DA72860403};
MLAS_DECLSPEC_ALIGN(static const uint64_t CB0[8], 64) = { 0x0000000000000000, 0xBF4207BC640D1509, 0x3F1C9744E36A5706, 0x3F4401AE28BA4DD5, 0xBF2F6DBBF4D6257F, 0xBEE397A9FA5686A2};

MLAS_DECLSPEC_ALIGN(static const uint64_t CD0[8], 64) = { 0x0000000000000000, 0x40312115B0932F24, 0xBF68C0D83DD22E02, 0x3FA1B3FD95EA9564, 0xC053FABD70601067, 0xC08A5C9D5FE8B9F6};
MLAS_DECLSPEC_ALIGN(static const uint64_t CD1[8], 64) = { 0x3FBCE2D77791DD77, 0xC0160D6CD0991EA3, 0x401C0A9EE4108F94, 0x40250CECD79A020A, 0x404A06640EE87808, 0x406EFF5F088CEC4B};
MLAS_DECLSPEC_ALIGN(static const uint64_t CD2[8], 64) = { 0x0000000000000000, 0xBFE04A567A6DBE4A, 0xC01056F9B5E387F5, 0xC0190DC96FF66CCD, 0xC0283F30817A3F08, 0xC03A5743DF38FDE0};
MLAS_DECLSPEC_ALIGN(static const uint64_t CD3[8], 64) = { 0xBF9B582755CDF345, 0x3FF0000000000000, 0x3FF0000000000000, 0x3FF0000000000000, 0x3FF0000000000000, 0x3FF0000000000000};

MLAS_DECLSPEC_ALIGN(static const uint64_t CA0[8], 64) = { 0x0000000000000000, 0xBD54E7E451AF0E36, 0xBE1ACEC2859CB55F, 0x3EABD5A2482B4979, 0xBF5BA377DDAB4E17, 0x3FF0E2920E0391AF};
MLAS_DECLSPEC_ALIGN(static const uint64_t CA1[8], 64) = { 0x3FF20DD7504270CB, 0x3FF20DD75043FE20, 0x3FF20DD75E8D2B64, 0x3FF20DCAA52085D5, 0x3FF2397F1D8FC0ED, 0xC00D249D1A95A5AE};
MLAS_DECLSPEC_ALIGN(static const uint64_t CA2[8], 64) = { 0x0000000000000000, 0xBE05680ACF8280E4, 0xBEABC6A83208FCFC, 0x3F13A994A348795B, 0xBF9945BFC1915C21, 0x40233905061C3803};
MLAS_DECLSPEC_ALIGN(static const uint64_t CA3[8], 64) = { 0xBFD8127465AFE719, 0xBFD812745E92C3D3, 0xBFD81253E42E7B99, 0xBFD8167B2DFCDE44, 0xBFD747AAABB690D8, 0xC027560B851F7690};

void
MLASCALL
MlasErffKernel_avx512(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine implements the generic kernel for the error function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    #if _MSC_VER && !__INTEL_COMPILER
    #pragma warning (push)
    #pragma warning (disable: 4310)
    #endif
    const __m512d mmCC0 = _mm512_load_pd(CC0);
    const __m512d mmCC1 = _mm512_load_pd(CC1);
    const __m512d mmCC2 = _mm512_load_pd(CC2);
    const __m512d mmCC3 = _mm512_load_pd(CC3);
    const __m512d mmCB0 = _mm512_load_pd(CB0);

    const __m512d mmCD0 = _mm512_load_pd(CD0);
    const __m512d mmCD1 = _mm512_load_pd(CD1);
    const __m512d mmCD2 = _mm512_load_pd(CD2);
    const __m512d mmCD3 = _mm512_load_pd(CD3);
	
	const __m512d mmCA0 = _mm512_load_pd(CA0);
    const __m512d mmCA1 = _mm512_load_pd(CA1);
    const __m512d mmCA2 = _mm512_load_pd(CA2);
    const __m512d mmCA3 = _mm512_load_pd(CA3);

    const __m512d mmZERO = _mm512_set1_pd(0);

    //TODO: will the aligned load/store be faster???
    while (N >= 8) {
        __m256 xf = _mm256_loadu_ps(Input);

        __m512d x = _mm512_cvtps_pd(xf);
        __m512i sign_bits = _mm512_and_epi64(_mm512_castpd_si512(x), _mm512_set1_epi64(0x8000000000000000));
        __m512d ax = _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(x), sign_bits));
        ax = _mm512_min_pd(ax, _mm512_castsi512_pd(_mm512_set1_epi64(0x400fffffffffffff))); // the bigest double less than 4.0
    
        __mmask8 ge_0_125 = _mm512_cmp_pd_mask(ax, _mm512_set1_pd(0.125), _CMP_GE_OQ);
        __m512i vindex = _mm512_srli_epi64(_mm512_castpd_si512(ax), 52);
        vindex = _mm512_sub_epi64(vindex, _mm512_set1_epi64(0x3FB));
        vindex = _mm512_mask_blend_epi64(ge_0_125, _mm512_castpd_si512(mmZERO), vindex);
    
        __m512d x2 = _mm512_mul_pd(ax, ax);

        __m512d resultpd = _mm512_permutexvar_pd(vindex, mmCA3);
        resultpd = _mm512_fmadd_pd(resultpd, ax, _mm512_permutexvar_pd(vindex, mmCA2));
        resultpd = _mm512_fmadd_pd(resultpd, ax, _mm512_permutexvar_pd(vindex, mmCA1));
        resultpd = _mm512_fmadd_pd(resultpd, ax, _mm512_permutexvar_pd(vindex, mmCA0));
    
        __m512d PolC = _mm512_permutexvar_pd(vindex, mmCB0);
        PolC = _mm512_fmadd_pd(PolC, ax, _mm512_permutexvar_pd(vindex, mmCC3));
        PolC = _mm512_fmadd_pd(PolC, ax, _mm512_permutexvar_pd(vindex, mmCC2));
        PolC = _mm512_fmadd_pd(PolC, ax, _mm512_permutexvar_pd(vindex, mmCC1));
        PolC = _mm512_fmadd_pd(PolC, ax, _mm512_permutexvar_pd(vindex, mmCC0));
    
        __m512d PolD = _mm512_permutexvar_pd(vindex, mmCD3);
        PolD = _mm512_fmadd_pd(PolD, ax, _mm512_permutexvar_pd(vindex, mmCD2));
        PolD = _mm512_fmadd_pd(PolD, ax, _mm512_permutexvar_pd(vindex, mmCD1));
        PolD = _mm512_fmadd_pd(PolD, ax, _mm512_permutexvar_pd(vindex, mmCD0));
    
        PolD = _mm512_mul_pd(PolD, x2);
        PolC = _mm512_mul_pd(PolC, x2);
        resultpd = _mm512_add_pd(resultpd, _mm512_mul_pd(PolC, PolD));
        resultpd = _mm512_castsi512_pd(_mm512_or_epi64(_mm512_castpd_si512(resultpd), sign_bits));

        _mm256_storeu_ps(Output, _mm512_cvtpd_ps(resultpd));

        Input += 8;
        Output += 8;
        N -= 8;
    }

    if (N > 0) {
        __mmask8 input_mask = (1 << N) - 1;
        __m256 xf = _mm256_maskz_loadu_ps(input_mask, Input);

        __m512d x = _mm512_cvtps_pd(xf);
        __m512i sign_bits = _mm512_and_epi64(_mm512_castpd_si512(x), _mm512_set1_epi64(0x8000000000000000));
        __m512d ax = _mm512_castsi512_pd(_mm512_xor_epi64(_mm512_castpd_si512(x), sign_bits));
        ax = _mm512_min_pd(ax, _mm512_castsi512_pd(_mm512_set1_epi64(0x400fffffffffffff))); // the bigest double less than 4.0
    
        __mmask8 ge_0_125 = _mm512_cmp_pd_mask(ax, _mm512_set1_pd(0.125), _CMP_GE_OQ);
        __m512i vindex = _mm512_srli_epi64(_mm512_castpd_si512(ax), 52);
        vindex = _mm512_sub_epi64(vindex, _mm512_set1_epi64(0x3FB));
        vindex = _mm512_mask_blend_epi64(ge_0_125, _mm512_castpd_si512(mmZERO), vindex);
    
        __m512d x2 = _mm512_mul_pd(ax, ax);

        __m512d resultpd = _mm512_permutexvar_pd(vindex, mmCA3);
        resultpd = _mm512_fmadd_pd(resultpd, ax, _mm512_permutexvar_pd(vindex, mmCA2));
        resultpd = _mm512_fmadd_pd(resultpd, ax, _mm512_permutexvar_pd(vindex, mmCA1));
        resultpd = _mm512_fmadd_pd(resultpd, ax, _mm512_permutexvar_pd(vindex, mmCA0));
    
        __m512d PolC = _mm512_permutexvar_pd(vindex, mmCB0);
        PolC = _mm512_fmadd_pd(PolC, ax, _mm512_permutexvar_pd(vindex, mmCC3));
        PolC = _mm512_fmadd_pd(PolC, ax, _mm512_permutexvar_pd(vindex, mmCC2));
        PolC = _mm512_fmadd_pd(PolC, ax, _mm512_permutexvar_pd(vindex, mmCC1));
        PolC = _mm512_fmadd_pd(PolC, ax, _mm512_permutexvar_pd(vindex, mmCC0));
    
        __m512d PolD = _mm512_permutexvar_pd(vindex, mmCD3);
        PolD = _mm512_fmadd_pd(PolD, ax, _mm512_permutexvar_pd(vindex, mmCD2));
        PolD = _mm512_fmadd_pd(PolD, ax, _mm512_permutexvar_pd(vindex, mmCD1));
        PolD = _mm512_fmadd_pd(PolD, ax, _mm512_permutexvar_pd(vindex, mmCD0));
    
        PolD = _mm512_mul_pd(PolD, x2);
        PolC = _mm512_mul_pd(PolC, x2);
        resultpd = _mm512_add_pd(resultpd, _mm512_mul_pd(PolC, PolD));
        resultpd = _mm512_castsi512_pd(_mm512_or_epi64(_mm512_castpd_si512(resultpd), sign_bits));

        _mm256_mask_storeu_ps(Output, input_mask, _mm512_cvtpd_ps(resultpd));
    }

    #if _MSC_VER && !__INTEL_COMPILER
    #pragma warning (pop)
    #endif
}

void
MLASCALL
MlasComputeErff(
    const float* Input,
    float* Output,
    size_t N
    )
/*++

Routine Description:

    This routine computes the error function.

Arguments:

    Input - Supplies the input buffer.

    Output - Supplies the output buffer.

    N - Supplies the number of elements to process.

Return Value:

    None.

--*/
{
    //TODO: support different platform
    MlasErffKernel_avx512(Input, Output, N);
// #if defined(MLAS_TARGET_AMD64)
//     MlasPlatform.TanhKernelRoutine(Input, Output, N);
// #else
//     MlasErffKernel_avx512(Input, Output, N);
// #endif
}
