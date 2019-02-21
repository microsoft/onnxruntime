#include "mlasi.h"

static inline 
__m256 expf_v256(__m256 a) {
    __m256 c = _mm256_set1_ps(1.25829120e+7f);
    __m256 r = _mm256_set1_ps(1.44269502e+0f);
    r = _mm256_fmadd_ps(r, a, c);
    r = _mm256_sub_ps(r, c);
    __m256 f = _mm256_fmadd_ps(r, _mm256_set1_ps(-6.93145752e-1f), a);
    f = _mm256_fmadd_ps(r, _mm256_set1_ps(-1.42860677e-6f), f);
    __m256i i = _mm256_cvtps_epi32(r);

    r = _mm256_set1_ps(1.38319808e-3f);
    r = _mm256_fmadd_ps(r, f, _mm256_set1_ps(8.37550033e-3f));
    r = _mm256_fmadd_ps(r, f, _mm256_set1_ps(4.16689515e-2f));
    r = _mm256_fmadd_ps(r, f, _mm256_set1_ps(1.66664466e-1f));
    r = _mm256_fmadd_ps(r, f, _mm256_set1_ps(4.99999851e-1f));
    r = _mm256_fmadd_ps(r, f, _mm256_set1_ps(1.00000000e+0f));
    r = _mm256_fmadd_ps(r, f, _mm256_set1_ps(1.00000000e+0f));

    // ldexpf(r, i)
    __m256i m = _mm256_castps_si256(_mm256_cmp_ps(r, _mm256_setzero_ps(), _CMP_NEQ_OQ));
    m = _mm256_and_si256(m, i);
    m = _mm256_slli_epi32(m, 23);
    r = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(r), m));
    return r;
}

static inline
__m256 erff_v256(__m256 xf) {
    __m256 sign_bits = _mm256_and_ps(xf, _mm256_set1_ps(-0.0f));
    __m256 ax = _mm256_min_ps(_mm256_xor_ps(xf, sign_bits), _mm256_set1_ps(3.99999f));  //almost 4.0
    __m256 gt_0_921875 = _mm256_cmp_ps(ax, _mm256_set1_ps(0.921875f), _CMP_GT_OQ);
    __m256 x2 = _mm256_mul_ps(ax, ax);

    // |xf| <= 0.921875f
    __m256 r_le = _mm256_set1_ps(-5.99104969e-4f);
    r_le = _mm256_fmadd_ps(r_le, x2, _mm256_set1_ps(4.99339588e-3f));
    r_le = _mm256_fmadd_ps(r_le, x2, _mm256_set1_ps(-2.67667342e-2f));
    r_le = _mm256_fmadd_ps(r_le, x2, _mm256_set1_ps(1.12818025e-1f));
    r_le = _mm256_fmadd_ps(r_le, x2, _mm256_set1_ps(-3.76124859e-1f));
    r_le = _mm256_fmadd_ps(r_le, x2, _mm256_set1_ps(1.28379151e-1f));
    r_le = _mm256_fmadd_ps(r_le, ax, ax);

    // |xf| > 0.921875f
    __m256 u = _mm256_set1_ps(3.88393435e-3f);
    __m256 r_gt = _mm256_set1_ps(1.72948930e-5f);
    u = _mm256_fmadd_ps(u, ax, _mm256_set1_ps(-2.42545605e-2f));
    r_gt = _mm256_fmadd_ps(r_gt, ax, _mm256_set1_ps(-3.83208680e-4f));
    r_gt = _mm256_fmadd_ps(r_gt, x2, u);
    r_gt = _mm256_fmadd_ps(r_gt, ax, _mm256_set1_ps(1.06777847e-1f));
    r_gt = _mm256_fmadd_ps(r_gt, ax, _mm256_set1_ps(6.34846687e-1f));
    r_gt = _mm256_fmadd_ps(r_gt, ax, _mm256_set1_ps(1.28717512e-1f));
    r_gt = _mm256_fmadd_ps(r_gt, ax, ax);
    r_gt = _mm256_xor_ps(r_gt, _mm256_set1_ps(-0.0f));
    r_gt = expf_v256(r_gt);
    r_gt = _mm256_sub_ps(_mm256_set1_ps(1.0f), r_gt);

    return _mm256_or_ps(_mm256_blendv_ps(r_le, r_gt, gt_0_921875), sign_bits);
}

void
MLASCALL
MlasErffKernelFma3(
    const float* Input,
    float* Output,
    size_t N)
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
    for (; N >= 8; N -= 8) {
        _mm256_storeu_ps(Output, erff_v256(_mm256_loadu_ps(Input)));
        Input += 8;
        Output += 8;
    }

    if (N > 0) {
        __m256i v = _mm256_castps_si256(_mm256_setzero_ps());

        v = _mm256_insert_epi32(v, *(const int*)(Input + 0), 0);
        if (N > 1) v = _mm256_insert_epi32(v, *(const int*)(Input + 1), 1);
        if (N > 2) v = _mm256_insert_epi32(v, *(const int*)(Input + 2), 2);
        if (N > 3) v = _mm256_insert_epi32(v, *(const int*)(Input + 3), 3);
        if (N > 4) v = _mm256_insert_epi32(v, *(const int*)(Input + 4), 4);
        if (N > 5) v = _mm256_insert_epi32(v, *(const int*)(Input + 5), 5);
        if (N > 6) v = _mm256_insert_epi32(v, *(const int*)(Input + 6), 6);

        v = _mm256_castps_si256(erff_v256(_mm256_castsi256_ps(v)));

        *(int*)(Output + 0) = _mm256_extract_epi32(v, 0);
        if (N > 1) *(int*)(Output + 1) = _mm256_extract_epi32(v, 1);
        if (N > 2) *(int*)(Output + 2) = _mm256_extract_epi32(v, 2);
        if (N > 3) *(int*)(Output + 3) = _mm256_extract_epi32(v, 3);
        if (N > 4) *(int*)(Output + 4) = _mm256_extract_epi32(v, 4);
        if (N > 5) *(int*)(Output + 5) = _mm256_extract_epi32(v, 5);
        if (N > 6) *(int*)(Output + 6) = _mm256_extract_epi32(v, 6);
    }
}


void
MLASCALL
MlasErffKernel(
    const float* Input,
    float* Output,
    size_t N)
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
    for (; N > 0; --N) {
        *Output++ = std::erf(*Input++);
    }
}

void
MLASCALL
MlasComputeErff(
    const float* Input,
    float* Output,
    size_t N)
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
#if defined(MLAS_TARGET_AMD64)
  MlasPlatform.ErffKernelRoutine(Input, Output, N);
#else
  MlasErffKernel(Input, Output, N);
#endif
}
