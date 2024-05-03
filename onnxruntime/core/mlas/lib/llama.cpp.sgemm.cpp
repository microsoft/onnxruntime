// ported/adapted from https://github.com/ggerganov/llama.cpp/pull/6414
#define __AVX2__ 1

#include <assert.h>
#include <immintrin.h>
#include "llama.cpp.sgemm.h"
#include "sqnbitgemm.h"
//#include "sqnbitgemm_kernel_avx_common.h"
#include <algorithm>
#include <cassert>

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((__noinline__))
#endif

#if defined(__ARM_NEON) || defined(__AVX512F__)
#define VECTOR_REGISTERS 32
#else
#define VECTOR_REGISTERS 16
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED FUSED MULTIPLY ADD

/**
 * Computes a * b + c.
 */
template <typename T, typename U>
inline U
madd(T a, T b, U c)
{
    return add(mul(a, b), c);
}

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m256
madd(__m256 a, __m256 b, __m256 c)
{
    return _mm256_fmadd_ps(a, b, c);
}
#endif
#if defined(__AVX512F__)
template <>
inline __m512
madd(__m512 a, __m512 b, __m512 c)
{
    return _mm512_fmadd_ps(a, b, c);
}
#endif


template <typename TA, typename TB, typename TC>
class tinyBLAS_Q0_AVX2
{
   public:
    tinyBLAS_Q0_AVX2(int64_t k, const TA *A, int64_t lda, const TB *B, int64_t ldb, TC *C, int64_t ldc,
      const float *QuantBScale, int64_t StrideQuantBScale)
        : A_q4_(A), B_q8_(B), C(C), k(k), lda_q4_(lda), ldb_q8_(ldb), ldc_(ldc),
      Quant4Scale_(QuantBScale), StrideQuant4Scale_(StrideQuantBScale)
    {
    }

    void matmul(int64_t m, int64_t n)
    {
        mnpack(0, m, 0, n);
    }

   private:
    void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n)
    {
        int64_t mc, nc, mp, np;
        switch ((std::min(m - m0, (int64_t)4) << 4) | std::min(n - n0, (int64_t)4)) {
#if VECTOR_REGISTERS == 32
            case 0x44:
                mc = 4;
                nc = 4;
                gemm<4, 4>(m0, m, n0, n);
                break;
            case 0x43:
                mc = 4;
                nc = 3;
                gemm<4, 3>(m0, m, n0, n);
                break;
            case 0x34:
                mc = 3;
                nc = 4;
                gemm<3, 4>(m0, m, n0, n);
                break;
            case 0x33:
                mc = 3;
                nc = 3;
                gemm<3, 3>(m0, m, n0, n);
                break;
            case 0x42:
                mc = 4;
                nc = 2;
                gemm<4, 2>(m0, m, n0, n);
                break;
            case 0x24:
                mc = 2;
                nc = 4;
                gemm<2, 4>(m0, m, n0, n);
                break;
#else
            case 0x44:
            case 0x43:
            case 0x42:
                mc = 4;
                nc = 2;
                gemm<4, 2>(m0, m, n0, n);
                break;
            case 0x34:
            case 0x24:
                mc = 2;
                nc = 4;
                gemm<2, 4>(m0, m, n0, n);
                break;
            case 0x33:
#endif
            case 0x32:
                mc = 3;
                nc = 2;
                gemm<3, 2>(m0, m, n0, n);
                break;
            case 0x23:
                mc = 2;
                nc = 3;
                gemm<2, 3>(m0, m, n0, n);
                break;
            case 0x41:
                mc = 4;
                nc = 1;
                gemm<4, 1>(m0, m, n0, n);
                break;
            case 0x22:
                mc = 2;
                nc = 2;
                gemm<2, 2>(m0, m, n0, n);
                break;
            case 0x14:
                mc = 1;
                nc = 4;
                gemm<1, 4>(m0, m, n0, n);
                break;
            case 0x31:
                mc = 3;
                nc = 1;
                gemm<3, 1>(m0, m, n0, n);
                break;
            case 0x13:
                mc = 1;
                nc = 3;
                gemm<1, 3>(m0, m, n0, n);
                break;
            case 0x21:
                mc = 2;
                nc = 1;
                gemm<2, 1>(m0, m, n0, n);
                break;
            case 0x12:
                mc = 1;
                nc = 2;
                gemm<1, 2>(m0, m, n0, n);
                break;
            case 0x11:
                mc = 1;
                nc = 1;
                gemm<1, 1>(m0, m, n0, n);
                break;
            default:
                return;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

    template <int RM, int RN>
    NOINLINE void gemm(int64_t m0, int64_t m, int64_t n0, int64_t n)
    {
        constexpr size_t BlkBitWidth4 = 4;
        constexpr size_t BlkLen32 = 32;
        constexpr size_t BlkDataSizeInBytes16 = MlasQNBitBlkDataSizeInBytes(BlkBitWidth4, BlkLen32);
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        for (int64_t tile = 0; tile < tiles; ++tile) {
            int64_t ii = m0 + tile / xtiles * RM;
            int64_t jj = n0 + tile % xtiles * RN;
            __m256 Cv[RN][RM] = {};
            for (int64_t l = 0; l < k; ++l)       // blk count (BlockCountK)
                for (int64_t j = 0; j < RN; ++j)  //
                    for (int64_t i = 0; i < RM; ++i) {
                        const std::byte *Quant4ABlk = A_q4_ + lda_q4_ * (ii + i) + l * BlkDataSizeInBytes16;
                        const std::byte *Quant8BBlk = B_q8_ + ldb_q8_ * (jj + j) + l * Q8BlkSize(BlkLen32);
                        const float &scale_q8 = Q8BlkScale(Quant8BBlk);
                        const float &scale_q4 = *(Quant4Scale_ + (ii + i) * StrideQuant4Scale_ + l);

                        const int8_t zp = 8;
                        const __m256i q4_v = load_q4(Quant4ABlk, zp);
                        const __m256i q8_v = load_q8(Quant8BBlk);
                        Cv[j][i] = madd(
                            _mm256_set1_ps(scale_q8 * scale_q4),
                            updot(_mm256_sign_epi8(q4_v, q4_v), _mm256_sign_epi8(q8_v, q4_v)),
                            Cv[j][i]
                        );
                    }
            for (int64_t j = 0; j < RN; ++j)
                for (int64_t i = 0; i < RM; ++i)
                    C[ldc_ * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    inline float hsum(__m128 x)
    {
          x = _mm_add_ps(x, _mm_movehl_ps(x, x));
          x = _mm_add_ss(x, _mm_movehdup_ps(x));
          return _mm_cvtss_f32(x);
    }
    inline float hsum(__m256 x)
    {
        return hsum(_mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x)));
    }

    inline __m256i load_q8(const std::byte *Quant8Blk)
    {
        return _mm256_loadu_si256((const __m256i *)Q8BlkData(Quant8Blk));
    }

    inline __m256i load_q4(const std::byte *Quant4DataPtr, const int8_t zp)
    {
        // | v0  v16 | v1  v17 | ... | v14 v30 | v15 v31 |
        // | v32 v48 | v33 v49 | ... | v46 v62 | v47 v63 |
        const __m128i bv_packed0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(Quant4DataPtr));

        const __m128i low_mask = _mm_set1_epi8(15);
        const __m128i bv_lo0 = _mm_and_si128(bv_packed0, low_mask);                     // 0, 1, 2, 3,...
        const __m128i bv_hi0 = _mm_and_si128(_mm_srli_epi16(bv_packed0, 4), low_mask);  // 16, 17, 18, 19,...
        __m256i bv_32_epi8 = _mm256_set_m128i(bv_hi0, bv_lo0);
        const __m256i bzp0 = _mm256_set1_epi8(zp);
        bv_32_epi8 = _mm256_sub_epi8(bv_32_epi8, bzp0);

        return bv_32_epi8;
    }

    inline __m256 updot(__m256i u, __m256i s)
    {
        __m256i res;
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
        res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
#else
        res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
#endif
        return _mm256_cvtepi32_ps(res);
    }

    const TA *const A_q4_;
    const TB *const B_q8_;
    TC *const C;
    const int64_t k;
    const int64_t lda_q4_;
    const int64_t ldb_q8_;
    const int64_t ldc_;
    const float *Quant4Scale_;
    int64_t StrideQuant4Scale_;
};

/**
 * Performs optimized matrix multiplication on CPU.
 *
 * This subroutine may compute C = Aᵀ * B with column major ordering.
 * Despite its name, this isn't a generalized implementation. Work is
 * only performed when a handwritten kernel is written and available.
 * Otherwise the caller should fall back to a general matmul routine.
 *
 * For example, for single-threaded single-precision GEMM you can say
 *
 *     llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc,
 *                     0, 1, GGML_TASK_TYPE_COMPUTE,
 *                     GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32);
 *
 * @param m is rows in `A` and `C`
 * @param n is cols in `B` and `C`
 * @param k is cols in `A` and rows in `B`
 * @param A is first input matrix (always transposed)
 * @param lda is row stride of `A`
 * @param B is second input matrix (never transposed)
 * @param ldb is row stride of `B`
 * @param C is input/output array of output matrices
 * @param ldc is row stride of `C`
 * @param ith is thread id (must be less than `nth`)
 * @param nth is number of threads (must be greater than zero)
 * @param task is GGML task type
 * @param Atype is GGML data type of `A`
 * @param Btype is GGML data type of `B`
 * @param Ctype is GGML data type of `C`
 * @return true if this function was able to service the matmul request
 */
bool
llamafile_sgemm(
    int64_t m,
    int64_t n,
    int64_t k,
    const std::byte *A,
    int64_t lda,
    const std::byte *B,
    int64_t ldb,
    float *C,
    int64_t ldc,
    const float *QuantBScale,
    int64_t StrideQuantBScale
)
{
    tinyBLAS_Q0_AVX2<std::byte, std::byte, float> tb{k, A, lda, B, ldb, C, ldc, QuantBScale, StrideQuantBScale};
    tb.matmul(m, n);
    return true;
}
