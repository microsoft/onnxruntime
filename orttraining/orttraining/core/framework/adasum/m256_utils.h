// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef ENABLE_CPU_FP16_TRAINING_OPS
#include <emmintrin.h>
#include <immintrin.h>

namespace onnxruntime {
namespace training {
// reduce 4xfloat64 into one double
inline double Mm256ReductionPd(__m256d v) {
  __m128d vlow = _mm256_castpd256_pd128(v);
  __m128d vhigh = _mm256_extractf128_pd(v, 1);  // high 128
  vlow = _mm_add_pd(vlow, vhigh);               // reduce down to 128

  __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
  return _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
}

// load 8 float16s from a and return the __m256 register
inline __m256 MmLoaduPh(const uint16_t* a) {
  __m128i r = _mm_loadu_si128((__m128i*)(a));
  return _mm256_cvtph_ps(r);
}

// store 8 float16 from val into a
inline void MmStorePh(uint16_t* a, __m256 val) {
  __m128i r = _mm256_cvtps_ph(val, 0);
  _mm_storeu_si128((__m128i*)a, r);
}

// load len (<= 8) float16s from a, fill the rest with 0s, and return the
// __m256 register
inline __m256 MmLoaduPhPartial(const uint16_t* a, int len) {
  short e[8];
  std::memset(e, 0, sizeof(e));
  std::memcpy(e, a, std::min(len, 8) * sizeof(short));
  __m128i es = _mm_set_epi16(e[7], e[6], e[5], e[4], e[3], e[2], e[1], e[0]);
  return _mm256_cvtph_ps(es);
}

// store the first len (< 8) float16s from val and store into a
inline void MmStorePhPartial(uint16_t* a, __m256 val, int len) {
  __m128i r = _mm256_cvtps_ph(val, 0);
  // for (int i = 0; i < std::min(len, 8); i++)
  //    a[i].value = _mm_extract_epi16(r, i);
  // but we cannot do this because the second argument to _mm_extract_epi16
  // has to be a compile time constant
  if (0 < len)
    a[0] = (short)_mm_extract_epi16(r, 0);
  if (1 < len)
    a[1] = (short)_mm_extract_epi16(r, 1);
  if (2 < len)
    a[2] = (short)_mm_extract_epi16(r, 2);
  if (3 < len)
    a[3] = (short)_mm_extract_epi16(r, 3);
  if (4 < len)
    a[4] = (short)_mm_extract_epi16(r, 4);
  if (5 < len)
    a[5] = (short)_mm_extract_epi16(r, 5);
  if (6 < len)
    a[6] = (short)_mm_extract_epi16(r, 6);
  if (7 < len)
    a[7] = (short)_mm_extract_epi16(r, 7);
}

inline void ComputeDotAndNormSqrdsfp16(const uint16_t* __restrict__ a,
                                       const uint16_t* __restrict__ b,
                                       int len, double& dotProduct,
                                       double& anormsq, double& bnormsq) {
  int i;
  __m256d dotProductVec = _mm256_setzero_pd();
  __m256d anormVec = _mm256_setzero_pd();
  __m256d bnormVec = _mm256_setzero_pd();
  for (i = 0; i < len - 7; i += 8) {
    __m256 aVec = MmLoaduPh(&a[i]);
    __m256 bVec = MmLoaduPh(&b[i]);
    __m256d aBot = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 0));
    __m256d aTop = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 1));
    __m256d bBot = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 0));
    __m256d bTop = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 1));
    dotProductVec = _mm256_fmadd_pd(aBot, bBot, dotProductVec);
    dotProductVec = _mm256_fmadd_pd(aTop, bTop, dotProductVec);
    anormVec = _mm256_fmadd_pd(aBot, aBot, anormVec);
    anormVec = _mm256_fmadd_pd(aTop, aTop, anormVec);
    bnormVec = _mm256_fmadd_pd(bBot, bBot, bnormVec);
    bnormVec = _mm256_fmadd_pd(bTop, bTop, bnormVec);
  }
  if (i < len) {
    __m256 aVec = MmLoaduPhPartial(&a[i], len - i);
    __m256 bVec = MmLoaduPhPartial(&b[i], len - i);
    __m256d aBot = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 0));
    __m256d aTop = _mm256_cvtps_pd(_mm256_extractf128_ps(aVec, 1));
    __m256d bBot = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 0));
    __m256d bTop = _mm256_cvtps_pd(_mm256_extractf128_ps(bVec, 1));
    dotProductVec = _mm256_fmadd_pd(aBot, bBot, dotProductVec);
    dotProductVec = _mm256_fmadd_pd(aTop, bTop, dotProductVec);
    anormVec = _mm256_fmadd_pd(aBot, aBot, anormVec);
    anormVec = _mm256_fmadd_pd(aTop, aTop, anormVec);
    bnormVec = _mm256_fmadd_pd(bBot, bBot, bnormVec);
    bnormVec = _mm256_fmadd_pd(bTop, bTop, bnormVec);
  }

  dotProduct = Mm256ReductionPd(dotProductVec);
  anormsq = Mm256ReductionPd(anormVec);
  bnormsq = Mm256ReductionPd(bnormVec);
}

inline void ScaledAddfp16(int len, double acoeff, uint16_t* __restrict__ a,
                          double bcoeff, uint16_t* __restrict__ b) {
  int i;
  __m256 acoeffVec = _mm256_set1_ps((float)(acoeff));
  __m256 bcoeffVec = _mm256_set1_ps((float)bcoeff);
  for (i = 0; i < len - 7; i += 8) {
    __m256 aVec = MmLoaduPh(&a[i]);
    __m256 bVec = MmLoaduPh(&b[i]);
    aVec = _mm256_mul_ps(acoeffVec, aVec);
    MmStorePh(&a[i], _mm256_fmadd_ps(bcoeffVec, bVec, aVec));
  }
  if (i < len) {
    __m256 aVec = MmLoaduPhPartial(&a[i], len - i);
    __m256 bVec = MmLoaduPhPartial(&b[i], len - i);
    aVec = _mm256_mul_ps(acoeffVec, aVec);
    MmStorePhPartial(&a[i], _mm256_fmadd_ps(bcoeffVec, bVec, aVec), len - i);
  }
}

}  // namespace training
}  // namespace onnxruntime
#endif  // ENABLE_CPU_FP16_TRAINING_OPS