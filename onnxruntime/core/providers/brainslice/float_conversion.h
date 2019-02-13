#pragma once

#include <cstdint>
#include "core/common/cpuid_info.h"
#include "3rdparty/half.hpp"

// Disable the warning: error C4752: found Intel(R) Advanced Vector Extensions; consider using /arch:AVX
#ifdef _MSC_VER
#pragma warning(disable:4752)
#endif

namespace onnxruntime {
namespace brainslice {

// Convert float-32 to float-16 with round-to-nearest without SIMD instructions.
inline void Float32ToFloat16NearestNoAVX2(const float* input, uint16_t* output, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    half_float::half float16(*input++);
    *output++ = *reinterpret_cast<uint16_t*>(&float16);
  }
}

// Convert float-32 to float-16 with round-to-nearest using AVX2.
inline void Float32ToFloat16NearestAVX2(const float* input, uint16_t* output, size_t count) {
  size_t remaining = count;

  while (remaining >= 64) {
    __m256 i1 = _mm256_loadu_ps(input);
    __m256 i2 = _mm256_loadu_ps(input + 8);
    __m256 i3 = _mm256_loadu_ps(input + 16);
    __m256 i4 = _mm256_loadu_ps(input + 24);
    __m256 i5 = _mm256_loadu_ps(input + 32);
    __m256 i6 = _mm256_loadu_ps(input + 40);
    __m256 i7 = _mm256_loadu_ps(input + 48);
    __m256 i8 = _mm256_loadu_ps(input + 56);

    __m128i o1 = _mm256_cvtps_ph(i1, 0);
    __m128i o2 = _mm256_cvtps_ph(i2, 0);
    __m128i o3 = _mm256_cvtps_ph(i3, 0);
    __m128i o4 = _mm256_cvtps_ph(i4, 0);
    __m128i o5 = _mm256_cvtps_ph(i5, 0);
    __m128i o6 = _mm256_cvtps_ph(i6, 0);
    __m128i o7 = _mm256_cvtps_ph(i7, 0);
    __m128i o8 = _mm256_cvtps_ph(i8, 0);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(output), o1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 8), o2);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 16), o3);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 24), o4);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 32), o5);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 40), o6);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 48), o7);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 56), o8);
    input += 64;
    output += 64;
    remaining -= 64;
  }

  while (remaining > 8) {
    __m256 i1 = _mm256_loadu_ps(input);
    __m128i o1 = _mm256_cvtps_ph(i1, 0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(output), o1);

    input += 8;
    output += 8;
    remaining -= 8;
  }

  if (remaining > 0) {
    Float32ToFloat16NearestNoAVX2(input, output, remaining);
  }
}

// Convert float-16 to float-32 without SIMD instructions.
inline void Float16ToFloat32NoAVX2(const uint16_t* input, float* output, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    const half_float::half* float16 = reinterpret_cast<const half_float::half*>(input);
    *output = static_cast<float>(*float16);

    ++input;
    ++output;
  }
}

// Convert float-16 to float-32 using AVX2.
inline void Float16ToFloat32AVX2(const uint16_t* input, float* output, size_t count) {
  size_t remaining = count;

  while (remaining >= 64) {
    __m128i i1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
    __m128i i2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 8));
    __m128i i3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 16));
    __m128i i4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 24));
    __m128i i5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 32));
    __m128i i6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 40));
    __m128i i7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 48));
    __m128i i8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 56));

    __m256 o1 = _mm256_cvtph_ps(i1);
    __m256 o2 = _mm256_cvtph_ps(i2);
    __m256 o3 = _mm256_cvtph_ps(i3);
    __m256 o4 = _mm256_cvtph_ps(i4);
    __m256 o5 = _mm256_cvtph_ps(i5);
    __m256 o6 = _mm256_cvtph_ps(i6);
    __m256 o7 = _mm256_cvtph_ps(i7);
    __m256 o8 = _mm256_cvtph_ps(i8);

    _mm256_storeu_ps(output, o1);
    _mm256_storeu_ps(output + 8, o2);
    _mm256_storeu_ps(output + 16, o3);
    _mm256_storeu_ps(output + 24, o4);
    _mm256_storeu_ps(output + 32, o5);
    _mm256_storeu_ps(output + 40, o6);
    _mm256_storeu_ps(output + 48, o7);
    _mm256_storeu_ps(output + 56, o8);
    input += 64;
    output += 64;
    remaining -= 64;
  }

  while (remaining >= 8) {
    __m128i i1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input));
    __m256 o1 = _mm256_cvtph_ps(i1);
    _mm256_storeu_ps(output, o1);
    input += 8;
    output += 8;
    remaining -= 8;
  }

  if (remaining > 0) {
    Float16ToFloat32NoAVX2(input, output, remaining);
  }
}

static const auto Float32ToFloat16 = CPUIDInfo::GetCPUIDInfo().HasF16C() ? Float32ToFloat16NearestAVX2 : Float32ToFloat16NearestNoAVX2;
static const auto Float16ToFloat32 = CPUIDInfo::GetCPUIDInfo().HasF16C() ? Float16ToFloat32AVX2 : Float16ToFloat32NoAVX2;

}  // namespace brainslice
}  // namespace onnxruntime
