#include "lib/Api.Image/pch.h"
#include "inc/NominalRangeConverter.h"

namespace _winml {
NominalRangeConverter::NominalRangeConverter(winml::LearningModelPixelRange pixelRange) {
  // For Normalization: the formula is input_range[min, max] / scale - shift
  // For DeNormalization: the formula is (input_range[min, max] + shift) * scale
  if (pixelRange == winml::LearningModelPixelRange::ZeroTo255) {
    scale = 1.f;
    shift = 0;
  } else if (pixelRange == winml::LearningModelPixelRange::ZeroToOne) {
    scale = 255.f;
    shift = 0;
  } else if (pixelRange == winml::LearningModelPixelRange::MinusOneToOne) {
    scale = (255.f / 2.f);
    shift = 1;
  }
};

// [0, 255] --> [0, 255]
// [0, 255] / 255 --> [0, 1]
// [0, 255] * 2 / 255 - 1 --> [-1, 1]
float NominalRangeConverter::Normalize(float val) const {
  return val / scale - shift;
}

DirectX::PackedVector::HALF NominalRangeConverter::Normalize(DirectX::PackedVector::HALF val) const {
  return static_cast<DirectX::PackedVector::HALF>(val / scale - shift);
}

#if defined(_M_AMD64) || defined(_M_IX86)
__m128 NominalRangeConverter::Normalize(__m128 sse_data) const {
  __m128 sse_shift = _mm_set1_ps(static_cast<float>(shift));
  __m128 sse_scale = _mm_set1_ps(scale);

  auto sse_dived = _mm_div_ps(sse_data, sse_scale);
  return _mm_sub_ps(sse_dived, sse_shift);
}
#endif

// [0, 255] --> [0, 255]
// ([0, 1] + 0 ) * 255 -> [0, 1]
// ([-1, 1] + 1) * 255 / 2 --> [-1, 1]
float NominalRangeConverter::Denormalize(float val) const {
  return scale * (val + shift);
}

DirectX::PackedVector::HALF NominalRangeConverter::Denormalize(DirectX::PackedVector::HALF val) const {
  return static_cast<DirectX::PackedVector::HALF>(scale * (val + shift));
}

#if defined(_M_AMD64) || defined(_M_IX86)
__m128 NominalRangeConverter::Denormalize(__m128 sse_data) const {
  __m128 sse_shift = _mm_set1_ps(static_cast<float>(shift));
  __m128 sse_scale = _mm_set1_ps(scale);

  auto sse_added = _mm_add_ps(sse_data, sse_shift);
  return _mm_mul_ps(sse_added, sse_scale);
}
#endif
}  // namespace _winml
