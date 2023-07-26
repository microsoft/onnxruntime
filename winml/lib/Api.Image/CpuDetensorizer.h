// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "inc/ImageConversionTypes.h"
#include "inc/NominalRangeConverter.h"

namespace _winml {

class CpuDetensorizer {
 public:
  template <typename T>
  static HRESULT Detensorize(
    _In_ ImageTensorChannelType formatFrom,
    _In_ ImageTensorChannelType formatTo,
    _In_ winml::LearningModelPixelRange pixelRange,
    _In_ T* pCPUTensor,
    _In_ uint32_t bufferWidth,
    _In_ uint32_t tensorHeight,
    _In_ uint32_t tensorWidth,
    _Inout_ BYTE* pData
  ) {
#pragma warning(push)
#pragma warning(disable : 26014 \
)  // warning about possible out of bounds accesing pData, but input is checked for BGRA8 format, so uiCapacity should be in multiples of 4 \
    // output is BGRA8: so blue at i, green is at i + 1, red is at i + 2

    uint32_t bytesPerPixel = formatTo == kImageTensorChannelTypeGRAY8 ? 1 : 4;

    // bufferWidth may have padding because of optimization, but bytesPerRow includes only the real tensor data. We need to jump
    // over bufferWidth's extra padding
    uint32_t bytesPerRow = tensorWidth * bytesPerPixel;
    uint32_t end = bufferWidth * tensorHeight;
    size_t tensorPlaneSize = tensorWidth * tensorHeight;

    auto nominalRangeConverter = NominalRangeConverter(pixelRange);

    if (formatFrom == formatTo && (formatFrom == kImageTensorChannelTypeBGR8 || formatFrom == kImageTensorChannelTypeRGB8)) {
      for (uint32_t i = 0; i < tensorHeight; i++) {
        BYTE* pPixel = pData;

        InterleaveRowFloatToByte(
          pCPUTensor + i * tensorWidth,
          pCPUTensor + tensorPlaneSize + i * tensorWidth,
          pCPUTensor + tensorPlaneSize * 2 + i * tensorWidth,
          tensorWidth,
          pPixel,
          bytesPerPixel,
          nominalRangeConverter
        );

        pData += bufferWidth;
      }
    } else if ((formatFrom == kImageTensorChannelTypeRGB8 && formatTo == kImageTensorChannelTypeBGR8) || (formatFrom == kImageTensorChannelTypeBGR8 && formatTo == kImageTensorChannelTypeRGB8)) {
      for (uint32_t i = 0; i < tensorHeight; i++) {
        BYTE* pPixel = pData;

        InterleaveRowFloatToByte(
          pCPUTensor + tensorPlaneSize * 2 + i * tensorWidth,
          pCPUTensor + tensorPlaneSize + i * tensorWidth,
          pCPUTensor + i * tensorWidth,
          tensorWidth,
          pPixel,
          bytesPerPixel,
          nominalRangeConverter
        );

        pData += bufferWidth;
      }
    } else if (formatFrom == kImageTensorChannelTypeGRAY8 && (formatTo == kImageTensorChannelTypeBGR8 || formatTo == kImageTensorChannelTypeRGB8)) {
      // just replicate the gray data across each channel
      for (uint32_t i = 0; i < end; i += bufferWidth) {
        for (uint32_t j = i; j < i + bytesPerRow; j += 4) {
          BYTE bGray = DetensorizeValue<T>(pCPUTensor, nominalRangeConverter);
          pData[j] = bGray;
          pData[j + 1] = bGray;
          pData[j + 2] = bGray;
          pData[j + 3] = 255;
          pCPUTensor++;
        }
      }
    } else if (formatFrom == kImageTensorChannelTypeGRAY8 && formatTo == kImageTensorChannelTypeGRAY8) {
      for (uint32_t i = 0; i < end; i += bufferWidth) {
        for (uint32_t j = i; j < i + bytesPerRow; j += 1) {
          BYTE bGray = DetensorizeValue<T>(pCPUTensor, nominalRangeConverter);
          pData[j] = bGray;
          pCPUTensor++;
        }
      }
    } else if (formatFrom == kImageTensorChannelTypeBGR8 && formatTo == kImageTensorChannelTypeGRAY8) {
      for (uint32_t i = 0; i < end; i += bufferWidth) {
        for (uint32_t j = i; j < i + bytesPerRow; j += 1) {
          BYTE red, green, blue;

          blue = DetensorizeValue(pCPUTensor, nominalRangeConverter);
          green = DetensorizeValue(pCPUTensor + tensorPlaneSize, nominalRangeConverter);
          red = DetensorizeValue(pCPUTensor + tensorPlaneSize * 2, nominalRangeConverter);

          pData[j] = static_cast<BYTE>(0.2126f * red + 0.7152f * green + 0.0722f * blue);
          pCPUTensor++;
        }
      }
    } else if (formatFrom == kImageTensorChannelTypeRGB8 && formatTo == kImageTensorChannelTypeGRAY8) {
      for (uint32_t i = 0; i < end; i += bufferWidth) {
        for (uint32_t j = i; j < i + bytesPerRow; j += 1) {
          BYTE red, green, blue;

          red = DetensorizeValue(pCPUTensor, nominalRangeConverter);
          green = DetensorizeValue(pCPUTensor + tensorPlaneSize, nominalRangeConverter);
          blue = DetensorizeValue(pCPUTensor + tensorPlaneSize * 2, nominalRangeConverter);

          pData[j] = static_cast<BYTE>(0.2126f * red + 0.7152f * green + 0.0722f * blue);
          pCPUTensor++;
        }
      }
    }
#pragma warning(pop)
    else {
      return E_INVALIDARG;
    }
    return S_OK;
  }

 private:
  template <typename T>
  static float ReadTensor(const T* pCPUTensor, const NominalRangeConverter& nominalRangeConverter) {
    return nominalRangeConverter.Denormalize(*pCPUTensor);
  }

  template <>
  static float ReadTensor<DirectX::PackedVector::HALF>(
    const DirectX::PackedVector::HALF* pCPUTensor, const NominalRangeConverter& nominalRangeConverter
  ) {
    return nominalRangeConverter.Denormalize(DirectX::PackedVector::XMConvertHalfToFloat(*pCPUTensor));
  }

  template <typename T>
  static BYTE DetensorizeValue(const T* pCPUTensor, const NominalRangeConverter& nominalRangeConverter) {
    return static_cast<BYTE>(std::max(0.0f, std::min(255.0f, ReadTensor(pCPUTensor, nominalRangeConverter) + 0.5f)));
  }

  template <typename T>
  static void InterleaveRowFloatToByte(
    const T* xChannel,
    const T* yChannel,
    const T* zChannel,
    uint32_t tensorWidth,
    BYTE* pData,
    uint32_t bytesPerPixel,
    const NominalRangeConverter& nominalRangeConverter
  ) {
    BYTE* pPixel = pData;
    uint32_t tensorWidthRemaining = tensorWidth;

    while (tensorWidthRemaining > 0) {
      pPixel[0] = DetensorizeValue(xChannel, nominalRangeConverter);
      pPixel[1] = DetensorizeValue(yChannel, nominalRangeConverter);
      pPixel[2] = DetensorizeValue(zChannel, nominalRangeConverter);
      pPixel[3] = 255;

      pPixel += 4;
      xChannel++;
      yChannel++;
      zChannel++;
      tensorWidthRemaining--;
    }
  }

#if defined(_M_AMD64) || defined(_M_IX86)
  template <>
  static void InterleaveRowFloatToByte(
    const float* xChannel,
    const float* yChannel,
    const float* zChannel,
    uint32_t tensorWidth,
    BYTE* pData,
    uint32_t bytesPerPixel,
    const NominalRangeConverter& nominalRangeConverter
  ) {
    BYTE* pPixel = pData;
    uint32_t tensorWidthRemaining = tensorWidth;

    __m128 maxv = _mm_set1_ps(255.0f);
    __m128 zero = _mm_setzero_ps();

    // Prep an alpha register with 8 bit - 255 alpha values
    __m128i alpha = _mm_setzero_si128();
    alpha = _mm_cmpeq_epi32(alpha, alpha);
    alpha = _mm_srli_epi16(alpha, 8);

    while (tensorWidthRemaining >= 8) {
      // Load, saturate, and convert to ints, 8 - 32 bit floats from X channel
      __m128i vXIntsLo = _mm_cvtps_epi32(_mm_min_ps(nominalRangeConverter.Denormalize(_mm_loadu_ps(xChannel)), maxv));
      __m128i vXIntsHi =
        _mm_cvtps_epi32(_mm_min_ps(nominalRangeConverter.Denormalize(_mm_loadu_ps(xChannel + 4)), maxv));

      // Pack 32 bit ints into 16 bit ints
      __m128i vXWords = _mm_packs_epi32(vXIntsLo, vXIntsHi);

      // Load, saturate, and convert to ints, 8 - 32 bit floats from Y channel
      __m128i vYIntsLo = _mm_cvtps_epi32(_mm_min_ps(nominalRangeConverter.Denormalize(_mm_loadu_ps(yChannel)), maxv));
      __m128i vYIntsHi =
        _mm_cvtps_epi32(_mm_min_ps(nominalRangeConverter.Denormalize(_mm_loadu_ps(yChannel + 4)), maxv));

      // Pack 32 bit ints into 16 bit ints
      __m128i vYWords = _mm_packs_epi32(vYIntsLo, vYIntsHi);

      // Load, saturate, and convert to ints, 8 - 32 bit floats from Z channel
      __m128i vZIntsLo = _mm_cvtps_epi32(_mm_min_ps(nominalRangeConverter.Denormalize(_mm_loadu_ps(zChannel)), maxv));
      __m128i vZIntsHi =
        _mm_cvtps_epi32(_mm_min_ps(nominalRangeConverter.Denormalize(_mm_loadu_ps(zChannel + 4)), maxv));

      // Pack 32 bit ints into 16 bit ints
      __m128i vZWords = _mm_packs_epi32(vZIntsLo, vZIntsHi);

      // Pack 16 bit ints into 8 bit uints
      __m128i vXZBytes = _mm_packus_epi16(vXWords, vZWords);
      __m128i vYABytes = _mm_packus_epi16(vYWords, alpha);

      // Interleave bytes into XY order
      __m128i vXYBytesInterleaved = _mm_unpacklo_epi8(vXZBytes, vYABytes);
      // Interleave bytes into ZA order
      __m128i vZABytesInterleaved = _mm_unpackhi_epi8(vXZBytes, vYABytes);

      // Interleave 16 bits to get XYZA XYZA ordering
      __m128i vPixelBytesLo = _mm_unpacklo_epi16(vXYBytesInterleaved, vZABytesInterleaved);
      __m128i vPixelBytesHi = _mm_unpackhi_epi16(vXYBytesInterleaved, vZABytesInterleaved);

      // Write out bytes now in proper order
      _mm_storeu_si128((__m128i*)pPixel, vPixelBytesLo);
      _mm_storeu_si128((__m128i*)(pPixel + 16), vPixelBytesHi);

      xChannel += 8;
      yChannel += 8;
      zChannel += 8;
      pPixel += 8 * static_cast<uint64_t>(bytesPerPixel);
      tensorWidthRemaining -= 8;
    }

    // Anything remaining deal with it one at a time
    while (tensorWidthRemaining > 0) {
      pPixel[0] = DetensorizeValue(xChannel, nominalRangeConverter);
      pPixel[1] = DetensorizeValue(yChannel, nominalRangeConverter);
      pPixel[2] = DetensorizeValue(zChannel, nominalRangeConverter);
      pPixel[3] = 255;

      pPixel += static_cast<uint64_t>(bytesPerPixel);
      xChannel++;
      yChannel++;
      zChannel++;
      tensorWidthRemaining--;
    }
  }
#endif
};
}  // namespace _winml
