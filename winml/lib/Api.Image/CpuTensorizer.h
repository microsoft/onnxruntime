// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "inc/ImageConversionTypes.h"
#include "inc/NominalRangeConverter.h"

namespace _winml {

class CpuTensorizer {
 public:
  template <typename T>
  static HRESULT TensorizeData(
    _In_ ImageTensorChannelType formatFrom,
    _In_ ImageTensorChannelType formatTo,
    _In_ winml::LearningModelPixelRange pixelRange,
    _In_ BYTE* pBuffer,
    _In_ UINT32 bufferWidth,
    _In_ const wgi::BitmapBounds& inputBounds,
    _Inout_ T* pCPUTensor
  ) {
#pragma warning(push)
#pragma warning(disable : 26014 \
)  // warning about possible out of bounds accesing pData, but input is checked for BGRA8 format, so uiCapacity should be in multiples of 4 \
    // input is BGRA8: so blue at i, green is at i + 1, red is at i + 2

    uint32_t bytesPerPixel = formatFrom == kImageTensorChannelTypeGRAY8 ? 1 : 4;

    // bufferWidth may have padding because of optimization, but bytesPerRow includes only the real tensor data. We need to jump
    // over bufferWidth's extra padding
    uint32_t bytesPerRow = inputBounds.Width * bytesPerPixel;
    uint32_t start = (inputBounds.Y * bufferWidth) + (inputBounds.X * bytesPerPixel);
    uint32_t end = start + bufferWidth * inputBounds.Height;
    uint32_t pixelInd = 0;

    uint32_t xElements = inputBounds.Width - inputBounds.X;
    uint32_t yElements = inputBounds.Height - inputBounds.Y;

    auto nominalRangeConverter = NominalRangeConverter(pixelRange);

    if (formatFrom == kImageTensorChannelTypeBGR8 && formatTo == kImageTensorChannelTypeBGR8 || formatFrom == kImageTensorChannelTypeRGB8 && formatTo == kImageTensorChannelTypeRGB8) {
      // Convert BGR8 -> BGR8 or RGB8 -> RGB8
      for (uint64_t y = 0; y < yElements; y++) {
        DeinterleaveRowByteToFloat(
          pBuffer + y * bufferWidth + start,
          pCPUTensor + y * inputBounds.Width,
          pCPUTensor + (inputBounds.Height * inputBounds.Width) + y * inputBounds.Width,
          pCPUTensor + (inputBounds.Height * inputBounds.Width) * 2 + y * inputBounds.Width,
          xElements,
          bytesPerPixel,
          nominalRangeConverter
        );
      }
    } else if (formatFrom == kImageTensorChannelTypeBGR8 && formatTo == kImageTensorChannelTypeRGB8 || formatFrom == kImageTensorChannelTypeRGB8 && formatTo == kImageTensorChannelTypeBGR8) {
      // Convert RGB8 -> BGR8 or BGR8 -> RGB8
      for (uint32_t y = 0; y < yElements; y++) {
        DeinterleaveRowByteToFloat(
          pBuffer + y * bufferWidth + start,
          pCPUTensor + (inputBounds.Height * inputBounds.Width) * 2 + y * inputBounds.Width,
          pCPUTensor + (inputBounds.Height * inputBounds.Width) + y * inputBounds.Width,
          pCPUTensor + y * inputBounds.Width,
          xElements,
          bytesPerPixel,
          nominalRangeConverter
        );
      }
    } else if (formatTo == kImageTensorChannelTypeGRAY8 && (formatFrom == kImageTensorChannelTypeBGR8 || formatFrom == kImageTensorChannelTypeRGB8)) {
      // Convert BGR8 -> GRAY8 or RGB8 -> GRAY8
      uint32_t blueIncrement = formatFrom == kImageTensorChannelTypeBGR8 ? 0 : 2;
      uint32_t redIncrement = formatFrom == kImageTensorChannelTypeBGR8 ? 2 : 0;

      for (UINT32 i = start; i < end; i += bufferWidth) {
        for (UINT32 j = i; j < i + bytesPerRow; j += bytesPerPixel) {
          float red = float(pBuffer[j + redIncrement]);
          float green = float(pBuffer[j + 1]);
          float blue = float(pBuffer[j + blueIncrement]);
          float gray = 0.2126f * red + 0.7152f * green + 0.0722f * blue;
          pCPUTensor[pixelInd] = ConvertByteToFloat<T>(static_cast<BYTE>(gray), nominalRangeConverter);
          pixelInd++;
        }
      }
    } else if (formatFrom == kImageTensorChannelTypeGRAY8 && (formatTo == kImageTensorChannelTypeBGR8 || formatTo == kImageTensorChannelTypeRGB8)) {
      // Convert GRAY8 -> BGR8 or GRAY8 -> RGB8
      for (UINT32 i = start; i < end; i += bufferWidth) {
        for (UINT32 j = i; j < i + bytesPerRow; j += bytesPerPixel) {
          pCPUTensor[pixelInd] = ConvertByteToFloat<T>(pBuffer[j], nominalRangeConverter);
          pCPUTensor[(inputBounds.Height * inputBounds.Width) + pixelInd] =
            ConvertByteToFloat<T>(pBuffer[j], nominalRangeConverter);
          pCPUTensor[(inputBounds.Height * inputBounds.Width * 2) + pixelInd] =
            ConvertByteToFloat<T>(pBuffer[j], nominalRangeConverter);
          pixelInd++;
        }
      }
    } else if (formatFrom == kImageTensorChannelTypeGRAY8 && formatTo == kImageTensorChannelTypeGRAY8) {
      // Convert GRAY8 -> GRAY8
      for (UINT32 i = start; i < end; i += bufferWidth) {
        for (UINT32 j = i; j < i + bytesPerRow; j += bytesPerPixel) {
          pCPUTensor[pixelInd] = ConvertByteToFloat<T>(pBuffer[j], nominalRangeConverter);
          pixelInd++;
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
  static T ConvertByteToFloat(const BYTE& input, const NominalRangeConverter& nominalRangeConverter);

  // clang-format off
  template <>
#if _MSVC_LANG < 202002L
  static
#endif
  float ConvertByteToFloat(const BYTE& input, const NominalRangeConverter& nominalRangeConverter) {
    return nominalRangeConverter.Normalize(static_cast<float>(input));
  }

  // clang-format off
  template <>
#if _MSVC_LANG < 202002L
  static
#endif
  DirectX::PackedVector::HALF ConvertByteToFloat(
    const BYTE& input,
    const NominalRangeConverter& nominalRangeConverter
  ) {
    return nominalRangeConverter.Normalize(DirectX::PackedVector::XMConvertFloatToHalf(input));
  }

  template <typename T>
  static void DeinterleaveRowByteToFloat(
    _In_ BYTE* pBuffer,
    _Inout_ T* xChannel,
    _Inout_ T* yChannel,
    _Inout_ T* zChannel,
    uint32_t pixelElements,
    uint32_t bytesPerPixel,
    const NominalRangeConverter& nominalRangeConverter
  ) {
    UINT32 j;

    for (j = 0; j < (pixelElements & 0xFFFFFFFC); j += 4) {
      xChannel[j] = ConvertByteToFloat<T>(pBuffer[0], nominalRangeConverter);
      yChannel[j] = ConvertByteToFloat<T>(pBuffer[1], nominalRangeConverter);
      zChannel[j] = ConvertByteToFloat<T>(pBuffer[2], nominalRangeConverter);
      xChannel[j + 1] = ConvertByteToFloat<T>(pBuffer[4], nominalRangeConverter);
      yChannel[j + 1] = ConvertByteToFloat<T>(pBuffer[5], nominalRangeConverter);
      zChannel[j + 1] = ConvertByteToFloat<T>(pBuffer[6], nominalRangeConverter);
      xChannel[j + 2] = ConvertByteToFloat<T>(pBuffer[8], nominalRangeConverter);
      yChannel[j + 2] = ConvertByteToFloat<T>(pBuffer[9], nominalRangeConverter);
      zChannel[j + 2] = ConvertByteToFloat<T>(pBuffer[10], nominalRangeConverter);
      xChannel[j + 3] = ConvertByteToFloat<T>(pBuffer[12], nominalRangeConverter);
      yChannel[j + 3] = ConvertByteToFloat<T>(pBuffer[13], nominalRangeConverter);
      zChannel[j + 3] = ConvertByteToFloat<T>(pBuffer[14], nominalRangeConverter);
      pBuffer += bytesPerPixel * 4;
    }

    for (; j < pixelElements; j++) {
      xChannel[j] = ConvertByteToFloat<T>(pBuffer[0], nominalRangeConverter);
      yChannel[j] = ConvertByteToFloat<T>(pBuffer[1], nominalRangeConverter);
      zChannel[j] = ConvertByteToFloat<T>(pBuffer[2], nominalRangeConverter);
      pBuffer += bytesPerPixel;
    }
  }

  // clang-format off
#if defined(_M_AMD64) || defined(_M_IX86)
  template <>
#if _MSVC_LANG < 202002L
  static
#endif
  void DeinterleaveRowByteToFloat(
    _In_ BYTE* pBuffer,
    _Inout_ float* xChannel,
    _Inout_ float* yChannel,
    _Inout_ float* zChannel,
    uint32_t pixelElements,
    uint32_t bytesPerPixel,
    const NominalRangeConverter& nominalRangeConverter
  ) {
    assert(bytesPerPixel == 4);

    __m128i ZeroVector = _mm_setzero_si128();
    while (pixelElements >= 8) {
      // Load 8 Pixels into 2 Registers
      // vBytes0 = X0 Y0 Z0 A0 X1 Y1...
      // vBytes0 = X4 Y4 Z4 A4 X2 Y2...
      __m128i vBytes0 = _mm_loadu_si128((__m128i*)pBuffer);
      __m128i vBytes1 = _mm_loadu_si128((__m128i*)(pBuffer + 16));

      // Shuffle to get
      // vi0 = X0 X4 Y0 Y4...A1 A5 (A is Alpha which is ignored)
      // vi1 = X2 X6 Y2 Y6...A2 A6
      __m128i vi0 = _mm_unpacklo_epi8(vBytes0, vBytes1);
      __m128i vi1 = _mm_unpackhi_epi8(vBytes0, vBytes1);

      // Shuffle again to get
      // vi0 = X0 X2 X4 X6...A4 A6 (All even byes)
      // vi1 = X1 X3 X5 X7...A3 A7 (All odd bytes)
      __m128i vi2 = _mm_unpacklo_epi8(vi0, vi1);
      __m128i vi3 = _mm_unpackhi_epi8(vi0, vi1);

      // Shuffle last time to get desired order
      // vi0 = X0 X1 X2 X3...Y6 Y7 (All even byes)
      // vi1 = Z0 Z1 Z2 Z3...A6 A7 (All odd bytes)
      __m128i vi4 = _mm_unpacklo_epi8(vi2, vi3);
      __m128i vi5 = _mm_unpackhi_epi8(vi2, vi3);

      // unpack with zeros to get 16 bit ints
      // vXWords = X0 X1...X6 X7
      __m128i vXWords = _mm_unpacklo_epi8(vi4, ZeroVector);

      // unpack again with zeros to get 32 bit ints
      __m128i vXIntsLo = _mm_unpacklo_epi16(vXWords, ZeroVector);
      __m128i vXIntsHi = _mm_unpackhi_epi16(vXWords, ZeroVector);

      // store 256 bits of X channel Floats
      _mm_storeu_ps(xChannel, nominalRangeConverter.Normalize(_mm_cvtepi32_ps(vXIntsLo)));
      _mm_storeu_ps(xChannel + 4, nominalRangeConverter.Normalize(_mm_cvtepi32_ps(vXIntsHi)));
      xChannel += 8;

      // unpack again for Y
      __m128i vYWords = _mm_unpackhi_epi8(vi4, ZeroVector);

      __m128i vYIntsLo = _mm_unpacklo_epi16(vYWords, ZeroVector);
      __m128i vYIntsHi = _mm_unpackhi_epi16(vYWords, ZeroVector);

      _mm_storeu_ps(yChannel, nominalRangeConverter.Normalize(_mm_cvtepi32_ps(vYIntsLo)));
      _mm_storeu_ps(yChannel + 4, nominalRangeConverter.Normalize(_mm_cvtepi32_ps(vYIntsHi)));
      yChannel += 8;

      // unpack again for Z
      __m128i vZWords = _mm_unpacklo_epi8(vi5, ZeroVector);

      __m128i vZIntsLo = _mm_unpacklo_epi16(vZWords, ZeroVector);
      __m128i vZIntsHi = _mm_unpackhi_epi16(vZWords, ZeroVector);

      _mm_storeu_ps(zChannel, nominalRangeConverter.Normalize(_mm_cvtepi32_ps(vZIntsLo)));
      _mm_storeu_ps(zChannel + 4, nominalRangeConverter.Normalize(_mm_cvtepi32_ps(vZIntsHi)));
      zChannel += 8;

      pBuffer += 32;
      pixelElements -= 8;
    }
    if (pixelElements >= 4) {
      // load 4 pixels = 16 values
      __m128i vBytes = _mm_loadu_si128((__m128i*)pBuffer);

      // unpack to 16 bits
      __m128i vWords0 = _mm_unpacklo_epi8(vBytes, ZeroVector);
      __m128i vWords1 = _mm_unpackhi_epi8(vBytes, ZeroVector);

      // unpack to 32 bits
      __m128i vInts0 = _mm_unpacklo_epi16(vWords0, ZeroVector);
      __m128i vInts1 = _mm_unpackhi_epi16(vWords0, ZeroVector);
      __m128i vInts2 = _mm_unpacklo_epi16(vWords1, ZeroVector);
      __m128i vInts3 = _mm_unpackhi_epi16(vWords1, ZeroVector);

      // Normalize to floats
      __m128 vFloats0 = _mm_cvtepi32_ps(vInts0);
      __m128 vFloats1 = _mm_cvtepi32_ps(vInts1);
      __m128 vFloats2 = _mm_cvtepi32_ps(vInts2);
      __m128 vFloats3 = _mm_cvtepi32_ps(vInts3);

      // We want have row but need cols so transpose 4x4 matrix
      _MM_TRANSPOSE4_PS(vFloats0, vFloats1, vFloats2, vFloats3);

      // Drop alpha channel transposed to vFloats3 write out rest
      _mm_storeu_ps(xChannel, nominalRangeConverter.Normalize(vFloats0));
      _mm_storeu_ps(yChannel, nominalRangeConverter.Normalize(vFloats1));
      _mm_storeu_ps(zChannel, nominalRangeConverter.Normalize(vFloats2));

      xChannel += 4;
      yChannel += 4;
      zChannel += 4;
      pBuffer += 4 * 4;
      pixelElements -= 4;
    }

    // Any remainder just do one at a time
    for (uint32_t j = 0; j < pixelElements; j++) {
      xChannel[j] = nominalRangeConverter.Normalize(static_cast<float>(pBuffer[0]));
      yChannel[j] = nominalRangeConverter.Normalize(static_cast<float>(pBuffer[1]));
      zChannel[j] = nominalRangeConverter.Normalize(static_cast<float>(pBuffer[2]));
      pBuffer += bytesPerPixel;
    }
  }
#endif
};
}  // namespace _winml
