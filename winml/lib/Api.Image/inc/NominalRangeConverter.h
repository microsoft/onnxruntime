// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "inc/ImageConversionTypes.h"

namespace _winml {

class NominalRangeConverter {
 public:
  NominalRangeConverter() = delete;
  NominalRangeConverter(ImageNominalPixelRange pixelRange);

  float Normalize(float val) const;

  DirectX::PackedVector::HALF Normalize(DirectX::PackedVector::HALF val) const;

  __m128 Normalize(__m128 sse_data) const;

  float Denormalize(float val) const;

  DirectX::PackedVector::HALF Denormalize(DirectX::PackedVector::HALF val) const;

  __m128 Denormalize(__m128 sse_data) const;

 private:
  float scale;
  int32_t shift;
};
} // namespace _winml