// Copyright (c) 2026 Arm Limited. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(DISABLE_FLOAT8_TYPES)

#include "core/common/float8.h"
#include "core/mlas/inc/mlas.h"

#include <cstdint>

namespace onnxruntime {

inline float Fp8ByteToFloat(uint8_t value, mlas_fp8_mode mode) {
  switch (mode) {
    case MLAS_FP8_MODE_E4M3_INF:
      return Float8E4M3FN(value, Float8E4M3FN::FromBits()).ToFloat();
    case MLAS_FP8_MODE_E4M3_SAT:
      return Float8E4M3FNUZ(value, Float8E4M3FNUZ::FromBits()).ToFloat();
    case MLAS_FP8_MODE_E5M2_INF:
      return Float8E5M2(value, Float8E5M2::FromBits()).ToFloat();
    case MLAS_FP8_MODE_E5M2_SAT:
      return Float8E5M2FNUZ(value, Float8E5M2FNUZ::FromBits()).ToFloat();
    default:
      return 0.0f;
  }
}

inline uint8_t FloatToFp8Byte(float value, mlas_fp8_mode mode) {
  switch (mode) {
    case MLAS_FP8_MODE_E4M3_INF: {
      const Float8E4M3FN fp8(value, true);
      return fp8.val;
    }
    case MLAS_FP8_MODE_E4M3_SAT: {
      const Float8E4M3FNUZ fp8(value, true);
      return fp8.val;
    }
    case MLAS_FP8_MODE_E5M2_INF: {
      const Float8E5M2 fp8(value, true);
      return fp8.val;
    }
    case MLAS_FP8_MODE_E5M2_SAT: {
      const Float8E5M2FNUZ fp8(value, true);
      return fp8.val;
    }
    default:
      return 0;
  }
}

inline bool IsValidFp8Mode(mlas_fp8_mode mode) {
  switch (mode) {
    case MLAS_FP8_MODE_E4M3_INF:
    case MLAS_FP8_MODE_E4M3_SAT:
    case MLAS_FP8_MODE_E5M2_INF:
    case MLAS_FP8_MODE_E5M2_SAT:
      return true;
    default:
      return false;
  }
}

}  // namespace onnxruntime

#endif  // !defined(DISABLE_FLOAT8_TYPES)
