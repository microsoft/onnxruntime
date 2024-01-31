// Copyright (c) 2021, Apple Inc. All rights reserved.
//
// Use of this source code is governed by a BSD-3-clause license that can be
// found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

#include "MILBlob/Fp16.hpp"

// ORT_EDIT: Exclude clang specific pragmas from other builds
#if defined(__clang__)
// fp16 lib code has some conversion warnings we don't want to globally ignore
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wincompatible-pointer-types"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wconversion"
#include "fp16/fp16.h"
#pragma clang diagnostic pop
#else
#include "fp16/fp16.h"
#endif

using namespace MILBlob;

/* static */ Fp16 Fp16::FromFloat(float f) {
  return Fp16(fp16_ieee_from_fp32_value(f));
}

float Fp16::GetFloat() const {
  return fp16_ieee_to_fp32_value(bytes);
}

void Fp16::SetFloat(float f) {
  bytes = fp16_ieee_from_fp32_value(f);
}
