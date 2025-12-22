/*++

Copyright 2025 FUJITSU LIMITED

Module Name:

   erf_neon_fp16.h

Abstract:

    This module contains the procedure prototypes for the ERF NEON FP16 intrinsics.

--*/

#pragma once

#include <arm_neon.h>

#include "mlasi.h"
#include "fp16_common.h"
#include "softmax_kernel_neon.h"

using _mlas_fp16_ = uint16_t;
void MlasNeonErfF16Kernel(const _mlas_fp16_* Input, _mlas_fp16_* Output, size_t N);
