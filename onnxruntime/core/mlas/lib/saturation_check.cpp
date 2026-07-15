/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    saturation_check.cpp

Abstract:

    This module implements logic to check saturation of the VPMADDUBSW
    instruction.

--*/

#include "mlasi.h"

namespace onnxruntime
{

#if defined(MLAS_TARGET_AMD64)

std::atomic<int> saturation_count{0};

void
reset_saturation_count()
{
    saturation_count = 0;
}

#else

void
reset_saturation_count()
{
}

#endif

}  // namespace onnxruntime
