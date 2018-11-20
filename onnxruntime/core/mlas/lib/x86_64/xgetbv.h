/*++

Copyright (c) Microsoft Corporation.  All rights reserved.

Licensed under the MIT License.

Module Name:

    xgetbv.h

Abstract:

    This module contains a wrapper for the XGETBV instruction for compilers
    lacking an intrinsic alternative.

--*/

#pragma once
// clang-format off

#if !defined(_XCR_XFEATURE_ENABLED_MASK)
#define _XCR_XFEATURE_ENABLED_MASK 0
#endif

inline
uint64_t
xgetbv(
    unsigned int ext_ctrl_reg
    )
{
    uint32_t eax, edx;

    __asm__
    (
        "xgetbv"
        : "=a" (eax), "=d" (edx)
        : "c" (ext_ctrl_reg)
    );

    return ((uint64_t)edx << 32) | eax;
}
