/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    platform.cpp

Abstract:

    This module implements logic to select the best configuration for the
    this platform.

--*/

#include "mlasi.h"

//
// Stores the platform information.
//

MLAS_PLATFORM MlasPlatform;

#ifdef MLAS_TARGET_AMD64_IX86

//
// Reads the processor extended control register to determine platform
// capabilities.
//

#if !defined(_XCR_XFEATURE_ENABLED_MASK)
#define _XCR_XFEATURE_ENABLED_MASK 0
#endif

inline
uint64_t
MlasReadExtendedControlRegister(
    unsigned int ext_ctrl_reg
    )
{
#if defined(_WIN32)
    return _xgetbv(ext_ctrl_reg);
#else
    uint32_t eax, edx;

    __asm__
    (
        "xgetbv"
        : "=a" (eax), "=d" (edx)
        : "c" (ext_ctrl_reg)
    );

    return ((uint64_t)edx << 32) | eax;
#endif
}

#endif

MLAS_PLATFORM::MLAS_PLATFORM(
    void
    )
/*++

Routine Description:

    This routine initializes the platform support for this library.

Arguments:

    None.

Return Value:

    None.

--*/
{

#if defined(MLAS_TARGET_AMD64_IX86)

    //
    // Default to the baseline SSE2 support.
    //

    this->KernelZeroRoutine = MlasSgemmKernelZeroSse;
    this->KernelAddRoutine = MlasSgemmKernelAddSse;
#if defined(MLAS_TARGET_AMD64)
    this->TransposePackB16x4Routine = MlasSgemmTransposePackB16x4Sse;
    this->LogisticKernelRoutine = MlasLogisticKernel;
    this->TanhKernelRoutine = MlasTanhKernel;
#endif

    //
    // Check if the processor supports the AVX and OSXSAVE features.
    //

    unsigned Cpuid1[4];
#if defined(_WIN32)
    __cpuid((int*)Cpuid1, 1);
#else
    __cpuid(1, Cpuid1[0], Cpuid1[1], Cpuid1[2], Cpuid1[3]);
#endif

    if ((Cpuid1[2] & 0x18000000) == 0x18000000) {

        //
        // Check if the operating system supports saving SSE and AVX states.
        //

        uint64_t xcr0 = MlasReadExtendedControlRegister(_XCR_XFEATURE_ENABLED_MASK);

        if ((xcr0 & 0x6) == 0x6) {

#if defined(MLAS_TARGET_IX86)

            this->KernelZeroRoutine = MlasSgemmKernelZeroAvx;
            this->KernelAddRoutine = MlasSgemmKernelAddAvx;

#else

            //
            // Check if the processor supports AVX512F (and the operating
            // system supports saving AVX512F state) or AVX2/FMA3 features.
            //

            unsigned Cpuid7[4];
#if defined(_WIN32)
            __cpuidex((int*)Cpuid7, 7, 0);
#else
            __cpuid_count(7, 0, Cpuid7[0], Cpuid7[1], Cpuid7[2], Cpuid7[3]);
#endif

            if (((Cpuid1[2] & 0x1000) != 0) && ((Cpuid7[1] & 0x20) != 0)) {

                if (((Cpuid7[1] & 0x10000) != 0) && ((xcr0 & 0xE0) == 0xE0)) {
                    this->KernelZeroRoutine = MlasSgemmKernelZeroAvx512F;
                    this->KernelAddRoutine = MlasSgemmKernelAddAvx512F;
                } else {
                    this->KernelZeroRoutine = MlasSgemmKernelZeroFma3;
                    this->KernelAddRoutine = MlasSgemmKernelAddFma3;
                }

                this->LogisticKernelRoutine = MlasLogisticKernelFma3;
                this->TanhKernelRoutine = MlasTanhKernelFma3;

            } else {

                this->KernelZeroRoutine = MlasSgemmKernelZeroAvx;
                this->KernelAddRoutine = MlasSgemmKernelAddAvx;
            }

            this->KernelM1Routine = MlasSgemmKernelM1Avx;
            this->KernelM1TransposeBRoutine = MlasSgemmKernelM1TransposeBAvx;
            this->TransposePackB16x4Routine = MlasSgemmTransposePackB16x4Avx;

#endif

        }
    }

#endif

}
