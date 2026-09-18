/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    linear_attention_sve_impl.cpp

Abstract:

    SVE intrinsics reference implementation of the linear (recurrent) attention
    compute kernels, and the regeneration source for the portable machine-code
    variant (aarch64/linear_attention_sve_asm.S, script: sve/gen_sve_asm.py).

    This is the only LinearAttention translation unit that requires an
    SVE-capable compiler (-march=armv8.2-a+sve); the driver is plain C++. The
    exported extern "C" symbols match the generated assembly exactly, so the
    build can link either implementation.

    Self-containment contract (verified by the generator): the kernels make no
    calls, reference no global data and use no literal pools -- every input
    arrives through the parameter block, so the frozen machine code is fully
    position-independent. This is why exp() is the driver's job: it would be a
    call, and the generator hard-fails on bl/blr.

    Regenerate from onnxruntime/core/mlas/lib with:

      python3 sve/gen_sve_asm.py \
          --src sve/linear_attention_sve_impl.cpp \
          --out aarch64/linear_attention_sve_asm.S \
          --march armv8.2-a+sve \
          --symbols MlasLinearAttentionSveVectorBytes,MlasLinearAttentionSveHeadN1Impl,MlasLinearAttentionSveHeadN2Impl,MlasLinearAttentionSveHeadN4Impl,MlasLinearAttentionSveHeadN8Impl \
          --module linear_attention_sve \
          --cflag=-fno-jump-tables

    -fno-jump-tables must also be on the intrinsics build in cmake, or the
    ASM=ON and ASM=OFF paths would be built from differently-compiled code.

--*/

#include "linear_attention_sve.h"

#include <arm_sve.h>

// Deliberately self-sufficient (no mlasi.h): the smaller the translation unit,
// the simpler the frozen object is to verify. This TU is only ever compiled by
// an SVE-capable gcc/clang.
#if !defined(MLAS_FORCEINLINE)
#define MLAS_FORCEINLINE __attribute__((always_inline)) inline
#endif

#include "linear_attention_asm_sve.h"
#include "linear_attention_compute_sve.h"

extern "C" {

size_t
MlasLinearAttentionSveVectorBytes(void)
{
    return svcntb();
}

void
MlasLinearAttentionSveHeadN1Impl(
    const MLAS_LINEAR_ATTENTION_SVE_CHUNK* Chunk
    )
{
    MlasLinearAttentionSveHead<1>(Chunk);
}

void
MlasLinearAttentionSveHeadN2Impl(
    const MLAS_LINEAR_ATTENTION_SVE_CHUNK* Chunk
    )
{
    MlasLinearAttentionSveHead<2>(Chunk);
}

void
MlasLinearAttentionSveHeadN4Impl(
    const MLAS_LINEAR_ATTENTION_SVE_CHUNK* Chunk
    )
{
    MlasLinearAttentionSveHead<4>(Chunk);
}

void
MlasLinearAttentionSveHeadN8Impl(
    const MLAS_LINEAR_ATTENTION_SVE_CHUNK* Chunk
    )
{
    MlasLinearAttentionSveHead<8>(Chunk);
}

}
