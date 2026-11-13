/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    linear_attention_sve.h

Abstract:

    Prototypes and the parameter block for the SVE linear (recurrent) attention
    compute kernels.

    The symbols are extern "C" so the two interchangeable implementations link
    identically: the SVE intrinsics reference translation unit
    (linear_attention_sve_impl.cpp) and the generated KleidiAI-style
    machine-code variant (aarch64/linear_attention_sve_asm.S). The build links
    exactly one of the two. This header is includable from translation units
    compiled WITHOUT SVE support (the driver), on any platform, so it contains
    no SVE types and no intrinsics.

    The frozen machine code may reference no global data and may make no calls
    (see sve/gen_sve_asm.py), so everything the kernel needs arrives through the
    parameter block below. In particular exp() cannot be evaluated there, which
    is why Decay arrives already exponentiated.

--*/

#pragma once

#include <cstddef>

//
// Largest KHeadSize the SVE kernel accepts. It bounds the two-pass
// pre-weighting buffers, which the compute kernel holds on its stack: one
// vector of d_k for the key plus one per readout head, so 9 * 256 * 4 = 9 KB at
// the widest instantiation. Lives here rather than in the compute header
// because the driver's eligibility check needs it and must not include
// <arm_sve.h>.
//
constexpr size_t MlasLinearAttentionSveMaxKHeadSize = 256;

//
// One kernel invocation: a run of TokenCount consecutive tokens for a single
// state matrix S.
//
// A chunk rather than a single token because the frozen entry point costs a
// prologue plus a dozen argument loads; at the smallest benchmarked shape
// (d_k=32, d_v=64) a token is only a few hundred FMAs, so a per-token call
// would be a double-digit percentage of the work. When the rule carries no
// decay the driver passes the whole sequence as one chunk.
//
struct MLAS_LINEAR_ATTENTION_SVE_CHUNK {
    //
    // State matrix S, [KHeadSize, VHeadSize] row-major with leading dimension
    // VHeadSize. Read and updated in place; never initialized by the kernel.
    //
    float* State;

    //
    // Token 0 of this chunk. Query points at the first of HeadsPerGroup
    // consecutive query heads, Output at the first of HeadsPerGroup consecutive
    // output heads.
    //
    const float* Query;
    const float* Key;
    const float* Value;
    float* Output;

    //
    // nullptr when the rule carries no decay. Otherwise TokenCount * KHeadSize
    // floats, ALREADY EXPONENTIATED and laid out with a contiguous KHeadSize
    // stride by the driver -- exp() is a call, which the frozen kernel cannot
    // make. The per-head decay layout is splatted to KHeadSize by the driver so
    // the kernel has exactly one code path.
    //
    const float* Decay;

    //
    // nullptr when the rule carries no beta term.
    //
    const float* Beta;

    size_t QueryTokenStride;
    size_t KeyTokenStride;
    size_t ValueTokenStride;
    size_t BetaTokenStride;
    size_t OutputTokenStride;

    size_t TokenCount;
    size_t KHeadSize;   // d_k
    size_t VHeadSize;   // d_v

    float Scale;
};

extern "C" {

//
// svcntb(). Lets the plain-C++ driver make a vector-length dependent routing
// decision without itself being compiled for SVE.
//
size_t
MlasLinearAttentionSveVectorBytes(void);

//
// One entry point per readout-head count. Separate symbols rather than one
// routine with a runtime switch: the head count fixes the accumulator layout,
// which must be compile-time (SVE vectors cannot be array elements), and a
// switch would risk a jump table, which the freezer rejects.
//
// HasDecay is (Decay != nullptr) and HasBeta is (Beta != nullptr); both are
// hoisted out of the token loop by the kernel.
//
void
MlasLinearAttentionSveHeadN1Impl(
    const MLAS_LINEAR_ATTENTION_SVE_CHUNK* Chunk
    );

void
MlasLinearAttentionSveHeadN2Impl(
    const MLAS_LINEAR_ATTENTION_SVE_CHUNK* Chunk
    );

void
MlasLinearAttentionSveHeadN4Impl(
    const MLAS_LINEAR_ATTENTION_SVE_CHUNK* Chunk
    );

void
MlasLinearAttentionSveHeadN8Impl(
    const MLAS_LINEAR_ATTENTION_SVE_CHUNK* Chunk
    );

}
