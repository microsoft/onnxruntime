/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    linear_attention.h

Abstract:

    This module includes the work-item description and kernel dispatch
    structure for the linear (recurrent) attention kernels.

--*/

#pragma once

#include "mlasi.h"

//
// One unit of linear-attention work: the complete token loop for a single
// (batch, kv head) pair, i.e. a single state matrix S.
//
// The threaded routine resolves every base pointer and stride before invoking
// a kernel, so a kernel performs no tensor index arithmetic and knows nothing
// about batching, GQA head mapping or key-head sharing.
//
struct MLAS_LINEAR_ATTENTION_WORK {
    //
    // State matrix S, [KHeadSize, VHeadSize] row-major with leading dimension
    // VHeadSize. Read and updated in place; never initialized by the kernel.
    //
    float* State;

    //
    // Token 0 of this (batch, head). Advance by the matching token stride to
    // reach token t. Query points at the first of HeadsPerGroup consecutive
    // query heads, Output at the first of HeadsPerGroup consecutive output
    // heads.
    //
    const float* Query;
    const float* Key;
    const float* Value;
    const float* Decay;  // nullptr when the rule carries no decay term
    const float* Beta;   // nullptr when the rule carries no beta term
    float* Output;

    size_t QueryTokenStride;
    size_t KeyTokenStride;
    size_t ValueTokenStride;
    size_t DecayTokenStride;  // 0 when Decay is nullptr
    size_t BetaTokenStride;   // 0 when Beta is nullptr
    size_t OutputTokenStride;

    size_t SequenceLength;
    size_t KHeadSize;      // d_k
    size_t VHeadSize;      // d_v
    size_t HeadsPerGroup;  // query heads read / output heads written per token

    float Scale;

    MLAS_LINEAR_ATTENTION_RULE Rule;

    //
    // Distinguishes a scalar broadcast (PerHead) from a per-row vector
    // (PerKeyDim). The beta layout needs no equivalent field: the base pointer
    // and BetaTokenStride already encode both cases.
    //
    MLAS_LINEAR_ATTENTION_DECAY_LAYOUT DecayLayout;

    //
    // At least MlasLinearAttentionBufferSizePerThread(KHeadSize, VHeadSize)
    // bytes of thread-private scratch.
    //
    float* Scratch;
};

/**
 * @brief Dispatch structure for linear-attention recurrence kernels.
 *
 * The dispatch granularity is one complete (batch, kv head) sequence: a kernel
 * owns the loop over all timesteps for a single state matrix. This is
 * deliberate - the state matrix is d_k * d_v floats (8-64 KB, L1/L2 resident),
 * so a vectorized kernel must be free to fuse decay, retrieval, update and
 * readout into a single streaming pass over it. Dispatching finer-grained
 * primitives would force one traversal of that working set per primitive.
 */
struct MLAS_LINEAR_ATTENTION_DISPATCH {
    typedef void(ProcessHead_Fn)(
        const MLAS_LINEAR_ATTENTION_WORK* Work
    );

    ProcessHead_Fn* ProcessHead = nullptr;
};

//
// Portable reference kernel. Registered as the baseline dispatch entry for
// every target, and used directly if no dispatch entry is available.
//
void
MlasLinearAttentionProcessHead(
    const MLAS_LINEAR_ATTENTION_WORK* Work
);
