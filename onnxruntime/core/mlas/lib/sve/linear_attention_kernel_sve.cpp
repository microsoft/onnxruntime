/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    linear_attention_kernel_sve.cpp

Abstract:

    Driver and dispatch for the SVE linear (recurrent) attention kernel.

    This translation unit is plain C++: it needs no SVE compiler support and is
    built without -march=...+sve. Only the compute kernel carries that flag --
    see sve/linear_attention_sve_impl.cpp and cmake/onnxruntime_mlas.cmake. The
    dispatch is installed when MLAS_CPUIDINFO::HasArmSve() is true.

    The driver owns exactly three things the frozen compute kernel cannot do:

      * eligibility, and handing declined work back to the NEON kernel;
      * exp(g_t), because the frozen machine code may make no calls;
      * chunking the token loop so the decay staging buffer stays bounded.

    Everything else, including the decay pre-weighting that exp feeds, lives in
    the kernel where it is vectorized.

    Why the fallback is the NEON kernel and not the portable one: the dispatch
    structure has a single slot, so installing this dispatch REPLACES
    MlasLinearAttentionDispatchNeon. Falling back to the portable kernel would
    turn every declined shape into a regression against main.

--*/

#include "../mlasi.h"
#include "../linear_attention.h"

#include "linear_attention_sve.h"

#include <algorithm>
#include <cmath>

namespace
{

//
// Decay staging, in floats. exp() is evaluated by the driver into this buffer
// for a run of tokens at a time; the kernel then reads it with a contiguous
// KHeadSize stride. 2048 floats is 8 KB, which holds 64 tokens at d_k = 32 and
// 8 at d_k = 256.
//
constexpr size_t MlasLinearAttentionSveDecayChunkFloats = 2048;

typedef void(LinearAttentionSveHeadFn)(const MLAS_LINEAR_ATTENTION_SVE_CHUNK*);

//
// if/else rather than a table or a switch: this is plain C++ so a jump table
// would be harmless here, but keeping the shape identical to the frozen
// kernel's own dispatch makes the two easy to compare.
//
LinearAttentionSveHeadFn*
SelectHeadFn(size_t n_out)
{
    if (n_out == 1) {
        return MlasLinearAttentionSveHeadN1Impl;
    }
    if (n_out == 2) {
        return MlasLinearAttentionSveHeadN2Impl;
    }
    if (n_out == 4) {
        return MlasLinearAttentionSveHeadN4Impl;
    }
    return MlasLinearAttentionSveHeadN8Impl;
}

}  // namespace

void
MlasLinearAttentionProcessHeadSve(
    const MLAS_LINEAR_ATTENTION_WORK* Work
)
{
    const size_t d_k = Work->KHeadSize;
    const size_t d_v = Work->VHeadSize;
    const size_t n_out = Work->HeadsPerGroup;

    //
    // d_k bounds the pre-weighting buffers the kernel holds on the stack. There
    // is deliberately no constraint on d_v: SVE predication makes a partial
    // lane cost the same as a full one, so the panel loops need no tail and the
    // d_v % 32 restriction the NEON kernel carries has no analogue here. Nor is
    // there a d_k % 4 constraint -- that one exists on NEON only because its
    // q.k tail steps by four.
    //
    const bool shape_ok = (d_k >= 1) && (d_k <= MlasLinearAttentionSveMaxKHeadSize);
    const bool n_out_ok = (n_out == 1 || n_out == 2 || n_out == 4 || n_out == 8);

    //
    // No vector-length condition. This kernel ran behind the NEON one at
    // VL=128 for as long as it applied one weight per LD1RW broadcast, and an
    // earlier version of this comment justified a routing gate by claiming SVE
    // has no scalar-operand FMA. That claim was wrong: SVE's FMLA/FMUL
    // (indexed) take the weight from a lane of a quad, which is the same
    // mechanism NEON's vfmaq_n_f32 uses. The compute kernel now uses it, so at
    // a 128-bit vector length -- where a Z register is exactly a Q register --
    // this kernel does the same work with the same instruction mix as the NEON
    // one, and there is no vector length at which it should be declined.
    //
    // Groups larger than one were never in question at any vector length: the
    // NEON kernel declines them outright, so the alternative is the portable
    // path.
    //
    if (!shape_ok || !n_out_ok) {
        MlasLinearAttentionProcessHeadNeon(Work);
        return;
    }

    const bool needs_decay = (Work->Rule == MlasLinearAttentionRuleGated ||
                              Work->Rule == MlasLinearAttentionRuleGatedDelta);
    const bool needs_beta = (Work->Rule == MlasLinearAttentionRuleDelta ||
                             Work->Rule == MlasLinearAttentionRuleGatedDelta);
    const bool decay_per_key_dim = (Work->DecayLayout == MlasLinearAttentionDecayPerKeyDim);

    LinearAttentionSveHeadFn* Fn = SelectHeadFn(n_out);

    //
    // Without decay the whole sequence is one chunk and the kernel is entered
    // once per head; the per-call cost is then irrelevant. With decay the chunk
    // is bounded by the staging buffer.
    //
    float decay_chunk[MlasLinearAttentionSveDecayChunkFloats];

    const size_t tokens_per_chunk =
        needs_decay ? std::max<size_t>(1, MlasLinearAttentionSveDecayChunkFloats / d_k)
                    : Work->SequenceLength;

    for (size_t t0 = 0; t0 < Work->SequenceLength; t0 += tokens_per_chunk) {
        const size_t count = std::min(tokens_per_chunk, Work->SequenceLength - t0);

        if (needs_decay) {
            //
            // The only part of the recurrence that cannot live in the frozen
            // kernel. MlasComputeExp routes through the platform dispatch in an
            // SVE build, so this is the SVE exp kernel -- itself already frozen
            // in elementwise_sve_asm.S -- rather than a second copy of the same
            // polynomial. The per-head layout is splatted to d_k so the kernel
            // sees one uniform layout.
            //
            for (size_t u = 0; u < count; ++u) {
                const float* gt = Work->Decay + (t0 + u) * Work->DecayTokenStride;
                float* d = decay_chunk + u * d_k;

                if (decay_per_key_dim) {
                    MlasComputeExp<float>(gt, d, d_k);
                } else {
                    std::fill_n(d, d_k, std::exp(gt[0]));
                }
            }
        }

        MLAS_LINEAR_ATTENTION_SVE_CHUNK chunk;

        chunk.State = Work->State;
        chunk.Query = Work->Query + t0 * Work->QueryTokenStride;
        chunk.Key = Work->Key + t0 * Work->KeyTokenStride;
        chunk.Value = Work->Value + t0 * Work->ValueTokenStride;
        chunk.Output = Work->Output + t0 * Work->OutputTokenStride;
        chunk.Decay = needs_decay ? decay_chunk : nullptr;
        chunk.Beta = needs_beta ? (Work->Beta + t0 * Work->BetaTokenStride) : nullptr;

        chunk.QueryTokenStride = Work->QueryTokenStride;
        chunk.KeyTokenStride = Work->KeyTokenStride;
        chunk.ValueTokenStride = Work->ValueTokenStride;
        chunk.BetaTokenStride = Work->BetaTokenStride;
        chunk.OutputTokenStride = Work->OutputTokenStride;

        chunk.TokenCount = count;
        chunk.KHeadSize = d_k;
        chunk.VHeadSize = d_v;
        chunk.Scale = Work->Scale;

        Fn(&chunk);
    }
}

//
// Kernel dispatch structure definition.
//
const MLAS_LINEAR_ATTENTION_DISPATCH MlasLinearAttentionDispatchSve = {
    MlasLinearAttentionProcessHeadSve
};
