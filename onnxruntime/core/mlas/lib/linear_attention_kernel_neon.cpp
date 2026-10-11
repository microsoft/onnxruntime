/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    linear_attention_kernel_neon.cpp

Abstract:

    This module implements the ARM NEON linear (recurrent) attention kernel.

    The portable kernel walks the state matrix S once per step of the
    recurrence - decay, retrieval, rank-1 update and readout - which is roughly
    4 reads and 2 writes of S per token. This kernel fuses all four steps, and
    processes S one column panel at a time so the panel stays hot in L1.

    With dec[i] = exp(g_t[i]) (or 1 when the rule has no decay):

        retrieved[j] = sum_i (dec[i]*k[i] * S_old[i,j])
        upd[j]       = beta * (v[j] - retrieved[j])     (= v[j] with no beta)
        S_new[i,j]   = dec[i] * S_old[i,j] + k[i] * upd[j]
        o[j]         = scale * sum_i (q[i] * S_new[i,j])

    Retrieval is what forces two passes: upd[j] depends on a reduction over all
    of S_old, so nothing downstream of it can be computed in the same
    traversal. That splits the rules into two shapes, and this kernel
    implements both rather than forcing one form on all four:

    * linear / gated (no beta, hence no retrieval). upd is just v, which is
      known up front, so each S_new element can be written and consumed by the
      readout in the same iteration - a SINGLE pass, 1 read and 1 write of S,
      the theoretical minimum. Reading S_new also means the readout weight is
      plain q[i], so this form needs no decay pre-weighting, no staging buffer
      and no q.k dot product at all.

    * delta / gated_delta (beta). Two passes, using the AVX-512 kernel's
      identity, which re-expresses the readout over S_old plus a rank-1
      correction so it never needs the written-back S_new:

          o[j] = scale * ( sum_i (dec[i]*q[i] * S_old[i,j]) + (q . k) * upd[j] )

      Pass 1 accumulates retrieved and the readout from one read of S_old;
      pass 2 re-reads the panel from L1 and writes S_new in place.

    linear_attention_kernel_avx512f.cpp uses the two-pass form for all four
    rules. Taking the single pass where it is available matters more here: on a
    128-bit vector a two-pass `linear` costs 2 loads, 2 FP ops and 1 store per
    element-vector, which is a dead tie with the portable path's update plus
    readout gemv, so it could only ever win on SGEMM call overhead. The
    single-pass form removes one of those loads outright.

    The other NEON-specific difference is register pressure. A 32-column panel
    is 8 float32x4_t rather than 2 __m512, so the two-pass live set is 8
    retrieval + 8 readout accumulators. That fits only because this kernel
    handles a single readout head; the GQA variant scales accumulators by the
    group size and will need a narrower panel.

    NEON also has no embedded broadcast, so the AVX-512 "set1 from memory"
    operand becomes vfmaq_n_f32, which takes the scalar directly and costs no
    vector register. At 8 lanes each scalar is amortized over 8 FMAs, so
    loading weights four at a time for vfmaq_laneq_f32 would buy nothing here.

    Only a single readout head (HeadsPerGroup == 1) is handled: MHA, and
    inverse GQA where one query head is shared. Larger groups fall back to the
    portable kernel.

    This reassociates the floating-point sums relative to the portable kernel,
    so results agree to tolerance rather than bit-exactly.

    This translation unit is baseline ARMv8-A ASIMD: no fp16, dot product, i8mm
    or SVE, so it needs no -march flag and no runtime feature check. cmake
    compiles it only for ARM64, hence no MLAS_TARGET_ARM64 guard here.

--*/

#include "linear_attention.h"

#include <arm_neon.h>
#include <cmath>
#include <utility>

//
// The two-pass kernel holds the decay-weighted vectors on the stack, which is
// what bounds KHeadSize. Two buffers of 256 floats is 2 KB.
//
constexpr size_t MlasLinearAttentionNeonMaxKHeadSize = 256;

//
// Column panel width, in floats. Eight float32x4_t lanes, matching the AVX-512
// kernel's 32-column panel: d_k x 32 floats is 16 KB at d_k = 128 and 32 KB at
// the d_k bound, so the panel stays inside a typical 64 KB L1 and the two-pass
// form re-reads it from there.
//
constexpr size_t MlasLinearAttentionNeonPanelWidth = 32;
constexpr size_t MlasLinearAttentionNeonLanes = MlasLinearAttentionNeonPanelWidth / 4;

namespace
{

//
// Compile-time loop unroll. The lane index must be a constant expression or the
// accumulator arrays below are addressed dynamically and gcc keeps them on the
// stack, reloading and restoring every accumulator on every iteration -- which
// turns the FMA-bound inner loop into a load/store-bound one. Same helper as
// qnbitgemm_kernel_neon.h, repeated here rather than including that header,
// which would pull in the whole QNBitGemm surface.
//
template <typename IterationFn, size_t... Indices>
MLAS_FORCEINLINE void
UnrolledLoopIterations(IterationFn&& f, std::index_sequence<Indices...> /* indices */)
{
    (f(Indices), ...);
}

template <size_t N, typename IterationFn>
MLAS_FORCEINLINE void
UnrolledLoop(IterationFn&& f)
{
    UnrolledLoopIterations(std::forward<IterationFn>(f), std::make_index_sequence<N>());
}


//
// q . k, accumulated in four vectors to keep the FMA pipeline full, then a
// single horizontal reduce. The tail loop covers d_k that is a multiple of 4
// but not of 16. Both operands are the raw query and key rows: the decay
// belongs to the sum term, not to this rank-1 coefficient.
//
MLAS_FORCEINLINE
float
LinearAttentionDotNeon(
    const float* __restrict q0,
    const float* __restrict kt,
    size_t d_k
)
{
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);

    size_t i = 0;

    for (; i + 16 <= d_k; i += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(q0 + i), vld1q_f32(kt + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(q0 + i + 4), vld1q_f32(kt + i + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(q0 + i + 8), vld1q_f32(kt + i + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(q0 + i + 12), vld1q_f32(kt + i + 12));
    }

    for (; i < d_k; i += 4) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(q0 + i), vld1q_f32(kt + i));
    }

    return vaddvq_f32(vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3)));
}

//
// linear / gated: one pass over S.
//
// upd is v, known before the traversal starts, so each S_new element is
// produced and consumed by the readout in the same iteration. S is read once
// and written once, and because the readout reads S_new the weight is plain
// q[i].
//
// Live registers: NLANE readout accumulators + NLANE holding v for the panel +
// the in-flight S value, i.e. ~17 of 32 at NLANE = 8.
//
template <bool HAS_DECAY>
MLAS_FORCEINLINE
void
FusedTokenSinglePassNeon(
    float* __restrict S,
    size_t d_k,
    size_t d_v,
    const float* __restrict decvec,
    const float* __restrict kt,
    const float* __restrict vt,
    const float* __restrict q0,
    float* __restrict o0,
    float scale
)
{
    constexpr size_t NLANE = MlasLinearAttentionNeonLanes;
    constexpr size_t PW = MlasLinearAttentionNeonPanelWidth;

    for (size_t j0 = 0; j0 < d_v; j0 += PW) {
        float32x4_t a[NLANE];
        float32x4_t vv[NLANE];

        UnrolledLoop<NLANE>([&](size_t l) {
            a[l] = vdupq_n_f32(0.0f);
            vv[l] = vld1q_f32(vt + j0 + l * 4);
        });

        for (size_t i = 0; i < d_k; ++i) {
            const float kk = kt[i];
            const float qq = q0[i];
            float* __restrict Si = S + i * d_v + j0;

            if constexpr (HAS_DECAY) {
                const float dc = decvec[i];
                UnrolledLoop<NLANE>([&](size_t l) {
                    const float32x4_t s =
                        vfmaq_n_f32(vmulq_n_f32(vv[l], kk), vld1q_f32(Si + l * 4), dc);
                    vst1q_f32(Si + l * 4, s);
                    a[l] = vfmaq_n_f32(a[l], s, qq);
                });
            } else {
                UnrolledLoop<NLANE>([&](size_t l) {
                    const float32x4_t s = vfmaq_n_f32(vld1q_f32(Si + l * 4), vv[l], kk);
                    vst1q_f32(Si + l * 4, s);
                    a[l] = vfmaq_n_f32(a[l], s, qq);
                });
            }
        }

        UnrolledLoop<NLANE>([&](size_t l) {
            vst1q_f32(o0 + j0 + l * 4, vmulq_n_f32(a[l], scale));
        });
    }
}

//
// delta / gated_delta: two passes over each panel.
//
// upd depends on the completed retrieval reduction, so pass 1 accumulates
// retrieved and the S_old readout from one read of the panel, and pass 2
// re-reads the panel from L1 to write S_new. The readout is closed with the
// (q . k) * upd rank-1 correction.
//
// Live registers in pass 1: NLANE retrieval + NLANE readout accumulators plus
// the in-flight S value, i.e. ~17 of 32 at NLANE = 8, with 16 independent FMA
// chains to cover the FMLA latency.
//
template <bool HAS_DECAY>
MLAS_FORCEINLINE
void
FusedTokenTwoPassNeon(
    float* __restrict S,
    size_t d_k,
    size_t d_v,
    const float* __restrict decvec,
    const float* __restrict kt,
    const float* __restrict vt,
    const float* __restrict q0,
    float* __restrict o0,
    float scale,
    float beta_val
)
{
    constexpr size_t NLANE = MlasLinearAttentionNeonLanes;
    constexpr size_t PW = MlasLinearAttentionNeonPanelWidth;

    //
    // Staging for the decay-weighted vectors. Without decay the kernel reads q0
    // and kt directly, so the buffers are never written and are sized away.
    //
    constexpr size_t StageK = HAS_DECAY ? MlasLinearAttentionNeonMaxKHeadSize : 1;
    float wkv_buf[StageK];
    float wqv_buf[StageK];

    const float* __restrict wqv = q0;
    const float* __restrict wkv = kt;

    if constexpr (HAS_DECAY) {
        for (size_t i = 0; i < d_k; ++i) {
            wqv_buf[i] = decvec[i] * q0[i];
            wkv_buf[i] = decvec[i] * kt[i];
        }
        wqv = wqv_buf;
        wkv = wkv_buf;
    }

    const float qk = LinearAttentionDotNeon(q0, kt, d_k);

    //
    // One panel at a time: pass 1 -> upd -> output -> pass 2, all while the
    // panel (d_k x PW) is hot in L1, so pass 2 re-reads it from L1 rather than
    // L2 and the retrieved / a / upd intermediates never leave registers.
    //
    for (size_t j0 = 0; j0 < d_v; j0 += PW) {
        float32x4_t r[NLANE];
        float32x4_t a[NLANE];

        UnrolledLoop<NLANE>([&](size_t l) {
            r[l] = vdupq_n_f32(0.0f);
            a[l] = vdupq_n_f32(0.0f);
        });

        for (size_t i = 0; i < d_k; ++i) {
            const float wk = wkv[i];
            const float wq = wqv[i];
            const float* __restrict Si = S + i * d_v + j0;
            UnrolledLoop<NLANE>([&](size_t l) {
                const float32x4_t s = vld1q_f32(Si + l * 4);
                r[l] = vfmaq_n_f32(r[l], s, wk);
                a[l] = vfmaq_n_f32(a[l], s, wq);
            });
        }

        float32x4_t u[NLANE];

        UnrolledLoop<NLANE>([&](size_t l) {
            const float32x4_t vv = vld1q_f32(vt + j0 + l * 4);
            u[l] = vmulq_n_f32(vsubq_f32(vv, r[l]), beta_val);
        });

        UnrolledLoop<NLANE>([&](size_t l) {
            vst1q_f32(o0 + j0 + l * 4, vmulq_n_f32(vfmaq_n_f32(a[l], u[l], qk), scale));
        });

        for (size_t i = 0; i < d_k; ++i) {
            const float kk = kt[i];
            float* __restrict Si = S + i * d_v + j0;
            if constexpr (HAS_DECAY) {
                const float dc = decvec[i];
                UnrolledLoop<NLANE>([&](size_t l) {
                    vst1q_f32(Si + l * 4,
                              vfmaq_n_f32(vmulq_n_f32(u[l], kk), vld1q_f32(Si + l * 4), dc));
                });
            } else {
                UnrolledLoop<NLANE>([&](size_t l) {
                    vst1q_f32(Si + l * 4,
                              vfmaq_n_f32(vld1q_f32(Si + l * 4), u[l], kk));
                });
            }
        }
    }
}

//
// Owns the whole token loop, so the rule dispatch and the staging buffers are
// resolved once per head rather than once per token.
//
template <bool HAS_DECAY, bool HAS_BETA>
void
ProcessHeadNeon(
    const MLAS_LINEAR_ATTENTION_WORK* Work
)
{
    const size_t d_k = Work->KHeadSize;
    const size_t d_v = Work->VHeadSize;

    const bool decay_per_key_dim = (Work->DecayLayout == MlasLinearAttentionDecayPerKeyDim);

    float* __restrict S = Work->State;
    float* __restrict decvec = Work->Scratch;  // d_k floats

    for (size_t t = 0; t < Work->SequenceLength; ++t) {
        const float* __restrict kt = Work->Key + t * Work->KeyTokenStride;
        const float* __restrict vt = Work->Value + t * Work->ValueTokenStride;
        const float* __restrict q0 = Work->Query + t * Work->QueryTokenStride;
        float* __restrict o0 = Work->Output + t * Work->OutputTokenStride;

        if constexpr (HAS_DECAY) {
            //
            // Materialize the decay vector once per token, so both the passes
            // and the pre-weighting read it instead of recomputing exp.
            //
            // The per-key-dim layout needs d_k exponentials, which is a real
            // cost next to a 128-bit-wide recurrence - at d_k = 128 a scalar
            // expf loop is a sizeable fraction of the token - so it goes
            // through the vectorized MlasComputeExp. On ARM64 that resolves
            // directly to the NEON MlasComputeExpF32Kernel. The per-head
            // layout is one exponential splatted across d_k, which keeps the
            // inner loops on a single uniform path.
            //
            const float* gt = Work->Decay + t * Work->DecayTokenStride;
            if (decay_per_key_dim) {
                MlasComputeExp<float>(gt, decvec, d_k);
            } else {
                const float exp_g = std::exp(gt[0]);
                for (size_t i = 0; i < d_k; ++i) {
                    decvec[i] = exp_g;
                }
            }
        }

        const float* __restrict dec = HAS_DECAY ? decvec : nullptr;

        if constexpr (HAS_BETA) {
            const float beta_val = Work->Beta[t * Work->BetaTokenStride];
            FusedTokenTwoPassNeon<HAS_DECAY>(S, d_k, d_v, dec, kt, vt, q0, o0,
                                             Work->Scale, beta_val);
        } else {
            FusedTokenSinglePassNeon<HAS_DECAY>(S, d_k, d_v, dec, kt, vt, q0, o0,
                                                Work->Scale);
        }
    }
}

}  // namespace

void
MlasLinearAttentionProcessHeadNeon(
    const MLAS_LINEAR_ATTENTION_WORK* Work
)
{
    const size_t d_k = Work->KHeadSize;
    const size_t d_v = Work->VHeadSize;

    //
    // The panel loops are unmasked, and the two-pass staging buffers are
    // fixed-size. Anything outside that envelope goes to the portable kernel,
    // which has no shape restrictions. Only a single readout head is handled
    // here; the GQA group sizes need per-group accumulators and a narrower
    // panel.
    //
    // d_k only has to be a multiple of 4 rather than the AVX-512 kernel's 16,
    // because the q . k reduction is 4 wide here, so this accepts shapes the
    // AVX-512 kernel declines.
    //
    const bool shape_ok = (d_k % 4 == 0) &&
                          (d_k <= MlasLinearAttentionNeonMaxKHeadSize) &&
                          (d_v % MlasLinearAttentionNeonPanelWidth == 0);

    if (!shape_ok || Work->HeadsPerGroup != 1) {
        MlasLinearAttentionProcessHead(Work);
        return;
    }

    switch (Work->Rule) {
        case MlasLinearAttentionRuleLinear:
            ProcessHeadNeon<false, false>(Work);
            return;
        case MlasLinearAttentionRuleGated:
            ProcessHeadNeon<true, false>(Work);
            return;
        case MlasLinearAttentionRuleDelta:
            ProcessHeadNeon<false, true>(Work);
            return;
        case MlasLinearAttentionRuleGatedDelta:
            ProcessHeadNeon<true, true>(Work);
            return;
    }

    //
    // Deliberately no default label above: -Wswitch turns a newly added rule
    // into a compile error here rather than silently routing it to one of the
    // existing specializations. A value outside the enum can still arrive at
    // runtime, so defer to the portable kernel rather than guess at its
    // semantics.
    //
    MlasLinearAttentionProcessHead(Work);
}

//
// Kernel dispatch structure definition.
//
const MLAS_LINEAR_ATTENTION_DISPATCH MlasLinearAttentionDispatchNeon = {
    .ProcessHead = MlasLinearAttentionProcessHeadNeon
};
