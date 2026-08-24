/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    linear_attention_kernel_avx512f.cpp

Abstract:

    This module implements the AVX-512 linear (recurrent) attention kernel.

    The portable kernel walks the state matrix S once per step of the
    recurrence - decay, retrieval, rank-1 update and readout - which is roughly
    4 reads and 2 writes of S per token. This kernel fuses all four into two
    passes (one read, one write) using the identity below, and processes S one
    column panel at a time so the second pass re-reads the panel from L1.

    With dec[i] = exp(g_t[i]) (or 1 when the rule has no decay), and the
    pre-weighted vectors k'[i] = dec[i]*k[i] and q'_g[i] = dec[i]*q_g[i]:

        retrieved[j] = sum_i (k'[i] * S_old[i,j])
        upd[j]       = beta * (v[j] - retrieved[j])     (= v[j] with no beta)
        S_new[i,j]   = dec[i] * S_old[i,j] + k[i] * upd[j]
        o_g[j]       = scale * ( sum_i (q'_g[i] * S_old[i,j]) + (q_g . k) * upd[j] )

    The last line is the load-bearing one: the readout is expressible from
    S_old plus a rank-1 correction, so it never needs the written-back S_new.
    Note the rank-1 term sits outside the sum. Pass 1 accumulates `retrieved`
    and every head's a_g = sum_i (q'_g[i] * S_old[i,:]) from a single read of
    S_old; pass 2 writes S_new in place.

    Both kernels are templated on HAS_DECAY. Without decay the pre-weighting
    disappears (k' = k, q' = q, so the kernels read the input vectors directly
    and copy nothing), and pass 2 collapses from a multiply plus an FMA to a
    single FMA. That matters: for the `linear` rule, which is the one the
    portable path handles in the fewest passes, the redundant multiply by 1 was
    enough to lose to a tuned SGEMM.

    This reassociates the floating-point sums relative to the portable kernel,
    so results agree to tolerance rather than bit-exactly.

--*/

#include "linear_attention.h"

#include <immintrin.h>
#include <cmath>

//
// With decay, the kernels hold the pre-weighted vectors on the stack, which is
// what bounds KHeadSize. NOUT <= 8 puts the worst case at 9 * 256 * 4 = 9 KB.
//
constexpr size_t MlasLinearAttentionAvx512MaxKHeadSize = 256;

namespace {

//
// Single readout head (MHA, or inverse GQA where one query head is shared).
//
// Requires d_k % 16 == 0, d_k <= 256, d_v % 32 == 0. Accumulators stay in ZMM
// registers across both passes; nothing here relies on auto-vectorization,
// which previously spilled the pass-1 accumulators and emitted a serial scalar
// horizontal reduction for q.k.
//
template <bool HAS_DECAY>
MLAS_FORCEINLINE
void
FusedTokenN1Avx512(
    float* __restrict S,
    size_t d_k,
    size_t d_v,
    const float* __restrict decvec,
    const float* __restrict kt,
    const float* __restrict vt,
    const float* __restrict q0,
    float* __restrict o0,
    float scale,
    float beta_val,
    bool needs_beta,
    bool needs_retrieval
)
{
    //
    // Staging for the decay-weighted vectors. Without decay the kernel reads q0
    // and kt directly, so the buffers are never written and are sized away.
    //
    constexpr size_t StageK = HAS_DECAY ? MlasLinearAttentionAvx512MaxKHeadSize : 1;
    float wkv_buf[StageK];
    float wqv_buf[StageK];

    const float* __restrict wqv = q0;
    const float* __restrict wkv = kt;

    if constexpr (HAS_DECAY) {
        for (size_t i = 0; i < d_k; ++i) {
            wqv_buf[i] = decvec[i] * q0[i];
        }
        wqv = wqv_buf;
        if (needs_retrieval) {
            for (size_t i = 0; i < d_k; ++i) {
                wkv_buf[i] = decvec[i] * kt[i];
            }
            wkv = wkv_buf;
        }
    }

    //
    // qk = q0 . k, accumulated in a vector with a single horizontal reduce.
    //
    __m512 qkacc = _mm512_setzero_ps();
    for (size_t i = 0; i < d_k; i += 16) {
        qkacc = _mm512_fmadd_ps(_mm512_loadu_ps(q0 + i), _mm512_loadu_ps(kt + i), qkacc);
    }

    const __m512 sc = _mm512_set1_ps(scale);
    const __m512 qkv = _mm512_set1_ps(_mm512_reduce_add_ps(qkacc));
    const __m512 b = _mm512_set1_ps(beta_val);

    //
    // One 32-column panel of S at a time: pass 1 -> upd -> output -> pass 2,
    // all while the panel (d_k x 32) is hot in L1, so pass 2 re-reads it from
    // L1 rather than L2 and the retrieved / a / upd intermediates never leave
    // ZMM registers.
    //
    for (size_t j0 = 0; j0 < d_v; j0 += 32) {
        __m512 r0 = _mm512_setzero_ps();
        __m512 r1 = _mm512_setzero_ps();
        __m512 a0 = _mm512_setzero_ps();
        __m512 a1 = _mm512_setzero_ps();

        if (needs_retrieval) {
            for (size_t i = 0; i < d_k; ++i) {
                const __m512 wk = _mm512_set1_ps(wkv[i]);
                const __m512 wq = _mm512_set1_ps(wqv[i]);
                const float* __restrict Si = S + i * d_v + j0;
                const __m512 s0 = _mm512_loadu_ps(Si);
                const __m512 s1 = _mm512_loadu_ps(Si + 16);
                r0 = _mm512_fmadd_ps(wk, s0, r0);
                r1 = _mm512_fmadd_ps(wk, s1, r1);
                a0 = _mm512_fmadd_ps(wq, s0, a0);
                a1 = _mm512_fmadd_ps(wq, s1, a1);
            }
        } else {
            for (size_t i = 0; i < d_k; ++i) {
                const __m512 wq = _mm512_set1_ps(wqv[i]);
                const float* __restrict Si = S + i * d_v + j0;
                a0 = _mm512_fmadd_ps(wq, _mm512_loadu_ps(Si), a0);
                a1 = _mm512_fmadd_ps(wq, _mm512_loadu_ps(Si + 16), a1);
            }
        }

        __m512 u0;
        __m512 u1;
        if (needs_beta) {
            u0 = _mm512_mul_ps(b, _mm512_sub_ps(_mm512_loadu_ps(vt + j0), r0));
            u1 = _mm512_mul_ps(b, _mm512_sub_ps(_mm512_loadu_ps(vt + j0 + 16), r1));
        } else {
            u0 = _mm512_loadu_ps(vt + j0);
            u1 = _mm512_loadu_ps(vt + j0 + 16);
        }

        _mm512_storeu_ps(o0 + j0, _mm512_mul_ps(sc, _mm512_fmadd_ps(qkv, u0, a0)));
        _mm512_storeu_ps(o0 + j0 + 16, _mm512_mul_ps(sc, _mm512_fmadd_ps(qkv, u1, a1)));

        for (size_t i = 0; i < d_k; ++i) {
            const __m512 kk = _mm512_set1_ps(kt[i]);
            float* __restrict Si = S + i * d_v + j0;
            if constexpr (HAS_DECAY) {
                const __m512 dc = _mm512_set1_ps(decvec[i]);
                _mm512_storeu_ps(Si,
                                 _mm512_fmadd_ps(dc, _mm512_loadu_ps(Si), _mm512_mul_ps(kk, u0)));
                _mm512_storeu_ps(Si + 16,
                                 _mm512_fmadd_ps(dc, _mm512_loadu_ps(Si + 16), _mm512_mul_ps(kk, u1)));
            } else {
                _mm512_storeu_ps(Si, _mm512_fmadd_ps(kk, u0, _mm512_loadu_ps(Si)));
                _mm512_storeu_ps(Si + 16, _mm512_fmadd_ps(kk, u1, _mm512_loadu_ps(Si + 16)));
            }
        }
    }
}

//
// Standard GQA: NOUT query heads share this state. Direct generalization of
// FusedTokenN1Avx512 - one read of each S_old panel feeds `retrieved` and all
// NOUT heads' a_g accumulators (1 + NOUT FMAs per S load), then a shared upd,
// per-head output, and the shared in-place pass 2.
//
// Templated on NOUT so the per-head accumulators have compile-time indices and
// stay in registers; a runtime loop would spill them. NLANE is the panel width
// in ZMM registers. Live registers are NLANE * (1 + NOUT) accumulators plus
// NLANE loads - AVX-512 embedded broadcast makes the wkv/wqv multipliers memory
// operands, so the weights cost no registers.
//
// Requires d_k % 16 == 0, d_k <= 256, d_v % (NLANE * 16) == 0.
//
template <int NOUT, int NLANE, bool HAS_DECAY>
MLAS_FORCEINLINE
void
FusedTokenGQAAvx512(
    float* __restrict S,
    size_t d_k,
    size_t d_v,
    const float* __restrict decvec,
    const float* __restrict kt,
    const float* __restrict vt,
    const float* __restrict q_base,
    float* __restrict o_base,
    float scale,
    float beta_val,
    bool needs_beta,
    bool needs_retrieval
)
{
    constexpr size_t MaxK = MlasLinearAttentionAvx512MaxKHeadSize;
    constexpr size_t PW = NLANE * 16;  // panel width in floats

    //
    // Staging for the decay-weighted vectors, sized away without decay: the
    // kernel then reads the query rows and kt directly and copies nothing.
    //
    float wkv_buf[HAS_DECAY ? MaxK : 1];
    float wqv_buf[HAS_DECAY ? NOUT * MaxK : 1];

    //
    // Without decay the per-head weights are just the query rows themselves,
    // so nothing is copied. NOUT is a compile-time bound, so this array of
    // pointers stays in registers.
    //
    const float* wqv[NOUT];
    const float* __restrict wkv = kt;

    for (int g = 0; g < NOUT; ++g) {
        const float* __restrict qg = q_base + static_cast<size_t>(g) * d_k;
        if constexpr (HAS_DECAY) {
            float* __restrict wq = wqv_buf + static_cast<size_t>(g) * MaxK;
            for (size_t i = 0; i < d_k; ++i) {
                wq[i] = decvec[i] * qg[i];
            }
            wqv[g] = wq;
        } else {
            wqv[g] = qg;
        }
    }

    if constexpr (HAS_DECAY) {
        if (needs_retrieval) {
            for (size_t i = 0; i < d_k; ++i) {
                wkv_buf[i] = decvec[i] * kt[i];
            }
            wkv = wkv_buf;
        }
    }

    float qk[NOUT];
    for (int g = 0; g < NOUT; ++g) {
        const float* __restrict qg = q_base + static_cast<size_t>(g) * d_k;
        __m512 acc = _mm512_setzero_ps();
        for (size_t i = 0; i < d_k; i += 16) {
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(qg + i), _mm512_loadu_ps(kt + i), acc);
        }
        qk[g] = _mm512_reduce_add_ps(acc);
    }

    const __m512 sc = _mm512_set1_ps(scale);
    const __m512 bb = _mm512_set1_ps(beta_val);

    for (size_t j0 = 0; j0 < d_v; j0 += PW) {
        __m512 r[NLANE];
        __m512 a[NOUT][NLANE];
        for (int l = 0; l < NLANE; ++l) {
            r[l] = _mm512_setzero_ps();
        }
        for (int g = 0; g < NOUT; ++g) {
            for (int l = 0; l < NLANE; ++l) {
                a[g][l] = _mm512_setzero_ps();
            }
        }

        for (size_t i = 0; i < d_k; ++i) {
            const float* __restrict Si = S + i * d_v + j0;
            __m512 s[NLANE];
            for (int l = 0; l < NLANE; ++l) {
                s[l] = _mm512_loadu_ps(Si + l * 16);
            }
            if (needs_retrieval) {
                const __m512 wk = _mm512_set1_ps(wkv[i]);
                for (int l = 0; l < NLANE; ++l) {
                    r[l] = _mm512_fmadd_ps(wk, s[l], r[l]);
                }
            }
            for (int g = 0; g < NOUT; ++g) {
                const __m512 wq = _mm512_set1_ps(wqv[g][i]);
                for (int l = 0; l < NLANE; ++l) {
                    a[g][l] = _mm512_fmadd_ps(wq, s[l], a[g][l]);
                }
            }
        }

        __m512 u[NLANE];
        for (int l = 0; l < NLANE; ++l) {
            const __m512 vv = _mm512_loadu_ps(vt + j0 + l * 16);
            u[l] = needs_beta ? _mm512_mul_ps(bb, _mm512_sub_ps(vv, r[l])) : vv;
        }

        for (int g = 0; g < NOUT; ++g) {
            const __m512 qkg = _mm512_set1_ps(qk[g]);
            float* __restrict og = o_base + static_cast<size_t>(g) * d_v + j0;
            for (int l = 0; l < NLANE; ++l) {
                _mm512_storeu_ps(og + l * 16,
                                 _mm512_mul_ps(sc, _mm512_fmadd_ps(qkg, u[l], a[g][l])));
            }
        }

        for (size_t i = 0; i < d_k; ++i) {
            const __m512 kk = _mm512_set1_ps(kt[i]);
            float* __restrict Si = S + i * d_v + j0;
            if constexpr (HAS_DECAY) {
                const __m512 dc = _mm512_set1_ps(decvec[i]);
                for (int l = 0; l < NLANE; ++l) {
                    _mm512_storeu_ps(Si + l * 16,
                                     _mm512_fmadd_ps(dc, _mm512_loadu_ps(Si + l * 16),
                                                     _mm512_mul_ps(kk, u[l])));
                }
            } else {
                for (int l = 0; l < NLANE; ++l) {
                    _mm512_storeu_ps(Si + l * 16,
                                     _mm512_fmadd_ps(kk, u[l], _mm512_loadu_ps(Si + l * 16)));
                }
            }
        }
    }
}

template <bool HAS_DECAY>
MLAS_FORCEINLINE
void
FusedTokenAvx512(
    size_t n_out,
    float* __restrict S,
    size_t d_k,
    size_t d_v,
    const float* __restrict decvec,
    const float* __restrict kt,
    const float* __restrict vt,
    const float* __restrict q_base,
    float* __restrict o_base,
    float scale,
    float beta_val,
    bool needs_beta,
    bool needs_retrieval
)
{
    switch (n_out) {
        case 1:
            FusedTokenN1Avx512<HAS_DECAY>(S, d_k, d_v, decvec, kt, vt, q_base, o_base,
                                          scale, beta_val, needs_beta, needs_retrieval);
            break;
        case 2:
            FusedTokenGQAAvx512<2, 2, HAS_DECAY>(S, d_k, d_v, decvec, kt, vt, q_base, o_base,
                                                 scale, beta_val, needs_beta, needs_retrieval);
            break;
        case 4:
            FusedTokenGQAAvx512<4, 2, HAS_DECAY>(S, d_k, d_v, decvec, kt, vt, q_base, o_base,
                                                 scale, beta_val, needs_beta, needs_retrieval);
            break;
        default:  // 8
            FusedTokenGQAAvx512<8, 2, HAS_DECAY>(S, d_k, d_v, decvec, kt, vt, q_base, o_base,
                                                 scale, beta_val, needs_beta, needs_retrieval);
            break;
    }
}

}  // namespace

void
MlasLinearAttentionProcessHeadAvx512F(
    const MLAS_LINEAR_ATTENTION_WORK* Work
)
{
    const size_t d_k = Work->KHeadSize;
    const size_t d_v = Work->VHeadSize;
    const size_t n_out = Work->HeadsPerGroup;

    //
    // The panel loops are unmasked, and the pre-weight buffers are fixed-size.
    // Anything outside that envelope goes to the portable kernel, which has no
    // shape restrictions. n_out == 16 is deliberately excluded: benchmarking
    // showed the 16-column single-lane variant it would need cannot match a
    // batched GEMM, unlike n_out in {1, 2, 4, 8}.
    //
    const bool shape_ok = (d_k % 16 == 0) &&
                          (d_k <= MlasLinearAttentionAvx512MaxKHeadSize) &&
                          (d_v % 32 == 0);
    const bool n_out_ok = (n_out == 1 || n_out == 2 || n_out == 4 || n_out == 8);

    if (!shape_ok || !n_out_ok) {
        MlasLinearAttentionProcessHead(Work);
        return;
    }

    const bool needs_decay = (Work->Rule == MlasLinearAttentionRuleGated ||
                              Work->Rule == MlasLinearAttentionRuleGatedDelta);
    const bool needs_beta = (Work->Rule == MlasLinearAttentionRuleDelta ||
                             Work->Rule == MlasLinearAttentionRuleGatedDelta);
    const bool needs_retrieval = needs_beta;
    const bool decay_per_key_dim = (Work->DecayLayout == MlasLinearAttentionDecayPerKeyDim);

    float* __restrict S = Work->State;
    float* __restrict decvec = Work->Scratch;  // d_k floats

    for (size_t t = 0; t < Work->SequenceLength; ++t) {
        const float* __restrict kt = Work->Key + t * Work->KeyTokenStride;
        const float* __restrict vt = Work->Value + t * Work->ValueTokenStride;

        const float beta_val = needs_beta ? Work->Beta[t * Work->BetaTokenStride] : 0.0f;

        const float* __restrict q_base = Work->Query + t * Work->QueryTokenStride;
        float* __restrict o_base = Work->Output + t * Work->OutputTokenStride;

        if (needs_decay) {
            //
            // Materialize the decay vector once per token, so the two passes
            // and the pre-weighting all read it instead of recomputing exp.
            //
            if (decay_per_key_dim) {
                const float* gt = Work->Decay + t * Work->DecayTokenStride;
                for (size_t i = 0; i < d_k; ++i) {
                    decvec[i] = std::exp(gt[i]);
                }
            } else {
                const float exp_g = std::exp(Work->Decay[t * Work->DecayTokenStride]);
                for (size_t i = 0; i < d_k; ++i) {
                    decvec[i] = exp_g;
                }
            }

            FusedTokenAvx512<true>(n_out, S, d_k, d_v, decvec, kt, vt, q_base, o_base,
                                   Work->Scale, beta_val, needs_beta, needs_retrieval);
        } else {
            FusedTokenAvx512<false>(n_out, S, d_k, d_v, nullptr, kt, vt, q_base, o_base,
                                    Work->Scale, beta_val, needs_beta, needs_retrieval);
        }
    }
}

const MLAS_LINEAR_ATTENTION_DISPATCH MlasLinearAttentionDispatchAvx512F = {
    MlasLinearAttentionProcessHeadAvx512F
};
