/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    linear_attention_compute_sve.h

Abstract:

    SVE compute core for the linear (recurrent) attention kernels, shared by the
    four readout-head instantiations. This is the regeneration source for
    aarch64/linear_attention_sve_asm.S (script: sve/gen_sve_asm.py).

    With dec[i] = exp(g_t[i]) (or 1 when the rule has no decay):

        retrieved[j] = sum_i (dec[i]*k[i] * S_old[i,j])
        upd[j]       = beta * (v[j] - retrieved[j])     (= v[j] with no beta)
        S_new[i,j]   = dec[i] * S_old[i,j] + k[i] * upd[j]
        o_g[j]       = scale * sum_i (q_g[i] * S_new[i,j])

    Retrieval is what forces two passes: upd[j] depends on a reduction over all
    of S_old, so nothing downstream of it can be computed in the same traversal.
    That splits the rules into two shapes, as in the NEON kernel:

    * linear / gated (no beta). upd is v, known before the traversal, so each
      S_new element is written and consumed by the readout in the same
      iteration -- a SINGLE pass, one read and one write of S. Because the
      readout reads S_new the weight is plain q_g[i]: no pre-weighting, no
      staging, no q.k dot product.

    * delta / gated_delta (beta). Two passes over each column panel, using the
      identity that re-expresses the readout over S_old plus a rank-1
      correction, so it never needs the written-back S_new:

          o_g[j] = scale * ( sum_i (dec[i]*q_g[i] * S_old[i,j]) + (q_g.k)*upd[j] )

      Pass 1 accumulates retrieved and every head's readout from one read of
      S_old; pass 2 re-reads the panel from L1 and writes S_new in place.

    NOTE the two weight vectors are NOT interchangeable. Two-pass reads S_old,
    so its readout weight is dec[i]*q_g[i] and the rank-1 coefficient uses the
    RAW q and k. Single-pass reads S_new, so its readout weight is the RAW
    q_g[i] and there is no q.k term at all. Swapping them yields a result that
    is correct at TokenCount == 1 and wrong from the second token on.

    VECTOR-LENGTH AGNOSTIC. The number of Z registers per accumulator group
    (NLANE) is fixed at compile time; the column count follows svcntw(), so a
    panel is NLANE*4 columns at VL=128, NLANE*8 at VL=256 and NLANE*16 at
    VL=512. That factoring is forced rather than chosen: SVE vectors are
    sizeless and cannot be array elements, so the accumulator grid must be named
    locals, which means its extent must be a compile-time constant.

    TWO THINGS THE FIRST VERSION GOT WRONG, both found by classifying the
    generated inner loops rather than by reasoning:

    1. Address arithmetic, not broadcasts, dominated. Indexing the weights as
       wkv[i] and wq_buf[g*MaxK + i] -- with g coming from a macro expansion --
       defeated strength reduction, so every iteration recomputed the addresses
       (lsl + add + sub, twice) while gcc happily strength-reduced the S row
       pointer. That cost ~8 scalar ops against 4 FMLAs. Every stream the inner
       loop touches is now an induction pointer advanced by one element.

    2. Panels were narrower than they needed to be. Predicate registers are
       scarce -- Pg is three bits, so only p0-p7 exist -- and holding one per
       lane capped NLANE at 4. But only the LAST panel of a row can be partial:
       the full panels are all-true. Splitting the panel loop into a full body
       (one shared ptrue, no per-lane predicates) and a single trailing
       predicated body removes the cap from the hot path, which is what lets
       the single-head case run 8 lanes and match the NEON kernel's geometry.

    Everything here is base SVE -- no SVE2 -- because Neoverse V1 (Graviton3),
    the primary wide-vector target, implements SVE only.

--*/

#pragma once

#include <arm_sve.h>

#include "linear_attention_sve.h"

#if !defined(MLAS_FORCEINLINE)
#if defined(_MSC_VER)
#define MLAS_FORCEINLINE __forceinline
#else
#define MLAS_FORCEINLINE __attribute__((always_inline)) inline
#endif
#endif

#if !defined(MLAS_LA_RESTRICT)
#define MLAS_LA_RESTRICT __restrict
#endif

//
// Maximum accumulator grid. The generated code only ever touches NLANE lanes of
// NOUT heads; the rest are dead stores that the compiler removes. Declaring the
// full grid is what lets one template body serve every (NOUT, NLANE) pair
// without arrays of SVE types, which the language forbids.
//
#define MLAS_LA_SVE_MAX_LANES 8
#define MLAS_LA_SVE_MAX_HEADS 8

// Expand M(g, l) over the whole grid.
#define MLAS_LA_SVE_LANES(M, g) \
    M(g, 0) M(g, 1) M(g, 2) M(g, 3) M(g, 4) M(g, 5) M(g, 6) M(g, 7)
#define MLAS_LA_SVE_GRID(M)                                    \
    MLAS_LA_SVE_LANES(M, 0) MLAS_LA_SVE_LANES(M, 1)            \
    MLAS_LA_SVE_LANES(M, 2) MLAS_LA_SVE_LANES(M, 3)            \
    MLAS_LA_SVE_LANES(M, 4) MLAS_LA_SVE_LANES(M, 5)            \
    MLAS_LA_SVE_LANES(M, 6) MLAS_LA_SVE_LANES(M, 7)

// Expand M(l) over the lanes only, and M(g) over the heads only.
#define MLAS_LA_SVE_EACH_LANE(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)
#define MLAS_LA_SVE_EACH_HEAD(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)

#define MLAS_LA_SVE_ACC(g, l) acc##g##_##l

//
// Per-lane predicate. FULL panels are all-true, so they share one ptrue and
// spend no predicate registers; only the trailing partial panel pays.
//
#define MLAS_LA_SVE_PRED(l)                                                     \
    const svbool_t pg##l =                                                      \
        FULL ? svptrue_b32()                                                    \
             : svwhilelt_b32_u64(uint64_t(j0 + (l) * W), uint64_t(d_v));

//
// Z registers per accumulator group, by readout-head count.
//
// The binding constraint is the accumulator count: pass 1 of the two-pass form
// holds NLANE*(1+NOUT) of them, and the single-pass form holds NOUT*NLANE
// readouts plus NLANE for v plus NLANE in-flight -- which come to the same
// thing, so one table serves both. Against 32 Z registers that allows 8 lanes
// at a single head and 2 at eight heads.
//
// It is no longer capped by the predicate file: see note 2 in the header.
//
template <size_t NOUT>
struct MlasLinearAttentionSveLanes {
    static constexpr size_t Max = (NOUT == 1) ? 8 : (NOUT <= 4) ? 4 : 2;
};

//
// Halve the panel when it would not fit in roughly half of L1, which the second
// pass depends on re-reading it from.
//
MLAS_FORCEINLINE
size_t
MlasLinearAttentionSveLaneCount(
    size_t NLaneMax,
    size_t d_k
)
{
    const size_t Words = svcntw();
    size_t n = NLaneMax;

    // Integer arithmetic only: a float comparison here would risk a literal
    // pool, which the freezer rejects. 8192 floats is 32 KB.
    while (n > 1 && n * Words * d_k > 8192) {
        n >>= 1;
    }

    return n;
}

//
// Horizontal dot product over d_k, predicated so any length works. Both
// operands are the raw query and key rows: the decay belongs to the sum term,
// not to this rank-1 coefficient. Four accumulators keep the FMA pipeline fed.
//
MLAS_FORCEINLINE
float
MlasLinearAttentionSveDot(
    const float* q0,
    const float* kt,
    size_t d_k
)
{
    const size_t W = svcntw();

    svfloat32_t d0 = svdup_n_f32(0.0f);
    svfloat32_t d1 = svdup_n_f32(0.0f);
    svfloat32_t d2 = svdup_n_f32(0.0f);
    svfloat32_t d3 = svdup_n_f32(0.0f);

    size_t i = 0;

    while (i < d_k) {
        const svbool_t p0 = svwhilelt_b32_u64(uint64_t(i), uint64_t(d_k));
        const svbool_t p1 = svwhilelt_b32_u64(uint64_t(i + W), uint64_t(d_k));
        const svbool_t p2 = svwhilelt_b32_u64(uint64_t(i + 2 * W), uint64_t(d_k));
        const svbool_t p3 = svwhilelt_b32_u64(uint64_t(i + 3 * W), uint64_t(d_k));

        d0 = svmla_f32_m(p0, d0, svld1_f32(p0, q0 + i), svld1_f32(p0, kt + i));
        d1 = svmla_f32_m(p1, d1, svld1_f32(p1, q0 + i + W), svld1_f32(p1, kt + i + W));
        d2 = svmla_f32_m(p2, d2, svld1_f32(p2, q0 + i + 2 * W), svld1_f32(p2, kt + i + 2 * W));
        d3 = svmla_f32_m(p3, d3, svld1_f32(p3, q0 + i + 3 * W), svld1_f32(p3, kt + i + 3 * W));

        i += 4 * W;
    }

    d0 = svadd_f32_x(svptrue_b32(), d0, d1);
    d2 = svadd_f32_x(svptrue_b32(), d2, d3);

    return svaddv_f32(svptrue_b32(), svadd_f32_x(svptrue_b32(), d0, d2));
}

//
// linear / gated, one column panel. Every stream the i loop reads -- the key
// row, the decay row, each head's query row and the state row -- is an
// induction pointer, so the loop body carries one increment per stream instead
// of recomputing an address from i.
//
template <size_t NOUT, size_t NLANE, bool HAS_DECAY, bool FULL>
MLAS_FORCEINLINE
void
MlasLinearAttentionSveSinglePanel(
    float* MLAS_LA_RESTRICT S,
    size_t d_k,
    size_t d_v,
    size_t j0,
    const float* MLAS_LA_RESTRICT dec,
    const float* MLAS_LA_RESTRICT kt,
    const float* MLAS_LA_RESTRICT vt,
    const float* MLAS_LA_RESTRICT q_base,
    float* MLAS_LA_RESTRICT o_base,
    float scale
)
{
    const size_t W = svcntw();

    MLAS_LA_SVE_EACH_LANE(MLAS_LA_SVE_PRED)

#define MLAS_LA_VV(l)                          \
    svfloat32_t vv##l = svdup_n_f32(0.0f);     \
    if constexpr (NLANE > (l)) {               \
        vv##l = svld1_f32(pg##l, vt + j0 + (l) * W); \
    }
    MLAS_LA_SVE_EACH_LANE(MLAS_LA_VV)
#undef MLAS_LA_VV

#define MLAS_LA_DECL(g, l) svfloat32_t MLAS_LA_SVE_ACC(g, l) = svdup_n_f32(0.0f);
    MLAS_LA_SVE_GRID(MLAS_LA_DECL)
#undef MLAS_LA_DECL

    //
    // Induction pointers, one per stream.
    //
    const float* MLAS_LA_RESTRICT ktp = kt;
    const float* MLAS_LA_RESTRICT decp = dec;
    float* MLAS_LA_RESTRICT Sp = S + j0;

#define MLAS_LA_QP(g)                                                   \
    const float* MLAS_LA_RESTRICT qp##g = (NOUT > (g)) ? (q_base + (g) * d_k) : q_base;
    MLAS_LA_SVE_EACH_HEAD(MLAS_LA_QP)
#undef MLAS_LA_QP

    for (size_t i = 0; i < d_k; ++i) {
        const svfloat32_t kk = svdup_n_f32(*ktp++);

        svfloat32_t dc = svdup_n_f32(0.0f);
        if constexpr (HAS_DECAY) {
            dc = svdup_n_f32(*decp++);
        }

        //
        // Build S_new for this lane, store it, then feed every readout head
        // from the value already in a register.
        //
#define MLAS_LA_STEP(l)                                                        \
    svfloat32_t s##l = svdup_n_f32(0.0f);                                      \
    if constexpr (NLANE > (l)) {                                               \
        const svfloat32_t old##l = svld1_f32(pg##l, Sp + (l) * W);             \
        if constexpr (HAS_DECAY) {                                             \
            s##l = svmla_f32_m(pg##l, svmul_f32_m(pg##l, vv##l, kk), old##l, dc); \
        } else {                                                               \
            s##l = svmla_f32_m(pg##l, old##l, vv##l, kk);                      \
        }                                                                      \
        svst1_f32(pg##l, Sp + (l) * W, s##l);                                  \
    }
        MLAS_LA_SVE_EACH_LANE(MLAS_LA_STEP)
#undef MLAS_LA_STEP

#define MLAS_LA_QW(g)                          \
    svfloat32_t qw##g = svdup_n_f32(0.0f);     \
    if constexpr (NOUT > (g)) {                \
        qw##g = svdup_n_f32(*qp##g);           \
        ++qp##g;                               \
    }
        MLAS_LA_SVE_EACH_HEAD(MLAS_LA_QW)
#undef MLAS_LA_QW

#define MLAS_LA_READ(g, l)                                                     \
    if constexpr (NOUT > (g) && NLANE > (l)) {                                 \
        MLAS_LA_SVE_ACC(g, l) =                                                \
            svmla_f32_m(pg##l, MLAS_LA_SVE_ACC(g, l), s##l, qw##g);            \
    }
        MLAS_LA_SVE_GRID(MLAS_LA_READ)
#undef MLAS_LA_READ

        Sp += d_v;
    }

    const svfloat32_t sc = svdup_n_f32(scale);

#define MLAS_LA_STORE(g, l)                                                    \
    if constexpr (NOUT > (g) && NLANE > (l)) {                                 \
        svst1_f32(pg##l, o_base + (g) * d_v + j0 + (l) * W,                    \
                  svmul_f32_m(pg##l, MLAS_LA_SVE_ACC(g, l), sc));              \
    }
    MLAS_LA_SVE_GRID(MLAS_LA_STORE)
#undef MLAS_LA_STORE
}

//
// delta / gated_delta, one column panel. Pass 1 accumulates retrieved and every
// head's S_old readout from one read; pass 2 re-reads the panel from L1 and
// writes S_new. Same induction-pointer discipline as the single-pass panel.
//
template <size_t NOUT, size_t NLANE, bool HAS_DECAY, bool FULL>
MLAS_FORCEINLINE
void
MlasLinearAttentionSveTwoPassPanel(
    float* MLAS_LA_RESTRICT S,
    size_t d_k,
    size_t d_v,
    size_t j0,
    const float* MLAS_LA_RESTRICT dec,
    const float* MLAS_LA_RESTRICT kt,
    const float* MLAS_LA_RESTRICT vt,
    float* MLAS_LA_RESTRICT o_base,
    float scale,
    float beta_val,
    const float* MLAS_LA_RESTRICT wkv,
    const float* MLAS_LA_RESTRICT wqv,
    size_t wq_stride,
    const float* MLAS_LA_RESTRICT qk
)
{
    const size_t W = svcntw();

    MLAS_LA_SVE_EACH_LANE(MLAS_LA_SVE_PRED)

#define MLAS_LA_DECLR(l) svfloat32_t r##l = svdup_n_f32(0.0f);
    MLAS_LA_SVE_EACH_LANE(MLAS_LA_DECLR)
#undef MLAS_LA_DECLR

#define MLAS_LA_DECL(g, l) svfloat32_t MLAS_LA_SVE_ACC(g, l) = svdup_n_f32(0.0f);
    MLAS_LA_SVE_GRID(MLAS_LA_DECL)
#undef MLAS_LA_DECL

    {
        const float* MLAS_LA_RESTRICT wkp = wkv;
        const float* MLAS_LA_RESTRICT Sp = S + j0;

#define MLAS_LA_WQP(g)                                                          \
    const float* MLAS_LA_RESTRICT wqp##g = (NOUT > (g)) ? (wqv + (g) * wq_stride) : wqv;
        MLAS_LA_SVE_EACH_HEAD(MLAS_LA_WQP)
#undef MLAS_LA_WQP

        for (size_t i = 0; i < d_k; ++i) {
            const svfloat32_t wk = svdup_n_f32(*wkp++);

#define MLAS_LA_LOAD(l)                                            \
    svfloat32_t s##l = svdup_n_f32(0.0f);                          \
    if constexpr (NLANE > (l)) {                                   \
        s##l = svld1_f32(pg##l, Sp + (l) * W);                     \
        r##l = svmla_f32_m(pg##l, r##l, s##l, wk);                 \
    }
            MLAS_LA_SVE_EACH_LANE(MLAS_LA_LOAD)
#undef MLAS_LA_LOAD

#define MLAS_LA_WQW(g)                         \
    svfloat32_t wq##g = svdup_n_f32(0.0f);     \
    if constexpr (NOUT > (g)) {                \
        wq##g = svdup_n_f32(*wqp##g);          \
        ++wqp##g;                              \
    }
            MLAS_LA_SVE_EACH_HEAD(MLAS_LA_WQW)
#undef MLAS_LA_WQW

#define MLAS_LA_ACCUM(g, l)                                                    \
    if constexpr (NOUT > (g) && NLANE > (l)) {                                 \
        MLAS_LA_SVE_ACC(g, l) =                                                \
            svmla_f32_m(pg##l, MLAS_LA_SVE_ACC(g, l), s##l, wq##g);            \
    }
            MLAS_LA_SVE_GRID(MLAS_LA_ACCUM)
#undef MLAS_LA_ACCUM

            Sp += d_v;
        }
    }

    //
    // upd = beta * (v - retrieved).
    //
    const svfloat32_t bb = svdup_n_f32(beta_val);

#define MLAS_LA_UPD(l)                                                              \
    svfloat32_t u##l = svdup_n_f32(0.0f);                                           \
    if constexpr (NLANE > (l)) {                                                    \
        u##l = svmul_f32_m(pg##l,                                                   \
                           svsub_f32_m(pg##l, svld1_f32(pg##l, vt + j0 + (l) * W), r##l), \
                           bb);                                                     \
    }
    MLAS_LA_SVE_EACH_LANE(MLAS_LA_UPD)
#undef MLAS_LA_UPD

    {
        const svfloat32_t sc = svdup_n_f32(scale);

#define MLAS_LA_STORE(g, l)                                                        \
    if constexpr (NOUT > (g) && NLANE > (l)) {                                     \
        const svfloat32_t o =                                                      \
            svmla_f32_m(pg##l, MLAS_LA_SVE_ACC(g, l), u##l, svdup_n_f32(qk[g]));   \
        svst1_f32(pg##l, o_base + (g) * d_v + j0 + (l) * W,                        \
                  svmul_f32_m(pg##l, o, sc));                                      \
    }
        MLAS_LA_SVE_GRID(MLAS_LA_STORE)
#undef MLAS_LA_STORE
    }

    //
    // Pass 2: re-read the panel from L1 and write S_new in place.
    //
    {
        const float* MLAS_LA_RESTRICT ktp = kt;
        const float* MLAS_LA_RESTRICT decp = dec;
        float* MLAS_LA_RESTRICT Sp = S + j0;

        for (size_t i = 0; i < d_k; ++i) {
            const svfloat32_t kk = svdup_n_f32(*ktp++);

            svfloat32_t dc = svdup_n_f32(0.0f);
            if constexpr (HAS_DECAY) {
                dc = svdup_n_f32(*decp++);
            }

#define MLAS_LA_WRITE(l)                                                              \
    if constexpr (NLANE > (l)) {                                                      \
        const svfloat32_t old = svld1_f32(pg##l, Sp + (l) * W);                       \
        if constexpr (HAS_DECAY) {                                                    \
            svst1_f32(pg##l, Sp + (l) * W,                                            \
                      svmla_f32_m(pg##l, svmul_f32_m(pg##l, u##l, kk), old, dc));     \
        } else {                                                                      \
            svst1_f32(pg##l, Sp + (l) * W, svmla_f32_m(pg##l, old, u##l, kk));        \
        }                                                                             \
    }
            MLAS_LA_SVE_EACH_LANE(MLAS_LA_WRITE)
#undef MLAS_LA_WRITE

            Sp += d_v;
        }
    }
}

//
// Panel loops. Full panels take the all-true body; at most one trailing
// partial panel takes the predicated one.
//
template <size_t NOUT, size_t NLANE, bool HAS_DECAY>
MLAS_FORCEINLINE
void
MlasLinearAttentionSveSinglePass(
    float* MLAS_LA_RESTRICT S,
    size_t d_k,
    size_t d_v,
    const float* MLAS_LA_RESTRICT dec,
    const float* MLAS_LA_RESTRICT kt,
    const float* MLAS_LA_RESTRICT vt,
    const float* MLAS_LA_RESTRICT q_base,
    float* MLAS_LA_RESTRICT o_base,
    float scale
)
{
    const size_t PW = NLANE * svcntw();
    size_t j0 = 0;

    for (; j0 + PW <= d_v; j0 += PW) {
        //
        // NOUT==1 at eight lanes is the shape this kernel is judged on against
        // the NEON one, and it is the one both compilers spill on when the K
        // loop is written with indexed FMA. That body is hand-scheduled; see
        // linear_attention_asm_sve.h. Everything else keeps the intrinsics.
        //
        if constexpr (NOUT == 1 && NLANE == 8) {
            MlasLinearAttentionSveSinglePanelAsmN1<HAS_DECAY>(
                S + j0, kt, dec, q_base, vt + j0, o_base + j0, d_k,
                d_v * sizeof(float), &scale);
        } else {
            MlasLinearAttentionSveSinglePanel<NOUT, NLANE, HAS_DECAY, true>(
                S, d_k, d_v, j0, dec, kt, vt, q_base, o_base, scale);
        }
    }

    if (j0 < d_v) {
        MlasLinearAttentionSveSinglePanel<NOUT, NLANE, HAS_DECAY, false>(
            S, d_k, d_v, j0, dec, kt, vt, q_base, o_base, scale);
    }
}

//
// Token-blocked single pass: two consecutive tokens per sweep of S, NOUT == 1
// only. The kernel is L1-port-bound (0.5% miss rate, ~2.3 accesses/cycle), so
// the win is that one state load and one state store serve both tokens --
// state traffic per token halves while the FMA count per token is unchanged.
// The per-element operation order matches the serial kernel exactly, so the
// result is bit-exact with the unblocked path.
//
// Full panels are four column vectors wide (two tokens need twice the value
// vectors and accumulators) and run the hand-scheduled body; the trailing
// partial panel runs the existing predicated intrinsics panel once per token
// -- unblocked but correct, and it executes at most once per row.
//
template <bool HAS_DECAY>
MLAS_FORCEINLINE
void
MlasLinearAttentionSveSinglePass2(
    float* MLAS_LA_RESTRICT S,
    size_t d_k,
    size_t d_v,
    const float* MLAS_LA_RESTRICT dec0,
    const float* MLAS_LA_RESTRICT dec1,
    const float* MLAS_LA_RESTRICT kt0,
    const float* MLAS_LA_RESTRICT kt1,
    const float* MLAS_LA_RESTRICT vt0,
    const float* MLAS_LA_RESTRICT vt1,
    const float* MLAS_LA_RESTRICT q0,
    const float* MLAS_LA_RESTRICT q1,
    float* MLAS_LA_RESTRICT o0,
    float* MLAS_LA_RESTRICT o1,
    float scale
)
{
    const size_t PW = 4 * svcntw();
    size_t j0 = 0;

    for (; j0 + PW <= d_v; j0 += PW) {
        MlasLinearAttentionSveSinglePanelAsmN1x2<HAS_DECAY>(
            S + j0, kt0, dec0, q0, vt0 + j0, o0 + j0,
            kt1, dec1, q1, vt1 + j0, o1 + j0,
            d_k, d_v * sizeof(float), &scale);
    }

    if (j0 < d_v) {
        MlasLinearAttentionSveSinglePanel<1, 8, HAS_DECAY, false>(
            S, d_k, d_v, j0, dec0, kt0, vt0, q0, o0, scale);
        MlasLinearAttentionSveSinglePanel<1, 8, HAS_DECAY, false>(
            S, d_k, d_v, j0, dec1, kt1, vt1, q1, o1, scale);
    }
}

template <size_t NOUT, size_t NLANE, bool HAS_DECAY>
MLAS_FORCEINLINE
void
MlasLinearAttentionSveTwoPass(
    float* MLAS_LA_RESTRICT S,
    size_t d_k,
    size_t d_v,
    const float* MLAS_LA_RESTRICT dec,
    const float* MLAS_LA_RESTRICT kt,
    const float* MLAS_LA_RESTRICT vt,
    const float* MLAS_LA_RESTRICT q_base,
    float* MLAS_LA_RESTRICT o_base,
    float scale,
    float beta_val,
    float* MLAS_LA_RESTRICT wk_buf,
    float* MLAS_LA_RESTRICT wq_buf
)
{
    constexpr size_t MaxK = MlasLinearAttentionSveMaxKHeadSize;
    const size_t W = svcntw();
    const size_t PW = NLANE * W;

    //
    // Pre-weighting stays inside the kernel -- it is plain vector work with no
    // relocations, and only exp() has to be done by the driver. Without decay
    // the weights are the input rows themselves and nothing is copied.
    //
    const float* MLAS_LA_RESTRICT wkv = kt;
    const float* MLAS_LA_RESTRICT wqv = q_base;
    size_t wq_stride = d_k;

    if constexpr (HAS_DECAY) {
        for (size_t i = 0; i < d_k; i += W) {
            const svbool_t p = svwhilelt_b32_u64(uint64_t(i), uint64_t(d_k));
            svst1_f32(p, wk_buf + i,
                      svmul_f32_m(p, svld1_f32(p, dec + i), svld1_f32(p, kt + i)));
        }
        wkv = wk_buf;

        for (size_t g = 0; g < NOUT; ++g) {
            for (size_t i = 0; i < d_k; i += W) {
                const svbool_t p = svwhilelt_b32_u64(uint64_t(i), uint64_t(d_k));
                svst1_f32(p, wq_buf + g * MaxK + i,
                          svmul_f32_m(p, svld1_f32(p, dec + i),
                                      svld1_f32(p, q_base + g * d_k + i)));
            }
        }
        wqv = wq_buf;
        wq_stride = MaxK;
    }

    //
    // The rank-1 coefficient uses the raw rows, not the pre-weighted ones.
    //
    float qk[MLAS_LA_SVE_MAX_HEADS];

    for (size_t g = 0; g < NOUT; ++g) {
        qk[g] = MlasLinearAttentionSveDot(q_base + g * d_k, kt, d_k);
    }

    size_t j0 = 0;

    for (; j0 + PW <= d_v; j0 += PW) {
        MlasLinearAttentionSveTwoPassPanel<NOUT, NLANE, HAS_DECAY, true>(
            S, d_k, d_v, j0, dec, kt, vt, o_base, scale, beta_val, wkv, wqv, wq_stride, qk);
    }

    if (j0 < d_v) {
        MlasLinearAttentionSveTwoPassPanel<NOUT, NLANE, HAS_DECAY, false>(
            S, d_k, d_v, j0, dec, kt, vt, o_base, scale, beta_val, wkv, wqv, wq_stride, qk);
    }
}

//
// Token loop for one chunk at a fixed readout-head count. HasDecay and HasBeta
// are resolved to template parameters here, once per chunk, so neither the
// panel loops nor the token loop carries a branch on them.
//
template <size_t NOUT, size_t NLANE, bool HAS_DECAY, bool HAS_BETA>
MLAS_FORCEINLINE
void
MlasLinearAttentionSveTokens(
    const MLAS_LINEAR_ATTENTION_SVE_CHUNK* Chunk
)
{
    const size_t d_k = Chunk->KHeadSize;
    const size_t d_v = Chunk->VHeadSize;
    constexpr size_t MaxK = MlasLinearAttentionSveMaxKHeadSize;

    float* MLAS_LA_RESTRICT S = Chunk->State;

    //
    // Sized away without decay, where the kernel reads the input rows directly.
    //
    float wk_buf[HAS_DECAY ? MaxK : 1];
    float wq_buf[HAS_DECAY ? NOUT * MaxK : 1];

    size_t t = 0;

    //
    // Single-head single-pass tokens go two at a time: one sweep of S serves
    // both, halving the state traffic per token on an L1-port-bound loop. Any
    // odd trailing token falls through to the serial loop below.
    //
    if constexpr (!HAS_BETA && NOUT == 1 && NLANE >= 4) {
        for (; t + 2 <= Chunk->TokenCount; t += 2) {
            const size_t u = t + 1;
            MlasLinearAttentionSveSinglePass2<HAS_DECAY>(
                S, d_k, d_v,
                HAS_DECAY ? (Chunk->Decay + t * d_k) : nullptr,
                HAS_DECAY ? (Chunk->Decay + u * d_k) : nullptr,
                Chunk->Key + t * Chunk->KeyTokenStride,
                Chunk->Key + u * Chunk->KeyTokenStride,
                Chunk->Value + t * Chunk->ValueTokenStride,
                Chunk->Value + u * Chunk->ValueTokenStride,
                Chunk->Query + t * Chunk->QueryTokenStride,
                Chunk->Query + u * Chunk->QueryTokenStride,
                Chunk->Output + t * Chunk->OutputTokenStride,
                Chunk->Output + u * Chunk->OutputTokenStride,
                Chunk->Scale);
        }
    }

    for (; t < Chunk->TokenCount; ++t) {
        const float* MLAS_LA_RESTRICT kt = Chunk->Key + t * Chunk->KeyTokenStride;
        const float* MLAS_LA_RESTRICT vt = Chunk->Value + t * Chunk->ValueTokenStride;
        const float* MLAS_LA_RESTRICT q0 = Chunk->Query + t * Chunk->QueryTokenStride;
        float* MLAS_LA_RESTRICT o0 = Chunk->Output + t * Chunk->OutputTokenStride;

        const float* MLAS_LA_RESTRICT dec = HAS_DECAY ? (Chunk->Decay + t * d_k) : nullptr;

        if constexpr (HAS_BETA) {
            const float beta_val = Chunk->Beta[t * Chunk->BetaTokenStride];
            MlasLinearAttentionSveTwoPass<NOUT, NLANE, HAS_DECAY>(
                S, d_k, d_v, dec, kt, vt, q0, o0, Chunk->Scale, beta_val, wk_buf, wq_buf);
        } else {
            MlasLinearAttentionSveSinglePass<NOUT, NLANE, HAS_DECAY>(
                S, d_k, d_v, dec, kt, vt, q0, o0, Chunk->Scale);
        }
    }
}

//
// Resolve the lane count and the two rule flags, then run the chunk. if/else
// rather than switch: a jump table would emit adrp, which the freezer rejects.
//
//
// One (NOUT, NLANE) instantiation, with the runtime rule flags folded to
// template parameters. Factored out so the lane-count ladder above it does not
// repeat the four-way branch per lane count.
//
template <size_t NOUT, size_t NLANE>
MLAS_FORCEINLINE
void
MlasLinearAttentionSveRunTokens(
    const MLAS_LINEAR_ATTENTION_SVE_CHUNK* Chunk,
    bool has_decay,
    bool has_beta
)
{
    if (has_beta) {
        if (has_decay) {
            MlasLinearAttentionSveTokens<NOUT, NLANE, true, true>(Chunk);
        } else {
            MlasLinearAttentionSveTokens<NOUT, NLANE, false, true>(Chunk);
        }
    } else {
        if (has_decay) {
            MlasLinearAttentionSveTokens<NOUT, NLANE, true, false>(Chunk);
        } else {
            MlasLinearAttentionSveTokens<NOUT, NLANE, false, false>(Chunk);
        }
    }
}

template <size_t NOUT>
MLAS_FORCEINLINE
void
MlasLinearAttentionSveHead(
    const MLAS_LINEAR_ATTENTION_SVE_CHUNK* Chunk
)
{
    constexpr size_t NLaneMax = MlasLinearAttentionSveLanes<NOUT>::Max;

    const size_t lanes = MlasLinearAttentionSveLaneCount(NLaneMax, Chunk->KHeadSize);
    const bool has_decay = (Chunk->Decay != nullptr);
    const bool has_beta = (Chunk->Beta != nullptr);

    //
    // Dispatch the EXACT lane count the L1 helper returns, not merely
    // "wide or halved". The helper can halve more than once: at VL=512 with
    // d_k=256 the eight-lane budget for NOUT=1 comes back as 2, and at the
    // architectural maximum VL=2048 it reaches 1 for every head count. An
    // earlier revision collapsed the return to a boolean and hardcoded a
    // single halving, which silently broke the half-L1 two-pass bound at
    // VL >= 512. Every power of two down to 1 is instantiated; the token
    // pairing and the hand-scheduled bodies gate themselves on NLANE, so the
    // narrow instantiations honor the cap they exist for.
    //
    // An if/else ladder rather than a switch, as everywhere else in this
    // kernel: a switch risks a jump table, which the freezer rejects.
    //
    if (lanes >= NLaneMax) {
        MlasLinearAttentionSveRunTokens<NOUT, NLaneMax>(Chunk, has_decay, has_beta);
        return;
    }
    if constexpr (NLaneMax >= 4) {
        if (lanes == NLaneMax / 2) {
            MlasLinearAttentionSveRunTokens<NOUT, NLaneMax / 2>(Chunk, has_decay, has_beta);
            return;
        }
    }
    if constexpr (NLaneMax >= 8) {
        if (lanes == NLaneMax / 4) {
            MlasLinearAttentionSveRunTokens<NOUT, NLaneMax / 4>(Chunk, has_decay, has_beta);
            return;
        }
    }

    //
    // lanes == 1 (the helper never returns 0), reached only when even two
    // lanes overflow the L1 budget.
    //
    MlasLinearAttentionSveRunTokens<NOUT, 1>(Chunk, has_decay, has_beta);
}
