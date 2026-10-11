/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    linear_attention_kernel_rvv.cpp

Abstract:

    This module implements the RISC-V Vector (RVV) linear (recurrent)
    attention kernel.

    The recurrence is the one the NEON and SVE kernels implement. With
    dec[i] = exp(g_t[i]) (or 1 when the rule has no decay):

        retrieved[j] = sum_i (dec[i]*k[i] * S_old[i,j])
        upd[j]       = beta * (v[j] - retrieved[j])     (= v[j] with no beta)
        S_new[i,j]   = dec[i] * S_old[i,j] + k[i] * upd[j]
        o_g[j]       = scale * sum_i (q_g[i] * S_new[i,j])

    and it splits the same way into two shapes:

    * linear / gated (no beta). upd is v, known before the traversal, so each
      S_new element is written and consumed by every readout head in the same
      iteration -- a SINGLE pass, one read and one write of S. Because the
      readout reads S_new the weight is the raw q_g[i].

    * delta / gated_delta (beta). Two passes over each column panel, using the
      identity that re-expresses the readout over S_old plus a rank-1
      correction, so it never needs the written-back S_new:

          o_g[j] = scale * ( sum_i (dec[i]*q_g[i] * S_old[i,j]) + (q_g.k)*upd[j] )

      Pass 1 accumulates retrieved and every head's readout from one read of
      S_old; pass 2 writes S_new in place. Here the two passes of CONSECUTIVE
      tokens are fused: pass 1 of token t+1 reads exactly the S_new that pass 2
      of token t writes, so one sweep stores each row and accumulates the next
      token's reduction from the value still in registers. S is read once and
      written once per token, as in the single-pass form, and the reduction
      results cross the token boundary through two small per-column buffers.
      Only the first token runs pass 1 on its own. The plain two-pass form,
      which re-reads each panel from L1 instead, remains for d_v beyond those
      buffers.

    Where this differs from the fixed-width kernels is the panel geometry. A
    NEON panel is eight named 128-bit registers and an SVE panel is NLANE named
    Z registers, because neither ISA can widen a single register. RVV can: a
    register group of LMUL registers is one operand, so a column panel here is
    ONE vector value of vl floats, with vl following vsetvl. That removes the
    per-lane macro grid entirely -- the only sizeless values that need names
    are the NOUT readout accumulators -- and makes the trailing partial panel
    free, since a short vl is an ordinary iteration rather than a predicated
    copy of the body.

    The single-pass form additionally runs two tokens per sweep of S at one
    and two readout heads. The loop is bound by vector issue and half of its
    slots are the state load and store, so chaining token t+1 onto token t
    while the row is still in a register halves the state traffic per token.

    LMUL is chosen per head, not fixed. The register budget is what bounds it
    from above: the single pass holds v + NOUT readouts + the in-flight row
    (NOUT + 2 groups), the fused two-pass sweep holds upd + the row + the next
    token's retrieval + NOUT readouts (NOUT + 3), and the paired single pass
    holds two of everything (2 * NOUT + 3). Against 32 vector registers, with
    one group kept spare for temporaries, that is LMUL 4 for one to four heads
    and 2 at eight. It is then narrowed for two reasons: a group wider than
    d_v wastes its upper registers on inactive lanes, and the unfused
    two-pass panel of d_k x vl floats must fit in roughly half of L1 for its
    second pass to re-read it from there.

    Every per-row weight -- the key, the decay, each head's query -- is a
    scalar operand of a .vf instruction, so the inner loops carry no broadcast
    and no vector register for it, the same mechanism as NEON's vfmaq_n_f32.
    The decayed update is evaluated as (dec * S_old) + k * upd, multiplying
    the loaded row in place and then fusing the rank-1 term onto it, so it
    needs no temporary group; the other association, k * upd first, holds a
    product live alongside the row and spills at the widest instantiations.

    This reassociates the floating-point sums relative to the portable kernel,
    so results agree to tolerance rather than bit-exactly.

--*/

#include "mlasi.h"
#include "linear_attention.h"

#if defined(MLAS_USE_RVV)

#include <riscv_vector.h>

#include <cmath>

namespace
{

//
// Largest KHeadSize accepted. It bounds the two-pass pre-weighting buffers,
// which live on the stack: one row of d_k for the key plus one per readout
// head, 9 * 256 * 4 = 9 KB at the widest instantiation. Same bound as the NEON
// and SVE kernels.
//
constexpr size_t MlasLinearAttentionRvvMaxKHeadSize = 256;

//
// Largest VHeadSize at which the two-pass form carries its retrieval and
// readout accumulators across tokens (see the fused sweep below). They live
// on the stack, (1 + NOUT) rows of d_v, so 9 * 256 * 4 = 9 KB at the widest
// instantiation. Larger d_v takes the unfused two-pass form, which keeps no
// per-column state between tokens and so has no bound.
//
constexpr size_t MlasLinearAttentionRvvMaxVHeadSize = 256;

//
// Half of a 64 KB L1, in floats. The unfused two-pass form re-reads its
// panel, so the panel (d_k x vl floats) is kept under this by narrowing LMUL.
//
constexpr size_t MlasLinearAttentionRvvPanelBudget = 8192;

//
// One vector "shape" per LMUL. The intrinsics carry the LMUL in their name, so
// the panel bodies are written once against this interface and instantiated
// per LMUL. Only the operations the panels need are wrapped.
//
#define MLAS_LA_RVV_SHAPE(LMUL)                                                            \
    struct RvvF32M##LMUL {                                                                 \
        using VType = vfloat32m##LMUL##_t;                                                 \
        static MLAS_FORCEINLINE size_t SetVl(size_t n)                                     \
        {                                                                                  \
            return __riscv_vsetvl_e32m##LMUL(n);                                           \
        }                                                                                  \
        static MLAS_FORCEINLINE VType Zero(size_t vl)                                      \
        {                                                                                  \
            return __riscv_vfmv_v_f_f32m##LMUL(0.0f, vl);                                  \
        }                                                                                  \
        static MLAS_FORCEINLINE VType Load(const float* p, size_t vl)                      \
        {                                                                                  \
            return __riscv_vle32_v_f32m##LMUL(p, vl);                                      \
        }                                                                                  \
        static MLAS_FORCEINLINE void Store(float* p, VType v, size_t vl)                   \
        {                                                                                  \
            __riscv_vse32_v_f32m##LMUL(p, v, vl);                                          \
        }                                                                                  \
        static MLAS_FORCEINLINE VType MulF(VType a, float b, size_t vl)                    \
        {                                                                                  \
            return __riscv_vfmul_vf_f32m##LMUL(a, b, vl);                                  \
        }                                                                                  \
        static MLAS_FORCEINLINE VType Sub(VType a, VType b, size_t vl)                     \
        {                                                                                  \
            return __riscv_vfsub_vv_f32m##LMUL(a, b, vl);                                  \
        }                                                                                  \
        /* acc + a * b */                                                                  \
        static MLAS_FORCEINLINE VType FmaccF(VType acc, float a, VType b, size_t vl)       \
        {                                                                                  \
            return __riscv_vfmacc_vf_f32m##LMUL(acc, a, b, vl);                            \
        }                                                                                  \
    };

MLAS_LA_RVV_SHAPE(1)
MLAS_LA_RVV_SHAPE(2)
MLAS_LA_RVV_SHAPE(4)
MLAS_LA_RVV_SHAPE(8)

#undef MLAS_LA_RVV_SHAPE

//
// Expand M(g) over every readout head the widest instantiation can hold. The
// accumulators are sizeless vector values, which cannot be array elements, so
// they are named through this rather than indexed. Bodies test NOUT > g, and
// the dead heads fold away.
//
#define MLAS_LA_RVV_EACH_HEAD(M) M(0) M(1) M(2) M(3) M(4) M(5) M(6) M(7)

//
// q . k over d_k, at a fixed LMUL of 4 independent of the panel shape. Both
// operands are the raw query and key rows: the decay belongs to the sum term,
// not to this rank-1 coefficient.
//
MLAS_FORCEINLINE
float
LinearAttentionDotRvv(
    const float* __restrict q0,
    const float* __restrict kt,
    size_t d_k
)
{
    const size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t acc = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

    for (size_t i = 0; i < d_k;) {
        const size_t vl = __riscv_vsetvl_e32m4(d_k - i);
        acc = __riscv_vfmacc_vv_f32m4_tu(acc, __riscv_vle32_v_f32m4(q0 + i, vl),
                                         __riscv_vle32_v_f32m4(kt + i, vl), vl);
        i += vl;
    }

    vfloat32m1_t sum = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    sum = __riscv_vfredusum_vs_f32m4_f32m1(acc, sum, vlmax);
    return __riscv_vfmv_f_s_f32m1_f32(sum);
}

//
// out[i] = a[i] * b[i] over d_k.
//
MLAS_FORCEINLINE
void
LinearAttentionMulRvv(
    float* __restrict out,
    const float* __restrict a,
    const float* __restrict b,
    size_t d_k
)
{
    for (size_t i = 0; i < d_k;) {
        const size_t vl = __riscv_vsetvl_e32m4(d_k - i);
        __riscv_vse32_v_f32m4(out + i,
                              __riscv_vfmul_vv_f32m4(__riscv_vle32_v_f32m4(a + i, vl),
                                                     __riscv_vle32_v_f32m4(b + i, vl), vl),
                              vl);
        i += vl;
    }
}

//
// linear / gated, one column panel of vl floats starting at column j0.
//
// Every stream the i loop reads -- the key row, the decay row, each head's
// query row and the state row -- advances by one element per iteration.
//
template <typename V, size_t NOUT, bool HAS_DECAY>
MLAS_FORCEINLINE
void
SinglePassPanelRvv(
    float* __restrict S,
    size_t d_k,
    size_t d_v,
    size_t j0,
    size_t vl,
    const float* __restrict dec,
    const float* __restrict kt,
    const float* __restrict vt,
    const float* __restrict q_base,
    float* __restrict o_base,
    float scale
)
{
    using VT = typename V::VType;

    const VT vv = V::Load(vt + j0, vl);

#define MLAS_LA_DECL(g) VT acc##g = V::Zero(vl);
    MLAS_LA_RVV_EACH_HEAD(MLAS_LA_DECL)
#undef MLAS_LA_DECL

    const float* __restrict ktp = kt;
    const float* __restrict decp = dec;
    float* __restrict Sp = S + j0;

    const float* qp[NOUT];
    for (size_t g = 0; g < NOUT; ++g) {
        qp[g] = q_base + g * d_k;
    }

    for (size_t i = 0; i < d_k; ++i) {
        const float kk = *ktp++;

        //
        // Build S_new for this row, store it, then feed every readout head
        // from the value still in registers.
        //
        VT s;
        if constexpr (HAS_DECAY) {
            const float dc = *decp++;
            s = V::FmaccF(V::MulF(V::Load(Sp, vl), dc, vl), kk, vv, vl);
        } else {
            s = V::FmaccF(V::Load(Sp, vl), kk, vv, vl);
        }
        V::Store(Sp, s, vl);

#define MLAS_LA_READ(g)                                                 \
    if constexpr (NOUT > (g)) {                                         \
        acc##g = V::FmaccF(acc##g, qp[g][i], s, vl);                    \
    }
        MLAS_LA_RVV_EACH_HEAD(MLAS_LA_READ)
#undef MLAS_LA_READ

        Sp += d_v;
    }

#define MLAS_LA_STORE(g)                                                        \
    if constexpr (NOUT > (g)) {                                                 \
        V::Store(o_base + (g) * d_v + j0, V::MulF(acc##g, scale, vl), vl);      \
    }
    MLAS_LA_RVV_EACH_HEAD(MLAS_LA_STORE)
#undef MLAS_LA_STORE
}

//
// linear / gated, one column panel, TWO consecutive tokens per sweep of S.
//
// The single-pass loop is bound by vector issue, and half of its issue slots
// are the state load and store. Chaining token t+1 onto token t while the row
// is still in a register serves both tokens from one load and one store, so
// the state traffic per token halves while the FMA count per token is
// unchanged. The per-element operation order is exactly that of the unpaired
// panel applied twice, so the result is bit-exact with it.
//
// Live groups: two v rows, the in-flight S value and 2*NOUT accumulators, so
// this is instantiated only for NOUT <= 2, where the group can still be LMUL 4.
//
template <typename V, size_t NOUT, bool HAS_DECAY>
MLAS_FORCEINLINE
void
SinglePassPanelPairRvv(
    float* __restrict S,
    size_t d_k,
    size_t d_v,
    size_t j0,
    size_t vl,
    const float* __restrict dec0,
    const float* __restrict dec1,
    const float* __restrict kt0,
    const float* __restrict kt1,
    const float* __restrict vt0,
    const float* __restrict vt1,
    const float* __restrict q_base0,
    const float* __restrict q_base1,
    float* __restrict o_base0,
    float* __restrict o_base1,
    float scale
)
{
    static_assert(NOUT <= 2, "paired single pass is register-bound at two heads");

    using VT = typename V::VType;

    const VT vv0 = V::Load(vt0 + j0, vl);
    const VT vv1 = V::Load(vt1 + j0, vl);

#define MLAS_LA_DECL(g)                 \
    VT acc0_##g = V::Zero(vl);          \
    VT acc1_##g = V::Zero(vl);
    MLAS_LA_RVV_EACH_HEAD(MLAS_LA_DECL)
#undef MLAS_LA_DECL

    float* __restrict Sp = S + j0;

    const float* qp0[NOUT];
    const float* qp1[NOUT];
    for (size_t g = 0; g < NOUT; ++g) {
        qp0[g] = q_base0 + g * d_k;
        qp1[g] = q_base1 + g * d_k;
    }

    for (size_t i = 0; i < d_k; ++i) {
        const float kk0 = kt0[i];
        const float kk1 = kt1[i];

        VT s = V::Load(Sp, vl);

        if constexpr (HAS_DECAY) {
            s = V::FmaccF(V::MulF(s, dec0[i], vl), kk0, vv0, vl);
        } else {
            s = V::FmaccF(s, kk0, vv0, vl);
        }

#define MLAS_LA_READ0(g)                                                \
    if constexpr (NOUT > (g)) {                                         \
        acc0_##g = V::FmaccF(acc0_##g, qp0[g][i], s, vl);               \
    }
        MLAS_LA_RVV_EACH_HEAD(MLAS_LA_READ0)
#undef MLAS_LA_READ0

        if constexpr (HAS_DECAY) {
            s = V::FmaccF(V::MulF(s, dec1[i], vl), kk1, vv1, vl);
        } else {
            s = V::FmaccF(s, kk1, vv1, vl);
        }
        V::Store(Sp, s, vl);

#define MLAS_LA_READ1(g)                                                \
    if constexpr (NOUT > (g)) {                                         \
        acc1_##g = V::FmaccF(acc1_##g, qp1[g][i], s, vl);               \
    }
        MLAS_LA_RVV_EACH_HEAD(MLAS_LA_READ1)
#undef MLAS_LA_READ1

        Sp += d_v;
    }

#define MLAS_LA_STORE(g)                                                            \
    if constexpr (NOUT > (g)) {                                                     \
        V::Store(o_base0 + (g) * d_v + j0, V::MulF(acc0_##g, scale, vl), vl);       \
        V::Store(o_base1 + (g) * d_v + j0, V::MulF(acc1_##g, scale, vl), vl);       \
    }
    MLAS_LA_RVV_EACH_HEAD(MLAS_LA_STORE)
#undef MLAS_LA_STORE
}

//
// delta / gated_delta, one column panel. Pass 1 accumulates retrieved and
// every head's S_old readout from one read; pass 2 re-reads the panel from L1
// and writes S_new. wkv and wqv[g] are the decay-weighted key and query rows
// (or the raw rows without decay); qk[g] is the raw q_g . k.
//
template <typename V, size_t NOUT, bool HAS_DECAY>
MLAS_FORCEINLINE
void
TwoPassPanelRvv(
    float* __restrict S,
    size_t d_k,
    size_t d_v,
    size_t j0,
    size_t vl,
    const float* __restrict dec,
    const float* __restrict kt,
    const float* __restrict vt,
    float* __restrict o_base,
    float scale,
    float beta_val,
    const float* __restrict wkv,
    const float* const* wqv,
    const float* __restrict qk
)
{
    using VT = typename V::VType;

    VT r = V::Zero(vl);

#define MLAS_LA_DECL(g) VT acc##g = V::Zero(vl);
    MLAS_LA_RVV_EACH_HEAD(MLAS_LA_DECL)
#undef MLAS_LA_DECL

    {
        const float* __restrict wkp = wkv;
        const float* __restrict Sp = S + j0;

        for (size_t i = 0; i < d_k; ++i) {
            const VT s = V::Load(Sp, vl);
            r = V::FmaccF(r, *wkp++, s, vl);

#define MLAS_LA_ACCUM(g)                                                \
    if constexpr (NOUT > (g)) {                                         \
        acc##g = V::FmaccF(acc##g, wqv[g][i], s, vl);                   \
    }
            MLAS_LA_RVV_EACH_HEAD(MLAS_LA_ACCUM)
#undef MLAS_LA_ACCUM

            Sp += d_v;
        }
    }

    //
    // upd = beta * (v - retrieved), then close each readout with the rank-1
    // correction.
    //
    const VT u = V::MulF(V::Sub(V::Load(vt + j0, vl), r, vl), beta_val, vl);

#define MLAS_LA_STORE(g)                                                        \
    if constexpr (NOUT > (g)) {                                                 \
        V::Store(o_base + (g) * d_v + j0,                                       \
                 V::MulF(V::FmaccF(acc##g, qk[g], u, vl), scale, vl), vl);      \
    }
    MLAS_LA_RVV_EACH_HEAD(MLAS_LA_STORE)
#undef MLAS_LA_STORE

    //
    // Pass 2: re-read the panel from L1 and write S_new in place.
    //
    {
        const float* __restrict ktp = kt;
        const float* __restrict decp = dec;
        float* __restrict Sp = S + j0;

        for (size_t i = 0; i < d_k; ++i) {
            const float kk = *ktp++;
            const VT old = V::Load(Sp, vl);

            if constexpr (HAS_DECAY) {
                const float dc = *decp++;
                V::Store(Sp, V::FmaccF(V::MulF(old, dc, vl), kk, u, vl), vl);
            } else {
                V::Store(Sp, V::FmaccF(old, kk, u, vl), vl);
            }

            Sp += d_v;
        }
    }
}

//
// Pass 1 alone: retrieved and every head's S_old readout for one panel, into
// the cross-token buffers. Runs once per head, for the first token; every
// later token's pass 1 is fused into the previous token's sweep.
//
template <typename V, size_t NOUT>
MLAS_FORCEINLINE
void
Pass1PanelRvv(
    const float* __restrict S,
    size_t d_k,
    size_t d_v,
    size_t j0,
    size_t vl,
    const float* __restrict wkv,
    const float* const* wqv,
    float* __restrict r_buf,
    float* __restrict a_buf
)
{
    using VT = typename V::VType;

    VT r = V::Zero(vl);

#define MLAS_LA_DECL(g) VT acc##g = V::Zero(vl);
    MLAS_LA_RVV_EACH_HEAD(MLAS_LA_DECL)
#undef MLAS_LA_DECL

    const float* __restrict Sp = S + j0;

    for (size_t i = 0; i < d_k; ++i) {
        const VT s = V::Load(Sp, vl);
        r = V::FmaccF(r, wkv[i], s, vl);

#define MLAS_LA_ACCUM(g)                                                \
    if constexpr (NOUT > (g)) {                                         \
        acc##g = V::FmaccF(acc##g, wqv[g][i], s, vl);                   \
    }
        MLAS_LA_RVV_EACH_HEAD(MLAS_LA_ACCUM)
#undef MLAS_LA_ACCUM

        Sp += d_v;
    }

    V::Store(r_buf + j0, r, vl);

#define MLAS_LA_STORE(g)                                                \
    if constexpr (NOUT > (g)) {                                         \
        V::Store(a_buf + (g) * d_v + j0, acc##g, vl);                   \
    }
    MLAS_LA_RVV_EACH_HEAD(MLAS_LA_STORE)
#undef MLAS_LA_STORE
}

//
// delta / gated_delta, one column panel, with pass 2 of this token FUSED with
// pass 1 of the next. Pass 1 of token t+1 reads exactly the S_new that pass 2
// of token t writes, so the two are one sweep: build S_new for the row, store
// it, and accumulate the next token's retrieval and readouts from the value
// still in registers. S is then read once and written once per token, as in
// the single-pass form, rather than read twice.
//
// r_buf / a_buf carry this token's pass-1 results in and, when HAS_NEXT, the
// next token's out. wkv / wqv are the NEXT token's decay-weighted rows.
//
// Live groups: u, the in-flight S value, r and NOUT readouts.
//
template <typename V, size_t NOUT, bool HAS_DECAY, bool HAS_NEXT>
MLAS_FORCEINLINE
void
FusedPassPanelRvv(
    float* __restrict S,
    size_t d_k,
    size_t d_v,
    size_t j0,
    size_t vl,
    const float* __restrict dec,
    const float* __restrict kt,
    const float* __restrict vt,
    float* __restrict o_base,
    float scale,
    float beta_val,
    const float* __restrict qk,
    float* __restrict r_buf,
    float* __restrict a_buf,
    const float* __restrict wkv,
    const float* const* wqv
)
{
    using VT = typename V::VType;

    if constexpr (!HAS_NEXT) {
        MLAS_UNREFERENCED_PARAMETER(wkv);
        MLAS_UNREFERENCED_PARAMETER(wqv);
    }

    //
    // upd = beta * (v - retrieved), then close each readout with the rank-1
    // correction.
    //
    const VT u = V::MulF(V::Sub(V::Load(vt + j0, vl), V::Load(r_buf + j0, vl), vl),
                         beta_val, vl);

#define MLAS_LA_OUT(g)                                                              \
    if constexpr (NOUT > (g)) {                                                     \
        V::Store(o_base + (g) * d_v + j0,                                           \
                 V::MulF(V::FmaccF(V::Load(a_buf + (g) * d_v + j0, vl), qk[g], u, vl), \
                         scale, vl),                                                \
                 vl);                                                               \
    }
    MLAS_LA_RVV_EACH_HEAD(MLAS_LA_OUT)
#undef MLAS_LA_OUT

    VT r = V::Zero(vl);

#define MLAS_LA_DECL(g) VT acc##g = V::Zero(vl);
    MLAS_LA_RVV_EACH_HEAD(MLAS_LA_DECL)
#undef MLAS_LA_DECL

    float* __restrict Sp = S + j0;

    for (size_t i = 0; i < d_k; ++i) {
        const float kk = kt[i];

        VT s = V::Load(Sp, vl);
        if constexpr (HAS_DECAY) {
            s = V::FmaccF(V::MulF(s, dec[i], vl), kk, u, vl);
        } else {
            s = V::FmaccF(s, kk, u, vl);
        }
        V::Store(Sp, s, vl);

        if constexpr (HAS_NEXT) {
            r = V::FmaccF(r, wkv[i], s, vl);

#define MLAS_LA_ACCUM(g)                                                \
    if constexpr (NOUT > (g)) {                                         \
        acc##g = V::FmaccF(acc##g, wqv[g][i], s, vl);                   \
    }
            MLAS_LA_RVV_EACH_HEAD(MLAS_LA_ACCUM)
#undef MLAS_LA_ACCUM
        }

        Sp += d_v;
    }

    if constexpr (HAS_NEXT) {
        V::Store(r_buf + j0, r, vl);

#define MLAS_LA_STORE(g)                                                \
    if constexpr (NOUT > (g)) {                                         \
        V::Store(a_buf + (g) * d_v + j0, acc##g, vl);                   \
    }
        MLAS_LA_RVV_EACH_HEAD(MLAS_LA_STORE)
#undef MLAS_LA_STORE
    }
}

//
// The decay-weighted rows for the two-pass reduction. Without decay they are
// the input rows themselves and nothing is copied.
//
template <size_t NOUT, bool HAS_DECAY>
MLAS_FORCEINLINE
void
LinearAttentionWeightsRvv(
    const float* __restrict kt,
    const float* __restrict q0,
    const float* __restrict dec,
    size_t d_k,
    float* __restrict wk_buf,
    float* __restrict wq_buf,
    const float*& wkv,
    const float** wqv
)
{
    constexpr size_t MaxK = MlasLinearAttentionRvvMaxKHeadSize;

    if constexpr (HAS_DECAY) {
        LinearAttentionMulRvv(wk_buf, dec, kt, d_k);
        wkv = wk_buf;
        for (size_t g = 0; g < NOUT; ++g) {
            LinearAttentionMulRvv(wq_buf + g * MaxK, dec, q0 + g * d_k, d_k);
            wqv[g] = wq_buf + g * MaxK;
        }
    } else {
        MLAS_UNREFERENCED_PARAMETER(dec);
        MLAS_UNREFERENCED_PARAMETER(wk_buf);
        MLAS_UNREFERENCED_PARAMETER(wq_buf);
        wkv = kt;
        for (size_t g = 0; g < NOUT; ++g) {
            wqv[g] = q0 + g * d_k;
        }
    }
}

//
// exp(g_t) for one token into a d_k buffer. The per-key-dim layout is d_k
// exponentials, which go through MlasComputeExp and so resolve to the RVV exp
// kernel. The per-head layout is one exponential splatted across d_k, so the
// panels see a single layout.
//
MLAS_FORCEINLINE
void
LinearAttentionDecayRvv(
    const MLAS_LINEAR_ATTENTION_WORK* Work,
    size_t t,
    float* __restrict decvec
)
{
    const size_t d_k = Work->KHeadSize;
    const float* gt = Work->Decay + t * Work->DecayTokenStride;

    if (Work->DecayLayout == MlasLinearAttentionDecayPerKeyDim) {
        MlasComputeExp<float>(gt, decvec, d_k);
    } else {
        const float exp_g = std::exp(gt[0]);
        for (size_t i = 0; i < d_k; ++i) {
            decvec[i] = exp_g;
        }
    }
}

//
// Whether the single-pass form runs two tokens per sweep at this head count.
//
template <size_t NOUT, bool HAS_BETA>
struct LinearAttentionRvvPairs {
    static constexpr bool Value = !HAS_BETA && (NOUT <= 2);
};

//
// delta / gated_delta token loop with the fused sweep. The first token's
// pass 1 runs on its own; from then on each token's sweep writes S_new and
// produces the next token's pass-1 results, so the last token's sweep has
// nothing to accumulate and takes the HAS_NEXT = false body.
//
template <typename V, size_t NOUT, bool HAS_DECAY>
void
ProcessHeadTwoPassFusedRvv(
    const MLAS_LINEAR_ATTENTION_WORK* Work,
    float* __restrict dec_a,
    float* __restrict dec_b,
    float* __restrict wk_buf,
    float* __restrict wq_buf,
    float* __restrict r_buf,
    float* __restrict a_buf
)
{
    const size_t d_k = Work->KHeadSize;
    const size_t d_v = Work->VHeadSize;
    const size_t T = Work->SequenceLength;
    const float scale = Work->Scale;

    float* __restrict S = Work->State;

    //
    // dec_cur is this token's decay, applied to S in the sweep; dec_next is
    // the next token's, which only the pre-weighting reads. They swap roles
    // each token.
    //
    float* dec_cur = dec_a;
    float* dec_next = dec_b;

    const float* wkv;
    const float* wqv[NOUT];
    float qk[NOUT];

    if constexpr (HAS_DECAY) {
        LinearAttentionDecayRvv(Work, 0, dec_cur);
    }
    LinearAttentionWeightsRvv<NOUT, HAS_DECAY>(Work->Key, Work->Query, dec_cur, d_k,
                                               wk_buf, wq_buf, wkv, wqv);

    for (size_t j0 = 0; j0 < d_v;) {
        const size_t vl = V::SetVl(d_v - j0);
        Pass1PanelRvv<V, NOUT>(S, d_k, d_v, j0, vl, wkv, wqv, r_buf, a_buf);
        j0 += vl;
    }

    for (size_t t = 0; t < T; ++t) {
        const float* __restrict kt = Work->Key + t * Work->KeyTokenStride;
        const float* __restrict vt = Work->Value + t * Work->ValueTokenStride;
        const float* __restrict q0 = Work->Query + t * Work->QueryTokenStride;
        float* __restrict o0 = Work->Output + t * Work->OutputTokenStride;
        const float beta_val = Work->Beta[t * Work->BetaTokenStride];

        for (size_t g = 0; g < NOUT; ++g) {
            qk[g] = LinearAttentionDotRvv(q0 + g * d_k, kt, d_k);
        }

        const float* __restrict dec = HAS_DECAY ? dec_cur : nullptr;

        if (t + 1 < T) {
            const size_t u = t + 1;

            if constexpr (HAS_DECAY) {
                LinearAttentionDecayRvv(Work, u, dec_next);
            }
            LinearAttentionWeightsRvv<NOUT, HAS_DECAY>(
                Work->Key + u * Work->KeyTokenStride, Work->Query + u * Work->QueryTokenStride,
                dec_next, d_k, wk_buf, wq_buf, wkv, wqv);

            for (size_t j0 = 0; j0 < d_v;) {
                const size_t vl = V::SetVl(d_v - j0);
                FusedPassPanelRvv<V, NOUT, HAS_DECAY, true>(
                    S, d_k, d_v, j0, vl, dec, kt, vt, o0, scale, beta_val, qk, r_buf, a_buf,
                    wkv, wqv);
                j0 += vl;
            }

            float* swap = dec_cur;
            dec_cur = dec_next;
            dec_next = swap;
        } else {
            for (size_t j0 = 0; j0 < d_v;) {
                const size_t vl = V::SetVl(d_v - j0);
                FusedPassPanelRvv<V, NOUT, HAS_DECAY, false>(
                    S, d_k, d_v, j0, vl, dec, kt, vt, o0, scale, beta_val, qk, r_buf, a_buf,
                    nullptr, nullptr);
                j0 += vl;
            }
        }
    }
}

//
// Owns the whole token loop at a fixed (LMUL, head count, rule), so the rule
// dispatch and the staging buffers are resolved once per head rather than
// once per token.
//
template <typename V, size_t NOUT, bool HAS_DECAY, bool HAS_BETA>
void
ProcessHeadRvv(
    const MLAS_LINEAR_ATTENTION_WORK* Work
)
{
    constexpr size_t MaxK = MlasLinearAttentionRvvMaxKHeadSize;
    constexpr size_t MaxV = MlasLinearAttentionRvvMaxVHeadSize;
    constexpr bool kPairs = LinearAttentionRvvPairs<NOUT, HAS_BETA>::Value;

    const size_t d_k = Work->KHeadSize;
    const size_t d_v = Work->VHeadSize;
    const float scale = Work->Scale;

    float* __restrict S = Work->State;

    //
    // Decay staging: one d_k row from the caller's scratch, and a second on
    // the stack for the loops that hold two tokens' decay at once. Sized away
    // without decay.
    //
    float* __restrict decvec0 = Work->Scratch;  // d_k floats
    float decvec1_buf[(HAS_DECAY && (kPairs || HAS_BETA)) ? MaxK : 1];

    //
    // Two-pass staging: the decay-weighted rows, and the cross-token
    // retrieval / readout accumulators of the fused sweep. Sized away when
    // unused.
    //
    constexpr bool kStage = HAS_DECAY && HAS_BETA;
    float wk_buf[kStage ? MaxK : 1];
    float wq_buf[kStage ? NOUT * MaxK : 1];
    float r_buf[HAS_BETA ? MaxV : 1];
    float a_buf[HAS_BETA ? NOUT * MaxV : 1];

    if constexpr (HAS_BETA) {
        if (d_v <= MaxV) {
            ProcessHeadTwoPassFusedRvv<V, NOUT, HAS_DECAY>(Work, decvec0, decvec1_buf, wk_buf,
                                                           wq_buf, r_buf, a_buf);
            return;
        }
    }

    size_t t = 0;

    if constexpr (kPairs) {
        for (; t + 2 <= Work->SequenceLength; t += 2) {
            const size_t u = t + 1;

            const float* __restrict dec0 = nullptr;
            const float* __restrict dec1 = nullptr;
            if constexpr (HAS_DECAY) {
                LinearAttentionDecayRvv(Work, t, decvec0);
                LinearAttentionDecayRvv(Work, u, decvec1_buf);
                dec0 = decvec0;
                dec1 = decvec1_buf;
            }

            for (size_t j0 = 0; j0 < d_v;) {
                const size_t vl = V::SetVl(d_v - j0);
                SinglePassPanelPairRvv<V, NOUT, HAS_DECAY>(
                    S, d_k, d_v, j0, vl, dec0, dec1,
                    Work->Key + t * Work->KeyTokenStride,
                    Work->Key + u * Work->KeyTokenStride,
                    Work->Value + t * Work->ValueTokenStride,
                    Work->Value + u * Work->ValueTokenStride,
                    Work->Query + t * Work->QueryTokenStride,
                    Work->Query + u * Work->QueryTokenStride,
                    Work->Output + t * Work->OutputTokenStride,
                    Work->Output + u * Work->OutputTokenStride,
                    scale);
                j0 += vl;
            }
        }
    }

    for (; t < Work->SequenceLength; ++t) {
        const float* __restrict kt = Work->Key + t * Work->KeyTokenStride;
        const float* __restrict vt = Work->Value + t * Work->ValueTokenStride;
        const float* __restrict q0 = Work->Query + t * Work->QueryTokenStride;
        float* __restrict o0 = Work->Output + t * Work->OutputTokenStride;

        const float* __restrict dec = nullptr;
        if constexpr (HAS_DECAY) {
            LinearAttentionDecayRvv(Work, t, decvec0);
            dec = decvec0;
        }

        if constexpr (HAS_BETA) {
            //
            // d_v beyond the fused sweep's buffers: the unfused two-pass form,
            // which re-reads each panel from L1 instead.
            //
            const float beta_val = Work->Beta[t * Work->BetaTokenStride];

            const float* wkv;
            const float* wqv[NOUT];
            float qk[NOUT];

            for (size_t g = 0; g < NOUT; ++g) {
                qk[g] = LinearAttentionDotRvv(q0 + g * d_k, kt, d_k);
            }
            LinearAttentionWeightsRvv<NOUT, HAS_DECAY>(kt, q0, dec, d_k, wk_buf, wq_buf, wkv, wqv);

            for (size_t j0 = 0; j0 < d_v;) {
                const size_t vl = V::SetVl(d_v - j0);
                TwoPassPanelRvv<V, NOUT, HAS_DECAY>(S, d_k, d_v, j0, vl, dec, kt, vt, o0,
                                                    scale, beta_val, wkv, wqv, qk);
                j0 += vl;
            }
        } else {
            for (size_t j0 = 0; j0 < d_v;) {
                const size_t vl = V::SetVl(d_v - j0);
                SinglePassPanelRvv<V, NOUT, HAS_DECAY>(S, d_k, d_v, j0, vl, dec, kt, vt, q0,
                                                       o0, scale);
                j0 += vl;
            }
        }
    }
}

//
// Widest LMUL the live set fits at, against 32 vector registers, leaving at
// least one group free for the temporaries between the named values. The
// unpaired single pass holds NOUT + 2 groups, the fused two-pass sweep
// NOUT + 3, and the paired single pass 2 * NOUT + 3.
//
template <size_t NOUT, bool HAS_BETA>
struct LinearAttentionRvvMaxLmul {
    static constexpr size_t Groups =
        LinearAttentionRvvPairs<NOUT, HAS_BETA>::Value ? (2 * NOUT + 3)
        : HAS_BETA                                     ? (NOUT + 3)
                                                       : (NOUT + 2);
    static constexpr size_t Value = (Groups < 4) ? 8 : (Groups < 8) ? 4 : (Groups < 16) ? 2 : 1;
};

//
// Pick the LMUL for this head, then run the token loop at it. Starting from
// the narrowest group, widen until one group covers d_v or the register budget
// is reached. The unfused two-pass form then narrows again while a d_k x vl
// panel would overflow the L1 budget its second pass depends on; every other
// form streams S once per sweep and has no such bound.
//
template <size_t NOUT, bool HAS_DECAY, bool HAS_BETA>
void
ProcessHeadRvvSelectLmul(
    const MLAS_LINEAR_ATTENTION_WORK* Work
)
{
    constexpr size_t MaxLmul = LinearAttentionRvvMaxLmul<NOUT, HAS_BETA>::Value;

    const size_t d_k = Work->KHeadSize;
    const size_t d_v = Work->VHeadSize;
    const size_t vlmax_m1 = __riscv_vsetvlmax_e32m1();

    size_t lmul = 1;
    while (lmul < MaxLmul && lmul * vlmax_m1 < d_v) {
        lmul *= 2;
    }
    if constexpr (HAS_BETA) {
        if (d_v > MlasLinearAttentionRvvMaxVHeadSize) {
            while (lmul > 1 && lmul * vlmax_m1 * d_k > MlasLinearAttentionRvvPanelBudget) {
                lmul /= 2;
            }
        }
    }

    if constexpr (MaxLmul >= 8) {
        if (lmul == 8) {
            ProcessHeadRvv<RvvF32M8, NOUT, HAS_DECAY, HAS_BETA>(Work);
            return;
        }
    }
    if constexpr (MaxLmul >= 4) {
        if (lmul == 4) {
            ProcessHeadRvv<RvvF32M4, NOUT, HAS_DECAY, HAS_BETA>(Work);
            return;
        }
    }
    if (lmul == 2) {
        ProcessHeadRvv<RvvF32M2, NOUT, HAS_DECAY, HAS_BETA>(Work);
        return;
    }

    ProcessHeadRvv<RvvF32M1, NOUT, HAS_DECAY, HAS_BETA>(Work);
}

template <size_t NOUT>
void
ProcessHeadRvvSelectRule(
    const MLAS_LINEAR_ATTENTION_WORK* Work
)
{
    switch (Work->Rule) {
        case MlasLinearAttentionRuleLinear:
            ProcessHeadRvvSelectLmul<NOUT, false, false>(Work);
            return;
        case MlasLinearAttentionRuleGated:
            ProcessHeadRvvSelectLmul<NOUT, true, false>(Work);
            return;
        case MlasLinearAttentionRuleDelta:
            ProcessHeadRvvSelectLmul<NOUT, false, true>(Work);
            return;
        case MlasLinearAttentionRuleGatedDelta:
            ProcessHeadRvvSelectLmul<NOUT, true, true>(Work);
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

}  // namespace

void
MlasLinearAttentionProcessHeadRvv(
    const MLAS_LINEAR_ATTENTION_WORK* Work
)
{
    const size_t d_k = Work->KHeadSize;
    const size_t n_out = Work->HeadsPerGroup;

    //
    // d_k bounds the fixed-size staging buffers. There is no constraint on
    // d_v: vsetvl makes a partial panel an ordinary iteration, so the d_v % 32
    // restriction the NEON kernel carries has no analogue here, and no d_k
    // alignment either, since the reductions over d_k are vl-driven too.
    //
    const bool shape_ok = (d_k >= 1) && (d_k <= MlasLinearAttentionRvvMaxKHeadSize);

    if (!shape_ok) {
        MlasLinearAttentionProcessHead(Work);
        return;
    }

    switch (n_out) {
        case 1:
            ProcessHeadRvvSelectRule<1>(Work);
            return;
        case 2:
            ProcessHeadRvvSelectRule<2>(Work);
            return;
        case 4:
            ProcessHeadRvvSelectRule<4>(Work);
            return;
        case 8:
            ProcessHeadRvvSelectRule<8>(Work);
            return;
        default:
            MlasLinearAttentionProcessHead(Work);
            return;
    }
}

//
// Kernel dispatch structure definition.
//
const MLAS_LINEAR_ATTENTION_DISPATCH MlasLinearAttentionDispatchRvv = {
    MlasLinearAttentionProcessHeadRvv
};

#endif  // MLAS_USE_RVV
