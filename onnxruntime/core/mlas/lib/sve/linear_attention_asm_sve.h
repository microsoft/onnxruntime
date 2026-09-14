/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    linear_attention_asm_sve.h

Abstract:

    Hand-scheduled innermost K loops for the SVE linear attention kernel.

    Why these are not intrinsics. The loops want SVE's indexed FMA, which takes
    the multiplier from a lane of a quad that one LD1RQW loads for four i steps
    at once. Expressing that through intrinsics was tried twice and both gcc
    13.3 and clang 18.1.3 spilled Z registers inside the loop, which costs far
    more than the broadcasts it saves:

        variant                          in-loop ldr z / str z
        intrinsics, naive unroll         gcc 125   clang 298
        intrinsics, one live S vector    gcc  98   clang 375

    The second variant already carries the correct schedule -- load one state
    vector, update it, store it, feed every readout head, discard it -- and the
    peak live set is only 20-21 Z registers at NOUT=1. The compilers simply do
    not hold that allocation. So the register assignment is written out here.

    THE KEY INVARIANT, which the failed intrinsics versions violated: unrolling
    by four packs four WEIGHTS into a quad. It does NOT require four complete i
    iterations to be live at once. Only one S vector is ever in flight.

    Register map, single-pass NOUT=1 (peak 21 of 32):

        z0        four key weights      (indexed multiplier, must be z0-z7)
        z1        four query weights    (indexed multiplier, must be z0-z7)
        z2        four decay weights    (indexed multiplier, must be z0-z7)
        z3        scale broadcast, epilogue only
        z8-z15    eight readout accumulators, one per column vector
        z16-z23   eight persistent value vectors
        z24       current S vector
        z25       arithmetic temporary, decay form only
        p0        all-true; this body serves FULL panels only

    Eight column vectors fit the LD1W/ST1W `mul vl` immediate range exactly
    (-8..7), so every lane is addressed off one base pointer with no extra
    scalar arithmetic.

    The trailing partial panel keeps the intrinsics path: it needs one predicate
    per lane, runs at most once per row, and is not worth a second body.

--*/

#pragma once

#include <arm_sve.h>

#if !defined(MLAS_FORCEINLINE)
#define MLAS_FORCEINLINE __attribute__((always_inline)) inline
#endif

#if !defined(MLAS_LA_RESTRICT)
#define MLAS_LA_RESTRICT __restrict
#endif

//
// One column vector of one sub-iteration.
//
// Decay form:     S' = v*k[h] + S*dec[h]      then  acc += S'*q[h]
// No-decay form:  S' = S + v*k[h]             then  acc += S'*q[h]
//
// L, ACC, VV and H arrive already stringified so the lane immediate lands in
// the instruction text as a literal, which is what the indexed encoding needs.
//
#define MLAS_LA_ASM_LANE_DECAY(L, ACC, VV, H)                                  \
    "ld1w  z24.s, p0/z, [%[st], #" L ", mul vl]\n"                             \
    "fmul  z25.s, " VV ".s, z0.s[" H "]\n"                                     \
    "fmla  z25.s, z24.s, z2.s[" H "]\n"                                        \
    "st1w  z25.s, p0, [%[st], #" L ", mul vl]\n"                               \
    "fmla  " ACC ".s, z25.s, z1.s[" H "]\n"

#define MLAS_LA_ASM_LANE_PLAIN(L, ACC, VV, H)                                  \
    "ld1w  z24.s, p0/z, [%[st], #" L ", mul vl]\n"                             \
    "fmla  z24.s, " VV ".s, z0.s[" H "]\n"                                     \
    "st1w  z24.s, p0, [%[st], #" L ", mul vl]\n"                               \
    "fmla  " ACC ".s, z24.s, z1.s[" H "]\n"

// Eight column vectors at one lane index.
#define MLAS_LA_ASM_LANES(M, H)                                                \
    M("0", "z8",  "z16", H) M("1", "z9",  "z17", H)                            \
    M("2", "z10", "z18", H) M("3", "z11", "z19", H)                            \
    M("4", "z12", "z20", H) M("5", "z13", "z21", H)                            \
    M("6", "z14", "z22", H) M("7", "z15", "z23", H)

// Four sub-iterations off one set of quads. The state pointer advances one row
// between them; the weight pointers advance once, after all four.
#define MLAS_LA_ASM_QUAD(M)                                                    \
    MLAS_LA_ASM_LANES(M, "0") "add %[st], %[st], %[dvb]\n"                     \
    MLAS_LA_ASM_LANES(M, "1") "add %[st], %[st], %[dvb]\n"                     \
    MLAS_LA_ASM_LANES(M, "2") "add %[st], %[st], %[dvb]\n"                     \
    MLAS_LA_ASM_LANES(M, "3") "add %[st], %[st], %[dvb]\n"

// Scalar remainder, one i at a time. LD1RW rather than a quad, so no caller row
// is ever read past its end.
#define MLAS_LA_ASM_TAIL_DECAY                                                 \
    "ld1rw z0.s, p0/z, [%[kt]]\n"                                              \
    "ld1rw z2.s, p0/z, [%[dec]]\n"                                             \
    "ld1rw z1.s, p0/z, [%[q0]]\n"                                              \
    MLAS_LA_ASM_LANES(MLAS_LA_ASM_LANE_DECAY, "0")                             \
    "add %[st], %[st], %[dvb]\n"                                               \
    "add %[kt], %[kt], #4\n"                                                   \
    "add %[dec], %[dec], #4\n"                                                 \
    "add %[q0], %[q0], #4\n"

#define MLAS_LA_ASM_TAIL_PLAIN                                                 \
    "ld1rw z0.s, p0/z, [%[kt]]\n"                                              \
    "ld1rw z1.s, p0/z, [%[q0]]\n"                                              \
    MLAS_LA_ASM_LANES(MLAS_LA_ASM_LANE_PLAIN, "0")                             \
    "add %[st], %[st], %[dvb]\n"                                               \
    "add %[kt], %[kt], #4\n"                                                   \
    "add %[q0], %[q0], #4\n"

//
// Prologue: all-true predicate, zeroed accumulators, eight value vectors held
// for the whole panel.
//
#define MLAS_LA_ASM_PROLOGUE                                                   \
    "ptrue p0.s\n"                                                             \
    "dup z8.s,  #0\n" "dup z9.s,  #0\n" "dup z10.s, #0\n" "dup z11.s, #0\n"    \
    "dup z12.s, #0\n" "dup z13.s, #0\n" "dup z14.s, #0\n" "dup z15.s, #0\n"    \
    "ld1w z16.s, p0/z, [%[vt], #0, mul vl]\n"                                  \
    "ld1w z17.s, p0/z, [%[vt], #1, mul vl]\n"                                  \
    "ld1w z18.s, p0/z, [%[vt], #2, mul vl]\n"                                  \
    "ld1w z19.s, p0/z, [%[vt], #3, mul vl]\n"                                  \
    "ld1w z20.s, p0/z, [%[vt], #4, mul vl]\n"                                  \
    "ld1w z21.s, p0/z, [%[vt], #5, mul vl]\n"                                  \
    "ld1w z22.s, p0/z, [%[vt], #6, mul vl]\n"                                  \
    "ld1w z23.s, p0/z, [%[vt], #7, mul vl]\n"

//
// Epilogue: scale and write the eight readout vectors.
//
#define MLAS_LA_ASM_EPILOGUE                                                   \
    "ld1rw z3.s, p0/z, [%[scale]]\n"                                           \
    "fmul z8.s,  z8.s,  z3.s\n" "st1w z8.s,  p0, [%[o], #0, mul vl]\n"         \
    "fmul z9.s,  z9.s,  z3.s\n" "st1w z9.s,  p0, [%[o], #1, mul vl]\n"         \
    "fmul z10.s, z10.s, z3.s\n" "st1w z10.s, p0, [%[o], #2, mul vl]\n"         \
    "fmul z11.s, z11.s, z3.s\n" "st1w z11.s, p0, [%[o], #3, mul vl]\n"         \
    "fmul z12.s, z12.s, z3.s\n" "st1w z12.s, p0, [%[o], #4, mul vl]\n"         \
    "fmul z13.s, z13.s, z3.s\n" "st1w z13.s, p0, [%[o], #5, mul vl]\n"         \
    "fmul z14.s, z14.s, z3.s\n" "st1w z14.s, p0, [%[o], #6, mul vl]\n"         \
    "fmul z15.s, z15.s, z3.s\n" "st1w z15.s, p0, [%[o], #7, mul vl]\n"

#define MLAS_LA_ASM_CLOBBERS                                                   \
    "z0", "z1", "z2", "z3", "z8", "z9", "z10", "z11", "z12", "z13", "z14",     \
    "z15", "z16", "z17", "z18", "z19", "z20", "z21", "z22", "z23", "z24",      \
    "z25", "p0", "cc", "memory"

//
// Single-pass (linear / gated), NOUT == 1, FULL panel of exactly eight column
// vectors. Owns the whole panel -- value loads, the K loop, and the scaled
// output store -- so nothing has to be handed across the asm boundary in a
// particular register.
//
template <bool HAS_DECAY>
MLAS_FORCEINLINE
void
MlasLinearAttentionSveSinglePanelAsmN1(
    float* MLAS_LA_RESTRICT St,          // S + j0, advanced by the body
    const float* MLAS_LA_RESTRICT kt,
    const float* MLAS_LA_RESTRICT dec,
    const float* MLAS_LA_RESTRICT q0,
    const float* MLAS_LA_RESTRICT vt,    // vt + j0
    float* MLAS_LA_RESTRICT o,           // o_base + j0
    size_t d_k,
    size_t d_v_bytes,                    // d_v * sizeof(float)
    const float* MLAS_LA_RESTRICT scale
)
{
    size_t quads = d_k >> 2;
    size_t rem = d_k & 3;

    if constexpr (HAS_DECAY) {
        __asm__ volatile(
            MLAS_LA_ASM_PROLOGUE
            "cbz %[quads], 2f\n"
            "1:\n"
            "ld1rqw z0.s, p0/z, [%[kt]]\n"
            "ld1rqw z2.s, p0/z, [%[dec]]\n"
            "ld1rqw z1.s, p0/z, [%[q0]]\n"
            MLAS_LA_ASM_QUAD(MLAS_LA_ASM_LANE_DECAY)
            "add %[kt],  %[kt],  #16\n"
            "add %[dec], %[dec], #16\n"
            "add %[q0],  %[q0],  #16\n"
            "subs %[quads], %[quads], #1\n"
            "b.ne 1b\n"
            "2:\n"
            "cbz %[rem], 4f\n"
            "3:\n"
            MLAS_LA_ASM_TAIL_DECAY
            "subs %[rem], %[rem], #1\n"
            "b.ne 3b\n"
            "4:\n"
            MLAS_LA_ASM_EPILOGUE
            : [st] "+&r"(St), [kt] "+&r"(kt), [dec] "+&r"(dec), [q0] "+&r"(q0),
              [quads] "+&r"(quads), [rem] "+&r"(rem)
            : [vt] "r"(vt), [o] "r"(o), [dvb] "r"(d_v_bytes), [scale] "r"(scale)
            : MLAS_LA_ASM_CLOBBERS);
    } else {
        (void)dec;
        __asm__ volatile(
            MLAS_LA_ASM_PROLOGUE
            "cbz %[quads], 2f\n"
            "1:\n"
            "ld1rqw z0.s, p0/z, [%[kt]]\n"
            "ld1rqw z1.s, p0/z, [%[q0]]\n"
            MLAS_LA_ASM_QUAD(MLAS_LA_ASM_LANE_PLAIN)
            "add %[kt], %[kt], #16\n"
            "add %[q0], %[q0], #16\n"
            "subs %[quads], %[quads], #1\n"
            "b.ne 1b\n"
            "2:\n"
            "cbz %[rem], 4f\n"
            "3:\n"
            MLAS_LA_ASM_TAIL_PLAIN
            "subs %[rem], %[rem], #1\n"
            "b.ne 3b\n"
            "4:\n"
            MLAS_LA_ASM_EPILOGUE
            : [st] "+&r"(St), [kt] "+&r"(kt), [q0] "+&r"(q0),
              [quads] "+&r"(quads), [rem] "+&r"(rem)
            : [vt] "r"(vt), [o] "r"(o), [dvb] "r"(d_v_bytes), [scale] "r"(scale)
            : MLAS_LA_ASM_CLOBBERS);
    }
}

//
// Token-blocked single-pass body: two tokens per sweep of S.
//
// The kernel is bound by L1 port throughput, not misses -- measured 0.5% L1D
// refill rate with 2.3 sustained accesses/cycle -- so the only lever left is
// issuing fewer loads and stores per token. The recurrence composes in
// registers:
//
//     load S                       (one LD1W)
//     S'  = v0*k0[h] + S *d0[h] ;  acc0 += S' *q0[h]     token t
//     S'' = v1*k1[h] + S'*d1[h] ;  acc1 += S''*q1[h]     token t+1
//     store S''                    (one ST1W)
//
// One load and one store now serve two tokens, halving the state traffic per
// token; the FMA count per token is unchanged. The per-element operation order
// is identical to the serial kernel -- S' simply never round-trips through
// memory -- so the result is bit-exact with the unblocked path.
//
// Register map (peak 26 of 32; the no-decay form skips z2/z5 for 24):
//
//     z0,z1,z2   k/q/dec quads, token t     (indexed multipliers, z0-z7)
//     z3,z4,z5   k/q/dec quads, token t+1   (indexed multipliers, z0-z7)
//     z6         scale broadcast, epilogue only
//     z8-z11     4 readout accumulators, token t
//     z12-z15    4 readout accumulators, token t+1
//     z16-z19    4 persistent value vectors, token t
//     z20-z23    4 persistent value vectors, token t+1
//     z24,z25    S in flight + arithmetic temporary
//
// Two tokens need twice the value vectors and accumulators, which is what
// forces the panel down from eight column vectors to four. Per-token traffic
// does not depend on the panel width -- only the panel-loop trip count does --
// so nothing is lost by that.
//
#define MLAS_LA_ASM2_LANE_DECAY(L, A0, A1, V0, V1, H)                          \
    "ld1w  z24.s, p0/z, [%[st], #" L ", mul vl]\n"                             \
    "fmul  z25.s, " V0 ".s, z0.s[" H "]\n"                                     \
    "fmla  z25.s, z24.s, z2.s[" H "]\n"                                        \
    "fmla  " A0 ".s, z25.s, z1.s[" H "]\n"                                     \
    "fmul  z24.s, " V1 ".s, z3.s[" H "]\n"                                     \
    "fmla  z24.s, z25.s, z5.s[" H "]\n"                                        \
    "st1w  z24.s, p0, [%[st], #" L ", mul vl]\n"                               \
    "fmla  " A1 ".s, z24.s, z4.s[" H "]\n"

#define MLAS_LA_ASM2_LANE_PLAIN(L, A0, A1, V0, V1, H)                          \
    "ld1w  z24.s, p0/z, [%[st], #" L ", mul vl]\n"                             \
    "fmla  z24.s, " V0 ".s, z0.s[" H "]\n"                                     \
    "fmla  " A0 ".s, z24.s, z1.s[" H "]\n"                                     \
    "fmla  z24.s, " V1 ".s, z3.s[" H "]\n"                                     \
    "st1w  z24.s, p0, [%[st], #" L ", mul vl]\n"                               \
    "fmla  " A1 ".s, z24.s, z4.s[" H "]\n"

// Four column vectors at one lane index.
#define MLAS_LA_ASM2_LANES(M, H)                                               \
    M("0", "z8",  "z12", "z16", "z20", H)                                      \
    M("1", "z9",  "z13", "z17", "z21", H)                                      \
    M("2", "z10", "z14", "z18", "z22", H)                                      \
    M("3", "z11", "z15", "z19", "z23", H)

#define MLAS_LA_ASM2_QUAD(M)                                                   \
    MLAS_LA_ASM2_LANES(M, "0") "add %[st], %[st], %[dvb]\n"                    \
    MLAS_LA_ASM2_LANES(M, "1") "add %[st], %[st], %[dvb]\n"                    \
    MLAS_LA_ASM2_LANES(M, "2") "add %[st], %[st], %[dvb]\n"                    \
    MLAS_LA_ASM2_LANES(M, "3") "add %[st], %[st], %[dvb]\n"

#define MLAS_LA_ASM2_PROLOGUE                                                  \
    "ptrue p0.s\n"                                                             \
    "dup z8.s,  #0\n" "dup z9.s,  #0\n" "dup z10.s, #0\n" "dup z11.s, #0\n"    \
    "dup z12.s, #0\n" "dup z13.s, #0\n" "dup z14.s, #0\n" "dup z15.s, #0\n"    \
    "ld1w z16.s, p0/z, [%[vt0], #0, mul vl]\n"                                 \
    "ld1w z17.s, p0/z, [%[vt0], #1, mul vl]\n"                                 \
    "ld1w z18.s, p0/z, [%[vt0], #2, mul vl]\n"                                 \
    "ld1w z19.s, p0/z, [%[vt0], #3, mul vl]\n"                                 \
    "ld1w z20.s, p0/z, [%[vt1], #0, mul vl]\n"                                 \
    "ld1w z21.s, p0/z, [%[vt1], #1, mul vl]\n"                                 \
    "ld1w z22.s, p0/z, [%[vt1], #2, mul vl]\n"                                 \
    "ld1w z23.s, p0/z, [%[vt1], #3, mul vl]\n"

#define MLAS_LA_ASM2_EPILOGUE                                                  \
    "ld1rw z6.s, p0/z, [%[scale]]\n"                                           \
    "fmul z8.s,  z8.s,  z6.s\n" "st1w z8.s,  p0, [%[o0], #0, mul vl]\n"        \
    "fmul z9.s,  z9.s,  z6.s\n" "st1w z9.s,  p0, [%[o0], #1, mul vl]\n"        \
    "fmul z10.s, z10.s, z6.s\n" "st1w z10.s, p0, [%[o0], #2, mul vl]\n"        \
    "fmul z11.s, z11.s, z6.s\n" "st1w z11.s, p0, [%[o0], #3, mul vl]\n"        \
    "fmul z12.s, z12.s, z6.s\n" "st1w z12.s, p0, [%[o1], #0, mul vl]\n"        \
    "fmul z13.s, z13.s, z6.s\n" "st1w z13.s, p0, [%[o1], #1, mul vl]\n"        \
    "fmul z14.s, z14.s, z6.s\n" "st1w z14.s, p0, [%[o1], #2, mul vl]\n"        \
    "fmul z15.s, z15.s, z6.s\n" "st1w z15.s, p0, [%[o1], #3, mul vl]\n"

#define MLAS_LA_ASM2_CLOBBERS                                                  \
    "z0", "z1", "z2", "z3", "z4", "z5", "z6", "z8", "z9", "z10", "z11",        \
    "z12", "z13", "z14", "z15", "z16", "z17", "z18", "z19", "z20", "z21",      \
    "z22", "z23", "z24", "z25", "p0", "cc", "memory"

//
// Token-blocked single-pass, NOUT == 1, FULL panel of exactly four column
// vectors, two consecutive tokens per call.
//
template <bool HAS_DECAY>
MLAS_FORCEINLINE
void
MlasLinearAttentionSveSinglePanelAsmN1x2(
    float* MLAS_LA_RESTRICT St,          // S + j0, advanced by the body
    const float* MLAS_LA_RESTRICT kt0,
    const float* MLAS_LA_RESTRICT dec0,
    const float* MLAS_LA_RESTRICT q0,
    const float* MLAS_LA_RESTRICT vt0,   // vt(t)   + j0
    float* MLAS_LA_RESTRICT o0,          // o(t)    + j0
    const float* MLAS_LA_RESTRICT kt1,
    const float* MLAS_LA_RESTRICT dec1,
    const float* MLAS_LA_RESTRICT q1,
    const float* MLAS_LA_RESTRICT vt1,   // vt(t+1) + j0
    float* MLAS_LA_RESTRICT o1,          // o(t+1)  + j0
    size_t d_k,
    size_t d_v_bytes,
    const float* MLAS_LA_RESTRICT scale
)
{
    size_t quads = d_k >> 2;
    size_t rem = d_k & 3;

    if constexpr (HAS_DECAY) {
        __asm__ volatile(
            MLAS_LA_ASM2_PROLOGUE
            "cbz %[quads], 2f\n"
            "1:\n"
            "ld1rqw z0.s, p0/z, [%[kt0]]\n"
            "ld1rqw z1.s, p0/z, [%[q0]]\n"
            "ld1rqw z2.s, p0/z, [%[dec0]]\n"
            "ld1rqw z3.s, p0/z, [%[kt1]]\n"
            "ld1rqw z4.s, p0/z, [%[q1]]\n"
            "ld1rqw z5.s, p0/z, [%[dec1]]\n"
            MLAS_LA_ASM2_QUAD(MLAS_LA_ASM2_LANE_DECAY)
            "add %[kt0],  %[kt0],  #16\n"
            "add %[q0],   %[q0],   #16\n"
            "add %[dec0], %[dec0], #16\n"
            "add %[kt1],  %[kt1],  #16\n"
            "add %[q1],   %[q1],   #16\n"
            "add %[dec1], %[dec1], #16\n"
            "subs %[quads], %[quads], #1\n"
            "b.ne 1b\n"
            "2:\n"
            "cbz %[rem], 4f\n"
            "3:\n"
            "ld1rw z0.s, p0/z, [%[kt0]]\n"
            "ld1rw z1.s, p0/z, [%[q0]]\n"
            "ld1rw z2.s, p0/z, [%[dec0]]\n"
            "ld1rw z3.s, p0/z, [%[kt1]]\n"
            "ld1rw z4.s, p0/z, [%[q1]]\n"
            "ld1rw z5.s, p0/z, [%[dec1]]\n"
            MLAS_LA_ASM2_LANES(MLAS_LA_ASM2_LANE_DECAY, "0")
            "add %[st], %[st], %[dvb]\n"
            "add %[kt0],  %[kt0],  #4\n"
            "add %[q0],   %[q0],   #4\n"
            "add %[dec0], %[dec0], #4\n"
            "add %[kt1],  %[kt1],  #4\n"
            "add %[q1],   %[q1],   #4\n"
            "add %[dec1], %[dec1], #4\n"
            "subs %[rem], %[rem], #1\n"
            "b.ne 3b\n"
            "4:\n"
            MLAS_LA_ASM2_EPILOGUE
            : [st] "+&r"(St), [kt0] "+&r"(kt0), [dec0] "+&r"(dec0), [q0] "+&r"(q0),
              [kt1] "+&r"(kt1), [dec1] "+&r"(dec1), [q1] "+&r"(q1),
              [quads] "+&r"(quads), [rem] "+&r"(rem)
            : [vt0] "r"(vt0), [vt1] "r"(vt1), [o0] "r"(o0), [o1] "r"(o1),
              [dvb] "r"(d_v_bytes), [scale] "r"(scale)
            : MLAS_LA_ASM2_CLOBBERS);
    } else {
        (void)dec0;
        (void)dec1;
        __asm__ volatile(
            MLAS_LA_ASM2_PROLOGUE
            "cbz %[quads], 2f\n"
            "1:\n"
            "ld1rqw z0.s, p0/z, [%[kt0]]\n"
            "ld1rqw z1.s, p0/z, [%[q0]]\n"
            "ld1rqw z3.s, p0/z, [%[kt1]]\n"
            "ld1rqw z4.s, p0/z, [%[q1]]\n"
            MLAS_LA_ASM2_QUAD(MLAS_LA_ASM2_LANE_PLAIN)
            "add %[kt0], %[kt0], #16\n"
            "add %[q0],  %[q0],  #16\n"
            "add %[kt1], %[kt1], #16\n"
            "add %[q1],  %[q1],  #16\n"
            "subs %[quads], %[quads], #1\n"
            "b.ne 1b\n"
            "2:\n"
            "cbz %[rem], 4f\n"
            "3:\n"
            "ld1rw z0.s, p0/z, [%[kt0]]\n"
            "ld1rw z1.s, p0/z, [%[q0]]\n"
            "ld1rw z3.s, p0/z, [%[kt1]]\n"
            "ld1rw z4.s, p0/z, [%[q1]]\n"
            MLAS_LA_ASM2_LANES(MLAS_LA_ASM2_LANE_PLAIN, "0")
            "add %[st], %[st], %[dvb]\n"
            "add %[kt0], %[kt0], #4\n"
            "add %[q0],  %[q0],  #4\n"
            "add %[kt1], %[kt1], #4\n"
            "add %[q1],  %[q1],  #4\n"
            "subs %[rem], %[rem], #1\n"
            "b.ne 3b\n"
            "4:\n"
            MLAS_LA_ASM2_EPILOGUE
            : [st] "+&r"(St), [kt0] "+&r"(kt0), [q0] "+&r"(q0),
              [kt1] "+&r"(kt1), [q1] "+&r"(q1),
              [quads] "+&r"(quads), [rem] "+&r"(rem)
            : [vt0] "r"(vt0), [vt1] "r"(vt1), [o0] "r"(o0), [o1] "r"(o1),
              [dvb] "r"(d_v_bytes), [scale] "r"(scale)
            : MLAS_LA_ASM2_CLOBBERS);
    }
}
