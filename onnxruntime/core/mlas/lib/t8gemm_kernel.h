// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/*++

Module Name:

    t8gemm_kernel.h

Abstract:

    Shared pieces of the portable, AVX2 and AVX-512F T8 ternary GEMM kernels.

    Each T8 byte holds five trits in base 3 and is stored as eight bit planes,
    so one 32-bit plane word covers 32 consecutive inputs. The decode below is
    the bitsliced gate network used by the NEON kernel: it turns the eight
    plane words (or vectors of them) of a group directly into per-trit zero
    and positive masks. The kernels then use those masks to add or subtract A
    in natural input order.

    Everything here has internal linkage: the header is included by
    translation units compiled for different instruction sets.

--*/

#pragma once

#include "mlasi.h"

namespace
{

constexpr size_t T8GroupSize = 128;
constexpr size_t T8WordsPerGroup = 4;
constexpr size_t T8TritsPerUnit = 5;
constexpr size_t T8TiledWordsPerGroup = 32;
constexpr size_t T8MaskWordsPerGroup = 40;

//
// Bitsliced decode of eight T8 bit planes into five trits. Ops supplies the
// three-input gates for one vector type; Z receives the "trit is 0" masks and
// P the "trit is 1" masks.
//

template <typename Ops>
MLAS_FORCEINLINE void
T8Decode(
    typename Ops::V b0,
    typename Ops::V b1,
    typename Ops::V b2,
    typename Ops::V b3,
    typename Ops::V b4,
    typename Ops::V b5,
    typename Ops::V b6,
    typename Ops::V b7,
    typename Ops::V* z,
    typename Ops::V* p
)
{
    using V = typename Ops::V;

    V nt = Ops::Nt(b7, b6, b5), pt = Ops::Pt(b7, b6, b5, nt);
    b5 = Ops::Bq(b7, b6, b5);
    b6 = Ops::And(b6, b7);
    V n = Ops::Neg(nt, pt, b4), q = Ops::Pos(nt, pt, b4);
    b4 = Ops::Carry(nt, pt, b4);
    nt = Ops::Neg(n, q, b3);
    pt = Ops::Pos(n, q, b3);
    b3 = Ops::Carry(n, q, b3);
    n = Ops::Neg(nt, pt, b2);
    q = Ops::Pos(nt, pt, b2);
    b2 = Ops::Carry(nt, pt, b2);
    nt = Ops::Neg(n, q, b1);
    pt = Ops::Pos(n, q, b1);
    b1 = Ops::Carry(n, q, b1);
    z[0] = Ops::Zero(nt, pt, b0);
    p[0] = Ops::Pos(nt, pt, b0);
    b0 = Ops::Carry(nt, pt, b0);

    nt = Ops::Nt(b6, b5, b4);
    pt = Ops::Pt(b6, b5, b4, nt);
    b4 = Ops::Bq(b6, b5, b4);
    b5 = Ops::And(b5, b6);
    n = Ops::Neg(nt, pt, b3);
    q = Ops::Pos(nt, pt, b3);
    b3 = Ops::Carry(nt, pt, b3);
    nt = Ops::Neg(n, q, b2);
    pt = Ops::Pos(n, q, b2);
    b2 = Ops::Carry(n, q, b2);
    n = Ops::Neg(nt, pt, b1);
    q = Ops::Pos(nt, pt, b1);
    b1 = Ops::Carry(nt, pt, b1);
    z[1] = Ops::Zero(n, q, b0);
    p[1] = Ops::Pos(n, q, b0);
    b0 = Ops::Carry(n, q, b0);

    nt = Ops::Nt(b4, b3, b2);
    pt = Ops::Pt(b4, b3, b2, nt);
    b2 = Ops::Bq(b4, b3, b2);
    b3 = Ops::And(b3, b4);
    n = Ops::Neg(nt, pt, b1);
    q = Ops::Pos(nt, pt, b1);
    b1 = Ops::Carry(nt, pt, b1);
    z[2] = Ops::Zero(n, q, b0);
    p[2] = Ops::Pos(n, q, b0);
    b0 = Ops::Carry(n, q, b0);

    nt = Ops::Nt(b3, b2, b1);
    pt = Ops::Pt(b3, b2, b1, nt);
    V n4 = Ops::Bq(b3, b2, b1);
    z[3] = Ops::Zero(nt, pt, b0);
    p[3] = Ops::Pos(nt, pt, b0);
    V p4 = Ops::Carry(nt, pt, b0);
    z[4] = Ops::NotOr(n4, p4);
    p[4] = p4;
}

//
// Truth-table immediates for vpternlogd, with the operands in order a, b, c.
//

constexpr uint8_t T8TernA = 0xF0;
constexpr uint8_t T8TernB = 0xCC;
constexpr uint8_t T8TernC = 0xAA;

//
// Returns the address of word 0 of group g of plane b, for the two layouts
// that store bit planes. Each group's four words for one plane are adjacent.
//

MLAS_FORCEINLINE const uint32_t*
T8PlaneWords(
    const uint32_t* B,
    MLAS_T8_WEIGHT_LAYOUT Layout,
    size_t K,
    size_t unit,
    size_t g,
    size_t b
)
{
    const size_t groups = K / T8GroupSize;
    if (Layout == MLAS_T8_WEIGHT_LAYOUT::Planes) {
        const size_t sets = K / 32;
        return B + (unit * 8 + b) * sets + g * T8WordsPerGroup;
    }
    return B + (unit * groups + g) * T8TiledWordsPerGroup + b * T8WordsPerGroup;
}

}  // namespace
