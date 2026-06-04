/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_sqnbitgemm_2bit.cpp

Abstract:

    Unit tests for the 2-bit AVX-512-VNNI weight-GEMM helpers.

    Phase 2a coverage: pack / unpack round-trip of the BlkLen=64 packed
    layout. The tests exercise sqnbitgemm_kernel_avx512_2bit.h directly;
    they do not depend on platform dispatch being wired up, so they run on
    every host (the helpers are pure scalar bit-twiddling).

--*/

#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include "core/mlas/lib/sqnbitgemm_kernel_avx512_2bit.h"

namespace {

namespace sq2 = onnxruntime::mlas::sq2bit_avx512;

// Standard ONNX 2-bit packing: byte_i = w[4i] | w[4i+1]<<2 | w[4i+2]<<4 | w[4i+3]<<6.
void
PackSourceBlock_BlkLen64(const uint8_t weights[sq2::kBlkLen], std::byte* src_out)
{
    for (size_t i = 0; i < sq2::kBlkBytes; ++i) {
        const uint8_t v0 = weights[4 * i + 0] & 0x03u;
        const uint8_t v1 = weights[4 * i + 1] & 0x03u;
        const uint8_t v2 = weights[4 * i + 2] & 0x03u;
        const uint8_t v3 = weights[4 * i + 3] & 0x03u;
        src_out[i] = static_cast<std::byte>(
            static_cast<uint8_t>(v0 | (v1 << 2) | (v2 << 4) | (v3 << 6))
        );
    }
}

}  // namespace

//
// Pack then immediately unpack a single block. Each 2-bit position must
// survive the layout permutation exactly.
//
TEST(MlasSq2BitTest, PackUnpackRoundTrip_BlkLen64_DeterministicPattern)
{
    // Use a deterministic pattern that touches every (position, value) pair:
    // weight i gets value (i % 4). This guarantees that if any bit-position
    // accounting is off the failing index pinpoints the bug.
    std::array<uint8_t, sq2::kBlkLen> weights{};
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<uint8_t>(i % 4);
    }

    std::array<std::byte, sq2::kBlkBytes> src{};
    PackSourceBlock_BlkLen64(weights.data(), src.data());

    // Sanity check: the source unpack reproduces the original weights.
    std::array<uint8_t, sq2::kBlkLen> via_src{};
    sq2::UnpackSourceBlock_BlkLen64_Reference(src.data(), via_src.data());
    for (size_t i = 0; i < weights.size(); ++i) {
        ASSERT_EQ(via_src[i], weights[i]) << "Source-unpack disagrees at i=" << i;
    }

    // Pack into the new layout, then unpack via the reference inverse.
    std::array<std::byte, sq2::kPackedBlkBytes> packed{};
    sq2::PackBlock_BlkLen64(src.data(), packed.data());

    std::array<uint8_t, sq2::kBlkLen> recovered{};
    sq2::UnpackBlock_BlkLen64_Reference(packed.data(), recovered.data());

    for (size_t i = 0; i < weights.size(); ++i) {
        ASSERT_EQ(recovered[i], weights[i])
            << "Round-trip mismatch at i=" << i
            << ": expected " << static_cast<int>(weights[i])
            << ", got " << static_cast<int>(recovered[i]);
    }
}

//
// Same round-trip but with pseudo-random weights, repeated across many
// blocks and seeds. Catches accidental position-dependent bugs that the
// deterministic pattern above might mask.
//
TEST(MlasSq2BitTest, PackUnpackRoundTrip_BlkLen64_Randomized)
{
    constexpr size_t kBlockCount = 17;  // arbitrary, > 1, not a SIMD-friendly number
    constexpr unsigned kSeeds = 8;

    for (unsigned seed = 0; seed < kSeeds; ++seed) {
        std::mt19937 rng(seed * 7919u + 1u);
        std::uniform_int_distribution<unsigned> dist(0u, 3u);

        std::vector<uint8_t> weights(kBlockCount * sq2::kBlkLen);
        for (auto& w : weights) {
            w = static_cast<uint8_t>(dist(rng));
        }

        std::vector<std::byte> src(kBlockCount * sq2::kBlkBytes);
        for (size_t blk = 0; blk < kBlockCount; ++blk) {
            PackSourceBlock_BlkLen64(weights.data() + blk * sq2::kBlkLen,
                                     src.data() + blk * sq2::kBlkBytes);
        }

        std::vector<std::byte> packed(kBlockCount * sq2::kPackedBlkBytes);
        for (size_t blk = 0; blk < kBlockCount; ++blk) {
            sq2::PackBlock_BlkLen64(src.data() + blk * sq2::kBlkBytes,
                                    packed.data() + blk * sq2::kPackedBlkBytes);
        }

        for (size_t blk = 0; blk < kBlockCount; ++blk) {
            std::array<uint8_t, sq2::kBlkLen> recovered{};
            sq2::UnpackBlock_BlkLen64_Reference(packed.data() + blk * sq2::kPackedBlkBytes,
                                                recovered.data());
            for (size_t i = 0; i < sq2::kBlkLen; ++i) {
                ASSERT_EQ(recovered[i], weights[blk * sq2::kBlkLen + i])
                    << "Random round-trip mismatch seed=" << seed
                    << " blk=" << blk
                    << " i=" << i;
            }
        }
    }
}

//
// Bit-position invariant: writing only value v into every weight slot must
// yield a packed buffer where every byte is 0x55 * v (== v repeated at
// positions 0,2,4,6). This catches confusion between low/high nibbles or
// reversed-bit packing.
//
TEST(MlasSq2BitTest, PackUnpackRoundTrip_BlkLen64_ConstantValues)
{
    for (uint8_t v = 0; v < 4; ++v) {
        std::array<uint8_t, sq2::kBlkLen> weights{};
        weights.fill(v);

        std::array<std::byte, sq2::kBlkBytes> src{};
        PackSourceBlock_BlkLen64(weights.data(), src.data());

        std::array<std::byte, sq2::kPackedBlkBytes> packed{};
        sq2::PackBlock_BlkLen64(src.data(), packed.data());

        const uint8_t expected_byte = static_cast<uint8_t>(v * 0x55u);  // v at bits {0..1,2..3,4..5,6..7}
        for (size_t i = 0; i < sq2::kPackedBlkBytes; ++i) {
            ASSERT_EQ(static_cast<uint8_t>(packed[i]), expected_byte)
                << "Constant-fill v=" << static_cast<int>(v)
                << " byte_i=" << i;
        }

        std::array<uint8_t, sq2::kBlkLen> recovered{};
        sq2::UnpackBlock_BlkLen64_Reference(packed.data(), recovered.data());
        for (size_t i = 0; i < sq2::kBlkLen; ++i) {
            ASSERT_EQ(recovered[i], v) << "Constant-fill v=" << static_cast<int>(v) << " i=" << i;
        }
    }
}
