/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    test_sqnbitgemm_2bit.cpp

Abstract:

    Unit tests for the 2-bit AVX-512 weight-GEMM pack/unpack helpers
    (block-group layout, BlkLen=64). Pure scalar bit-twiddling; runs on
    every host because the exercised helpers do not contain SIMD code.

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
void PackSourceBlock_BlkLen64(const uint8_t weights[sq2::kBlkLen], std::byte* src_out) {
  for (size_t i = 0; i < sq2::kBlkBytes; ++i) {
    const uint8_t v0 = weights[4 * i + 0] & 0x03u;
    const uint8_t v1 = weights[4 * i + 1] & 0x03u;
    const uint8_t v2 = weights[4 * i + 2] & 0x03u;
    const uint8_t v3 = weights[4 * i + 3] & 0x03u;
    src_out[i] = static_cast<std::byte>(
        static_cast<uint8_t>(v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)));
  }
}

}  // namespace

// -----------------------------------------------------------------------------
// block-group (4-K-block) round-trip tests
// -----------------------------------------------------------------------------
//
// These exercise PackBlockGroup_BlkLen64 / UnPackBlockGroup_BlkLen64_Reference,
// which underpin the fast-unpack path (single 64-byte load + 4 fixed
// shift-and-mask producing 4 block ZMMs).

//
// Deterministic-pattern block-group round-trip. Block k assigns weight i the
// value ((i + k) % 4), giving every (block_index, position, value) a unique
// fingerprint that pinpoints a layout swap if any.
//
TEST(MlasSq2BitTest, PackUnpackRoundTrip_BlockGroup_BlkLen64_DeterministicPattern) {
  std::array<std::array<uint8_t, sq2::kBlkLen>, sq2::kBlockGroupBlks> weights{};
  for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
    for (size_t i = 0; i < sq2::kBlkLen; ++i) {
      weights[k][i] = static_cast<uint8_t>((i + k) % 4);
    }
  }

  std::array<std::array<std::byte, sq2::kBlkBytes>, sq2::kBlockGroupBlks> src{};
  for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
    PackSourceBlock_BlkLen64(weights[k].data(), src[k].data());
  }

  std::array<std::byte, sq2::kBlockGroupBytes> packed{};
  sq2::PackBlockGroup_BlkLen64(src[0].data(), src[1].data(), src[2].data(), src[3].data(),
                               packed.data());

  std::array<std::array<uint8_t, sq2::kBlkLen>, sq2::kBlockGroupBlks> recovered{};
  sq2::UnPackBlockGroup_BlkLen64_Reference(packed.data(),
                                           recovered[0].data(), recovered[1].data(),
                                           recovered[2].data(), recovered[3].data());

  for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
    for (size_t i = 0; i < sq2::kBlkLen; ++i) {
      ASSERT_EQ(recovered[k][i], weights[k][i])
          << "block-group round-trip mismatch k=" << k << " i=" << i;
    }
  }
}

//
// Randomized block-group round-trip across several seeds. The four input
// blocks are independent random fills; the test fails fast if any (block,
// weight) entry is mis-routed by the packed-byte layout.
//
TEST(MlasSq2BitTest, PackUnpackRoundTrip_BlockGroup_BlkLen64_Randomized) {
  constexpr unsigned kSeeds = 8;

  for (unsigned seed = 0; seed < kSeeds; ++seed) {
    std::mt19937 rng(seed * 5051u + 13u);
    std::uniform_int_distribution<unsigned> dist(0u, 3u);

    std::array<std::array<uint8_t, sq2::kBlkLen>, sq2::kBlockGroupBlks> weights{};
    for (auto& blk : weights) {
      for (auto& w : blk) {
        w = static_cast<uint8_t>(dist(rng));
      }
    }

    std::array<std::array<std::byte, sq2::kBlkBytes>, sq2::kBlockGroupBlks> src{};
    for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
      PackSourceBlock_BlkLen64(weights[k].data(), src[k].data());
    }

    std::array<std::byte, sq2::kBlockGroupBytes> packed{};
    sq2::PackBlockGroup_BlkLen64(src[0].data(), src[1].data(), src[2].data(), src[3].data(),
                                 packed.data());

    std::array<std::array<uint8_t, sq2::kBlkLen>, sq2::kBlockGroupBlks> recovered{};
    sq2::UnPackBlockGroup_BlkLen64_Reference(packed.data(),
                                             recovered[0].data(), recovered[1].data(),
                                             recovered[2].data(), recovered[3].data());

    for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
      for (size_t i = 0; i < sq2::kBlkLen; ++i) {
        ASSERT_EQ(recovered[k][i], weights[k][i])
            << "Random block-group mismatch seed=" << seed << " k=" << k << " i=" << i;
      }
    }
  }
}

//
// Constant-value invariants for the block-group layout:
//   - All four blocks set to the same value v produces packed bytes equal to
//     0x55 * v (v repeated at bit positions {0..1,2..3,4..5,6..7}).
//   - Block_k set to value v with all other blocks zero produces packed bytes
//     equal to (v << (2*k)) -- exclusively occupying the k-th bit slot.
//
TEST(MlasSq2BitTest, PackUnpackRoundTrip_BlockGroup_BlkLen64_ConstantValues) {
  // Case 1: every block filled with v.
  for (uint8_t v = 0; v < 4; ++v) {
    std::array<std::array<uint8_t, sq2::kBlkLen>, sq2::kBlockGroupBlks> weights{};
    for (auto& blk : weights) {
      blk.fill(v);
    }

    std::array<std::array<std::byte, sq2::kBlkBytes>, sq2::kBlockGroupBlks> src{};
    for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
      PackSourceBlock_BlkLen64(weights[k].data(), src[k].data());
    }

    std::array<std::byte, sq2::kBlockGroupBytes> packed{};
    sq2::PackBlockGroup_BlkLen64(src[0].data(), src[1].data(), src[2].data(), src[3].data(),
                                 packed.data());

    const uint8_t expected_byte = static_cast<uint8_t>(v * 0x55u);
    for (size_t i = 0; i < sq2::kBlockGroupBytes; ++i) {
      ASSERT_EQ(static_cast<uint8_t>(packed[i]), expected_byte)
          << "Uniform-fill v=" << static_cast<int>(v) << " byte_i=" << i;
    }
  }

  // Case 2: only one block at a time carries a non-zero value.
  for (size_t target_k = 0; target_k < sq2::kBlockGroupBlks; ++target_k) {
    for (uint8_t v = 1; v < 4; ++v) {
      std::array<std::array<uint8_t, sq2::kBlkLen>, sq2::kBlockGroupBlks> weights{};
      weights[target_k].fill(v);

      std::array<std::array<std::byte, sq2::kBlkBytes>, sq2::kBlockGroupBlks> src{};
      for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
        PackSourceBlock_BlkLen64(weights[k].data(), src[k].data());
      }

      std::array<std::byte, sq2::kBlockGroupBytes> packed{};
      sq2::PackBlockGroup_BlkLen64(src[0].data(), src[1].data(), src[2].data(), src[3].data(),
                                   packed.data());

      const uint8_t expected_byte = static_cast<uint8_t>(v << (2 * target_k));
      for (size_t i = 0; i < sq2::kBlockGroupBytes; ++i) {
        ASSERT_EQ(static_cast<uint8_t>(packed[i]), expected_byte)
            << "Isolated block target_k=" << target_k
            << " v=" << static_cast<int>(v)
            << " byte_i=" << i;
      }

      std::array<std::array<uint8_t, sq2::kBlkLen>, sq2::kBlockGroupBlks> recovered{};
      sq2::UnPackBlockGroup_BlkLen64_Reference(
          packed.data(),
          recovered[0].data(), recovered[1].data(),
          recovered[2].data(), recovered[3].data());
      for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
        const uint8_t expect_val = (k == target_k) ? v : uint8_t{0};
        for (size_t i = 0; i < sq2::kBlkLen; ++i) {
          ASSERT_EQ(recovered[k][i], expect_val)
              << "Isolated round-trip target_k=" << target_k
              << " k=" << k << " v=" << static_cast<int>(v) << " i=" << i;
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Block-group round-trip tests for BlkLen=128.
// Mirror the BlkLen=64 tests above. Each block-group still aggregates 4
// K-blocks; only the per-block byte width doubles (16 -> 32) and the per-
// group byte total doubles (64 -> 128). The packing rule is identical so
// the same kinds of invariants hold.
// -----------------------------------------------------------------------------

namespace {
// Standard ONNX 2-bit source packing for a BlkLen=128 block (32 bytes).
void PackSourceBlock_BlkLen128(const uint8_t weights[sq2::kBlkLen128], std::byte* src_out) {
  for (size_t i = 0; i < sq2::kBlkBytes128; ++i) {
    const uint8_t v0 = weights[4 * i + 0] & 0x03u;
    const uint8_t v1 = weights[4 * i + 1] & 0x03u;
    const uint8_t v2 = weights[4 * i + 2] & 0x03u;
    const uint8_t v3 = weights[4 * i + 3] & 0x03u;
    src_out[i] = static_cast<std::byte>(
        static_cast<uint8_t>(v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)));
  }
}
}  // namespace

TEST(MlasSq2BitTest, PackUnpackRoundTrip_BlockGroup_BlkLen128_DeterministicPattern) {
  std::array<std::array<uint8_t, sq2::kBlkLen128>, sq2::kBlockGroupBlks> weights{};
  for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
    for (size_t i = 0; i < sq2::kBlkLen128; ++i) {
      weights[k][i] = static_cast<uint8_t>((i + k) % 4);
    }
  }

  std::array<std::array<std::byte, sq2::kBlkBytes128>, sq2::kBlockGroupBlks> src{};
  for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
    PackSourceBlock_BlkLen128(weights[k].data(), src[k].data());
  }

  std::array<std::byte, sq2::kBlockGroupBytes128> packed{};
  sq2::PackBlockGroup_BlkLen128(src[0].data(), src[1].data(), src[2].data(), src[3].data(),
                                packed.data());

  std::array<std::array<uint8_t, sq2::kBlkLen128>, sq2::kBlockGroupBlks> recovered{};
  sq2::UnPackBlockGroup_BlkLen128_Reference(packed.data(),
                                            recovered[0].data(), recovered[1].data(),
                                            recovered[2].data(), recovered[3].data());

  for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
    for (size_t i = 0; i < sq2::kBlkLen128; ++i) {
      ASSERT_EQ(recovered[k][i], weights[k][i])
          << "BlkLen128 round-trip mismatch k=" << k << " i=" << i;
    }
  }
}

TEST(MlasSq2BitTest, PackUnpackRoundTrip_BlockGroup_BlkLen128_Randomized) {
  constexpr unsigned kSeeds = 8;

  for (unsigned seed = 0; seed < kSeeds; ++seed) {
    std::mt19937 rng(seed * 5051u + 13u);
    std::uniform_int_distribution<unsigned> dist(0u, 3u);

    std::array<std::array<uint8_t, sq2::kBlkLen128>, sq2::kBlockGroupBlks> weights{};
    for (auto& blk : weights) {
      for (auto& w : blk) {
        w = static_cast<uint8_t>(dist(rng));
      }
    }

    std::array<std::array<std::byte, sq2::kBlkBytes128>, sq2::kBlockGroupBlks> src{};
    for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
      PackSourceBlock_BlkLen128(weights[k].data(), src[k].data());
    }

    std::array<std::byte, sq2::kBlockGroupBytes128> packed{};
    sq2::PackBlockGroup_BlkLen128(src[0].data(), src[1].data(), src[2].data(), src[3].data(),
                                  packed.data());

    std::array<std::array<uint8_t, sq2::kBlkLen128>, sq2::kBlockGroupBlks> recovered{};
    sq2::UnPackBlockGroup_BlkLen128_Reference(packed.data(),
                                              recovered[0].data(), recovered[1].data(),
                                              recovered[2].data(), recovered[3].data());

    for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
      for (size_t i = 0; i < sq2::kBlkLen128; ++i) {
        ASSERT_EQ(recovered[k][i], weights[k][i])
            << "BlkLen128 random round-trip seed=" << seed << " k=" << k << " i=" << i;
      }
    }
  }
}

TEST(MlasSq2BitTest, PackUnpackRoundTrip_BlockGroup_BlkLen128_ConstantValues) {
  // Case 1: every block filled with v.
  for (uint8_t v = 0; v < 4; ++v) {
    std::array<std::array<uint8_t, sq2::kBlkLen128>, sq2::kBlockGroupBlks> weights{};
    for (auto& blk : weights) {
      blk.fill(v);
    }

    std::array<std::array<std::byte, sq2::kBlkBytes128>, sq2::kBlockGroupBlks> src{};
    for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
      PackSourceBlock_BlkLen128(weights[k].data(), src[k].data());
    }

    std::array<std::byte, sq2::kBlockGroupBytes128> packed{};
    sq2::PackBlockGroup_BlkLen128(src[0].data(), src[1].data(), src[2].data(), src[3].data(),
                                  packed.data());

    const uint8_t expected_byte = static_cast<uint8_t>(v * 0x55u);
    for (size_t i = 0; i < sq2::kBlockGroupBytes128; ++i) {
      ASSERT_EQ(static_cast<uint8_t>(packed[i]), expected_byte)
          << "BlkLen128 uniform-fill v=" << static_cast<int>(v) << " byte_i=" << i;
    }
  }

  // Case 2: only one block at a time carries a non-zero value.
  for (size_t target_k = 0; target_k < sq2::kBlockGroupBlks; ++target_k) {
    for (uint8_t v = 1; v < 4; ++v) {
      std::array<std::array<uint8_t, sq2::kBlkLen128>, sq2::kBlockGroupBlks> weights{};
      weights[target_k].fill(v);

      std::array<std::array<std::byte, sq2::kBlkBytes128>, sq2::kBlockGroupBlks> src{};
      for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
        PackSourceBlock_BlkLen128(weights[k].data(), src[k].data());
      }

      std::array<std::byte, sq2::kBlockGroupBytes128> packed{};
      sq2::PackBlockGroup_BlkLen128(src[0].data(), src[1].data(), src[2].data(), src[3].data(),
                                    packed.data());

      const uint8_t expected_byte = static_cast<uint8_t>(v << (2 * target_k));
      for (size_t i = 0; i < sq2::kBlockGroupBytes128; ++i) {
        ASSERT_EQ(static_cast<uint8_t>(packed[i]), expected_byte)
            << "BlkLen128 isolated block target_k=" << target_k
            << " v=" << static_cast<int>(v) << " byte_i=" << i;
      }

      std::array<std::array<uint8_t, sq2::kBlkLen128>, sq2::kBlockGroupBlks> recovered{};
      sq2::UnPackBlockGroup_BlkLen128_Reference(
          packed.data(),
          recovered[0].data(), recovered[1].data(),
          recovered[2].data(), recovered[3].data());
      for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
        const uint8_t expect_val = (k == target_k) ? v : uint8_t{0};
        for (size_t i = 0; i < sq2::kBlkLen128; ++i) {
          ASSERT_EQ(recovered[k][i], expect_val)
              << "BlkLen128 isolated round-trip target_k=" << target_k
              << " k=" << k << " v=" << static_cast<int>(v) << " i=" << i;
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
// Block-group round-trip tests for BlkLen=32.
// Each block-group is still 4 K-blocks; per-block byte width is 8 (vs 16 for
// BlkLen=64 and 32 for BlkLen=128), per-group bytes is 32.
// -----------------------------------------------------------------------------

namespace {
void PackSourceBlock_BlkLen32(const uint8_t weights[sq2::kBlkLen32], std::byte* src_out) {
  for (size_t i = 0; i < sq2::kBlkBytes32; ++i) {
    const uint8_t v0 = weights[4 * i + 0] & 0x03u;
    const uint8_t v1 = weights[4 * i + 1] & 0x03u;
    const uint8_t v2 = weights[4 * i + 2] & 0x03u;
    const uint8_t v3 = weights[4 * i + 3] & 0x03u;
    src_out[i] = static_cast<std::byte>(
        static_cast<uint8_t>(v0 | (v1 << 2) | (v2 << 4) | (v3 << 6)));
  }
}
}  // namespace

TEST(MlasSq2BitTest, PackUnpackRoundTrip_BlockGroup_BlkLen32_DeterministicPattern) {
  std::array<std::array<uint8_t, sq2::kBlkLen32>, sq2::kBlockGroupBlks> weights{};
  for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
    for (size_t i = 0; i < sq2::kBlkLen32; ++i) {
      weights[k][i] = static_cast<uint8_t>((i + k) % 4);
    }
  }

  std::array<std::array<std::byte, sq2::kBlkBytes32>, sq2::kBlockGroupBlks> src{};
  for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
    PackSourceBlock_BlkLen32(weights[k].data(), src[k].data());
  }

  std::array<std::byte, sq2::kBlockGroupBytes32> packed{};
  sq2::PackBlockGroup_BlkLen32(src[0].data(), src[1].data(), src[2].data(), src[3].data(),
                               packed.data());

  std::array<std::array<uint8_t, sq2::kBlkLen32>, sq2::kBlockGroupBlks> recovered{};
  sq2::UnPackBlockGroup_BlkLen32_Reference(packed.data(),
                                           recovered[0].data(), recovered[1].data(),
                                           recovered[2].data(), recovered[3].data());

  for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
    for (size_t i = 0; i < sq2::kBlkLen32; ++i) {
      ASSERT_EQ(recovered[k][i], weights[k][i])
          << "BlkLen32 round-trip mismatch k=" << k << " i=" << i;
    }
  }
}

TEST(MlasSq2BitTest, PackUnpackRoundTrip_BlockGroup_BlkLen32_Randomized) {
  constexpr unsigned kSeeds = 8;

  for (unsigned seed = 0; seed < kSeeds; ++seed) {
    std::mt19937 rng(seed * 5051u + 13u);
    std::uniform_int_distribution<unsigned> dist(0u, 3u);

    std::array<std::array<uint8_t, sq2::kBlkLen32>, sq2::kBlockGroupBlks> weights{};
    for (auto& blk : weights) {
      for (auto& w : blk) {
        w = static_cast<uint8_t>(dist(rng));
      }
    }

    std::array<std::array<std::byte, sq2::kBlkBytes32>, sq2::kBlockGroupBlks> src{};
    for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
      PackSourceBlock_BlkLen32(weights[k].data(), src[k].data());
    }

    std::array<std::byte, sq2::kBlockGroupBytes32> packed{};
    sq2::PackBlockGroup_BlkLen32(src[0].data(), src[1].data(), src[2].data(), src[3].data(),
                                 packed.data());

    std::array<std::array<uint8_t, sq2::kBlkLen32>, sq2::kBlockGroupBlks> recovered{};
    sq2::UnPackBlockGroup_BlkLen32_Reference(packed.data(),
                                             recovered[0].data(), recovered[1].data(),
                                             recovered[2].data(), recovered[3].data());

    for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
      for (size_t i = 0; i < sq2::kBlkLen32; ++i) {
        ASSERT_EQ(recovered[k][i], weights[k][i])
            << "BlkLen32 random round-trip seed=" << seed << " k=" << k << " i=" << i;
      }
    }
  }
}

TEST(MlasSq2BitTest, PackUnpackRoundTrip_BlockGroup_BlkLen32_ConstantValues) {
  for (uint8_t v = 0; v < 4; ++v) {
    std::array<std::array<uint8_t, sq2::kBlkLen32>, sq2::kBlockGroupBlks> weights{};
    for (auto& blk : weights) {
      blk.fill(v);
    }

    std::array<std::array<std::byte, sq2::kBlkBytes32>, sq2::kBlockGroupBlks> src{};
    for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
      PackSourceBlock_BlkLen32(weights[k].data(), src[k].data());
    }

    std::array<std::byte, sq2::kBlockGroupBytes32> packed{};
    sq2::PackBlockGroup_BlkLen32(src[0].data(), src[1].data(), src[2].data(), src[3].data(),
                                 packed.data());

    const uint8_t expected_byte = static_cast<uint8_t>(v * 0x55u);
    for (size_t i = 0; i < sq2::kBlockGroupBytes32; ++i) {
      ASSERT_EQ(static_cast<uint8_t>(packed[i]), expected_byte)
          << "BlkLen32 uniform-fill v=" << static_cast<int>(v) << " byte_i=" << i;
    }
  }

  for (size_t target_k = 0; target_k < sq2::kBlockGroupBlks; ++target_k) {
    for (uint8_t v = 1; v < 4; ++v) {
      std::array<std::array<uint8_t, sq2::kBlkLen32>, sq2::kBlockGroupBlks> weights{};
      weights[target_k].fill(v);

      std::array<std::array<std::byte, sq2::kBlkBytes32>, sq2::kBlockGroupBlks> src{};
      for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
        PackSourceBlock_BlkLen32(weights[k].data(), src[k].data());
      }

      std::array<std::byte, sq2::kBlockGroupBytes32> packed{};
      sq2::PackBlockGroup_BlkLen32(src[0].data(), src[1].data(), src[2].data(), src[3].data(),
                                   packed.data());

      const uint8_t expected_byte = static_cast<uint8_t>(v << (2 * target_k));
      for (size_t i = 0; i < sq2::kBlockGroupBytes32; ++i) {
        ASSERT_EQ(static_cast<uint8_t>(packed[i]), expected_byte)
            << "BlkLen32 isolated block target_k=" << target_k
            << " v=" << static_cast<int>(v) << " byte_i=" << i;
      }

      std::array<std::array<uint8_t, sq2::kBlkLen32>, sq2::kBlockGroupBlks> recovered{};
      sq2::UnPackBlockGroup_BlkLen32_Reference(
          packed.data(),
          recovered[0].data(), recovered[1].data(),
          recovered[2].data(), recovered[3].data());
      for (size_t k = 0; k < sq2::kBlockGroupBlks; ++k) {
        const uint8_t expect_val = (k == target_k) ? v : uint8_t{0};
        for (size_t i = 0; i < sq2::kBlkLen32; ++i) {
          ASSERT_EQ(recovered[k][i], expect_val)
              << "BlkLen32 isolated round-trip target_k=" << target_k
              << " k=" << k << " v=" << static_cast<int>(v) << " i=" << i;
        }
      }
    }
  }
}
