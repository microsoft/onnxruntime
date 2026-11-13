// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#ifdef __CUDACC__
#define ORT_PACK_HD __host__ __device__
#else
#define ORT_PACK_HD
#endif

namespace onnxruntime {
namespace packing {

struct Parameters {
  uint64_t radix;
  uint64_t group_size;
  int width;
};

struct Layout {
  uint64_t count;
  uint64_t words;
  uint64_t groups;
  uint64_t inner;
};

ORT_PACK_HD inline uint64_t Mask(int bits) {
  return (uint64_t{1} << bits) - 1;
}

ORT_PACK_HD inline uint64_t EncodeGroup(const uint8_t* row, uint64_t group,
                                        const Parameters& p, const Layout& l) {
  const uint64_t start = group * p.group_size;
  const uint64_t digits = l.count - start < p.group_size ? l.count - start : p.group_size;
  uint64_t value = 0;
  for (uint64_t j = digits; j != 0; --j) {
    value = value * p.radix + row[(start + j - 1) * l.inner];
  }
  return value;
}

// One owner per storage word, even when group indices cross word boundaries.
template <typename Word, int Width = 0>
ORT_PACK_HD inline Word EncodeWord(const uint8_t* row, uint64_t word,
                                   const Parameters& p, const Layout& l) {
  constexpr int word_bits = sizeof(Word) * 8;
  const int width = Width == 0 ? p.width : Width;
  // Quotient/remainder addressing avoids overflowing word * word_bits.
  uint64_t group = (word / width) * word_bits + (word % width) * word_bits / width;
  int skip = static_cast<int>((word % width) * word_bits % width);
  int placed = 0;
  uint64_t result = 0;
  while (placed < word_bits && group < l.groups) {
    const uint64_t index = Width == 0 ? EncodeGroup(row, group, p, l) : row[group * l.inner];
    const int take = width - skip < word_bits - placed ? width - skip : word_bits - placed;
    result |= ((index >> skip) & Mask(take)) << placed;
    placed += take;
    skip = 0;
    ++group;
  }
  return static_cast<Word>(result);
}

template <typename Word, int Width = 0>
ORT_PACK_HD inline uint64_t DecodeGroup(const Word* row, uint64_t group,
                                        const Parameters& p, const Layout& l) {
  constexpr int word_bits = sizeof(Word) * 8;
  const int width = Width == 0 ? p.width : Width;
  uint64_t word = (group / word_bits) * width + (group % word_bits) * width / word_bits;
  int skip = static_cast<int>((group % word_bits) * width % word_bits);
  int placed = 0;
  uint64_t result = 0;
  while (placed < width) {
    const int take = word_bits - skip < width - placed ? word_bits - skip : width - placed;
    result |= ((static_cast<uint64_t>(row[word * l.inner]) >> skip) & Mask(take)) << placed;
    placed += take;
    skip = 0;
    ++word;
  }
  return result;
}

template <typename Word, int Width = 0>
ORT_PACK_HD inline void DecodeAndStoreGroup(const Word* row, uint8_t* output, uint64_t group,
                                            const Parameters& p, const Layout& l) {
  uint64_t value = DecodeGroup<Word, Width>(row, group, p, l);
  if constexpr (Width != 0) {
    output[group * l.inner] = static_cast<uint8_t>(value % p.radix);
  } else {
    const uint64_t start = group * p.group_size;
    const uint64_t digits = l.count - start < p.group_size ? l.count - start : p.group_size;
    for (uint64_t j = 0; j < digits; ++j) {
      output[(start + j) * l.inner] = static_cast<uint8_t>(value % p.radix);
      value /= p.radix;
    }
  }
}

}  // namespace packing
}  // namespace onnxruntime

#undef ORT_PACK_HD
