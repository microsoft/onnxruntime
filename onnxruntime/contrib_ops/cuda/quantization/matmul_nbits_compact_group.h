#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if defined(USE_COMPACT_FPA_INTB_GEMM) && USE_COMPACT_FPA_INTB_GEMM
inline constexpr bool kMatMulNBitsCompactFpAIntB = true;
#else
inline constexpr bool kMatMulNBitsCompactFpAIntB = false;
#endif

inline int64_t MatMulNBitsFpAIntBBlockSize(int64_t bits, int64_t block_size, bool compact) {
  return compact && bits == 4 && (block_size == 64 || block_size == 128) ? 32 : block_size;
}

template <typename Scale>
bool TransposeAndRepeatMatMulScales(const Scale* source, size_t source_count,
                                     size_t columns, size_t groups, size_t repeats,
                                     Scale* destination, size_t destination_count) {
  const size_t limit = std::numeric_limits<size_t>::max();
  if (source == nullptr || destination == nullptr || columns == 0 || groups == 0 || repeats == 0 ||
      columns > limit / groups || columns * groups != source_count ||
      source_count > limit / repeats || source_count * repeats != destination_count) {
    return false;
  }
  for (size_t group = 0; group < groups * repeats; ++group) {
    for (size_t column = 0; column < columns; ++column) {
      destination[group * columns + column] = source[column * groups + group / repeats];
    }
  }
  return true;
}

}
}
}