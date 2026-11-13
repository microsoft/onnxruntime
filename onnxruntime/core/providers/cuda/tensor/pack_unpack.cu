// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <limits>

#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "core/providers/cuda/tensor/pack_unpack_impl.h"

namespace onnxruntime {
namespace cuda {
namespace {

constexpr uint64_t kMinVectorWork = 131072;
constexpr uint64_t kMinVectorRow = 4096;

template <typename Word, int Width>
__global__ void UnpackContiguousKernel(const Word* input, uint8_t* output, uint64_t work,
                                       packing::Layout l, fast_divmod count_div) {
  const uint64_t vectors = work / 4 + (work % 4 != 0);
  const packing::Parameters p{uint64_t{1} << Width, 1, Width};
  l.inner = 1;
  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < vectors; i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    const uint64_t first = i * 4;
    int outer, unit;
    count_div.divmod(static_cast<int>(first), outer, unit);
    if (l.count - static_cast<uint64_t>(unit) >= 4 && work - first >= 4) {
      const Word* row = input + static_cast<uint64_t>(outer) * l.words;
      uint8_t values[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        values[j] = static_cast<uint8_t>(packing::DecodeGroup<Word, Width>(row, unit + j, p, l));
      }
      reinterpret_cast<uchar4*>(output)[i] = make_uchar4(values[0], values[1], values[2], values[3]);
    } else {
      // A vector can cross a padded row boundary or the end of the output.
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (first + j < work) {
          int tail_outer, tail_unit;
          count_div.divmod(static_cast<int>(first + j), tail_outer, tail_unit);
          output[first + j] = static_cast<uint8_t>(packing::DecodeGroup<Word, Width>(
              input + static_cast<uint64_t>(tail_outer) * l.words, tail_unit, p, l));
        }
      }
    }
  }
}

template <typename Word, bool Unpack, typename Index, int Width = 0, uint64_t Radix = 0>
__global__ void PackingKernel(const void* input, void* output, uint64_t work,
                              packing::Parameters p, packing::Layout l,
                              DivMod<Index> inner_div, DivMod<Index> units_div) {
  if constexpr (Radix != 0) p.radix = Radix;
  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < work; i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    Index row, inner_index, outer_index, unit;
    inner_div.divmod(static_cast<Index>(i), row, inner_index);
    units_div.divmod(row, outer_index, unit);
    const uint64_t inner = static_cast<uint64_t>(inner_index);
    const uint64_t outer = static_cast<uint64_t>(outer_index);
    if constexpr (Unpack) {
      packing::DecodeAndStoreGroup<Word, Width>(
          static_cast<const Word*>(input) + outer * l.words * l.inner + inner,
          static_cast<uint8_t*>(output) + outer * l.count * l.inner + inner, unit, p, l);
    } else {
      static_cast<Word*>(output)[i] = packing::EncodeWord<Word, Width>(
          static_cast<const uint8_t*>(input) + outer * l.count * l.inner + inner, unit, p, l);
    }
  }
}

template <typename Word, bool Unpack, typename Index>
void DispatchPackingWithIndex(cudaStream_t stream, int blocks, const void* input, void* output, uint64_t work,
                              packing::Parameters p, packing::Layout l) {
  const DivMod<Index> inner_div(static_cast<Index>(l.inner));
  const DivMod<Index> units_div(static_cast<Index>(Unpack ? l.groups : l.words));
  if constexpr (Unpack) {
    if (p.radix == 3 && p.group_size > 1) {
      PackingKernel<Word, true, Index, 0, 3><<<blocks, 256, 0, stream>>>(input, output, work, p, l, inner_div, units_div);
      return;
    }
  }
  if (p.group_size == 1) {
    switch (p.width) {
      case 1:
        PackingKernel<Word, Unpack, Index, 1><<<blocks, 256, 0, stream>>>(input, output, work, p, l, inner_div, units_div);
        return;
      case 2:
        PackingKernel<Word, Unpack, Index, 2><<<blocks, 256, 0, stream>>>(input, output, work, p, l, inner_div, units_div);
        return;
      case 3:
        PackingKernel<Word, Unpack, Index, 3><<<blocks, 256, 0, stream>>>(input, output, work, p, l, inner_div, units_div);
        return;
      case 4:
        PackingKernel<Word, Unpack, Index, 4><<<blocks, 256, 0, stream>>>(input, output, work, p, l, inner_div, units_div);
        return;
      case 5:
        PackingKernel<Word, Unpack, Index, 5><<<blocks, 256, 0, stream>>>(input, output, work, p, l, inner_div, units_div);
        return;
      case 6:
        PackingKernel<Word, Unpack, Index, 6><<<blocks, 256, 0, stream>>>(input, output, work, p, l, inner_div, units_div);
        return;
      case 7:
        PackingKernel<Word, Unpack, Index, 7><<<blocks, 256, 0, stream>>>(input, output, work, p, l, inner_div, units_div);
        return;
      case 8:
        PackingKernel<Word, Unpack, Index, 8><<<blocks, 256, 0, stream>>>(input, output, work, p, l, inner_div, units_div);
        return;
    }
  }
  PackingKernel<Word, Unpack, Index><<<blocks, 256, 0, stream>>>(input, output, work, p, l, inner_div, units_div);
}

template <typename Word, bool Unpack>
void DispatchPacking(cudaStream_t stream, int blocks, const void* input, void* output, uint64_t work,
                     packing::Parameters p, packing::Layout l) {
  if constexpr (Unpack) {
    // Small tensors and short padded rows are faster with scalar decoding.
    if (work >= kMinVectorWork && work <= static_cast<uint64_t>(std::numeric_limits<int>::max()) &&
        l.count >= kMinVectorRow && l.inner == 1 && p.group_size == 1 &&
        (p.width == 1 || p.width == 2 || p.width == 4 || p.width == 8) &&
        p.radix == (uint64_t{1} << p.width) &&
        reinterpret_cast<uintptr_t>(output) % alignof(uchar4) == 0) {
      const uint64_t vectors = work / 4 + (work % 4 != 0);
      const int vector_blocks = static_cast<int>(std::min<uint64_t>(vectors / 256 + (vectors % 256 != 0), 65535));
      const fast_divmod count_div(static_cast<int>(l.count));
      const auto* words = static_cast<const Word*>(input);
      auto* codes = static_cast<uint8_t*>(output);
      switch (p.width) {
        case 1:
          UnpackContiguousKernel<Word, 1><<<vector_blocks, 256, 0, stream>>>(words, codes, work, l, count_div);
          return;
        case 2:
          UnpackContiguousKernel<Word, 2><<<vector_blocks, 256, 0, stream>>>(words, codes, work, l, count_div);
          return;
        case 4:
          UnpackContiguousKernel<Word, 4><<<vector_blocks, 256, 0, stream>>>(words, codes, work, l, count_div);
          return;
        case 8:
          UnpackContiguousKernel<Word, 8><<<vector_blocks, 256, 0, stream>>>(words, codes, work, l, count_div);
          return;
      }
    }
  }
  // Work bounds both divisors, but byte offsets can still exceed INT_MAX.
  if (work <= static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    DispatchPackingWithIndex<Word, Unpack, int>(stream, blocks, input, output, work, p, l);
  } else {
    DispatchPackingWithIndex<Word, Unpack, uint64_t>(stream, blocks, input, output, work, p, l);
  }
}
}  // namespace

Status LaunchPacking(cudaStream_t stream, bool unpack, bool word32, const void* input, void* output,
                     uint64_t work, packing::Parameters parameters, packing::Layout layout) {
  const int blocks = static_cast<int>(std::min<uint64_t>(work / 256 + (work % 256 != 0), 65535));
  if (unpack) {
    if (word32) {
      DispatchPacking<uint32_t, true>(stream, blocks, input, output, work, parameters, layout);
    } else {
      DispatchPacking<uint8_t, true>(stream, blocks, input, output, work, parameters, layout);
    }
  } else if (word32) {
    DispatchPacking<uint32_t, false>(stream, blocks, input, output, work, parameters, layout);
  } else {
    DispatchPacking<uint8_t, false>(stream, blocks, input, output, work, parameters, layout);
  }
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
