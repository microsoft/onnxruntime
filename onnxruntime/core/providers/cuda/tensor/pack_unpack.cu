// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>

#include "core/providers/cuda/tensor/pack_unpack_impl.h"

namespace onnxruntime {
namespace cuda {
namespace {

template <typename Word, bool Unpack, int Width = 0>
__global__ void PackingKernel(const void* input, void* output, uint64_t work,
                              packing::Parameters p, packing::Layout l) {
  const uint64_t units = Unpack ? l.groups : l.words;
  for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < work; i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    const uint64_t inner = i % l.inner;
    const uint64_t unit = (i / l.inner) % units;
    const uint64_t outer = i / l.inner / units;
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

template <typename Word, bool Unpack>
void DispatchPacking(cudaStream_t stream, int blocks, const void* input, void* output, uint64_t work,
                     packing::Parameters p, packing::Layout l) {
  if (p.group_size == 1) {
    switch (p.width) {
      case 1:
        PackingKernel<Word, Unpack, 1><<<blocks, 256, 0, stream>>>(input, output, work, p, l);
        return;
      case 2:
        PackingKernel<Word, Unpack, 2><<<blocks, 256, 0, stream>>>(input, output, work, p, l);
        return;
      case 4:
        PackingKernel<Word, Unpack, 4><<<blocks, 256, 0, stream>>>(input, output, work, p, l);
        return;
      case 8:
        PackingKernel<Word, Unpack, 8><<<blocks, 256, 0, stream>>>(input, output, work, p, l);
        return;
    }
  }
  PackingKernel<Word, Unpack><<<blocks, 256, 0, stream>>>(input, output, work, p, l);
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
