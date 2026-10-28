// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <type_traits>

#include "core/providers/cuda/atomic/common.cuh"

namespace onnxruntime::test {
namespace {

template <typename T>
__global__ void AtomicFunctorBitsKernel(std::make_unsigned_t<T>* storage, int count) {
  const int index = threadIdx.x;
  if (index >= count) return;
  T lhs;
  T rhs;
  memcpy(&lhs, storage + index, sizeof(T));
  memcpy(&rhs, storage + count + index, sizeof(T));
  const T added = onnxruntime::cuda::AddFunc{}(lhs, rhs);
  const T multiplied = onnxruntime::cuda::MulFunc{}(lhs, rhs);
  memcpy(storage + 2 * count + index, &added, sizeof(T));
  memcpy(storage + 3 * count + index, &multiplied, sizeof(T));
}

template <typename T>
void Launch(void* storage, int count, cudaStream_t stream) {
  AtomicFunctorBitsKernel<T><<<1, 32, 0, stream>>>(static_cast<std::make_unsigned_t<T>*>(storage), count);
}

}  // namespace

cudaError_t LaunchAtomicFunctorBits(void* storage, int element_bytes, bool is_signed, int count, cudaStream_t stream) {
  switch (element_bytes) {
    case 1:
      is_signed ? Launch<int8_t>(storage, count, stream) : Launch<uint8_t>(storage, count, stream);
      break;
    case 2:
      is_signed ? Launch<int16_t>(storage, count, stream) : Launch<uint16_t>(storage, count, stream);
      break;
    case 4:
      is_signed ? Launch<int32_t>(storage, count, stream) : Launch<uint32_t>(storage, count, stream);
      break;
    case 8:
      is_signed ? Launch<int64_t>(storage, count, stream) : Launch<uint64_t>(storage, count, stream);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace onnxruntime::test
