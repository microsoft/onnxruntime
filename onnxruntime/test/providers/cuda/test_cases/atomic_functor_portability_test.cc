// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

namespace onnxruntime::test {

cudaError_t LaunchAtomicFunctorBits(void* storage, int element_bytes, bool is_signed, int count, cudaStream_t stream);

namespace {

template <typename T>
void TestAtomicFunctorBits() {
  using U = std::make_unsigned_t<T>;
  constexpr int count = 10;
  constexpr uint64_t sign = uint64_t{1} << (8 * sizeof(T) - 1);
  constexpr uint64_t mask = std::numeric_limits<U>::max();
  const std::array<uint64_t, count> lhs{sign - 1, sign, mask, mask, sign, sign, mask, sign | 1, 0, sign - 1};
  const std::array<uint64_t, count> rhs{1, mask, 1, mask, sign, mask, mask, 3, mask, 2};
  std::array<U, 4 * count> storage{};
  for (int i = 0; i < count; ++i) {
    storage[i] = static_cast<U>(lhs[i]);
    storage[count + i] = static_cast<U>(rhs[i]);
  }
  void* raw = nullptr;
  ASSERT_EQ(cudaMalloc(&raw, sizeof(storage)), cudaSuccess);
  auto free_buffer = [](void* buffer) { EXPECT_EQ(cudaFree(buffer), cudaSuccess); };
  std::unique_ptr<void, decltype(free_buffer)> device(raw, free_buffer);
  ASSERT_EQ(cudaMemcpy(device.get(), storage.data(), sizeof(storage), cudaMemcpyHostToDevice), cudaSuccess);
  ASSERT_EQ(LaunchAtomicFunctorBits(device.get(), sizeof(T), std::is_signed_v<T>, count, nullptr), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(storage.data(), device.get(), sizeof(storage), cudaMemcpyDeviceToHost), cudaSuccess);
  for (int i = 0; i < count; ++i) {
    EXPECT_EQ(storage[2 * count + i], static_cast<U>(lhs[i] + rhs[i])) << "addition case " << i;
    EXPECT_EQ(storage[3 * count + i], static_cast<U>(lhs[i] * rhs[i])) << "multiplication case " << i;
  }
}

TEST(CUDA_EP_Unittest, AtomicFunctorBitsInt8) {
  TestAtomicFunctorBits<int8_t>();
}
TEST(CUDA_EP_Unittest, AtomicFunctorBitsUint8) {
  TestAtomicFunctorBits<uint8_t>();
}
TEST(CUDA_EP_Unittest, AtomicFunctorBitsInt16) {
  TestAtomicFunctorBits<int16_t>();
}
TEST(CUDA_EP_Unittest, AtomicFunctorBitsUint16) {
  TestAtomicFunctorBits<uint16_t>();
}
TEST(CUDA_EP_Unittest, AtomicFunctorBitsInt32) {
  TestAtomicFunctorBits<int32_t>();
}
TEST(CUDA_EP_Unittest, AtomicFunctorBitsUint32) {
  TestAtomicFunctorBits<uint32_t>();
}
TEST(CUDA_EP_Unittest, AtomicFunctorBitsInt64) {
  TestAtomicFunctorBits<int64_t>();
}
TEST(CUDA_EP_Unittest, AtomicFunctorBitsUint64) {
  TestAtomicFunctorBits<uint64_t>();
}

}  // namespace
}  // namespace onnxruntime::test
