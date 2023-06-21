// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {
namespace test {

namespace {
struct CudaDeviceMemoryDeleter {
  template <typename T>
  void operator()(T* p) {
    cudaFree(p);
  }
};

template <typename TElement>
void TestFillCorrectness(size_t num_elements, TElement value) {
  void* raw_buffer;
  CUDA_CALL_THROW(cudaMalloc(&raw_buffer, num_elements * sizeof(TElement)));
  std::unique_ptr<TElement, CudaDeviceMemoryDeleter> buffer{
      reinterpret_cast<TElement*>(raw_buffer)};

  Fill<TElement>(nullptr, buffer.get(), value, num_elements);

  auto cpu_buffer = std::make_unique<TElement[]>(num_elements);
  CUDA_CALL_THROW(cudaMemcpy(cpu_buffer.get(), buffer.get(), num_elements * sizeof(TElement), cudaMemcpyKind::cudaMemcpyDeviceToHost));

  std::vector<TElement> expected_data(num_elements, value);
  EXPECT_EQ(std::memcmp(cpu_buffer.get(), expected_data.data(), num_elements * sizeof(TElement)), 0);
}
}  // namespace

TEST(CudaUtilsTest, FillCorrectness) {
  TestFillCorrectness<int8_t>(1 << 20, 1);
  TestFillCorrectness<int16_t>(1 << 20, 2);
  TestFillCorrectness<int32_t>(1 << 20, 3);
  TestFillCorrectness<float>(1 << 20, 4.0f);
  TestFillCorrectness<double>(1 << 20, 5.0);
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
