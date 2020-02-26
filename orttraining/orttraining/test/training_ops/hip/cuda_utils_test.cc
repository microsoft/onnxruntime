// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_HIP

#include "core/providers/hip/hip_utils.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/providers/hip/hip_call.h"

namespace onnxruntime {
namespace hip {
namespace test {

namespace {
struct HipDeviceMemoryDeleter {
  template <typename T>
  void operator()(T* p) {
    hipFree(p);
  }
};
/*
template <typename TElement>
void TestFillCorrectness(size_t num_elements, TElement value) {
  void* raw_buffer;
  HIP_CALL_THROW(hipMalloc(&raw_buffer, num_elements * sizeof(TElement)));
  std::unique_ptr<TElement, HipDeviceMemoryDeleter> buffer{
      reinterpret_cast<TElement*>(raw_buffer)};

  Fill<TElement>(buffer.get(), value, num_elements);

  auto cpu_buffer = onnxruntime::make_unique<TElement[]>(num_elements);
  HIP_CALL_THROW(hipMemcpy(cpu_buffer.get(), buffer.get(), num_elements * sizeof(TElement), hipMemcpyKind::hipMemcpyDeviceToHost));

  std::vector<TElement> expected_data(num_elements, value);
  EXPECT_EQ(std::memcmp(cpu_buffer.get(), expected_data.data(), num_elements * sizeof(TElement)), 0);
}*/
}  // namespace
/*
TEST(HipUtilsTest, FillCorrectness) {
  TestFillCorrectness<int8_t>(1 << 20, 1);
  TestFillCorrectness<int16_t>(1 << 20, 2);
  TestFillCorrectness<int32_t>(1 << 20, 3);
  TestFillCorrectness<float>(1 << 20, 4.0f);
  TestFillCorrectness<double>(1 << 20, 5.0);
}
*/
}  // namespace test
}  // namespace hip
}  // namespace onnxruntime

#endif  // USE_HIP
