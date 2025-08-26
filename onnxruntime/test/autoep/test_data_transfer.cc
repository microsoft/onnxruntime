

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// registration/selection is only supported on windows as there's no device discovery on other platforms
#ifdef _WIN32

#include <algorithm>
#include <gsl/gsl>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"

#include "test/autoep/test_autoep_utils.h"
#include "test/common/random_generator.h"
#include "test/util/include/api_asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

TEST(OrtEpLibrary, DataTransfer) {
  RegisteredEpDeviceUniquePtr example_ep;
  Utils::RegisterAndGetExampleEp(*ort_env, example_ep);
  Ort::ConstEpDevice ep_device(example_ep.get());

  auto device_memory_info = ep_device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT);

  // create a tensor using the default CPU allocator
  Ort::AllocatorWithDefaultOptions cpu_allocator;
  constexpr const std::array<int64_t, 3U> shape{2, 3, 4};  // shape doesn't matter
  const size_t num_elements = 2 * 3 * 4;

  RandomValueGenerator random{};
  std::vector<float> input_data = random.Gaussian<float>(shape, 0.0f, 2.f);
  Ort::Value cpu_tensor = Ort::Value::CreateTensor<float>(cpu_allocator.GetInfo(),
                                                          input_data.data(), input_data.size(),
                                                          shape.data(), shape.size());

  // create an on-device Tensor using the example EPs alternative CPU allocator.
  // it has a different vendor to the default ORT CPU allocator so we can copy between them even though both are
  // really CPU based.
  auto allocator = ort_env->GetSharedAllocator(device_memory_info);
  ASSERT_NE(allocator, nullptr);
  Ort::Value device_tensor = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());

  std::vector<Ort::Value> src_tensor;
  src_tensor.push_back(std::move(cpu_tensor));
  std::vector<Ort::Value> dst_tensor;
  dst_tensor.push_back(std::move(device_tensor));

  ASSERT_CXX_ORTSTATUS_OK(ort_env->CopyTensors(src_tensor, dst_tensor, nullptr));

  const float* src_data = src_tensor[0].GetTensorData<float>();
  const float* dst_data = dst_tensor[0].GetTensorData<float>();

  size_t bytes = src_tensor[0].GetTensorSizeInBytes();
  ASSERT_EQ(bytes, num_elements * sizeof(float));

  ASSERT_NE(src_data, dst_data) << "Should have copied between two different memory locations";

  auto src_span = gsl::make_span(src_data, num_elements);
  auto dst_span = gsl::make_span(dst_data, num_elements);

  EXPECT_THAT(src_span, ::testing::ContainerEq(dst_span));

  // must release this before we unload the EP and the allocator is deleted
  device_tensor = Ort::Value();
}

}  // namespace test
}  // namespace onnxruntime

#endif  // _WIN32
