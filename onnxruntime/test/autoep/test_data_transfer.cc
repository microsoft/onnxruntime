

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
  const OrtApi& c_api = Ort::GetApi();
  const OrtEpDevice* ep_device;
  Utils::RegisterAndGetExampleEp(*ort_env, ep_device);

  const OrtMemoryInfo* device_memory_info = c_api.EpDevice_MemoryInfo(ep_device, OrtDeviceMemoryType_DEFAULT);

  // create a tensor using the default CPU allocator
  Ort::AllocatorWithDefaultOptions cpu_allocator;
  std::vector<int64_t> shape{2, 3, 4};  // shape doesn't matter
  const size_t num_elements = 2 * 3 * 4;

  RandomValueGenerator random{};
  std::vector<float> input_data = random.Gaussian<float>(shape, 0.0f, 2.f);
  Ort::Value cpu_tensor = Ort::Value::CreateTensor<float>(cpu_allocator.GetInfo(),
                                                          input_data.data(), input_data.size(),
                                                          shape.data(), shape.size());

  // create an on-device Tensor using the example EPs alternative CPU allocator.
  // it has a different vendor to the default ORT CPU allocator so we can copy between them even though both are
  // really CPU based.
  OrtAllocator* allocator = nullptr;
  ASSERT_ORTSTATUS_OK(c_api.GetSharedAllocator(*ort_env, device_memory_info, &allocator));
  ASSERT_NE(allocator, nullptr);
  Ort::Value device_tensor = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());

  std::vector<const OrtValue*> src_tensor_ptrs{cpu_tensor};
  std::vector<OrtValue*> dst_tensor_ptrs{device_tensor};

  ASSERT_ORTSTATUS_OK(c_api.CopyTensors(*ort_env, src_tensor_ptrs.data(), dst_tensor_ptrs.data(), nullptr,
                                        src_tensor_ptrs.size()));

  const float* src_data = nullptr;
  const float* dst_data = nullptr;
  ASSERT_ORTSTATUS_OK(c_api.GetTensorData(cpu_tensor, reinterpret_cast<const void**>(&src_data)));
  ASSERT_ORTSTATUS_OK(c_api.GetTensorData(device_tensor, reinterpret_cast<const void**>(&dst_data)));

  size_t bytes;
  ASSERT_ORTSTATUS_OK(c_api.GetTensorSizeInBytes(cpu_tensor, &bytes));
  ASSERT_EQ(bytes, num_elements * sizeof(float));

  ASSERT_NE(src_data, dst_data) << "Should have copied between two different memory locations";

  auto src_span = gsl::make_span(src_data, num_elements);
  auto dst_span = gsl::make_span(dst_data, num_elements);

  EXPECT_THAT(src_span, ::testing::ContainerEq(dst_span));

  // must release this before we unload the EP and the allocator is deleted
  device_tensor = Ort::Value();

  ort_env->UnregisterExecutionProviderLibrary(Utils::example_ep_info.registration_name.c_str());
}

}  // namespace test
}  // namespace onnxruntime

#endif  // _WIN32
