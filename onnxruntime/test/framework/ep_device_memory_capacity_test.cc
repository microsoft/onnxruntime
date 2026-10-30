#if defined(USE_CUDA) && !defined(ORT_MINIMAL_BUILD)

#include <cuda.h>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <string_view>

#include <gsl/gsl>
#include <gtest/gtest.h>

#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/api_asserts.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

TEST(EpDeviceMemoryCapacityTest, CudaCapacityPreservesContext) {
  const auto init_result = cuInit(0);
  if (init_result == CUDA_ERROR_NO_DEVICE) {
    GTEST_SKIP() << "No CUDA device available";
  }
  ASSERT_EQ(init_result, CUDA_SUCCESS);
  int device_count = 0;
  ASSERT_EQ(cuDeviceGetCount(&device_count), CUDA_SUCCESS);
  if (device_count == 0) {
    GTEST_SKIP() << "No CUDA device available";
  }

  constexpr const char* registration_name = "cuda_memory_capacity_test";
  bool registered = false;
  const auto unregister = gsl::finally([&]() {
    if (registered) {
      Ort::Status status{Ort::GetApi().UnregisterExecutionProviderLibrary(*ort_env, registration_name)};
      EXPECT_TRUE(status.IsOK()) << status.GetErrorMessage();
    }
  });
  auto devices = ort_env->GetEpDevices();
  const auto is_cuda = [](const Ort::ConstEpDevice& device) {
    return std::string_view(device.EpName()) == "CUDAExecutionProvider";
  };
  if (std::none_of(devices.begin(), devices.end(), is_cuda)) {
#ifdef _WIN32
    constexpr const ORTCHAR_T* library_path = ORT_TSTR("onnxruntime_providers_cuda.dll");
#else
    constexpr const ORTCHAR_T* library_path = ORT_TSTR("./libonnxruntime_providers_cuda.so");
#endif
    ort_env->RegisterExecutionProviderLibrary(registration_name, library_path);
    registered = true;
    devices = ort_env->GetEpDevices();
  }

  CUcontext original_context = nullptr;
  ASSERT_EQ(cuCtxGetCurrent(&original_context), CUDA_SUCCESS);
  const auto restore_context = gsl::finally([&]() {
    EXPECT_EQ(cuCtxSetCurrent(original_context), CUDA_SUCCESS);
  });
  CUdevice caller_device = 0;
  ASSERT_EQ(cuDeviceGet(&caller_device, device_count - 1), CUDA_SUCCESS);
  CUcontext caller_context = nullptr;
  ASSERT_EQ(cuCtxCreate_v2(&caller_context, 0, caller_device), CUDA_SUCCESS);
  const auto destroy_context = gsl::finally([&]() {
    EXPECT_EQ(cuCtxDestroy(caller_context), CUDA_SUCCESS);
  });

  size_t tested = 0;
  for (const auto& device : devices) {
    if (!is_cuda(device)) {
      continue;
    }
    const int ordinal = device.GetMemoryInfo(OrtDeviceMemoryType_DEFAULT).GetDeviceId();
    SCOPED_TRACE(ordinal);
    CUdevice cuda_device = 0;
    ASSERT_EQ(cuDeviceGet(&cuda_device, ordinal), CUDA_SUCCESS);
    size_t expected_total = 0;
    ASSERT_EQ(cuDeviceTotalMem(&expected_total, cuda_device), CUDA_SUCCESS);
    ASSERT_GT(expected_total, 0u);

    for (CUcontext current_context : {original_context, caller_context}) {
      ASSERT_EQ(cuCtxSetCurrent(current_context), CUDA_SUCCESS);
      uint64_t total = 0;
      uint64_t available = 0;
      ASSERT_ORTSTATUS_OK(Ort::GetApi().EpDevice_GetMemoryCapacity(device, &total, &available));
      CUcontext after = nullptr;
      ASSERT_EQ(cuCtxGetCurrent(&after), CUDA_SUCCESS);
      EXPECT_EQ(after, current_context);
      EXPECT_EQ(total, static_cast<uint64_t>(expected_total));
      EXPECT_LE(available, total);

      const auto capacity = device.GetMemoryCapacity();
      ASSERT_EQ(cuCtxGetCurrent(&after), CUDA_SUCCESS);
      EXPECT_EQ(after, current_context);
      EXPECT_EQ(capacity.first, static_cast<uint64_t>(expected_total));
      EXPECT_LE(capacity.second, capacity.first);
    }
    ++tested;
  }
  EXPECT_GT(tested, 0u) << "CUDA is available but no CUDA EP devices were discovered";
}

}  // namespace test
}  // namespace onnxruntime

#endif