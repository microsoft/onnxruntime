// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <memory>
#include <vector>

#include "cuda_runtime.h"
#include "gtest/gtest.h"

#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {
namespace {

struct CudaFreeDeleter {
  void operator()(void* p) const {
    if (p != nullptr) {
      cudaFree(p);
    }
  }
};

bool IsCudaDeviceUnavailable() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    cudaGetLastError();
    return true;
  }

  return false;
}

}  // namespace

TEST(CUDAExternalMemoryTest, SessionAllocatesBoundOutputFromExternalMemory) {
  if (IsCudaDeviceUnavailable()) {
    GTEST_SKIP() << "No CUDA device available.";
  }

  constexpr OrtDevice::DeviceId cuda_device_id = 0;
  ASSERT_EQ(cudaSetDevice(cuda_device_id), cudaSuccess);

  constexpr size_t external_mem_size = 64 * 1024 * 1024;
  void* external_mem_ptr = nullptr;
  ASSERT_EQ(cudaMalloc(&external_mem_ptr, external_mem_size), cudaSuccess);
  std::unique_ptr<void, CudaFreeDeleter> external_memory{external_mem_ptr};

  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.do_copy_in_default_stream = true;
  provider_options.use_tf32 = false;
  provider_options.gpu_external_mem_ptr = external_mem_ptr;
  provider_options.gpu_external_mem_size = external_mem_size;

  SessionOptions so;
  so.session_logid = "CUDAExternalMemorySessionAllocatorTest";
  InferenceSession session_object{so, GetEnvironment()};
  auto cuda_provider = CudaExecutionProviderWithOptions(&provider_options);
  ASSERT_NE(cuda_provider, nullptr);
  ASSERT_STATUS_OK(session_object.RegisterExecutionProvider(std::move(cuda_provider)));
  ASSERT_STATUS_OK(session_object.Load(ORT_TSTR("testdata/mul_1.onnx")));
  ASSERT_STATUS_OK(session_object.Initialize());

  const std::vector<int64_t> dims = {3, 2};
  const std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  OrtValue ml_value_x;
  CreateMLValue<float>(std::make_shared<CPUAllocator>(), dims, values, &ml_value_x);

  std::unique_ptr<IOBinding> io_binding;
  ASSERT_STATUS_OK(session_object.NewIOBinding(&io_binding));
  ASSERT_STATUS_OK(io_binding->BindInput("X", ml_value_x));

  OrtDevice output_device(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NVIDIA, cuda_device_id);
  ASSERT_STATUS_OK(io_binding->BindOutput("Y", output_device));
  ASSERT_STATUS_OK(io_binding->SynchronizeInputs());

  RunOptions run_options;
  run_options.run_tag = so.session_logid;
  ASSERT_STATUS_OK(session_object.Run(run_options, *io_binding));
  ASSERT_STATUS_OK(io_binding->SynchronizeOutputs());

  std::vector<OrtValue>& outputs = io_binding->GetOutputs();
  ASSERT_EQ(1u, outputs.size());
  const auto& output_tensor = outputs.front().Get<Tensor>();
  EXPECT_EQ(output_tensor.Location().device.Type(), OrtDevice::GPU);
  EXPECT_EQ(output_tensor.Location().device.Id(), cuda_device_id);

  const auto base_address = reinterpret_cast<uintptr_t>(external_mem_ptr);
  const auto output_address = reinterpret_cast<uintptr_t>(output_tensor.DataRaw());
  ASSERT_GE(output_address, base_address);
  ASSERT_LT(output_address, base_address + external_mem_size);
  ASSERT_LE(output_tensor.SizeInBytes(), base_address + external_mem_size - output_address);

  std::vector<float> output_values(values.size());
  ASSERT_EQ(cudaMemcpy(output_values.data(), output_tensor.Data<float>(), output_tensor.SizeInBytes(),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  const std::vector<float> expected_values = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
  EXPECT_EQ(output_values, expected_values);
}

}  // namespace test
}  // namespace onnxruntime
