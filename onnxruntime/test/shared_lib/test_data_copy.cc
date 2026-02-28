// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <filesystem>
#include <vector>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/graph/constants.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/api_asserts.h"
#include "test/shared_lib/utils.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_options.h"
#include <cuda_runtime.h>
#endif

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

using StreamUniquePtr = std::unique_ptr<OrtSyncStream, std::function<void(OrtSyncStream*)>>;

#ifdef USE_CUDA
// test copying input to CUDA using an OrtEpFactory based EP.
// tests GetSharedAllocator, CreateSyncStreamForEpDevice and CopyTensors APIs
TEST(PluginEpDataCopyTest, CopyInputsToCudaDevice) {
#ifdef _WIN32
  std::string cuda_lib = "onnxruntime_providers_cuda.dll";
#else
  std::string cuda_lib = "onnxruntime_providers_cuda.so";
#endif

  if (!std::filesystem::exists(cuda_lib)) {
    GTEST_SKIP() << "CUDA library was not found";
  }

  // register the provider bridge based CUDA EP so allocator and data transfer is available
  // not all the CIs have the provider library in the expected place so we allow for that
  const char* ep_registration_name = "ORT CUDA";
  ort_env->RegisterExecutionProviderLibrary(ep_registration_name,
                                            ORT_TSTR("onnxruntime_providers_cuda"));

  Ort::ConstEpDevice cuda_device{nullptr};
  for (const auto& ep_device : ort_env->GetEpDevices()) {
    std::string vendor{ep_device.EpVendor()};
    std::string name = {ep_device.EpName()};
    if (vendor == std::string("Microsoft") && name == kCudaExecutionProvider) {
      cuda_device = ep_device;
      break;
    }
  }

  if (!cuda_device) {  // device running tests may not have an nvidia card
    return;
  }

  const auto run_test = [&](bool use_streams) {
    Ort::SessionOptions options;
    options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);

    // we pass in the CUDA cudaStream_t from the OrtSyncStream via provider options so need to create it upfront.
    // in the future the stream should be an input to the Session Run.
    Ort::SyncStream stream{nullptr};
    if (use_streams) {
      stream = cuda_device.CreateSyncStream();

      size_t stream_addr = reinterpret_cast<size_t>(stream.GetHandle());
      options.AddConfigEntry("ep.cudaexecutionprovider.user_compute_stream", std::to_string(stream_addr).c_str());
      // we explicitly specify user_compute_stream, so why do we also need to set has_user_compute_stream?
      options.AddConfigEntry("ep.cudaexecutionprovider.has_user_compute_stream", "1");
    }

    Ort::Session session(*ort_env, ORT_TSTR("testdata/mnist.onnx"), options);

    size_t num_inputs = session.GetInputCount();

    // find the input location so we know which inputs can be provided on device.
    auto input_locations = session.GetMemoryInfoForInputs();
    ASSERT_EQ(session.GetInputCount(), input_locations.size());

    // Testing coverage
    auto input_ep_devices = session.GetEpDeviceForInputs();
    ASSERT_EQ(session.GetInputCount(), input_ep_devices.size());

    // This is for testing
    auto output_locations = session.GetMemoryInfoForOutputs();
    ASSERT_EQ(session.GetOutputCount(), output_locations.size());

    std::vector<Ort::Value> cpu_tensors;

    // info for device copy
    std::vector<Ort::Value> device_tensors;

    ASSERT_EQ(num_inputs, 1);

    // create cpu based input data.
    Ort::AllocatorWithDefaultOptions cpu_allocator;
    constexpr const std::array<int64_t, 4U> shape{1, 1, 28, 28};
    std::vector<float> input_data(28 * 28, 0.5f);
    Ort::Value input_value = Ort::Value::CreateTensor<float>(cpu_allocator.GetInfo(),
                                                             input_data.data(), input_data.size(),
                                                             shape.data(), shape.size());
    cpu_tensors.push_back(std::move(input_value));

    for (size_t idx = 0; idx < num_inputs; ++idx) {
      auto mem_info = input_locations[idx];
      OrtDeviceMemoryType mem_type = mem_info.GetDeviceMemoryType();
      OrtMemoryInfoDeviceType device_type = mem_info.GetDeviceType();

      if (device_type == OrtMemoryInfoDeviceType_GPU && mem_type == OrtDeviceMemoryType_DEFAULT) {
        // copy to device
        auto allocator = ort_env->GetSharedAllocator(mem_info);

        // allocate new on-device memory
        auto src_shape = cpu_tensors[idx].GetTensorTypeAndShapeInfo().GetShape();
        Ort::Value device_value = Ort::Value::CreateTensor<float>(allocator, src_shape.data(), src_shape.size());

        /* if you have existing memory on device use one of these instead of CreateTensorAsOrtValue + CopyTensors
        void* existing_data;
        size_t data_length = 128 * sizeof(float);
        api->CreateTensorWithDataAsOrtValue(input_locations[0], existing_data, data_length, shape, 2,
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value);

        // existing with ownership transfer. ORT will use the allocator to free the memory once it is no longer required
        api->CreateTensorWithDataAndDeleterAsOrtValue(allocator, existing_data, data_length, shape, 2,
                                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value);
        */

        device_tensors.push_back(std::move(device_value));
      }
    }

    if (!device_tensors.empty()) {
      // Test original CopyTensors (backward compatible)
      ASSERT_CXX_ORTSTATUS_OK(ort_env->CopyTensors(cpu_tensors, device_tensors, stream));

      // Stream support is still a work in progress.
      //
      // CUDA EP can use a user provided stream via provider options, so we can pass in the cudaStream_t from the
      // OrtSyncStream used in CopyTensors call that way.
      //
      // Alternatively you can manually sync the device via IoBinding.
      // Ort::IoBinding iobinding(session);
      // iobinding.SynchronizeInputs();  // this doesn't actually require any bound inputs
    }

    const auto& input_tensors = (!device_tensors.empty()) ? device_tensors : cpu_tensors;

    constexpr const std::array<const char*, 1U> input_names = {"Input3"};
    constexpr const std::array<const char*, 1U> output_names = {"Plus214_Output_0"};
    Ort::Value output;

    session.Run(Ort::RunOptions{}, input_names.data(), input_tensors.data(), input_tensors.size(),
                output_names.data(), &output, 1);

    const float* results = output.GetTensorData<float>();

    // expected results from the CPU EP. can check/re-create by running with PREFER_CPU.
    constexpr const std::array<float, 10U> expected = {
        -0.701670527f,
        -0.583666623f,
        0.0480501056f,
        0.550699294f,
        -1.25372827f,
        1.17879760f,
        0.838122189f,
        -1.51267099f,
        0.902430952f,
        0.243748352f,
    };

    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_NEAR(expected[i], results[i], 1e-3) << "i=" << i;
    }
  };

  run_test(/*use_streams*/ true);
  run_test(/*use_streams*/ false);

  ort_env->UnregisterExecutionProviderLibrary(ep_registration_name);
}
#endif  // USE_CUDA

// -----------------------------------------------------------------------------
// Tests for Value::CreateTensor overloads with a byte/element offset.
//
// The new overloads allow creating a tensor that is a zero-copy view into a
// sub-region of an existing buffer. This is useful for, e.g., preparing a
// partial copy via CopyTensors:
//
//   auto src_view = Ort::Value::CreateTensor<float>(info, buffer.data(),
//                       /*element_count=*/N, /*element_offset=*/start,
//                       shape.data(), shape.size());
//   ort_env->CopyTensors({src_view}, {dst_tensor}, stream);
// -----------------------------------------------------------------------------

// Test the typed template overload (element-based offset).
TEST(CreateTensorWithOffsetTest, TypedTemplateElementOffset) {
  // Buffer: [0, 1, 2, 3, 4, 5, 6, 7]
  std::vector<float> buffer = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};

  Ort::AllocatorWithDefaultOptions cpu_alloc;
  const OrtMemoryInfo* cpu_info = cpu_alloc.GetInfo();

  // Create a tensor view over elements [4..7] of the buffer.
  constexpr std::array<int64_t, 1> shape = {4};
  auto tensor = Ort::Value::CreateTensor<float>(
      cpu_info, buffer.data(),
      /*p_data_element_count=*/4,
      /*p_data_element_offset=*/4,
      shape.data(), shape.size());

  ASSERT_TRUE(tensor.IsTensor());
  const float* data = tensor.GetTensorData<float>();

  // The tensor data should point to buffer[4].
  EXPECT_EQ(data, buffer.data() + 4) << "Tensor data pointer should point to buffer[4]";

  EXPECT_EQ(data[0], 4.f);
  EXPECT_EQ(data[1], 5.f);
  EXPECT_EQ(data[2], 6.f);
  EXPECT_EQ(data[3], 7.f);
}

// Test the void* overload (byte-based offset).
TEST(CreateTensorWithOffsetTest, VoidStarByteOffset) {
  std::vector<int32_t> buffer = {10, 20, 30, 40, 50, 60};

  Ort::AllocatorWithDefaultOptions cpu_alloc;
  const OrtMemoryInfo* cpu_info = cpu_alloc.GetInfo();

  // View elements [2..4] (byte offset = 2 * sizeof(int32_t) = 8).
  constexpr std::array<int64_t, 1> shape = {3};
  const size_t byte_offset = 2 * sizeof(int32_t);
  const size_t view_byte_count = 3 * sizeof(int32_t);

  auto tensor = Ort::Value::CreateTensor(
      cpu_info, static_cast<void*>(buffer.data()),
      view_byte_count, byte_offset,
      shape.data(), shape.size(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);

  ASSERT_TRUE(tensor.IsTensor());
  const int32_t* data = tensor.GetTensorData<int32_t>();

  EXPECT_EQ(data, buffer.data() + 2) << "Tensor data pointer should point to buffer[2]";

  EXPECT_EQ(data[0], 30);
  EXPECT_EQ(data[1], 40);
  EXPECT_EQ(data[2], 50);
}

// Test that a zero offset produces the same result as no-offset CreateTensor.
TEST(CreateTensorWithOffsetTest, ZeroOffsetMatchesNoOffset) {
  std::vector<double> buffer = {1.1, 2.2, 3.3};

  Ort::AllocatorWithDefaultOptions cpu_alloc;
  const OrtMemoryInfo* cpu_info = cpu_alloc.GetInfo();
  constexpr std::array<int64_t, 1> shape = {3};

  auto t_no_offset = Ort::Value::CreateTensor<double>(
      cpu_info, buffer.data(), 3, shape.data(), shape.size());

  auto t_zero_offset = Ort::Value::CreateTensor<double>(
      cpu_info, buffer.data(), 3, /*element_offset=*/0, shape.data(), shape.size());

  EXPECT_EQ(t_no_offset.GetTensorData<double>(), t_zero_offset.GetTensorData<double>());
}

// Test the intended usage pattern: use CreateTensor-with-offset together with
// CopyTensors to copy a sub-region of a source tensor into a destination.
// (CPU -> CPU copy via the default CPU data transfer.)
TEST(CreateTensorWithOffsetTest, UsageWithCopyTensors) {
  // Source buffer: [0, 1, 2, 3, 4, 5, 6, 7]
  std::vector<float> src_buffer = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
  std::vector<float> dst_buffer(4, -1.f);

  Ort::AllocatorWithDefaultOptions cpu_alloc;
  const OrtMemoryInfo* cpu_info = cpu_alloc.GetInfo();

  // Create a view over src[4..7].
  constexpr std::array<int64_t, 1> shape = {4};
  auto src_view = Ort::Value::CreateTensor<float>(
      cpu_info, src_buffer.data(),
      /*element_count=*/4, /*element_offset=*/4,
      shape.data(), shape.size());

  // Create a full destination tensor.
  auto dst = Ort::Value::CreateTensor<float>(
      cpu_info, dst_buffer.data(), 4, shape.data(), shape.size());

  // Copy src view -> dst using the standard CopyTensors API.
  // Ort::Value is move-only, so we can't use brace-init to construct a vector from lvalues.
  std::vector<Ort::Value> copy_src, copy_dst;
  copy_src.push_back(std::move(src_view));
  copy_dst.push_back(std::move(dst));
  ASSERT_CXX_ORTSTATUS_OK(ort_env->CopyTensors(copy_src, copy_dst, /*stream=*/nullptr));

  EXPECT_EQ(dst_buffer[0], 4.f);
  EXPECT_EQ(dst_buffer[1], 5.f);
  EXPECT_EQ(dst_buffer[2], 6.f);
  EXPECT_EQ(dst_buffer[3], 7.f);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
