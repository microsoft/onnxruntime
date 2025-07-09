// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

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
TEST(DataCopyTest, CopyInputsToCudaDevice) {
  OrtEnv* env = *ort_env;
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

  // register the provider bridge based CUDA EP so allocator and data transfer is available
  api->RegisterExecutionProviderLibrary(env, "ORT CUDA", ORT_TSTR("onnxruntime_providers_cuda"));

  const OrtEpDevice* cuda_device = nullptr;
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
    OrtSyncStream* stream = nullptr;
    StreamUniquePtr stream_ptr;
    if (use_streams) {
      ASSERT_ORTSTATUS_OK(api->CreateSyncStreamForEpDevice(cuda_device, /*options*/ nullptr, &stream));
      stream_ptr = StreamUniquePtr(stream, [api](OrtSyncStream* stream) { api->ReleaseSyncStream(stream); });

      size_t stream_addr = reinterpret_cast<size_t>(api->SyncStream_GetHandle(stream));
      options.AddConfigEntry("ep.cudaexecutionprovider.user_compute_stream", std::to_string(stream_addr).c_str());
      // no idea why this is needed...
      options.AddConfigEntry("ep.cudaexecutionprovider.has_user_compute_stream", "1");
    }

    Ort::Session session(*ort_env, ORT_TSTR("testdata/mnist.opset12.onnx"), options);

    size_t num_inputs = session.GetInputCount();

    // find the input location so we know which inputs can be provided on device.
    std::vector<const OrtMemoryInfo*> input_locations;
    input_locations.resize(num_inputs, nullptr);
    api->SessionGetMemoryInfoForInputs(session, input_locations.data(), num_inputs);

    std::vector<Ort::Value> cpu_tensors;

    // info for device copy
    std::vector<const OrtValue*> src_tensor_ptrs;
    std::vector<OrtValue*> dst_tensor_ptrs;

    // values we'll call Run with
    std::vector<Ort::Value> input_tensors;

    ASSERT_EQ(num_inputs, 1);

    // create cpu based input data.
    Ort::AllocatorWithDefaultOptions cpu_allocator;
    int64_t shape[4] = {1, 1, 28, 28};
    std::vector<float> input_data(28 * 28, 0.5f);
    size_t data_len = 28 * 28 * sizeof(float);
    Ort::Value input_value = Ort::Value::CreateTensor<float>(cpu_allocator.GetInfo(), input_data.data(), data_len,
                                                             shape, 4);
    cpu_tensors.push_back(std::move(input_value));

    for (size_t idx = 0; idx < num_inputs; ++idx) {
      const OrtMemoryInfo* mem_info = input_locations[idx];
      OrtDeviceMemoryType mem_type;
      OrtMemoryInfoDeviceType device_type;
      ASSERT_ORTSTATUS_OK(api->MemoryInfoGetDeviceMemType(mem_info, &mem_type));
      api->MemoryInfoGetDeviceType(mem_info, &device_type);

      if (device_type == OrtMemoryInfoDeviceType_GPU && mem_type == OrtDeviceMemoryType_DEFAULT) {
        // copy to device
        OrtAllocator* allocator = nullptr;
        ASSERT_ORTSTATUS_OK(api->GetSharedAllocator(env, mem_info, &allocator));

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

        src_tensor_ptrs.push_back(cpu_tensors[idx]);
        dst_tensor_ptrs.push_back(device_value);
        input_tensors.push_back(std::move(device_value));
      } else {
        // input is on CPU accessible memory. move to input_tensors
        input_tensors.push_back(std::move(cpu_tensors[idx]));
      }
    }

    if (!src_tensor_ptrs.empty()) {
      api->CopyTensors(env, src_tensor_ptrs.data(), dst_tensor_ptrs.data(), stream, cpu_tensors.size());

      // Stream support is still a work in progress.
      //
      // CUDA EP can use a user provided stream via provider options, so we can pass in the cudaStream_t from the
      // OrtSyncStream used in CopyTensors call that way.
      //
      // Alternatively you can manually sync the device via IoBinding.
      // Ort::IoBinding iobinding(session);
      // iobinding.SynchronizeInputs();  // this doesn't actually require any bound inputs
    }

    std::vector<const char*> input_names = {"Input3"};
    std::vector<const char*> output_names = {"Plus214_Output_0"};
    Ort::Value output;

    session.Run(Ort::RunOptions{}, input_names.data(), input_tensors.data(), input_tensors.size(),
                output_names.data(), &output, 1);

    const float* results = nullptr;
    ASSERT_ORTSTATUS_OK(api->GetTensorData(output, reinterpret_cast<const void**>(&results)));

    // expected results from the CPU EP. can check/re-create by running with PREFER_CPU.
    std::vector<float> expected = {
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
      EXPECT_NEAR(expected[i], results[i], 1e-5) << "i=" << i;
    }
  };

  run_test(/*use_streams*/ true);
  run_test(/*use_streams*/ false);
}
#endif  // USE_CUDA

}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
