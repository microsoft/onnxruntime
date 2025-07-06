// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <vector>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/util/include/api_asserts.h"

#ifdef USE_CUDA
#include "core/providers/cuda/cuda_provider_options.h"
#include <cuda_runtime.h>
#endif

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

using StreamUniquePtr = std::unique_ptr<OrtSyncStream, std::function<void(OrtSyncStream*)>>;

TEST(DataCopyTest, CopyInputsToDevice) {
  OrtEnv* env = *ort_env;

  // OrtEnv* env = ort_env.get()->operator OrtEnv*();
  const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);

  // register the provider bridge based CUDA EP so allocator and data transfer is available
  api->RegisterExecutionProviderLibrary(env, "ORT CUDA", ORT_TSTR("onnxruntime_providers_cuda"));

  Ort::SessionOptions options;
  options.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);

  const auto run_test = [&](bool use_streams) {
    Ort::Session session(*ort_env, ORT_TSTR("testdata/mnist.opset12.onnx"), options);

    size_t num_inputs = session.GetInputCount();

    std::vector<const OrtMemoryInfo*> input_locations;
    input_locations.resize(num_inputs, nullptr);
    api->SessionGetMemoryInfoForInputs(session, input_locations.data(), num_inputs);

    std::vector<const OrtEpDevice*> input_ep_devices;
    if (use_streams) {
      input_ep_devices.resize(num_inputs, nullptr);
      api->SessionGetEpDeviceForInputs(session, input_ep_devices.data(), num_inputs);
    }

    std::vector<StreamUniquePtr> streams;

    std::vector<const OrtValue*> src_tensors;
    std::vector<OrtValue*> dst_tensors;
    std::vector<OrtSyncStream*> copy_streams;

    src_tensors.reserve(num_inputs);
    dst_tensors.reserve(num_inputs);
    copy_streams.reserve(num_inputs);

    for (size_t idx = 0; idx < num_inputs; ++idx) {
      const OrtMemoryInfo* mem_info = input_locations[idx];
      OrtDeviceMemoryType mem_type;
      OrtMemoryInfoDeviceType device_type;
      api->MemoryInfoGetDeviceMemType(mem_info, &mem_type);
      api->MemoryInfoGetDeviceType(mem_info, &device_type);

      if (device_type != OrtMemoryInfoDeviceType_CPU &&
          mem_type == OrtDeviceMemoryType_DEFAULT) {
        // copy to device
        OrtAllocator* allocator = nullptr;
        ASSERT_ORTSTATUS_OK(api->GetSharedAllocator(env, mem_info, &allocator));

        // allocate new on device memory
        int64_t shape[4] = {1, 1, 28, 28};
        OrtValue* dst_value = nullptr;
        ASSERT_ORTSTATUS_OK(api->CreateTensorAsOrtValue(allocator, shape, sizeof(shape),
                                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &dst_value));
        /* if you have existing memory on device use one of these instead of CreateTensorAsOrtValue + CopyTensors
        // use existing
        void* existing_data;
        size_t data_length = 128 * sizeof(float);
        api->CreateTensorWithDataAsOrtValue(input_locations[0], existing_data, data_length, shape, 2,
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value);

        // existing with ownership transfer. ORT will use the allocator to free the memory once it is no longer required
        api->CreateTensorWithDataAndDeleterAsOrtValue(allocator, existing_data, data_length, shape, 2,
                                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value);

        */

        // create cpu based input data.
        std::vector<float> input_data(28 * 28, 0.5f);
        size_t data_len = 28 * 28 * sizeof(float);
        OrtValue* src_value = nullptr;
        ASSERT_ORTSTATUS_OK(api->CreateTensorWithDataAsOrtValue(input_locations[0], input_data.data(), data_len,
                                                                shape, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                                                &src_value));

        src_tensors.push_back(src_value);
        dst_tensors.push_back(dst_value);

        // we create one stream per input here but we could also use the same stream instance for multiple inputs.
        OrtSyncStream* stream = nullptr;
        if (use_streams && input_ep_devices[idx] != nullptr) {
          ASSERT_ORTSTATUS_OK(api->CreateSyncStreamForEpDevice(input_ep_devices[idx], &stream));
          StreamUniquePtr stream_ptr(stream, [api](OrtSyncStream* stream) { api->ReleaseSyncStream(stream); });
          streams.push_back(std::move(stream_ptr));
        }

        copy_streams.push_back(stream);

      } else {
        // input is on CPU accessible memory
      }
    }

    api->CopyTensors(env, src_tensors.data(), dst_tensors.data(), copy_streams.data(), src_tensors.size());

    // currently we still need explicit device sync until we update h
  };

  run_test(/*use_streams*/ false);
  run_test(true);
}
}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
