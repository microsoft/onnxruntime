// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <vector>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

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

    std::vector<Ort::Value> src_tensors;
    std::vector<Ort::Value> dst_tensors;
    std::vector<OrtSyncStream*> copy_streams;
    std::vector<OrtValue*> input_tensors;

    src_tensors.reserve(num_inputs);
    dst_tensors.reserve(num_inputs);
    copy_streams.reserve(num_inputs);
    input_tensors.reserve(num_inputs);

    Ort::AllocatorWithDefaultOptions cpu_allocator;

    for (size_t idx = 0; idx < num_inputs; ++idx) {
      const OrtMemoryInfo* mem_info = input_locations[idx];
      OrtDeviceMemoryType mem_type;
      OrtMemoryInfoDeviceType device_type;
      api->MemoryInfoGetDeviceMemType(mem_info, &mem_type);
      api->MemoryInfoGetDeviceType(mem_info, &device_type);

      // create cpu based input data.
      int64_t shape[4] = {1, 1, 28, 28};
      std::vector<float> input_data(28 * 28, 0.5f);
      size_t data_len = 28 * 28 * sizeof(float);
      Ort::Value src_value = Ort::Value::CreateTensor<float>(cpu_allocator.GetInfo(), input_data.data(), data_len,
                                                             shape, 4);
      src_tensors.push_back(std::move(src_value));

      if (device_type != OrtMemoryInfoDeviceType_CPU &&
          mem_type == OrtDeviceMemoryType_DEFAULT) {
        // copy to device
        OrtAllocator* allocator = nullptr;
        ASSERT_ORTSTATUS_OK(api->GetSharedAllocator(env, mem_info, &allocator));

        // allocate new on-device memory
        Ort::Value dst_value = Ort::Value::CreateTensor<float>(allocator, shape, sizeof(shape) / sizeof(shape[0]));

        /* if you have existing memory on device use one of these instead of CreateTensorAsOrtValue + CopyTensors
        void* existing_data;
        size_t data_length = 128 * sizeof(float);
        api->CreateTensorWithDataAsOrtValue(input_locations[0], existing_data, data_length, shape, 2,
                                            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value);

        // existing with ownership transfer. ORT will use the allocator to free the memory once it is no longer required
        api->CreateTensorWithDataAndDeleterAsOrtValue(allocator, existing_data, data_length, shape, 2,
                                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &value);
        */

        dst_tensors.push_back(std::move(dst_value));
        input_tensors.push_back(dst_tensors.back());

        // we create one stream per input here but we could also use the same stream instance for multiple inputs.
        OrtSyncStream* stream = nullptr;
        if (use_streams && input_ep_devices[idx] != nullptr) {
          ASSERT_ORTSTATUS_OK(api->CreateSyncStreamForEpDevice(input_ep_devices[idx], &stream));
          StreamUniquePtr stream_ptr(stream, [api](OrtSyncStream* stream) { api->ReleaseSyncStream(stream); });
          streams.push_back(std::move(stream_ptr));

          // Add a Notification so we can connect to the internal Stream used for inferencing
          // ASSERT_ORTSTATUS_OK(api->CreateInputSyncNotification(stream));
        }

        copy_streams.push_back(stream);

      } else {
        // input is on CPU accessible memory
        input_tensors.push_back(src_tensors.back());
      }
    }

    if (!dst_tensors.empty()) {
      std::vector<const OrtValue*> src_tensor_ptrs;
      std::vector<OrtValue*> dst_tensor_ptrs;
      std::transform(src_tensors.begin(), src_tensors.end(), std::back_inserter(src_tensor_ptrs),
                     [](const Ort::Value& ort_value) -> const OrtValue* {
                       return ort_value;
                     });

      std::transform(dst_tensors.begin(), dst_tensors.end(), std::back_inserter(dst_tensor_ptrs),
                     [](const Ort::Value& ort_value) -> OrtValue* {
                       return ort_value;
                     });

      api->CopyTensors(env, src_tensor_ptrs.data(), dst_tensor_ptrs.data(), copy_streams.data(), src_tensors.size());

      // manual sync until streams are hooked up to the inference session
      Ort::IoBinding iobinding(session);
      iobinding.SynchronizeInputs();  // this doesn't actually require any bound inputs

      // for (const auto& stream : streams) {
      //   ASSERT_ORTSTATUS_OK(api->ActivateInputSyncNotification(stream.get()));
      // }

      // TODO: Need a Run function that takes the input streams and an optional output stream
      // Do we need multiple streams for outputs?
      // Should we also have a simplified version that takes one overall input and output stream?
    }

    std::vector<const char*> input_names = {"Input3"};
    const char* output_name = "Plus214_Output_0";
    OrtValue* output = nullptr;
    // TODO: The C++ API should support providing Ort::Value for outputs without requiring IoBinding to be used.
    ASSERT_ORTSTATUS_OK(api->Run(session, Ort::RunOptions{}, input_names.data(),
                                 input_tensors.data(), input_tensors.size(),
                                 &output_name, 1, &output));

    const float* results = nullptr;
    ASSERT_ORTSTATUS_OK(api->GetTensorData(output, reinterpret_cast<const void**>(&results)));

    for (size_t i = 0; i < 10; ++i) {
      std::cout << i << ": " << results[i] << " ";
    }

    std::cout << std::endl;
  };

  run_test(/*use_streams*/ false);

  // CUDA EP doesn't implement streams support yet
  // run_test(true);
}
}  // namespace test
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
