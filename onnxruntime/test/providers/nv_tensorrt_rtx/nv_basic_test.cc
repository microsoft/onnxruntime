// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"

#include "test/util/include/scoped_env_vars.h"
#include "test/common/trt_op_test_utils.h"
#include "test/common/random_generator.h"
#include "test/providers/nv_tensorrt_rtx/test_nv_trt_rtx_ep_util.h"

#include <thread>
#include <chrono>

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;
extern std::unique_ptr<Ort::Env> ort_env;
namespace onnxruntime {

namespace test {

TEST(NvExecutionProviderTest, ContextEmbedAndReload) {
  PathString model_name = ORT_TSTR("nv_execution_provider_test.onnx");
  PathString model_name_ctx = ORT_TSTR("nv_execution_provider_test_ctx.onnx");
  auto model_name_ctx_str = PathToUTF8(model_name_ctx);
  clearFileIfExists(model_name_ctx);
  std::string graph_name = "test";
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model_name, graph_name, dims);
  // AOT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, model_name_ctx_str.c_str());
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(*ort_env, model_name.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation AOT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }

  // JIT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(*ort_env, model_name_ctx.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation JIT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }
}

TEST(NvExecutionProviderTest, ContextEmbedAndReloadDynamic) {
  PathString model_name = ORT_TSTR("nv_execution_provider_dyn_test.onnx");
  PathString model_name_ctx = ORT_TSTR("nv_execution_provider_dyn_test_ctx.onnx");
  auto model_name_ctx_str = PathToUTF8(model_name_ctx);
  clearFileIfExists(model_name_ctx);
  std::string graph_name = "test";
  std::vector<int> dims = {1, -1, -1};

  CreateBaseModel(model_name, graph_name, dims);

  // AOT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, model_name_ctx_str.c_str());
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(*ort_env, model_name.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation AOT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }

  // JIT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(*ort_env, model_name_ctx.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation JIT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    std::map<std::string, std::vector<int64_t>> shape_overwrites;
    shape_overwrites["X"] = {1, 5, 5};
    shape_overwrites["Y"] = {1, 5, 1};
    auto io_binding = generate_io_binding(session_object, shape_overwrites);
    session_object.Run(run_options, io_binding);
  }
}

TEST(NvExecutionProviderTest, ContextEmbedAndReloadDataDynamic) {
  PathString model_name = ORT_TSTR("nv_execution_provider_data_dyn_test.onnx");
  PathString model_name_ctx = ORT_TSTR("nv_execution_provider_data_dyn_test_ctx.onnx");
  auto model_name_ctx_str = PathToUTF8(model_name_ctx);
  clearFileIfExists(model_name_ctx);
  std::string graph_name = "test";
  std::vector<int> dims = {1, -1, -1};

  CreateBaseModel(model_name, graph_name, dims);

  // AOT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, model_name_ctx_str.c_str());
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(*ort_env, model_name.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation AOT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }

  // JIT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(*ort_env, model_name_ctx.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation JIT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    std::map<std::string, std::vector<int64_t>> shape_overwrites;
    shape_overwrites["X"] = {1, 5, 5};
    shape_overwrites["Y"] = {1, 5, 5};
    auto io_binding = generate_io_binding(session_object, shape_overwrites);
    session_object.Run(run_options, io_binding);
  }
}

std::string getTypeAsName(ONNX_NAMESPACE::TensorProto_DataType dtype) {
  switch (dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "fp64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "fp32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "fp16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return "bf16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4:
      return "int4";
    default:
      return "Unkwon type";
  }
}

class TypeTests : public ::testing::TestWithParam<ONNX_NAMESPACE::TensorProto_DataType> {
 public:
};

TEST_P(TypeTests, IOTypes) {
  const std::string dtype_name = getTypeAsName(GetParam());
  ASSERT_FALSE(dtype_name.empty());
  const std::string model_name_str = "nv_execution_provider_" + dtype_name + ".onnx";
  const PathString model_name = ToPathString(model_name_str);
  const std::string graph_name = "test" + dtype_name;
  const std::vector<int> dims = {1, 5, 10};

  CreateBaseModel(model_name, graph_name, dims, false, GetParam());

  // AOT time
  {
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(*ort_env, model_name.c_str(), so);

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }
}

INSTANTIATE_TEST_SUITE_P(NvExecutionProviderTest, TypeTests,
                         ::testing::Values(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32
                                           // disabled low precision integer types since a specific quantize/dequantize model is required
                                           // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
                                           // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
                                           // ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4
                                           ),
                         [](const testing::TestParamInfo<TypeTests::ParamType>& info) { return getTypeAsName(info.param); });

#if defined(WIN32)
static bool SessionHasEp(Ort::Session& session, const char* ep_name) {
  // Access the underlying InferenceSession.
  const OrtSession* ort_session = session;
  const InferenceSession* s = reinterpret_cast<const InferenceSession*>(ort_session);
  bool has_ep = false;

  for (const auto& provider : s->GetRegisteredProviderTypes()) {
    if (provider == ep_name) {
      has_ep = true;
      break;
    }
  }
  return has_ep;
}

// Tests autoEP feature to automatically select an EP that supports the GPU.
// Currently only works on Windows.
TEST(NvExecutionProviderTest, AutoEp_PreferGpu) {
  PathString model_name = ORT_TSTR("nv_execution_provider_auto_ep.onnx");
  std::string graph_name = "test";

  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model_name, graph_name, dims);

  {
    ort_env->RegisterExecutionProviderLibrary(kNvTensorRTRTXExecutionProvider, ORT_TSTR("onnxruntime_providers_nv_tensorrt_rtx.dll"));

    Ort::SessionOptions so;
    so.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);
    Ort::Session session_object(*ort_env, model_name.c_str(), so);
    EXPECT_TRUE(SessionHasEp(session_object, kNvTensorRTRTXExecutionProvider));
  }

  ort_env->UnregisterExecutionProviderLibrary(kNvTensorRTRTXExecutionProvider);
}

TEST(NvExecutionProviderTest, GetSharedAllocator) {
  const OrtApi& c_api = Ort::GetApi();
  RegisteredEpDeviceUniquePtr nv_tensorrt_rtx_ep;
  Utils::RegisterAndGetNvTensorRtRtxEp(*ort_env, nv_tensorrt_rtx_ep);

  const auto* ep_memory_info = c_api.EpDevice_MemoryInfo(nv_tensorrt_rtx_ep.get(), OrtDeviceMemoryType_DEFAULT);

  // validate there is a shared allocator
  OrtAllocator* allocator = nullptr;
  ASSERT_ORTSTATUS_OK(c_api.GetSharedAllocator(*ort_env, ep_memory_info, &allocator));
  ASSERT_NE(allocator, nullptr);

  const auto* ep_host_accessible_memory_info = c_api.EpDevice_MemoryInfo(nv_tensorrt_rtx_ep.get(), OrtDeviceMemoryType_HOST_ACCESSIBLE);
  OrtAllocator* host_accessible_allocator = nullptr;
  ASSERT_ORTSTATUS_OK(c_api.GetSharedAllocator(*ort_env, ep_host_accessible_memory_info, &host_accessible_allocator));
  ASSERT_NE(host_accessible_allocator, nullptr);
}

TEST(NvExecutionProviderTest, LoadUnloadPluginLibrary) {
  const std::filesystem::path& library_path = Utils::nv_tensorrt_rtx_ep_info.library_path;
  const std::string& registration_name = Utils::nv_tensorrt_rtx_ep_info.registration_name;

  const OrtApi* c_api = &Ort::GetApi();
  // this should load the library and create OrtEpDevice
  ASSERT_ORTSTATUS_OK(Ort::GetApi().RegisterExecutionProviderLibrary(*ort_env, registration_name.c_str(),
                                                                     library_path.c_str()));

  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_devices = 0;

  ASSERT_ORTSTATUS_OK(Ort::GetApi().GetEpDevices(*ort_env, &ep_devices, &num_devices));
  // should be one device for the example EP
  auto num_test_ep_devices = std::count_if(ep_devices, ep_devices + num_devices,
                                           [&registration_name, &c_api](const OrtEpDevice* device) {
                                             // the example uses the registration name for the EP name
                                             // but that is not a requirement and the two can differ.
                                             return c_api->EpDevice_EpName(device) == registration_name;
                                           });
  ASSERT_EQ(num_test_ep_devices, 1) << "Expected an OrtEpDevice to have been created by the test library.";

  // and this should unload it
  ASSERT_ORTSTATUS_OK(Ort::GetApi().UnregisterExecutionProviderLibrary(*ort_env,
                                                                       registration_name.c_str()));
}

TEST(NvExecutionProviderTest, LoadUnloadPluginLibraryCxxApi) {
  const std::filesystem::path& library_path = Utils::nv_tensorrt_rtx_ep_info.library_path;
  const std::string& registration_name = Utils::nv_tensorrt_rtx_ep_info.registration_name;
  const OrtApi* c_api = &Ort::GetApi();
  // this should load the library and create OrtEpDevice
  ort_env->RegisterExecutionProviderLibrary(registration_name.c_str(), library_path.c_str());

  std::vector<Ort::ConstEpDevice> ep_devices = ort_env->GetEpDevices();

  auto test_ep_device = std::find_if(ep_devices.begin(), ep_devices.end(),
                                     [&registration_name, &c_api](const Ort::ConstEpDevice& device) {
                                       return device.EpName() == registration_name;
                                     });
  ASSERT_NE(test_ep_device, ep_devices.end()) << "Expected an OrtEpDevice to have been created by the test library.";

  // test all the C++ getters. expected values are from \onnxruntime\test\autoep\library\example_plugin_ep.cc
  ASSERT_STREQ(test_ep_device->EpVendor(), "NVIDIA");

  auto metadata = test_ep_device->EpMetadata();
  ASSERT_STREQ(metadata.GetValue(kOrtEpDevice_EpMetadataKey_Version), ORT_VERSION);

  // the GPU device info will vary by machine so check for the lowest common denominator values
  Ort::ConstHardwareDevice device = test_ep_device->Device();
  ASSERT_EQ(device.Type(), OrtHardwareDeviceType_GPU);
  ASSERT_GE(device.VendorId(), 0);
  ASSERT_GE(device.DeviceId(), 0);
  ASSERT_NE(device.Vendor(), nullptr);
  Ort::ConstKeyValuePairs device_metadata = device.Metadata();
  std::unordered_map<std::string, std::string> metadata_entries = device_metadata.GetKeyValuePairs();
  ASSERT_GT(metadata_entries.size(), 0);  // should have at least SPDRP_HARDWAREID on Windows

  // and this should unload it without throwing
  ort_env->UnregisterExecutionProviderLibrary(registration_name.c_str());
}

TEST(NvExecutionProviderTest, DataTransfer) {
  const OrtApi& c_api = Ort::GetApi();
  RegisteredEpDeviceUniquePtr nv_tensorrt_rtx_ep;
  Utils::RegisterAndGetNvTensorRtRtxEp(*ort_env, nv_tensorrt_rtx_ep);
  const OrtEpDevice* ep_device = nv_tensorrt_rtx_ep.get();

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

  // create an on-device Tensor using the NV TensorRT RTX EP  GPU allocator.

  OrtAllocator* allocator = nullptr;
  ASSERT_ORTSTATUS_OK(c_api.GetSharedAllocator(*ort_env, device_memory_info, &allocator));
  ASSERT_NE(allocator, nullptr);
  Ort::Value device_tensor = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());

  std::vector<const OrtValue*> src_tensor_ptrs{cpu_tensor};
  std::vector<OrtValue*> dst_tensor_ptrs{device_tensor};

  ASSERT_ORTSTATUS_OK(c_api.CopyTensors(*ort_env, src_tensor_ptrs.data(), dst_tensor_ptrs.data(), nullptr,
                                        src_tensor_ptrs.size()));

  // Copy data back from device_tensor to a new CPU tensor and verify the contents

  // Create a new CPU tensor to receive the data
  Ort::Value cpu_tensor_copy = Ort::Value::CreateTensor<float>(cpu_allocator, shape.data(), shape.size());

  std::vector<const OrtValue*> src_tensor_ptrs_back{device_tensor};
  std::vector<OrtValue*> dst_tensor_ptrs_back{cpu_tensor_copy};

  ASSERT_ORTSTATUS_OK(c_api.CopyTensors(*ort_env, src_tensor_ptrs_back.data(), dst_tensor_ptrs_back.data(), nullptr,
                                        src_tensor_ptrs_back.size()));

  const float* src_data = nullptr;
  ASSERT_ORTSTATUS_OK(c_api.GetTensorData(cpu_tensor, reinterpret_cast<const void**>(&src_data)));

  const float* cpu_copy_data = nullptr;
  ASSERT_ORTSTATUS_OK(c_api.GetTensorData(cpu_tensor_copy, reinterpret_cast<const void**>(&cpu_copy_data)));

  ASSERT_NE(src_data, cpu_copy_data) << "Should have copied between two different memory locations";

  size_t bytes;
  ASSERT_ORTSTATUS_OK(c_api.GetTensorSizeInBytes(cpu_tensor, &bytes));
  ASSERT_EQ(bytes, num_elements * sizeof(float));

  auto src_span = gsl::make_span(src_data, num_elements);
  auto cpu_copy_span = gsl::make_span(cpu_copy_data, num_elements);

  EXPECT_THAT(cpu_copy_span, ::testing::ContainerEq(src_span));

  // must release this before we unload the EP and the allocator is deleted
  device_tensor = Ort::Value();
}

#endif  // defined(WIN32)

}  // namespace test
}  // namespace onnxruntime
