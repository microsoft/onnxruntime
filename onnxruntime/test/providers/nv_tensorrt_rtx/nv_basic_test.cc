// Copyright (c) Microsoft Corporation. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test/framework/test_utils.h"
#include "gtest/gtest.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/common/trt_op_test_utils.h"
#include "test/providers/nv_tensorrt_rtx/test_nv_trt_rtx_ep_util.h"

#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_ep_device_ep_metadata_keys.h>
#include <onnxruntime_run_options_config_keys.h>
#include <onnxruntime_session_options_config_keys.h>
#include <string>
#include <thread>
#include <filesystem>
#include <chrono>

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;
extern std::unique_ptr<Ort::Env> ort_env;
namespace onnxruntime {

namespace test {

template <typename T>
class NvExecutionProviderTest : public ::testing::Test {
 protected:
  std::string getTypeAsName() {
    std::string dtype_name = "";
    if constexpr (std::is_same<T, double>::value) {
      dtype_name = "fp64";
    } else if constexpr (std::is_same<T, float>::value) {
      dtype_name = "fp32";
    } else if constexpr (std::is_same<T, BFloat16>::value) {
      dtype_name = "bf16";
    } else if constexpr (std::is_same<T, MLFloat16>::value) {
      dtype_name = "fp16";
    } else if constexpr (std::is_same<T, int8_t>::value) {
      dtype_name = "int8";
    } else if constexpr (std::is_same<T, uint8_t>::value) {
      dtype_name = "uint8";
    } else if constexpr (std::is_same<T, int32_t>::value) {
      dtype_name = "int32";
    } else if constexpr (std::is_same<T, int64_t>::value) {
      dtype_name = "int64";
    }
    return dtype_name;
  }
};

using NvExecutionProviderTestTypes = ::testing::Types<double, float, MLFloat16, BFloat16, uint8_t, int8_t, int32_t, int64_t>;  // double,
TYPED_TEST_SUITE(NvExecutionProviderTest, NvExecutionProviderTestTypes);

std::string PathToUTF8(const PathString& path) {
#ifdef WIN32
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.to_bytes(path);
#else
  return path.c_str();
#endif
}

void clearFileIfExists(PathString path) {
  if (std::filesystem::exists(path)) {
    std::filesystem::remove(path);
  }
}

template <typename T>
void VerifyOutputs(const std::vector<OrtValue>& fetches, const std::vector<int64_t>& expected_dims,
                   const std::vector<T>& expected_values) {
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<T> found(rtensor.Data<T>(), rtensor.Data<T>() + expected_values.size());
  ASSERT_EQ(expected_values, found);
}

/**
 * Create a simple model with dynamic or non-dynamic input shape.
 * \param model_name - model name
 * \param graph_name - graph name
 * \param dims - input dimensions
 * \param add_fast_gelu - add FastGelu node which makes the whole model partition into TRT EP and CUDA EP subgraphs.
 *
 * input: "X", "Y" and "Z"
 *        you can specify input dimensions, for example (1, 3, 2), (1, 2) or (1, -1, -1)). Note: -1 means the dimension is dynamic.
 *        All three inputs have the same dimensions.
 * output: "M"
 *
 *      "X"  "Y"
 *        \  /
 *    "Z"  Add
 *      \  /
 *       Add
 *       /
 *       Add (+ float scalar "S")
 *       /
 *     "O"
 *
 *     or
 *
 *      "X"  "Y"
 *        \  /
 *    "Z"  Add
 *      \  /
 *       Add
 *       /
 *    FastGelu (This node will be placed on CUDA EP)
 *     /
 *     *       Add (+ float scalar "S")
 *    /
 *   "O"
 */
static void CreateBaseModel(const PathString& model_name,
                            std::string graph_name,
                            std::vector<int> dims,
                            bool add_fast_gelu = false,
                            ONNX_NAMESPACE::TensorProto_DataType dtype = ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();
  std::vector<onnxruntime::NodeArg*> inputs;
  std::vector<onnxruntime::NodeArg*> outputs;

  // FLOAT tensor
  ONNX_NAMESPACE::TypeProto float_tensor;
  float_tensor.mutable_tensor_type()->set_elem_type(dtype);

  for (auto dim : dims) {
    float_tensor.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim);
  }
  ONNX_NAMESPACE::TypeProto dyn_float_tensor;
  dyn_float_tensor.mutable_tensor_type()->set_elem_type(dtype);

  auto& input_arg_1 = graph.GetOrCreateNodeArg("X", &float_tensor);
  auto& input_arg_2 = graph.GetOrCreateNodeArg("Y", &float_tensor);
  inputs.push_back(&input_arg_1);
  inputs.push_back(&input_arg_2);
  auto& output_arg = graph.GetOrCreateNodeArg("node_1_out_1", &float_tensor);
  outputs.push_back(&output_arg);
  graph.AddNode("node_1", "Add", "node 1.", inputs, outputs);

  auto& input_arg_3 = graph.GetOrCreateNodeArg("Z", &float_tensor);
  inputs.clear();
  inputs.push_back(&output_arg);
  inputs.push_back(&input_arg_3);

  auto& output_arg_2 = graph.GetOrCreateNodeArg("node_2_out_1", &float_tensor);
  outputs.clear();
  outputs.push_back(&output_arg_2);
  graph.AddNode("node_2", "Add", "node 2.", inputs, outputs);

  inputs.clear();
  inputs.push_back(&output_arg_2);

  if (add_fast_gelu) {
    auto& output_arg_3 = graph.GetOrCreateNodeArg("node_3_out_1", &dyn_float_tensor);
    outputs.clear();
    outputs.push_back(&output_arg_3);

    graph.AddNode("node_3", "FastGelu", "node 3.", inputs, outputs,
                  /* attributes */ nullptr, kMSDomain);

    inputs.clear();
    inputs.push_back(&output_arg_3);
  }

  ONNX_NAMESPACE::TypeProto float_scalar;
  float_scalar.mutable_tensor_type()->set_elem_type(dtype);
  float_scalar.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  auto& input_scalar = graph.GetOrCreateNodeArg("S", &float_scalar);
  inputs.push_back(&input_scalar);

  auto& output_arg_4 = graph.GetOrCreateNodeArg("O", &dyn_float_tensor);

  outputs.clear();
  outputs.push_back(&output_arg_4);
  graph.AddNode("node_5", "Add", "node 5.", inputs, outputs);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK());
  status = onnxruntime::Model::Save(model, model_name);
  ASSERT_TRUE(status.IsOK());
}

static Ort::IoBinding generate_io_binding(Ort::Session& session, std::map<std::string, std::vector<int64_t>> shape_overwrites = {}) {
  Ort::IoBinding binding(session);
  auto allocator = Ort::AllocatorWithDefaultOptions();
  for (int input_idx = 0; input_idx < int(session.GetInputCount()); ++input_idx) {
    auto input_name = session.GetInputNameAllocated(input_idx, Ort::AllocatorWithDefaultOptions());
    auto full_tensor_info = session.GetInputTypeInfo(input_idx);
    auto tensor_info = full_tensor_info.GetTensorTypeAndShapeInfo();
    auto shape = tensor_info.GetShape();
    auto type = tensor_info.GetElementType();
    if (shape_overwrites.find(input_name.get()) == shape_overwrites.end()) {
      for (auto& v : shape) {
        if (v == -1) {
          v = 1;
        }
      }
    } else {
      shape = shape_overwrites[input_name.get()];
    }
    auto input_value = Ort::Value::CreateTensor(allocator,
                                                shape.data(),
                                                shape.size(),
                                                type);
    binding.BindInput(input_name.get(), input_value);
  }

  for (int output_idx = 0; output_idx < int(session.GetOutputCount()); ++output_idx) {
    auto output_name = session.GetOutputNameAllocated(output_idx, Ort::AllocatorWithDefaultOptions());
    binding.BindOutput(output_name.get(), allocator.GetInfo());
  }
  return binding;
}

TEST(NvExecutionProviderTest, ContextEmbedAndReload) {
  PathString model_name = ORT_TSTR("nv_execution_provider_test.onnx");
  PathString model_name_ctx = ORT_TSTR("nv_execution_provider_test_ctx.onnx");
  auto model_name_ctx_str = PathToUTF8(model_name_ctx);
  clearFileIfExists(model_name_ctx);
  std::string graph_name = "test";
  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model_name, graph_name, dims);

  auto env = Ort::Env();
  auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
  env.UpdateEnvWithCustomLogLevel(logging_level);

  // AOT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, model_name_ctx_str.c_str());
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(env, model_name.c_str(), so);
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
    Ort::Session session_object(env, model_name_ctx.c_str(), so);
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

  auto env = Ort::Env();
  auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
  env.UpdateEnvWithCustomLogLevel(logging_level);

  // AOT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, model_name_ctx_str.c_str());
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(env, model_name.c_str(), so);
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
    Ort::Session session_object(env, model_name_ctx.c_str(), so);
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

  auto env = Ort::Env();
  auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
  env.UpdateEnvWithCustomLogLevel(logging_level);

  // AOT time
  {
    auto start = std::chrono::high_resolution_clock::now();
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextFilePath, model_name_ctx_str.c_str());
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(env, model_name.c_str(), so);
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
    Ort::Session session_object(env, model_name_ctx.c_str(), so);
    auto stop = std::chrono::high_resolution_clock::now();
    std::cout << "Session creation JIT: " << std::chrono::duration_cast<std::chrono::milliseconds>((stop - start)).count() << " ms" << std::endl;

    std::map<std::string, std::vector<int64_t>> shape_overwrites;
    shape_overwrites["X"] = {1, 5, 5};
    shape_overwrites["Y"] = {1, 5, 5};
    auto io_binding = generate_io_binding(session_object, shape_overwrites);
    session_object.Run(run_options, io_binding);
  }
}

TYPED_TEST(NvExecutionProviderTest, IOTypeTests) {
  std::string dtype_name = this->getTypeAsName();
  ASSERT_FALSE(dtype_name.empty());
  const std::string model_name_str = "nv_execution_provider_" + dtype_name + ".onnx";
  const PathString model_name = ToPathString(model_name_str);
  std::string graph_name = "test" + dtype_name;
  std::vector<int> dims = {1, -1, -1};

  CreateBaseModel(model_name, graph_name, dims);

  auto env = Ort::Env();
  auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
  env.UpdateEnvWithCustomLogLevel(logging_level);

  // AOT time
  {
    Ort::SessionOptions so;
    Ort::RunOptions run_options;
    so.AppendExecutionProvider(kNvTensorRTRTXExecutionProvider, {});
    Ort::Session session_object(env, model_name.c_str(), so);

    auto io_binding = generate_io_binding(session_object);
    session_object.Run(run_options, io_binding);
  }
}

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

#if defined(WIN32)
// Tests autoEP feature to automatically select an EP that supports the GPU.
// Currently only works on Windows.
TEST(NvExecutionProviderTest, AutoEp_PreferGpu) {
  PathString model_name = ORT_TSTR("nv_execution_provider_data_dyn_test.onnx");
  std::string graph_name = "test";

  std::vector<int> dims = {1, 3, 2};

  CreateBaseModel(model_name, graph_name, dims);

  auto env = Ort::Env();
  auto logging_level = OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING;
  env.UpdateEnvWithCustomLogLevel(logging_level);

  {
    env.RegisterExecutionProviderLibrary(kNvTensorRTRTXExecutionProvider, ORT_TSTR("onnxruntime_providers_nv_tensorrt_rtx.dll"));

    Ort::SessionOptions so;
    so.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);
    Ort::Session session_object(env, model_name.c_str(), so);
    EXPECT_TRUE(SessionHasEp(session_object, kNvTensorRTRTXExecutionProvider));
  }

  env.UnregisterExecutionProviderLibrary(kNvTensorRTRTXExecutionProvider);
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
  //ASSERT_STREQ(metadata.GetValue("supported_devices"), "NVIDIA RTX");

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

#endif  // defined(WIN32)

}  // namespace test
}  // namespace onnxruntime
