// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testPch.h"

#include <functional>

#include "cppwinrt_onnx.h"

#include "AdapterSessionTest.h"
#include "ILotusValueProviderPrivate.h"
#include "onnxruntime_c_api.h"
#include "OnnxruntimeEngine.h"
#include "OnnxruntimeErrors.h"
#include "OnnxruntimeModel.h"
#include "core/common/logging/isink.h"
#include "core/common/logging/logging.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/ort_env.h"

using namespace _winml;
using namespace winrt::Windows::Foundation::Collections;
using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Media;
using namespace winrt::Windows::Storage;
using namespace winrt::Windows::Storage::Streams;

namespace {
winrt::com_ptr<_winml::OnnxruntimeEngineFactory> engine_factory;
const OrtApi *ort_api;
const WinmlAdapterApi *winml_adapter_api;
OrtEnv* ort_env;

void AdapterSessionTestSetup() {
  winrt::init_apartment();
#ifdef BUILD_INBOX
  winrt_activation_handler = WINRT_RoGetActivationFactory;
#endif
  WINML_EXPECT_HRESULT_SUCCEEDED(Microsoft::WRL::MakeAndInitialize<_winml::OnnxruntimeEngineFactory>(engine_factory.put()));
  WINML_EXPECT_HRESULT_SUCCEEDED(engine_factory->GetOrtEnvironment(&ort_env));
  WINML_EXPECT_NOT_EQUAL(nullptr, winml_adapter_api = engine_factory->UseWinmlAdapterApi());
  WINML_EXPECT_NOT_EQUAL(nullptr, ort_api = engine_factory->UseOrtApi());
}

void AdapterSessionTestTeardown() {
  engine_factory = nullptr;
}

UniqueOrtSessionOptions CreateUniqueOrtSessionOptions() {
  OrtSessionOptions *options;
  THROW_IF_NOT_OK_MSG(ort_api->CreateSessionOptions(&options), ort_api);
  return UniqueOrtSessionOptions(options, ort_api->ReleaseSessionOptions);
}

void AppendExecutionProvider_CPU() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  THROW_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_CPU(session_options.get(), true), ort_api);
}

winrt::com_ptr<ID3D12Device> CreateD3DDevice() {
  winrt::com_ptr<ID3D12Device> device;
  WINML_EXPECT_NO_THROW(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.put())));
  return device;
}

winrt::com_ptr<ID3D12CommandQueue> CreateD3DQueue(ID3D12Device* device) {
  winrt::com_ptr<ID3D12CommandQueue> queue;
  D3D12_COMMAND_QUEUE_DESC command_queue_desc = {};
  command_queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  device->CreateCommandQueue(&command_queue_desc, IID_PPV_ARGS(queue.put()));
  return queue;
}

UniqueOrtSession CreateUniqueOrtSession(const UniqueOrtSessionOptions& session_options) {
  OrtSession* session;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->CreateSessionWithoutModel(ort_env, session_options.get(), &session), ort_api);
  return UniqueOrtSession(session, ort_api->ReleaseSession);
}

UniqueOrtSession CreateUniqueOrtSession(const std::wstring& model_path, const UniqueOrtSessionOptions& session_options) {
  OrtSession* session;
  ort_api->SetIntraOpNumThreads(session_options.get(), 1);
  ort_api->SetSessionGraphOptimizationLevel(session_options.get(), ORT_ENABLE_BASIC);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_CPU(session_options.get(), true), ort_api);
  THROW_IF_NOT_OK_MSG(ort_api->CreateSession(ort_env, model_path.c_str(), session_options.get(), &session), ort_api);
  return UniqueOrtSession(session, ort_api->ReleaseSession);
}

void AppendExecutionProvider_DML() {
  const auto session_options = CreateUniqueOrtSessionOptions();

  const auto device = CreateD3DDevice();
  const auto queue = CreateD3DQueue(device.get());
  THROW_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_DML(session_options.get(), device.get(), queue.get(), true), ort_api);
}

void CreateWithoutModel() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  CreateUniqueOrtSession(session_options);
}

void GetExecutionProvider() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  const auto model_path = FileHelpers::GetModulePath() + L"fns-candy.onnx";
  auto session = CreateUniqueOrtSession(model_path, session_options);

  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
}

void GetExecutionProvider_DML() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  THROW_IF_NOT_OK_MSG(ort_api->DisableMemPattern(session_options.get()), ort_api);
  const auto device = CreateD3DDevice();
  const auto queue = CreateD3DQueue(device.get());
  THROW_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_DML(session_options.get(), device.get(), queue.get(), true), ort_api);

  const auto model_path = FileHelpers::GetModulePath() + L"fns-candy.onnx";
  auto session = CreateUniqueOrtSession(model_path, session_options);

  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  // Test if DML EP method can be called
  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlExecutionProviderFlushContext(ort_provider), ort_api);
}

void RegisterGraphTransformers() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  auto session = CreateUniqueOrtSession(session_options);
  winml_adapter_api->SessionRegisterGraphTransformers(session.get());
}

void RegisterGraphTransformers_DML() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  auto session = CreateUniqueOrtSession(session_options);
  winml_adapter_api->SessionRegisterGraphTransformers(session.get());
}

void RegisterCustomRegistry() {
  IMLOperatorRegistry* registry;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->CreateCustomRegistry(&registry), ort_api);
  if (registry) {
    const auto session_options = CreateUniqueOrtSessionOptions();
    auto session = CreateUniqueOrtSession(session_options);
    THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionRegisterCustomRegistry(session.get(), registry), ort_api);
  }
}

void RegisterCustomRegistry_DML() {
  IMLOperatorRegistry* registry;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->CreateCustomRegistry(&registry), ort_api);
  WINML_EXPECT_NOT_EQUAL(nullptr, registry);
  const auto session_options = CreateUniqueOrtSessionOptions();
  auto session = CreateUniqueOrtSession(session_options);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionRegisterCustomRegistry(session.get(), registry), ort_api);
}

void LoadAndPurloinModel(const UniqueOrtSession& session, const std::string& model_path) {
  winrt::com_ptr<_winml::IModel> model;
  WINML_THROW_IF_FAILED(engine_factory->CreateModel(model_path.c_str(), sizeof(model_path), model.put()));

  winrt::com_ptr<_winml::IOnnxruntimeModel> onnxruntime_model;
  WINML_EXPECT_NO_THROW(onnxruntime_model = model.as<_winml::IOnnxruntimeModel>());
  OrtModel* ort_model = nullptr;
  WINML_EXPECT_HRESULT_SUCCEEDED(onnxruntime_model->DetachOrtModel(&ort_model));
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionLoadAndPurloinModel(session.get(), ort_model), ort_api);
}

void LoadAndPurloinModel() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  auto session = CreateUniqueOrtSession(session_options);
  LoadAndPurloinModel(session, "fns-candy.onnx");
}

void Initialize() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  auto session = CreateUniqueOrtSession(session_options);

  winrt::com_ptr<_winml::IModel> model;
  const auto model_path = "fns-candy.onnx";
  WINML_THROW_IF_FAILED(engine_factory->CreateModel(model_path, sizeof(model_path), model.put()));

  winrt::com_ptr<_winml::IOnnxruntimeModel> onnxruntime_model;
  WINML_EXPECT_NO_THROW(onnxruntime_model = model.as<_winml::IOnnxruntimeModel>());
  OrtModel* ort_model = nullptr;
  WINML_EXPECT_HRESULT_SUCCEEDED(onnxruntime_model->DetachOrtModel(&ort_model));
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionLoadAndPurloinModel(session.get(), ort_model), ort_api);

  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionInitialize(session.get()), ort_api);
}

static bool logging_called = false, profile_called = false;
void Profiling() {
  const auto logging_callback = [](void*, OrtLoggingLevel, const char*, const char*, const char*, const char*) {
    logging_called = true;
  };
  const auto profile_callback = [](const OrtProfilerEventRecord*) {
    profile_called = true;
  };

  THROW_IF_NOT_OK_MSG(winml_adapter_api->EnvConfigureCustomLoggerAndProfiler(
      ort_env,
      logging_callback,
      profile_callback,
      nullptr,
      OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
      "Default",
      &ort_env),
    ort_api);

  const auto session_options = CreateUniqueOrtSessionOptions();
  auto session = CreateUniqueOrtSession(session_options);

  winml_adapter_api->SessionStartProfiling(ort_env, session.get());
  LoadAndPurloinModel(session, "fns-candy.onnx");
  winml_adapter_api->SessionEndProfiling(session.get());
  WINML_EXPECT_TRUE(logging_called);
  WINML_EXPECT_TRUE(profile_called);
}

void CopyInputAcrossDevices() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  auto session = CreateUniqueOrtSession(L"fns-candy.onnx", session_options);

  constexpr std::array<int64_t, 4> dimensions{1, 3, 720, 720};
  constexpr size_t input_tensor_size = [&dimensions]() {
    size_t size = 1;
    for (auto dim : dimensions)
      size *= static_cast<size_t>(dim);
    return size;
  } ();

  OrtMemoryInfo* memory_info;
  THROW_IF_NOT_OK_MSG(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info), ort_api);
  std::vector<float> input_tensor_values(input_tensor_size);
  OrtValue* input_tensor;
  THROW_IF_NOT_OK_MSG(ort_api->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size * sizeof(float), dimensions.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor), ort_api);

  int is_tensor;
  THROW_IF_NOT_OK_MSG(ort_api->IsTensor(input_tensor, &is_tensor), ort_api);
  WINML_EXPECT_TRUE(is_tensor);

  OrtValue* dest_ort_value;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionCopyOneInputAcrossDevices(session.get(), "inputImage", input_tensor, &dest_ort_value), ort_api);

  ort_api->ReleaseValue(input_tensor);
  ort_api->ReleaseValue(dest_ort_value);
  ort_api->ReleaseMemoryInfo(memory_info);
}

void CopyInputAcrossDevices_DML() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  THROW_IF_NOT_OK_MSG(ort_api->DisableMemPattern(session_options.get()), ort_api);
  const auto device = CreateD3DDevice();
  const auto queue = CreateD3DQueue(device.get());
  THROW_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_DML(session_options.get(), device.get(), queue.get(), true), ort_api);
  auto session = CreateUniqueOrtSession(session_options);

  LoadAndPurloinModel(session, "fns-candy.onnx");
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionInitialize(session.get()), ort_api);
  constexpr std::array<int64_t, 4> dimensions{1, 3, 720, 720};
  constexpr size_t input_tensor_size = [&dimensions]() {
    size_t size = 1;
    for (auto dim : dimensions)
      size *= static_cast<size_t>(dim);
    return size;
  } ();

  OrtMemoryInfo* memory_info;
  THROW_IF_NOT_OK_MSG(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info), ort_api);
  std::vector<float> input_tensor_values(input_tensor_size);
  OrtValue* input_tensor;
  THROW_IF_NOT_OK_MSG(ort_api->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size * sizeof(float), dimensions.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor), ort_api);

  int is_tensor;
  THROW_IF_NOT_OK_MSG(ort_api->IsTensor(input_tensor, &is_tensor), ort_api);
  WINML_EXPECT_TRUE(is_tensor);

  OrtValue* dest_ort_value = nullptr;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionCopyOneInputAcrossDevices(session.get(), "inputImage", input_tensor, &dest_ort_value), ort_api);

  ort_api->ReleaseValue(input_tensor);
  ort_api->ReleaseMemoryInfo(memory_info);
}

void GetNumberOfIntraOpThreads(){
  const auto session_options = CreateUniqueOrtSessionOptions();
  uint32_t desired_num_threads = std::thread::hardware_concurrency() / 2;
  ort_api->SetIntraOpNumThreads(session_options.get(), desired_num_threads);
  const auto session = CreateUniqueOrtSession(session_options);
  uint32_t num_threads;
  winml_adapter_api->SessionGetNumberOfIntraOpThreads(session.get(), &num_threads);
  WINML_EXPECT_EQUAL(num_threads, desired_num_threads);
}
}

const AdapterSessionTestAPI& getapi() {
  static AdapterSessionTestAPI api =
  {
    AdapterSessionTestSetup,
    AdapterSessionTestTeardown,
    AppendExecutionProvider_CPU,
    AppendExecutionProvider_DML,
    CreateWithoutModel,
    GetExecutionProvider,
    GetExecutionProvider_DML,
    Initialize,
    RegisterGraphTransformers,
    RegisterGraphTransformers_DML,
    RegisterCustomRegistry,
    RegisterCustomRegistry_DML,
    LoadAndPurloinModel,
    Profiling,
    CopyInputAcrossDevices,
    CopyInputAcrossDevices_DML,
    GetNumberOfIntraOpThreads
  };

  if (SkipGpuTests()) {
    api.AppendExecutionProvider_DML = SkipTest;
    api.GetExecutionProvider_DML = SkipTest;
    api.RegisterGraphTransformers_DML = SkipTest;
    api.RegisterCustomRegistry_DML = SkipTest;
    api.CopyInputAcrossDevices_DML = SkipTest;
  }
  if (SkipTestsImpactedByOpenMP()) {
    api.GetNumberOfIntraOpThreads = SkipTest;
  }
  return api;
}
