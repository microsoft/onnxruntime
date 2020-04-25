// // Copyright (c) Microsoft Corporation. All rights reserved.
// // Licensed under the MIT License.

#include "testPch.h"

#include "AdapterDmlEpTest.h"

#include "common.h"
#include "winml_adapter_c_api.h"
#include "core/providers/winml/winml_provider_factory.h"
#include "OnnxruntimeErrors.h"
#include "UniqueOrtPtr.h"

namespace {
const WinmlAdapterApi* winml_adapter_api;
const OrtApi* ort_api;
OrtEnv* ort_env;

void AdapterDmlEpTestSetup() {
  GPUTEST;
  ort_api = OrtGetApiBase()->GetApi(2);
  winml_adapter_api = OrtGetWinMLAdapter(ort_api);
  THROW_IF_NOT_OK_MSG(ort_api->CreateEnv(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "Default", &ort_env), ort_api);
}

void AdapterDmlEpTestTeardown() {
  ort_api->ReleaseEnv(ort_env);
}

UniqueOrtSessionOptions CreateUniqueOrtSessionOptions() {
  OrtSessionOptions *options;
  THROW_IF_NOT_OK_MSG(ort_api->CreateSessionOptions(&options), ort_api);
  return UniqueOrtSessionOptions(options, ort_api->ReleaseSessionOptions);
}

UniqueOrtSession CreateUniqueOrtSession(const std::wstring& model_path, const UniqueOrtSessionOptions& session_options) {
  OrtSession* session;
  THROW_IF_NOT_OK_MSG(ort_api->CreateSession(ort_env, model_path.c_str(), session_options.get(), &session), ort_api);
  return UniqueOrtSession(session, ort_api->ReleaseSession);
}

UniqueOrtSession CreateDmlSession() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  THROW_IF_NOT_OK_MSG(ort_api->DisableMemPattern(session_options.get()), ort_api);

  winrt::com_ptr<ID3D12Device> device;
  WINML_EXPECT_NO_THROW(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.put())));

  winrt::com_ptr<ID3D12CommandQueue> queue;
  D3D12_COMMAND_QUEUE_DESC command_queue_desc = {};
  command_queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  device->CreateCommandQueue(&command_queue_desc, IID_PPV_ARGS(queue.put()));

  THROW_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_DML(session_options.get(), device.get(), queue.get()), ort_api);

  const auto model_path = FileHelpers::GetModulePath() + L"fns-candy.onnx";
  return CreateUniqueOrtSession(model_path, session_options);
}

void DmlExecutionProviderSetDefaultRoundingMode() {
  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlExecutionProviderSetDefaultRoundingMode(ort_provider, false), ort_api);
}

void DmlExecutionProviderFlushContext() {
  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlExecutionProviderFlushContext(ort_provider), ort_api);
}

void DmlExecutionProviderTrimUploadHeap() {
  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlExecutionProviderTrimUploadHeap(ort_provider), ort_api);
}

void DmlExecutionProviderReleaseCompletedReferences() {
  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlExecutionProviderReleaseCompletedReferences(ort_provider), ort_api);
}

winrt::com_ptr<ID3D12Resource> CreateD3D12Resource(ID3D12Device& device) {
  constexpr uint64_t buffer_size = 720 * 720 * 3 * sizeof(float);
  constexpr D3D12_HEAP_PROPERTIES heap_properties = {
      D3D12_HEAP_TYPE_DEFAULT,
      D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
      D3D12_MEMORY_POOL_UNKNOWN,
      0,
      0
  };
  constexpr D3D12_RESOURCE_DESC resource_desc = {
      D3D12_RESOURCE_DIMENSION_BUFFER,
      0,
      buffer_size,
      1,
      1,
      1,
      DXGI_FORMAT_UNKNOWN,
      {1, 0},
      D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
  };
  winrt::com_ptr<ID3D12Resource> d3d12_resource;
  device.CreateCommittedResource(
      &heap_properties,
      D3D12_HEAP_FLAG_NONE,
      &resource_desc,
      D3D12_RESOURCE_STATE_COMMON,
      nullptr,
      IID_PPV_ARGS(d3d12_resource.put()));
  return d3d12_resource;
}

void DmlCreateAndFreeGPUAllocationFromD3DResource() {
  winrt::com_ptr<ID3D12Device> device;
  WINML_EXPECT_NO_THROW(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.put())));

  auto d3d12_resource = CreateD3D12Resource(*device);
  void* dml_allocator_resource;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlCreateGPUAllocationFromD3DResource(d3d12_resource.get(), &dml_allocator_resource), ort_api);
  winml_adapter_api->DmlFreeGPUAllocation(dml_allocator_resource);
}

void DmlGetD3D12ResourceFromAllocation() {
  winrt::com_ptr<ID3D12Device> device;
  WINML_EXPECT_NO_THROW(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.put())));

  auto d3d12_resource = CreateD3D12Resource(*device);
  void* gpu_allocation;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlCreateGPUAllocationFromD3DResource(d3d12_resource.get(), &gpu_allocation), ort_api);

  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  winrt::com_ptr<ID3D12Resource> d3d12_resource_from_allocation;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlGetD3D12ResourceFromAllocation(ort_provider, gpu_allocation, d3d12_resource_from_allocation.put()), ort_api);
  // Ensure resource is the same
  WINML_EXPECT_EQUAL(d3d12_resource, d3d12_resource_from_allocation);

  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlFreeGPUAllocation(gpu_allocation), ort_api);
}

UniqueOrtValue CreateTensorFromMemoryInfo(OrtMemoryInfo* memory_info) {
  constexpr std::array<int64_t, 4> dimensions{1, 3, 720, 720};
  auto input_tensor_size = std::accumulate(begin(dimensions), end(dimensions), static_cast<int64_t>(1), std::multiplies<int64_t>());
  std::vector<float> input_tensor_values(input_tensor_size);

  OrtValue* tensor;
  THROW_IF_NOT_OK_MSG(ort_api->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size * sizeof(float), dimensions.data(), dimensions.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &tensor), ort_api);
  return UniqueOrtValue(tensor, ort_api->ReleaseValue);
}

void GetProviderMemoryInfo() {
  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  OrtMemoryInfo *memory_info;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->GetProviderMemoryInfo(ort_provider, &memory_info), ort_api);
  // Ensure tensor can be created with the provided OrtMemoryInfo
  CreateTensorFromMemoryInfo(memory_info);
  ort_api->ReleaseMemoryInfo(memory_info);
}

void GetAndFreeProviderAllocator() {
  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  OrtAllocator *allocator;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->GetProviderAllocator(ort_provider, &allocator), ort_api);

  // Ensure allocation works
  void *data = nullptr;
  THROW_IF_NOT_OK_MSG(ort_api->AllocatorAlloc(allocator, 1024, &data), ort_api);
  WINML_EXPECT_NOT_EQUAL(nullptr, data);
  THROW_IF_NOT_OK_MSG(ort_api->AllocatorFree(allocator, data), ort_api);

  THROW_IF_NOT_OK_MSG(winml_adapter_api->FreeProviderAllocator(allocator), ort_api);
}

void GetValueMemoryInfo() {
  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  OrtMemoryInfo *memory_info;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->GetProviderMemoryInfo(ort_provider, &memory_info), ort_api);
  auto tensor = CreateTensorFromMemoryInfo(memory_info);

  OrtMemoryInfo *value_memory_info;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->GetValueMemoryInfo(tensor.get(), &value_memory_info), ort_api);
  CreateTensorFromMemoryInfo(value_memory_info);

  ort_api->ReleaseMemoryInfo(value_memory_info);
  ort_api->ReleaseMemoryInfo(memory_info);
}

void ExecutionProviderSync() {
  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->ExecutionProviderSync(ort_provider), ort_api);
}

// winrt::com_ptr<ID3D12Device> CreateD3DDevice() {
//   winrt::com_ptr<ID3D12Device> device;
//   WINML_EXPECT_NO_THROW(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.put())));
//   return device;
// }

// winrt::com_ptr<ID3D12CommandQueue> CreateD3DQueue(ID3D12Device* device) {
//   winrt::com_ptr<ID3D12CommandQueue> queue;
//   D3D12_COMMAND_QUEUE_DESC command_queue_desc = {};
//   command_queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
//   device->CreateCommandQueue(&command_queue_desc, IID_PPV_ARGS(queue.put()));
//   return queue;
// }

// UniqueOrtSession CreateUniqueOrtSession(const UniqueOrtSessionOptions& session_options) {
//   OrtSession* session;
//   THROW_IF_NOT_OK_MSG(winml_adapter_api->CreateSessionWithoutModel(ort_env, session_options.get(), &session), ort_api);
//   return UniqueOrtSession(session, ort_api->ReleaseSession);
// }

// void LoadAndPurloinModel(const UniqueOrtSession& session, const std::string& model_path) {
//   winrt::com_ptr<_winml::IModel> model;
//   WINML_THROW_IF_FAILED(engine_factory->CreateModel(model_path.c_str(), sizeof(model_path), model.put()));

//   winrt::com_ptr<_winml::IOnnxruntimeModel> onnxruntime_model;
//   WINML_EXPECT_NO_THROW(onnxruntime_model = model.as<_winml::IOnnxruntimeModel>());
//   OrtModel* ort_model;
//   WINML_EXPECT_HRESULT_SUCCEEDED(onnxruntime_model->DetachOrtModel(&ort_model));
//   THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionLoadAndPurloinModel(session.get(), ort_model), ort_api);
// }

void DmlCopyTensor() {
  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  OrtMemoryInfo *memory_info;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->GetProviderMemoryInfo(ort_provider, &memory_info), ort_api);
  auto tensor = CreateTensorFromMemoryInfo(memory_info);

  OrtMemoryInfo* cpu_memory_info;
  THROW_IF_NOT_OK_MSG(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_memory_info), ort_api);
  auto cpu_tensor = CreateTensorFromMemoryInfo(cpu_memory_info);

  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlCopyTensor(ort_provider, tensor.get(), cpu_tensor.get()), ort_api);

  // const auto session_options = CreateUniqueOrtSessionOptions();
  // THROW_IF_NOT_OK_MSG(ort_api->DisableMemPattern(session_options.get()), ort_api);
  // const auto device = CreateD3DDevice();
  // const auto queue = CreateD3DQueue(device.get());
  // THROW_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_DML(session_options.get(), device.get(), queue.get()), ort_api);
  // auto session = CreateUniqueOrtSession(session_options);

  // LoadAndPurloinModel(session, "fns-candy.onnx");
  // THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionInitialize(session.get()), ort_api);
  // constexpr std::array<int64_t, 4> dimensions{1, 3, 720, 720};
  // constexpr size_t input_tensor_size = [&dimensions]() {
  //   size_t size = 1;
  //   for (auto dim : dimensions)
  //     size *= dim;
  //   return size;
  // } ();

  // OrtMemoryInfo* memory_info;
  // THROW_IF_NOT_OK_MSG(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info), ort_api);
  // std::vector<float> input_tensor_values(input_tensor_size);
  // OrtValue* input_tensor;
  // THROW_IF_NOT_OK_MSG(ort_api->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size * sizeof(float), dimensions.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor), ort_api);

  // int is_tensor;
  // THROW_IF_NOT_OK_MSG(ort_api->IsTensor(input_tensor, &is_tensor), ort_api);
  // WINML_EXPECT_TRUE(is_tensor);

  // OrtValue* dest_ort_value = nullptr;
  // THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionCopyOneInputAcrossDevices(session.get(), "inputImage", input_tensor, &dest_ort_value), ort_api);

  // ort_api->ReleaseValue(input_tensor);
  // ort_api->ReleaseMemoryInfo(memory_info);
}

void CreateCustomRegistry() {
  // THROW_IF_NOT_OK_MSG(winml_adapter_api->CreateCustomRegistry(), ort_api);
}

void ValueGetDeviceId() {
  // THROW_IF_NOT_OK_MSG(winml_adapter_api->ValueGetDeviceId(), ort_api);
}

void SessionGetInputRequiredDeviceId() {
  // THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetInputRequiredDeviceId(), ort_api);
}
}

const AdapterDmlEpTestApi& getapi() {
  static constexpr AdapterDmlEpTestApi api =
  {
    AdapterDmlEpTestSetup,
    AdapterDmlEpTestTeardown,
    DmlExecutionProviderSetDefaultRoundingMode,
    DmlExecutionProviderFlushContext,
    DmlExecutionProviderTrimUploadHeap,
    DmlExecutionProviderReleaseCompletedReferences,
    DmlCreateAndFreeGPUAllocationFromD3DResource,
    DmlGetD3D12ResourceFromAllocation,
    GetProviderMemoryInfo,
    GetAndFreeProviderAllocator,
    GetValueMemoryInfo,
    ExecutionProviderSync,
    DmlCopyTensor,
    CreateCustomRegistry,
    ValueGetDeviceId,
    SessionGetInputRequiredDeviceId
  };
  return api;
}
