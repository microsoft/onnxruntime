// // Copyright (c) Microsoft Corporation. All rights reserved.
// // Licensed under the MIT License.

#include "testPch.h"

#include "AdapterDmlEpTest.h"
#include <wil/result.h>

#include "common.h"
#include "winml_adapter_c_api.h"
#include "core/framework/execution_provider.h"
#include "core/providers/winml/winml_provider_factory.h"
#include "OnnxruntimeEngine.h"
#include "OnnxruntimeErrors.h"
#include "OnnxruntimeModel.h"
#include "UniqueOrtPtr.h"

namespace {
const WinmlAdapterApi* winml_adapter_api;
const OrtApi* ort_api;
OrtEnv* ort_env;
winrt::com_ptr<_winml::OnnxruntimeEngineFactory> engine_factory;
const OrtApi *ort_api;
const WinmlAdapterApi *winml_adapter_api;
OrtEnv* ort_env;
IEngine* engine;

void AdapterDmlEpTestSetup() {
  GPUTEST;
  winrt::init_apartment();
  WINML_EXPECT_HRESULT_SUCCEEDED(Microsoft::WRL::MakeAndInitialize<_winml::OnnxruntimeEngineFactory>(engine_factory.put()));
  IEngineBuilder *engine_builder;
  engine_factory->CreateEngineBuilder(&engine_builder);
  engine_builder->CreateEngine(&engine);
  ort_api = OrtGetApiBase()->GetApi(2);
  winml_adapter_api = OrtGetWinMLAdapter(ort_api);
  THROW_IF_NOT_OK_MSG(ort_api->CreateEnv(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "Default", &ort_env), ort_api);
}

void AdapterDmlEpTestTeardown() {
  ort_api->ReleaseEnv(ort_env);
  engine_factory = nullptr;
  engine = nullptr;
}

UniqueOrtSessionOptions CreateUniqueOrtSessionOptions() {
  OrtSessionOptions *options;
  THROW_IF_NOT_OK_MSG(ort_api->CreateSessionOptions(&options), ort_api);
  return UniqueOrtSessionOptions(options, ort_api->ReleaseSessionOptions);
}

UniqueOrtSession CreateUniqueOrtSession(const UniqueOrtSessionOptions& session_options) {
  OrtSession* session;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->CreateSessionWithoutModel(ort_env, session_options.get(), &session), ort_api);
  return UniqueOrtSession(session, ort_api->ReleaseSession);
}

UniqueOrtSession CreateUniqueOrtSession(const std::string& model_path, const UniqueOrtSessionOptions& session_options) {
  auto session = CreateUniqueOrtSession(session_options);
  OrtModel* ort_model;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->CreateModelFromPath(model_path.c_str(), model_path.size(), &ort_model), ort_api);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionLoadAndPurloinModel(session.get(), ort_model), ort_api);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionInitialize(session.get()), ort_api);
  return session;
}

UniqueOrtSession CreateDmlSession() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  THROW_IF_NOT_OK_MSG(ort_api->DisableMemPattern(session_options.get()), ort_api);

  winrt::com_ptr<ID3D12Device> device;
  WINML_EXPECT_NO_THROW(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.put())));

  winrt::com_ptr<ID3D12CommandQueue> queue;
  D3D12_COMMAND_QUEUE_DESC command_queue_desc = {};
  command_queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  WINML_EXPECT_HRESULT_SUCCEEDED(device->CreateCommandQueue(&command_queue_desc, IID_PPV_ARGS(queue.put())));

  THROW_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_DML(session_options.get(), device.get(), queue.get()), ort_api);
  const auto module_path = std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(FileHelpers::GetModulePath());
  return CreateUniqueOrtSession(module_path + "fns-candy.onnx", session_options);
}

UniqueOrtSession CreateCpuSession() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  const auto module_path = std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(FileHelpers::GetModulePath());
  return CreateUniqueOrtSession(module_path + "fns-candy.onnx", session_options);
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

UniqueOrtMemoryInfo GetCpuOutputMemoryInfo(OrtExecutionProvider* provider) {
  const auto execution_provider = reinterpret_cast<onnxruntime::IExecutionProvider*>(provider);
  auto allocator = execution_provider->GetAllocator(0, OrtMemTypeCPUOutput);
  const auto& info = allocator->Info();
  return UniqueOrtMemoryInfo(new OrtMemoryInfo(info.name, info.alloc_type, info.device, info.id, info.mem_type), ort_api->ReleaseMemoryInfo);
}

UniqueOrtValue CreateDmlTensor(OrtExecutionProvider* execution_provider, ID3D12Device* device) {
//  OnnxruntimeEngine::CreateTensorValueFromExternalD3DResource(ID3D12Resource* d3d_resource, const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out)
//  auto memory_info = GetCpuOutputMemoryInfo(execution_provider);
  OrtMemoryInfo* dml_memory = nullptr;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->GetProviderMemoryInfo(execution_provider, &dml_memory), ort_api);

  auto resource = CreateD3D12Resource(*device);
  std::array<int64_t, 4> shape = {720, 720, 3};

  void* dml_allocator_resource;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlCreateGPUAllocationFromD3DResource(resource.get(), &dml_allocator_resource), ort_api);

  OrtValue* ort_value;
  THROW_IF_NOT_OK_MSG(ort_api->CreateTensorWithDataAsOrtValue(
        dml_memory,
        dml_allocator_resource,
        static_cast<size_t>(resource->GetDesc().Width),
        shape.data(),
        3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &ort_value),
      ort_api);

  Microsoft::WRL::ComPtr<OnnxruntimeValue> out_value;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(&out_value, this, std::move(unique_value), UniqueOrtAllocator(nullptr, nullptr)));

  // Cache the allocator on the value so it destructs appropriately when the value is dropped
  Microsoft::WRL::ComPtr<DmlAllocatorWrapper> dml_allocator_resource_wrapper;
  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<DmlAllocatorWrapper>(&dml_allocator_resource_wrapper, std::move(unique_dml_allocator_resource)));

  RETURN_IF_FAILED(out_value->SetParameter(dml_allocator_resource_wrapper.Get()));

  *out = out_value.Detach();

//  winml_adapter_api->DmlFreeGPUAllocation(dml_allocator_resource);
  return UniqueOrtValue(ort_value, ort_api->ReleaseValue);
}

void DmlCopyTensor() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  THROW_IF_NOT_OK_MSG(ort_api->DisableMemPattern(session_options.get()), ort_api);

  winrt::com_ptr<ID3D12Device> device;
  WINML_EXPECT_NO_THROW(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.put())));

  winrt::com_ptr<ID3D12CommandQueue> queue;
  D3D12_COMMAND_QUEUE_DESC command_queue_desc = {};
  command_queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  WINML_EXPECT_HRESULT_SUCCEEDED(device->CreateCommandQueue(&command_queue_desc, IID_PPV_ARGS(queue.put())));

  THROW_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_DML(session_options.get(), device.get(), queue.get()), ort_api);
  const auto module_path = std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(FileHelpers::GetModulePath());
  auto session = CreateUniqueOrtSession(module_path + "fns-candy.onnx", session_options);


//  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);

  auto memory_info = GetCpuOutputMemoryInfo(ort_provider);
//  THROW_IF_NOT_OK_MSG(winml_adapter_api->GetProviderMemoryInfo(ort_provider, &memory_info), ort_api);
  auto gpu_tensor = CreateDmlTensor(ort_provider, device.get());

  OrtMemoryInfo* cpu_memory_info;
  THROW_IF_NOT_OK_MSG(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_memory_info), ort_api);
  auto dst_cpu_tensor = CreateTensorFromMemoryInfo(cpu_memory_info);

  // CPU to CPU is not supported
//  WINML_EXPECT_THROW_SPECIFIC(winml_adapter_api->DmlCopyTensor(ort_provider, cpu_tensor.get(), dst_cpu_tensor.get()), wil::ResultException, [](const wil::ResultException& e) { return e.GetFailureInfo().hr == E_INVALIDARG; });

  // GPU to CPU
  const char *name;
  THROW_IF_NOT_OK_MSG(ort_api->MemoryInfoGetName(memory_info.get(), &name), ort_api);
  std::cout << name << "\n";
  int id;
  THROW_IF_NOT_OK_MSG(ort_api->MemoryInfoGetId(memory_info.get(), &id), ort_api);
  std::cout << id << "\n";
  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlCopyTensor(ort_provider, gpu_tensor.get(), dst_cpu_tensor.get()), ort_api);

//  ORT_API2_STATUS(MemoryInfoGetMemType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtMemType* out);
//  ORT_API2_STATUS(MemoryInfoGetType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtAllocatorType* out);


  OrtMemoryInfo *dml_cpu_input_memory_info;
//  THROW_IF_NOT_OK_MSG(ort_api->CreateMemoryInfo("DML allocator", OrtDeviceAllocator, 0, OrtMemTypeCPUInput, &dml_cpu_input_memory_info), ort_api);
//  ORT_API2_STATUS(CreateMemoryInfo, _In_ const char* name1, enum OrtAllocatorType type, int id1,
//                  enum OrtMemType mem_type1, _Outptr_ OrtMemoryInfo** out);
}

void CreateCustomRegistry() {
  IMLOperatorRegistry* registry;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->CreateCustomRegistry(&registry), ort_api);
  WINML_EXPECT_NOT_EQUAL(nullptr, registry);
}

void ValueGetDeviceId() {
  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  OrtMemoryInfo *memory_info;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->GetProviderMemoryInfo(ort_provider, &memory_info), ort_api);
  auto gpu_tensor = CreateTensorFromMemoryInfo(memory_info);

  int16_t device_id;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->ValueGetDeviceId(gpu_tensor.get(), &device_id), ort_api);

  OrtMemoryInfo* cpu_memory_info;
  THROW_IF_NOT_OK_MSG(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_memory_info), ort_api);
  auto cpu_tensor = CreateTensorFromMemoryInfo(cpu_memory_info);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->ValueGetDeviceId(cpu_tensor.get(), &device_id), ort_api);
  WINML_EXPECT_EQUAL(0, device_id);
}

void SessionGetInputRequiredDeviceId() {
  auto session = CreateDmlSession();
  int16_t device_id;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetInputRequiredDeviceId(session.get(), "inputImage", &device_id), ort_api);

  auto cpu_session = CreateCpuSession();
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetInputRequiredDeviceId(cpu_session.get(), "inputImage", &device_id), ort_api);
  WINML_EXPECT_EQUAL(0, device_id);
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
