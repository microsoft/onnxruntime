// // Copyright (c) Microsoft Corporation. All rights reserved.
// // Licensed under the MIT License.

#include "testPch.h"

#include "AdapterDmlEpTest.h"
#include <wil/result.h>

#include "common.h"
#include "iengine.h"
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
winrt::com_ptr<_winml::IEngine> engine;

void AdapterDmlEpTestSetup() {
  GPUTEST;
  winrt::init_apartment();
  WINML_EXPECT_HRESULT_SUCCEEDED(Microsoft::WRL::MakeAndInitialize<_winml::OnnxruntimeEngineFactory>(engine_factory.put()));
  WINML_EXPECT_HRESULT_SUCCEEDED(engine_factory->GetOrtEnvironment(&ort_env));
  winrt::com_ptr<_winml::IEngineBuilder> engine_builder;
  engine_factory->CreateEngineBuilder(engine_builder.put());
  engine_builder->CreateEngine(engine.put());

  ort_api = OrtGetApiBase()->GetApi(2);
  winml_adapter_api = OrtGetWinMLAdapter(ort_api);
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

UniqueOrtSession CreateUniqueOrtSession(const std::wstring& model_path, const UniqueOrtSessionOptions& session_options) {
  THROW_IF_NOT_OK_MSG(ort_api->SetIntraOpNumThreads(session_options.get(), 1), ort_api);
  THROW_IF_NOT_OK_MSG(ort_api->SetSessionGraphOptimizationLevel(session_options.get(), ORT_ENABLE_BASIC), ort_api);
  OrtSession *session;
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
  WINML_EXPECT_HRESULT_SUCCEEDED(device->CreateCommandQueue(&command_queue_desc, IID_PPV_ARGS(queue.put())));

  THROW_IF_NOT_OK_MSG(winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_DML(session_options.get(), device.get(), queue.get()), ort_api);
  return CreateUniqueOrtSession(FileHelpers::GetModulePath() + L"fns-candy.onnx", session_options);
}

UniqueOrtSession CreateCpuSession() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  return CreateUniqueOrtSession(FileHelpers::GetModulePath() + L"fns-candy.onnx", session_options);
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
  auto unique_memory_info = UniqueOrtMemoryInfo(memory_info, ort_api->ReleaseMemoryInfo);
  // Ensure tensor can be created with the provided OrtMemoryInfo
  CreateTensorFromMemoryInfo(unique_memory_info.get());
}

void GetAndFreeProviderAllocator() {
  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  OrtAllocator *allocator;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->GetProviderAllocator(ort_provider, &allocator), ort_api);
  auto unique_allocator = UniqueOrtAllocator(allocator, winml_adapter_api->FreeProviderAllocator);

  // Ensure allocation works
  void *data = nullptr;
  THROW_IF_NOT_OK_MSG(ort_api->AllocatorAlloc(unique_allocator.get(), 1024, &data), ort_api);
  WINML_EXPECT_NOT_EQUAL(nullptr, data);
  THROW_IF_NOT_OK_MSG(ort_api->AllocatorFree(unique_allocator.get(), data), ort_api);
}

void GetValueMemoryInfo() {
  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);

  OrtMemoryInfo *memory_info;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->GetProviderMemoryInfo(ort_provider, &memory_info), ort_api);
  auto unique_memory_info = UniqueOrtMemoryInfo(memory_info, ort_api->ReleaseMemoryInfo);
  auto tensor = CreateTensorFromMemoryInfo(unique_memory_info.get());

  OrtMemoryInfo *value_memory_info;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->GetValueMemoryInfo(tensor.get(), &value_memory_info), ort_api);
  auto unique_value_memory_info = UniqueOrtMemoryInfo(value_memory_info, ort_api->ReleaseMemoryInfo);
  CreateTensorFromMemoryInfo(unique_value_memory_info.get());
}

void ExecutionProviderSync() {
  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->ExecutionProviderSync(ort_provider), ort_api);
}

//UniqueOrtMemoryInfo GetCpuOutputMemoryInfo(OrtExecutionProvider* provider) {
//  const auto execution_provider = reinterpret_cast<onnxruntime::IExecutionProvider*>(provider);
//  auto allocator = execution_provider->GetAllocator(0, OrtMemTypeCPUOutput);
//  const auto& info = allocator->Info();
//  return UniqueOrtMemoryInfo(new OrtMemoryInfo(info.name, info.alloc_type, info.device, info.id, info.mem_type), ort_api->ReleaseMemoryInfo);
//}
using DmlAllocatorResource = std::unique_ptr<void, void (*)(void*)>;
UniqueOrtValue CreateDmlTensor(OrtMemoryInfo* dml_memory, void* dml_allocator_resource, ID3D12Device* device) {
//  UniqueOrtValue CreateDmlTensor(OrtExecutionProvider* execution_provider, ID3D12Device* device) {
//  OnnxruntimeEngine::CreateTensorValueFromExternalD3DResource(ID3D12Resource* d3d_resource, const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out)
//  auto memory_info = GetCpuOutputMemoryInfo(execution_provider);
  std::array<int64_t, 3> shape = {720, 720, 3};

//  void* dml_allocator_resource;
//  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlCreateGPUAllocationFromD3DResource(resource.get(), &dml_allocator_resource), ort_api);
//
//  OrtValue* ort_value;
//  THROW_IF_NOT_OK_MSG(ort_api->CreateTensorWithDataAsOrtValue(
//        dml_memory,
//        dml_allocator_resource,
//        static_cast<size_t>(resource->GetDesc().Width),
//        shape.data(),
//        3,
//        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
//        &ort_value),
//      ort_api);




//  winrt::com_ptr<_winml::IValue> value;
//  engine->CreateTensorValueFromExternalD3DResource(resource.get(), shape.data(), 3, winml::TensorKind::Float, value.put());
//  OrtValue *ort_value = static_cast<_winml::OnnxruntimeValue*>(value.detach())->UseOrtValue();


  // create the OrtValue as a tensor letting ort know that we own the data buffer
  OrtValue* ort_value;
  THROW_IF_NOT_OK_MSG(ort_api->CreateTensorWithDataAsOrtValue(
      dml_memory,
      dml_allocator_resource,
//      static_cast<size_t>(resource->GetDesc().Width),
      720 * 720 * 3 * 4,
      shape.data(),
      shape.size(),
      ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      &ort_value),
    ort_api);
  auto unique_value = UniqueOrtValue(ort_value, ort_api->ReleaseValue);

  Microsoft::WRL::ComPtr<_winml::OnnxruntimeValue> out_value;
  WINML_EXPECT_HRESULT_SUCCEEDED(Microsoft::WRL::MakeAndInitialize<_winml::OnnxruntimeValue>(&out_value, static_cast<_winml::OnnxruntimeEngine*>(engine.get()), std::move(unique_value), UniqueOrtAllocator(nullptr, nullptr)));
  out_value.Detach();  // FIXME leak


//  (ID3D12Resource* resource, const int64_t* shape, size_t count, winml::TensorKind kind, _Out_ IValue** out) override;
//   return UniqueOrtValue(ort_value, ort_api->ReleaseValue);

//  Microsoft::WRL::ComPtr<OnnxruntimeValue> out_value;
//  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<OnnxruntimeValue>(&out_value, this, std::move(unique_value), UniqueOrtAllocator(nullptr, nullptr)));
//
//  // Cache the allocator on the value so it destructs appropriately when the value is dropped
//  Microsoft::WRL::ComPtr<DmlAllocatorWrapper> dml_allocator_resource_wrapper;
//  RETURN_IF_FAILED(Microsoft::WRL::MakeAndInitialize<DmlAllocatorWrapper>(&dml_allocator_resource_wrapper, std::move(unique_dml_allocator_resource)));
//
//  RETURN_IF_FAILED(out_value->SetParameter(dml_allocator_resource_wrapper.Get()));
//
//  *out = out_value.Detach();
//
////  winml_adapter_api->DmlFreeGPUAllocation(dml_allocator_resource);
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
  auto session = CreateUniqueOrtSession(FileHelpers::GetModulePath() + L"fns-candy.onnx", session_options);


//  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);

//  auto memory_info = GetCpuOutputMemoryInfo(ort_provider);
//  THROW_IF_NOT_OK_MSG(winml_adapter_api->GetProviderMemory  Info(ort_provider, &memory_info), ort_api);
  // TODO free these
  OrtMemoryInfo* dml_memory;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->GetProviderMemoryInfo(ort_provider, &dml_memory), ort_api);
  auto resource = CreateD3D12Resource(*device);
  void* dml_allocator_resource;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlCreateGPUAllocationFromD3DResource(resource.get(), &dml_allocator_resource), ort_api);
  auto gpu_tensor = CreateDmlTensor(dml_memory, dml_allocator_resource, device.get());

  OrtMemoryInfo* cpu_memory_info;
  THROW_IF_NOT_OK_MSG(ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_memory_info), ort_api);
  auto dst_cpu_tensor = CreateTensorFromMemoryInfo(cpu_memory_info);

  // CPU to CPU is not supported
//  WINML_EXPECT_THROW_SPECIFIC(winml_adapter_api->DmlCopyTensor(ort_provider, cpu_tensor.get(), dst_cpu_tensor.get()), wil::ResultException, [](const wil::ResultException& e) { return e.GetFailureInfo().hr == E_INVALIDARG; });

  // GPU to CPU
  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlCopyTensor(ort_provider, gpu_tensor.get(), dst_cpu_tensor.get()), ort_api);

//  ORT_API2_STATUS(MemoryInfoGetMemType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtMemType* out);
//  ORT_API2_STATUS(MemoryInfoGetType, _In_ const OrtMemoryInfo* ptr, _Out_ OrtAllocatorType* out);

  winml_adapter_api->DmlFreeGPUAllocation(dml_allocator_resource);
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
