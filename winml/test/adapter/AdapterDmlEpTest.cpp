// // Copyright (c) Microsoft Corporation. All rights reserved.
// // Licensed under the MIT License.

#include "testPch.h"

#include "AdapterDmlEpTest.h"

#include "common.h"
#include "iengine.h"
#include "winml_adapter_c_api.h"
#include "core/framework/execution_provider.h"
#include "core/providers/winml/winml_provider_factory.h"
#include "core/providers/dml/dml_provider_factory.h"
#include "OnnxruntimeEngine.h"
#include "OnnxruntimeErrors.h"
#include "OnnxruntimeModel.h"
#include "UniqueOrtPtr.h"

namespace {
const WinmlAdapterApi* winml_adapter_api;
const OrtDmlApi* ort_dml_api;
const OrtApi* ort_api;
OrtEnv* ort_env;

void AdapterDmlEpTestSetup() {
  GPUTEST;
  winrt::init_apartment();
  ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  THROW_IF_NOT_OK_MSG(
    ort_api->GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ort_dml_api)), ort_api
  );
  winml_adapter_api = OrtGetWinMLAdapter(ORT_API_VERSION);
  THROW_IF_NOT_OK_MSG(ort_api->CreateEnv(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "Default", &ort_env), ort_api);
#ifdef BUILD_INBOX
  winrt_activation_handler = WINRT_RoGetActivationFactory;
#endif
}

void AdapterDmlEpTestTeardown() {
  GPUTEST;
  ort_api->ReleaseEnv(ort_env);
}

UniqueOrtSessionOptions CreateUniqueOrtSessionOptions() {
  OrtSessionOptions* options;
  THROW_IF_NOT_OK_MSG(ort_api->CreateSessionOptions(&options), ort_api);
  return UniqueOrtSessionOptions(options, ort_api->ReleaseSessionOptions);
}

UniqueOrtSession CreateUniqueOrtSession(const UniqueOrtSessionOptions& session_options) {
  OrtSession* session;
  THROW_IF_NOT_OK_MSG(
    winml_adapter_api->CreateSessionWithoutModel(ort_env, session_options.get(), nullptr, nullptr, &session), ort_api
  );
  return UniqueOrtSession(session, ort_api->ReleaseSession);
}

UniqueOrtSession CreateUniqueOrtSession(
  const std::wstring& model_path, const UniqueOrtSessionOptions& session_options
) {
  THROW_IF_NOT_OK_MSG(ort_api->SetIntraOpNumThreads(session_options.get(), 1), ort_api);
  THROW_IF_NOT_OK_MSG(ort_api->SetSessionGraphOptimizationLevel(session_options.get(), ORT_ENABLE_BASIC), ort_api);
  OrtSession* session;
  THROW_IF_NOT_OK_MSG(ort_api->CreateSession(ort_env, model_path.c_str(), session_options.get(), &session), ort_api);
  return UniqueOrtSession(session, ort_api->ReleaseSession);
}

UniqueOrtSession CreateDmlSession() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  THROW_IF_NOT_OK_MSG(ort_api->DisableMemPattern(session_options.get()), ort_api);

  winrt::com_ptr<ID3D12Device> device;
  WINML_EXPECT_NO_THROW(
    D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.put()))
  );

  winrt::com_ptr<ID3D12CommandQueue> queue;
  D3D12_COMMAND_QUEUE_DESC command_queue_desc = {};
  command_queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  WINML_EXPECT_HRESULT_SUCCEEDED(device->CreateCommandQueue(&command_queue_desc, IID_PPV_ARGS(queue.put())));

  THROW_IF_NOT_OK_MSG(
    winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_DML(
      session_options.get(), device.get(), queue.get(), false
    ),
    ort_api
  );
  return CreateUniqueOrtSession(FileHelpers::GetModulePath() + L"fns-candy.onnx", session_options);
}

UniqueOrtSession CreateCpuSession() {
  const auto session_options = CreateUniqueOrtSessionOptions();
  return CreateUniqueOrtSession(FileHelpers::GetModulePath() + L"fns-candy.onnx", session_options);
}

void DmlExecutionProviderFlushContext() {
  GPUTEST;
  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlExecutionProviderFlushContext(ort_provider), ort_api);
}

void DmlExecutionProviderReleaseCompletedReferences() {
  GPUTEST;
  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlExecutionProviderReleaseCompletedReferences(ort_provider), ort_api);
}

constexpr std::array<int64_t, 4> dimensions{1, 3, 720, 720};
constexpr uint64_t tensor_size = 3 * 720 * 720;
std::array<float, tensor_size> tensor_values = {};

winrt::com_ptr<ID3D12Resource> CreateD3D12Resource(ID3D12Device& device) {
  constexpr uint64_t buffer_size = tensor_size * sizeof(float);
  constexpr D3D12_HEAP_PROPERTIES heap_properties = {
    D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0};
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
  WINML_EXPECT_HRESULT_SUCCEEDED(device.CreateCommittedResource(
    &heap_properties,
    D3D12_HEAP_FLAG_NONE,
    &resource_desc,
    D3D12_RESOURCE_STATE_COMMON,
    nullptr,
    IID_PPV_ARGS(d3d12_resource.put())
  ));
  return d3d12_resource;
}

void DmlCreateAndFreeGPUAllocationFromD3DResource() {
  GPUTEST;
  winrt::com_ptr<ID3D12Device> device;
  WINML_EXPECT_NO_THROW(
    D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.put()))
  );

  auto d3d12_resource = CreateD3D12Resource(*device);
  void* dml_allocator_resource;
  THROW_IF_NOT_OK_MSG(
    ort_dml_api->CreateGPUAllocationFromD3DResource(d3d12_resource.get(), &dml_allocator_resource), ort_api
  );
  THROW_IF_NOT_OK_MSG(ort_dml_api->FreeGPUAllocation(dml_allocator_resource), ort_api);
}

void DmlGetD3D12ResourceFromAllocation() {
  GPUTEST;
  winrt::com_ptr<ID3D12Device> device;
  WINML_EXPECT_NO_THROW(
    D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.put()))
  );

  auto d3d12_resource = CreateD3D12Resource(*device);
  void* gpu_allocation;
  THROW_IF_NOT_OK_MSG(ort_dml_api->CreateGPUAllocationFromD3DResource(d3d12_resource.get(), &gpu_allocation), ort_api);

  auto session = CreateDmlSession();

  OrtMemoryInfo* ort_memory_info;
  THROW_IF_NOT_OK_MSG(
    ort_api->CreateMemoryInfo(
      "DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault, &ort_memory_info
    ),
    ort_api
  );

  OrtAllocator* ort_allocator;
  THROW_IF_NOT_OK_MSG(ort_api->CreateAllocator(session.get(), ort_memory_info, &ort_allocator), ort_api);
  auto allocator = UniqueOrtAllocator(ort_allocator, ort_api->ReleaseAllocator);

  winrt::com_ptr<ID3D12Resource> d3d12_resource_from_allocation;
  THROW_IF_NOT_OK_MSG(
    ort_dml_api->GetD3D12ResourceFromAllocation(allocator.get(), gpu_allocation, d3d12_resource_from_allocation.put()),
    ort_api
  );
  // Ensure resource is the same
  WINML_EXPECT_EQUAL(d3d12_resource, d3d12_resource_from_allocation);

  THROW_IF_NOT_OK_MSG(ort_dml_api->FreeGPUAllocation(gpu_allocation), ort_api);
}

UniqueOrtValue CreateTensorFromMemoryInfo(const OrtMemoryInfo* memory_info) {
  OrtValue* tensor;
  THROW_IF_NOT_OK_MSG(
    ort_api->CreateTensorWithDataAsOrtValue(
      memory_info,
      tensor_values.data(),
      tensor_size * sizeof(float),
      dimensions.data(),
      dimensions.size(),
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      &tensor
    ),
    ort_api
  );
  return UniqueOrtValue(tensor, ort_api->ReleaseValue);
}

void GetTensorMemoryInfo() {
  GPUTEST;
  auto session = CreateDmlSession();

  OrtMemoryInfo* ort_memory_info;
  THROW_IF_NOT_OK_MSG(
    ort_api->CreateMemoryInfo(
      "DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault, &ort_memory_info
    ),
    ort_api
  );
  auto tensor = CreateTensorFromMemoryInfo(ort_memory_info);

  const OrtMemoryInfo* value_memory_info;
  THROW_IF_NOT_OK_MSG(ort_api->GetTensorMemoryInfo(tensor.get(), &value_memory_info), ort_api);
  CreateTensorFromMemoryInfo(value_memory_info);
}

void ExecutionProviderSync() {
  GPUTEST;
  auto session = CreateDmlSession();
  OrtExecutionProvider* ort_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &ort_provider), ort_api);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->ExecutionProviderSync(ort_provider), ort_api);
}

void DmlCopyTensor() {
  GPUTEST;
  const auto session_options = CreateUniqueOrtSessionOptions();
  THROW_IF_NOT_OK_MSG(ort_api->DisableMemPattern(session_options.get()), ort_api);

  winrt::com_ptr<ID3D12Device> device;
  WINML_EXPECT_NO_THROW(
    D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(device.put()))
  );

  winrt::com_ptr<ID3D12CommandQueue> queue;
  D3D12_COMMAND_QUEUE_DESC command_queue_desc = {};
  command_queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  WINML_EXPECT_HRESULT_SUCCEEDED(device->CreateCommandQueue(&command_queue_desc, IID_PPV_ARGS(queue.put())));

  THROW_IF_NOT_OK_MSG(
    winml_adapter_api->OrtSessionOptionsAppendExecutionProvider_DML(
      session_options.get(), device.get(), queue.get(), false
    ),
    ort_api
  );
  auto session = CreateUniqueOrtSession(FileHelpers::GetModulePath() + L"fns-candy.onnx", session_options);

  OrtExecutionProvider* dml_provider;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->SessionGetExecutionProvider(session.get(), 0, &dml_provider), ort_api);

  // CPU to CPU is not supported
  OrtMemoryInfo* cpu_memory_info;
  THROW_IF_NOT_OK_MSG(ort_api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &cpu_memory_info), ort_api);
  auto cpu_tensor = CreateTensorFromMemoryInfo(cpu_memory_info);
  auto dst_cpu_tensor = CreateTensorFromMemoryInfo(cpu_memory_info);
  WINML_EXPECT_NOT_EQUAL(
    nullptr, winml_adapter_api->DmlCopyTensor(dml_provider, cpu_tensor.get(), dst_cpu_tensor.get())
  );

  // GPU to CPU
  OrtMemoryInfo* ort_memory_info;
  THROW_IF_NOT_OK_MSG(
    ort_api->CreateMemoryInfo(
      "DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault, &ort_memory_info
    ),
    ort_api
  );

  auto resource = CreateD3D12Resource(*device);
  void* dml_allocator_resource;
  THROW_IF_NOT_OK_MSG(
    ort_dml_api->CreateGPUAllocationFromD3DResource(resource.get(), &dml_allocator_resource), ort_api
  );

  std::array<int64_t, 3> shape = {720, 720, 3};
  OrtValue* gpu_value;
  THROW_IF_NOT_OK_MSG(
    ort_api->CreateTensorWithDataAsOrtValue(
      ort_memory_info,
      dml_allocator_resource,
      static_cast<size_t>(resource->GetDesc().Width),
      shape.data(),
      shape.size(),
      ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      &gpu_value
    ),
    ort_api
  );
  dst_cpu_tensor = CreateTensorFromMemoryInfo(cpu_memory_info);
  THROW_IF_NOT_OK_MSG(winml_adapter_api->DmlCopyTensor(dml_provider, gpu_value, dst_cpu_tensor.get()), ort_api);

  THROW_IF_NOT_OK_MSG(ort_dml_api->FreeGPUAllocation(dml_allocator_resource), ort_api);
}

void CreateCustomRegistry() {
  GPUTEST;
  IMLOperatorRegistry* registry;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->CreateCustomRegistry(&registry), ort_api);
  WINML_EXPECT_NOT_EQUAL(nullptr, registry);
}

void ValueGetDeviceId() {
  GPUTEST;
  auto session = CreateDmlSession();

  OrtMemoryInfo* ort_memory_info;
  THROW_IF_NOT_OK_MSG(
    ort_api->CreateMemoryInfo(
      "DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault, &ort_memory_info
    ),
    ort_api
  );
  auto gpu_tensor = CreateTensorFromMemoryInfo(ort_memory_info);

  int16_t device_id;
  THROW_IF_NOT_OK_MSG(winml_adapter_api->ValueGetDeviceId(gpu_tensor.get(), &device_id), ort_api);

  OrtMemoryInfo* cpu_memory_info;
  THROW_IF_NOT_OK_MSG(ort_api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &cpu_memory_info), ort_api);
  auto unique_cpu_memory_info = UniqueOrtMemoryInfo(cpu_memory_info, ort_api->ReleaseMemoryInfo);
  auto cpu_tensor = CreateTensorFromMemoryInfo(unique_cpu_memory_info.get());
  THROW_IF_NOT_OK_MSG(winml_adapter_api->ValueGetDeviceId(cpu_tensor.get(), &device_id), ort_api);
  WINML_EXPECT_EQUAL(0, device_id);
}

void SessionGetInputRequiredDeviceId() {
  GPUTEST;
  auto session = CreateDmlSession();
  int16_t device_id;
  THROW_IF_NOT_OK_MSG(
    winml_adapter_api->SessionGetInputRequiredDeviceId(session.get(), "inputImage", &device_id), ort_api
  );

  auto cpu_session = CreateCpuSession();
  THROW_IF_NOT_OK_MSG(
    winml_adapter_api->SessionGetInputRequiredDeviceId(cpu_session.get(), "inputImage", &device_id), ort_api
  );
  WINML_EXPECT_EQUAL(0, device_id);
}
}// namespace

const AdapterDmlEpTestApi& getapi() {
  static constexpr AdapterDmlEpTestApi api = {
    AdapterDmlEpTestSetup,
    AdapterDmlEpTestTeardown,
    DmlExecutionProviderFlushContext,
    DmlExecutionProviderReleaseCompletedReferences,
    DmlCreateAndFreeGPUAllocationFromD3DResource,
    DmlGetD3D12ResourceFromAllocation,
    GetTensorMemoryInfo,
    ExecutionProviderSync,
    DmlCopyTensor,
    CreateCustomRegistry,
    ValueGetDeviceId,
    SessionGetInputRequiredDeviceId};
  return api;
}
