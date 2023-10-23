#include "dml_interop.h"
#include <algorithm>
#include <limits>
#include <set>
#include <type_traits>
#include <core/session/onnxruntime_cxx_api.h>
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/providers/tensorrt/tensorrt_provider_options.h"
#include "core/providers/dnnl/dnnl_provider_options.h"
#include "core/providers/dml/dml_provider_factory.h"
#include <assert.h>
#include "providers.h"


#include <core/session/onnxruntime_cxx_api.h>
#include "test_configuration.h"
#include "test_session.h"

#include <wrl/client.h>
#include <d3d12.h>

using UniqueNativePtr = std::unique_ptr<void, void (*)(void*)>;

size_t GetSizeFromType(ONNXTensorElementDataType type) {
#define CASE_FOR_TYPE(T)                         \
  case Ort::TypeToTensorType<T>::type: {         \
    return sizeof(T);                            \
  }

  switch (type) {
    CASE_FOR_TYPE(Ort::Float16_t);
    CASE_FOR_TYPE(Ort::BFloat16_t);
    CASE_FOR_TYPE(float);
    CASE_FOR_TYPE(double);
    CASE_FOR_TYPE(int8_t);
    CASE_FOR_TYPE(int16_t);
    CASE_FOR_TYPE(int32_t);
    CASE_FOR_TYPE(int64_t);
    CASE_FOR_TYPE(uint8_t);
    CASE_FOR_TYPE(uint16_t);
    CASE_FOR_TYPE(uint32_t);
    CASE_FOR_TYPE(uint64_t);
    CASE_FOR_TYPE(bool);
#if !defined(DISABLE_FLOAT8_TYPES)
    CASE_FOR_TYPE(Ort::Float8E4M3FN_t);
    CASE_FOR_TYPE(Ort::Float8E4M3FNUZ_t);
    CASE_FOR_TYPE(Ort::Float8E5M2_t);
    CASE_FOR_TYPE(Ort::Float8E5M2FNUZ_t);
#endif
    default:
      ORT_THROW("Unsupported tensor data type: ", type);
  }
#undef CASE_FOR_TYPE
}

Microsoft::WRL::ComPtr<ID3D12Resource> CreateD3D12Resource(
  ID3D12Device* device,
  ONNXTensorElementDataType type,
  const std::vector<int64_t>& shape,
  D3D12_HEAP_TYPE heap_type) {
  // Try to allocate the backing memory for the caller
  auto bufferSize =
    std::accumulate(
        std::begin(shape),
        std::end(shape),
        static_cast<int64_t>(1),
        std::multiplies<int64_t>());

  auto bufferByteSize = GetSizeFromType(type) * bufferSize;

  // DML needs the resources' sizes to be a multiple of 4 bytes
  if (bufferByteSize % 4 != 0) {
    bufferByteSize += 4 - (bufferByteSize % 4);
  }

  auto resource_flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  if (heap_type == D3D12_HEAP_TYPE_UPLOAD ||
      heap_type == D3D12_HEAP_TYPE_READBACK) {
    resource_flags = D3D12_RESOURCE_FLAG_NONE;
  }

  D3D12_HEAP_PROPERTIES heapProperties = {
    heap_type, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0};
  D3D12_RESOURCE_DESC resourceDesc = {
    D3D12_RESOURCE_DIMENSION_BUFFER,
    0,
    static_cast<uint64_t>(bufferByteSize),
    1,
    1,
    1,
    DXGI_FORMAT_UNKNOWN,
    {1, 0},
    D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
    resource_flags
  };

  Microsoft::WRL::ComPtr<ID3D12Resource> resource;
  device->CreateCommittedResource(
    &heapProperties,
    D3D12_HEAP_FLAG_NONE,
    &resourceDesc,
    D3D12_RESOURCE_STATE_COMMON,
    nullptr,
    __uuidof(ID3D12Resource),
    &resource);

  return resource;
}

static void InitializeDmlValueFromCpuValue(
    const Ort::Session& session,
    const Ort::Value& cpu_value,
    const char* name,
    Ort::Value& dml_value) {
  auto& ort_api = Ort::GetApi();
  const OrtDmlApi* ort_dml_api;
  Ort::ThrowOnError(ort_api.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ort_dml_api)));
  Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue = nullptr;
  Ort::ThrowOnError(ort_dml_api->GetCommandQueueForSessionInput(session, name, &queue));
  Microsoft::WRL::ComPtr<ID3D12Device> device = nullptr;
  queue->GetDevice(IID_PPV_ARGS(&device));

  // Get cpu data
  const void* cpu_mutable_data = cpu_value.GetTensorRawData();
  const auto type_and_shape_info = cpu_value.GetTensorTypeAndShapeInfo();
  auto cpu_buffer_size_in_bytes = GetSizeFromType(type_and_shape_info.GetElementType()) * type_and_shape_info.GetElementCount();
  auto node_dim = type_and_shape_info.GetShape();

  // Copy to upload resource
  auto upload_resource = CreateD3D12Resource(device.Get(), type_and_shape_info.GetElementType(), node_dim, D3D12_HEAP_TYPE_UPLOAD);
  void* mapped_dml_data;
  (void)(upload_resource->Map(0, nullptr, &mapped_dml_data));
  memcpy(mapped_dml_data, cpu_mutable_data, cpu_buffer_size_in_bytes);
  upload_resource->Unmap(0, nullptr);

  // Get dml resource
  void* mutable_data = dml_value.GetTensorMutableRawData();
  auto ort_memory_info = dml_value.GetTensorMemoryInfo();
  auto ort_allocator = Ort::Allocator(session, ort_memory_info);
  Microsoft::WRL::ComPtr<ID3D12Resource> dml_resource;
  Ort::ThrowOnError(ort_dml_api->GetD3D12ResourceFromAllocation(ort_allocator, mutable_data, &dml_resource));

  Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> command_list;
  Microsoft::WRL::ComPtr<ID3D12CommandAllocator> command_allocator;

  device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&command_allocator));
  device->CreateCommandList(
      0,
      D3D12_COMMAND_LIST_TYPE_DIRECT,
      command_allocator.Get(),
      nullptr,
      IID_PPV_ARGS(&command_list)
    );

  Microsoft::WRL::ComPtr<ID3D12Fence> fence;
  device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));


  D3D12_RESOURCE_BARRIER barrier = {};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
  barrier.Transition.pResource = dml_resource.Get();
  barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;

  command_list->ResourceBarrier(1, &barrier);
  command_list->CopyBufferRegion(dml_resource.Get(), 0, upload_resource.Get(), 0, cpu_buffer_size_in_bytes);

  barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
  barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;

  command_list->ResourceBarrier(1, &barrier);
  command_list->Close();

  ID3D12CommandList* ppCommandLists[] = {command_list.Get()};
  queue->ExecuteCommandLists(_countof(ppCommandLists), ppCommandLists);
  queue->Signal(fence.Get(), 1);

  // Wait until the fence is completed.
  auto fence_event = CreateEventEx(NULL, false, false, EVENT_ALL_ACCESS);
  fence->SetEventOnCompletion(1, fence_event);
  WaitForSingleObject(fence_event, INFINITE);
}

std::pair<Ort::Value, UniqueNativePtr> CreateDmlValue(
    const Ort::ConstTensorTypeAndShapeInfo& tensor_info,
    const Ort::Session& session,
    Ort::Value&& default_value,
    const char* name,
    bool is_input) {

  auto& ort_api = Ort::GetApi();
  const OrtDmlApi* ort_dml_api;
  Ort::ThrowOnError(ort_api.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ort_dml_api)));
  Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue = nullptr;
  if (is_input) {
    Ort::ThrowOnError(ort_dml_api->GetCommandQueueForSessionInput(session, name, &queue));
  } else {
    Ort::ThrowOnError(ort_dml_api->GetCommandQueueForSessionOutput(session, name, &queue));
  }
  Microsoft::WRL::ComPtr<ID3D12Device> device = nullptr;
  queue->GetDevice(IID_PPV_ARGS(&device));

  if (!device) {
    return { std::move(default_value), UniqueNativePtr(nullptr, nullptr) };
  }

  auto node_dim = tensor_info.GetShape();
  auto d3d_resource = CreateD3D12Resource(device.Get(), tensor_info.GetElementType(), node_dim, D3D12_HEAP_TYPE_DEFAULT);
  void* dml_allocator_resource;
  Ort::ThrowOnError(ort_dml_api->CreateGPUAllocationFromD3DResource(d3d_resource.Get(), &dml_allocator_resource));

  auto unique_dml_allocator_resource = UniqueNativePtr(dml_allocator_resource, [](void* ptr) {
    auto& ort_api = Ort::GetApi();
    const OrtDmlApi* ort_dml_api;
    Ort::ThrowOnError(ort_api.GetExecutionProviderApi("DML", ORT_API_VERSION, reinterpret_cast<const void**>(&ort_dml_api)));
    Ort::ThrowOnError(ort_dml_api->FreeGPUAllocation(ptr));
  });

  auto dml_memory_info = Ort::MemoryInfo("DML", OrtAllocatorType::OrtDeviceAllocator, 0, OrtMemType::OrtMemTypeDefault);

  // create the OrtValue as a tensor letting ort know that we own the data buffer
  OrtValue* dml_value_ptr;
  Ort::ThrowOnError(ort_api.CreateTensorWithDataAsOrtValue(
      dml_memory_info,
      unique_dml_allocator_resource.get(),
      static_cast<size_t>(d3d_resource->GetDesc().Width),
      node_dim.data(),
      node_dim.size(),
      tensor_info.GetElementType(),
      &dml_value_ptr));
  Ort::Value dml_value(dml_value_ptr);

  return {std::move(dml_value), std::move(unique_dml_allocator_resource)};
}

std::pair<Ort::Value, UniqueNativePtr> CreateDmlValueFromCpuValue(
    Ort::Value&& cpu_value,
    const Ort::Session& session,
    const char* input_name) {
  auto type_info = cpu_value.GetTypeInfo();
  auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
  auto dml_value_pair = CreateDmlValue(tensor_info, session, std::move(cpu_value), input_name, true);
  InitializeDmlValueFromCpuValue(session, cpu_value, input_name, dml_value_pair.first);
  return dml_value_pair;
}
