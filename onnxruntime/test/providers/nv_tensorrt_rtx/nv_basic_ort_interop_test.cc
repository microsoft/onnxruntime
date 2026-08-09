// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "core/graph/model.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/util/include/asserts.h"
#include "test/common/trt_op_test_utils.h"
#include "test/common/random_generator.h"
#include "test/providers/nv_tensorrt_rtx/test_nv_trt_rtx_ep_util.h"
#include "test/unittest_util/conversion.h"

#include <cstring>
#include <gtest/gtest.h>
#include <onnxruntime_cxx_api.h>

#if defined(USE_DX_INTEROP) && USE_DX_INTEROP && defined(_WIN32)
#include <Windows.h>
#include <d3d12.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;
#endif

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {

#if defined(USE_DX_INTEROP) && USE_DX_INTEROP && defined(_WIN32)
namespace {

// Load d3d12.dll at runtime and get D3D12CreateDevice so the test exe does not need to link d3d12.lib.
// Caller must keep .module loaded for the duration of D3D12 API use (do not FreeLibrary).
struct D3D12CreateDeviceLoadResult {
  HMODULE module = nullptr;
  typedef HRESULT(WINAPI* PFN_D3D12CreateDevice)(IUnknown* pAdapter, D3D_FEATURE_LEVEL MinimumFeatureLevel,
                                                 REFIID riid, void** ppDevice);
  PFN_D3D12CreateDevice pfn = nullptr;
};
inline D3D12CreateDeviceLoadResult LoadD3D12CreateDevice() {
  D3D12CreateDeviceLoadResult r;
  r.module = LoadLibraryW(L"d3d12.dll");
  if (r.module) {
    r.pfn = reinterpret_cast<D3D12CreateDeviceLoadResult::PFN_D3D12CreateDevice>(
        GetProcAddress(r.module, "D3D12CreateDevice"));
  }
  return r;
}

void CreateD3D12Buffer(ID3D12Device* pDevice, size_t size, ID3D12Resource** ppResource,
                       D3D12_RESOURCE_STATES initState) {
  D3D12_RESOURCE_DESC bufferDesc = {};
  bufferDesc.MipLevels = 1;
  bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
  bufferDesc.Width = size;
  bufferDesc.Height = 1;
  bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  bufferDesc.DepthOrArraySize = 1;
  bufferDesc.SampleDesc.Count = 1;
  bufferDesc.SampleDesc.Quality = 0;
  bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

  D3D12_HEAP_PROPERTIES heapProps = {};
  heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;
  heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  heapProps.CreationNodeMask = 1;
  heapProps.VisibleNodeMask = 1;

  HRESULT hr = pDevice->CreateCommittedResource(
      &heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc, initState, nullptr, IID_PPV_ARGS(ppResource));
  if (FAILED(hr)) {
    GTEST_FAIL() << "Failed creating D3D12 resource, HRESULT: 0x" << std::hex << hr;
  }
}

void CreateUploadBuffer(ID3D12Device* pDevice, size_t size, ID3D12Resource** ppResource) {
  D3D12_HEAP_PROPERTIES heapProps = {};
  heapProps.Type = D3D12_HEAP_TYPE_UPLOAD;
  heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  heapProps.CreationNodeMask = 1;
  heapProps.VisibleNodeMask = 1;

  D3D12_RESOURCE_DESC bufferDesc = {};
  bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  bufferDesc.Alignment = 0;
  bufferDesc.Width = size;
  bufferDesc.Height = 1;
  bufferDesc.DepthOrArraySize = 1;
  bufferDesc.MipLevels = 1;
  bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
  bufferDesc.SampleDesc.Count = 1;
  bufferDesc.SampleDesc.Quality = 0;
  bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

  HRESULT hr = pDevice->CreateCommittedResource(
      &heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
      D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(ppResource));
  if (FAILED(hr)) {
    GTEST_FAIL() << "Failed creating D3D12 upload resource, HRESULT: 0x" << std::hex << hr;
  }
}

void CreateReadBackBuffer(ID3D12Device* pDevice, size_t size, ID3D12Resource** ppResource) {
  D3D12_HEAP_PROPERTIES heapProps = {};
  heapProps.Type = D3D12_HEAP_TYPE_READBACK;
  heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
  heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
  heapProps.CreationNodeMask = 1;
  heapProps.VisibleNodeMask = 1;

  D3D12_RESOURCE_DESC bufferDesc = {};
  bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  bufferDesc.Alignment = 0;
  bufferDesc.Width = size;
  bufferDesc.Height = 1;
  bufferDesc.DepthOrArraySize = 1;
  bufferDesc.MipLevels = 1;
  bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
  bufferDesc.SampleDesc.Count = 1;
  bufferDesc.SampleDesc.Quality = 0;
  bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  bufferDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

  HRESULT hr = pDevice->CreateCommittedResource(
      &heapProps, D3D12_HEAP_FLAG_NONE, &bufferDesc,
      D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(ppResource));
  if (FAILED(hr)) {
    GTEST_FAIL() << "Failed creating D3D12 readback resource, HRESULT: 0x" << std::hex << hr;
  }
}

void FlushAndWait(ID3D12Device* pDevice, ID3D12CommandQueue* pQueue) {
  HANDLE hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
  ComPtr<ID3D12Fence> pFence;
  pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&pFence));
  pQueue->Signal(pFence.Get(), 1);
  pFence->SetEventOnCompletion(1, hEvent);
  WaitForSingleObject(hEvent, INFINITE);
  CloseHandle(hEvent);
}

}  // namespace

// Test InitGraphicsInteropForEpDevice with command_queue = nullptr (graceful exit).
// Only built when D3D12 interop is enabled; otherwise Init would return ORT_NOT_IMPLEMENTED.
TEST(NvExecutionProviderTest, GraphicsInteropInitWithoutCommandQueue) {
  ASSERT_NE(ort_env.get(), nullptr);

  RegisteredEpDeviceUniquePtr nv_ep;
  Utils::RegisterAndGetNvTensorRtRtxEp(*ort_env, nv_ep);
  const OrtEpDevice* ep_device = nv_ep.get();
  ASSERT_NE(ep_device, nullptr);

  const OrtInteropApi& interop_api = Ort::GetInteropApi();
  OrtGraphicsInteropConfig config = {};
  config.version = ORT_API_VERSION;
  config.graphics_api = ORT_GRAPHICS_API_D3D12;
  config.command_queue = nullptr;
  config.additional_options = nullptr;

  {
    Ort::Status init_status(interop_api.InitGraphicsInteropForEpDevice(ep_device, &config));
    ASSERT_TRUE(init_status.IsOK()) << init_status.GetErrorMessage();
  }
  {
    Ort::Status deinit_status(interop_api.DeinitGraphicsInteropForEpDevice(ep_device));
    ASSERT_TRUE(deinit_status.IsOK()) << deinit_status.GetErrorMessage();
  }
}

// Test full D3D12 graphics interop: create D3D12 device and command queue,
// init graphics interop, create sync stream, release, deinit.
TEST(NvExecutionProviderTest, GraphicsInteropD3D12InitStreamDeinit) {
  ASSERT_NE(ort_env.get(), nullptr);

  D3D12CreateDeviceLoadResult d3d12 = LoadD3D12CreateDevice();
  if (!d3d12.pfn) {
    GTEST_SKIP() << "d3d12.dll or D3D12CreateDevice not available";
  }
  ComPtr<ID3D12Device> pDevice;
  ComPtr<ID3D12CommandQueue> pCommandQueue;
  HRESULT hr = d3d12.pfn(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&pDevice));
  if (FAILED(hr)) {
    GTEST_SKIP() << "D3D12 device creation failed, HRESULT: 0x" << std::hex << hr;
  }
  (void)d3d12.module;  // keep d3d12.dll loaded for the duration of the test

  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  hr = pDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&pCommandQueue));
  if (FAILED(hr)) {
    GTEST_SKIP() << "D3D12 command queue creation failed, HRESULT: 0x" << std::hex << hr;
  }

  RegisteredEpDeviceUniquePtr nv_ep;
  Utils::RegisterAndGetNvTensorRtRtxEp(*ort_env, nv_ep);
  const OrtEpDevice* ep_device = nv_ep.get();
  ASSERT_NE(ep_device, nullptr);

  const OrtInteropApi& interop_api = Ort::GetInteropApi();
  OrtGraphicsInteropConfig config = {};
  config.version = ORT_API_VERSION;
  config.graphics_api = ORT_GRAPHICS_API_D3D12;
  config.command_queue = pCommandQueue.Get();
  config.additional_options = nullptr;

  {
    Ort::Status init_status(interop_api.InitGraphicsInteropForEpDevice(ep_device, &config));
    if (!init_status.IsOK()) {
      GTEST_SKIP() << "InitGraphicsInteropForEpDevice failed (e.g. DX interop not built): "
                   << init_status.GetErrorMessage();
    }
  }

  const OrtApi& ort_api = Ort::GetApi();
  OrtSyncStream* stream = nullptr;
  {
    Ort::Status stream_status(ort_api.CreateSyncStreamForEpDevice(ep_device, nullptr, &stream));
    ASSERT_TRUE(stream_status.IsOK()) << stream_status.GetErrorMessage();
  }
  ASSERT_NE(stream, nullptr);
  ort_api.ReleaseSyncStream(stream);

  {
    Ort::Status deinit_status(interop_api.DeinitGraphicsInteropForEpDevice(ep_device));
    ASSERT_TRUE(deinit_status.IsOK()) << deinit_status.GetErrorMessage();
  }
}

// Full D3D12 interop + inference: mirrors SimpleDXInterop_cig_only Main.cpp with random input data.
// Uses ORT Interop API for semaphore (ImportSemaphore, WaitSemaphore, SignalSemaphore) and runs inference on a Relu model.
TEST(NvExecutionProviderTest, GraphicsInteropD3D12FullInference) {
  ASSERT_NE(ort_env.get(), nullptr);

  constexpr int image_dim = 64;  // Smaller than 1080 for faster unit test
  const size_t tensor_num_elements = 3 * image_dim * image_dim;
  const size_t tensor_byte_size = tensor_num_elements * sizeof(uint16_t);

  // Create simple Relu model (1, 3, image_dim, image_dim) FLOAT16
  PathString model_name = ORT_TSTR("nv_interop_inference_test.onnx");
  clearFileIfExists(model_name);
  {
    onnxruntime::Model model("interop_test", false, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();
    ONNX_NAMESPACE::TypeProto tensor_type;
    tensor_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
    tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
    tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(image_dim);
    tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(image_dim);

    auto& input_arg = graph.GetOrCreateNodeArg("input", &tensor_type);
    auto& output_arg = graph.GetOrCreateNodeArg("output", &tensor_type);
    graph.AddNode("relu_node", "Relu", "Relu operation", {&input_arg}, {&output_arg});
    ASSERT_STATUS_OK(graph.Resolve());
    ASSERT_STATUS_OK(onnxruntime::Model::Save(model, model_name));
  }

  // Random input data in valid float16 range (not raw uint16 bit patterns)
  std::vector<uint16_t> cpuInputHalf(tensor_num_elements);
  std::vector<uint16_t> cpuOutputHalf(tensor_num_elements);
  {
    RandomValueGenerator random{};
    std::vector<int64_t> shape{3, image_dim, image_dim};
    std::vector<MLFloat16> input_fp16 = random.Uniform<MLFloat16>(shape, -2.f, 2.f);
    memcpy(cpuInputHalf.data(), input_fp16.data(), tensor_byte_size);
  }

  D3D12CreateDeviceLoadResult d3d12 = LoadD3D12CreateDevice();
  if (!d3d12.pfn) {
    GTEST_SKIP() << "d3d12.dll or D3D12CreateDevice not available";
  }
  ComPtr<ID3D12Device> pDevice;
  ComPtr<ID3D12CommandQueue> pCommandQueue;
  HRESULT hr = d3d12.pfn(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&pDevice));
  if (FAILED(hr)) {
    GTEST_SKIP() << "D3D12 device creation failed, HRESULT: 0x" << std::hex << hr;
  }
  (void)d3d12.module;  // keep d3d12.dll loaded for the duration of the test
  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  hr = pDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&pCommandQueue));
  if (FAILED(hr)) {
    GTEST_SKIP() << "D3D12 command queue creation failed, HRESULT: 0x" << std::hex << hr;
  }

  RegisteredEpDeviceUniquePtr nv_ep;
  Utils::RegisterAndGetNvTensorRtRtxEp(*ort_env, nv_ep);
  const OrtEpDevice* trt_ep_device = nv_ep.get();
  ASSERT_NE(trt_ep_device, nullptr);

  const OrtApi& ort_api = Ort::GetApi();
  const OrtInteropApi& interop_api = Ort::GetInteropApi();

  OrtGraphicsInteropConfig graphics_config = {};
  graphics_config.version = ORT_API_VERSION;
  graphics_config.graphics_api = ORT_GRAPHICS_API_D3D12;
  graphics_config.command_queue = pCommandQueue.Get();
  graphics_config.additional_options = nullptr;
  {
    Ort::Status init_status(interop_api.InitGraphicsInteropForEpDevice(trt_ep_device, &graphics_config));
    if (!init_status.IsOK()) {
      GTEST_SKIP() << "InitGraphicsInteropForEpDevice failed: " << init_status.GetErrorMessage();
    }
  }

  // Optional: D3D12 fence/semaphore for GPU sync between D3D12 and ORT. Only call
  // CreateExternalResourceImporterForDevice when the EP implements it (e.g. not NV TensorRT RTX).
  // Otherwise run inference with CPU sync (FlushAndWait) instead.
  OrtExternalResourceImporter* importer = nullptr;
  OrtExternalSemaphoreHandle* ort_sem_handle = nullptr;
  ComPtr<ID3D12Fence> pFence;
  HANDLE sharedFenceHandle = nullptr;
  const bool ep_supports_external_importer =
      (strcmp(ort_api.EpDevice_EpName(trt_ep_device), Utils::nv_tensorrt_rtx_ep_info.registration_name.c_str()) != 0);
  if (ep_supports_external_importer) {
    Ort::Status s(interop_api.CreateExternalResourceImporterForDevice(trt_ep_device, &importer));
    if (s.IsOK() && importer != nullptr) {
      enum FenceState {
        FENCE_UPLOAD_DONE = 1,
        FENCE_KERNEL_DONE = 2
      };
      pDevice->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&pFence));
      pDevice->CreateSharedHandle(pFence.Get(), nullptr, GENERIC_ALL, nullptr, &sharedFenceHandle);
      ASSERT_NE(sharedFenceHandle, nullptr);
      OrtExternalSemaphoreDescriptor sem_desc = {};
      sem_desc.version = ORT_API_VERSION;
      sem_desc.type = ORT_EXTERNAL_SEMAPHORE_D3D12_FENCE;
      sem_desc.native_handle = sharedFenceHandle;
      Ort::Status import_s(interop_api.ImportSemaphore(importer, &sem_desc, &ort_sem_handle));
      ASSERT_TRUE(import_s.IsOK()) << import_s.GetErrorMessage();
      ASSERT_NE(ort_sem_handle, nullptr);
    }
  }

  OrtSyncStream* stream = nullptr;
  {
    Ort::Status s(ort_api.CreateSyncStreamForEpDevice(trt_ep_device, nullptr, &stream));
    ASSERT_TRUE(s.IsOK()) << s.GetErrorMessage();
  }
  ASSERT_NE(stream, nullptr);

  // IHV-agnostic memory info for binding D3D12 buffers
  OrtMemoryInfo* memory_info_agnostic = nullptr;
  const OrtHardwareDevice* hw_device = ort_api.EpDevice_Device(trt_ep_device);
  UINT vID = ort_api.HardwareDevice_VendorId(hw_device);
  {
    Ort::Status mem_status(ort_api.CreateMemoryInfo_V2(
        "Device_Agnostic", OrtMemoryInfoDeviceType_GPU, vID, 0,
        OrtDeviceMemoryType_DEFAULT, 0, OrtArenaAllocator, &memory_info_agnostic));
    ASSERT_TRUE(mem_status.IsOK()) << mem_status.GetErrorMessage();
  }
  ASSERT_NE(memory_info_agnostic, nullptr);

  // Session options: user compute stream for non-CPU EPs
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  sessionOptions.DisableMemPattern();
  sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  sessionOptions.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
  sessionOptions.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);

  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_ep_devices = 0;
  {
    Ort::Status get_devices_status(ort_api.GetEpDevices(*ort_env, &ep_devices, &num_ep_devices));
    ASSERT_TRUE(get_devices_status.IsOK()) << get_devices_status.GetErrorMessage();
  }
  char stream_address[32];
  sprintf_s(stream_address, "%llu", static_cast<unsigned long long>(reinterpret_cast<size_t>(ort_api.SyncStream_GetHandle(stream))));
  const char* option_keys[] = {"user_compute_stream", "has_user_compute_stream"};
  const char* option_values[] = {stream_address, "1"};
  for (size_t i = 0; i < num_ep_devices; i++) {
    if (strcmp(ort_api.EpDevice_EpName(ep_devices[i]), "CPUExecutionProvider") != 0) {
      Ort::Status append_ep_status(ort_api.SessionOptionsAppendExecutionProvider_V2(
          sessionOptions, *ort_env, &ep_devices[i], 1, option_keys, option_values, 2));
      ASSERT_TRUE(append_ep_status.IsOK()) << append_ep_status.GetErrorMessage();
    }
  }

  // D3D12 resources
  ComPtr<ID3D12Resource> pInput;
  ComPtr<ID3D12Resource> pOutput;
  ComPtr<ID3D12Resource> pUploadRes;
  ComPtr<ID3D12Resource> pDownloadRes;
  CreateD3D12Buffer(pDevice.Get(), tensor_byte_size, pInput.GetAddressOf(), D3D12_RESOURCE_STATE_COPY_DEST);
  CreateD3D12Buffer(pDevice.Get(), tensor_byte_size, pOutput.GetAddressOf(), D3D12_RESOURCE_STATE_COPY_SOURCE);
  CreateUploadBuffer(pDevice.Get(), tensor_byte_size, pUploadRes.GetAddressOf());
  CreateReadBackBuffer(pDevice.Get(), tensor_byte_size, pDownloadRes.GetAddressOf());

  ComPtr<ID3D12CommandAllocator> pAllocatorCopy;
  pDevice->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&pAllocatorCopy));
  ComPtr<ID3D12GraphicsCommandList> pUploadCommandList;
  pDevice->CreateCommandList(1, D3D12_COMMAND_LIST_TYPE_COMPUTE, pAllocatorCopy.Get(), nullptr,
                             IID_PPV_ARGS(&pUploadCommandList));
  pUploadCommandList->CopyResource(pInput.Get(), pUploadRes.Get());
  pUploadCommandList->Close();
  ComPtr<ID3D12GraphicsCommandList> pDownloadCommandList;
  pDevice->CreateCommandList(1, D3D12_COMMAND_LIST_TYPE_COMPUTE, pAllocatorCopy.Get(), nullptr,
                             IID_PPV_ARGS(&pDownloadCommandList));
  pDownloadCommandList->CopyResource(pDownloadRes.Get(), pOutput.Get());
  pDownloadCommandList->Close();

  // Session and inference in inner scope so TensorRT engine is destroyed before we destroy the CIG context in Deinit.
  {
    Ort::Session session(*ort_env, model_name.c_str(), sessionOptions);
    Ort::IoBinding ioBinding(session);
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);
    int64_t input_dim[] = {1, 3, image_dim, image_dim};
    int64_t output_dim[] = {1, 3, image_dim, image_dim};

    // Upload random input to D3D12 upload buffer
    void* pData = nullptr;
    pUploadRes->Map(0, nullptr, &pData);
    memcpy(pData, cpuInputHalf.data(), tensor_byte_size);
    pUploadRes->Unmap(0, nullptr);

    Ort::Value input_tensor(Ort::Value::CreateTensor(
        memory_info_agnostic, reinterpret_cast<void*>(pInput->GetGPUVirtualAddress()),
        tensor_byte_size, input_dim, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
    Ort::Value output_tensor(Ort::Value::CreateTensor(
        memory_info_agnostic, reinterpret_cast<void*>(pOutput->GetGPUVirtualAddress()),
        tensor_byte_size, output_dim, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
    ioBinding.BindInput(input_name.get(), input_tensor);
    ioBinding.BindOutput(output_name.get(), output_tensor);
    ioBinding.SynchronizeInputs();

    // Run 2 iterations: upload -> sync -> inference -> download
    const bool use_semaphore_sync = (importer != nullptr && ort_sem_handle != nullptr);
    enum FenceState {
      FENCE_UPLOAD_DONE = 1,
      FENCE_KERNEL_DONE = 2
    };
    Ort::RunOptions run_options;
    run_options.AddConfigEntry("disable_synchronize_execution_providers", "1");
    for (int iter = 0; iter < 2; iter++) {
      ID3D12CommandList* upload_list = pUploadCommandList.Get();
      pCommandQueue->ExecuteCommandLists(1, &upload_list);
      if (use_semaphore_sync) {
        pCommandQueue->Signal(pFence.Get(), FENCE_UPLOAD_DONE);
        Ort::Status s(interop_api.WaitSemaphore(importer, ort_sem_handle, stream, FENCE_UPLOAD_DONE));
        ASSERT_TRUE(s.IsOK()) << s.GetErrorMessage();
      } else {
        FlushAndWait(pDevice.Get(), pCommandQueue.Get());
      }
      session.Run(run_options, ioBinding);
      if (use_semaphore_sync) {
        Ort::Status s(interop_api.SignalSemaphore(importer, ort_sem_handle, stream, FENCE_KERNEL_DONE));
        ASSERT_TRUE(s.IsOK()) << s.GetErrorMessage();
        pCommandQueue->Wait(pFence.Get(), FENCE_KERNEL_DONE);
      }
      ID3D12CommandList* download_list = pDownloadCommandList.Get();
      pCommandQueue->ExecuteCommandLists(1, &download_list);
      FlushAndWait(pDevice.Get(), pCommandQueue.Get());
    }

    // Read back output and validate Relu: interpret as float16, compare to max(0, input)
    void* pOutputData = nullptr;
    pDownloadRes->Map(0, nullptr, &pOutputData);
    memcpy(cpuOutputHalf.data(), pOutputData, tensor_byte_size);
    pDownloadRes->Unmap(0, nullptr);
    std::vector<float> input_float(tensor_num_elements);
    std::vector<float> output_float(tensor_num_elements);
    ConvertMLFloat16ToFloat(reinterpret_cast<const MLFloat16*>(cpuInputHalf.data()), input_float.data(), tensor_num_elements);
    ConvertMLFloat16ToFloat(reinterpret_cast<const MLFloat16*>(cpuOutputHalf.data()), output_float.data(), tensor_num_elements);
    const float tol = 1e-2f;  // float16 limited precision
    for (size_t i = 0; i < tensor_num_elements; i++) {
      float expected = std::max(0.f, input_float[i]);
      ASSERT_NEAR(output_float[i], expected, tol) << "Relu mismatch at index " << i;
    }
  }

  // Cleanup: release resources that use the interop context before DeinitGraphicsInteropForEpDevice
  ort_api.ReleaseSyncStream(stream);
  if (importer != nullptr) {
    interop_api.ReleaseExternalSemaphoreHandle(ort_sem_handle);
    interop_api.ReleaseExternalResourceImporter(importer);
    CloseHandle(sharedFenceHandle);
  }
  ort_api.ReleaseMemoryInfo(memory_info_agnostic);
  {
    Ort::Status deinit_status(interop_api.DeinitGraphicsInteropForEpDevice(trt_ep_device));
    ASSERT_TRUE(deinit_status.IsOK()) << deinit_status.GetErrorMessage();
  }
}

#endif  // USE_DX_INTEROP && _WIN32

}  // namespace test
}  // namespace onnxruntime
