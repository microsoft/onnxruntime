// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/framework_test_utils.h"

#include "test/util/include/scoped_env_vars.h"
#include "test/common/trt_op_test_utils.h"
#include "test/common/random_generator.h"
#include "test/providers/nv_tensorrt_rtx/test_nv_trt_rtx_ep_util.h"

#include <thread>
#include <chrono>

// #include <onnxruntime_cxx_api.h>
#if DX_FOR_INTEROP && _WIN32
#include <d3d12.h>
#endif

using namespace std;
using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::logging;
extern std::unique_ptr<Ort::Env> ort_env;
namespace onnxruntime {

  #if DX_FOR_INTEROP && _WIN32
void CreateD3D12Buffer(ID3D12Device* pDevice, const size_t size, ID3D12Resource** ppResource, D3D12_RESOURCE_STATES initState)
{
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
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        initState,
        nullptr,
        IID_PPV_ARGS(ppResource));

    if (FAILED(hr))
    {
        printf("\nFailed creating a resource\n");
        exit(0);
    }
}

void CreateUploadBuffer(ID3D12Device* pDevice, const size_t size, ID3D12Resource** ppResource)
{
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
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_GENERIC_READ,
        nullptr,
        IID_PPV_ARGS(ppResource));
}

void CreateReadBackBuffer(ID3D12Device* pDevice, const size_t size, ID3D12Resource** ppResource)
{
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
        &heapProps,
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_COPY_DEST,
        nullptr,
        IID_PPV_ARGS(ppResource));
}


void FlushAndWait(ID3D12Device* pDevice, ID3D12CommandQueue* pQueue)
{
    // Event and D3D12 Fence to manage CPU<->GPU sync (we want to keep 2 iterations in "flight")
    HANDLE hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    ID3D12Fence* pFence = nullptr;
    pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&pFence));

    pQueue->Signal(pFence, 1);
    pFence->SetEventOnCompletion(1, hEvent);
    DWORD retVal = WaitForSingleObject(hEvent, INFINITE);

    pFence->Release();
    CloseHandle(hEvent);
}
#endif
namespace test {

TEST(NvExecutionProviderTest, GraphicsORTInteropTest) {
  PathString model_name = ORT_TSTR("nv_execution_provider_test.onnx");
  std::string graph_name = "test";
  constexpr int image_dim = 1080;
  
  // Create a simple 1-input, 1-output Relu model
  onnxruntime::Model model(graph_name, false, DefaultLoggingManager().DefaultLogger());
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

#if DX_FOR_INTEROP && _WIN32
  {
    std::vector<uint16_t> cpuInputHalf(3 * image_dim * image_dim);
    std::vector<uint16_t> cpuOutputHalf(3 * image_dim * image_dim);
    using StreamUniquePtr = std::unique_ptr<OrtSyncStream, std::function<void(OrtSyncStream*)>>;

  // Generate random data for input
  {
    RandomValueGenerator random{};
    std::vector<int64_t> shape{3, image_dim, image_dim};
    std::vector<uint16_t> input_data = random.Uniform<uint16_t>(shape, static_cast<uint16_t>(0), static_cast<uint16_t>(65535));
    memcpy(cpuInputHalf.data(), input_data.data(), cpuInputHalf.size() * sizeof(uint16_t));
  }  // input_data is freed here


  // set up d3d12
  ID3D12Device* pDevice;
  ID3D12CommandQueue* pCommandQueue;
  ID3D12Resource* pInput;
  ID3D12Resource* pOutput;
  ID3D12Resource* pUploadRes;
  ID3D12Resource* pUploadResCorrupt;
  ID3D12Resource* pDownloadRes;
  ID3D12GraphicsCommandList* pUploadCommandList;      // pUploadRes->pInput copy command
  ID3D12GraphicsCommandList* pDownloadCommandList;    // pOutput->pDownloadRes copy command
  ID3D12CommandAllocator* pAllocatorCopy;
  
  uint64_t fenceValue = 0;
  GraphicsInteropParams graphicsInteropParams;
  graphicsInteropParams.extSyncPrimitive = ExternalSyncPrimitive_D3D12Fence;
  graphicsInteropParams.DevicePtr.DXDeviceParams.pDevice = nullptr;
  graphicsInteropParams.DevicePtr.DXDeviceParams.pCommandQueue = nullptr;
  HANDLE sharedFenceHandle = nullptr;
  FenceInteropParams fenceInteropParams;
  fenceInteropParams.extSyncPrimitive = ExternalSyncPrimitive_D3D12Fence;
  fenceInteropParams.FencePtr.pFence = nullptr;
  OrtFence* ortFence = nullptr;

  HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&graphicsInteropParams.DevicePtr.DXDeviceParams.pDevice));
  if (FAILED(hr))
  {
      std::cerr << "Failed to create D3D12 device" <<std::endl;
  }

  D3D12_COMMAND_QUEUE_DESC queueDesc = {};
  queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
  hr = reinterpret_cast<ID3D12Device*>(graphicsInteropParams.DevicePtr.DXDeviceParams.pDevice)->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&pCommandQueue));
  graphicsInteropParams.DevicePtr.DXDeviceParams.pCommandQueue = pCommandQueue;

  // Use ORT APIs to load the model
  OrtApi const& ortApi = Ort::GetApi();
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
  sessionOptions.DisableMemPattern();
  sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  sessionOptions.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
  ortApi.AddFreeDimensionOverrideByName(sessionOptions, "batch_size", 1);

  std::string trtLibPath = "onnxruntime_providers_nv_tensorrt_rtx.dll";
  std::wstring wideTrtLibPath = std::wstring(trtLibPath.begin(), trtLibPath.end());
  
  OrtStatus* status = ortApi.RegisterExecutionProviderLibrary(*ort_env, "NvTensorRtRtx", wideTrtLibPath.c_str());
  if (status != nullptr) {
    std::string error_message = ortApi.GetErrorMessage(status);
    ortApi.ReleaseStatus(status);
    FAIL() << "Failed to register EP library: " << error_message;
  }

  sessionOptions.SetEpSelectionPolicy(OrtExecutionProviderDevicePolicy_PREFER_GPU);

  reinterpret_cast<ID3D12Device*>(graphicsInteropParams.DevicePtr.DXDeviceParams.pDevice)->CreateFence(fenceValue, D3D12_FENCE_FLAG_SHARED, __uuidof(ID3D12Fence), reinterpret_cast<void**>(&fenceInteropParams.FencePtr.pFence));
  reinterpret_cast<ID3D12Device*>(graphicsInteropParams.DevicePtr.DXDeviceParams.pDevice)->CreateSharedHandle(reinterpret_cast<ID3D12Fence*>(fenceInteropParams.FencePtr.pFence), nullptr, GENERIC_ALL, nullptr, &sharedFenceHandle);

  const OrtEpDevice* const* ep_devices = nullptr;
  size_t num_ep_devices;
  ortApi.GetEpDevices(*ort_env, &ep_devices, &num_ep_devices);
  const OrtEpDevice* trt_ep_device = nullptr;
  for (UINT i = 0; i < num_ep_devices; i++)
  {
      if (strcmp(ortApi.EpDevice_EpName(ep_devices[i]), "NvTensorRTRTXExecutionProvider") == 0)
      {
          trt_ep_device = ep_devices[i];
          break;
      }
  }

  // Must be called before other interop functions to create the context
  ortApi.SetupGraphicsInteropContextForEpDevice(trt_ep_device, &graphicsInteropParams);

  // Create ORT stream - this will be created on the context we just set up
  OrtSyncStream* stream = nullptr;
  StreamUniquePtr stream_ptr;
  ortApi.CreateSyncStreamForEpDevice(trt_ep_device, nullptr, &stream);
  stream_ptr = StreamUniquePtr(stream, [ortApi](OrtSyncStream* stream) { ortApi.ReleaseSyncStream(stream); });

  // Create IHV-agnostic memory info using hardware device vendor ID
  OrtMemoryInfo* memory_info_agnostic = nullptr;
  const OrtHardwareDevice* hw_device = ortApi.EpDevice_Device(trt_ep_device);
  UINT vID = ortApi.HardwareDevice_VendorId(hw_device);
  ortApi.CreateMemoryInfo_V2("Device_Agnostic", OrtMemoryInfoDeviceType_GPU, 
                              /*vendor_id*/vID, /*device_id*/0, 
                              OrtDeviceMemoryType_DEFAULT, /*default alignment*/0, 
                              OrtArenaAllocator, &memory_info_agnostic);
  
  auto memory_info_cleanup = std::unique_ptr<OrtMemoryInfo, std::function<void(OrtMemoryInfo*)>>(
    memory_info_agnostic,
    [&ortApi](OrtMemoryInfo* ptr) {
      if (ptr) ortApi.ReleaseMemoryInfo(ptr);
    }
  );

  char streamAddress[32];
  size_t stream_addr_val = reinterpret_cast<size_t>(ortApi.SyncStream_GetHandle(stream));
  sprintf_s(streamAddress, "%llu", static_cast<uint64_t>(stream_addr_val));
  const char* option_keys[] = { "user_compute_stream", "has_user_compute_stream" };
  const char* option_values[] = { streamAddress, "1" };
  for (size_t i = 0; i < num_ep_devices; i++)
  {
      if (strcmp(ortApi.EpDevice_EpName(ep_devices[i]), "CPUExecutionProvider") != 0)
          ortApi.SessionOptionsAppendExecutionProvider_V2(sessionOptions, *ort_env, &ep_devices[i], 1, option_keys, option_values, 2);
  }

  // default resources
  CreateD3D12Buffer(reinterpret_cast<ID3D12Device*>(graphicsInteropParams.DevicePtr.DXDeviceParams.pDevice), 3 * image_dim * image_dim * sizeof(uint16_t), &pInput, D3D12_RESOURCE_STATE_COPY_DEST);
  CreateD3D12Buffer(reinterpret_cast<ID3D12Device*>(graphicsInteropParams.DevicePtr.DXDeviceParams.pDevice), 3 * image_dim * image_dim * sizeof(uint16_t), &pOutput, D3D12_RESOURCE_STATE_COPY_SOURCE);

  // upload and download resources
  CreateUploadBuffer(reinterpret_cast<ID3D12Device*>(graphicsInteropParams.DevicePtr.DXDeviceParams.pDevice), 3 * image_dim * image_dim * sizeof(uint16_t), &pUploadRes);
  CreateUploadBuffer(reinterpret_cast<ID3D12Device*>(graphicsInteropParams.DevicePtr.DXDeviceParams.pDevice), 3 * image_dim * image_dim * sizeof(uint16_t), &pUploadResCorrupt);  // Ankan - for test!

  CreateReadBackBuffer(reinterpret_cast<ID3D12Device*>(graphicsInteropParams.DevicePtr.DXDeviceParams.pDevice), 3 * image_dim * image_dim * sizeof(uint16_t), &pDownloadRes);

  hr = reinterpret_cast<ID3D12Device*>(graphicsInteropParams.DevicePtr.DXDeviceParams.pDevice)->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&pAllocatorCopy));

  hr = reinterpret_cast<ID3D12Device*>(graphicsInteropParams.DevicePtr.DXDeviceParams.pDevice)->CreateCommandList(1, D3D12_COMMAND_LIST_TYPE_COMPUTE, pAllocatorCopy, NULL, IID_PPV_ARGS(&pUploadCommandList));


  // heavy GPU load for reproducing race condition
  for (int i = 0; i < 1000; i++)
  {
      pUploadCommandList->CopyResource(pInput, pUploadResCorrupt);

      D3D12_RESOURCE_BARRIER barrier = {};
      barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
      barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
      barrier.UAV.pResource = nullptr; // This makes it a NULL UAV barrier

      pUploadCommandList->ResourceBarrier(1, &barrier);
  }

  pUploadCommandList->CopyResource(pInput, pUploadRes);
  pUploadCommandList->Close();

  std::cerr << "Test completed successfully1" <<std::endl;

  // record the commands in the download command list
  hr = reinterpret_cast<ID3D12Device*>(graphicsInteropParams.DevicePtr.DXDeviceParams.pDevice)->CreateCommandList(1, D3D12_COMMAND_LIST_TYPE_COMPUTE, pAllocatorCopy, NULL, IID_PPV_ARGS(&pDownloadCommandList));
  pDownloadCommandList->CopyResource(pDownloadRes, pOutput);
  pDownloadCommandList->Close();

  Ort::Session session = Ort::Session(*ort_env, L"nv_execution_provider_test.onnx", sessionOptions);

  ortApi.GetOrtFenceForGraphicsInterop(session, &graphicsInteropParams, &fenceInteropParams, &ortFence);

  Ort::IoBinding ioBinding = Ort::IoBinding::IoBinding(session);

  Ort::AllocatorWithDefaultOptions allocator;
  Ort::AllocatedStringPtr InputTensorName = session.GetInputNameAllocated(0, allocator);
  Ort::AllocatedStringPtr OuptutTensorName = session.GetOutputNameAllocated(0, allocator);

  int64_t inputDim[] = { 1, 3, image_dim, image_dim };
  int64_t outputDim[] = { 1, 3, image_dim, image_dim };

  // upload the input
  void* pData;
  pUploadRes->Map(0, nullptr, (void**)&pData);
  memcpy(pData, cpuInputHalf.data(), cpuInputHalf.size() * sizeof(uint16_t));
  pUploadRes->Unmap(0, nullptr);

  // Upload corrupted data to test synchronization (should not affect the output)
  void* pDataCorrupt;
  pUploadResCorrupt->Map(0, nullptr, (void**)&pDataCorrupt);
  std::fill_n((uint8_t*)pDataCorrupt, 3 * image_dim * image_dim * sizeof(uint16_t), 0xFF);
  pUploadResCorrupt->Unmap(0, nullptr);

  // bind the resources using IHV-agnostic memory info but keep zero-copy external memory sharing
  Ort::Value inputTensor(Ort::Value::CreateTensor(memory_info_agnostic, (void*)pInput->GetGPUVirtualAddress(), cpuInputHalf.size() * sizeof(uint16_t), inputDim, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
  Ort::Value outputTensor(Ort::Value::CreateTensor(memory_info_agnostic, (void*)pOutput->GetGPUVirtualAddress(), cpuOutputHalf.size() * sizeof(uint16_t), outputDim, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16));
  ioBinding.BindInput(InputTensorName.get(), inputTensor);
  ioBinding.BindOutput(OuptutTensorName.get(), outputTensor);
  ioBinding.SynchronizeInputs();

  std::cerr << "Test completed successfully2" <<std::endl;

  for (int i = 0; i < 2; i++)
  {
      fenceValue++;
      // upload inputs (using DX)
      pCommandQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&pUploadCommandList);

      // make ORT wait for upload
      pCommandQueue->Signal(reinterpret_cast<ID3D12Fence*>(fenceInteropParams.FencePtr.pFence), fenceValue);                         // signal the fence from D3D12 side
      ortApi.InteropEpWait(session, ortFence, stream, fenceValue);    // make ORT wait on the fence (on CUDA side internally)

      // run the model
      Ort::RunOptions runOptions;
      runOptions.AddConfigEntry("disable_synchronize_execution_providers", "1");
      session.Run(runOptions, ioBinding);

      fenceValue++;
      // make DX wait for ORT
      ortApi.InteropEpSignal(session, ortFence, stream, fenceValue);  // signal from CUDA side (internally)
      pCommandQueue->Wait(reinterpret_cast<ID3D12Fence*>(fenceInteropParams.FencePtr.pFence), fenceValue);                           // make D3D12 wait on the fence

      // download the output to cpu memory (again using DX)
      pCommandQueue->ExecuteCommandLists(1, (ID3D12CommandList**)&pDownloadCommandList);
      FlushAndWait(reinterpret_cast<ID3D12Device*>(graphicsInteropParams.DevicePtr.DXDeviceParams.pDevice), pCommandQueue);
  }


  std::cerr << "Test completed successfully3" <<std::endl;
  void* pOutputData;
  pDownloadRes->Map(0, nullptr, (void**)&pOutputData);
  memcpy(cpuOutputHalf.data(), pOutputData, cpuOutputHalf.size() * sizeof(uint16_t));
  pDownloadRes->Unmap(0, nullptr);

  std::cerr << "First 20 elements of cpuInputHalf:\n";
  for (int i = 0; i < 20; i++) {
    std::cerr << cpuInputHalf[i] << " ";
  }
  std::cerr << std::endl;
  
  // Print first 20 elements of output
  std::cerr << "First 20 elements of cpuOutputHalf:\n";
  for (int i = 0; i < 20; i++) {
    std::cerr << cpuOutputHalf[i] << " ";
  }
  std::cerr << std::endl;


  pInput->Release();
  pOutput->Release();
  pUploadRes->Release();
  pUploadResCorrupt->Release();
  pDownloadRes->Release();
  pUploadCommandList->Release();
  pDownloadCommandList->Release();
  pAllocatorCopy->Release();
  reinterpret_cast<ID3D12Fence*>(fenceInteropParams.FencePtr.pFence)->Release();
  CloseHandle(sharedFenceHandle);
  pCommandQueue->Release();
  reinterpret_cast<ID3D12Device*>(graphicsInteropParams.DevicePtr.DXDeviceParams.pDevice)->Release();

  std::cerr << "\nInference done. Check output image." <<std::endl;

  }  // End scope - cpuInputHalf and cpuOutputHalf are freed here
#endif
std::cerr << "Test completed successfully" <<std::endl;
}

}  // namespace test
}  // namespace onnxruntime
