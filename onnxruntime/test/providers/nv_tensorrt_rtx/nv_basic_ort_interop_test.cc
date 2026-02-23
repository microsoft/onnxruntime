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
#endif  // USE_DX_INTEROP && _WIN32

// Test InitGraphicsInteropForEpDevice with command_queue = nullptr (graceful exit).
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

#if defined(USE_DX_INTEROP) && USE_DX_INTEROP && defined(_WIN32)
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

#endif  // USE_DX_INTEROP && _WIN32

}  // namespace test
}  // namespace onnxruntime
