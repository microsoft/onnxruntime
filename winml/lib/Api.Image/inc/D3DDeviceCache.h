// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "pch.h"

//
// Exception information
//
#ifndef FACILITY_VISUALCPP
#define FACILITY_VISUALCPP ((LONG)0x6d)
#endif

#define VcppException(sev, err) ((sev) | (FACILITY_VISUALCPP << 16) | err)

namespace _winml {

enum class PipelineStateCacheType : unsigned char {
  kFloat32 = 0,
  kFloat16 = 1,
  kCount = 2
};

enum class PipelineStateCacheFormat : unsigned char {
  kRGB8 = 0,
  kBGR8 = 1,
  kGRAY8 = 2,
  kCount = 3
};

enum class PipelineStateCacheOperation : unsigned char {
  kTensorize = 0,
  kDetensorize = 1,
  kCount = 2
};

template <typename E>
constexpr auto underlying(E e) noexcept {
  return static_cast<typename std::underlying_type<E>::type>(e);
}

class D3DDeviceCache {
 public:
  ~D3DDeviceCache();
  D3DDeviceCache(winml::LearningModelDeviceKind const& device_kind);
  D3DDeviceCache(wgdx::Direct3D11::IDirect3DDevice const& device);
  D3DDeviceCache(ID3D12CommandQueue* queue);

  ID3D11Device* GetD3D11Device();
  ID3D11DeviceContext4* GetD3D11DeviceContext();

  ID3D12Device1* GetD3D12Device() { return device_.get(); }
  ID3D12CommandQueue* GetCommandQueue() { return command_queue_.get(); }

  wgdx::Direct3D11::IDirect3DDevice GetWinrtDevice();

  ID3D12RootSignature* GetTensorizeRootSignature();
  ID3D12RootSignature* GetDetensorizeRootSignature();
  ID3D12PipelineState* GetCachedPipelineState(
    PipelineStateCacheType type,
    PipelineStateCacheFormat format_from,
    PipelineStateCacheFormat format_to,
    PipelineStateCacheOperation operation
  );

  ID3D12Resource* GetDetensorizeVertexBuffer(_Out_ UINT* vertex_buffer_size);

  HANDLE GetConverterFenceHandle();

  const GUID& GetFenceGuid() const;

  void GPUSyncD3D11ToD3D12();
  void GPUSyncD3D12ToD3D11();
  void SyncD3D12ToCPU();

  void SyncConverterToD3D11Device(_In_ ID3D11Fence* d3d11_fence_);
  void SyncD3D11DeviceToConverter(_In_ ID3D11Fence* d3d11_fence_);

  UINT64 QueueFenceToD3D12();
  void WaitForFenceValue(UINT64 fence_value);

  const LUID& GetDeviceLuid() { return device_luid_; };

  bool IsFloat16Supported();
  bool SharedHandleInitialized();

 private:
  void EnsureD3D11FromD3D12();
  void EnsureD3D12Fence();
  void EnsureSharedFences();
  void InitializeCommandQueue(ID3D12Device1* device);

  ID3D12PipelineState* CreateTensorizePipelineState(
    PipelineStateCacheType type, PipelineStateCacheFormat format_from, PipelineStateCacheFormat format_to
  );
  ID3D12PipelineState* CreateDetensorizePipelineState(
    PipelineStateCacheType type, PipelineStateCacheFormat format_from, PipelineStateCacheFormat format_to
  );

  winrt::com_ptr<ID3D12Device1> device_;
  winrt::com_ptr<ID3D12CommandQueue> command_queue_;
  winrt::com_ptr<ID3D12SharingContract> sharing_contract_;

  winrt::com_ptr<ID3D11Device> device_11_;
  wgdx::Direct3D11::IDirect3DDevice winrt_device_;
  winrt::com_ptr<ID3D11DeviceContext4> device_context11_;

  winrt::com_ptr<ID3D12RootSignature> tensorize_root_signature_;
  winrt::com_ptr<ID3D12RootSignature> detensorize_root_signature_;

  // clang-format off
  winrt::com_ptr<ID3D12PipelineState>
    cached_pipeline_state[underlying(PipelineStateCacheType::kCount)][underlying(PipelineStateCacheFormat::kCount)]
                         [underlying(PipelineStateCacheFormat::kCount)][underlying(PipelineStateCacheOperation::kCount)];

  winrt::com_ptr<ID3D12Resource> detensorize_vertex_buffer_;

  winrt::com_ptr<ID3D11Fence> d3d11_fence_;
  winrt::com_ptr<ID3D12Fence> d3d12_fence_;
  std::atomic<UINT64> fence_value_ = 1;

  GUID fence_guid_;

  winrt::com_ptr<ID3D12Fence> converter_fence_;
  wil::unique_handle converter_fence_handle_;
  std::atomic<UINT64> converter_fence_value_ = 1;

  LUID device_luid_;
  static const UINT sc_vertexBufferSize = sizeof(DirectX::XMFLOAT3) * 4;

  // added a lock when we added delay loading to the device cache.   Since parts of
  // initialization happen later, we need make it thread safe.
  CWinMLLock lock_;
};
}  // namespace _winml
