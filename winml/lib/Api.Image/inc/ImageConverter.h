// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <wrl/implements.h>
#include "WinML_Lock.h"
#include "ImageConversionHelpers.h"

// Assign a name to the object to aid with debugging.
#if defined(_DEBUG)
inline void SetName(ID3D12Object* object, LPCWSTR name) {
  object->SetName(name);
}
inline void SetNameIndexed(ID3D12Object* object, LPCWSTR name, UINT index) {
  WCHAR full_name[50];
  if (swprintf_s(full_name, L"%s[%u]", name, index) > 0) {
    object->SetName(full_name);
  }
}
#else
inline void SetName(ID3D12Object*, LPCWSTR) {
}
inline void SetNameIndexed(ID3D12Object*, LPCWSTR, UINT) {
}
#endif

// Forward declaration
namespace winrt::Windows::AI::MachineLearning::implementation {
class D3DDeviceCache;
}

namespace Windows::AI::MachineLearning::Internal {
struct ConstantBufferCS {
  UINT height;
  UINT width;
};

class ImageConverter {
 public:
  ImageConverter() : converted_video_frame_(nullptr) {}
  void ResetAllocator();

 protected:
  // Indices of shader resources in the descriptor heap.
  enum DescriptorHeapIndex : UINT32 {
    SrvBufferIdx = 0,
    UavBufferIdx = SrvBufferIdx + 1,
    DescriptorCount = UavBufferIdx + 1
  };

  Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> command_list_;
  Microsoft::WRL::ComPtr<ID3D12CommandAllocator> command_allocator_;
  Microsoft::WRL::ComPtr<ID3D12RootSignature> root_signature_;
  Microsoft::WRL::ComPtr<ID3D12PipelineState> pipeline_state_;
  Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> descriptor_heap_;
  Microsoft::WRL::ComPtr<ID3D11Texture2D> D3D11_cached_texture_;
  winrt::Windows::Media::VideoFrame converted_video_frame_;
  CWinMLLock lock_;

  void SyncD3D11ToD3D12(_In_ winrt::Windows::AI::MachineLearning::implementation::D3DDeviceCache& device_cache, _In_ ID3D11Texture2D* D3D11_texture);
  void SyncD3D12ToD3D11(_In_ winrt::Windows::AI::MachineLearning::implementation::D3DDeviceCache& device_cache, _In_ ID3D11Texture2D* texture);
  void ResetCommandList(_In_ winrt::Windows::AI::MachineLearning::implementation::D3DDeviceCache& device_cache);
  Microsoft::WRL::ComPtr<ID3D11Fence> FetchOrCreateFenceOnDevice(_In_ winrt::Windows::AI::MachineLearning::implementation::D3DDeviceCache& device_cache, _In_ ID3D11Device* D3D11_device);

  Microsoft::WRL::ComPtr<ID3D11Texture2D> CreateTextureFromUnsupportedColorFormat(
      const winrt::Windows::Media::IVideoFrame& video_frame,
      const winrt::Windows::Graphics::Imaging::BitmapBounds& input_bounds,
      const winrt::Windows::Graphics::Imaging::BitmapBounds& output_bounds,
      winrt::Windows::Graphics::DirectX::DirectXPixelFormat new_format);

  static void CopyTextureIntoTexture(
      _In_ ID3D11Texture2D* texture_from,
      _In_ const winrt::Windows::Graphics::Imaging::BitmapBounds& input_bounds,
      _Inout_ ID3D11Texture2D* texture_to);
};
}  // namespace Windows::AI::MachineLearning::Internal