// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ImageConverter.h"
#include "ImageConversionHelpers.h"
#include "ImageConversionTypes.h"

namespace _winml {

class VideoFrameToTensorConverter : public ImageConverter {
 public:
  VideoFrameToTensorConverter() : shared_handle_(nullptr) {}

  // Function takes in a VideoFrame backed by either a SoftwareBitmap or D3DSurface,
  // and converts to a tensor DX12 Resource.
  // CommandQueue and commandlist should be a compute resource,
  // commandlist will be passed in open, closed and executing when function exits
  // User should pass in a BitmapBounds describing the region of interest, in the form of
  // {upperleft X, upperleft Y, width, height} to be turned into a tensor.
  // If the region of interest is the entire VideoFrame, the input BitmapBounds should describe the entire image.
  void VideoFrameToDX12Tensor(
    _In_ const UINT32 batch_index,
    _In_ winml::LearningModelSession& session,
    _In_ const wm::IVideoFrame& input_video_frame,
    _In_ const wgi::BitmapBounds& input_bounds,
    _In_ const ImageTensorDescription& tensor_description,
    _Inout_ ID3D12Resource* output_tensor
  );

  // Function takes in a VideoFrame backed by either a SoftwareBitmap or D3DSurface,
  // and converts to a tensor returned in a buffer.
  // User should pass in a BitmapBounds describing the region of interest, in the form of
  // {upperleft X, upperleft Y, width, height} to be turned into a tensor.
  // If the region of interest is the entire VideoFrame, the input BitmapBounds should describe the entire image.
  void VideoFrameToSoftwareTensor(
    _In_ const wm::IVideoFrame& input_video_frame,
    _In_ const wgi::BitmapBounds& input_bounds,
    _In_ const ImageTensorDescription& tensor_description,
    _Out_ BYTE* output_CPU_tensor
  );

  void ConvertBuffersToBatchedGPUTensor(
    _In_ const std::vector<wss::IBuffer>& buffers,
    _In_ size_t buffer_size_in_bytes,
    _In_ _winml::D3DDeviceCache& device_cache,
    _Inout_ ID3D12Resource* output_resource
  );

 private:
  GUID d3d11_texture_GUID_ = {
    0x485e4bb3, 0x3fe8, 0x497b, {0x85, 0x9e, 0xc7, 0x5, 0x18, 0xdb, 0x11, 0x2a}
  };  // {485E4BB3-3FE8-497B-859E-C70518DB112A}
  GUID handle_GUID_ = {
    0xce43264e, 0x41f7, 0x4882, {0x9e, 0x20, 0xfa, 0xa5, 0x1e, 0x37, 0x64, 0xfc}
  };
  ;  // CE43264E-41F7-4882-9E20-FAA51E3764FC
  Microsoft::WRL::ComPtr<ID3D12Resource> upload_heap_;
  Microsoft::WRL::ComPtr<ID3D12Resource> input_D3D12_resource_;
  HANDLE shared_handle_;

  Microsoft::WRL::ComPtr<ID3D12Resource> ShareD3D11Texture(ID3D11Texture2D* pTexture, ID3D12Device* pDevice);

  void ConvertSoftwareBitmapToGPUTensor(
    _In_ const UINT32 batch_index,
    _In_ const wm::IVideoFrame& videoFrame,
    _In_ _winml::D3DDeviceCache& device_cache,
    _In_ const wgi::BitmapBounds& input_bounds,
    _In_ const ImageTensorDescription& tensor_description,
    _Inout_ ID3D12Resource* pOutputResource
  );

  void ConvertDX12TextureToGPUTensor(
    _In_ const UINT32 batch_index,
    _In_ ID3D12Resource* pInputResource,
    _In_ _winml::D3DDeviceCache& device_cache,
    _In_ const ImageTensorDescription& tensor_description,
    _Inout_ ID3D12Resource* output_resource
  );

  static D3D12_UNORDERED_ACCESS_VIEW_DESC CreateUAVDescription(
    const UINT32 batch_index, const D3D12_RESOURCE_DESC& resource_description, const ImageTensorDescription& description
  );

  static void ConvertSoftwareBitmapToCPUTensor(
    _In_ const wgi::SoftwareBitmap& software_bitmap,
    _In_ const ImageTensorDescription& tensor_description,
    _In_ const wgi::BitmapBounds& input_bounds,
    _Inout_ void* CPU_tensor
  );
};
}  // namespace _winml
