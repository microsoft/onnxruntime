// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "image_loader.h"
#include <sstream>
#include <wincodec.h>
#include <wincodecsdk.h>
#include <atlbase.h>

bool CreateImageLoader(void** out) {
  IWICImagingFactory* piFactory;
  auto hr = CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&piFactory));
  if (!SUCCEEDED(hr)) return false;
  *out = piFactory;
  return true;
}

void ReleaseImageLoader(void* p) {
  auto piFactory = reinterpret_cast<IWICImagingFactory*>(p);
  piFactory->Release();
}

template <typename T>
static void PrintErrorDescription(HRESULT hr, std::basic_ostringstream<T>& oss) {
  if (FACILITY_WINDOWS == HRESULT_FACILITY(hr)) hr = HRESULT_CODE(hr);
  TCHAR* szErrMsg;

  if (FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM, NULL, hr,
                    MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&szErrMsg, 0, NULL) != 0) {
    oss << szErrMsg;
    LocalFree(szErrMsg);
  } else {
    oss << TEXT("[Could not find a description for error # ") << hr;
  }
}

OrtStatus* LoadImageFromFileAndCrop(void* loader, const ORTCHAR_T* filename, double central_crop_fraction, float** out,
                                    int* out_width, int* out_height) {
  auto piFactory = reinterpret_cast<IWICImagingFactory*>(loader);
  const int channels = 3;
  try {
    CComPtr<IWICBitmapDecoder> piDecoder;
    ATLENSURE_SUCCEEDED(
        piFactory->CreateDecoderFromFilename(filename, NULL, GENERIC_READ,
                                             WICDecodeMetadataCacheOnDemand,  // defer parsing non-critical metadata
                                             &piDecoder));

    UINT count = 0;
    ATLENSURE_SUCCEEDED(piDecoder->GetFrameCount(&count));
    if (count != 1) {
      return Ort::GetApi().CreateStatus(ORT_FAIL, "The image has multiple frames, I don't know which to choose");
    }

    CComPtr<IWICBitmapFrameDecode> piFrameDecode;
    ATLENSURE_SUCCEEDED(piDecoder->GetFrame(0, &piFrameDecode));
    UINT width, height;
    ATLENSURE_SUCCEEDED(piFrameDecode->GetSize(&width, &height));
    CComPtr<IWICFormatConverter> ppIFormatConverter;
    ATLENSURE_SUCCEEDED(piFactory->CreateFormatConverter(&ppIFormatConverter));
    ATLENSURE_SUCCEEDED(ppIFormatConverter->Initialize(piFrameDecode,                // Source frame to convert
                                                       GUID_WICPixelFormat24bppRGB,  // The desired pixel format
                                                       WICBitmapDitherTypeNone,      // The desired dither pattern
                                                       NULL,                         // The desired palette
                                                       0.f,                          // The desired alpha threshold
                                                       WICBitmapPaletteTypeCustom    // Palette translation type
                                                       ));
    int bbox_h_start =
        static_cast<int>((static_cast<double>(height) - static_cast<double>(height) * central_crop_fraction) / 2);
    int bbox_w_start =
        static_cast<int>((static_cast<double>(width) - static_cast<double>(width) * central_crop_fraction) / 2);
    int bbox_h_size = height - bbox_h_start * 2;
    int bbox_w_size = width - bbox_w_start * 2;
    UINT stride = bbox_w_size * channels;
    UINT result_buffer_size = bbox_h_size * bbox_w_size * channels;
    // TODO: check result_buffer_size <= UNIT_MAX
    std::vector<uint8_t> data(result_buffer_size);
    WICRect rect;
    memset(&rect, 0, sizeof(WICRect));
    rect.X = bbox_w_start;
    rect.Y = bbox_h_start;
    rect.Height = bbox_h_size;
    rect.Width = bbox_w_size;

    ATLENSURE_SUCCEEDED(ppIFormatConverter->CopyPixels(&rect, stride, static_cast<UINT>(data.size()), data.data()));
    float* float_file_data = (float*)malloc(data.size() * sizeof(float));
    size_t len = data.size();
    for (size_t i = 0; i != len; ++i) {
      float_file_data[i] = static_cast<float>(data[i]) / 255;
    }

    *out = float_file_data;
    *out_width = bbox_w_size;
    *out_height = bbox_h_size;
    return nullptr;
  } catch (const std::exception& ex) {
    std::ostringstream oss;
    oss << "Load " << filename << " failed:" << ex.what();
    return Ort::GetApi().CreateStatus(ORT_FAIL, oss.str().c_str());
  } catch (const CAtlException& ex) {
    std::ostringstream oss;
    oss << "Load " << filename << " failed:";
    PrintErrorDescription(ex.m_hr, oss);
    return Ort::GetApi().CreateStatus(ORT_FAIL, oss.str().c_str());
  }
}
