/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#ifdef HAVE_JPEG
#include <jpeglib.h>
#endif
#include <assert.h>
#include "image_loader.h"
#include "cached_interpolation.h"
#include "local_filesystem.h"
#ifdef HAVE_JPEG
#include "jpeg_mem.h"
#endif

namespace {
/**
 * CalculateResizeScale determines the float scaling factor.
 * @param in_size
 * @param out_size
 * @param align_corners If true, the centers of the 4 corner pixels of the input and output tensors are aligned,
 *                        preserving the values at the corner pixels
 * @return
 */
inline float CalculateResizeScale(int64_t in_size, int64_t out_size, bool align_corners) {
  return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float>(out_size - 1)
                                         : in_size / static_cast<float>(out_size);
}

inline void compute_interpolation_weights(const int64_t out_size, const int64_t in_size, const float scale,
                                          CachedInterpolation* interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (int64_t i = out_size - 1; i >= 0; --i) {
    const float in = i * scale;
    interpolation[i].lower = static_cast<int64_t>(in);
    interpolation[i].upper = std::min(interpolation[i].lower + 1, in_size - 1);
    interpolation[i].lerp = in - interpolation[i].lower;
  }
}

/**
 * Computes the bilinear interpolation from the appropriate 4 float points
 * and the linear interpolation weights.
 */
inline float compute_lerp(const float top_left, const float top_right, const float bottom_left,
                          const float bottom_right, const float x_lerp, const float y_lerp) {
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

}  // namespace
template <typename T>
void ResizeImageInMemory(const T* input_data, float* output_data, int in_height, int in_width, int out_height,
                         int out_width, int channels) {
  float height_scale = CalculateResizeScale(in_height, out_height, false);
  float width_scale = CalculateResizeScale(in_width, out_width, false);

  std::vector<CachedInterpolation> ys(out_height + 1);
  std::vector<CachedInterpolation> xs(out_width + 1);

  // Compute the cached interpolation weights on the x and y dimensions.
  compute_interpolation_weights(out_height, in_height, height_scale, ys.data());
  compute_interpolation_weights(out_width, in_width, width_scale, xs.data());

  // Scale x interpolation weights to avoid a multiplication during iteration.
  for (int i = 0; i < xs.size(); ++i) {
    xs[i].lower *= channels;
    xs[i].upper *= channels;
  }

  const int64_t in_row_size = in_width * channels;
  const int64_t in_batch_num_values = in_height * in_row_size;
  const int64_t out_row_size = out_width * channels;

  const T* input_b_ptr = input_data;
  float* output_y_ptr = output_data;
  const int batch_size = 1;

  if (channels == 3) {
    for (int b = 0; b < batch_size; ++b) {
      for (int64_t y = 0; y < out_height; ++y) {
        const T* ys_input_lower_ptr = input_b_ptr + ys[y].lower * in_row_size;
        const T* ys_input_upper_ptr = input_b_ptr + ys[y].upper * in_row_size;
        const float ys_lerp = ys[y].lerp;
        for (int64_t x = 0; x < out_width; ++x) {
          const int64_t xs_lower = xs[x].lower;
          const int64_t xs_upper = xs[x].upper;
          const float xs_lerp = xs[x].lerp;

          // Read channel 0.
          const float top_left0(ys_input_lower_ptr[xs_lower + 0]);
          const float top_right0(ys_input_lower_ptr[xs_upper + 0]);
          const float bottom_left0(ys_input_upper_ptr[xs_lower + 0]);
          const float bottom_right0(ys_input_upper_ptr[xs_upper + 0]);

          // Read channel 1.
          const float top_left1(ys_input_lower_ptr[xs_lower + 1]);
          const float top_right1(ys_input_lower_ptr[xs_upper + 1]);
          const float bottom_left1(ys_input_upper_ptr[xs_lower + 1]);
          const float bottom_right1(ys_input_upper_ptr[xs_upper + 1]);

          // Read channel 2.
          const float top_left2(ys_input_lower_ptr[xs_lower + 2]);
          const float top_right2(ys_input_lower_ptr[xs_upper + 2]);
          const float bottom_left2(ys_input_upper_ptr[xs_lower + 2]);
          const float bottom_right2(ys_input_upper_ptr[xs_upper + 2]);

          // Compute output.
          output_y_ptr[x * channels + 0] =
              compute_lerp(top_left0, top_right0, bottom_left0, bottom_right0, xs_lerp, ys_lerp);
          output_y_ptr[x * channels + 1] =
              compute_lerp(top_left1, top_right1, bottom_left1, bottom_right1, xs_lerp, ys_lerp);
          output_y_ptr[x * channels + 2] =
              compute_lerp(top_left2, top_right2, bottom_left2, bottom_right2, xs_lerp, ys_lerp);
        }
        output_y_ptr += out_row_size;
      }
      input_b_ptr += in_batch_num_values;
    }
  } else {
    for (int b = 0; b < batch_size; ++b) {
      for (int64_t y = 0; y < out_height; ++y) {
        const T* ys_input_lower_ptr = input_b_ptr + ys[y].lower * in_row_size;
        const T* ys_input_upper_ptr = input_b_ptr + ys[y].upper * in_row_size;
        const float ys_lerp = ys[y].lerp;
        for (int64_t x = 0; x < out_width; ++x) {
          auto xs_lower = xs[x].lower;
          auto xs_upper = xs[x].upper;
          auto xs_lerp = xs[x].lerp;
          for (int c = 0; c < channels; ++c) {
            const float top_left(ys_input_lower_ptr[xs_lower + c]);
            const float top_right(ys_input_lower_ptr[xs_upper + c]);
            const float bottom_left(ys_input_upper_ptr[xs_lower + c]);
            const float bottom_right(ys_input_upper_ptr[xs_upper + c]);
            output_y_ptr[x * channels + c] =
                compute_lerp(top_left, top_right, bottom_left, bottom_right, xs_lerp, ys_lerp);
          }
        }
        output_y_ptr += out_row_size;
      }
      input_b_ptr += in_batch_num_values;
    }
  }
}

template void ResizeImageInMemory(const float* input_data, float* output_data, int in_height, int in_width,
                                  int out_height, int out_width, int channels);

template void ResizeImageInMemory(const uint8_t* input_data, float* output_data, int in_height, int in_width,
                                  int out_height, int out_width, int channels);

InceptionPreprocessing::InceptionPreprocessing(int out_height, int out_width, int channels)
    : out_height_(out_height), out_width_(out_width), channels_(channels) {
  if (!CreateImageLoader(&image_loader_)) {
    throw std::runtime_error("create image loader failed");
  }
}

#ifdef HAVE_JPEG
#else
bool CreateImageLoader(void** out) {
  IWICImagingFactory* piFactory;
  auto hr = CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&piFactory));
  if (!SUCCEEDED(hr)) return false;
  *out = piFactory;
  return true;
}

void ReleaseImageLoader(void* p){
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
    assert(count == 1);

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
  } catch (std::exception& ex) {
    std::ostringstream oss;
    oss << "Load " << filename << " failed:" << ex.what();
    return OrtCreateStatus(ORT_FAIL, oss.str().c_str());
  } catch (const CAtlException& ex) {
    std::ostringstream oss;
    oss << "Load " << filename << " failed:";
    PrintErrorDescription(ex.m_hr, oss);
    return OrtCreateStatus(ORT_FAIL, oss.str().c_str());
  }
}
#endif

// see: https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py
// function: preprocess_for_eval
void InceptionPreprocessing::operator()(const void* input_data, void* output_data) const {
  const TCharString& file_name = *reinterpret_cast<const TCharString*>(input_data);
#ifdef HAVE_JPEG
  UncompressFlags flags;
  flags.components = channels_;
  // The TensorFlow-chosen default for jpeg decoding is IFAST, sacrificing
  // image quality for speed.
  flags.dct_method = JDCT_IFAST;
  size_t file_len;
  void* file_data;
  ReadFileAsString(file_name.c_str(), file_data, file_len);
  int width;
  int height;
  int channels;
  std::unique_ptr<uint8_t[]> decompressed_image(
      Uncompress(file_data, static_cast<int>(file_len), flags, &width, &height, &channels, nullptr));
  free(file_data);

  if (decompressed_image == nullptr) {
    std::ostringstream oss;
    oss << "decompress '" << file_name.c_str() << "' failed";
    throw std::runtime_error(oss.str());
  }

  if (channels != channels_) {
    std::ostringstream oss;
    oss << "input format error, expect 3 channels, got " << channels;
    throw std::runtime_error(oss.str());
  }

  // cast uint8 to float
  // See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/image_ops_impl.py of
  // tf.image.convert_image_dtype

  // crop it, and cast each pixel value from uint8 to float in range of [0,1]
  // TODO: should the result be in range of [0,1) or [0,1]?

  int bbox_h_start =
      static_cast<int>((static_cast<double>(height) - static_cast<double>(height) * central_fraction_) / 2);
  int bbox_w_start =
      static_cast<int>((static_cast<double>(width) - static_cast<double>(width) * central_fraction_) / 2);
  int bbox_h_size = height - bbox_h_start * 2;
  int bbox_w_size = width - bbox_w_start * 2;
  std::vector<float> float_file_data(bbox_h_size * bbox_w_size * channels);
  {
    auto p = decompressed_image.get() + (bbox_h_start * width + bbox_w_start) * channels;

    size_t len = bbox_w_size * channels;
    float* wptr = float_file_data.data();
    for (int i = 0; i != bbox_h_size; ++i) {
      for (int j = 0; j != len; ++j) {
        // TODO: should it be divided by 255 or 256?
        *wptr++ = static_cast<float>(p[j]) / 255;
      }
      p += width * channels;
    }
    assert(wptr == float_file_data.data() + float_file_data.size());
  }
  float* float_file_data_pointer = float_file_data.data();
#else
  float* float_file_data_pointer;
  int bbox_h_size, bbox_w_size;
  int channels = 3;
  ORT_THROW_ON_ERROR(LoadImageFromFileAndCrop(nullptr, file_name.c_str(), central_fraction_, &float_file_data_pointer,
                                              &bbox_w_size, &bbox_h_size));
#endif
  auto output_data_ = reinterpret_cast<float*>(output_data);
  ResizeImageInMemory(float_file_data_pointer, output_data_, bbox_h_size, bbox_w_size, out_height_, out_width_,
                      channels);
  size_t output_data_len = channels_ * out_height_ * out_width_;
  for (size_t i = 0; i != output_data_len; ++i) {
    output_data_[i] = (output_data_[i] - 0.5f) * 2.f;
  }
}
