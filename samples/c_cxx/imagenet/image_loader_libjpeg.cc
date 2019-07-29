// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "image_loader.h"
#include <jpeglib.h>
#include "jpeg_mem.h"
#include "local_filesystem.h"
#include <assert.h>

bool CreateImageLoader(void** out) {
  *out = nullptr;
  return true;
}

void ReleaseImageLoader(void*) {}

OrtStatus* LoadImageFromFileAndCrop(void*, const ORTCHAR_T* filename, double central_crop_fraction, float** out,
                                    int* out_width, int* out_height) {
  const int channels_ = 3;
  UncompressFlags flags;
  flags.components = channels_;
  // The TensorFlow-chosen default for jpeg decoding is IFAST, sacrificing
  // image quality for speed.
  flags.dct_method = JDCT_IFAST;
  size_t file_len;
  void* file_data;
  ReadFileAsString(filename, file_data, file_len);
  int width;
  int height;
  int channels;
  std::unique_ptr<uint8_t[]> decompressed_image(
      Uncompress(file_data, static_cast<int>(file_len), flags, &width, &height, &channels, nullptr));
  free(file_data);

  if (decompressed_image == nullptr) {
    std::ostringstream oss;
    oss << "decompress '" << filename << "' failed";
    return OrtCreateStatus(ORT_FAIL, oss.str().c_str());
  }

  if (channels != channels_) {
    std::ostringstream oss;
    oss << "input format error, expect 3 channels, got " << channels;
    return OrtCreateStatus(ORT_FAIL, oss.str().c_str());
  }

  // cast uint8 to float
  // See: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/image_ops_impl.py of
  // tf.image.convert_image_dtype

  // crop it, and cast each pixel value from uint8 to float in range of [0,1]
  // TODO: should the result be in range of [0,1) or [0,1]?

  int bbox_h_start =
      static_cast<int>((static_cast<double>(height) - static_cast<double>(height) * central_crop_fraction) / 2);
  int bbox_w_start =
      static_cast<int>((static_cast<double>(width) - static_cast<double>(width) * central_crop_fraction) / 2);
  int bbox_h_size = height - bbox_h_start * 2;
  int bbox_w_size = width - bbox_w_start * 2;
  const size_t ele_count = bbox_h_size * bbox_w_size * channels;
  float* float_file_data = (float*)malloc(ele_count * sizeof(float));
  if (float_file_data == nullptr) {
    return OrtCreateStatus(ORT_FAIL, "out of memory");
  }

  {
    auto p = decompressed_image.get() + (bbox_h_start * width + bbox_w_start) * channels;

    size_t len = bbox_w_size * channels;
    float* wptr = float_file_data;
    for (int i = 0; i != bbox_h_size; ++i) {
      for (int j = 0; j != len; ++j) {
        // TODO: should it be divided by 255 or 256?
        *wptr++ = static_cast<float>(p[j]) / 255;
      }
      p += width * channels;
    }
    assert(wptr == float_file_data + ele_count);
  }

  *out = float_file_data;
  *out_width = bbox_w_size;
  *out_height = bbox_h_size;
  return nullptr;
}
