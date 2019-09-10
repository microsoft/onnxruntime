// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// A simple tool to test if the image resizing code works

#include "image_loader.h"
#include "CachedInterpolation.h"
#include <jpeglib.h>
#include "local_filesystem.h"
#include "jpeg_mem.h"

#include <png.h>

int main(int argc, char* argv[]) {
  std::string file_name(argv[1]);
  std::string output_file_name(argv[2]);
  int out_width = 299;
  int out_height = 299;

  int width;
  int height;
  int channels;

  UncompressFlags flags;
  flags.components = 3;
  // The TensorFlow-chosen default for jpeg decoding is IFAST, sacrificing
  // image quality for speed.
  flags.dct_method = JDCT_IFAST;
  size_t file_len;
  void* file_data;
  ReadFileAsString(file_name.c_str(), file_data, file_len);
  uint8_t* image_data = Uncompress(file_data, file_len, flags, &width, &height, &channels, nullptr);
  free(file_data);

  if (channels != 3) {
    std::ostringstream oss;
    oss << "input format error, expect 3 channels, got " << channels;
    throw std::runtime_error(oss.str());
  }

  std::vector<float> output_data(height * width * channels);

  ResizeImageInMemory((uint8_t*)image_data, output_data.data(), height, width, out_height, out_width, channels);
  delete[](uint8*) image_data;

  std::vector<png_byte> model_output_bytes(output_data.size());
  for (size_t i = 0; i != output_data.size(); ++i) {
    model_output_bytes[i] = (png_byte)(output_data[i]);
  }

  png_image image;
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  image.format = PNG_FORMAT_RGB;
  image.height = out_height;
  image.width = out_width;

  if (png_image_write_to_file(&image, output_file_name.c_str(), 0 /*convert_to_8bit*/, model_output_bytes.data(),
                              0 /*row_stride*/, nullptr /*colormap*/) == 0) {
    printf("write to '%s' failed:%s\n", output_file_name.c_str(), image.message);
    return -1;
  }

  return 0;
}