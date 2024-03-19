// Originally from https://github.com/opencv/opencv/tree/4.x/modules/imgcodecs/src
// Modified to remove opencv dependencies

/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#pragma once

#include "vision/impl/image_encoder_decoder.hpp"

#include <cassert>
#include <vector>

namespace ort_extensions {

class PngDecoder : public BaseImageDecoder {
 public:
  static bool IsPng(const uint8_t* bytes, uint64_t num_bytes);

  PngDecoder(const uint8_t* bytes, uint64_t num_bytes)
      : BaseImageDecoder{bytes, num_bytes} {
    ReadHeader();
  }

  virtual ~PngDecoder();

  // ImageDecoder newDecoder() const CV_OVERRIDE;

 private:
  static void ReadDataFromBuf(void* png_ptr, uint8_t* dst, size_t size);

  bool ReadHeader();
  bool ReadData();

  bool DecodeImpl(uint8_t* output, uint64_t out_bytes) override;

  uint64_t cur_offset_{0};  // current read offset from bytes_

  int bit_depth_{0};
  int color_type_{0};

  // TODO: These are opaque assumably to keep png.h from being included here. not sure it's worth it.
  // Created in ReadHeder. Freed in dtor.
  void* png_ptr_{nullptr};   // pointer to decompression structure
  void* info_ptr_{nullptr};  // pointer to image information structure
  void* end_info_{nullptr};  // pointer to one more image information structure
};

class PngEncoder : public BaseImageEncoder {
 public:
  PngEncoder(const uint8_t* bytes, const std::vector<int64_t>& shape)
      : BaseImageEncoder(bytes, shape) {
  }

 private:
  bool EncodeImpl() override;

  static void WriteDataToBuf(void* png_ptr, uint8_t* src, size_t size);
  static void FlushBuf(void* png_ptr);

  uint64_t cur_offset_{0};  // current read offset from bytes_
};

}  // namespace ort_extensions
