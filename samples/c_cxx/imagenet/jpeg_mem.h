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

// This file defines functions to compress and uncompress JPEG files
// to and from memory.  It provides interfaces for raw images
// (data array and size fields).
// Direct manipulation of JPEG strings are supplied: Flip, Rotate, Crop..

#pragma once

#include <functional>
#include <string>

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/types.h>
extern "C" {
#include "jerror.h"
#include "jpeglib.h"
}

using uint8 = std::uint8_t;
using int64 = std::int64_t;

// Flags for Uncompress
struct UncompressFlags {
  // ratio can be 1, 2, 4, or 8 and represent the denominator for the scaling
  // factor (eg ratio = 4 means that the resulting image will be at 1/4 original
  // size in both directions).
  int ratio = 1;

  // The number of bytes per pixel (1, 3 or 4), or 0 for autodetect.
  int components = 0;

  // If true, decoder will use a slower but nicer upscaling of the chroma
  // planes (yuv420/422 only).
  bool fancy_upscaling = true;

  // If true, will attempt to fill in missing lines of truncated files
  bool try_recover_truncated_jpeg = false;

  // The minimum required fraction of lines read before the image is accepted.
  float min_acceptable_fraction = 1.0;

  // The distance in bytes from one scanline to the other.  Should be at least
  // equal to width*components*sizeof(JSAMPLE).  If 0 is passed, the stride
  // used will be this minimal value.
  int stride = 0;

  // Setting of J_DCT_METHOD enum in jpeglib.h, for choosing which
  // algorithm to use for DCT/IDCT.
  //
  // Setting this has a quality/speed trade-off implication.
  J_DCT_METHOD dct_method = JDCT_DEFAULT;
};

// Uncompress some raw JPEG data given by the pointer srcdata and the length
// datasize.
// - width and height are the address where to store the size of the
//   uncompressed image in pixels.  May be nullptr.
// - components is the address where the number of read components are
//   stored.  This is *output only*: to request a specific number of
//   components use flags.components.  May be nullptr.
// - nwarn is the address in which to store the number of warnings.
//   May be nullptr.
// The function returns a pointer to the raw uncompressed data or NULL if
// there was an error. The caller of the function is responsible for
// freeing the memory (using delete []).
uint8* Uncompress(const void* srcdata, int datasize, const UncompressFlags& flags, int* width, int* height,
                  int* components,  // Output only: useful with autodetect
                  int64* nwarn);

// Version of Uncompress that allocates memory via a callback.  The callback
// arguments are (width, height, components).  If the size is known ahead of
// time this function can return an existing buffer; passing a callback allows
// the buffer to be shaped based on the JPEG header.  The caller is responsible
// for freeing the memory *even along error paths*.
uint8* Uncompress(const void* srcdata, int datasize, const UncompressFlags& flags, int64* nwarn,
                  std::function<uint8*(int, int, int)> allocate_output);
