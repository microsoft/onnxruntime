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

#pragma once

#include <stdint.h>
// Compute the interpolation indices only once.
struct CachedInterpolation {
  int64_t lower;  // Lower source index used in the interpolation
  int64_t upper;  // Upper source index used in the interpolation
  // 1-D linear interpolation scale (see:
  // https://en.wikipedia.org/wiki/Bilinear_interpolation)
  float lerp;
};
