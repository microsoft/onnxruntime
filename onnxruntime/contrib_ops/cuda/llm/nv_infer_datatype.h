/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>

// This is corresponding to nvinfer1 namespace used by TensorRT. Add it to avoid dependency on TensorRT.
namespace onnxruntime::llm::nvinfer {

enum class DataType : int32_t {
  //! 32-bit floating point format.
  kFLOAT = 0,

  //! IEEE 16-bit floating-point format -- has a 5 bit exponent and 11 bit significand.
  kHALF = 1,

  //! Signed 8-bit integer representing a quantized floating-point value.
  kINT8 = 2,

  //! Signed 32-bit integer format.
  kINT32 = 3,

  //! 8-bit boolean. 0 = false, 1 = true, other values undefined.
  kBOOL = 4,

  //! Unsigned 8-bit integer format.
  //! Cannot be used to represent quantized floating-point values.
  kUINT8 = 5,

  //! Signed 8-bit floating point with
  //! 1 sign bit, 4 exponent bits, 3 mantissa bits, and exponent-bias 7.
  kFP8 = 6,

  //! Brain float -- has an 8 bit exponent and 8 bit significand.
  kBF16 = 7,

  //! Signed 64-bit integer type.
  kINT64 = 8,

  //! Signed 4-bit integer type.
  kINT4 = 9,

  kFP4 = 10,
};
}  // namespace onnxruntime::llm::nvinfer
