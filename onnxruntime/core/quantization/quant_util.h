// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/mlas/inc/mlas.h"
#include "core/framework/allocator.h"
#include "core/framework/buffer_deleter.h"

namespace onnxruntime {
namespace quantization {
// Transpose the input and store it to a new allocated buffer
inline uint8_t* TransPoseInputData(const uint8_t* input, BufferUniquePtr& buffer_holder, AllocatorPtr& allocator, size_t M, size_t N) {
  uint8_t* output = static_cast<uint8_t*>(allocator->Alloc(M * N * sizeof(uint8_t)));
  MlasTranspose(input, output, M, N);
  buffer_holder.reset(output);
  return output;
}

}  // namespace quantization
}  // namespace onnxruntime
