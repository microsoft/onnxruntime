// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace webgpu {

// Quantized cache layout per head:
//   one fp32 scale followed by head_size * bit_width packed bits.
// Callers that preallocate present_key/present_value must express this byte
// span in the output tensor's element type because shape inference reports
// the model's uncompressed head size.
constexpr int KvCacheQuantizedHeadSizeU32(int head_size, uint32_t bit_width) {
  return 1 + head_size * static_cast<int>(bit_width) / 32;
}

constexpr int64_t KvCacheQuantizedHeadSize(int head_size, uint32_t bit_width,
                                           size_t element_size) {
  return static_cast<int64_t>(KvCacheQuantizedHeadSizeU32(head_size, bit_width)) * 4 /
         static_cast<int64_t>(element_size);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
