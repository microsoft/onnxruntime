// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "flatbuffers_utils.h"
#include "ort.fbs.h"

namespace onnxruntime {
namespace experimental {
namespace utils {

using namespace onnxruntime::experimental;

bool IsOrtFormatModelBytes(const void* bytes, int num_bytes) {
  return num_bytes > 8 &&  // check buffer is large enough to contain identifier so we don't read random memory
         fbs::InferenceSessionBufferHasIdentifier(bytes);
}

}  // namespace utils
}  // namespace experimental
}  // namespace onnxruntime