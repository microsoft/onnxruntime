// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "heap_buffer.h"
#include "core/session/onnxruntime_c_api.h"

HeapBuffer::~HeapBuffer() {
  for (void* p : buffers_) {
    free(p);
  }
}