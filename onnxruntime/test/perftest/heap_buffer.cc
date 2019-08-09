// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "heap_buffer.h"

HeapBuffer::~HeapBuffer() {
  for (void* p : buffers_) {
    free(p);
  }
}