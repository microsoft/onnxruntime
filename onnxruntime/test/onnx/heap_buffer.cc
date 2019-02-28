// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "heap_buffer.h"

#include "core/session/onnxruntime_c_api.h"

void HeapBuffer::AddDeleter(OrtCallback* d) {
  if (d != nullptr && d->f != nullptr) deleters_.push_back(d);
}

HeapBuffer::~HeapBuffer() {
  for (auto d : deleters_) {
    if (d->f) d->f(d->param);
    delete d;
  }
}