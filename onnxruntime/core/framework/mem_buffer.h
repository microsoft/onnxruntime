// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"

namespace onnxruntime {
/**
 * A simple POD for using with tensor deserialization
 */
class MemBuffer {
 public:
  MemBuffer(void* buffer, size_t len, const OrtAllocatorInfo& alloc_info)
      : buffer_(buffer), len_(len), alloc_info_(alloc_info) {}
  void* GetBuffer() const { return buffer_; }

  size_t GetLen() const { return len_; }
  const OrtAllocatorInfo& GetAllocInfo() const { return alloc_info_; }

 private:
  void* const buffer_;
  const size_t len_;
  const OrtAllocatorInfo& alloc_info_;
};
};  // namespace onnxruntime