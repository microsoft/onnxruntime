// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/allocator.h"

// Placeholder for an MKL allocators
namespace onnxruntime {
constexpr const char* MKLDNN = "MklDnn";
constexpr const char* MKLDNN_CPU = "MklDnnCpu";

class MKLDNNAllocator : public CPUAllocator {
 public:
  const OrtAllocatorInfo& Info() const override;
};
class MKLDNNCPUAllocator : public CPUAllocator {
 public:
  const OrtAllocatorInfo& Info() const override;
};
}  // namespace onnxruntime
