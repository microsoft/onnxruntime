// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/allocator.h"
#include <tvm/build_module.h>

namespace onnxruntime {
namespace nuphar {

// NupharRuntimeHandle holds necessary meta data from nuphar provider
struct NupharRuntimeHandle {
  bool allow_unaligned_buffers;
  const DLContext& dl_ctx;
  bool enable_model_parallelism;

  AllocatorPtr allocator;

  NupharRuntimeHandle(const DLContext& _dl_ctx)
      : dl_ctx(_dl_ctx) {}
};

}  // namespace nuphar
}  // namespace onnxruntime
