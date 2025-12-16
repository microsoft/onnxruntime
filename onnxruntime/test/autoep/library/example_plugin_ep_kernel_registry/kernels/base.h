// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../../plugin_ep_utils.h"

// Base class for kernel implementations.
//
// Note: BaseKernelImpl has virtual functions so care should be taken when casting BaseKernelImpl to a OrtKernelImpl,
// which is a C API struct type. Specifically, a static_cast or implicit cast should be used. A reinterpret_cast
// will result in an invalid object due to the presence of the vtable.
class BaseKernelImpl : public OrtKernelImpl {
 public:
  BaseKernelImpl(const OrtKernelInfo* info, void* state);
  virtual ~BaseKernelImpl() = default;

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

 private:
  // Derived classes implement DoCompute.
  // DoCompute is called by BaseKernelImpl::ComputeImpl, which also catches exceptions thrown by DoCompute
  // implementations and converts them into OrtStatus*.
  virtual OrtStatus* DoCompute(OrtKernelContext* kernel_ctx) = 0;

 protected:
  const OrtKernelInfo* info_;
  void* state_;  // Custom state passed from OrtEp
};
