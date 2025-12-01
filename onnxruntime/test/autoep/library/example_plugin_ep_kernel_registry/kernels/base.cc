// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "base.h"

BaseKernelImpl::BaseKernelImpl(const OrtKernelInfo* info, void* state) : info_{info}, state_{state} {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
}

/*static*/
OrtStatus* ORT_API_CALL BaseKernelImpl::ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept {
  BaseKernelImpl* base_kernel = static_cast<BaseKernelImpl*>(this_ptr);
  return base_kernel->DoCompute(kernel_ctx);
}

/*static*/
void ORT_API_CALL BaseKernelImpl::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<BaseKernelImpl*>(this_ptr);
}
