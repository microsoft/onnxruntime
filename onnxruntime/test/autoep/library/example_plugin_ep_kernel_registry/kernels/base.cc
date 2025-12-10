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
  try {
    BaseKernelImpl* base_kernel = static_cast<BaseKernelImpl*>(this_ptr);
    return base_kernel->DoCompute(kernel_ctx);
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }
}

/*static*/
void ORT_API_CALL BaseKernelImpl::ReleaseImpl(OrtKernelImpl* this_ptr) noexcept {
  delete static_cast<BaseKernelImpl*>(this_ptr);
}
