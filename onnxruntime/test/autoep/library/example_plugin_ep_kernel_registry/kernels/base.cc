// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "base.h"

BaseKernelImpl::BaseKernelImpl(const OrtKernelInfo* info, void* state)
    : info_{info},
      data_transfer_impl_{reinterpret_cast<OrtDataTransferImpl*>(state)} {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
  PrePackConstantTensor = PrePackConstantTensorImpl;
}

OrtStatus* BaseKernelImpl::DoPrePackConstantTensor(const OrtValue* /*tensor*/, int /*input_index*/,
                                                   OrtAllocator* /*alloc*/, /*out*/ bool& is_packed) {
  // Default implementation that does not pack weights
  is_packed = false;
  return nullptr;
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

/*static*/
OrtStatus* ORT_API_CALL BaseKernelImpl::PrePackConstantTensorImpl(OrtKernelImpl* this_ptr, const OrtValue* tensor,
                                                                  int input_index, OrtAllocator* alloc,
                                                                  /*out*/ bool* is_packed) noexcept {
  try {
    BaseKernelImpl* base_kernel = static_cast<BaseKernelImpl*>(this_ptr);
    return base_kernel->DoPrePackConstantTensor(tensor, input_index, alloc, *is_packed);
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }
}
