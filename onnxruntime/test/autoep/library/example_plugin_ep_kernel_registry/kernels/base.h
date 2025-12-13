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

  // Static functions assigned to the OrtKernelImpl fields:

  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;

  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;

  static OrtStatus* ORT_API_CALL PrePackWeightImpl(OrtKernelImpl* this_ptr, const OrtValue* tensor,
                                                   int input_index, OrtAllocator* alloc,
                                                   OrtSharedPrePackedWeightCache* prepacked_weight_cache,
                                                   /*out*/ bool* is_packed) noexcept;

  static OrtStatus* ORT_API_CALL SetSharedPrePackedWeightImpl(OrtKernelImpl* this_ptr,
                                                              const void* const* buffer_data_ptrs,
                                                              size_t num_buffers, int input_index) noexcept;

 private:
  // Methods that a derived class can override. Some are required (pure virtual) and others are optional
  // (have a default implementation):

  // Required.
  // DoCompute is called by BaseKernelImpl::ComputeImpl, which also catches exceptions thrown by DoCompute
  // implementations and converts them into OrtStatus*.
  virtual OrtStatus* DoCompute(OrtKernelContext* kernel_ctx) = 0;

  // Optional. The default implementation (BaseKernel::DoPrePackWeight) sets `is_packed` to false.
  // DoPrePackWeight is called by BaseKernelImpl::PrePackWeightImpl, which also catches exceptions thrown
  // by DoPrePackWeight.
  virtual OrtStatus* DoPrePackWeight(const OrtValue* tensor, int input_index, OrtAllocator* alloc,
                                     OrtSharedPrePackedWeightCache* prepacked_weight_cache, /*out*/ bool& is_packed);

  // Optional. The default implementation (BaseKernel::DoSetSharedPrePackedWeight) sets `used_shared_weight` to false.
  // DoSetSharedPrePackedWeight is called by BaseKernelImpl::SetSharedPrePackedWeightImpl, which also catches
  // exceptions thrown by DoSetSharedPrePackedWeight.
  virtual OrtStatus* DoSetSharedPrePackedWeight(const void* const* buffer_data_ptrs, size_t num_buffers,
                                                int input_index);

 protected:
  // Copies the source tensor into the destination tensor using the OrtDataTransferImpl defined by EP.
  OrtStatus* CopyTensor(Ort::ConstValue src_tensor, Ort::UnownedValue dst_tensor) noexcept;

  const OrtKernelInfo* info_;
  OrtDataTransferImpl* data_transfer_impl_;  // Custom state passed from OrtEp
};
