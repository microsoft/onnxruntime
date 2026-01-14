// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#include "../../plugin_ep_utils.h"
#include "../ep_allocator.h"

/// <summary>
/// An OrtKernelImpl class for binary element-wise operations.
/// Only Sub and Mul are supported currently.
/// </summary>
class BinaryOp : public OrtKernelImpl {
 private:
  struct PrivateTag {};

  struct PackedWeightInfo {
    Ort::ConstMemoryInfo mem_info{nullptr};
    std::vector<int64_t> shape;
    ONNXTensorElementDataType elem_type;
    size_t num_bytes;

    // Only one of the following data fields will be set.
    // If pre-packed data is shared with other kernels, `shared_data` will be non-null. Otherwise, this kernel
    // sets `owned_data`, whose lifetime it manages.
    AllocationUniquePtr owned_data{};
    const void* shared_data{nullptr};  // not owned by this kernel.
  };

 public:
  static OrtStatus* CreateKernelImpl(const OrtKernelInfo* info, void* state, /*out*/ OrtKernelImpl*& kernel) noexcept;
  BinaryOp(Ort::ConstKernelInfo info, void* state, PrivateTag);

  // Static functions assigned to the OrtKernelImpl fields:
  static OrtStatus* ORT_API_CALL ComputeImpl(OrtKernelImpl* this_ptr, OrtKernelContext* kernel_ctx) noexcept;
  static void ORT_API_CALL ReleaseImpl(OrtKernelImpl* this_ptr) noexcept;
  static OrtStatus* ORT_API_CALL PrePackWeightImpl(OrtKernelImpl* this_ptr, const OrtValue* tensor,
                                                   int input_index, OrtAllocator* alloc,
                                                   OrtSharedPrePackedWeightCache* prepacked_weight_cache,
                                                   /*out*/ bool* is_packed) noexcept;
  static OrtStatus* ORT_API_CALL SetSharedPrePackedWeightImpl(OrtKernelImpl* this_ptr,
                                                              const void* const* buffer_data_ptrs,
                                                              const size_t* buffer_data_sizes,
                                                              size_t num_buffers, int input_index) noexcept;

 private:
  Ort::ConstKernelInfo info_;
  OrtDataTransferImpl* data_transfer_impl_;  // Custom state passed from OrtEp
  std::optional<PackedWeightInfo> packed_weight_1_info_ = std::nullopt;
};
