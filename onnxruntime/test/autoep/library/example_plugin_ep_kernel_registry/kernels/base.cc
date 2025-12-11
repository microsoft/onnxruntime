// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "base.h"

BaseKernelImpl::BaseKernelImpl(const OrtKernelInfo* info, void* state)
    : OrtKernelImpl{},  // Initialize all OrtKernelImpl functions to NULL
      info_{info},
      data_transfer_impl_{reinterpret_cast<OrtDataTransferImpl*>(state)} {
  ort_version_supported = ORT_API_VERSION;
  Compute = ComputeImpl;
  Release = ReleaseImpl;
  PrePackWeight = PrePackWeightImpl;
  SetSharedPrePackedWeight = SetSharedPrePackedWeightImpl;
}

// Default implementation that does not pack weights.
OrtStatus* BaseKernelImpl::DoPrePackWeight(const OrtValue* /*tensor*/, int /*input_index*/,
                                           OrtAllocator* /*alloc*/, OrtSharedPrePackedWeightCache* /*weight_cache*/,
                                           /*out*/ bool& is_packed) {
  is_packed = false;
  return nullptr;
}

// Default implementation that does not use shared weights.
OrtStatus* BaseKernelImpl::DoSetSharedPrePackedWeight(const void* const* /*buffer_data_ptrs*/, size_t /*num_buffers*/,
                                                      int /*input_index*/) {
  return nullptr;
}

OrtStatus* BaseKernelImpl::CopyTensor(Ort::ConstValue src_tensor, Ort::UnownedValue dst_tensor) noexcept {
  const OrtMemoryDevice* src_device = Ort::GetEpApi().MemoryInfo_GetMemoryDevice(src_tensor.GetTensorMemoryInfo());
  const OrtMemoryDevice* dst_device = Ort::GetEpApi().MemoryInfo_GetMemoryDevice(dst_tensor.GetTensorMemoryInfo());

  RETURN_IF(!data_transfer_impl_->CanCopy(data_transfer_impl_, src_device, dst_device), Ort::GetApi(),
            "OrtDataTransferImpl cannot copy src tensor to dst tensor.");

  auto src_type_shape = src_tensor.GetTensorTypeAndShapeInfo();
  auto dst_type_shape = dst_tensor.GetTensorTypeAndShapeInfo();
  bool same_elem_type = src_type_shape.GetElementType() == dst_type_shape.GetElementType();
  bool same_elem_count = src_type_shape.GetElementCount() == dst_type_shape.GetElementCount();
  RETURN_IF(!same_elem_type || !same_elem_count, Ort::GetApi(), "Cannot copy tensors of different types or size.");

  std::array<const OrtValue*, 1> src_tensors = {src_tensor};
  std::array<OrtValue*, 1> dst_tensors = {dst_tensor};

  RETURN_IF_ERROR(data_transfer_impl_->CopyTensors(data_transfer_impl_, src_tensors.data(), dst_tensors.data(),
                                                   /*streams*/ nullptr, src_tensors.size()));

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
OrtStatus* ORT_API_CALL BaseKernelImpl::PrePackWeightImpl(OrtKernelImpl* this_ptr, const OrtValue* tensor,
                                                          int input_index, OrtAllocator* alloc,
                                                          OrtSharedPrePackedWeightCache* prepacked_weight_cache,
                                                          /*out*/ bool* is_packed) noexcept {
  try {
    BaseKernelImpl* base_kernel = static_cast<BaseKernelImpl*>(this_ptr);
    return base_kernel->DoPrePackWeight(tensor, input_index, alloc, prepacked_weight_cache, *is_packed);
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }
}

/*static*/
OrtStatus* ORT_API_CALL BaseKernelImpl::SetSharedPrePackedWeightImpl(OrtKernelImpl* this_ptr,
                                                                     const void* const* buffer_data_ptrs,
                                                                     size_t num_buffers, int input_index) noexcept {
  try {
    BaseKernelImpl* base_kernel = static_cast<BaseKernelImpl*>(this_ptr);
    return base_kernel->DoSetSharedPrePackedWeight(buffer_data_ptrs, num_buffers, input_index);
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  }
}
