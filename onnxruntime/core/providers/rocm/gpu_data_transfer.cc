// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/gpu_data_transfer.h"

// use default stream for copy for now, to avoid racing in BFC arena as in issue #4829
// note this may cause some models to run slower if there are ops running on CPU
// so we leave it as optional, in case user need the previous behavior
// a full fix to BFC arena is being looked at, and once it's in, we can revert this change
namespace onnxruntime {
GPUDataTransfer::GPUDataTransfer(hipStream_t stream, bool do_copy_in_default_stream) {
  // create streams, default is nullptr
  do_copy_in_default_stream_ = do_copy_in_default_stream;
  streams_[kHipStreamDefault] = stream;
  if (do_copy_in_default_stream) {
    streams_[kHipStreamCopyIn] = stream;
    streams_[kHipStreamCopyOut] = stream;
  } else {
    HIP_CALL_THROW(hipStreamCreateWithFlags(&streams_[kHipStreamCopyIn], hipStreamNonBlocking));
    HIP_CALL_THROW(hipStreamCreateWithFlags(&streams_[kHipStreamCopyOut], hipStreamNonBlocking));
  }
}

GPUDataTransfer::~GPUDataTransfer() {
  if (!do_copy_in_default_stream_ && streams_[kHipStreamCopyIn] != nullptr) {
    ORT_IGNORE_RETURN_VALUE(HIP_CALL(hipStreamDestroy(streams_[kHipStreamCopyIn])));
  }
  if (!do_copy_in_default_stream_ && streams_[kHipStreamCopyOut] != nullptr) {
    ORT_IGNORE_RETURN_VALUE(HIP_CALL(hipStreamDestroy(streams_[kHipStreamCopyOut])));
  }
}

bool GPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return src_device.Type() == OrtDevice::GPU || src_device.MemType() == OrtDevice::MemType::HIP_PINNED ||
         dst_device.Type() == OrtDevice::GPU || dst_device.MemType() == OrtDevice::MemType::HIP_PINNED;
}

common::Status GPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst, int exec_queue_id) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  if (dst_device.Type() == OrtDevice::GPU) {
    if (src_device.Type() == OrtDevice::CPU && src_device.MemType() == OrtDevice::MemType::HIP_PINNED) {
      // copy from pinned memory to GPU, this is non-blocking
      HIP_RETURN_IF_ERROR(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyHostToDevice, GetStream(exec_queue_id)));
    } else if (src_device.Type() == OrtDevice::GPU) {
      // copying between GPU, this is non-blocking
      // Copy only if the two addresses are different.
      if (dst_data != src_data) {
        HIP_RETURN_IF_ERROR(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyDeviceToDevice, GetStream(kHipStreamDefault)));
      }
    } else {
      // copy from other CPU memory to GPU, this is blocking
      HIP_RETURN_IF_ERROR(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyHostToDevice, GetStream(kHipStreamDefault)));
      HIP_RETURN_IF_ERROR(hipStreamSynchronize(GetStream(kHipStreamDefault)));
    }
  } else if (src_device.Type() == OrtDevice::GPU) {
    if (dst_device.Type() == OrtDevice::CPU && dst_device.MemType() == OrtDevice::MemType::HIP_PINNED) {
      // copying from GPU to pinned memory, this is non-blocking
      HIP_RETURN_IF_ERROR(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyDeviceToHost, GetStream(exec_queue_id)));
    } else {
      // copying from GPU to CPU memory, this is blocking
      HIP_RETURN_IF_ERROR(hipMemcpyAsync(dst_data, src_data, bytes, hipMemcpyDeviceToHost, GetStream(kHipStreamDefault)));
      HIP_RETURN_IF_ERROR(hipStreamSynchronize(GetStream(kHipStreamDefault)));
    }
  } else {
    // copying between cpu memory
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}
}  // namespace onnxruntime
