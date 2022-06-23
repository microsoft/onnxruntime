// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/gpu_data_transfer.h"
#include "cuda_common.h"

// use default stream for copy for now, to avoid racing in BFC arena as in issue #4829
// note this may cause some models to run slower if there are ops running on CPU
// so we leave it as optional, in case user need the previous behavior
// a full fix to BFC arena is being looked at, and once it's in, we can revert this change
namespace onnxruntime {
GPUDataTransfer::GPUDataTransfer(cudaStream_t stream, bool do_copy_in_default_stream) {
  // create streams, default is nullptr
  do_copy_in_default_stream_ = do_copy_in_default_stream;
  streams_[kCudaStreamDefault] = stream;
  if (do_copy_in_default_stream) {
    streams_[kCudaStreamCopyIn] = stream;
    streams_[kCudaStreamCopyOut] = stream;
  } else {
    CUDA_CALL_THROW(cudaStreamCreateWithFlags(&streams_[kCudaStreamCopyIn], cudaStreamNonBlocking));
    CUDA_CALL_THROW(cudaStreamCreateWithFlags(&streams_[kCudaStreamCopyOut], cudaStreamNonBlocking));
  }
}

GPUDataTransfer::~GPUDataTransfer() {
  if (!do_copy_in_default_stream_ && streams_[kCudaStreamCopyIn] != nullptr) {
    CUDA_CALL(cudaStreamDestroy(streams_[kCudaStreamCopyIn]));
  }
  if (!do_copy_in_default_stream_ && streams_[kCudaStreamCopyOut] != nullptr) {
    CUDA_CALL(cudaStreamDestroy(streams_[kCudaStreamCopyOut]));
  }
}

bool GPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return src_device.Type() == OrtDevice::GPU || src_device.MemType() == OrtDevice::MemType::CUDA_PINNED ||
         dst_device.Type() == OrtDevice::GPU || dst_device.MemType() == OrtDevice::MemType::CUDA_PINNED;
}

common::Status GPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  // for the sync version of memcpy, launch to cuda default stream
  if (dst_device.Type() == OrtDevice::GPU) {
    if (src_device.Type() == OrtDevice::GPU) {
      // Copy only if the two addresses are different.
      if (dst_data != src_data) {
        CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice));
        CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(nullptr));
      }
    } else {
      // copy from other CPU memory to GPU, this is blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyHostToDevice));
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(nullptr));
    }
  } else if (src_device.Type() == OrtDevice::GPU) {
      // copying from GPU to CPU memory, this is blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToHost));
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(nullptr));
  } else {
    // copying between cpu memory
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}

common::Status GPUDataTransfer::CopyTensorAsync(const Tensor& src, Tensor& dst, Stream* stream) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();
  ORT_ENFORCE(stream && stream->provider && stream->provider->Type() == kCudaExecutionProvider);

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  if (dst_device.Type() == OrtDevice::GPU) {
    if (src_device.Type() == OrtDevice::CPU) {
      // copy from pinned memory to GPU, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream->handle)));
    } else if (src_device.Type() == OrtDevice::GPU) {
      // copying between GPU, this is non-blocking
      if (dst_data != src_data) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream->handle)));
      }
    }
  } else if (src_device.Type() == OrtDevice::GPU) {
    if (dst_device.Type() == OrtDevice::CPU) {
      // copying from GPU to pinned memory, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream->handle)));
    }
  } else {
    return ::onnxruntime::common::Status(::onnxruntime::common::ONNXRUNTIME,
                                         ::onnxruntime::common::FAIL,
                                         "Copy between CPU memories can't be async."); 
  }

  return Status::OK();
}

}  // namespace onnxruntime
