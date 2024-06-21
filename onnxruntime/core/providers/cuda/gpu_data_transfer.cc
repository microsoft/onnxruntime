// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"

#include "core/providers/cuda/gpu_data_transfer.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "cuda_common.h"
// #include "copy_impl.h"

namespace onnxruntime {
GPUDataTransfer::GPUDataTransfer() {}

GPUDataTransfer::~GPUDataTransfer() {}

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

common::Status GPUDataTransfer::CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  if (dst_device.Type() == OrtDevice::GPU) {
    if (src_device.Type() == OrtDevice::CPU) {
      // copy from pinned memory to GPU, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream.GetHandle())));
    } else if (src_device.Type() == OrtDevice::GPU) {
      // copying between GPU, this is non-blocking
      if (dst_data != src_data) {
        if (!src.IsContiguous() || !dst.IsContiguous()) {
          // const auto& src_stride_vec = src.Strides();
          const auto& dst_dims = dst.Shape().GetDims();
          // TensorPitches dst_pitches(src.Shape().GetDims());
          // TArray<int64_t> dst_stride_vec(input_pitches);
          std::vector<int64_t> input_strides(src.Strides().size());
          std::copy(src.Strides().begin(), src.Strides().end(), input_strides.begin());
          std::vector<int64_t> dst_dim_vec(dst_dims.size());
          std::copy(dst_dims.begin(), dst_dims.end(), dst_dim_vec.begin());
          std::cout << "111111111111111111111111" << std::endl;
          ORT_RETURN_IF_ERROR(cuda::StridedTensorCopyImpl(static_cast<cudaStream_t>(stream.GetHandle()),
                                                          src.DataType()->Size(), src.Shape().Size(),
                                                          src_data, dst_data, dst_dim_vec, input_strides));

          std::cout << "22222222222222" << std::endl;
        } else {
          CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice, static_cast<cudaStream_t>(stream.GetHandle())));
        }
      }
    }
  } else if (src_device.Type() == OrtDevice::GPU) {
    if (dst_device.Type() == OrtDevice::CPU) {
      // copying from GPU to pinned memory, this is non-blocking
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream.GetHandle())));
    }
  } else {
    if (src_device.MemType() == OrtDevice::MemType::CUDA_PINNED) {
      // sync the stream first to make sure the data arrived
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(static_cast<cudaStream_t>(stream.GetHandle())));
    }
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}

}  // namespace onnxruntime
