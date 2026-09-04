// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"

#include "core/providers/cuda/gpu_data_transfer.h"
#include "cuda_common.h"

namespace onnxruntime {
namespace {

constexpr size_t kPinnedStagingBufferSize = 64 * 1024 * 1024;
constexpr size_t kPinnedStagingThreshold = 16 * 1024 * 1024;

class CudaDeviceGuard {
 public:
  common::Status SetDevice(int device_id) {
    CUDA_RETURN_IF_ERROR(cudaGetDevice(&previous_device_));
    if (previous_device_ != device_id) {
      CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id));
      restore_device_ = true;
    }
    return Status::OK();
  }

  ~CudaDeviceGuard() {
    if (restore_device_) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaSetDevice(previous_device_)));
    }
  }

 private:
  int previous_device_{-1};
  bool restore_device_{false};
};

}  // namespace

struct GPUDataTransfer::PinnedStagingState {
  explicit PinnedStagingState(int device) : device_id(device) {}

  int device_id;
  std::array<void*, 2> buffers{};
  std::array<cudaStream_t, 2> streams{};
};

GPUDataTransfer::GPUDataTransfer() = default;

GPUDataTransfer::~GPUDataTransfer() {
  ReleaseAllPinnedStaging();
}

common::Status GPUDataTransfer::EnsurePinnedStaging(int device_id, PinnedStagingState*& state) const {
  const auto existing = pinned_staging_by_device_.find(device_id);
  if (existing != pinned_staging_by_device_.end()) {
    state = existing->second.get();
    return Status::OK();
  }

  auto new_state = std::make_unique<PinnedStagingState>(device_id);

  for (size_t i = 0; i < new_state->buffers.size(); ++i) {
    auto status = CUDA_CALL(cudaMallocHost(&new_state->buffers[i], kPinnedStagingBufferSize));
    if (!status.IsOK()) {
      ReleasePinnedStaging(*new_state);
      return status;
    }

    status = CUDA_CALL(cudaStreamCreateWithFlags(&new_state->streams[i], cudaStreamNonBlocking));
    if (!status.IsOK()) {
      ReleasePinnedStaging(*new_state);
      return status;
    }
  }

  state = new_state.get();
  pinned_staging_by_device_.emplace(device_id, std::move(new_state));
  return Status::OK();
}

void GPUDataTransfer::ReleasePinnedStaging(PinnedStagingState& state) noexcept {
  int previous_device = -1;
  const bool restore_device =
      cudaGetDevice(&previous_device) == cudaSuccess &&
      previous_device != state.device_id &&
      cudaSetDevice(state.device_id) == cudaSuccess;

  for (auto& stream : state.streams) {
    if (stream != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamSynchronize(stream)));
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamDestroy(stream)));
      stream = nullptr;
    }
  }

  for (auto& buffer : state.buffers) {
    if (buffer != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaFreeHost(buffer)));
      buffer = nullptr;
    }
  }

  if (restore_device) {
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaSetDevice(previous_device)));
  }
}

void GPUDataTransfer::ReleaseAllPinnedStaging() const noexcept {
  for (auto& [device_id, state] : pinned_staging_by_device_) {
    ORT_UNUSED_PARAMETER(device_id);
    ReleasePinnedStaging(*state);
  }
  pinned_staging_by_device_.clear();
}

common::Status GPUDataTransfer::CopyHostToDeviceWithPinnedStaging(const void* src_data, void* dst_data,
                                                                  size_t bytes, int device_id) const {
  std::lock_guard<std::mutex> lock(pinned_staging_mutex_);
  CudaDeviceGuard device_guard;
  ORT_RETURN_IF_ERROR(device_guard.SetDevice(device_id));

  PinnedStagingState* state = nullptr;
  ORT_RETURN_IF_ERROR(EnsurePinnedStaging(device_id, state));

  const auto* src_bytes = static_cast<const uint8_t*>(src_data);
  auto* dst_bytes = static_cast<uint8_t*>(dst_data);
  std::array<bool, 2> stream_used{};

  for (size_t offset = 0, chunk_index = 0; offset < bytes; ++chunk_index) {
    const size_t staging_index = chunk_index % state->buffers.size();
    const size_t chunk_size = std::min(kPinnedStagingBufferSize, bytes - offset);
    cudaStream_t stream = state->streams[staging_index];

    if (stream_used[staging_index]) {
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));
    }

    memcpy(state->buffers[staging_index], src_bytes + offset, chunk_size);
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_bytes + offset, state->buffers[staging_index],
                                         chunk_size, cudaMemcpyHostToDevice, stream));
    stream_used[staging_index] = true;
    offset += chunk_size;
  }

  for (size_t i = 0; i < state->streams.size(); ++i) {
    if (stream_used[i]) {
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(state->streams[i]));
    }
  }

  return Status::OK();
}

bool GPUDataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  OrtDevice::DeviceType src_type = src_device.Type();
  OrtDevice::DeviceType dst_type = dst_device.Type();

  if ((src_type == OrtDevice::GPU && src_device.Vendor() != OrtDevice::VendorIds::NVIDIA) ||
      (dst_type == OrtDevice::GPU && dst_device.Vendor() != OrtDevice::VendorIds::NVIDIA)) {
    return false;
  }

  // copy must involve a GPU, and be device to device or cpu (exclude other device types)
  return (src_type == OrtDevice::GPU || dst_type == OrtDevice::GPU) &&
         (src_type == OrtDevice::GPU || src_type == OrtDevice::CPU) &&
         (dst_type == OrtDevice::GPU || dst_type == OrtDevice::CPU);
}

common::Status GPUDataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  size_t bytes = src.SizeInBytes();
  const void* src_data = src.DataRaw();
  void* dst_data = dst.MutableDataRaw();

  auto& src_device = src.Location().device;
  auto& dst_device = dst.Location().device;

  const bool dst_is_gpu_default = dst_device.Type() == OrtDevice::GPU &&
                                  dst_device.MemType() == OrtDevice::MemType::DEFAULT;
  const bool src_is_gpu_default = src_device.Type() == OrtDevice::GPU &&
                                  src_device.MemType() == OrtDevice::MemType::DEFAULT;

  // for the sync version of memcpy, launch to cuda default stream
  if (dst_is_gpu_default) {
    if (src_is_gpu_default) {
      // Copy only if the two addresses are different.
      if (dst_data != src_data) {
        CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice));
        // For device memory to device memory copy, no host-side synchronization is performed by cudaMemcpy.
        // see https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
        CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(nullptr));
      }
    } else {
      // copy from other CPU memory to GPU, this is blocking
      if (src_device.MemType() != OrtDevice::MemType::HOST_ACCESSIBLE &&
          bytes >= kPinnedStagingThreshold) {
        ORT_RETURN_IF_ERROR(
            CopyHostToDeviceWithPinnedStaging(src_data, dst_data, bytes, dst_device.Id()));
      } else {
        CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyHostToDevice));
      }
      if (src_device.MemType() != OrtDevice::MemType::HOST_ACCESSIBLE &&
          bytes < kPinnedStagingThreshold) {
        // For cudaMemcpy from pageable host memory to device memory, DMA to final destination may not have completed.
        // see https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html
        CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(nullptr));
      }
    }
  } else if (src_is_gpu_default) {
    // copying from GPU to CPU memory, this is blocking
    CUDA_RETURN_IF_ERROR(cudaMemcpy(dst_data, src_data, bytes, cudaMemcpyDeviceToHost));
  } else {
    // copying between cpu memory
    ORT_ENFORCE(dst_data != src_data);
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

  const bool dst_is_gpu_default = dst_device.Type() == OrtDevice::GPU &&
                                  dst_device.MemType() == OrtDevice::MemType::DEFAULT;
  const bool src_is_gpu_default = src_device.Type() == OrtDevice::GPU &&
                                  src_device.MemType() == OrtDevice::MemType::DEFAULT;

  if (dst_is_gpu_default) {
    if (src_is_gpu_default) {
      // copying between GPU, this is non-blocking
      if (dst_data != src_data) {
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToDevice,
                                             static_cast<cudaStream_t>(stream.GetHandle())));
      }
    } else {
      // copy from pinned or non-pinned CPU memory to GPU
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyHostToDevice,
                                           static_cast<cudaStream_t>(stream.GetHandle())));
    }
  } else if (src_is_gpu_default) {
    // copy from GPU to pinned or non-pinned CPU memory.
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dst_data, src_data, bytes, cudaMemcpyDeviceToHost,
                                         static_cast<cudaStream_t>(stream.GetHandle())));
  } else {
    if (src_device.MemType() == OrtDevice::MemType::HOST_ACCESSIBLE) {
      // sync the stream first to make sure the data arrived
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(static_cast<cudaStream_t>(stream.GetHandle())));
    }

    ORT_ENFORCE(dst_data != src_data);
    memcpy(dst_data, src_data, bytes);
  }

  return Status::OK();
}

}  // namespace onnxruntime
