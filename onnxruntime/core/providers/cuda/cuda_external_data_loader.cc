// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// provider_api.h must be first to set SHARED_PROVIDER
#include "core/providers/shared_library/provider_api.h"

#include "core/providers/cuda/cuda_external_data_loader.h"

#include <algorithm>
#include <bit>
#include <future>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {
namespace {

constexpr size_t kBufferSize = 64 * 1024 * 1024;
constexpr size_t kParallelReadThreshold = 16 * 1024 * 1024;
constexpr size_t kReaderCount = 4;

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

common::Status ReadChunk(const Env& env, const std::filesystem::path& path,
                         FileOffsetType offset, size_t length, void* buffer) {
  const size_t reader_count = length >= kParallelReadThreshold ? kReaderCount : 1;
  std::vector<std::future<common::Status>> reads;
  reads.reserve(reader_count);

  for (size_t reader = 0; reader < reader_count; ++reader) {
    const size_t begin = length * reader / reader_count;
    const size_t end = length * (reader + 1) / reader_count;
    reads.emplace_back(std::async(std::launch::async, [&env, &path, offset, begin, end, buffer]() {
      return env.ReadFileIntoBuffer(path.native().c_str(), offset + begin, end - begin,
                                    gsl::span<char>{static_cast<char*>(buffer) + begin, end - begin});
    }));
  }

  for (auto& read : reads) {
    ORT_RETURN_IF_ERROR(read.get());
  }

  return Status::OK();
}

void SwapByteOrderInplace(void* buffer, size_t length, size_t element_size) {
  auto* bytes = static_cast<uint8_t*>(buffer);
  for (size_t offset = 0; offset < length; offset += element_size) {
    std::reverse(bytes + offset, bytes + offset + element_size);
  }
}

}  // namespace

ExternalDataLoader::ExternalDataLoader(int device_id) : device_id_(device_id) {}

ExternalDataLoader::~ExternalDataLoader() {
  ReleaseResources();
}

bool ExternalDataLoader::CanLoad(const OrtMemoryInfo& target_memory_info) const {
  const OrtDevice& device = target_memory_info.device;
  return device.Type() == OrtDevice::GPU &&
         device.MemType() == OrtDevice::MemType::DEFAULT &&
         device.Vendor() == OrtDevice::VendorIds::NVIDIA &&
         device.Id() == device_id_;
}

common::Status ExternalDataLoader::EnsureResources() const {
  if (buffers_[0] != nullptr) {
    return Status::OK();
  }

  for (size_t i = 0; i < buffers_.size(); ++i) {
    auto status = CUDA_CALL(cudaMallocHost(&buffers_[i], kBufferSize));
    if (!status.IsOK()) {
      ReleaseResources();
      return status;
    }

    status = CUDA_CALL(cudaStreamCreateWithFlags(&streams_[i], cudaStreamNonBlocking));
    if (!status.IsOK()) {
      ReleaseResources();
      return status;
    }
  }

  return Status::OK();
}

void ExternalDataLoader::ReleaseResources() const noexcept {
  int previous_device = -1;
  const bool restore_device =
      cudaGetDevice(&previous_device) == cudaSuccess &&
      previous_device != device_id_ &&
      cudaSetDevice(device_id_) == cudaSuccess;

  for (auto& stream : streams_) {
    if (stream != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamSynchronize(stream)));
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamDestroy(stream)));
      stream = nullptr;
    }
  }

  for (auto& buffer : buffers_) {
    if (buffer != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaFreeHost(buffer)));
      buffer = nullptr;
    }
  }

  if (restore_device) {
    ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaSetDevice(previous_device)));
  }
}

common::Status ExternalDataLoader::LoadTensor(const Env& env,
                                              const std::filesystem::path& data_file_path,
                                              FileOffsetType data_offset,
                                              SafeInt<size_t> data_length,
                                              Tensor& tensor) const {
  ORT_RETURN_IF_NOT(CanLoad(tensor.Location()), "Unsupported tensor location: ",
                    tensor.Location().ToString());
  ORT_RETURN_IF(tensor.IsDataTypeString(),
                "String external initializers cannot be loaded into CUDA memory.");

  const size_t length = data_length;
  ORT_RETURN_IF_NOT(length == tensor.SizeInBytes(), "External data length does not match tensor size.");

  size_t file_length = 0;
  ORT_RETURN_IF_ERROR(env.GetFileLength(data_file_path.native().c_str(), file_length));
  const SafeInt<FileOffsetType> end_offset = SafeInt<FileOffsetType>(data_offset) + length;
  ORT_RETURN_IF(data_offset < 0 || end_offset > file_length,
                "External data range is outside the file.");

  std::lock_guard<std::mutex> lock(mutex_);
  CudaDeviceGuard device_guard;
  ORT_RETURN_IF_ERROR(device_guard.SetDevice(device_id_));
  ORT_RETURN_IF_ERROR(EnsureResources());

  auto* destination = static_cast<uint8_t*>(tensor.MutableDataRaw());
  std::array<bool, 2> stream_used{};
  auto synchronize_streams = [&]() {
    common::Status status = Status::OK();
    for (size_t i = 0; i < streams_.size(); ++i) {
      if (stream_used[i]) {
        const auto sync_status = CUDA_CALL(cudaStreamSynchronize(streams_[i]));
        if (status.IsOK() && !sync_status.IsOK()) {
          status = sync_status;
        }
        stream_used[i] = false;
      }
    }
    return status;
  };

  for (size_t offset = 0, chunk_index = 0; offset < length; ++chunk_index) {
    const size_t buffer_index = chunk_index % buffers_.size();
    const size_t chunk_size = std::min(kBufferSize, length - offset);

    if (stream_used[buffer_index]) {
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(streams_[buffer_index]));
    }

    const auto read_status =
        ReadChunk(env, data_file_path, data_offset + offset, chunk_size, buffers_[buffer_index]);
    if (!read_status.IsOK()) {
      ORT_IGNORE_RETURN_VALUE(synchronize_streams());
      return read_status;
    }

    if (tensor.IsDataType<bool>()) {
      auto* bool_data = static_cast<uint8_t*>(buffers_[buffer_index]);
      std::transform(bool_data, bool_data + chunk_size, bool_data,
                     [](uint8_t value) { return value != 0; });
    }

    if constexpr (std::endian::native != std::endian::little) {
      const size_t element_size = tensor.DataType()->Size();
      if (element_size > 1) {
        SwapByteOrderInplace(buffers_[buffer_index], chunk_size, element_size);
      }
    }

    const auto copy_status =
        CUDA_CALL(cudaMemcpyAsync(destination + offset, buffers_[buffer_index], chunk_size,
                                  cudaMemcpyHostToDevice, streams_[buffer_index]));
    if (!copy_status.IsOK()) {
      ORT_IGNORE_RETURN_VALUE(synchronize_streams());
      return copy_status;
    }
    stream_used[buffer_index] = true;
    offset += chunk_size;
  }

  return synchronize_streams();
}

}  // namespace cuda
}  // namespace onnxruntime
