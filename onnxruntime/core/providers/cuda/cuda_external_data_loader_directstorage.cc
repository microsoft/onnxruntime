// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// provider_api.h must be first to set SHARED_PROVIDER
#include "core/providers/shared_library/provider_api.h"

#include "core/providers/cuda/cuda_external_data_loader_directstorage.h"

#if defined(ORT_CUDA_DIRECTSTORAGE_AVAILABLE)
#include <algorithm>
#include <cstring>
#include <d3d12.h>
#include <dstorage.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#endif

namespace onnxruntime {
namespace cuda {
namespace {

#if defined(ORT_CUDA_DIRECTSTORAGE_AVAILABLE)

using Microsoft::WRL::ComPtr;
constexpr size_t kDirectStorageBufferSize = 32 * 1024 * 1024;

common::Status CheckHResult(HRESULT result, const char* operation) {
  ORT_RETURN_IF(FAILED(result), operation, " failed, HRESULT=", static_cast<uint32_t>(result));
  return Status::OK();
}

class DirectStorageLibrary {
 public:
  DirectStorageLibrary() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(DirectStorageLibrary);
  ~DirectStorageLibrary() {
    if (handle != nullptr) {
      FreeLibrary(handle);
    }
  }
  HMODULE handle{nullptr};
};

class WindowsDirectStorageLoader final : public DirectStorageLoader {
 public:
  ~WindowsDirectStorageLoader() override {
    if (stream_ != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamSynchronize(stream_)));
    }
    if (queue_) {
      queue_->Close();
    }
    if (stream_ != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaStreamDestroy(stream_)));
    }
    if (cuda_fence_ != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaDestroyExternalSemaphore(cuda_fence_)));
    }
    if (buffer_ != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaFree(buffer_)));
    }
    if (cuda_memory_ != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaDestroyExternalMemory(cuda_memory_)));
    }
  }

  static common::Status Create(int device_id, std::unique_ptr<DirectStorageLoader>& loader) {
    auto candidate = std::unique_ptr<WindowsDirectStorageLoader>(new WindowsDirectStorageLoader());
    ORT_RETURN_IF_ERROR(candidate->Initialize(device_id));
    loader = std::move(candidate);
    return Status::OK();
  }

  common::Status Load(const std::filesystem::path& path, void* validated_file_handle,
                      int64_t data_offset, size_t data_length, Tensor& tensor) override {
    ORT_RETURN_IF(validated_file_handle == nullptr,
                  "Microsoft DirectStorage requires the validated Windows file handle.");
    BY_HANDLE_FILE_INFORMATION original{};
    ORT_RETURN_IF_NOT(GetFileInformationByHandle(validated_file_handle, &original),
                      "GetFileInformationByHandle failed: ", GetLastError());
    ComPtr<IDStorageFile> file;
    ORT_RETURN_IF_ERROR(CheckHResult(factory_->OpenFile(path.c_str(), IID_PPV_ARGS(&file)),
                                     "DirectStorage OpenFile"));
    bool pending = false;
    auto drain_on_error = gsl::finally([&]() {
      if (pending) {
        queue_->Close();
      }
    });
    BY_HANDLE_FILE_INFORMATION opened{};
    ORT_RETURN_IF_ERROR(CheckHResult(file->GetFileInformation(&opened), "DirectStorage GetFileInformation"));
    // DirectStorage opens by path. Never read a replacement for the file validated by the caller.
    ORT_RETURN_IF(original.dwVolumeSerialNumber != opened.dwVolumeSerialNumber ||
                      original.nFileIndexHigh != opened.nFileIndexHigh ||
                      original.nFileIndexLow != opened.nFileIndexLow,
                  "External-data file changed before DirectStorage opened it.");
    const uint64_t file_size = (static_cast<uint64_t>(opened.nFileSizeHigh) << 32) | opened.nFileSizeLow;
    ORT_RETURN_IF(data_offset < 0 || static_cast<uint64_t>(data_offset) > file_size ||
                      data_length > file_size - static_cast<uint64_t>(data_offset),
                  "DirectStorage external-data range is outside the file.");

    auto* destination = static_cast<uint8_t*>(tensor.MutableDataRaw());
    for (size_t offset = 0; offset < data_length;) {
      const auto chunk = static_cast<uint32_t>(std::min(kDirectStorageBufferSize, data_length - offset));
      DSTORAGE_REQUEST request{};
      request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
      request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_BUFFER;
      request.Source.File.Source = file.Get();
      request.Source.File.Offset = SafeInt<uint64_t>(data_offset) + offset;
      request.Source.File.Size = chunk;
      request.UncompressedSize = chunk;
      request.Destination.Buffer.Resource = resource_.Get();
      request.Destination.Buffer.Size = chunk;
      queue_->EnqueueRequest(&request);
      queue_->EnqueueStatus(status_.Get(), 0);
      queue_->EnqueueSignal(fence_.Get(), ++fence_value_);
      queue_->Submit();
      pending = true;

      cudaExternalSemaphoreWaitParams wait{};
      wait.params.fence.value = fence_value_;
      CUDA_RETURN_IF_ERROR(cudaWaitExternalSemaphoresAsync(&cuda_fence_, &wait, 1, stream_));
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream_));
      pending = false;
      ORT_RETURN_IF_ERROR(CheckHResult(status_->GetHResult(0), "DirectStorage read"));
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(destination + offset, buffer_, chunk,
                                           cudaMemcpyDeviceToDevice, stream_));
      // Complete CUDA reads before DirectStorage writes the shared buffer again.
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream_));
      offset += chunk;
    }
    return Status::OK();
  }

 private:
  WindowsDirectStorageLoader() = default;

  common::Status Initialize(int device_id) {
    CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id));
    cudaDeviceProp properties{};
    CUDA_RETURN_IF_ERROR(cudaGetDeviceProperties(&properties, device_id));
    const unsigned int node_mask = properties.luidDeviceNodeMask;
    ORT_RETURN_IF(node_mask == 0 || (node_mask & (node_mask - 1)) != 0,
                  "DirectStorage requires a single CUDA device node.");
    LUID adapter_luid{};
    static_assert(sizeof(properties.luid) == sizeof(adapter_luid));
    std::memcpy(&adapter_luid, properties.luid, sizeof(adapter_luid));
    ComPtr<IDXGIFactory4> dxgi;
    ORT_RETURN_IF_ERROR(CheckHResult(CreateDXGIFactory1(IID_PPV_ARGS(&dxgi)), "CreateDXGIFactory1"));
    ComPtr<IDXGIAdapter> adapter;
    ORT_RETURN_IF_ERROR(CheckHResult(dxgi->EnumAdapterByLuid(adapter_luid, IID_PPV_ARGS(&adapter)),
                                     "Find CUDA DXGI adapter"));
    ORT_RETURN_IF_ERROR(CheckHResult(
        D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device_)), "D3D12CreateDevice"));
    ORT_RETURN_IF(device_->GetNodeCount() != 1, "DirectStorage does not support linked D3D12 adapters.");

    library_.handle = LoadLibraryExW(L"dstorage.dll", nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    ORT_RETURN_IF(library_.handle == nullptr, "Microsoft DirectStorage is unavailable: dstorage.dll load failed: ",
                  GetLastError());
    const auto get_factory =
        reinterpret_cast<decltype(&DStorageGetFactory)>(GetProcAddress(library_.handle, "DStorageGetFactory"));
    ORT_RETURN_IF(get_factory == nullptr, "DStorageGetFactory is unavailable: ", GetLastError());
    ORT_RETURN_IF_ERROR(CheckHResult(get_factory(IID_PPV_ARGS(&factory_)), "DStorageGetFactory"));
    // The default DirectStorage staging size is 32 MiB. Do not reconfigure its process-wide factory.
    DSTORAGE_QUEUE_DESC queue_desc{};
    queue_desc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
    queue_desc.Capacity = DSTORAGE_MIN_QUEUE_CAPACITY;
    queue_desc.Priority = DSTORAGE_PRIORITY_NORMAL;
    queue_desc.Device = device_.Get();
    ORT_RETURN_IF_ERROR(CheckHResult(factory_->CreateQueue(&queue_desc, IID_PPV_ARGS(&queue_)),
                                     "DirectStorage CreateQueue"));
    ORT_RETURN_IF_ERROR(CheckHResult(factory_->CreateStatusArray(1, "ORT external data", IID_PPV_ARGS(&status_)),
                                     "DirectStorage CreateStatusArray"));

    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap.CreationNodeMask = node_mask;
    heap.VisibleNodeMask = node_mask;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = kDirectStorageBufferSize;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ORT_RETURN_IF_ERROR(CheckHResult(
        device_->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_SHARED, &desc,
                                         D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&resource_)),
        "Create shared DirectStorage buffer"));
    HANDLE shared_memory = nullptr;
    ORT_RETURN_IF_ERROR(CheckHResult(
        device_->CreateSharedHandle(resource_.Get(), nullptr, GENERIC_ALL, nullptr, &shared_memory),
        "Share DirectStorage buffer"));
    auto close_memory = gsl::finally([&]() { CloseHandle(shared_memory); });
    cudaExternalMemoryHandleDesc memory_desc{};
    memory_desc.type = cudaExternalMemoryHandleTypeD3D12Resource;
    memory_desc.handle.win32.handle = shared_memory;
    memory_desc.size = device_->GetResourceAllocationInfo(node_mask, 1, &desc).SizeInBytes;
    memory_desc.flags = cudaExternalMemoryDedicated;
    CUDA_RETURN_IF_ERROR(cudaImportExternalMemory(&cuda_memory_, &memory_desc));
    cudaExternalMemoryBufferDesc buffer_desc{};
    buffer_desc.size = kDirectStorageBufferSize;
    CUDA_RETURN_IF_ERROR(cudaExternalMemoryGetMappedBuffer(&buffer_, cuda_memory_, &buffer_desc));

    ORT_RETURN_IF_ERROR(CheckHResult(
        device_->CreateFence(0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence_)), "Create DirectStorage fence"));
    HANDLE shared_fence = nullptr;
    ORT_RETURN_IF_ERROR(CheckHResult(
        device_->CreateSharedHandle(fence_.Get(), nullptr, GENERIC_ALL, nullptr, &shared_fence),
        "Share DirectStorage fence"));
    auto close_fence = gsl::finally([&]() { CloseHandle(shared_fence); });
    cudaExternalSemaphoreHandleDesc fence_desc{};
    fence_desc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    fence_desc.handle.win32.handle = shared_fence;
    CUDA_RETURN_IF_ERROR(cudaImportExternalSemaphore(&cuda_fence_, &fence_desc));
    CUDA_RETURN_IF_ERROR(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    return Status::OK();
  }

  DirectStorageLibrary library_;
  ComPtr<ID3D12Device> device_;
  ComPtr<IDStorageFactory> factory_;
  ComPtr<ID3D12Resource> resource_;
  ComPtr<ID3D12Fence> fence_;
  ComPtr<IDStorageStatusArray> status_;
  ComPtr<IDStorageQueue> queue_;
  cudaExternalMemory_t cuda_memory_{nullptr};
  cudaExternalSemaphore_t cuda_fence_{nullptr};
  cudaStream_t stream_{nullptr};
  void* buffer_{nullptr};
  uint64_t fence_value_{0};
};

#endif

}  // namespace

common::Status DirectStorageLoader::Create(int device_id, std::unique_ptr<DirectStorageLoader>& loader) {
#if defined(ORT_CUDA_DIRECTSTORAGE_AVAILABLE)
  return WindowsDirectStorageLoader::Create(device_id, loader);
#else
  ORT_UNUSED_PARAMETER(device_id);
  ORT_UNUSED_PARAMETER(loader);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Microsoft DirectStorage requires a Windows CUDA build with "
                         "onnxruntime_USE_CUDA_DIRECTSTORAGE=ON.");
#endif
}

}  // namespace cuda
}  // namespace onnxruntime
