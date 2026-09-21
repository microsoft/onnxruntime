// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// provider_api.h must be first to set SHARED_PROVIDER
#include "core/providers/shared_library/provider_api.h"

#include "core/providers/cuda/cuda_external_data_loader_gds.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"

#if defined(__linux__) && __has_include(<cufile.h>)
#define ORT_CUDA_GDS_AVAILABLE 1
#include <cufile.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace onnxruntime {
namespace cuda {
namespace {

constexpr size_t kGdsBufferSize = 64 * 1024 * 1024;

#if defined(ORT_CUDA_GDS_AVAILABLE)

template <typename T>
common::Status LoadSymbol(void* library, const char* name, T& function) {
  dlerror();
  function = reinterpret_cast<T>(dlsym(library, name));
  const char* error = dlerror();
  ORT_RETURN_IF(function == nullptr || error != nullptr,
                "Failed to load ", name, " from libcufile: ",
                error == nullptr ? "symbol not found" : error);
  return Status::OK();
}

common::Status CheckCuFileStatus(CUfileError_t status, std::string_view operation) {
  ORT_RETURN_IF(status.err != CU_FILE_SUCCESS, operation, " failed: ",
                cufileop_status_error(status.err), " (", static_cast<int>(status.err), ")");
  return Status::OK();
}

class LinuxGdsLoader final : public GdsLoader {
 public:
  using DriverOpenFn = decltype(&cuFileDriverOpen);
  using DriverCloseFn = CUfileError_t (*)();
  using HandleRegisterFn = decltype(&cuFileHandleRegister);
  using HandleDeregisterFn = decltype(&cuFileHandleDeregister);
  using BufferRegisterFn = decltype(&cuFileBufRegister);
  using BufferDeregisterFn = decltype(&cuFileBufDeregister);
  using ReadFn = decltype(&cuFileRead);

  ~LinuxGdsLoader() override {
    if (gds_buffer_registered_) {
      ORT_IGNORE_RETURN_VALUE(buffer_deregister_(gds_buffer_));
    }
    if (gds_buffer_ != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaFree(gds_buffer_)));
    }
    if (driver_initialized_) {
      ORT_IGNORE_RETURN_VALUE(driver_close_());
    }
    if (library_ != nullptr) {
      ORT_IGNORE_RETURN_VALUE(dlclose(library_));
    }
  }

  static common::Status Create(int device_id, std::unique_ptr<GdsLoader>& loader) {
    auto candidate = std::unique_ptr<LinuxGdsLoader>(new LinuxGdsLoader());
    ORT_RETURN_IF_ERROR(candidate->Initialize(device_id));
    loader = std::move(candidate);
    return Status::OK();
  }

  common::Status Load(const std::filesystem::path& data_file_path,
                      int64_t data_offset,
                      size_t data_length,
                      Tensor& tensor) const override {
    const int file_descriptor = open(data_file_path.c_str(), O_RDONLY | O_DIRECT);
    ORT_RETURN_IF(file_descriptor < 0, "Failed to open external data for GPUDirect Storage: ",
                  std::strerror(errno));
    auto close_file = gsl::finally([file_descriptor]() { ORT_IGNORE_RETURN_VALUE(close(file_descriptor)); });

    CUfileDescr_t descriptor{};
    descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    descriptor.handle.fd = file_descriptor;
    CUfileHandle_t file_handle = nullptr;
    ORT_RETURN_IF_ERROR(CheckCuFileStatus(handle_register_(&file_handle, &descriptor),
                                          "cuFileHandleRegister"));
    auto deregister_file = gsl::finally([&]() { handle_deregister_(file_handle); });

    auto* destination = static_cast<uint8_t*>(tensor.MutableDataRaw());
    for (size_t offset = 0; offset < data_length;) {
      const size_t chunk_size = std::min(kGdsBufferSize, data_length - offset);
      const auto file_offset = SafeInt<off_t>(data_offset) + offset;
      const ssize_t bytes_read = read_(file_handle, gds_buffer_, chunk_size, file_offset, 0);
      if (bytes_read != static_cast<ssize_t>(chunk_size)) {
        if (bytes_read == -1) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "cuFileRead failed: ", std::strerror(errno));
        }
        if (bytes_read < 0) {
          const auto cu_file_error = static_cast<CUfileOpError>(-bytes_read);
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "cuFileRead failed: ",
                                 cufileop_status_error(cu_file_error),
                                 " (", static_cast<int>(cu_file_error), ")");
        }
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "cuFileRead returned ", bytes_read,
                               " bytes; expected ", chunk_size, ".");
      }

      CUDA_RETURN_IF_ERROR(
          cudaMemcpy(destination + offset, gds_buffer_, chunk_size, cudaMemcpyDeviceToDevice));
      CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(nullptr));
      offset += chunk_size;
    }

    return Status::OK();
  }

 private:
  common::Status Initialize(int device_id) {
    library_ = dlopen("libcufile.so.0", RTLD_NOW | RTLD_LOCAL);
    if (library_ == nullptr) {
      library_ = dlopen("libcufile.so", RTLD_NOW | RTLD_LOCAL);
    }
    const char* library_error = dlerror();
    ORT_RETURN_IF(library_ == nullptr, "GPUDirect Storage is unavailable: ",
                  library_error == nullptr ? "libcufile could not be loaded" : library_error);

    ORT_RETURN_IF_ERROR(LoadSymbol(library_, "cuFileDriverOpen", driver_open_));
    auto close_status = LoadSymbol(library_, "cuFileDriverClose_v2", driver_close_);
    if (!close_status.IsOK()) {
      ORT_RETURN_IF_ERROR(LoadSymbol(library_, "cuFileDriverClose", driver_close_));
    }
    ORT_RETURN_IF_ERROR(LoadSymbol(library_, "cuFileHandleRegister", handle_register_));
    ORT_RETURN_IF_ERROR(LoadSymbol(library_, "cuFileHandleDeregister", handle_deregister_));
    ORT_RETURN_IF_ERROR(LoadSymbol(library_, "cuFileBufRegister", buffer_register_));
    ORT_RETURN_IF_ERROR(LoadSymbol(library_, "cuFileBufDeregister", buffer_deregister_));
    ORT_RETURN_IF_ERROR(LoadSymbol(library_, "cuFileRead", read_));

    ORT_RETURN_IF_ERROR(CheckCuFileStatus(driver_open_(), "cuFileDriverOpen"));
    driver_initialized_ = true;

    CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id));
    CUDA_RETURN_IF_ERROR(cudaMalloc(&gds_buffer_, kGdsBufferSize));
    ORT_RETURN_IF_ERROR(CheckCuFileStatus(
        buffer_register_(gds_buffer_, kGdsBufferSize, 0), "cuFileBufRegister"));
    gds_buffer_registered_ = true;
    return Status::OK();
  }

  LinuxGdsLoader() = default;

  void* library_{nullptr};
  void* gds_buffer_{nullptr};
  bool driver_initialized_{false};
  bool gds_buffer_registered_{false};
  DriverOpenFn driver_open_{nullptr};
  DriverCloseFn driver_close_{nullptr};
  HandleRegisterFn handle_register_{nullptr};
  HandleDeregisterFn handle_deregister_{nullptr};
  BufferRegisterFn buffer_register_{nullptr};
  BufferDeregisterFn buffer_deregister_{nullptr};
  ReadFn read_{nullptr};
};

#endif

}  // namespace

common::Status GdsLoader::Create(int device_id, std::unique_ptr<GdsLoader>& loader) {
#if defined(ORT_CUDA_GDS_AVAILABLE)
  return LinuxGdsLoader::Create(device_id, loader);
#else
  ORT_UNUSED_PARAMETER(device_id);
  ORT_UNUSED_PARAMETER(loader);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "GPUDirect Storage requires Linux and a CUDA toolkit with cuFile headers.");
#endif
}

}  // namespace cuda
}  // namespace onnxruntime
