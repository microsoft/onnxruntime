// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// provider_api.h must be first to set SHARED_PROVIDER
#include "core/providers/shared_library/provider_api.h"

#include "core/providers/cuda/cuda_external_data_loader_gds.h"

#include "core/common/common.h"
#include "core/providers/cuda/cuda_common.h"

#if defined(ORT_CUDA_GDS_AVAILABLE)
#include "core/providers/cuda/cuda_external_data_loader_gds_api.h"

#include <dlfcn.h>
#endif

namespace onnxruntime {
namespace cuda {
namespace {

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

class CuFileDriver {
 public:
  using DriverOpenFn = decltype(&cuFileDriverOpen);
  using DriverCloseFn = CUfileError_t (*)();
  using HandleRegisterFn = decltype(&cuFileHandleRegister);
  using HandleDeregisterFn = decltype(&cuFileHandleDeregister);
  using BufferRegisterFn = decltype(&cuFileBufRegister);
  using BufferDeregisterFn = decltype(&cuFileBufDeregister);
  using SetBoolParameterFn = decltype(&cuFileSetParameterBool);
  using ReadFn = decltype(&cuFileRead);

  CuFileDriver() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CuFileDriver);

  ~CuFileDriver() {
    if (driver_initialized_) {
      ORT_IGNORE_RETURN_VALUE(driver_close_());
    }
    if (library_ != nullptr) {
      ORT_IGNORE_RETURN_VALUE(dlclose(library_));
    }
  }

  GdsReadApi GetReadApi() const {
    return {handle_register_, handle_deregister_, read_,
            [](void* destination, const void* source, size_t length, cudaMemcpyKind kind) {
              return CUDA_CALL(cudaMemcpy(destination, source, length, kind));
            },
            [](cudaStream_t stream) {
              return CUDA_CALL(cudaStreamSynchronize(stream));
            }};
  }

  CUfileError_t RegisterBuffer(const void* buffer, size_t length) const {
    return buffer_register_(buffer, length, 0);
  }

  CUfileError_t DeregisterBuffer(const void* buffer) const {
    return buffer_deregister_(buffer);
  }

  common::Status Initialize() {
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
    ORT_RETURN_IF_ERROR(LoadSymbol(library_, "cuFileSetParameterBool", set_bool_parameter_));
    ORT_RETURN_IF_ERROR(LoadSymbol(library_, "cuFileRead", read_));

    ORT_RETURN_IF_ERROR(CheckCuFileStatus(
        set_bool_parameter_(CUFILE_PARAM_USE_PCIP2PDMA, true),
        "Enabling cuFile PCI P2PDMA"));
    ORT_RETURN_IF_ERROR(CheckCuFileStatus(
        set_bool_parameter_(CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE, false),
        "Disabling cuFile compatibility mode"));
    ORT_RETURN_IF_ERROR(CheckCuFileStatus(driver_open_(), "cuFileDriverOpen"));
    driver_initialized_ = true;
    return Status::OK();
  }

 private:
  void* library_{nullptr};
  bool driver_initialized_{false};
  DriverOpenFn driver_open_{nullptr};
  DriverCloseFn driver_close_{nullptr};
  HandleRegisterFn handle_register_{nullptr};
  HandleDeregisterFn handle_deregister_{nullptr};
  BufferRegisterFn buffer_register_{nullptr};
  BufferDeregisterFn buffer_deregister_{nullptr};
  SetBoolParameterFn set_bool_parameter_{nullptr};
  ReadFn read_{nullptr};
};

class LinuxGdsLoader final : public GdsLoader {
 public:
  ~LinuxGdsLoader() override {
    if (gds_buffer_registered_) {
      ORT_IGNORE_RETURN_VALUE(driver_->DeregisterBuffer(gds_buffer_));
    }
    if (gds_buffer_ != nullptr) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaFree(gds_buffer_)));
    }
  }

  static common::Status Create(int device_id, std::unique_ptr<GdsLoader>& loader) {
    auto candidate = std::unique_ptr<LinuxGdsLoader>(new LinuxGdsLoader());
    ORT_RETURN_IF_ERROR(candidate->Initialize(device_id));
    loader = std::move(candidate);
    return Status::OK();
  }

  common::Status Load(int file_descriptor,
                      int64_t data_offset,
                      size_t data_length,
                      Tensor& tensor) const override {
    return LoadGdsFile(driver_->GetReadApi(), gds_buffer_,
                       file_descriptor, data_offset, data_length, tensor.MutableDataRaw());
  }

 private:
  common::Status Initialize(int device_id) {
    ORT_RETURN_IF_ERROR(driver_.Acquire());

    CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id));
    CUDA_RETURN_IF_ERROR(cudaMalloc(&gds_buffer_, kGdsBufferSize));
    ORT_RETURN_IF_ERROR(CheckCuFileStatus(
        driver_->RegisterBuffer(gds_buffer_, kGdsBufferSize), "cuFileBufRegister"));
    gds_buffer_registered_ = true;
    return Status::OK();
  }

  LinuxGdsLoader() = default;

  GdsDriverHandle<CuFileDriver> driver_;
  void* gds_buffer_{nullptr};
  bool gds_buffer_registered_{false};
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
                         "GPUDirect Storage requires Linux and cuFile headers with cuFileSetParameterBool, "
                         "CUFILE_PARAM_USE_PCIP2PDMA, and CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE.");
#endif
}

}  // namespace cuda
}  // namespace onnxruntime
