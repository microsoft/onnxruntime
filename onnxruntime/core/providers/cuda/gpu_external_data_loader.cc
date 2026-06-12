// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_GDS

#include "core/providers/cuda/gpu_external_data_loader.h"

#include <fcntl.h>
#include <unistd.h>
#include <cufile.h>

#include "core/framework/tensor.h"
#include "core/common/common.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {
namespace cuda {

namespace {
constexpr size_t kCuFileAlignment = 4096;  // cuFile requires 4K-aligned offsets for best perf
}

GpuExternalDataLoader::GpuExternalDataLoader(int device_id) : device_id_(device_id) {
  CUfileError_t status = cuFileDriverOpen();
  if (status.err == CU_FILE_SUCCESS) {
    driver_open_ = true;
  } else {
    LOGS_DEFAULT(WARNING) << "cuFileDriverOpen failed (err=" << status.err
                          << "). GDS direct loading will not be available.";
  }
}

GpuExternalDataLoader::~GpuExternalDataLoader() {
  if (driver_open_) {
    cuFileDriverClose();
  }
}

bool GpuExternalDataLoader::IsGdsAvailable() {
  CUfileError_t status = cuFileDriverOpen();
  if (status.err == CU_FILE_SUCCESS) {
    cuFileDriverClose();
    return true;
  }
  return false;
}

bool GpuExternalDataLoader::CanLoad(const OrtMemoryInfo& target_memory_info) const {
  return driver_open_ &&
         target_memory_info.device.Type() == OrtDevice::GPU &&
         target_memory_info.device.Id() == device_id_;
}

common::Status GpuExternalDataLoader::LoadTensor(const Env& /*env*/,
                                                  const std::filesystem::path& data_file_path,
                                                  FileOffsetType data_offset,
                                                  SafeInt<size_t> data_length,
                                                  Tensor& tensor) const {
  if (!driver_open_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "cuFile driver is not open");
  }

  const size_t length = data_length;
  void* gpu_ptr = tensor.MutableDataRaw();

  // Open file with O_DIRECT for GDS bypass
  int fd = open(data_file_path.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to open file for GDS: ", data_file_path.string(),
                           ", errno=", errno);
  }

  // Register file handle with cuFile
  CUfileDescr_t cf_descr{};
  cf_descr.handle.fd = fd;
  cf_descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

  CUfileHandle_t cf_handle{};
  CUfileError_t status = cuFileHandleRegister(&cf_handle, &cf_descr);
  if (status.err != CU_FILE_SUCCESS) {
    close(fd);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "cuFileHandleRegister failed for: ", data_file_path.string(),
                           ", err=", status.err);
  }

  // Register GPU buffer with cuFile
  status = cuFileBufRegister(gpu_ptr, length, 0);
  if (status.err != CU_FILE_SUCCESS) {
    cuFileHandleDeregister(cf_handle);
    close(fd);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "cuFileBufRegister failed, err=", status.err);
  }

  // Read directly from disk to GPU memory
  // cuFileRead: returns bytes read on success, negative on error
  ssize_t bytes_read = cuFileRead(cf_handle, gpu_ptr, length,
                                  static_cast<off_t>(data_offset), 0);

  // Cleanup
  cuFileBufDeregister(gpu_ptr);
  cuFileHandleDeregister(cf_handle);
  close(fd);

  if (bytes_read < 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "cuFileRead failed for: ", data_file_path.string(),
                           ", error=", bytes_read);
  }

  if (static_cast<size_t>(bytes_read) != length) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "cuFileRead incomplete: expected ", length,
                           " bytes but got ", bytes_read);
  }

  return common::Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

#endif  // USE_GDS
