// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ORT_CUDA_GDS_AVAILABLE)

#include "core/providers/cuda/cuda_external_data_loader_gds_api.h"

#include <algorithm>
#include <cerrno>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>
#include <gsl/gsl>

#include "core/common/common.h"
#include "core/common/safeint.h"

namespace onnxruntime {
namespace cuda {

common::Status CheckCuFileStatus(CUfileError_t status, std::string_view operation) {
  ORT_RETURN_IF(status.err != CU_FILE_SUCCESS, operation, " failed: ",
                cufileop_status_error(status.err), " (", static_cast<int>(status.err), ")");
  return Status::OK();
}

common::Status LoadGdsFile(const GdsReadApi& api, void* staging_buffer,
                           int file_descriptor, int64_t data_offset, size_t data_length,
                           void* destination_buffer) {
  ORT_RETURN_IF(file_descriptor < 0,
                "GPUDirect Storage requires an open POSIX file descriptor.");

  const int direct_descriptor = dup(file_descriptor);
  ORT_RETURN_IF(direct_descriptor < 0, "Failed to duplicate external-data file descriptor: ",
                std::strerror(errno));
  auto close_file = gsl::finally([direct_descriptor]() {
    ORT_IGNORE_RETURN_VALUE(close(direct_descriptor));
  });

  const int original_flags = fcntl(direct_descriptor, F_GETFL);
  ORT_RETURN_IF(original_flags < 0, "Failed to query external-data file flags: ",
                std::strerror(errno));
  ORT_RETURN_IF(fcntl(direct_descriptor, F_SETFL, original_flags | O_DIRECT) < 0,
                "Failed to enable O_DIRECT for GPUDirect Storage: ", std::strerror(errno));
  auto restore_flags = gsl::finally([direct_descriptor, original_flags]() {
    ORT_IGNORE_RETURN_VALUE(fcntl(direct_descriptor, F_SETFL, original_flags));
  });

  CUfileDescr_t descriptor{};
  descriptor.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
  descriptor.handle.fd = direct_descriptor;
  CUfileHandle_t file_handle = nullptr;
  ORT_RETURN_IF_ERROR(CheckCuFileStatus(api.handle_register(&file_handle, &descriptor),
                                        "cuFileHandleRegister"));
  auto deregister_file = gsl::finally([&]() { api.handle_deregister(file_handle); });

  auto* destination = static_cast<uint8_t*>(destination_buffer);
  for (size_t offset = 0; offset < data_length;) {
    const size_t chunk_size = std::min(kGdsBufferSize, data_length - offset);
    const auto file_offset = SafeInt<off_t>(data_offset) + offset;
    const ssize_t bytes_read = api.read(file_handle, staging_buffer, chunk_size, file_offset, 0);
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

    ORT_RETURN_IF_ERROR(api.copy(destination + offset, staging_buffer, chunk_size, cudaMemcpyDeviceToDevice));
    ORT_RETURN_IF_ERROR(api.synchronize(nullptr));
    offset += chunk_size;
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

#endif
