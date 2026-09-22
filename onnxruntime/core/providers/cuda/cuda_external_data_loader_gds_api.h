// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string_view>
#include <type_traits>

#include <cufile.h>
#include <cuda_runtime_api.h>

#include "core/common/status.h"

namespace onnxruntime {
namespace cuda {

inline constexpr size_t kGdsBufferSize = 64 * 1024 * 1024;

common::Status CheckCuFileStatus(CUfileError_t status, std::string_view operation);

struct GdsReadApi {
  std::function<std::remove_pointer_t<decltype(&cuFileHandleRegister)>> handle_register;
  std::function<std::remove_pointer_t<decltype(&cuFileHandleDeregister)>> handle_deregister;
  std::function<std::remove_pointer_t<decltype(&cuFileRead)>> read;
  std::function<common::Status(void*, const void*, size_t, cudaMemcpyKind)> copy;
  std::function<common::Status(cudaStream_t)> synchronize;
};

// The caller owns the registered staging buffer, which must hold at least kGdsBufferSize bytes.
common::Status LoadGdsFile(const GdsReadApi& api, void* staging_buffer,
                           int file_descriptor, int64_t data_offset, size_t data_length,
                           void* destination);

}  // namespace cuda
}  // namespace onnxruntime
