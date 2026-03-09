// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(_WIN32)

#include <winerror.h>

#include "core/common/status.h"

namespace onnxruntime::common {

constexpr HRESULT StatusCodeToHRESULT(StatusCode status) noexcept {
  switch (status) {
    case StatusCode::OK:
      return S_OK;
    case StatusCode::FAIL:
      return E_FAIL;
    case StatusCode::INVALID_ARGUMENT:
      return E_INVALIDARG;
    case StatusCode::NO_SUCHFILE:
      return HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
    case StatusCode::NO_MODEL:
      return HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
    case StatusCode::ENGINE_ERROR:
      return E_FAIL;
    case StatusCode::RUNTIME_EXCEPTION:
      return E_FAIL;
    case StatusCode::INVALID_PROTOBUF:
      return HRESULT_FROM_WIN32(ERROR_FILE_CORRUPT);
    case StatusCode::MODEL_LOADED:
      return HRESULT_FROM_WIN32(ERROR_INTERNAL_ERROR);
    case StatusCode::NOT_IMPLEMENTED:
      return E_NOTIMPL;
    case StatusCode::INVALID_GRAPH:
      return HRESULT_FROM_WIN32(ERROR_FILE_CORRUPT);
    case StatusCode::EP_FAIL:
      return HRESULT_FROM_WIN32(ERROR_INTERNAL_ERROR);
    case StatusCode::MODEL_LOAD_CANCELED:
      return HRESULT_FROM_WIN32(ERROR_CANCELLED);
    case StatusCode::MODEL_REQUIRES_COMPILATION:
      return HRESULT_FROM_WIN32(ERROR_NOT_SUPPORTED);
    case StatusCode::NOT_FOUND:
      return HRESULT_FROM_WIN32(ERROR_NOT_FOUND);
    default:
      return E_FAIL;
  }
}

}  // namespace onnxruntime::common

#endif  // defined(_WIN32)
