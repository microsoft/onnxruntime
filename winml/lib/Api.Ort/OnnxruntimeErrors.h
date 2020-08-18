// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "pch.h"
#include "core/providers/winml/winml_provider_factory.h"

#ifdef _WIN32
inline HRESULT OrtErrorCodeToHRESULT(OrtErrorCode status) noexcept {
  switch (status) {
    case OrtErrorCode::ORT_OK:
      return S_OK;
    case OrtErrorCode::ORT_FAIL:
      return E_FAIL;
    case OrtErrorCode::ORT_INVALID_ARGUMENT:
      return E_INVALIDARG;
    case OrtErrorCode::ORT_NO_SUCHFILE:
      return __HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
    case OrtErrorCode::ORT_NO_MODEL:
      return __HRESULT_FROM_WIN32(ERROR_FILE_NOT_FOUND);
    case OrtErrorCode::ORT_ENGINE_ERROR:
      return E_FAIL;
    case OrtErrorCode::ORT_RUNTIME_EXCEPTION:
      return E_FAIL;
    case OrtErrorCode::ORT_INVALID_PROTOBUF:
      return __HRESULT_FROM_WIN32(ERROR_FILE_CORRUPT);
    case OrtErrorCode::ORT_MODEL_LOADED:
      return __HRESULT_FROM_WIN32(ERROR_INTERNAL_ERROR);
    case OrtErrorCode::ORT_NOT_IMPLEMENTED:
      return E_NOTIMPL;
    case OrtErrorCode::ORT_INVALID_GRAPH:
      return __HRESULT_FROM_WIN32(ERROR_FILE_CORRUPT);
    case OrtErrorCode::ORT_EP_FAIL:
      return __HRESULT_FROM_WIN32(ERROR_INTERNAL_ERROR);
    default:
      return E_FAIL;
  }
}
#endif

static inline HRESULT OriginateError(HRESULT hresult, const char *message) {
#if WINVER >= _WIN32_WINNT_WIN8
    auto hstring_message = _winml::Strings::HStringFromUTF8(message);
    RoOriginateError(hresult, reinterpret_cast<HSTRING>(winrt::get_abi(hstring_message)));
#else
    RETURN_HR_MSG(hresult, message);
#endif
}

#define RETURN_HR_IF_NOT_OK_MSG(status, ort_api)                                                               \
  do {                                                                                                         \
    auto _status = status;                                                                                     \
    if (_status) {                                                                                             \
      auto error_code = ort_api->GetErrorCode(_status);                                                        \
      auto error_message = ort_api->GetErrorMessage(_status);                                                  \
      HRESULT hresult = OrtErrorCodeToHRESULT(error_code);                                                     \
      telemetry_helper.LogRuntimeError(hresult, error_message, __FILE__, __FUNCTION__, __LINE__); \
      return OriginateError(hresult, error_message);                                                           \
    }                                                                                                          \
  } while (0)

#define THROW_IF_NOT_OK_MSG(status, ort_api)                                                                   \
  do {                                                                                                         \
    auto _status = status;                                                                                     \
    if (_status) {                                                                                             \
      auto error_code = ort_api->GetErrorCode(_status);                                                        \
      auto error_message = ort_api->GetErrorMessage(_status);                                                  \
      HRESULT hresult = OrtErrorCodeToHRESULT(error_code);                                                     \
      telemetry_helper.LogRuntimeError(hresult, std::string(error_message), __FILE__, __FUNCTION__, __LINE__); \
      auto message = _winml::Strings::HStringFromUTF8(error_message);                                           \
      throw winrt::hresult_error(hresult, message);                                                            \
    }                                                                                                          \
  } while (0)
