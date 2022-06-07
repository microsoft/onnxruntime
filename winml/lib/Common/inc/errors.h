// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"

#define WINML_THROW_IF_NOT_OK(status)                                                                      \
  do {                                                                                                     \
    auto _status = status;                                                                                 \
    if (!_status.IsOK()) {                                                                                 \
      HRESULT hresult = StatusCodeToHRESULT(static_cast<StatusCode>(_status.Code()));                      \
      telemetry_helper.LogRuntimeError(hresult, _status.ErrorMessage(), __FILE__, __FUNCTION__, __LINE__); \
      winrt::hstring errorMessage(_winml::Strings::HStringFromUTF8(_status.ErrorMessage()));                \
      throw winrt::hresult_error(hresult, errorMessage);                                                   \
    }                                                                                                      \
  } while (0)

//
// WINML_THROW_IF_*_MSG Variants
//

#define WINML_THROW_HR_IF_FALSE_MSG(hr, value, message, ...)                        \
  do {                                                                              \
    auto _value = value;                                                            \
    if (_value == false) {                                                          \
      auto _hr = hr;                                                                \
      char msg[1024];                                                               \
      sprintf_s(msg, message, __VA_ARGS__);                                         \
      telemetry_helper.LogRuntimeError(_hr, msg, __FILE__, __FUNCTION__, __LINE__); \
      winrt::hstring errorMessage(_winml::Strings::HStringFromUTF8(msg));            \
      throw winrt::hresult_error(_hr, errorMessage);                                \
    }                                                                               \
  } while (0)

#define WINML_THROW_HR_IF_TRUE_MSG(hr, value, message, ...) WINML_THROW_HR_IF_FALSE_MSG(hr, !(value), message, __VA_ARGS__)
#define WINML_THROW_HR_IF_NULL_MSG(hr, value, message, ...) WINML_THROW_HR_IF_TRUE_MSG(hr, ((value) == nullptr), message, __VA_ARGS__)

//
// WINML_THROW_IF_FAILED* Variants
//

#define WINML_THROW_HR(hr)                                                           \
  {                                                                                  \
    auto _result = hr;                                                               \
    telemetry_helper.LogRuntimeError(_result, "", __FILE__, __FUNCTION__, __LINE__); \
    throw winrt::hresult_error(_result, winrt::hresult_error::from_abi);             \
  }

#define WINML_THROW_HR_MSG_NO_TELEMETRY_SENT(hr, message, ...)               \
  do {                                                                       \
    auto _hr = hr;                                                           \
    char msg[1024];                                                          \
    sprintf_s(msg, message, __VA_ARGS__);                                    \
    winrt::hstring errorMessage(_winml::Strings::HStringFromUTF8(msg));      \
    throw winrt::hresult_error(_hr, errorMessage);                           \
  } while (0)

#define WINML_THROW_IF_FAILED(hr)                                                  \
  do {                                                                             \
    HRESULT _hr = hr;                                                              \
    if (FAILED(_hr)) {                                                             \
      telemetry_helper.LogRuntimeError(_hr, "", __FILE__, __FUNCTION__, __LINE__); \
      throw winrt::hresult_error(_hr, winrt::hresult_error::from_abi);             \
    }                                                                              \
  } while (0)

#define WINML_THROW_IF_FAILED_MSG(hr, message, ...)                    \
  do {                                                                 \
    HRESULT _result = hr;                                              \
    if (FAILED(_result)) {                                             \
      WINML_THROW_HR_IF_TRUE_MSG(_result, true, message, __VA_ARGS__); \
    }                                                                  \
  } while (0)

using thrower = std::function<void(HRESULT)>;
using enforce = std::function<void(HRESULT, bool)>;
using enforce_succeeded = std::function<void(HRESULT)>;

inline void enforce_not_false(HRESULT hr, bool value, thrower fnThrower) {
  if (value == false) {
    fnThrower(hr);
  }
}

inline void enforce_not_failed(HRESULT hr, thrower fnThrower) {
  if (FAILED(hr)) {
    fnThrower(hr);
  }
}

inline __declspec(noinline) winrt::hresult_error _to_hresult() noexcept {
  try {
    throw;
  } catch (winrt::hresult_error const& e) {
    return e;
  } catch (wil::ResultException const& e) {
    return winrt::hresult_error(e.GetErrorCode(), winrt::to_hstring(e.what()));
  } catch (std::bad_alloc const&) {
    return winrt::hresult_error(E_OUTOFMEMORY);
  } catch (std::out_of_range const& e) {
    return winrt::hresult_out_of_bounds(winrt::to_hstring(e.what()));
  } catch (std::invalid_argument const& e) {
    return winrt::hresult_invalid_argument(winrt::to_hstring(e.what()));
  } catch (std::exception const& e) {
    return winrt::hresult_error(E_FAIL, winrt::to_hstring(e.what()));
  } catch (...) {
    return winrt::hresult_error(E_FAIL);
  }
}

#define WINML_CATCH_ALL  \
  catch (...) {          \
    throw _to_hresult(); \
  }

#define WINML_CATCH_ALL_COM        \
  catch (...) {                    \
    return _to_hresult().to_abi(); \
  }

#define WINML_CATCH_ALL_DONOTHING \
  catch (...) {                   \
    return;                       \
  }