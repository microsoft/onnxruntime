// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file status_impl.h
/// \brief Internal representation of ModelPackageStatus, shared by all
///        implementation units in the model_package library.

#pragma once

#include <new>
#include <string>
#include <system_error>
#include <utility>

#include "model_package_api.h"

/// Function-try-block for functions returning ModelPackageStatus*:
/// MODEL_PACKAGE_CATCH turns a std::system_error into MODEL_PACKAGE_ERR_IO.
/// Both expand to nothing when exceptions are disabled.
#if defined(__cpp_exceptions) || defined(_CPPUNWIND)
#define MODEL_PACKAGE_TRY try
#define MODEL_PACKAGE_CATCH                                                 \
  catch (const std::system_error& error) {                                  \
    return ::model_package::MakeStatus(MODEL_PACKAGE_ERR_IO, error.what()); \
  }
#else
#define MODEL_PACKAGE_TRY
#define MODEL_PACKAGE_CATCH
#endif

struct ModelPackageStatus {
  ModelPackageErrorCode code{MODEL_PACKAGE_ERR_INVALID_ARG};
  std::string message;
};

namespace model_package {

/// Allocate a new failure status. Returns nullptr if allocation fails (callers
/// should treat that as a generic error; we deliberately never throw out of the
/// C API).
inline ModelPackageStatus* MakeStatus(ModelPackageErrorCode code, std::string message) {
  return new (std::nothrow) ModelPackageStatus{code, std::move(message)};
}

}  // namespace model_package
