// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file status_impl.h
/// \brief Internal representation of ModelPackageStatus, shared by all
///        implementation units in the model_package library.

#pragma once

#include <new>
#include <string>
#include <utility>

#include "model_package_api.h"

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
