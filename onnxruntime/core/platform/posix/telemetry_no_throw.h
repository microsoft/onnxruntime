// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <exception>
#include <utility>

#include "core/common/common.h"

namespace onnxruntime::telemetry_internal {

template <typename Warning>
void ReportTelemetryFailureNoThrow(Warning& warning, const char* message) noexcept {
  ORT_TRY {
    warning(message);
  }
  ORT_CATCH(...) {
  }
}

template <typename Operation, typename Warning>
void RunTelemetryOperationNoThrow(Operation&& operation, Warning&& warning) noexcept {
  ORT_TRY {
    std::forward<Operation>(operation)();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      ReportTelemetryFailureNoThrow(warning, ex.what());
    });
  }
  ORT_CATCH(...) {
    ReportTelemetryFailureNoThrow(warning, nullptr);
  }
}

}  // namespace onnxruntime::telemetry_internal
