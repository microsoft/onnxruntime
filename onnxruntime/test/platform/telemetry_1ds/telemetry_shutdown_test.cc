// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdlib>
#include <memory>
#include <string_view>

#include "core/platform/telemetry_1ds.h"

namespace {

std::unique_ptr<onnxruntime::OneDsTelemetry> telemetry;

// Register this before SDK initialization so it runs after the SDK shutdown callback,
// but before the global provider is destroyed. No model or event is needed: this test
// exercises teardown ordering without depending on a network response or a crash.
void CheckShutdown() {
  if (telemetry && telemetry->IsEnabled()) {
    std::_Exit(1);
  }
  // Teardown must remain idempotent, and a late static destructor must not restart the SDK.
  telemetry.reset();
  onnxruntime::OneDsTelemetry late_provider;
  late_provider.EnableTelemetryEvents();
  if (late_provider.IsEnabled()) {
    std::_Exit(2);
  }
}

}  // namespace

int main(int argc, char* argv[]) {
  if (std::atexit(CheckShutdown) != 0) {
    return 3;
  }
  const std::string_view mode = argc > 1 ? argv[1] : "retained";
  telemetry = std::make_unique<onnxruntime::OneDsTelemetry>();
  if (mode == "disabled") {
    return telemetry->IsEnabled() ? 4 : 0;
  }
  if (!telemetry->IsEnabled()) {
    return 5;
  }
  if (mode == "recreated") {
    telemetry.reset();
    telemetry = std::make_unique<onnxruntime::OneDsTelemetry>();
    if (!telemetry->IsEnabled()) {
      return 6;
    }
  } else if (mode == "released") {
    telemetry.reset();
  }
  return 0;
}
