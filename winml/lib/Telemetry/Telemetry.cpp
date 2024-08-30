// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "lib/Telemetry/pch.h"

WinMLTelemetryHelper telemetry_helper;

TRACELOGGING_DEFINE_PROVIDER(
  winml_trace_logging_provider, WINML_PROVIDER_DESC, WINML_PROVIDER_GUID, TraceLoggingOptionMicrosoftTelemetry()
);
