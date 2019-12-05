// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"

WinMLTelemetryHelper telemetry_helper;

TRACELOGGING_DEFINE_PROVIDER(
    winml_trace_logging_provider,
    WINML_PROVIDER_DESC,
    WINML_PROVIDER_GUID,
    TraceLoggingOptionMicrosoftTelemetry());

