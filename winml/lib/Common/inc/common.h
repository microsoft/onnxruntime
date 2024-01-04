// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// STL
#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING
#define _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING

#include <algorithm>
#include <cassert>
#include <codecvt>
#include <functional>
#include <iterator>
#include <locale>
#include <numeric>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>
#include <thread>
#include <tuple>

// WIL
#include <wil/cppwinrt.h>
#include <wil/result.h>
#include <wil/winrt.h>

// Windows pollutes with preprocessor that redefine OPTIONAL.
// Undefine OPTIONAL to get onnx macros to resolve correctly.
#ifdef OPTIONAL
#undef OPTIONAL
#endif

#pragma warning(disable : 4100)

// Telemetry
#include "WinMLTelemetryHelper.h"
// Declare global telemetry helper
extern WinMLTelemetryHelper telemetry_helper;
#ifndef WINML_TELEMETRY_DISABLED
// Declare TraceLogging provider
TRACELOGGING_DECLARE_PROVIDER(winml_trace_logging_provider);
#endif  //WINML_TELEMETRY_DISABLED

// WinML
#include "errors.h"
#include "NamespaceAliases.h"
#include "StringHelpers.h"
#include "WinML_Lock.h"

template <typename T>
auto unmove_ptr(T&& t) {
  return &static_cast<T&>(t);
}
