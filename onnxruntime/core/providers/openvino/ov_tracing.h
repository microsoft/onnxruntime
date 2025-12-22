// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#ifdef _WIN32
#include <windows.h>
#include <TraceLoggingProvider.h>
#include <winmeta.h>

#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>
#include <optional>
#include <algorithm>
#include "core/providers/openvino/contexts.h"

TRACELOGGING_DECLARE_PROVIDER(ov_tracing_provider_handle);

namespace onnxruntime {
namespace openvino_ep {

class OVTracing {
 public:
  static OVTracing& Instance();
  bool IsEnabled() const;
  unsigned char Level() const;
  UINT64 Keyword() const;

  void LogAllRuntimeOptions(uint32_t session_id, const SessionContext& ctx) const;

  using EtwInternalCallback = std::function<void(
      LPCGUID, ULONG, UCHAR, ULONGLONG, ULONGLONG, PEVENT_FILTER_DESCRIPTOR, PVOID)>;
  static void RegisterInternalCallback(const EtwInternalCallback& callback);
  static void UnregisterInternalCallback(const EtwInternalCallback& callback);

 private:
  OVTracing();
  ~OVTracing();
  OVTracing(const OVTracing&) = delete;
  OVTracing& operator=(const OVTracing&) = delete;
  OVTracing(OVTracing&&) = delete;
  OVTracing& operator=(OVTracing&&) = delete;

  static std::mutex mutex_;
  static uint32_t global_register_count_;
  static bool enabled_;
  static std::vector<const EtwInternalCallback*> callbacks_;
  static std::mutex callbacks_mutex_;
  static std::mutex provider_change_mutex_;
  static UCHAR level_;
  static ULONGLONG keyword_;

  static void InvokeCallbacks(LPCGUID, ULONG, UCHAR, ULONGLONG, ULONGLONG, PEVENT_FILTER_DESCRIPTOR, PVOID);
  static void NTAPI ORT_TL_EtwEnableCallback(_In_ LPCGUID, _In_ ULONG, _In_ UCHAR, _In_ ULONGLONG,
                                             _In_ ULONGLONG, _In_opt_ PEVENT_FILTER_DESCRIPTOR, _In_opt_ PVOID);
};

}  // namespace openvino_ep
}  // namespace onnxruntime

#endif  // defined(_WIN32)
