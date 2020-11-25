// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <windows.h>
#include <stdio.h>
#include <wbemidl.h>
#include <wmistr.h>
#include <evntrace.h>
#include <assert.h>
#include <tdh.h>
#include <stdexcept>
#include <vector>
#include <sstream>
#include <in6addr.h>
#include <unordered_map>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>

void OrtEventHandler(EVENT_RECORD* pEventRecord, void* pContext);

class LoggingEventRecord {
 private:
  std::vector<char> buffer_;
  EVENT_RECORD* event_record_;

 public:
  const TRACE_EVENT_INFO* GetEventInfo() const { return (const TRACE_EVENT_INFO*)buffer_.data(); }
  TRACE_EVENT_INFO* GetEventInfo() { return (TRACE_EVENT_INFO*)buffer_.data(); }

  const wchar_t* GetTaskName() const {
    const TRACE_EVENT_INFO* p = GetEventInfo();
    return (const wchar_t*)(buffer_.data() + p->TaskNameOffset);
  }

  static LoggingEventRecord CreateLoggingEventRecord(EVENT_RECORD* pEvent, DWORD& status);  
};

struct OpStat {
  std::wstring name;
  size_t count = 0;
  uint64_t total_time = 0;
};

struct ProfilingInfo {
  int ortrun_count = 0;
  int ortrun_end_count = 0;
  int session_count = 0;
  bool session_started = false;
  bool session_ended = false;
  LARGE_INTEGER op_start_time;

  std::unordered_map<std::wstring, OpStat> op_stat;
  std::vector<ULONG64> time_per_run;
};


