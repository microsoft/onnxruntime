// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#pragma once

#include <algorithm>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <thread>

inline int DnnlCalcNumThreads() {
  int num_threads = 0;
#if _WIN32
  // Indeed 64 should be enough. However, it's harmless to have a little more.
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer[256];
  DWORD returnLength = sizeof(buffer);
  if (GetLogicalProcessorInformation(buffer, &returnLength) == FALSE) {
    num_threads = std::thread::hardware_concurrency();
  } else {
    int count = (int)(returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    for (int i = 0; i != count; ++i) {
      if (buffer[i].Relationship == RelationProcessorCore) {
        ++num_threads;
      }
    }
  }
#endif
  if (!num_threads)
    num_threads = std::thread::hardware_concurrency();

  return num_threads;
}