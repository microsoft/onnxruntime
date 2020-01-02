// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This header contains shared definitions for reading and writing
// via C/C++ API Opaque data types that are registered within ORT
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
// This structure is used to initialize and read
// OrtValue of opaque(com.microsoft.featurizers,DateTimeFeaturizer_TimePoint)
struct DateTimeFeaturizerTimePointData {
  int32_t year;
  uint8_t month;
  uint8_t day;
  uint8_t hour;
  uint8_t minute;
  uint8_t second;
  uint8_t dayOfWeek;
  uint16_t dayOfYear;
  uint8_t quarterOfYear;
  uint8_t weekOfMonth;
};

#ifdef __cplusplus
} // extern "C"
#endif
