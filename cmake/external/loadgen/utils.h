/* Copyright 2019 The MLPerf Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/// \file
/// \brief Various shared utility functions.

#ifndef MLPERF_LOADGEN_UTILS_H
#define MLPERF_LOADGEN_UTILS_H

#include <algorithm>
#include <chrono>
#include <string>

#include "query_sample.h"

namespace mlperf {

template <typename T>
void RemoveValue(T* container, const typename T::value_type& value_to_remove) {
  container->erase(std::remove_if(container->begin(), container->end(),
                                  [&](typename T::value_type v) {
                                    return v == value_to_remove;
                                  }),
                   container->end());
}

template <typename CountT, typename RatioT>
double DurationToSeconds(
    const std::chrono::duration<CountT, RatioT>& chrono_duration) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(
             chrono_duration)
      .count();
}

inline double QuerySampleLatencyToSeconds(QuerySampleLatency qsl) {
  return static_cast<double>(qsl) / std::nano::den;
}

template <typename DurationT>
inline DurationT SecondsToDuration(double seconds) {
  return std::chrono::duration_cast<DurationT>(
      std::chrono::duration<double>(seconds));
}

std::string CurrentDateTimeISO8601();

/// \brief Uses a format that matches the one used by SPEC power
/// measurement logging.
std::string DateTimeStringForPower(std::chrono::system_clock::time_point tp);

std::string DoubleToString(double value, int precision = 2);

bool FileExists(const std::string filename);

}  // namespace mlperf

#endif  // MLPERF_LOADGEN_UTILS_H
