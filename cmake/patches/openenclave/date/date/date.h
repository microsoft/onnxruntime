// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <ostream>
#include <chrono>
#include <iomanip>

// Partial replacement of the date/date.h header library for use within Open Enclave.

namespace date {

template <class Duration>
    using sys_time = std::chrono::time_point<std::chrono::system_clock, Duration>;

using days = std::chrono::duration
    <int, std::ratio_multiply<std::ratio<24>, std::chrono::hours::period>>;

template <class CharT, class Traits, class Duration>
inline
typename std::enable_if
<
    !std::chrono::treat_as_floating_point<typename Duration::rep>::value &&
        std::ratio_less<typename Duration::period, days::period>::value
    , std::basic_ostream<CharT, Traits>&
>::type
operator<<(std::basic_ostream<CharT, Traits>& os, const sys_time<Duration>& tp)
{
  std::time_t tt = std::chrono::system_clock::to_time_t(tp);
  auto utc_tm = *gmtime(&tt);
  auto ms_duration = std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch());
  auto ms = ms_duration.count() % 1000000;
  // cannot use std::put_time as Open Enclave doesn't support strftime
  return os  << utc_tm.tm_year + 1900 << "-"
      << std::setfill('0') << std::setw(2) << utc_tm.tm_mon + 1 << "-" 
      << std::setfill('0') << std::setw(2) << utc_tm.tm_mday << " " 
      << std::setfill('0') << std::setw(2) << utc_tm.tm_hour << ":"
      << std::setfill('0') << std::setw(2) << utc_tm.tm_min << ":"
      << std::setfill('0') << std::setw(2) << utc_tm.tm_sec << "." 
      << std::setfill('0') << std::setw(6) << ms;
}

}  // namespace date