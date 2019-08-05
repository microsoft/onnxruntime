// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------

#define CATCH_CONFIG_MAIN
#include "catch.hpp"
#include <cstdio>
#include "../DateTimeFeaturizer.h"

namespace Microsoft {
namespace Featurizer {
namespace DateTimeFeaturizer {

using SysClock = std::chrono::system_clock;

TEST_CASE("Past - 1976 Nov 17, 12:27:04", "[DateTimeFeaturizer][DateTime]") {
    const time_t date = 217081624;
    SysClock::time_point stp = SysClock::from_time_t(date);
    
    // Constructor
    TimePoint tp(stp);
    CHECK(tp.year == 1976);
    CHECK(tp.month == TimePoint::NOVEMBER);
    CHECK(tp.day == 17);
    CHECK(tp.hour == 12);
    CHECK(tp.minute == 27);
    CHECK(tp.second == 4);
    CHECK(tp.dayOfWeek == TimePoint::WEDNESDAY);
    CHECK(tp.dayOfYear == 321);
    CHECK(tp.quarterOfYear == 4);
    CHECK(tp.weekOfMonth == 2);

    // assignment
    TimePoint tp1 = stp;
    CHECK(tp1.year == 1976);
    CHECK(tp1.month == TimePoint::NOVEMBER);
    CHECK(tp1.day == 17);

    // function
    TimePoint tp2 = SystemToDPTimePoint(stp);
    CHECK(tp2.year == 1976);
    CHECK(tp2.month == TimePoint::NOVEMBER);
    CHECK(tp2.day == 17);
}

TEST_CASE("Past - 1976 Nov 17, 12:27:05", "[DateTimeFeaturizer][Transformer]") {
    const time_t date = 217081625;
    SysClock::time_point stp = SysClock::from_time_t(date);
    
    Transformer dt;
    TimePoint tp = dt.transform(stp);
    CHECK(tp.year == 1976);
    CHECK(tp.month == TimePoint::NOVEMBER);
    CHECK(tp.day == 17);
    CHECK(tp.hour == 12);
    CHECK(tp.minute == 27);
    CHECK(tp.second == 5);
    CHECK(tp.dayOfWeek == TimePoint::WEDNESDAY);
    CHECK(tp.dayOfYear == 321);
    CHECK(tp.quarterOfYear == 4);
    CHECK(tp.weekOfMonth == 2);

}

TEST_CASE("Future - 2025 June 30", "[DateTimeFeaturizer][Transformer]") {
    const time_t date = 1751241600;
    SysClock::time_point stp = SysClock::from_time_t(date);

    Transformer dt;
    TimePoint tp = dt.transform(stp);
    CHECK(tp.year == 2025);
    CHECK(tp.month == TimePoint::JUNE);
    CHECK(tp.day == 30);
    CHECK(tp.hour == 0);
    CHECK(tp.minute == 0);
    CHECK(tp.second == 0);
    CHECK(tp.dayOfWeek == TimePoint::MONDAY);
    CHECK(tp.dayOfYear == 180);
    CHECK(tp.quarterOfYear == 2);
    CHECK(tp.weekOfMonth == 4);
}

#ifdef _MSC_VER
// others define system_clock::time_point as nanoseconds (64-bit),
// which rolls over somewhere around 2260. Still a couple hundred years!
TEST_CASE("Far Future - 2998 March 2, 14:03:02", "[DateTimeFeaturizer][Transformer]") {
    const time_t date = 32445842582;
    SysClock::time_point stp = SysClock::from_time_t(date);

    Transformer dt;
    TimePoint tp = dt.transform(stp);
    CHECK(tp.year == 2998);
    CHECK(tp.month == TimePoint::MARCH);
    CHECK(tp.day == 2);
    CHECK(tp.hour == 14);
    CHECK(tp.minute == 3);
    CHECK(tp.second == 2);
    CHECK(tp.dayOfWeek == TimePoint::FRIDAY);
    CHECK(tp.dayOfYear == 60);
    CHECK(tp.quarterOfYear == 1);
    CHECK(tp.weekOfMonth == 0);
}

#else

// msvcrt doesn't support negative time_t, so nothing before 1970
TEST_CASE("Pre-Epoch - 1776 July 4", "[DateTimeFeaturizer][Transformer]")
{
    const time_t date = -6106060800;
    SysClock::time_point stp = SysClock::from_time_t(date);

    // Constructor
    Transformer dt;
    TimePoint tp = dt.transform(stp);
    CHECK(tp.year == 1776);
    CHECK(tp.month == TimePoint::JULY);
    CHECK(tp.day == 4);
}
#endif /* _MSVCRT */
} // namespace DateTimeFeaturizer
} // namespace Featurizer
} // namespace Microsoft
