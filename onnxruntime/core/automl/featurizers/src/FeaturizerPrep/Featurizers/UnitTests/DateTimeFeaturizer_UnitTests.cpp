// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------

#define CATCH_CONFIG_MAIN
#include <cstdio>
#include "gtest/gtest.h"

#include "../DateTimeFeaturizer.h"


namespace Microsoft {
namespace Featurizer {
namespace DateTimeFeaturizer {

using SysClock = std::chrono::system_clock;

TEST(DateTimeFeaturizer_DateTime, Past_1976_Nov_17__12_27_04) {
    const time_t date = 217081624;
    SysClock::time_point stp = SysClock::from_time_t(date);

    // Constructor
    TimePoint tp(stp);
    ASSERT_EQ(tp.year, 1976);
    ASSERT_EQ(tp.month, TimePoint::NOVEMBER);
    ASSERT_EQ(tp.day, 17);
    ASSERT_EQ(tp.hour, 12);
    ASSERT_EQ(tp.minute, 27);
    ASSERT_EQ(tp.second, 4);
    ASSERT_EQ(tp.dayOfWeek, TimePoint::WEDNESDAY);
    ASSERT_EQ(tp.dayOfYear, 321);
    ASSERT_EQ(tp.quarterOfYear, 4);
    ASSERT_EQ(tp.weekOfMonth, 2);

    // assignment
    TimePoint tp1 = stp;
    ASSERT_EQ(tp1.year, 1976);
    ASSERT_EQ(tp1.month, TimePoint::NOVEMBER);
    ASSERT_EQ(tp1.day, 17);

    // function
    TimePoint tp2 = SystemToDPTimePoint(stp);
    ASSERT_EQ(tp2.year, 1976);
    ASSERT_EQ(tp2.month, TimePoint::NOVEMBER);
    ASSERT_EQ(tp2.day, 17);
}

TEST(DateTimeFeaturizer_Transformer , Past_1976_Nov_17__12_27_05) {
    const time_t date = 217081625;
    SysClock::time_point stp = SysClock::from_time_t(date);
    
    Transformer dt;
    TimePoint tp = dt.transform(stp);
    ASSERT_EQ(tp.year, 1976);
    ASSERT_EQ(tp.month, TimePoint::NOVEMBER);
    ASSERT_EQ(tp.day, 17);
    ASSERT_EQ(tp.hour, 12);
    ASSERT_EQ(tp.minute, 27);
    ASSERT_EQ(tp.second, 5);
    ASSERT_EQ(tp.dayOfWeek, TimePoint::WEDNESDAY);
    ASSERT_EQ(tp.dayOfYear, 321);
    ASSERT_EQ(tp.quarterOfYear, 4);
    ASSERT_EQ(tp.weekOfMonth, 2);

}

TEST(DateTimeFeaturizer_Transformer , Future_2025_June_30) {
    const time_t date = 1751241600;
    SysClock::time_point stp = SysClock::from_time_t(date);

    Transformer dt;
    TimePoint tp = dt.transform(stp);
    ASSERT_EQ(tp.year, 2025);
    ASSERT_EQ(tp.month, TimePoint::JUNE);
    ASSERT_EQ(tp.day, 30);
    ASSERT_EQ(tp.hour, 0);
    ASSERT_EQ(tp.minute, 0);
    ASSERT_EQ(tp.second, 0);
    ASSERT_EQ(tp.dayOfWeek, TimePoint::MONDAY);
    ASSERT_EQ(tp.dayOfYear, 180);
    ASSERT_EQ(tp.quarterOfYear, 2);
    ASSERT_EQ(tp.weekOfMonth, 4);
}

#ifdef _MSC_VER
// others define system_clock::time_point as nanoseconds (64-bit),
// which rolls over somewhere around 2260. Still a couple hundred years!
TEST(DateTimeFeaturizer_Transformer , Far_Future__2998_March_2__14_03_02) {
    const time_t date = 32445842582;
    SysClock::time_point stp = SysClock::from_time_t(date);

    Transformer dt;
    TimePoint tp = dt.transform(stp);
    ASSERT_EQ(tp.year, 2998);
    ASSERT_EQ(tp.month, TimePoint::MARCH);
    ASSERT_EQ(tp.day, 2);
    ASSERT_EQ(tp.hour, 14);
    ASSERT_EQ(tp.minute, 3);
    ASSERT_EQ(tp.second, 2);
    ASSERT_EQ(tp.dayOfWeek, TimePoint::FRIDAY);
    ASSERT_EQ(tp.dayOfYear, 60);
    ASSERT_EQ(tp.quarterOfYear, 1);
    ASSERT_EQ(tp.weekOfMonth, 0);
}

#else

// msvcrt doesn't support negative time_t, so nothing before 1970
TEST(DateTimeFeaturizer_Transformer, Pre_Epoch__1776_July_4) {

    const time_t date = -6106060800;
    SysClock::time_point stp = SysClock::from_time_t(date);

    // Constructor
    Transformer dt;
    TimePoint tp = dt.transform(stp);
    ASSERT_EQ(tp.year, 1776);
    ASSERT_EQ(tp.month, TimePoint::JULY);
    ASSERT_EQ(tp.day, 4);
}
#endif /* _MSC_VER */
} // namespace DateTimeFeaturizer
} // namespace Featurizer
} // namespace Microsoft
