// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------
#pragma once

#include "../Featurizer.h"
#include <chrono>
#include <ctime>
#include <cstdint>
#include <stdexcept>

namespace Microsoft {
namespace Featurizer {

/////////////////////////////////////////////////////////////////////////
///  \namespace     DateTimeTransformer
///  \brief         A Transformer that takes a chrono::system_clock::time_point and
///                 returns a struct with all the data split out. 
///
namespace DateTimeFeaturizer {

    /////////////////////////////////////////////////////////////////////////
    ///  \struct        TimePoint
    ///  \brief         Struct to hold various components of DateTime information 
    ///
    struct TimePoint {
        std::int32_t year = 0;
        std::uint8_t month = 0;         /* 1-12 */
        std::uint8_t day = 0;           /* 1-31 */
        std::uint8_t hour = 0;          /* 0-23 */
        std::uint8_t minute = 0;        /* 0-59 */
        std::uint8_t second = 0;        /* 0-59 */
        std::uint8_t dayOfWeek = 0;     /* 0-6 */
        std::uint16_t dayOfYear = 0;    /* 0-365 */
        std::uint8_t quarterOfYear = 0; /* 1-4 */
        std::uint8_t weekOfMonth = 0;   /* 0-4 */

        // Need default __ctor to satisfy ORT type system
        TimePoint() = default;
        TimePoint(const std::chrono::system_clock::time_point& sysTime);

        TimePoint(TimePoint&&) = default;
        TimePoint& operator=(TimePoint&&) = default;

        TimePoint(const TimePoint&) = delete;
        TimePoint& operator=(const TimePoint&) = delete;

        bool operator==(const TimePoint& o) const {
          return year == o.year &&
                 month == o.month &&
                 day == o.day &&
                 hour == o.hour &&
                 minute == o.minute &&
                 second == o.second &&
                 dayOfWeek == o.dayOfWeek &&
                 dayOfYear == o.dayOfYear &&
                 quarterOfYear == o.quarterOfYear &&
                 weekOfMonth == o.weekOfMonth;
        }

        enum { 
            JANUARY = 1, FEBRUARY, MARCH, APRIL, MAY, JUNE, 
            JULY, AUGUST, SEPTEMBER, OCTOBER, NOVEMBER, DECEMBER
        };
        enum {
            SUNDAY = 0, MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY
        };
    };

    inline TimePoint SystemToDPTimePoint(const std::chrono::system_clock::time_point& sysTime) {
        return TimePoint (sysTime);
    }

    /////////////////////////////////////////////////////////////////////////
    ///  \class         DateTimeTransformer
    ///  \brief         Transformer
    ///
    class Transformer : public Microsoft::Featurizer::Transformer<Microsoft::Featurizer::DateTimeFeaturizer::TimePoint, std::chrono::system_clock::time_point> {
        public:
            Transformer(void) = default;
            ~Transformer(void) override = default;

            Transformer(Transformer const &) = delete;
            Transformer & operator =(Transformer const &) = delete;

            Transformer(Transformer &&) = default;
            Transformer & operator =(Transformer &&) = delete;

            return_type transform(arg_type const &arg) const override;

        private:
        // ----------------------------------------------------------------------
        // |  Private Methods
        template <typename ArchiveT>
        void serialize(ArchiveT &ar, unsigned int const version);
    };

} // Namespace DateTimeFeaturizer
} // Namespace Featurizer
} // Namespace Microsoft
