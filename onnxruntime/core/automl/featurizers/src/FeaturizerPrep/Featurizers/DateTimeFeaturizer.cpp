// ----------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License
// ----------------------------------------------------------------------
#include "DateTimeFeaturizer.h"

#ifdef _MSC_VER
inline struct tm *gmtime_r(time_t const* const timer, struct tm* const  result) {
    return gmtime_s(result, timer) == 0 ? result : nullptr;
}

#endif

namespace Microsoft {
namespace Featurizer {

namespace DateTimeFeaturizer {
    
    TimePoint::TimePoint(const std::chrono::system_clock::time_point& sysTime) {
        // Get to a tm to get what we need.
        // Eventually C++202x will have expanded chrono support that might 
        // have what we need, but not yet!
        std::tm tmt;
        time_t tt = std::chrono::system_clock::to_time_t(sysTime);
        std::tm* res = gmtime_r(&tt, &tmt);
        if (res) {
            year = static_cast<std::int32_t>(tmt.tm_year) + 1900;
            month = static_cast<std::uint8_t>(tmt.tm_mon) + 1;
            day = static_cast<std::uint8_t>(tmt.tm_mday);
            hour = static_cast<std::uint8_t>(tmt.tm_hour);
            minute = static_cast<std::uint8_t>(tmt.tm_min);
            second = static_cast<std::uint8_t>(tmt.tm_sec);
            dayOfWeek = static_cast<std::uint8_t>(tmt.tm_wday);
            dayOfYear = static_cast<std::uint16_t>(tmt.tm_yday);
            quarterOfYear = (month + 2) / 3;
            weekOfMonth = (day - 1) / 7;
        }
        else
        {
            if (tt < 0) {
                throw std::invalid_argument("Dates prior to 1970 are not supported.");
            }
            else {
                throw std::invalid_argument("Unknown error converting input date.");
            }
        }
    }

    Transformer::return_type Transformer::transform(arg_type const &arg) const /*override*/ {
        return Microsoft::Featurizer::DateTimeFeaturizer::TimePoint(arg);
    }


} // namespace DateTimeFeaturizer
} // namespace Featurizer
} // namespace Microsoft
