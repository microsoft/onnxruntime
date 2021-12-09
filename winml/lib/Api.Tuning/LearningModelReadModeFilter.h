#pragma once
#include "LearningModelReadModeFilter.g.h"

namespace WINML_TUNINGP
{
    struct LearningModelReadModeFilter : LearningModelReadModeFilterT<LearningModelReadModeFilter>
    {
        LearningModelReadModeFilter();

        winml_tuning::LearningModelReadModeFilter IncludeAll();

        winml_tuning::LearningModelReadModeFilter Include(
            winml_tuning::LearningModelReadMode const& mode);

        winml_tuning::LearningModelReadModeFilter Clear();

        wfc::IIterator<winml_tuning::LearningModelReadMode> First();

        winml_tuning::LearningModelReadMode GetAt(uint32_t index);

        uint32_t Size();

        bool IndexOf(winml_tuning::LearningModelReadMode const& value, uint32_t& index);

        uint32_t GetMany(
            uint32_t startIndex,
            array_view<winml_tuning::LearningModelReadMode> items);

    private:
        std::vector<winml_tuning::LearningModelReadMode> modes_;

    };
}
