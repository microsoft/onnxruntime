#pragma once
#include "LearningModelReadModeFilter.g.h"

namespace WINML_EXPERIMENTALP
{
    struct LearningModelReadModeFilter : LearningModelReadModeFilterT<LearningModelReadModeFilter>
    {
        LearningModelReadModeFilter();

        winml_experimental::LearningModelReadModeFilter IncludeAll();

        winml_experimental::LearningModelReadModeFilter Include(
            winml_experimental::LearningModelReadMode const& mode);

        winml_experimental::LearningModelReadModeFilter Clear();

        wfc::IIterator<winml_experimental::LearningModelReadMode> First();

        winml_experimental::LearningModelReadMode GetAt(uint32_t index);

        uint32_t Size();

        bool IndexOf(winml_experimental::LearningModelReadMode const& value, uint32_t& index);

        uint32_t GetMany(
            uint32_t startIndex,
            array_view<winml_experimental::LearningModelReadMode> items);

    private:
        std::vector<winml_experimental::LearningModelReadMode> modes_;

    };
}
