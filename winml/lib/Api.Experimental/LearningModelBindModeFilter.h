#pragma once
#include "LearningModelBindModeFilter.g.h"

namespace WINML_EXPERIMENTALP
{
    struct LearningModelBindModeFilter : LearningModelBindModeFilterT<LearningModelBindModeFilter>
    {
        LearningModelBindModeFilter();

        winml_experimental::LearningModelBindModeFilter IncludeAll();

        winml_experimental::LearningModelBindModeFilter Include(
            winml_experimental::LearningModelBindMode const& mode);

        winml_experimental::LearningModelBindModeFilter Clear();

        wfc::IIterator<winml_experimental::LearningModelBindMode> First();

        winml_experimental::LearningModelBindMode GetAt(uint32_t index);

        uint32_t Size();

        bool IndexOf(winml_experimental::LearningModelBindMode const& value, uint32_t& index);

        uint32_t GetMany(
            uint32_t startIndex,
            array_view<winml_experimental::LearningModelBindMode> items);

    private:
        std::vector<winml_experimental::LearningModelBindMode> modes_;

    };
}
