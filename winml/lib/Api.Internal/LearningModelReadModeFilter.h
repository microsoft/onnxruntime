#pragma once
#include "LearningModelReadModeFilter.g.h"

namespace WINML_INTERNALP
{
    struct LearningModelReadModeFilter : LearningModelReadModeFilterT<LearningModelReadModeFilter>
    {
        LearningModelReadModeFilter();

        winml_internal::LearningModelReadModeFilter IncludeAll();

        winml_internal::LearningModelReadModeFilter Include(
            winml_internal::LearningModelReadMode const& mode);

        winml_internal::LearningModelReadModeFilter Clear();

        wfc::IIterator<winml_internal::LearningModelReadMode> First();

        winml_internal::LearningModelReadMode GetAt(uint32_t index);

        uint32_t Size();

        bool IndexOf(winml_internal::LearningModelReadMode const& value, uint32_t& index);

        uint32_t GetMany(
            uint32_t startIndex,
            array_view<winml_internal::LearningModelReadMode> items);

    private:
        std::vector<winml_internal::LearningModelReadMode> modes_;

    };
}
