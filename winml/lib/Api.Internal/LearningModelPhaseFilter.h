#pragma once
#include "LearningModelPhaseFilter.g.h"

namespace WINML_INTERNALP
{
    struct LearningModelPhaseFilter : LearningModelPhaseFilterT<LearningModelPhaseFilter>
    {
        LearningModelPhaseFilter();

        winml_internal::LearningModelPhaseFilter IncludeAll();

        winml_internal::LearningModelPhaseFilter Include(
            winml_internal::LearningModelPhase const& strategy);

        winml_internal::LearningModelPhaseFilter Clear();

        wfc::IIterator<winml_internal::LearningModelPhase> First();

        winml_internal::LearningModelPhase GetAt(uint32_t index);

        uint32_t Size();

        bool IndexOf(winml_internal::LearningModelPhase const& value, uint32_t& index);

        uint32_t GetMany(
            uint32_t startIndex,
            array_view<winml_internal::LearningModelPhase> items);

    private:
        std::vector<winml_internal::LearningModelPhase> phases_;

    };
}
