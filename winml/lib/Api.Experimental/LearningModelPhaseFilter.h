#pragma once
#include "LearningModelPhaseFilter.g.h"

namespace WINML_EXPERIMENTALP
{
    struct LearningModelPhaseFilter : LearningModelPhaseFilterT<LearningModelPhaseFilter>
    {
        LearningModelPhaseFilter();

        winml_experimental::LearningModelPhaseFilter IncludeAll();

        winml_experimental::LearningModelPhaseFilter Include(
            winml_experimental::LearningModelPhase const& strategy);

        winml_experimental::LearningModelPhaseFilter Clear();

        wfc::IIterator<winml_experimental::LearningModelPhase> First();

        winml_experimental::LearningModelPhase GetAt(uint32_t index);

        uint32_t Size();

        bool IndexOf(winml_experimental::LearningModelPhase const& value, uint32_t& index);

        uint32_t GetMany(
            uint32_t startIndex,
            array_view<winml_experimental::LearningModelPhase> items);

    private:
        std::vector<winml_experimental::LearningModelPhase> phases_;

    };
}
