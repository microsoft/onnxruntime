#pragma once
#include "LearningModelPhaseFilter.g.h"

namespace WINML_TUNINGP
{
    struct LearningModelPhaseFilter : LearningModelPhaseFilterT<LearningModelPhaseFilter>
    {
        LearningModelPhaseFilter();

        winml_tuning::LearningModelPhaseFilter IncludeAll();

        winml_tuning::LearningModelPhaseFilter Include(
            winml_tuning::LearningModelPhase const& strategy);

        winml_tuning::LearningModelPhaseFilter Clear();

        wfc::IIterator<winml_tuning::LearningModelPhase> First();

        winml_tuning::LearningModelPhase GetAt(uint32_t index);

        uint32_t Size();

        bool IndexOf(winml_tuning::LearningModelPhase const& value, uint32_t& index);

        uint32_t GetMany(
            uint32_t startIndex,
            array_view<winml_tuning::LearningModelPhase> items);

    private:
        std::vector<winml_tuning::LearningModelPhase> phases_;

    };
}
