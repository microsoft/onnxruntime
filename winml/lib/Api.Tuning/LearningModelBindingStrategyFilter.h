#pragma once
#include "LearningModelBindingStrategyFilter.g.h"

namespace WINML_TUNINGP
{
    struct LearningModelBindingStrategyFilter : LearningModelBindingStrategyFilterT<LearningModelBindingStrategyFilter>
    {
        LearningModelBindingStrategyFilter();

        winml_tuning::LearningModelBindingStrategyFilter IncludeAll();

        winml_tuning::LearningModelBindingStrategyFilter Include(
            winml_tuning::LearningModelBindingStrategy const& strategy);

        winml_tuning::LearningModelBindingStrategyFilter Clear();

        wfc::IIterator<winml_tuning::LearningModelBindingStrategy> First();

        winml_tuning::LearningModelBindingStrategy GetAt(uint32_t index);

        uint32_t Size();

        bool IndexOf(winml_tuning::LearningModelBindingStrategy const& value,
                     uint32_t& index);

        uint32_t GetMany(uint32_t startIndex,
                         array_view<winml_tuning::LearningModelBindingStrategy> items);

    private:
        std::vector<winml_tuning::LearningModelBindingStrategy> strategies_;

    };
}
