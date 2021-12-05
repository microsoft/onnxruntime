#pragma once
#include "LearningModelBindingStrategyFilter.g.h"

namespace WINML_EXPERIMENTALP
{
    struct LearningModelBindingStrategyFilter : LearningModelBindingStrategyFilterT<LearningModelBindingStrategyFilter>
    {
        LearningModelBindingStrategyFilter();

        winml_experimental::LearningModelBindingStrategyFilter IncludeAll();

        winml_experimental::LearningModelBindingStrategyFilter Include(
            winml_experimental::LearningModelBindingStrategy const& strategy);

        winml_experimental::LearningModelBindingStrategyFilter Clear();

        wfc::IIterator<winml_experimental::LearningModelBindingStrategy> First();

        winml_experimental::LearningModelBindingStrategy GetAt(uint32_t index);

        uint32_t Size();

        bool IndexOf(winml_experimental::LearningModelBindingStrategy const& value,
                     uint32_t& index);

        uint32_t GetMany(uint32_t startIndex,
                         array_view<winml_experimental::LearningModelBindingStrategy> items);

    private:
        std::vector<winml_experimental::LearningModelBindingStrategy> strategies_;

    };
}
