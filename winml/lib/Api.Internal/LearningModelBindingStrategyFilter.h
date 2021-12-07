#pragma once
#include "LearningModelBindingStrategyFilter.g.h"

namespace WINML_INTERNALP
{
    struct LearningModelBindingStrategyFilter : LearningModelBindingStrategyFilterT<LearningModelBindingStrategyFilter>
    {
        LearningModelBindingStrategyFilter();

        winml_internal::LearningModelBindingStrategyFilter IncludeAll();

        winml_internal::LearningModelBindingStrategyFilter Include(
            winml_internal::LearningModelBindingStrategy const& strategy);

        winml_internal::LearningModelBindingStrategyFilter Clear();

        wfc::IIterator<winml_internal::LearningModelBindingStrategy> First();

        winml_internal::LearningModelBindingStrategy GetAt(uint32_t index);

        uint32_t Size();

        bool IndexOf(winml_internal::LearningModelBindingStrategy const& value,
                     uint32_t& index);

        uint32_t GetMany(uint32_t startIndex,
                         array_view<winml_internal::LearningModelBindingStrategy> items);

    private:
        std::vector<winml_internal::LearningModelBindingStrategy> strategies_;

    };
}
