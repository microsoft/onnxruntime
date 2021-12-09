#include "lib/Api.Tuning/pch/pch.h"
#include "LearningModelBindingStrategyFilter.h"

namespace WINML_TUNINGP
{
    LearningModelBindingStrategyFilter::LearningModelBindingStrategyFilter()
    {
        strategies_.push_back(winml_tuning::LearningModelBindingStrategy::CreateFromShapeArrayAndDataArray);
    }

    winml_tuning::LearningModelBindingStrategyFilter LearningModelBindingStrategyFilter::IncludeAll()
    {
        Clear();
        strategies_.push_back(winml_tuning::LearningModelBindingStrategy::CreateFromShape);
        strategies_.push_back(winml_tuning::LearningModelBindingStrategy::CreateFromArray);
        strategies_.push_back(winml_tuning::LearningModelBindingStrategy::CreateFromIterable);
        strategies_.push_back(winml_tuning::LearningModelBindingStrategy::CreateFromShapeArrayAndDataArray);
        strategies_.push_back(winml_tuning::LearningModelBindingStrategy::CreateFromBuffer);
        strategies_.push_back(winml_tuning::LearningModelBindingStrategy::CreateFromD3D12Resource);
        strategies_.push_back(winml_tuning::LearningModelBindingStrategy::CreateUnbound);
        return *this;
    }

    winml_tuning::LearningModelBindingStrategyFilter LearningModelBindingStrategyFilter::Include(
        winml_tuning::LearningModelBindingStrategy const& strategy)
    {
        auto found_it = std::find(strategies_.begin(), strategies_.end(), strategy);
        if (found_it == strategies_.end())
        {
            strategies_.push_back(strategy);
        }
        return *this;
    }

    winml_tuning::LearningModelBindingStrategyFilter LearningModelBindingStrategyFilter::Clear()
    {
        strategies_.clear();
        return *this;
    }

    Windows::Foundation::Collections::IIterator<winml_tuning::LearningModelBindingStrategy> LearningModelBindingStrategyFilter::First()
    {
        throw hresult_not_implemented();
    }

    winml_tuning::LearningModelBindingStrategy LearningModelBindingStrategyFilter::GetAt(uint32_t index)
    {
        return strategies_.at(index);
    }

    uint32_t LearningModelBindingStrategyFilter::Size()
    {
        return static_cast<uint32_t>(strategies_.size());
    }

    bool LearningModelBindingStrategyFilter::IndexOf(
        winml_tuning::LearningModelBindingStrategy const& value,
        uint32_t& index)
    {
        index = 0;
        auto found_it = std::find(strategies_.begin(), strategies_.end(), value);
        if (found_it != strategies_.end())
        {
            index = static_cast<uint32_t>(std::distance(strategies_.begin(), found_it));
            return true;
        }
        return false;
    }

    uint32_t LearningModelBindingStrategyFilter::GetMany(
        uint32_t startIndex,
        array_view<winml_tuning::LearningModelBindingStrategy> items)
    {
        for (uint32_t i = 0; i < items.size(); i++)
        {
            items[i] = strategies_[i + startIndex];
        }
        return items.size();
    }
}
