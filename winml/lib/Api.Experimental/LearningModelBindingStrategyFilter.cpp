#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelBindingStrategyFilter.h"

namespace WINML_EXPERIMENTALP
{
    LearningModelBindingStrategyFilter::LearningModelBindingStrategyFilter()
    {
        strategies_.push_back(winml_experimental::LearningModelBindingStrategy::CreateFromShapeArrayAndDataArray);
    }

    winml_experimental::LearningModelBindingStrategyFilter LearningModelBindingStrategyFilter::IncludeAll()
    {
        strategies_.push_back(winml_experimental::LearningModelBindingStrategy::CreateFromShape);
        strategies_.push_back(winml_experimental::LearningModelBindingStrategy::CreateFromArray);
        strategies_.push_back(winml_experimental::LearningModelBindingStrategy::CreateFromIterable);
        strategies_.push_back(winml_experimental::LearningModelBindingStrategy::CreateFromShapeArrayAndDataArray);
        strategies_.push_back(winml_experimental::LearningModelBindingStrategy::CreateFromBuffer);
        strategies_.push_back(winml_experimental::LearningModelBindingStrategy::CreateFromD3D12Resource);
        strategies_.push_back(winml_experimental::LearningModelBindingStrategy::CreateUnbound);
        return *this;
    }

    winml_experimental::LearningModelBindingStrategyFilter LearningModelBindingStrategyFilter::Include(
        winml_experimental::LearningModelBindingStrategy const& strategy)
    {
        auto found_it = std::find(strategies_.begin(), strategies_.end(), strategy);
        if (found_it == strategies_.end())
        {
            strategies_.push_back(strategy);
        }
        return *this;
    }

    winml_experimental::LearningModelBindingStrategyFilter LearningModelBindingStrategyFilter::Clear()
    {
        strategies_.clear();
        return *this;
    }

    Windows::Foundation::Collections::IIterator<winml_experimental::LearningModelBindingStrategy> LearningModelBindingStrategyFilter::First()
    {
        throw hresult_not_implemented();
    }

    winml_experimental::LearningModelBindingStrategy LearningModelBindingStrategyFilter::GetAt(uint32_t index)
    {
        return strategies_.at(index);
    }

    uint32_t LearningModelBindingStrategyFilter::Size()
    {
        return static_cast<uint32_t>(strategies_.size());
    }

    bool LearningModelBindingStrategyFilter::IndexOf(
        winml_experimental::LearningModelBindingStrategy const& value,
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
        array_view<winml_experimental::LearningModelBindingStrategy> items)
    {
        for (uint32_t i = 0; i < items.size(); i++)
        {
            items[i] = strategies_[i + startIndex];
        }
        return items.size();
    }
}
