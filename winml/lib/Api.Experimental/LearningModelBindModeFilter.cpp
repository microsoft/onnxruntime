#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelBindModeFilter.h"

namespace WINML_EXPERIMENTALP
{
    LearningModelBindModeFilter::LearningModelBindModeFilter()
    {
        modes_.push_back(winml_experimental::LearningModelBindMode::Bound);
    }

    winml_experimental::LearningModelBindModeFilter LearningModelBindModeFilter::IncludeAll()
    {
        modes_.push_back(winml_experimental::LearningModelBindMode::Bound);
        modes_.push_back(winml_experimental::LearningModelBindMode::Unbound);
        return *this;
    }

    winml_experimental::LearningModelBindModeFilter LearningModelBindModeFilter::Include(
        winml_experimental::LearningModelBindMode const& mode)
    {
        auto found_it = std::find(modes_.begin(), modes_.end(), mode);
        if (found_it == modes_.end())
        {
            modes_.push_back(mode);
        }
        return *this;
    }

    winml_experimental::LearningModelBindModeFilter LearningModelBindModeFilter::Clear()
    {
        modes_.clear();
        return *this;
    }

    wfc::IIterator<winml_experimental::LearningModelBindMode> LearningModelBindModeFilter::First()
    {
        throw hresult_not_implemented();
    }

    winml_experimental::LearningModelBindMode LearningModelBindModeFilter::GetAt(uint32_t index)
    {
        return modes_.at(index);
    }

    uint32_t LearningModelBindModeFilter::Size()
    {
        return static_cast<uint32_t>(modes_.size());
    }

    bool LearningModelBindModeFilter::IndexOf(
        winml_experimental::LearningModelBindMode const& value,
        uint32_t& index)
    {
        index = 0;
        auto found_it = std::find(modes_.begin(), modes_.end(), value);
        if (found_it != modes_.end())
        {
            index = static_cast<uint32_t>(std::distance(modes_.begin(), found_it));
            return true;
        }
        return false;
    }

    uint32_t LearningModelBindModeFilter::GetMany(
        uint32_t startIndex,
        array_view<winml_experimental::LearningModelBindMode> items)
    {
        for (uint32_t i = 0; i < items.size(); i++)
        {
            items[i] = modes_[i + startIndex];
        }
        return items.size();
    }
}
