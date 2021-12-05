#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelReadModeFilter.h"

namespace WINML_EXPERIMENTALP
{
    LearningModelReadModeFilter::LearningModelReadModeFilter()
    {
        modes_.push_back(winml_experimental::LearningModelReadMode::GetAsVectorView);
    }

    winml_experimental::LearningModelReadModeFilter LearningModelReadModeFilter::IncludeAll()
    {
        modes_.push_back(winml_experimental::LearningModelReadMode::GetAsVectorView);
        modes_.push_back(winml_experimental::LearningModelReadMode::GetFromNativeBufferAccess);
        modes_.push_back(winml_experimental::LearningModelReadMode::GetFromMemoryBufferReferenceAccess);
        modes_.push_back(winml_experimental::LearningModelReadMode::GetAsD3D12Resource);
        return *this;
    }

    winml_experimental::LearningModelReadModeFilter LearningModelReadModeFilter::Include(
        winml_experimental::LearningModelReadMode const& mode)
    {
        auto found_it = std::find(modes_.begin(), modes_.end(), mode);
        if (found_it == modes_.end())
        {
            modes_.push_back(mode);
        }
        return *this;
    }

    winml_experimental::LearningModelReadModeFilter LearningModelReadModeFilter::Clear()
    {
        modes_.clear();
        return *this;
    }

    wfc::IIterator<winml_experimental::LearningModelReadMode> LearningModelReadModeFilter::First()
    {
        throw hresult_not_implemented();
    }

    winml_experimental::LearningModelReadMode LearningModelReadModeFilter::GetAt(uint32_t index)
    {
        return modes_.at(index);
    }

    uint32_t LearningModelReadModeFilter::Size()
    {
        return static_cast<uint32_t>(modes_.size());
    }

    bool LearningModelReadModeFilter::IndexOf(
        winml_experimental::LearningModelReadMode const& value,
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

    uint32_t LearningModelReadModeFilter::GetMany(
        uint32_t startIndex,
        array_view<winml_experimental::LearningModelReadMode> items)
    {
        for (uint32_t i = 0; i < items.size(); i++)
        {
            items[i] = modes_[i + startIndex];
        }
        return items.size();
    }
}
