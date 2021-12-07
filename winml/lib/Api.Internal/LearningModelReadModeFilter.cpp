#include "lib/Api.Internal/pch/pch.h"
#include "LearningModelReadModeFilter.h"

namespace WINML_INTERNALP
{
    LearningModelReadModeFilter::LearningModelReadModeFilter()
    {
        modes_.push_back(winml_internal::LearningModelReadMode::GetAsVectorView);
    }

    winml_internal::LearningModelReadModeFilter LearningModelReadModeFilter::IncludeAll()
    {
        Clear();
        modes_.push_back(winml_internal::LearningModelReadMode::GetAsVectorView);
        modes_.push_back(winml_internal::LearningModelReadMode::GetFromNativeBufferAccess);
        modes_.push_back(winml_internal::LearningModelReadMode::GetFromMemoryBufferReferenceAccess);
        modes_.push_back(winml_internal::LearningModelReadMode::GetAsD3D12Resource);
        return *this;
    }

    winml_internal::LearningModelReadModeFilter LearningModelReadModeFilter::Include(
        winml_internal::LearningModelReadMode const& mode)
    {
        auto found_it = std::find(modes_.begin(), modes_.end(), mode);
        if (found_it == modes_.end())
        {
            modes_.push_back(mode);
        }
        return *this;
    }

    winml_internal::LearningModelReadModeFilter LearningModelReadModeFilter::Clear()
    {
        modes_.clear();
        return *this;
    }

    wfc::IIterator<winml_internal::LearningModelReadMode> LearningModelReadModeFilter::First()
    {
        throw hresult_not_implemented();
    }

    winml_internal::LearningModelReadMode LearningModelReadModeFilter::GetAt(uint32_t index)
    {
        return modes_.at(index);
    }

    uint32_t LearningModelReadModeFilter::Size()
    {
        return static_cast<uint32_t>(modes_.size());
    }

    bool LearningModelReadModeFilter::IndexOf(
        winml_internal::LearningModelReadMode const& value,
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
        array_view<winml_internal::LearningModelReadMode> items)
    {
        for (uint32_t i = 0; i < items.size(); i++)
        {
            items[i] = modes_[i + startIndex];
        }
        return items.size();
    }
}
