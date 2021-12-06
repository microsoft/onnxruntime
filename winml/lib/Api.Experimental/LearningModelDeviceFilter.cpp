#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelDeviceFilter.h"

namespace WINML_EXPERIMENTALP
{
    LearningModelDeviceFilter::LearningModelDeviceFilter()
    {
        IncludeAll();
    }

    winml_experimental::LearningModelDeviceFilter LearningModelDeviceFilter::IncludeAll()
    {
        device_kinds_.push_back(winml::LearningModelDeviceKind::Cpu);
        device_kinds_.push_back(winml::LearningModelDeviceKind::DirectX);
        return *this;
    }

    winml_experimental::LearningModelDeviceFilter LearningModelDeviceFilter::Include(
        winml::LearningModelDeviceKind const& phase)
    {
        auto found_it = std::find(device_kinds_.begin(), device_kinds_.end(), phase);
        if (found_it == device_kinds_.end())
        {
            device_kinds_.push_back(phase);
        }
        return *this;
    }

    winml_experimental::LearningModelDeviceFilter LearningModelDeviceFilter::Clear()
    {
        device_kinds_.clear();
        return *this;
    }

    wfc::IIterator<winml::LearningModelDeviceKind> LearningModelDeviceFilter::First()
    {
        throw hresult_not_implemented();
    }

    winml::LearningModelDeviceKind LearningModelDeviceFilter::GetAt(uint32_t index)
    {
        return device_kinds_.at(index);
    }

    uint32_t LearningModelDeviceFilter::Size()
    {
        return static_cast<uint32_t>(device_kinds_.size());
    }

    bool LearningModelDeviceFilter::IndexOf(
        winml::LearningModelDeviceKind const& value,
        uint32_t& index)
    {
        index = 0;
        auto found_it = std::find(device_kinds_.begin(), device_kinds_.end(), value);
        if (found_it != device_kinds_.end())
        {
            index = static_cast<uint32_t>(std::distance(device_kinds_.begin(), found_it));
            return true;
        }
        return false;
    }

    uint32_t LearningModelDeviceFilter::GetMany(
        uint32_t startIndex,
        array_view<winml::LearningModelDeviceKind> items)
    {
        for (uint32_t i = 0; i < items.size(); i++)
        {
            items[i] = device_kinds_[i + startIndex];
        }
        return items.size();
    }
}
