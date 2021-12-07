#include "lib/Api.Internal/pch/pch.h"
#include "LearningModelPhaseFilter.h"

namespace WINML_INTERNALP
{
    LearningModelPhaseFilter::LearningModelPhaseFilter()
    {
        phases_.push_back(winml_internal::LearningModelPhase::BindInputs);
        phases_.push_back(winml_internal::LearningModelPhase::BindOutputs);
        phases_.push_back(winml_internal::LearningModelPhase::Evaluate);
        phases_.push_back(winml_internal::LearningModelPhase::FetchResults);
    }

    winml_internal::LearningModelPhaseFilter LearningModelPhaseFilter::IncludeAll()
    {
        Clear();
        phases_.push_back(winml_internal::LearningModelPhase::LoadModel);
        phases_.push_back(winml_internal::LearningModelPhase::CreateSession);
        phases_.push_back(winml_internal::LearningModelPhase::BindInputs);
        phases_.push_back(winml_internal::LearningModelPhase::BindOutputs);
        phases_.push_back(winml_internal::LearningModelPhase::Evaluate);
        phases_.push_back(winml_internal::LearningModelPhase::FetchResults);
        return *this;
    }

    winml_internal::LearningModelPhaseFilter LearningModelPhaseFilter::Include(
        winml_internal::LearningModelPhase const& phase)
    {
        auto found_it = std::find(phases_.begin(), phases_.end(), phase);
        if (found_it == phases_.end())
        {
            phases_.push_back(phase);
        }
        return *this;
    }

    winml_internal::LearningModelPhaseFilter LearningModelPhaseFilter::Clear()
    {
        phases_.clear();
        return *this;
    }

    wfc::IIterator<winml_internal::LearningModelPhase> LearningModelPhaseFilter::First()
    {
        throw hresult_not_implemented();
    }

    winml_internal::LearningModelPhase LearningModelPhaseFilter::GetAt(uint32_t index)
    {
        return phases_.at(index);
    }

    uint32_t LearningModelPhaseFilter::Size()
    {
        return static_cast<uint32_t>(phases_.size());
    }

    bool LearningModelPhaseFilter::IndexOf(
        winml_internal::LearningModelPhase const& value,
        uint32_t& index)
    {
        index = 0;
        auto found_it = std::find(phases_.begin(), phases_.end(), value);
        if (found_it != phases_.end())
        {
            index = static_cast<uint32_t>(std::distance(phases_.begin(), found_it));
            return true;
        }
        return false;
    }

    uint32_t LearningModelPhaseFilter::GetMany(
        uint32_t startIndex,
        array_view<winml_internal::LearningModelPhase> items)
    {
        for (uint32_t i = 0; i < items.size(); i++)
        {
            items[i] = phases_[i + startIndex];
        }
        return items.size();
    }
}
