#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelPhaseFilter.h"

namespace WINML_EXPERIMENTALP
{
    LearningModelPhaseFilter::LearningModelPhaseFilter()
    {
        phases_.push_back(winml_experimental::LearningModelPhase::Evaluate);
    }

    winml_experimental::LearningModelPhaseFilter LearningModelPhaseFilter::IncludeAll()
    {
        phases_.push_back(winml_experimental::LearningModelPhase::LoadModel);
        phases_.push_back(winml_experimental::LearningModelPhase::CreateSession);
        phases_.push_back(winml_experimental::LearningModelPhase::BindInputs);
        phases_.push_back(winml_experimental::LearningModelPhase::BindOutputs);
        phases_.push_back(winml_experimental::LearningModelPhase::Evaluate);
        phases_.push_back(winml_experimental::LearningModelPhase::FetchResults);
        return *this;
    }

    winml_experimental::LearningModelPhaseFilter LearningModelPhaseFilter::Include(
        winml_experimental::LearningModelPhase const& phase)
    {
        auto found_it = std::find(phases_.begin(), phases_.end(), phase);
        if (found_it == phases_.end())
        {
            phases_.push_back(phase);
        }
        return *this;
    }

    winml_experimental::LearningModelPhaseFilter LearningModelPhaseFilter::Clear()
    {
        phases_.clear();
        return *this;
    }

    wfc::IIterator<winml_experimental::LearningModelPhase> LearningModelPhaseFilter::First()
    {
        throw hresult_not_implemented();
    }

    winml_experimental::LearningModelPhase LearningModelPhaseFilter::GetAt(uint32_t index)
    {
        return phases_.at(index);
    }

    uint32_t LearningModelPhaseFilter::Size()
    {
        return static_cast<uint32_t>(phases_.size());
    }

    bool LearningModelPhaseFilter::IndexOf(
        winml_experimental::LearningModelPhase const& value,
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
        array_view<winml_experimental::LearningModelPhase> items)
    {
        for (uint32_t i = 0; i < items.size(); i++)
        {
            items[i] = phases_[i + startIndex];
        }
        return items.size();
    }
}
