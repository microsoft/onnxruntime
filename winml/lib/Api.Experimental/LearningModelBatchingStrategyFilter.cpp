#include "lib/Api.Experimental/pch/pch.h"
#include "LearningModelBatchingStrategyFilter.h"

namespace WINML_EXPERIMENTALP {

    LearningModelBatchingStrategyFilter::LearningModelBatchingStrategyFilter() : batch_sizes_()
    {
        start_ = 1;
        stride_ = 1;
        total_ = 1;

        CreateVector();
    }

    void LearningModelBatchingStrategyFilter::CreateVector()
    {
        batch_sizes_.clear();
        for (uint32_t i = 0; i < total_; i++) {
            batch_sizes_.push_back(start_ + stride_* i);
        }
    }

    uint32_t LearningModelBatchingStrategyFilter::BatchSizeStart()
    {
        return start_;
    }

    void LearningModelBatchingStrategyFilter::BatchSizeStart(uint32_t value)
    {
        start_ = value;
        CreateVector();
    }

    uint32_t LearningModelBatchingStrategyFilter::BatchSizeStride()
    {
        return stride_;
    }

    void LearningModelBatchingStrategyFilter::BatchSizeStride(uint32_t value)
    {
        stride_ = value;
        CreateVector();
    }

    uint32_t LearningModelBatchingStrategyFilter::BatchSizeTotal()
    {
        return total_;
    }

    void LearningModelBatchingStrategyFilter::BatchSizeTotal(uint32_t value)
    {
        total_ = value;
        CreateVector();
    }

    Windows::Foundation::Collections::IIterator<uint32_t> LearningModelBatchingStrategyFilter::First()
    {
        throw hresult_not_implemented();
    }

    uint32_t LearningModelBatchingStrategyFilter::GetAt(uint32_t index)
    {
        return batch_sizes_.at(index);
    }

    uint32_t LearningModelBatchingStrategyFilter::Size()
    {
        return static_cast<uint32_t>(batch_sizes_.size());
    }

    bool LearningModelBatchingStrategyFilter::IndexOf(uint32_t const& value, uint32_t& index)
    {
        index = 0;
        auto found_it = std::find(batch_sizes_.begin(), batch_sizes_.end(), value);
        if (found_it != batch_sizes_.end())
        {
            index = static_cast<uint32_t>(std::distance(batch_sizes_.begin(), found_it));
            return true;
        }
        return false;
    }

    uint32_t LearningModelBatchingStrategyFilter::GetMany(uint32_t startIndex, array_view<uint32_t> items)
    {
      for (uint32_t i = 0; i < items.size(); i++) {
        items[i] = batch_sizes_[i + startIndex];
      }
      return items.size();
    }
}
