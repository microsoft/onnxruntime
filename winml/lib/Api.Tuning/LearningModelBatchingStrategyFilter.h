#pragma once
#include "LearningModelBatchingStrategyFilter.g.h"

namespace WINML_TUNINGP
{
    struct LearningModelBatchingStrategyFilter : LearningModelBatchingStrategyFilterT<LearningModelBatchingStrategyFilter>
    {
        LearningModelBatchingStrategyFilter();

        uint32_t BatchSizeStart();
        void BatchSizeStart(uint32_t value);

        uint32_t BatchSizeStride();
        void BatchSizeStride(uint32_t value);

        uint32_t BatchSizeTotal();
        void BatchSizeTotal(uint32_t value);

        wfc::IIterator<uint32_t> First();
        uint32_t GetAt(uint32_t index);
        uint32_t Size();
        bool IndexOf(uint32_t const& value, uint32_t& index);
        uint32_t GetMany(uint32_t startIndex, array_view<uint32_t> items);

    private:
        void CreateVector();

    private:
        uint32_t start_;
        uint32_t stride_;
        uint32_t total_;
        std::vector<uint32_t> batch_sizes_;
    };
}
