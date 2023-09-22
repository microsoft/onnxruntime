// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

using namespace Dml;

TensorDesc::TensorDesc(
    DML_TENSOR_DATA_TYPE dataType,
    gsl::span<const uint32_t> sizes,
    std::optional<gsl::span<const uint32_t>> strides,
    uint32_t guaranteedBaseOffsetAlignment
    )
{
    m_tensorType = DML_TENSOR_TYPE_BUFFER;
    m_bufferTensorDesc.DataType = dataType;
    m_mlOperatorTensorDataType = GetMlDataTypeFromDmlDataType(dataType);

    ML_CHECK_VALID_ARGUMENT(gsl::narrow_cast<size_t>(sizes.size()) <= std::size(m_sizes));
    std::copy(sizes.begin(), sizes.end(), m_sizes);
    m_bufferTensorDesc.Sizes = m_sizes;

    if (strides)
    {
        ML_CHECK_VALID_ARGUMENT(strides->size() == sizes.size());
        std::copy(strides->begin(), strides->end(), m_strides);
        m_bufferTensorDesc.Strides = m_strides;
    }

    m_bufferTensorDesc.DimensionCount = gsl::narrow_cast<uint32_t>(sizes.size());
    m_bufferTensorDesc.Flags = DML_TENSOR_FLAG_NONE;

    m_bufferTensorDesc.GuaranteedBaseOffsetAlignment = guaranteedBaseOffsetAlignment;
    m_bufferTensorDesc.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(
        m_bufferTensorDesc.DataType,
        m_bufferTensorDesc.DimensionCount,
        m_sizes,
        strides ? m_strides : nullptr
        );
}

TensorDesc::TensorDesc(MLOperatorTensorDataType dataType)
{
    m_mlOperatorTensorDataType = dataType;
    m_bufferTensorDesc.DataType = GetDmlDataTypeFromMlDataType(dataType);
    // Leave all other fields deferred, including m_tensorType = DML_TENSOR_TYPE_INVALID.
}

TensorDesc::TensorDesc(
    MLOperatorTensorDataType dataType,
    gsl::span<const uint32_t> dimensions, // Desired dimensions
    gsl::span<const uint32_t> nonBroadcastDimensions, // Actual physical dimensions
    int32_t coerceAxis,
    int32_t placement, // Adjustment offset of the passed dimensions within the minDimensionCount.
    int32_t leftAlignedDimensionCount, // Number of dimensions that remain left aligned when expanded to minimum count (INT32_MAX means all, 0 means all right aligned).
    uint32_t minDimensionCount,
    uint32_t guaranteedBaseOffsetAlignment
    )
{
    m_tensorType = DML_TENSOR_TYPE_BUFFER;
    m_mlOperatorTensorDataType = dataType;

    m_bufferTensorDesc.DataType = GetDmlDataTypeFromMlDataType(dataType);
    ML_CHECK_VALID_ARGUMENT(ApiTraits::IsValidEnumValue(m_bufferTensorDesc.DataType));
    ML_CHECK_VALID_ARGUMENT(coerceAxis >= 0);

    gsl::span<const uint32_t> sizes;

    // If needed, flatten the tensor dimensions to a 2D tensor of size [a_0 * ... * a_{coerceAxis-1}, a_{coerceAxis} * ... * a_{n-1}]
    // e.g. Flattening [1,2,3,4] with axis 2 yields [2,12].
    uint32_t coercedSizes[2];
    if (dimensions.size() > 1 && coerceAxis < gsl::narrow_cast<int32_t>(dimensions.size()))
    {
        uint32_t dimension0 = 1u;
        uint32_t dimension1 = dimensions[coerceAxis];

        for (int32_t i = 0; i < coerceAxis; ++i)
        {
            dimension0 *= dimensions[i];
        }

        for (size_t i = static_cast<int64_t>(coerceAxis) + 1, ci = dimensions.size(); i < ci; ++i)
        {
            dimension1 *= dimensions[i];
        }

        coercedSizes[0] = dimension0;
        coercedSizes[1] = dimension1;

        sizes = coercedSizes;
    }
    else
    {
        sizes = dimensions;
    }

    ////////////////////////////////////////
    // Align dimensions

    // Determine the number of dimensions that should be aligned to the left edge when promoting to the minimum dimension count.
    // Negative values mean align from the right.
    const int32_t rank = gsl::narrow_cast<int32_t>(sizes.size());
    leftAlignedDimensionCount = leftAlignedDimensionCount < 0 ? std::max(0, leftAlignedDimensionCount + rank) : std::min(rank, leftAlignedDimensionCount);

    ML_CHECK_VALID_ARGUMENT(rank <= MaximumDimensionCount);
    m_bufferTensorDesc.DimensionCount = std::max(rank, int32_t(minDimensionCount));

    // Many DirectML operators accept only certain dimension counts, but it's very common for ONNX models
    // to have fewer dimensions than that. So this logic massages the dimension count up to what is needed
    // by filling unused dimensions with size 1, by left-aligning, right-aligning, or even mid-filling.
    //
    // e.g.:
    //
    //      elementwise addition - [H W]   -> [1 1 H W], leftAlignedCount = 0, placement = 0
    //      1D convolution       - [N C W] -> [N C 1 W], leftAlignedCount = 2, placement = 0
    //      batch normalization  - [C]     -> [1 C 1 1], leftAlignedCount = 1, placement = 1
    //
    {
        // Compute the total number of additional dimensions to fill with 1's,
        // before, after, and in the middle.
        const int32_t totalFillerCount     = m_bufferTensorDesc.DimensionCount - rank;
        const int32_t leadingFillerCount   = std::clamp(placement, 0, totalFillerCount);
        const int32_t remainingFillerCount = totalFillerCount - leadingFillerCount;
        const int32_t trailingFillerCount  = std::clamp(-placement, 0, remainingFillerCount);
        const int32_t middleFillerCount    = remainingFillerCount - trailingFillerCount;
        const int32_t firstRightAlignedDim = leadingFillerCount + leftAlignedDimensionCount + middleFillerCount;

        int i = 0, j = 0;
        while (j < leadingFillerCount)          { m_sizes[j++] = 1; }
        while (i < leftAlignedDimensionCount)   { m_sizes[j++] = sizes[i++]; }
        while (j < firstRightAlignedDim)        { m_sizes[j++] = 1; }
        while (i < rank)                        { m_sizes[j++] = sizes[i++]; }
        while (j < MaximumDimensionCount)       { m_sizes[j++] = 1; }
    }

    ////////////////////////////////////////
    // Coerce the physical shape to the desired shape.

    // By default, assume strides are not necessary.
    bool useStrides = false;

    if (dimensions != nonBroadcastDimensions)
    {
        // This broadcasting and subset logic is only applicable to the simple case where all
        // dimensions are contiguously right aligned, which means no flattening coercion,
        // placement offset, or split alignment. In such cases, the right side of m_sizes
        // should match the original dimensions.
        ML_CHECK_VALID_ARGUMENT(std::equal(
            dimensions.begin(),
            dimensions.end(),
            &m_sizes[m_bufferTensorDesc.DimensionCount - rank],
            &m_sizes[m_bufferTensorDesc.DimensionCount]
        ));

        // Stretch any dimensions with a single element.
        //
        // e.g. physical [2,1,4]
        //       desired [2,3,4]
        //       output  [2,3,4]
        //       strides [4,0,1]
        //
        // If broadcasting is used, then strides are used.
        useStrides = true;

        // Walk backwards through both input shapes and broadcast or default each dimension.
        auto nonBroadcastDimsIter = nonBroadcastDimensions.rbegin();
        uint32_t elementCount = 1;

        for (int descDimIndex = m_bufferTensorDesc.DimensionCount - 1; descDimIndex >= 0; --descDimIndex)
        {
            if (nonBroadcastDimsIter == nonBroadcastDimensions.rend() || (*nonBroadcastDimsIter == 1))
            {
                m_strides[descDimIndex] = 0;
            }
            else
            {
                m_strides[descDimIndex] = elementCount;
                elementCount *= (*nonBroadcastDimsIter);
            }

            if (nonBroadcastDimsIter != nonBroadcastDimensions.rend())
            {
                ++nonBroadcastDimsIter;
            }
        }
    }

    if (useStrides)
    {
        m_bufferTensorDesc.Strides = m_strides;
    }

    m_bufferTensorDesc.Flags = DML_TENSOR_FLAG_NONE;
    m_bufferTensorDesc.GuaranteedBaseOffsetAlignment = guaranteedBaseOffsetAlignment;
    m_bufferTensorDesc.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(
        m_bufferTensorDesc.DataType,
        m_bufferTensorDesc.DimensionCount,
        m_sizes,
        useStrides ? m_strides : nullptr
        );
    assert(m_bufferTensorDesc.TotalTensorSizeInBytes >= ComputeByteSizeFromDimensions(nonBroadcastDimensions, dataType));
}

gsl::span<const uint32_t> TensorDesc::GetStrides() const
{
    if (m_bufferTensorDesc.Strides == nullptr)
    {
        return {};
    }
    return { m_strides, m_strides + m_bufferTensorDesc.DimensionCount };
}

void TensorDesc::SetStrides(gsl::span<const uint32_t> strides)
{
    m_bufferTensorDesc.Strides = strides.empty() ? nullptr : strides.data();

    if (!strides.empty())
    {
        ML_CHECK_VALID_ARGUMENT(strides.size() <= std::size(m_strides));
        ML_CHECK_VALID_ARGUMENT(strides.size() == m_bufferTensorDesc.DimensionCount);
        std::copy(strides.begin(), strides.end(), m_strides);
    }

    m_bufferTensorDesc.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(
        m_bufferTensorDesc.DataType,
        m_bufferTensorDesc.DimensionCount,
        m_sizes,
        strides.empty() ? nullptr : m_strides);
}

DML_TENSOR_DESC TensorDesc::GetDmlDesc()
{
    if (m_tensorType == DML_TENSOR_TYPE_INVALID)
    {
        return { m_tensorType, nullptr };
    }

    // Update the DML_BUFFER_TENSOR_DESC Sizes and Strides pointers to point internally to the TensorDesc fields.
    // This update matters whether it was a new instance or a copy from via copy constructor from another TensorDesc.
    m_bufferTensorDesc.Sizes = m_sizes;
    if (m_bufferTensorDesc.Strides)
    {
        m_bufferTensorDesc.Strides = m_strides;
    }

    // Only buffer tensors are supported right now.
    assert(m_tensorType == DML_TENSOR_TYPE_BUFFER);
    return { m_tensorType, &m_bufferTensorDesc };
}

// ONNX likes to use signed types for logically unsigned index data.  DML avoids this inconsistency and therefore
// requires coercion by the caller.
void TensorDesc::ForceUnsignedDataType()
{
    static_assert(ApiTraits::EnumValueCount<DML_TENSOR_DATA_TYPE> == 12, "New tensor data type.  Update cases.");

    switch (m_bufferTensorDesc.DataType)
    {
    case DML_TENSOR_DATA_TYPE_INT64:
        m_bufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_UINT64;
        break;

    case DML_TENSOR_DATA_TYPE_INT32:
        m_bufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_UINT32;
        break;

    case DML_TENSOR_DATA_TYPE_INT16:
        m_bufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_UINT16;
        break;

    case DML_TENSOR_DATA_TYPE_INT8:
        m_bufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_UINT8;
        break;

    // Nothing to do if already unsigned.
    case DML_TENSOR_DATA_TYPE_UINT64:
    case DML_TENSOR_DATA_TYPE_UINT32:
    case DML_TENSOR_DATA_TYPE_UINT16:
    case DML_TENSOR_DATA_TYPE_UINT8:
        break;

    default:
        ML_INVALID_ARGUMENT("Can't coerce unknown or non-integral data type");
    }
}

void TensorDesc::SetDimensionCount(uint32_t newDimensionCount, TensorAxis alignment)
{
    ML_CHECK_VALID_ARGUMENT(newDimensionCount <= MaximumDimensionCount);
    ML_CHECK_VALID_ARGUMENT(alignment == TensorAxis::RightAligned || alignment == TensorAxis::LeftAligned);

    const uint32_t oldDimensionCount = m_bufferTensorDesc.DimensionCount;
    const int32_t difference = static_cast<int32_t>(newDimensionCount - oldDimensionCount);
    if (difference == 0)
    {
        return;
    }

    int32_t fillOffset = oldDimensionCount;
    int32_t fillCount = std::max(0, difference);

    // alignment == TensorAxis::LeftAligned is the easy case.
    // Right alignment needs more work, shifting values over.
    if (alignment == TensorAxis::RightAligned)
    {
        fillOffset = 0; // Fill leading dimensions with 1's starting at the front.
        uint32_t moveCount = std::min(newDimensionCount, oldDimensionCount);
        memmove(&m_sizes[fillCount], &m_sizes[oldDimensionCount - moveCount], sizeof(m_sizes[0]) * moveCount);
        memmove(&m_strides[fillCount], &m_strides[oldDimensionCount - moveCount], sizeof(m_strides[0]) * moveCount);
    }
    if (fillCount > 0)
    {
        std::fill(&m_sizes[fillOffset], &m_sizes[fillOffset] + fillCount, 1u);
        std::fill(&m_strides[fillOffset], &m_strides[fillOffset] + fillCount, 0u);
    }
    m_bufferTensorDesc.DimensionCount = newDimensionCount;
}

// Uses dimensionMapping to reorder m_sizes and m_strides to match specific Tensor layout
void TensorDesc::PermuteDimensions(gsl::span<const uint32_t> dimensionMapping, const TensorAxis alignment)
{
    EnsureStridesExist();
    SetDimensionCount(static_cast<uint32_t>(dimensionMapping.size()), alignment);

    // Shuffle m_sizes and m_strides according to the indexes pointed by dimensionMapping
    std::vector<uint32_t> tempSizes{m_sizes, m_sizes + MaximumDimensionCount};
    std::vector<uint32_t> tempStrides{m_strides, m_strides + MaximumDimensionCount};

    for (size_t i = 0; i < dimensionMapping.size(); i++)
    {
        m_sizes[i] = tempSizes[dimensionMapping[i]];
        m_strides[i] = tempStrides[dimensionMapping[i]];
    }

    m_bufferTensorDesc.Sizes = m_sizes;
    m_bufferTensorDesc.Strides = m_strides;
}

void TensorDesc::EnsureStridesExist()
{
    if (m_bufferTensorDesc.Strides != nullptr)
    {
        // Strides are populated
        return;
    }

    uint32_t stride = 1;
    for (uint32_t i = m_bufferTensorDesc.DimensionCount; i-- > 0;)
    {
        m_strides[i] = stride;
        stride *= m_sizes[i];
    }
}
