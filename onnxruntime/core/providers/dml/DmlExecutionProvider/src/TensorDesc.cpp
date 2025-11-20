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

        SetBroadcastedShape(dimensions, nonBroadcastDimensions, m_bufferTensorDesc.DimensionCount);
    }

    m_bufferTensorDesc.Flags = DML_TENSOR_FLAG_NONE;
    m_bufferTensorDesc.GuaranteedBaseOffsetAlignment = guaranteedBaseOffsetAlignment;
    m_bufferTensorDesc.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(
        m_bufferTensorDesc.DataType,
        m_bufferTensorDesc.DimensionCount,
        m_sizes,
        m_bufferTensorDesc.Strides
        );
    assert(m_bufferTensorDesc.TotalTensorSizeInBytes >= ComputeByteSizeFromDimensions(nonBroadcastDimensions, dataType));
}

gsl::span<const uint32_t> TensorDesc::GetStrides() const noexcept
{
    if (m_bufferTensorDesc.Strides == nullptr)
    {
        return {};
    }
    return { m_strides, m_strides + m_bufferTensorDesc.DimensionCount };
}

void TensorDesc::SetStrides(gsl::span<const uint32_t> strides)
{
    if (!strides.empty())
    {
        ML_CHECK_VALID_ARGUMENT(strides.size() <= std::size(m_strides));
        ML_CHECK_VALID_ARGUMENT(strides.size() == m_bufferTensorDesc.DimensionCount);
        std::copy(strides.begin(), strides.end(), m_strides);
    }

    m_bufferTensorDesc.Strides = strides.empty() ? nullptr : m_strides;

    m_bufferTensorDesc.TotalTensorSizeInBytes = DMLCalcBufferTensorSize(
        m_bufferTensorDesc.DataType,
        m_bufferTensorDesc.DimensionCount,
        m_sizes,
        strides.empty() ? nullptr : m_strides);
}

DML_TENSOR_DESC TensorDesc::GetDmlDesc() noexcept
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
    static_assert(ApiTraits::EnumValueCount<DML_TENSOR_DATA_TYPE> == 14, "New tensor data type.  Update cases.");

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

    case DML_TENSOR_DATA_TYPE_INT4:
        m_bufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_UINT4;
        break;

    // Nothing to do if already unsigned.
    case DML_TENSOR_DATA_TYPE_UINT64:
    case DML_TENSOR_DATA_TYPE_UINT32:
    case DML_TENSOR_DATA_TYPE_UINT16:
    case DML_TENSOR_DATA_TYPE_UINT8:
    case DML_TENSOR_DATA_TYPE_UINT4:
        break;

    default:
        ML_INVALID_ARGUMENT("Can't coerce unknown or non-integral data type");
    }
}

void TensorDesc::BroadcastTo(gsl::span<const uint32_t> targetSizes)
{
    auto currentSizes = gsl::span<const uint32_t>(m_sizes, m_sizes + m_bufferTensorDesc.DimensionCount);
    size_t targetRank = std::max(currentSizes.size(), targetSizes.size());
    return SetBroadcastedShape(targetSizes, currentSizes, targetRank);
}

// The targetRank can be greater than or equal the targetSizes, which may apply if there is a minimum rank
// for an operator. targetSizes and sourceSizes may be different sizes, and they will be right-aligned within
// the targetRank.
void TensorDesc::SetBroadcastedShape(gsl::span<const uint32_t> targetSizes, gsl::span<const uint32_t> sourceSizes, size_t targetRank)
{
    ML_CHECK_VALID_ARGUMENT(targetSizes.size() <= targetRank);
    ML_CHECK_VALID_ARGUMENT(targetRank <= MaximumDimensionCount);

    // Update the tensor shape with the target shape, padding right-aligned if needed.
    size_t extraFill = targetRank - targetSizes.size();
    std::fill(&m_sizes[0], &m_sizes[0] + extraFill, 1u); // Insert any leading 1's.
    std::copy(targetSizes.data(), targetSizes.data() + targetSizes.size(), &m_sizes[0] + extraFill);
    m_bufferTensorDesc.DimensionCount = uint32_t(targetRank);

    if (targetSizes == sourceSizes)
    {
        m_bufferTensorDesc.Strides = nullptr; // Packed strides.
        return;
    }

    // Stretch any dimensions with a single element.
    //
    // e.g. physical [2,1,4]
    //       desired [2,3,4]
    //       output  [2,3,4]
    //       strides [4,0,1]

    // Walk backwards through each dimenions of the input shapes, either broadcasting or defaulting to packed.
    auto sourceSizesIter = sourceSizes.rbegin();
    uint32_t elementCount = 1;

    for (size_t dimensionIndex = targetRank; dimensionIndex-- > 0; )
    {
        uint32_t stride = 0; // Broadcast by default (any leading batch dimensions or dimensions of size 1)
        if (sourceSizesIter != sourceSizes.rend())
        {
            if (uint32_t size = *sourceSizesIter; size > 1)
            {
                stride = elementCount;
                elementCount *= size;
            }
            ++sourceSizesIter;
        }
        m_strides[dimensionIndex] = stride;
    }

    m_bufferTensorDesc.Strides = m_strides;
}

// Add additional padding 1's to ensure the count is at least that large.
void TensorDesc::EnsureMinimumDimensionCount(uint32_t minimumDimensionCount, TensorAxis alignment)
{
    if (m_bufferTensorDesc.DimensionCount < minimumDimensionCount)
    {
        SetDimensionCount(minimumDimensionCount, alignment);
    }
}

// Ensure the dimension count is less than or equal to the limit.
void TensorDesc::EnsureMaximumDimensionCount(uint32_t maximumDimensionCount, TensorAxis alignment)
{
    if (m_bufferTensorDesc.DimensionCount > maximumDimensionCount)
    {
        SetDimensionCount(maximumDimensionCount, alignment, /*foldEndDimensions*/ true);
    }
}

// Set a new dimension count, adding or removing dimensions as needed.
// If the new rank is larger, any new dimensions are filled with size 1 and stride 0.
// If the new rank is smaller and foldEndDimensions is true, then any removed dimensions are folded together.
// Otherwise those dimensions (leading or trailing, depending on alignment) are simply truncated.
void TensorDesc::SetDimensionCount(uint32_t newDimensionCount, TensorAxis alignment, bool foldEndDimensions)
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

    // If shrinking the rank and asked to fold dimensions, then collapse them into the first/last dimension.
    // e.g. Folding 4D dimensions [2,3,4,5] to 3D right-aligned yield [6,4,5]
    // e.g.         6D dimensions [2,3,4,5,6,7] to 3D left-aligned yield [1,2,840]
    //
    // Otherwise dimensions are simply truncated (which may be desired if they were modified before calling).
    if (foldEndDimensions && difference < 0 && newDimensionCount > 0)
    {
        uint32_t dimensionCountRemoved = -difference;
        uint32_t dimensionCountFolded = dimensionCountRemoved + 1; // If 2 dimensions are removed, then 3 dimensions are folded into one.
        uint32_t targetDimensionIndex;
        uint32_t firstFoldedDimensionIndex;

        // Determine the range to fold and which dimension to fold them into.
        // e.g. Right-aligned: was 4D [2, 3, 4, 5]
        //                     now 2D      [12, 5]
        //                     fold    <----->
        //                     target        *
        //
        //      Left-aligned:  was 4D [2, 3, 4, 5]
        //                     now 2D [2, 60]
        //                     fold       <----->
        //                     target     *
        //
        if (alignment == TensorAxis::RightAligned)
        {
            targetDimensionIndex = dimensionCountRemoved; // Fold extra dimensions into the first dimension of the new size.
            firstFoldedDimensionIndex = 0;
        }
        else // alignment == TensorAxis::LeftAligned
        {
            targetDimensionIndex = newDimensionCount - 1; // Fold extra dimensions into the last dimension of the new size.
            firstFoldedDimensionIndex = targetDimensionIndex;
        }
        auto sizeFoldBegin = &m_sizes[firstFoldedDimensionIndex];
        auto sizeFoldEnd = &m_sizes[firstFoldedDimensionIndex + dimensionCountFolded];

        // Ensure no stride broadcasting is lost during the fold, which would silently give incorrect results.
        ML_CHECK_VALID_ARGUMENT(
            m_bufferTensorDesc.Strides == nullptr ||
            AreStridesCollapsible(
                { sizeFoldBegin, sizeFoldEnd },
                { &m_strides[firstFoldedDimensionIndex], dimensionCountFolded }
            )
        );

        m_sizes[targetDimensionIndex] = std::accumulate(sizeFoldBegin, sizeFoldEnd, 1u, std::multiplies<uint32_t>());

        // Update strides too (right alignment has no extra work).
        if (alignment == TensorAxis::LeftAligned)
        {
            m_strides[targetDimensionIndex] = m_strides[oldDimensionCount - 1]; // Migrate the last stride to its new position.
        }
        // Ensure the target stride is at least 1, not 0, in case a dimension of size 1 was folded that had a stride
        // of 0 (which might happen because a stride of 0 for dimension of size 1 is ignorable), and other dimensions
        // were folded into the target too.
        m_strides[targetDimensionIndex] = std::max(m_strides[targetDimensionIndex], 1u);
    }

    // Left alignment is the easy case (just truncate the end).
    // Right alignment needs more work, shifting values over.
    if (alignment == TensorAxis::RightAligned)
    {
        fillOffset = 0; // Fill leading dimensions with 1's starting at the front.
        uint32_t moveCount = std::min(newDimensionCount, oldDimensionCount);
        memmove(&m_sizes[fillCount], &m_sizes[oldDimensionCount - moveCount], sizeof(m_sizes[0]) * moveCount);
        memmove(&m_strides[fillCount], &m_strides[oldDimensionCount - moveCount], sizeof(m_strides[0]) * moveCount);
    }

    // For any new dimensions, insert leading/trailing 1's for sizes and 0's for strides.
    if (fillCount > 0)
    {
        std::fill(&m_sizes[fillOffset], &m_sizes[fillOffset] + fillCount, 1u);
        std::fill(&m_strides[fillOffset], &m_strides[fillOffset] + fillCount, 0u);
    }
    m_bufferTensorDesc.DimensionCount = newDimensionCount;
}

void TensorDesc::SetDimensionsAndStrides(gsl::span<const uint32_t> sizes, gsl::span<const uint32_t> strides)
{
    static_assert(sizeof(m_sizes) == sizeof(m_strides));
    ML_CHECK_VALID_ARGUMENT(sizes.size() <= std::size(m_sizes));
    ML_CHECK_VALID_ARGUMENT(strides.empty() || strides.size() == sizes.size());

    std::copy(sizes.begin(), sizes.end(), m_sizes);
    m_bufferTensorDesc.DimensionCount = static_cast<uint32_t>(sizes.size());
    SetStrides(strides);
}

void TensorDesc::PermuteDimensions(gsl::span<const uint32_t> dimensionMapping, const TensorAxis alignment)
{
    const uint32_t oldRank = m_bufferTensorDesc.DimensionCount;
    EnsureStridesExist();
    SetDimensionCount(static_cast<uint32_t>(dimensionMapping.size()), alignment);

    // Shuffle m_sizes and m_strides according to the indexes pointed by dimensionMapping.
    // Note using MaximumDimensionCount instead of oldRank is intentional here, because the old rank could
    // be smaller or larger than the new rank, but it will never be larger than MaximumDimensionCount.
    std::vector<uint32_t> oldSizes{m_sizes, m_sizes + MaximumDimensionCount};
    std::vector<uint32_t> oldStrides{m_strides, m_strides + MaximumDimensionCount};

    for (size_t i = 0; i < dimensionMapping.size(); i++)
    {
        uint32_t sourceAxis = dimensionMapping[i];
        m_sizes[i] = sourceAxis < oldRank ? oldSizes[sourceAxis] : 1;
        m_strides[i] = sourceAxis < oldRank ? oldStrides[sourceAxis] : 0;
    }

    m_bufferTensorDesc.Sizes = m_sizes;
    m_bufferTensorDesc.Strides = m_strides;
}

void TensorDesc::EnsureStridesExist() noexcept
{
    if (m_bufferTensorDesc.Strides != nullptr)
    {
        // Strides are already populated
        return;
    }

    GetDescendingPackedStrides({m_sizes, m_bufferTensorDesc.DimensionCount}, {m_strides, m_bufferTensorDesc.DimensionCount});
    m_bufferTensorDesc.Strides = m_strides;
}

// A range of dimensions have collapsible strides if they all follow packed layout rules
// or are all broadcasted dimensions. e.g.
//
//  sizes [2,3,4] with strides [12,4,1] are collapsible
//  sizes [1,1,1] with strides [1,0,1] are collapsible
//  sizes [2,1,4] with strides [4,0,1] are not collapsible
bool TensorDesc::AreStridesCollapsible(gsl::span<const uint32_t> sizes, gsl::span<const uint32_t> strides) const noexcept
{
    if (strides.empty())
    {
        return true;
    }

    assert(sizes.size() == strides.size());

    // Start from the back and work towards the front, computing the accumulated packed stride
    // and comparing it to the actual stride. Skip dimensions of size 1 which make no difference,
    // as these are the logical target dimensions by now, not the original physical dimensions
    // (so it's stride can be ignored).
    uint32_t expectedStride = strides.back();
    for (size_t i = strides.size(); i-- > 0; )
    {
        if (sizes[i] != 1)
        {
            if (strides[i] != expectedStride)
            {
                return false;
            }
            expectedStride *= sizes[i];
        }
    }
    return true;
}
