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
    gsl::span<const uint32_t> dimensions,
    gsl::span<const uint32_t> nonBroadcastDimensions,
    uint32_t coerceAxis,
    uint32_t placement,
    uint32_t minDimensionCount,
    uint32_t guaranteedBaseOffsetAlignment
    )
{
    m_tensorType = DML_TENSOR_TYPE_BUFFER;
    m_mlOperatorTensorDataType = dataType;

    m_bufferTensorDesc.DataType = GetDmlDataTypeFromMlDataType(dataType);
    ML_CHECK_VALID_ARGUMENT(ApiTraits::IsValidEnumValue(m_bufferTensorDesc.DataType));

    // Coercion isn't always possible when striding is used to broadcast tensors, and must be different
    // among dimensions flattened together. Placement is not implemented in combination with broadcasting, 
    // though could be.
    ML_CHECK_VALID_ARGUMENT((nonBroadcastDimensions == dimensions) || ((coerceAxis == TensorAxis::DoNotCoerce) && (placement == W)));

    gsl::span<const uint32_t> sizes;

    // If needed, perform tensor coercion to 2D tensor of size [a_0 * ... * a_{coerceAxis-1}, a_{coerceAxis} * ... * a_{n-1}]
    uint32_t coercedSizes[2];
    if (dimensions.size() > 1 && coerceAxis < gsl::narrow_cast<uint32_t>(dimensions.size()))
    {
        uint32_t dimension0 = 1u;
        uint32_t dimension1 = dimensions[coerceAxis];

        for (uint32_t i = 0; i < coerceAxis; ++i)
        {
            dimension0 *= dimensions[i];
        }

        for (size_t i = coerceAxis + 1, ci = dimensions.size(); i < ci; ++i)
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

    ML_CHECK_VALID_ARGUMENT(sizes.size() <= MaximumDimensionCount);
    m_bufferTensorDesc.DimensionCount = std::max(gsl::narrow_cast<uint32_t>(sizes.size()), minDimensionCount);

    int leadingDims = std::max<int>(0, placement + 1 - gsl::narrow_cast<int>(sizes.size()));

    for (int i = 0; i < leadingDims; ++i)
    {
        m_sizes[i] = 1;
    }

    for (int i = 0, ci = gsl::narrow_cast<int>(sizes.size()); i < ci; ++i)
    {
        m_sizes[leadingDims + i] = gsl::narrow_cast<uint32_t>(sizes[i]);
    }

    for (int i = leadingDims + gsl::narrow_cast<int>(sizes.size()); i < MaximumDimensionCount; ++i)
    {
        m_sizes[i] = 1;
    }

    // By default, assume strides are not necessary.
    bool useStrides = false;

    if (dimensions != nonBroadcastDimensions)
    {
        // If broadcasting is used, then strides are used.
        useStrides = true;

        // Walk backwards through both input shapes and broadcast or default each dimension
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

    uint64_t endPaddingInBytes = 0;

    if (dataType == MLOperatorTensorDataType::UInt64 || dataType == MLOperatorTensorDataType::Int64)
    {
        // DirectML doesn't support tensor of int64 because Direct3D doesn't support 
        // the data type. A workaround is to use strides to fake 64-bit memory access
        // while only the lower 32 bits contains the data. This trick obviously doesn't
        // work if the data element is genuine 64-bit. It also doesn't work if the data
        // element is negative as the signed bit will be incorrectly interpreted.
        m_bufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_UINT32;

        // If the strides haven't been calculated yet, initialize them as packed.
        if (!useStrides)
        {
            uint32_t stride = 1;
            for (int i = m_bufferTensorDesc.DimensionCount - 1; i >= 0; i--)
            {
                m_strides[i] = stride;
                stride *= m_sizes[i];
            }
        }

        // Double the stride values to emulate 64-bit integer support.
        for (uint32_t i = 0; i < m_bufferTensorDesc.DimensionCount; ++i)
        {
            m_strides[i] *= 2;
        }

        useStrides = true;

        // The physical size of the tensor will have an extra 4 bytes at the end.
        // DMLCalcBufferTensorSize calculates the minimum implied size, which is based on the last
        // addressable element of the tensor plus the space for the last element. However, the size
        // of the last element is now halved from 8 bytes to 4 bytes.
        //
        // Example:
        // Original Tensor: size={2,3}, strides={3,1}, type=int64, size = (1+{1,2}*{3,1})*sizeof(int64) = 6 * 8 = 48
        // Emulated Tensor: size={2,3}, strides={6,2}, type=int32, size = (1+{1,2}*{6,2})*sizeof(int32) = 11 * 4 = 44
        //
        // DirectML itself won't read/write the last 4 bytes, but we want the total size to be accurate
        // so that the entire region can be zeroed.
        endPaddingInBytes = sizeof(uint32_t);
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
        ) + endPaddingInBytes;
}

gsl::span<const uint32_t> TensorDesc::GetStrides() const
{
    if (m_bufferTensorDesc.Strides == nullptr)
    {
        return {};
    }
    return { m_strides, m_strides + m_bufferTensorDesc.DimensionCount }; 
}

DML_TENSOR_DESC TensorDesc::GetDmlDesc()
{
    if (m_tensorType == DML_TENSOR_TYPE_INVALID)
    {
        return { m_tensorType, nullptr };
    }

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
    static_assert(ApiTraits::EnumValueCount<DML_TENSOR_DATA_TYPE> == 9, "New tensor data type.  Update cases.");

    switch (m_bufferTensorDesc.DataType)
    {
    case DML_TENSOR_DATA_TYPE_INT32:
        m_bufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_UINT32;
        break;

    case DML_TENSOR_DATA_TYPE_INT16:
        m_bufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_UINT16;
        break;

    case DML_TENSOR_DATA_TYPE_INT8:
        m_bufferTensorDesc.DataType = DML_TENSOR_DATA_TYPE_UINT8;
        break;

        // Nothing to do if already unsigned
    case DML_TENSOR_DATA_TYPE_UINT32:
    case DML_TENSOR_DATA_TYPE_UINT16:
    case DML_TENSOR_DATA_TYPE_UINT8:
        break;

    default:
        ML_INVALID_ARGUMENT("Can't coerce unknown or non-integral data type");
    }
}
