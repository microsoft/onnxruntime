// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

DML_TENSOR_DATA_TYPE GetDmlDataTypeFromMlDataTypeNoThrow(MLOperatorTensorDataType tensorDataType) noexcept
{
    switch (tensorDataType)
    {
    case MLOperatorTensorDataType::Float: return DML_TENSOR_DATA_TYPE_FLOAT32;
    case MLOperatorTensorDataType::UInt8: return DML_TENSOR_DATA_TYPE_UINT8;
    case MLOperatorTensorDataType::Int8: return DML_TENSOR_DATA_TYPE_INT8;
    case MLOperatorTensorDataType::UInt16: return DML_TENSOR_DATA_TYPE_UINT16;
    case MLOperatorTensorDataType::Int16: return DML_TENSOR_DATA_TYPE_INT16;
    case MLOperatorTensorDataType::Int32: return DML_TENSOR_DATA_TYPE_INT32;
    case MLOperatorTensorDataType::Int64: return DML_TENSOR_DATA_TYPE_INT64;
    case MLOperatorTensorDataType::String: return DML_TENSOR_DATA_TYPE_UNKNOWN;
    case MLOperatorTensorDataType::Bool: return DML_TENSOR_DATA_TYPE_UINT8;
    case MLOperatorTensorDataType::Float16: return DML_TENSOR_DATA_TYPE_FLOAT16;
    case MLOperatorTensorDataType::Double: return DML_TENSOR_DATA_TYPE_FLOAT64;
    case MLOperatorTensorDataType::UInt32: return DML_TENSOR_DATA_TYPE_UINT32;
    case MLOperatorTensorDataType::UInt64: return DML_TENSOR_DATA_TYPE_UINT64;
    case MLOperatorTensorDataType::Complex64: return DML_TENSOR_DATA_TYPE_UNKNOWN;
    case MLOperatorTensorDataType::Complex128: return DML_TENSOR_DATA_TYPE_UNKNOWN;
    case MLOperatorTensorDataType::Undefined:
    default: return DML_TENSOR_DATA_TYPE_UNKNOWN;
    };
}

DML_TENSOR_DATA_TYPE Remap64bitDmlDataTypeTo32bit(DML_TENSOR_DATA_TYPE dmlElementType) noexcept
{
    switch (dmlElementType)
    {
    case DML_TENSOR_DATA_TYPE_UINT64: return DML_TENSOR_DATA_TYPE_UINT32; break;
    case DML_TENSOR_DATA_TYPE_INT64:  return DML_TENSOR_DATA_TYPE_INT32;  break;
    default: return dmlElementType;
    }
}

bool IsSigned(DML_TENSOR_DATA_TYPE dataType)
{
    switch (dataType)
    {
        case DML_TENSOR_DATA_TYPE_FLOAT64: return true;
        case DML_TENSOR_DATA_TYPE_FLOAT32: return true;
        case DML_TENSOR_DATA_TYPE_FLOAT16: return true;
        case DML_TENSOR_DATA_TYPE_UINT64: return false;
        case DML_TENSOR_DATA_TYPE_UINT32: return false;
        case DML_TENSOR_DATA_TYPE_UINT16: return false;
        case DML_TENSOR_DATA_TYPE_UINT8: return false;
        case DML_TENSOR_DATA_TYPE_INT64: return true;
        case DML_TENSOR_DATA_TYPE_INT32: return true;
        case DML_TENSOR_DATA_TYPE_INT16: return true;
        case DML_TENSOR_DATA_TYPE_INT8: return true;
    }

    assert(false);
    return false;
}

DML_TENSOR_DATA_TYPE GetDmlDataTypeFromMlDataType(MLOperatorTensorDataType tensorDataType)
{
    DML_TENSOR_DATA_TYPE dmlTensorDataType = GetDmlDataTypeFromMlDataTypeNoThrow(tensorDataType);
    if (dmlTensorDataType == DML_TENSOR_DATA_TYPE_UNKNOWN)
    {
        ML_INVALID_ARGUMENT("MLOperatorTensorDataType has no equivalent data type in DML.");
    }
    return dmlTensorDataType;
}

MLOperatorTensorDataType GetMlDataTypeFromDmlDataType(DML_TENSOR_DATA_TYPE tensorDataType)
{
    switch (tensorDataType)
    {
    case DML_TENSOR_DATA_TYPE_FLOAT32:  return MLOperatorTensorDataType::Float;
    case DML_TENSOR_DATA_TYPE_UINT8:    return MLOperatorTensorDataType::UInt8;
    case DML_TENSOR_DATA_TYPE_INT8:     return MLOperatorTensorDataType::Int8;
    case DML_TENSOR_DATA_TYPE_UINT16:   return MLOperatorTensorDataType::UInt16;
    case DML_TENSOR_DATA_TYPE_INT16:    return MLOperatorTensorDataType::Int16;
    case DML_TENSOR_DATA_TYPE_INT32:    return MLOperatorTensorDataType::Int32;
    case DML_TENSOR_DATA_TYPE_FLOAT16:  return MLOperatorTensorDataType::Float16;
    case DML_TENSOR_DATA_TYPE_UINT32:   return MLOperatorTensorDataType::UInt32;
    case DML_TENSOR_DATA_TYPE_UINT64:   return MLOperatorTensorDataType::UInt64;
    case DML_TENSOR_DATA_TYPE_INT64:    return MLOperatorTensorDataType::Int64;
    case DML_TENSOR_DATA_TYPE_FLOAT64:  return MLOperatorTensorDataType::Double;

    default: ML_INVALID_ARGUMENT("Unknown DML_TENSOR_DATA_TYPE.");
    };
}

size_t ComputeByteSizeFromDimensions(gsl::span<const DimensionType> dimensions, MLOperatorTensorDataType tensorDataType)
{
    return ComputeElementCountFromDimensions(dimensions) * GetByteSizeFromMlDataType(tensorDataType);
}

size_t ComputeByteSizeFromTensor(IMLOperatorTensor& tensor)
{
    uint32_t dimensionCount = 0;
    dimensionCount = tensor.GetDimensionCount();
    ML_CHECK_VALID_ARGUMENT(dimensionCount <= MaximumDimensionCount, "Dimensions are beyond supported count.");

    std::array<DimensionType, MaximumDimensionCount> dimensions;
    THROW_IF_FAILED(tensor.GetShape(dimensionCount, /*out*/ dimensions.data()));

    return ComputeByteSizeFromDimensions(gsl::make_span(dimensions.data(), dimensionCount), tensor.GetTensorDataType());
}

uint32_t GetSupportedDeviceDataTypeMask(IDMLDevice* dmlDevice)
{
    uint32_t deviceTypeMask = 0u;

    // Form the bitmask of all supported data types.
    for (uint32_t i = 0; i <= DML_TENSOR_DATA_TYPE_INT64; ++i)
    {
        DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT dataTypeQuery = { static_cast<DML_TENSOR_DATA_TYPE>(i) };
        DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT dataTypeSupport = {};

        THROW_IF_FAILED(dmlDevice->CheckFeatureSupport(
            DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT,
            sizeof(dataTypeQuery),
            &dataTypeQuery,
            sizeof(dataTypeSupport),
            &dataTypeSupport
        ));

        deviceTypeMask |= (dataTypeSupport.IsSupported << i);
    }

    return deviceTypeMask;
}

void GetDescendingPackedStrides(gsl::span<const uint32_t> sizes, /*out*/ gsl::span<uint32_t> strides)
{
    assert(sizes.size() == strides.size());

    uint32_t stride = 1;
    for (size_t i = strides.size(); i-- > 0; )
    {
        strides[i] = stride;
        stride *= sizes[i];
    }
}

} // namespace Dml
