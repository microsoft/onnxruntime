// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <assert.h>
#include "core/providers/dml/OperatorAuthorHelper/Common.h"

namespace Dml
{
    using namespace OperatorHelper;

    static const int MaximumDimensionCount = DML_TENSOR_DIMENSION_COUNT_MAX1;

    DML_TENSOR_DATA_TYPE GetDmlDataTypeFromMlDataType(MLOperatorTensorDataType tensorDataType);
    DML_TENSOR_DATA_TYPE GetDmlDataTypeFromMlDataTypeNoThrow(MLOperatorTensorDataType tensorDataType) noexcept;
    DML_TENSOR_DATA_TYPE Remap64bitDmlDataTypeTo32bit(DML_TENSOR_DATA_TYPE dmlElementType) noexcept;
    MLOperatorTensorDataType GetMlDataTypeFromDmlDataType(DML_TENSOR_DATA_TYPE tensorDataType);
    size_t ComputeByteSizeFromDimensions(gsl::span<const DimensionType> dimensions, MLOperatorTensorDataType tensorDataType);
    size_t ComputeByteSizeFromTensor(IMLOperatorTensor& tensor);
    uint32_t GetSupportedDeviceDataTypeMask(IDMLDevice* dmlDevice);
    void GetDescendingPackedStrides(gsl::span<const uint32_t> sizes, /*out*/ gsl::span<uint32_t> strides);

    bool IsSigned(DML_TENSOR_DATA_TYPE dataType);

    /** Calculates the minimum number of bytes required to store a buffer tensor with the specified type, sizes, and
        strides. The formula can be expressed as the following:

        IndexOfLastElement = dot(Sizes - 1, Strides);
        MinimumImpliedSizeInBytes = roundup((IndexOfLastElement + 1) * ElementSizeInBytes, 4)

        In other words, the minimum size of a tensor is the index of the one-past-the-end element, multiplied by the
        element size (e.g. 2 bytes for a FLOAT16 tensor). Additionally DirectML requires that all buffers bound must have
        a total size which is DWORD-aligned, and hence the minimum implied size in bytes must be rounded up to the nearest
        4-byte boundary.
    */
    inline UINT64 DMLCalcBufferTensorSize(
        DML_TENSOR_DATA_TYPE dataType,
        UINT dimensionCount,
        _In_reads_(dimensionCount) const UINT* sizes,
        _In_reads_opt_(dimensionCount) const UINT* strides)
    {
        UINT elementSizeInBytes = 0;
        switch (dataType)
        {
        case DML_TENSOR_DATA_TYPE_FLOAT64:
        case DML_TENSOR_DATA_TYPE_UINT64:
        case DML_TENSOR_DATA_TYPE_INT64:
            elementSizeInBytes = 8;
            break;

        case DML_TENSOR_DATA_TYPE_FLOAT32:
        case DML_TENSOR_DATA_TYPE_UINT32:
        case DML_TENSOR_DATA_TYPE_INT32:
            elementSizeInBytes = 4;
            break;

        case DML_TENSOR_DATA_TYPE_FLOAT16:
        case DML_TENSOR_DATA_TYPE_UINT16:
        case DML_TENSOR_DATA_TYPE_INT16:
            elementSizeInBytes = 2;
            break;

        case DML_TENSOR_DATA_TYPE_UINT8:
        case DML_TENSOR_DATA_TYPE_INT8:
            elementSizeInBytes = 1;
            break;

        default:
            return 0; // Invalid data type
        }

        UINT64 minimumImpliedSizeInBytes = 0;
        if (!strides)
        {
            minimumImpliedSizeInBytes = sizes[0];
            for (UINT i = 1; i < dimensionCount; ++i)
            {
                minimumImpliedSizeInBytes *= sizes[i];
            }
            minimumImpliedSizeInBytes *= elementSizeInBytes;
        }
        else
        {
            UINT indexOfLastElement = 0;
            for (UINT i = 0; i < dimensionCount; ++i)
            {
                indexOfLastElement += (sizes[i] - 1) * strides[i];
            }

            minimumImpliedSizeInBytes = (indexOfLastElement + 1) * elementSizeInBytes;
        }

        // Round up to the nearest 4 bytes.
        minimumImpliedSizeInBytes = (minimumImpliedSizeInBytes + 3) & ~3ui64;

        return minimumImpliedSizeInBytes;
    }

    template <typename T>
    void CastToClampedScalarUnion(DML_TENSOR_DATA_TYPE dataType, T value, DML_SCALAR_UNION* outputValue)
    {
        switch (dataType)
        {
        case DML_TENSOR_DATA_TYPE_UINT8:    outputValue->UInt8   = clamp_cast<uint8_t, T>(value); break;
        case DML_TENSOR_DATA_TYPE_UINT16:   outputValue->UInt16  = clamp_cast<uint16_t, T>(value); break;
        case DML_TENSOR_DATA_TYPE_UINT32:   outputValue->UInt32  = clamp_cast<uint32_t, T>(value); break;
        case DML_TENSOR_DATA_TYPE_UINT64:   outputValue->UInt64  = clamp_cast<uint64_t, T>(value); break;
        case DML_TENSOR_DATA_TYPE_INT8:     outputValue->Int8    = clamp_cast<int8_t, T>(value); break;
        case DML_TENSOR_DATA_TYPE_INT16:    outputValue->Int16   = clamp_cast<int16_t, T>(value); break;
        case DML_TENSOR_DATA_TYPE_INT32:    outputValue->Int32   = clamp_cast<int32_t, T>(value); break;
        case DML_TENSOR_DATA_TYPE_INT64:    outputValue->Int64   = clamp_cast<int64_t, T>(value); break;
        case DML_TENSOR_DATA_TYPE_FLOAT16:  outputValue->Float32 = clamp_cast<float, T>(value); break;
        case DML_TENSOR_DATA_TYPE_FLOAT32:  outputValue->Float32 = clamp_cast<float, T>(value); break;
        case DML_TENSOR_DATA_TYPE_FLOAT64:  outputValue->Float64 = clamp_cast<double, T>(value); break;
        default: assert(false);
        }
    }
} // namespace Dml
