// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Dml
{
    class TensorDesc
    {
    public:
        // Constructs an invalid / optional tensor desc.
        TensorDesc() = default;

        // Constructs a minimal tensor desc that knows type but lacks size information yet.
        TensorDesc(MLOperatorTensorDataType dataType);

        TensorDesc(
            DML_TENSOR_DATA_TYPE dataType,
            gsl::span<const uint32_t> sizes,
            std::optional<gsl::span<const uint32_t>> strides = std::nullopt,
            uint32_t guaranteedBaseOffsetAlignment = 0
            );

        TensorDesc(
            MLOperatorTensorDataType dataType,
            gsl::span<const uint32_t> dimensions, // Desired dimensions of tensor (after any broadcasting).
            gsl::span<const uint32_t> nonBroadcastDimensions, // Original dimensions (before any broadcasting). Usually same as 'dimensions'.
            int32_t coerceAxis,
            int32_t placement, // Adjustment offset of the passed dimensions within the minDimensionCount.
            int32_t leftAlignedDimensionCount, // Number of dimensions that remain left aligned when expanded to minimum count (INT32_MAX means all, 0 means all right aligned).
            uint32_t minDimensionCount,
            uint32_t guaranteedBaseOffsetAlignment
            );

        DML_TENSOR_DESC GetDmlDesc();

        inline DML_TENSOR_DATA_TYPE GetDmlDataType() const { return m_bufferTensorDesc.DataType; }
        inline MLOperatorTensorDataType GetMlOperatorDataType() const { return m_mlOperatorTensorDataType; }
        void ForceUnsignedDataType();

        inline bool IsValid() const { return m_tensorType != DML_TENSOR_TYPE_INVALID; }
        inline uint32_t GetDimensionCount() const { return m_bufferTensorDesc.DimensionCount; }
        void SetDimensionCount(uint32_t newDimensionCount, TensorAxis alignment);
        gsl::span<const uint32_t> GetSizes() const { return { m_sizes, m_sizes + m_bufferTensorDesc.DimensionCount }; }
        gsl::span<const uint32_t> GetStrides() const;
        void SetStrides(gsl::span<const uint32_t> strides);
        void PermuteDimensions(gsl::span<const uint32_t> dimensionMapping, const TensorAxis alignment);

        inline uint64_t GetBufferSizeInBytes() const
        {
            assert(m_tensorType == DML_TENSOR_TYPE_BUFFER);
            return m_bufferTensorDesc.TotalTensorSizeInBytes;
        }

        static TensorDesc ConstructDefaultTensorDesc(
            const MLOperatorTensorDataType dataType,
            gsl::span<const uint32_t> tensorShape)
        {
            return TensorDesc(
                        dataType,
                        tensorShape, // desired
                        tensorShape, // actual
                        TensorAxis::DoNotCoerce,
                        TensorAxis::W,
                        TensorAxis::RightAligned,
                        1, // minDimensionCount
                        0
                    );
        }

        static TensorDesc ConstructBroadcastedTensorDesc(
            const MLOperatorTensorDataType dataType,
            gsl::span<const uint32_t> desiredTensorShape,
            gsl::span<const uint32_t> actualTensorShape)
        {
            return TensorDesc(
                        dataType,
                        desiredTensorShape,
                        actualTensorShape,
                        TensorAxis::DoNotCoerce,
                        TensorAxis::W,
                        TensorAxis::RightAligned,
                        1, // minDimensionCount
                        0
                    );
        }

    private:
        DML_TENSOR_TYPE m_tensorType = DML_TENSOR_TYPE_INVALID;
        MLOperatorTensorDataType m_mlOperatorTensorDataType = MLOperatorTensorDataType::Undefined;
        uint32_t m_sizes[MaximumDimensionCount] = {};
        uint32_t m_strides[MaximumDimensionCount] = {};
        DML_BUFFER_TENSOR_DESC m_bufferTensorDesc = {};

        void EnsureStridesExist();
    };

    class TensorDescBuilder
    {
    public:
        inline TensorDescBuilder& SetDataType(MLOperatorTensorDataType dataType)
        {
            m_dataType = dataType;
            return *this;
        }

        inline TensorDescBuilder& SetDimensions(gsl::span<const uint32_t> dimensions)
        {
            m_dimensions = dimensions;
            return *this;
        }

        inline TensorDescBuilder& SetNonBroadcastDimensions(gsl::span<const uint32_t> nonBroadcastDimensions)
        {
            m_nonBroadcastDimensions = nonBroadcastDimensions;
            return *this;
        }

        inline TensorDescBuilder& SetCoerceAxis(TensorAxis coerceAxis)
        {
            m_coerceAxis = coerceAxis;
            return *this;
        }

        inline TensorDescBuilder& SetPlacement(TensorAxis placement)
        {
            m_placement = placement;
            return *this;
        }

        inline TensorDescBuilder& SetMinDimensionCount(uint32_t minDimensionCount)
        {
            m_minDimensionCount = minDimensionCount;
            return *this;
        }

        inline TensorDescBuilder& SetGuaranteedBaseOffsetAlignment(uint32_t guaranteedBaseOffsetAlignment)
        {
            m_guaranteedBaseOffsetAlignment = guaranteedBaseOffsetAlignment;
            return *this;
        }

        inline TensorDescBuilder& SetLeftAlignedDimensionCount(uint32_t leftAlignedDimensionCount)
        {
            m_leftAlignedDimensionCount = leftAlignedDimensionCount;
            return *this;
        }

        inline TensorDesc Create()
        {
            return TensorDesc(
                m_dataType,
                m_dimensions,
                m_nonBroadcastDimensions.size() > 0 ? m_nonBroadcastDimensions : m_dimensions,
                m_coerceAxis,
                m_placement,
                m_leftAlignedDimensionCount,
                m_minDimensionCount,
                m_guaranteedBaseOffsetAlignment
                );
        }

    private:
        MLOperatorTensorDataType m_dataType = MLOperatorTensorDataType::Undefined;
        gsl::span<const uint32_t> m_dimensions = {};
        gsl::span<const uint32_t> m_nonBroadcastDimensions = {};
        TensorAxis m_coerceAxis = TensorAxis::DoNotCoerce;
        TensorAxis m_placement = TensorAxis::NoPlacementAdjustment;
        int32_t m_leftAlignedDimensionCount = TensorAxis::RightAligned;
        uint32_t m_minDimensionCount = NchwDimensionCount;
        uint32_t m_guaranteedBaseOffsetAlignment = 0;
    };

    using TensorSequenceDesc = std::vector<TensorDesc>;
}
