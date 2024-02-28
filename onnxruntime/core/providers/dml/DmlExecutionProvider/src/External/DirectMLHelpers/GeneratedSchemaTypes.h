// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

using ApiAttributeVariant = std::variant<
    const DML_TENSOR_DESC*, 
    const DML_OPERATOR_DESC*, 
    UINT, 
    UINT64, 
    INT, 
    FLOAT, 
    const UINT*, 
    const INT*, 
    const FLOAT*, 
    const DML_SCALE_BIAS*, 
    DML_SIZE_2D, 
    DML_SCALAR_UNION, 
    BOOL
    >;

namespace OperatorFieldTypes
{
    using TensorDesc = std::optional<DmlBufferTensorDesc>; // DML_SCHEMA_FIELD_TYPE_TENSOR_DESC
    using TensorDescArray = std::optional<std::vector<DmlBufferTensorDesc>>; // DML_SCHEMA_FIELD_TYPE_TENSOR_DESC_ARRAY
    using FusedActivationOperatorDesc = std::optional<AbstractOperatorDesc>; // DML_SCHEMA_FIELD_TYPE_OPERATOR_DESC
    using FusedActivationOperatorDescArray = std::optional<std::vector<AbstractOperatorDesc>>; // DML_SCHEMA_FIELD_TYPE_OPERATOR_DESC_ARRAY
    using UInt = uint32_t; // DML_SCHEMA_FIELD_TYPE_UINT
    using UInt64 = uint64_t; // DML_SCHEMA_FIELD_TYPE_UINT64
    using Int = int32_t; // DML_SCHEMA_FIELD_TYPE_INT
    using Float = float; // DML_SCHEMA_FIELD_TYPE_FLOAT
    using UIntArray = std::vector<uint32_t>; // DML_SCHEMA_FIELD_TYPE_UINT_ARRAY
    using IntArray = std::vector<int32_t>; // DML_SCHEMA_FIELD_TYPE_INT_ARRAY
    using FloatArray = std::vector<float>; // DML_SCHEMA_FIELD_TYPE_FLOAT_ARRAY
    using ScaleBias = std::optional<DML_SCALE_BIAS>; // DML_SCHEMA_FIELD_TYPE_SCALE_BIAS
    using Size2D = DML_SIZE_2D; // DML_SCHEMA_FIELD_TYPE_SIZE_2D
    using ScalarUnion = DML_SCALAR_UNION; // DML_SCHEMA_FIELD_TYPE_SCALAR_UNION
    using Bool = bool; // DML_SCHEMA_FIELD_TYPE_BOOL
}

using OperatorFieldVariant = std::variant<
    OperatorFieldTypes::TensorDesc, 
    OperatorFieldTypes::TensorDescArray, 
    OperatorFieldTypes::FusedActivationOperatorDesc, 
    OperatorFieldTypes::FusedActivationOperatorDescArray, 
    OperatorFieldTypes::UInt, 
    OperatorFieldTypes::UInt64, 
    OperatorFieldTypes::Int, 
    OperatorFieldTypes::Float, 
    OperatorFieldTypes::UIntArray, 
    OperatorFieldTypes::IntArray, 
    OperatorFieldTypes::FloatArray, 
    OperatorFieldTypes::ScaleBias, 
    OperatorFieldTypes::Size2D, 
    OperatorFieldTypes::ScalarUnion, 
    OperatorFieldTypes::Bool
    >;

class OperatorField
{
public:
    OperatorField() = default;
    explicit OperatorField(const DML_SCHEMA_FIELD* schema, OperatorFieldVariant&& data)
        : m_schema(schema)
        , m_data(std::move(data))
    {
        assert(m_schema->Type == (DML_SCHEMA_FIELD_TYPE)m_data.index());
    }

    const DML_SCHEMA_FIELD* GetSchema() const
    {
        return m_schema;
    }

    const OperatorFieldVariant& GetData() const
    {
        return m_data;
    }

    const OperatorFieldTypes::TensorDesc& AsTensorDesc() const { return std::get<OperatorFieldTypes::TensorDesc>(m_data); }
    OperatorFieldTypes::TensorDesc& AsTensorDesc() { return std::get<OperatorFieldTypes::TensorDesc>(m_data); }

    const OperatorFieldTypes::TensorDescArray& AsTensorDescArray() const { return std::get<OperatorFieldTypes::TensorDescArray>(m_data); }
    OperatorFieldTypes::TensorDescArray& AsTensorDescArray() { return std::get<OperatorFieldTypes::TensorDescArray>(m_data); }

    const OperatorFieldTypes::FusedActivationOperatorDesc& AsFusedActivationOperatorDesc() const { return std::get<OperatorFieldTypes::FusedActivationOperatorDesc>(m_data); }
    OperatorFieldTypes::FusedActivationOperatorDesc& AsFusedActivationOperatorDesc() { return std::get<OperatorFieldTypes::FusedActivationOperatorDesc>(m_data); }

    const OperatorFieldTypes::FusedActivationOperatorDescArray& AsFusedActivationOperatorDescArray() const { return std::get<OperatorFieldTypes::FusedActivationOperatorDescArray>(m_data); }
    OperatorFieldTypes::FusedActivationOperatorDescArray& AsFusedActivationOperatorDescArray() { return std::get<OperatorFieldTypes::FusedActivationOperatorDescArray>(m_data); }

    const OperatorFieldTypes::UInt& AsUInt() const { return std::get<OperatorFieldTypes::UInt>(m_data); }
    OperatorFieldTypes::UInt& AsUInt() { return std::get<OperatorFieldTypes::UInt>(m_data); }

    const OperatorFieldTypes::UInt64& AsUInt64() const { return std::get<OperatorFieldTypes::UInt64>(m_data); }
    OperatorFieldTypes::UInt64& AsUInt64() { return std::get<OperatorFieldTypes::UInt64>(m_data); }

    const OperatorFieldTypes::Int& AsInt() const { return std::get<OperatorFieldTypes::Int>(m_data); }
    OperatorFieldTypes::Int& AsInt() { return std::get<OperatorFieldTypes::Int>(m_data); }

    const OperatorFieldTypes::Float& AsFloat() const { return std::get<OperatorFieldTypes::Float>(m_data); }
    OperatorFieldTypes::Float& AsFloat() { return std::get<OperatorFieldTypes::Float>(m_data); }

    const OperatorFieldTypes::UIntArray& AsUIntArray() const { return std::get<OperatorFieldTypes::UIntArray>(m_data); }
    OperatorFieldTypes::UIntArray& AsUIntArray() { return std::get<OperatorFieldTypes::UIntArray>(m_data); }

    const OperatorFieldTypes::IntArray& AsIntArray() const { return std::get<OperatorFieldTypes::IntArray>(m_data); }
    OperatorFieldTypes::IntArray& AsIntArray() { return std::get<OperatorFieldTypes::IntArray>(m_data); }

    const OperatorFieldTypes::FloatArray& AsFloatArray() const { return std::get<OperatorFieldTypes::FloatArray>(m_data); }
    OperatorFieldTypes::FloatArray& AsFloatArray() { return std::get<OperatorFieldTypes::FloatArray>(m_data); }

    const OperatorFieldTypes::ScaleBias& AsScaleBias() const { return std::get<OperatorFieldTypes::ScaleBias>(m_data); }
    OperatorFieldTypes::ScaleBias& AsScaleBias() { return std::get<OperatorFieldTypes::ScaleBias>(m_data); }

    const OperatorFieldTypes::Size2D& AsSize2D() const { return std::get<OperatorFieldTypes::Size2D>(m_data); }
    OperatorFieldTypes::Size2D& AsSize2D() { return std::get<OperatorFieldTypes::Size2D>(m_data); }

    const OperatorFieldTypes::ScalarUnion& AsScalarUnion() const { return std::get<OperatorFieldTypes::ScalarUnion>(m_data); }
    OperatorFieldTypes::ScalarUnion& AsScalarUnion() { return std::get<OperatorFieldTypes::ScalarUnion>(m_data); }

    const OperatorFieldTypes::Bool& AsBool() const { return std::get<OperatorFieldTypes::Bool>(m_data); }
    OperatorFieldTypes::Bool& AsBool() { return std::get<OperatorFieldTypes::Bool>(m_data); }

private:
    const DML_SCHEMA_FIELD* m_schema;
    OperatorFieldVariant m_data;
};

