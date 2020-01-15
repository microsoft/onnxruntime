// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

using ApiAttributeVariant = std::variant<
    const DML_TENSOR_DESC*, 
    const DML_OPERATOR_DESC*, 
    UINT, 
    INT, 
    FLOAT, 
    const UINT*, 
    const FLOAT*, 
    const DML_SCALE_BIAS*, 
    DML_SIZE_2D
    >;

namespace OperatorFieldTypes
{
    using TensorDesc = std::optional<DmlBufferTensorDesc>; // DML_SCHEMA_FIELD_TYPE_TENSOR_DESC
    using TensorDescArray = std::optional<std::vector<DmlBufferTensorDesc>>; // DML_SCHEMA_FIELD_TYPE_TENSOR_DESC_ARRAY
    using OperatorDesc = std::optional<AbstractOperatorDesc>; // DML_SCHEMA_FIELD_TYPE_OPERATOR_DESC
    using OperatorDescArray = std::optional<std::vector<AbstractOperatorDesc>>; // DML_SCHEMA_FIELD_TYPE_OPERATOR_DESC_ARRAY
    using UInt = uint32_t; // DML_SCHEMA_FIELD_TYPE_UINT
    using Int = int32_t; // DML_SCHEMA_FIELD_TYPE_INT
    using Float = float; // DML_SCHEMA_FIELD_TYPE_FLOAT
    using UIntArray = std::optional<std::vector<uint32_t>>; // DML_SCHEMA_FIELD_TYPE_UINT_ARRAY
    using FloatArray = std::optional<std::vector<float>>; // DML_SCHEMA_FIELD_TYPE_FLOAT_ARRAY
    using ScaleBias = std::optional<DML_SCALE_BIAS>; // DML_SCHEMA_FIELD_TYPE_SCALE_BIAS
    using Size2D = DML_SIZE_2D; // DML_SCHEMA_FIELD_TYPE_SIZE_2D
}

using OperatorFieldVariant = std::variant<
    OperatorFieldTypes::TensorDesc, 
    OperatorFieldTypes::TensorDescArray, 
    OperatorFieldTypes::OperatorDesc, 
    OperatorFieldTypes::OperatorDescArray, 
    OperatorFieldTypes::UInt, 
    OperatorFieldTypes::Int, 
    OperatorFieldTypes::Float, 
    OperatorFieldTypes::UIntArray, 
    OperatorFieldTypes::FloatArray, 
    OperatorFieldTypes::ScaleBias, 
    OperatorFieldTypes::Size2D
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

    const OperatorFieldTypes::OperatorDesc& AsOperatorDesc() const { return std::get<OperatorFieldTypes::OperatorDesc>(m_data); }
    OperatorFieldTypes::OperatorDesc& AsOperatorDesc() { return std::get<OperatorFieldTypes::OperatorDesc>(m_data); }

    const OperatorFieldTypes::OperatorDescArray& AsOperatorDescArray() const { return std::get<OperatorFieldTypes::OperatorDescArray>(m_data); }
    OperatorFieldTypes::OperatorDescArray& AsOperatorDescArray() { return std::get<OperatorFieldTypes::OperatorDescArray>(m_data); }

    const OperatorFieldTypes::UInt& AsUInt() const { return std::get<OperatorFieldTypes::UInt>(m_data); }
    OperatorFieldTypes::UInt& AsUInt() { return std::get<OperatorFieldTypes::UInt>(m_data); }

    const OperatorFieldTypes::Int& AsInt() const { return std::get<OperatorFieldTypes::Int>(m_data); }
    OperatorFieldTypes::Int& AsInt() { return std::get<OperatorFieldTypes::Int>(m_data); }

    const OperatorFieldTypes::Float& AsFloat() const { return std::get<OperatorFieldTypes::Float>(m_data); }
    OperatorFieldTypes::Float& AsFloat() { return std::get<OperatorFieldTypes::Float>(m_data); }

    const OperatorFieldTypes::UIntArray& AsUIntArray() const { return std::get<OperatorFieldTypes::UIntArray>(m_data); }
    OperatorFieldTypes::UIntArray& AsUIntArray() { return std::get<OperatorFieldTypes::UIntArray>(m_data); }

    const OperatorFieldTypes::FloatArray& AsFloatArray() const { return std::get<OperatorFieldTypes::FloatArray>(m_data); }
    OperatorFieldTypes::FloatArray& AsFloatArray() { return std::get<OperatorFieldTypes::FloatArray>(m_data); }

    const OperatorFieldTypes::ScaleBias& AsScaleBias() const { return std::get<OperatorFieldTypes::ScaleBias>(m_data); }
    OperatorFieldTypes::ScaleBias& AsScaleBias() { return std::get<OperatorFieldTypes::ScaleBias>(m_data); }

    const OperatorFieldTypes::Size2D& AsSize2D() const { return std::get<OperatorFieldTypes::Size2D>(m_data); }
    OperatorFieldTypes::Size2D& AsSize2D() { return std::get<OperatorFieldTypes::Size2D>(m_data); }

private:
    const DML_SCHEMA_FIELD* m_schema;
    OperatorFieldVariant m_data;
};

