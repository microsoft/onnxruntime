// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

class OperatorField;

struct AbstractOperatorDesc
{
    const DML_OPERATOR_SCHEMA* schema = nullptr;
    std::vector<OperatorField> fields;

    // NOTE (clang-cl / two-phase name lookup):
    // OperatorField and AbstractOperatorDesc are mutually recursive - OperatorField holds a
    // std::optional<AbstractOperatorDesc> (see GeneratedSchemaTypes.h) - so OperatorField is
    // necessarily an incomplete type at this point. Any member function that instantiates
    // std::vector<OperatorField> operations requiring a complete element type - the special
    // member functions (which may need to destroy the vector) and the GetTensors iteration -
    // must therefore be declared here and defined out of line below, after OperatorField has
    // been completed. MSVC's cl.exe defers these instantiations and tolerates inline
    // definitions, but conforming two-phase compilers such as clang-cl instantiate them at
    // definition time and fail with "incomplete type 'OperatorField'".
    AbstractOperatorDesc();
    AbstractOperatorDesc(const DML_OPERATOR_SCHEMA* schema, std::vector<OperatorField>&& fields);
    AbstractOperatorDesc(const AbstractOperatorDesc&);
    AbstractOperatorDesc(AbstractOperatorDesc&&) noexcept;
    AbstractOperatorDesc& operator=(const AbstractOperatorDesc&);
    AbstractOperatorDesc& operator=(AbstractOperatorDesc&&) noexcept;
    ~AbstractOperatorDesc();

    std::vector<DmlBufferTensorDesc*> GetInputTensors();
    std::vector<const DmlBufferTensorDesc*> GetInputTensors() const;
    std::vector<DmlBufferTensorDesc*> GetOutputTensors();
    std::vector<const DmlBufferTensorDesc*> GetOutputTensors() const;

private:
    template <typename TensorType, DML_SCHEMA_FIELD_KIND Kind>
    std::vector<TensorType*> GetTensors() const;
};

// Complete OperatorField before defining the members that require it. GeneratedSchemaTypes.h is
// guarded by #pragma once, so its subsequent include from precomp.h is a no-op and the
// established include order is preserved.
#include "GeneratedSchemaTypes.h"

inline AbstractOperatorDesc::AbstractOperatorDesc() = default;

inline AbstractOperatorDesc::AbstractOperatorDesc(const DML_OPERATOR_SCHEMA* schema, std::vector<OperatorField>&& fields)
    : schema(schema)
    , fields(std::move(fields))
{
}

inline AbstractOperatorDesc::AbstractOperatorDesc(const AbstractOperatorDesc&) = default;
inline AbstractOperatorDesc::AbstractOperatorDesc(AbstractOperatorDesc&&) noexcept = default;
inline AbstractOperatorDesc& AbstractOperatorDesc::operator=(const AbstractOperatorDesc&) = default;
inline AbstractOperatorDesc& AbstractOperatorDesc::operator=(AbstractOperatorDesc&&) noexcept = default;
inline AbstractOperatorDesc::~AbstractOperatorDesc() = default;

inline std::vector<DmlBufferTensorDesc*> AbstractOperatorDesc::GetInputTensors()
{
    return GetTensors<DmlBufferTensorDesc, DML_SCHEMA_FIELD_KIND_INPUT_TENSOR>();
}

inline std::vector<const DmlBufferTensorDesc*> AbstractOperatorDesc::GetInputTensors() const
{
    return GetTensors<const DmlBufferTensorDesc, DML_SCHEMA_FIELD_KIND_INPUT_TENSOR>();
}

inline std::vector<DmlBufferTensorDesc*> AbstractOperatorDesc::GetOutputTensors()
{
    return GetTensors<DmlBufferTensorDesc, DML_SCHEMA_FIELD_KIND_OUTPUT_TENSOR>();
}

inline std::vector<const DmlBufferTensorDesc*> AbstractOperatorDesc::GetOutputTensors() const
{
    return GetTensors<const DmlBufferTensorDesc, DML_SCHEMA_FIELD_KIND_OUTPUT_TENSOR>();
}

template <typename TensorType, DML_SCHEMA_FIELD_KIND Kind>
std::vector<TensorType*> AbstractOperatorDesc::GetTensors() const
{
    std::vector<TensorType*> tensors;
    for (auto& field : fields)
    {
        const DML_SCHEMA_FIELD* fieldSchema = field.GetSchema();
        if (fieldSchema->Kind != Kind)
        {
            continue;
        }

        if (fieldSchema->Type == DML_SCHEMA_FIELD_TYPE_TENSOR_DESC)
        {
            auto& tensor = field.AsTensorDesc();
            tensors.push_back(tensor ? const_cast<TensorType*>(&*tensor) : nullptr);
        }
        else if (fieldSchema->Type == DML_SCHEMA_FIELD_TYPE_TENSOR_DESC_ARRAY)
        {
            auto& tensorArray = field.AsTensorDescArray();
            if (tensorArray)
            {
                for (auto& tensor : *tensorArray)
                {
                    tensors.push_back(const_cast<TensorType*>(&tensor));
                }
            }
        }
    }
    return tensors;
}
