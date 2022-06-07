// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

class OperatorField;

struct AbstractOperatorDesc
{
    const DML_OPERATOR_SCHEMA* schema = nullptr;
    std::vector<OperatorField> fields;

    AbstractOperatorDesc() = default;
    AbstractOperatorDesc(const DML_OPERATOR_SCHEMA* schema, std::vector<OperatorField>&& fields)
        : schema(schema)
        , fields(std::move(fields))
    {}

    std::vector<DmlBufferTensorDesc*> GetInputTensors()
    {
        return GetTensors<DmlBufferTensorDesc, DML_SCHEMA_FIELD_KIND_INPUT_TENSOR>();
    }

    std::vector<const DmlBufferTensorDesc*> GetInputTensors() const
    {
        return GetTensors<const DmlBufferTensorDesc, DML_SCHEMA_FIELD_KIND_INPUT_TENSOR>();
    }

    std::vector<DmlBufferTensorDesc*> GetOutputTensors()
    {
        return GetTensors<DmlBufferTensorDesc, DML_SCHEMA_FIELD_KIND_OUTPUT_TENSOR>();
    }

    std::vector<const DmlBufferTensorDesc*> GetOutputTensors() const
    {
        return GetTensors<const DmlBufferTensorDesc, DML_SCHEMA_FIELD_KIND_OUTPUT_TENSOR>();
    }

private:
    template <typename TensorType, DML_SCHEMA_FIELD_KIND Kind>
    std::vector<TensorType*> GetTensors() const
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
};
