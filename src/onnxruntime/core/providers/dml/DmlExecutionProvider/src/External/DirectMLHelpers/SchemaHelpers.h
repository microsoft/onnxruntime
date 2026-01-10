// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace SchemaHelpers
{
    inline AbstractOperatorDesc ConvertOperatorDesc(const DML_OPERATOR_DESC& opDesc);

    inline OperatorFieldTypes::TensorDesc ToOperatorFieldType(const DML_TENSOR_DESC* value)
    {
        return value ? OperatorFieldTypes::TensorDesc(*value) : std::nullopt;
    }

    inline OperatorFieldTypes::TensorDescArray ToOperatorFieldType(const DML_TENSOR_DESC* values, uint32_t count)
    {
        OperatorFieldTypes::TensorDescArray field;
        if (values && count != 0)
        {
            field.emplace(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                (*field)[i] = values[i];
            }
        }
        return field;
    }

    inline OperatorFieldTypes::FusedActivationOperatorDesc ToOperatorFieldType(const DML_OPERATOR_DESC* value)
    {
        return value ? OperatorFieldTypes::FusedActivationOperatorDesc(ConvertOperatorDesc(*value)) : std::nullopt;
    }

    inline OperatorFieldTypes::FusedActivationOperatorDescArray ToOperatorFieldType(const DML_OPERATOR_DESC* values, uint32_t count)
    {
        OperatorFieldTypes::FusedActivationOperatorDescArray field;
        if (values && count != 0)
        {
            field.emplace(count);
            for (uint32_t i = 0; i < count; ++i)
            {
                (*field)[i] = ConvertOperatorDesc(values[i]);
            }
        }
        return field;
    }

    inline OperatorFieldTypes::UInt ToOperatorFieldType(uint32_t value)
    {
        return value;
    }

    inline OperatorFieldTypes::UInt64 ToOperatorFieldType(uint64_t value)
    {
        return value;
    }

    inline OperatorFieldTypes::Int ToOperatorFieldType(int32_t value)
    {
        return value;
    }

    inline OperatorFieldTypes::Float ToOperatorFieldType(float value)
    {
        return value;
    }

    inline OperatorFieldTypes::Bool ToOperatorFieldType(bool value)
    {
        return value;
    }

    inline OperatorFieldTypes::UIntArray ToOperatorFieldType(const uint32_t* values, uint32_t count)
    {
        OperatorFieldTypes::UIntArray field;
        if (values && count != 0)
        {
            field.assign(values, values + count);
        }
        return field;
    }

    inline OperatorFieldTypes::IntArray ToOperatorFieldType(const int32_t* values, uint32_t count)
    {
        OperatorFieldTypes::IntArray field;
        if (values && count != 0)
        {
            field.assign(values, values + count);
        }
        return field;
    }

    inline OperatorFieldTypes::FloatArray ToOperatorFieldType(const float* values, uint32_t count)
    {
        OperatorFieldTypes::FloatArray field;
        if (values && count != 0)
        {
            field.assign(values, values + count);
        }
        return field;
    }

    inline OperatorFieldTypes::ScaleBias ToOperatorFieldType(const DML_SCALE_BIAS* value)
    {
        return value ? OperatorFieldTypes::ScaleBias(*value) : std::nullopt;
    }

    inline OperatorFieldTypes::Size2D ToOperatorFieldType(DML_SIZE_2D value)
    {
        return value;
    }

    inline OperatorFieldTypes::ScalarUnion ToOperatorFieldType(DML_SCALAR_UNION value)
    {
        return value;
    }

    class StructFieldWriter
    {
    public:
        explicit StructFieldWriter(gsl::span<byte> dst)
            : m_dst(dst)
            , m_bytesWritten(0)
        {}

        template <typename T>
        void Write(const T& value)
        {
            static_assert(std::is_trivial_v<T>, "Only trivial types are supported.");

            size_t dstOffset = RoundUpToMultiple(m_bytesWritten, alignof(T));
            size_t newBytesWritten = dstOffset + sizeof(value);

            assert(newBytesWritten <= gsl::narrow_cast<size_t>(m_dst.size()));
            memcpy(m_dst.data() + dstOffset, &value, sizeof(value));

            m_bytesWritten = newBytesWritten;
        }

    private:
        template <typename T>
        T RoundUpToMultiple(T value, T multiple)
        {
            static_assert(std::is_integral_v<T>);

            T remainder = value % multiple;
            if (remainder != 0)
            {
                value += multiple - remainder;
            }

            return value;
        }

        gsl::span<byte> m_dst;
        size_t m_bytesWritten;
    };

    template <size_t N>
    DML_BUFFER_TENSOR_DESC MakeBufferTensorDesc(const DmlBufferTensorDesc& src, StackAllocator<N>* allocator)
    {
        size_t dimensionCount = src.sizes.size();

        auto* sizes = allocator->template Allocate<UINT>(dimensionCount);
        std::copy_n(src.sizes.begin(), dimensionCount, sizes);

        UINT* strides = nullptr;
        if (src.strides)
        {
            strides = allocator->template Allocate<UINT>(dimensionCount);
            std::copy_n(src.strides->begin(), dimensionCount, strides);
        }

        DML_BUFFER_TENSOR_DESC dst;
        dst.DataType = src.dataType;
        dst.Flags = src.flags;
        dst.Sizes = sizes;
        dst.Strides = strides;
        dst.DimensionCount = static_cast<UINT>(dimensionCount);
        dst.TotalTensorSizeInBytes = src.totalTensorSizeInBytes;
        dst.GuaranteedBaseOffsetAlignment = src.guaranteedBaseOffsetAlignment;
        return dst;
    }

    template <size_t N>
    DML_TENSOR_DESC MakeTensorDesc(const DmlBufferTensorDesc& src, StackAllocator<N>* allocator)
    {
        auto* desc = allocator->template Allocate<DML_BUFFER_TENSOR_DESC>();
        *desc = MakeBufferTensorDesc(src, allocator);

        DML_TENSOR_DESC dst;
        dst.Type = DML_TENSOR_TYPE_BUFFER;
        dst.Desc = desc;
        return dst;
    }

    template <size_t N>
    DML_OPERATOR_DESC ConvertOperatorDesc(const AbstractOperatorDesc& abstractDesc, StackAllocator<N>* allocator);

    template <size_t N>
    void WriteOperatorDescField(const OperatorField& field, StructFieldWriter* dst, StackAllocator<N>* allocator)
    {
        const DML_SCHEMA_FIELD& schema = *field.GetSchema();

        switch (schema.Type)
        {
        case DML_SCHEMA_FIELD_TYPE_TENSOR_DESC:
        {
            DML_TENSOR_DESC* desc = nullptr;

            const auto& value = field.AsTensorDesc();
            if (value)
            {
                desc = allocator->template Allocate<DML_TENSOR_DESC>();
                *desc = MakeTensorDesc(*value, allocator);
            }

            dst->Write(desc);
        } break;

        case DML_SCHEMA_FIELD_TYPE_TENSOR_DESC_ARRAY:
        {
            DML_TENSOR_DESC* descs = nullptr;

            const auto& values = field.AsTensorDescArray();
            if (values)
            {
                descs = allocator->template Allocate<DML_TENSOR_DESC>(values->size());
                for (size_t i = 0; i < values->size(); ++i)
                {
                    descs[i] = MakeTensorDesc((*values)[i], allocator);
                }
            }

            dst->Write(descs);
        } break;

        case DML_SCHEMA_FIELD_TYPE_OPERATOR_DESC:
        {
            DML_OPERATOR_DESC* desc = nullptr;

            const auto& value = field.AsFusedActivationOperatorDesc();
            if (value)
            {
                desc = allocator->template Allocate<DML_OPERATOR_DESC>();
                *desc = ConvertOperatorDesc(*value, allocator);
            }

            dst->Write(desc);
        } break;

        case DML_SCHEMA_FIELD_TYPE_OPERATOR_DESC_ARRAY:
        {
            DML_OPERATOR_DESC* descs = nullptr;

            const auto& values = field.AsFusedActivationOperatorDescArray();
            if (values)
            {
                descs = allocator->template Allocate<DML_OPERATOR_DESC>(values->size());
                for (size_t i = 0; i < values->size(); ++i)
                {
                    descs[i] = ConvertOperatorDesc((*values)[i], allocator);
                }
            }

            dst->Write(descs);
        } break;

        case DML_SCHEMA_FIELD_TYPE_UINT:
        {
            uint32_t value = field.AsUInt();
            dst->Write(value);
        } break;

        case DML_SCHEMA_FIELD_TYPE_UINT64:
        {
            uint64_t value = field.AsUInt64();
            dst->Write(value);
        } break;

        case DML_SCHEMA_FIELD_TYPE_INT:
        {
            int32_t value = field.AsInt();
            dst->Write(value);
        } break;

        case DML_SCHEMA_FIELD_TYPE_FLOAT:
        {
            float value = field.AsFloat();
            dst->Write(value);
        } break;

        case DML_SCHEMA_FIELD_TYPE_BOOL:
        {
            // OperatorFieldTypes::Bool is a 'bool' (1 byte) but written as 'BOOL' in op descs (4 bytes).
            BOOL value = static_cast<BOOL>(field.AsBool());
            dst->Write(value);
        } break;

        case DML_SCHEMA_FIELD_TYPE_UINT_ARRAY:
        {
            uint32_t* arrayPtr = nullptr;

            const auto& values = field.AsUIntArray();
            arrayPtr = allocator->template Allocate<uint32_t>(values.size());
            std::copy(values.begin(), values.end(), arrayPtr);

            dst->Write(arrayPtr);
        } break;

        case DML_SCHEMA_FIELD_TYPE_INT_ARRAY:
        {
            int32_t* arrayPtr = nullptr;

            const auto& values = field.AsIntArray();
            arrayPtr = allocator->template Allocate<int32_t>(values.size());
            std::copy(values.begin(), values.end(), arrayPtr);

            dst->Write(arrayPtr);
        } break;

        case DML_SCHEMA_FIELD_TYPE_FLOAT_ARRAY:
        {
            float* arrayPtr = nullptr;

            const auto& values = field.AsFloatArray();
            arrayPtr = allocator->template Allocate<float>(values.size());
            std::copy(values.begin(), values.end(), arrayPtr);

            dst->Write(arrayPtr);
        } break;

        case DML_SCHEMA_FIELD_TYPE_SCALE_BIAS:
        {
            DML_SCALE_BIAS* scaleBias = nullptr;

            const auto& value = field.AsScaleBias();
            if (value)
            {
                scaleBias = allocator->template Allocate<DML_SCALE_BIAS>();
                *scaleBias = *value;
            }

            dst->Write(scaleBias);
        } break;

        case DML_SCHEMA_FIELD_TYPE_SIZE_2D:
        {
            DML_SIZE_2D value = field.AsSize2D();
            dst->Write(value);
        } break;

        case DML_SCHEMA_FIELD_TYPE_SCALAR_UNION:
        {
            uint64_t value = field.AsScalarUnion().UInt64;
            dst->Write(value);
        } break;

        default:
            assert(false);
            ORT_THROW_HR(E_UNEXPECTED);
        }
    }

    template <size_t N>
    DML_OPERATOR_DESC ConvertOperatorDesc(const AbstractOperatorDesc& abstractDesc, StackAllocator<N>* allocator)
    {
        const DML_OPERATOR_SCHEMA& schema = *abstractDesc.schema;

        // Retrieve the size of the ABI operator desc struct
        size_t abiDescSizeInBytes = ApiTraits::OperatorTypeVisitor(schema.OperatorType, [](auto tag) {
            using T = decltype(tag); // T is one of the DML_*_OPERATOR_DESC structs
            return sizeof(T);
        });

        // Allocate a blob of bytes to hold the struct
        byte* abiDesc = allocator->template Allocate<byte>(abiDescSizeInBytes);

        // Use the schema to write data into the blob

        StructFieldWriter writer(gsl::make_span(abiDesc, abiDescSizeInBytes));

        for (const OperatorField& field : abstractDesc.fields)
        {
            WriteOperatorDescField(field, &writer, allocator);
        }

        return DML_OPERATOR_DESC{ schema.OperatorType, abiDesc };
    }

} // namespace SchemaHelpers
