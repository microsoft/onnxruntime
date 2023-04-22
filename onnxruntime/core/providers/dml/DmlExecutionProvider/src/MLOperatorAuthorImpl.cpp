// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#include "core/framework/customregistry.h"
#include "core/framework/execution_frame.h"
#include "core/framework/TensorSeq.h"

#include "core/session/onnxruntime_c_api.h"
#include "core/providers/dml/DmlExecutionProvider/inc/MLOperatorAuthor.h"

#include "MLOperatorAuthorImpl.h"
#include "core/providers/dml/OperatorAuthorHelper/MLOperatorAuthorPrivate.h"

using namespace Microsoft::WRL;

namespace Windows::AI::MachineLearning::Adapter
{

#pragma warning(push)
#pragma warning(disable:4702)
    size_t AttributeValue::ElementCount() const
    {
        switch (type)
        {
        case MLOperatorAttributeType::Float:
            ML_CHECK_BOOL(floats.size() == 1);
            return 1;

        case MLOperatorAttributeType::Int:
            ML_CHECK_BOOL(ints.size() == 1);
            return 1;

        case MLOperatorAttributeType::String:
            ML_CHECK_BOOL(strings.size() == 1);
            return 1;

        case MLOperatorAttributeType::FloatArray:
            return floats.size();

        case MLOperatorAttributeType::IntArray:
            return ints.size();

        case MLOperatorAttributeType::StringArray:
            return strings.size();

        default:
            // The type is validated when default attributes are registered
            assert(false);
            ORT_THROW_HR(E_FAIL);
            return 0;
        }
#pragma warning(pop)
    }

    void AttributeValue::GetAttribute(
        MLOperatorAttributeType attributeType,
        uint32_t elementCount,
        size_t elementByteSize,
        void* value
        ) const
    {
        switch (attributeType)
        {
        case MLOperatorAttributeType::Float:
            ML_CHECK_BOOL(floats.size() == 1);
            __fallthrough;
        case MLOperatorAttributeType::FloatArray:
            ML_CHECK_BOOL(floats.size() == elementCount);
            ML_CHECK_BOOL(elementByteSize == sizeof(float));
            std::copy(floats.begin(), floats.end(), static_cast<float*>(value));
            break;

        case MLOperatorAttributeType::Int:
            ML_CHECK_BOOL(ints.size() == 1);
            __fallthrough;
        case MLOperatorAttributeType::IntArray:
            ML_CHECK_BOOL(ints.size() == elementCount);
            ML_CHECK_BOOL(elementByteSize == sizeof(int64_t));
            std::copy(ints.begin(), ints.end(), static_cast<int64_t*>(value));
            break;

        default:
            ORT_THROW_HR(E_INVALIDARG);
        }
    }

    const std::string* AttributeValue::GetStringAttribute(
        _In_z_ const char* attributeName,
        uint32_t elementIndex) const
    {
        ML_CHECK_BOOL((type == MLOperatorAttributeType::String && elementIndex == 0 && strings.size() == 1) ||
            (type == MLOperatorAttributeType::StringArray && elementIndex < strings.size()));

        return &strings.data()[elementIndex];
    }

    bool IsAllocationInterface(const ::OrtMemoryInfo& info)
    {
        return strcmp(info.name, onnxruntime::CPU) && !(info.mem_type == ::OrtMemType::OrtMemTypeCPUOutput || info.mem_type == ::OrtMemType::OrtMemTypeCPUInput);
    }

    // Translate the data object stored in a tensor to the type which will be returned through
    // the ABI. The translation is determined by the provider and based on options with which the
    // kernels are registered.
    void TranslateAllocationDataToAbi(
        IWinmlExecutionProvider* winmlProvider,
        bool isInternalOperator,
        const ::OrtMemoryInfo& allocInfo,
        IUnknown* allocation,
        IUnknown** abiAllocation)
    {
        if (winmlProvider)
        {
            winmlProvider->GetABIDataInterface(isInternalOperator, allocation, abiAllocation);
        }
        else
        {
            ComPtr<IUnknown> tmp = allocation;
            *abiAllocation = tmp.Detach();
        }
    }

    //
    // Traits for numeric attribute types
    //
    template <MLOperatorAttributeType T>
    struct MLAttributeTypeTraits {
    };

    template <>
    struct MLAttributeTypeTraits<MLOperatorAttributeType::Float> {
        using Type = float;
        static const onnx::AttributeProto_AttributeType ProtoType = onnx::AttributeProto_AttributeType_FLOAT;
        static const bool IsPrimitiveAttributeType = true;
        static const bool IsArray = false;
    };

    template <>
    struct MLAttributeTypeTraits<MLOperatorAttributeType::Int> {
        using Type = int64_t;
        static const onnx::AttributeProto_AttributeType ProtoType = onnx::AttributeProto_AttributeType_INT;
        static const bool IsPrimitiveAttributeType = true;
        static const bool IsArray = false;
    };

    template <>
    struct MLAttributeTypeTraits<MLOperatorAttributeType::String> {
        static const onnx::AttributeProto_AttributeType ProtoType = onnx::AttributeProto_AttributeType_STRING;
        static const bool IsPrimitiveAttributeType = true;
        static const bool IsArray = false;
    };

    template <>
    struct MLAttributeTypeTraits<MLOperatorAttributeTypeTensor> {
        static const onnx::AttributeProto_AttributeType ProtoType = onnx::AttributeProto_AttributeType_TENSOR;
        static const bool IsPrimitiveAttributeType = false;
        static const bool IsArray = false;
    };

    template <>
    struct MLAttributeTypeTraits<MLOperatorAttributeType::FloatArray> {
        using Type = float;
        static const onnx::AttributeProto_AttributeType ProtoType = onnx::AttributeProto_AttributeType_FLOATS;
        static const bool IsPrimitiveAttributeType = true;
        static const bool IsArray = true;
    };

    template <>
    struct MLAttributeTypeTraits<MLOperatorAttributeType::IntArray> {
        using Type = int64_t;
        static const onnx::AttributeProto_AttributeType ProtoType = onnx::AttributeProto_AttributeType_INTS;
        static const bool IsPrimitiveAttributeType = true;
        static const bool IsArray = true;
    };

    template <>
    struct MLAttributeTypeTraits<MLOperatorAttributeType::StringArray> {
        static const onnx::AttributeProto_AttributeType ProtoType = onnx::AttributeProto_AttributeType_STRINGS;
        static const bool IsPrimitiveAttributeType = true;
        static const bool IsArray = true;
    };

#define ML_ATTR_TO_PROTO_CASE(x)   \
  case MLOperatorAttributeType::x: \
    return MLAttributeTypeTraits<MLOperatorAttributeType::x>::ProtoType;

    onnx::AttributeProto_AttributeType ToProto(MLOperatorAttributeType type)
    {
        switch (type)
        {
        case MLOperatorAttributeType::Float:
            return MLAttributeTypeTraits<MLOperatorAttributeType::Float>::ProtoType;
        case MLOperatorAttributeType::Int:
            return MLAttributeTypeTraits<MLOperatorAttributeType::Int>::ProtoType;
        case MLOperatorAttributeType::FloatArray:
            return MLAttributeTypeTraits<MLOperatorAttributeType::FloatArray>::ProtoType;
#pragma warning(suppress:4063)
        case MLOperatorAttributeTypeTensor:
            return MLAttributeTypeTraits<MLOperatorAttributeTypeTensor>::ProtoType;
        case MLOperatorAttributeType::IntArray:
            return MLAttributeTypeTraits<MLOperatorAttributeType::IntArray>::ProtoType;
        case MLOperatorAttributeType::String:
            return MLAttributeTypeTraits<MLOperatorAttributeType::String>::ProtoType;
        case MLOperatorAttributeType::StringArray:
            return MLAttributeTypeTraits<MLOperatorAttributeType::StringArray>::ProtoType;
        default:
            return onnx::AttributeProto_AttributeType_UNDEFINED;
        }
    }

    bool IsPrimitiveAttributeType(MLOperatorAttributeType type)
    {
        switch (type)
        {
        case MLOperatorAttributeType::Float:
            return MLAttributeTypeTraits<MLOperatorAttributeType::Float>::IsPrimitiveAttributeType;
        case MLOperatorAttributeType::Int:
            return MLAttributeTypeTraits<MLOperatorAttributeType::Int>::IsPrimitiveAttributeType;
        case MLOperatorAttributeType::FloatArray:
            return MLAttributeTypeTraits<MLOperatorAttributeType::FloatArray>::IsPrimitiveAttributeType;
        case MLOperatorAttributeType::IntArray:
            return MLAttributeTypeTraits<MLOperatorAttributeType::IntArray>::IsPrimitiveAttributeType;
        case MLOperatorAttributeType::String:
            return MLAttributeTypeTraits<MLOperatorAttributeType::String>::IsPrimitiveAttributeType;
        case MLOperatorAttributeType::StringArray:
            return MLAttributeTypeTraits<MLOperatorAttributeType::StringArray>::IsPrimitiveAttributeType;
        default:
            return false;  // Including other types like tensor and graph...
        }
    }

#define ML_TENSOR_TYPE_CASE(x)          \
  if (onnxruntime::utils::IsPrimitiveDataType<x>(type)) \
  { \
    return MLTypeTraits<x>::TensorType; \
  }

#pragma warning(push)
#pragma warning(disable:4702)
    ::MLOperatorTensorDataType ToMLTensorDataType(onnxruntime::MLDataType type)
    {
        if (onnxruntime::utils::IsDataTypeString(type))
        {
            return MLOperatorTensorDataType::String;
        }

        ML_TENSOR_TYPE_CASE(float);
        ML_TENSOR_TYPE_CASE(uint8_t);
        ML_TENSOR_TYPE_CASE(int8_t);
        ML_TENSOR_TYPE_CASE(uint16_t);
        ML_TENSOR_TYPE_CASE(int16_t);
        ML_TENSOR_TYPE_CASE(int32_t);
        ML_TENSOR_TYPE_CASE(int64_t);
        ML_TENSOR_TYPE_CASE(bool);
        ML_TENSOR_TYPE_CASE(double);
        ML_TENSOR_TYPE_CASE(uint32_t);
        ML_TENSOR_TYPE_CASE(uint64_t);
        ML_TENSOR_TYPE_CASE(onnxruntime::MLFloat16);

        ORT_THROW_HR(E_NOTIMPL);
        return MLOperatorTensorDataType::Undefined;
#pragma warning(pop)
    }

#undef ML_TENSOR_TYPE_CASE
#define ML_TENSOR_TYPE_CASE(x)                            \
  if (type == MLTypeTraits<x>::TensorType)                \
  {                                                       \
    return onnxruntime::DataTypeImpl::GetTensorType<x>(); \
  }

#define ML_SEQUENCE_TENSOR_TYPE_CASE(x)                           \
  if (type == MLTypeTraits<x>::TensorType)                        \
  {                                                               \
    return onnxruntime::DataTypeImpl::GetSequenceTensorType<x>(); \
  }

#define ML_PRIMITIVE_TYPE_CASE(x)                   \
  if (type == MLTypeTraits<x>::TensorType)          \
  {                                                 \
    return onnxruntime::DataTypeImpl::GetType<x>(); \
  }

#pragma warning(push)
#pragma warning(disable:4702)
    onnxruntime::MLDataType ToMLDataType(::MLOperatorEdgeType edgeType, ::MLOperatorTensorDataType type)
    {
        if (edgeType == ::MLOperatorEdgeType::Tensor)
        {
            if (type == MLOperatorTensorDataType::String)
                return onnxruntime::DataTypeImpl::GetTensorType<std::string>();

            ML_TENSOR_TYPE_CASE(float);
            ML_TENSOR_TYPE_CASE(uint8_t);
            ML_TENSOR_TYPE_CASE(int8_t);
            ML_TENSOR_TYPE_CASE(uint16_t);
            ML_TENSOR_TYPE_CASE(int16_t);
            ML_TENSOR_TYPE_CASE(int32_t);
            ML_TENSOR_TYPE_CASE(int64_t);
            ML_TENSOR_TYPE_CASE(bool);
            ML_TENSOR_TYPE_CASE(double);
            ML_TENSOR_TYPE_CASE(uint32_t);
            ML_TENSOR_TYPE_CASE(uint64_t);
            ML_TENSOR_TYPE_CASE(onnxruntime::MLFloat16);

            ORT_THROW_HR(E_NOTIMPL);
            return onnxruntime::DataTypeImpl::GetTensorType<float>();
        }
        else if (edgeType == ::MLOperatorEdgeType::SequenceTensor)
        {
            if (type == MLOperatorTensorDataType::String)
                return onnxruntime::DataTypeImpl::GetSequenceTensorType<std::string>();

            ML_SEQUENCE_TENSOR_TYPE_CASE(float);
            ML_SEQUENCE_TENSOR_TYPE_CASE(uint8_t);
            ML_SEQUENCE_TENSOR_TYPE_CASE(int8_t);
            ML_SEQUENCE_TENSOR_TYPE_CASE(uint16_t);
            ML_SEQUENCE_TENSOR_TYPE_CASE(int16_t);
            ML_SEQUENCE_TENSOR_TYPE_CASE(int32_t);
            ML_SEQUENCE_TENSOR_TYPE_CASE(int64_t);
            ML_SEQUENCE_TENSOR_TYPE_CASE(bool);
            ML_SEQUENCE_TENSOR_TYPE_CASE(double);
            ML_SEQUENCE_TENSOR_TYPE_CASE(uint32_t);
            ML_SEQUENCE_TENSOR_TYPE_CASE(uint64_t);
            ML_SEQUENCE_TENSOR_TYPE_CASE(onnxruntime::MLFloat16);

            ORT_THROW_HR(E_NOTIMPL);
            return onnxruntime::DataTypeImpl::GetSequenceTensorType<float>();
        }
        else if (edgeType == ::MLOperatorEdgeType::Primitive)
        {
            if (type == MLOperatorTensorDataType::String)
                return onnxruntime::DataTypeImpl::GetType<std::string>();

            ML_PRIMITIVE_TYPE_CASE(float);
            ML_PRIMITIVE_TYPE_CASE(uint8_t);
            ML_PRIMITIVE_TYPE_CASE(int8_t);
            ML_PRIMITIVE_TYPE_CASE(uint16_t);
            ML_PRIMITIVE_TYPE_CASE(int16_t);
            ML_PRIMITIVE_TYPE_CASE(int32_t);
            ML_PRIMITIVE_TYPE_CASE(int64_t);
            ML_PRIMITIVE_TYPE_CASE(bool);
            ML_PRIMITIVE_TYPE_CASE(double);
            ML_PRIMITIVE_TYPE_CASE(uint32_t);
            ML_PRIMITIVE_TYPE_CASE(uint64_t);
            ML_PRIMITIVE_TYPE_CASE(onnxruntime::MLFloat16);

            ORT_THROW_HR(E_NOTIMPL);
            return onnxruntime::DataTypeImpl::GetType<float>();
        }
#pragma warning(pop)
        ORT_THROW_HR(E_NOTIMPL);
    }


#pragma warning(push)
#pragma warning(disable:4702)
    ::MLOperatorTensorDataType ToMLTensorDataType(onnx::TensorProto_DataType type)
    {
        switch (type)
        {
        case onnx::TensorProto_DataType_FLOAT:
            return MLOperatorTensorDataType::Float;

        case onnx::TensorProto_DataType_UINT8:
            return MLOperatorTensorDataType::UInt8;

        case onnx::TensorProto_DataType_INT8:
            return MLOperatorTensorDataType::Int8;

        case onnx::TensorProto_DataType_UINT16:
            return MLOperatorTensorDataType::UInt16;

        case onnx::TensorProto_DataType_INT16:
            return MLOperatorTensorDataType::Int16;

        case onnx::TensorProto_DataType_INT32:
            return MLOperatorTensorDataType::Int32;

        case onnx::TensorProto_DataType_INT64:
            return MLOperatorTensorDataType::Int64;

        case onnx::TensorProto_DataType_STRING:
            return MLOperatorTensorDataType::String;

        case onnx::TensorProto_DataType_BOOL:
            return MLOperatorTensorDataType::Bool;

        case onnx::TensorProto_DataType_FLOAT16:
            return MLOperatorTensorDataType::Float16;

        case onnx::TensorProto_DataType_DOUBLE:
            return MLOperatorTensorDataType::Double;

        case onnx::TensorProto_DataType_UINT32:
            return MLOperatorTensorDataType::UInt32;

        case onnx::TensorProto_DataType_UINT64:
            return MLOperatorTensorDataType::UInt64;

        case onnx::TensorProto_DataType_COMPLEX64:
            return MLOperatorTensorDataType::Complex64;

        case onnx::TensorProto_DataType_COMPLEX128:
            return MLOperatorTensorDataType::Complex128;

        default:
            ORT_THROW_HR(E_NOTIMPL);
            return MLOperatorTensorDataType::Undefined;
        }
#pragma warning(pop)
    }

    ::MLOperatorEdgeDescription ToMLEdgeDesc(const onnx::TypeProto* type)
    {
        // Initialized to undefined class and data type
        MLOperatorEdgeDescription ret = {};

        ML_CHECK_BOOL(type->value_case() == onnx::TypeProto::kTensorType ||
            type->value_case() == onnx::TypeProto::kSequenceType ||
            type->value_case() == onnx::TypeProto::VALUE_NOT_SET);

        if (type->value_case() == onnx::TypeProto::kTensorType)
        {
            ret.edgeType = MLOperatorEdgeType::Tensor;
            const onnx::TypeProto_Tensor tensorType = type->tensor_type();
            if (tensorType.has_elem_type())
            {
                ret.tensorDataType = ToMLTensorDataType(onnx::TensorProto_DataType(tensorType.elem_type()));
            }
        }
        else if (type->value_case() == onnx::TypeProto::kSequenceType)
        {
            ret.edgeType = MLOperatorEdgeType::SequenceTensor;
            const auto& tensorType = type->sequence_type().elem_type().tensor_type();
            if (tensorType.has_elem_type())
            {
                ret.tensorDataType = ToMLTensorDataType(onnx::TensorProto_DataType(tensorType.elem_type()));
            }
        }

        return ret;
    }

#pragma warning(push)
#pragma warning(disable:4702)
    std::string ToTypeString(MLOperatorEdgeDescription desc)
    {
        if (desc.edgeType == MLOperatorEdgeType::Tensor)
        {
            switch (desc.tensorDataType)
            {
            case MLOperatorTensorDataType::Float:
                return "tensor(float)";

            case MLOperatorTensorDataType::UInt8:
                return "tensor(uint8)";

            case MLOperatorTensorDataType::Int8:
                return "tensor(int8)";

            case MLOperatorTensorDataType::UInt16:
                return "tensor(uint16)";

            case MLOperatorTensorDataType::Int16:
                return "tensor(int16)";

            case MLOperatorTensorDataType::Int32:
                return "tensor(int32)";

            case MLOperatorTensorDataType::Int64:
                return "tensor(int64)";

            case MLOperatorTensorDataType::String:
                return "tensor(string)";

            case MLOperatorTensorDataType::Bool:
                return "tensor(bool)";

            case MLOperatorTensorDataType::Float16:
                return "tensor(float16)";

            case MLOperatorTensorDataType::Double:
                return "tensor(double)";

            case MLOperatorTensorDataType::UInt32:
                return "tensor(uint32)";

            case MLOperatorTensorDataType::UInt64:
                return "tensor(uint64)";

            case MLOperatorTensorDataType::Complex64:
                return "tensor(complext64)";

            case MLOperatorTensorDataType::Complex128:
                return "tensor(complext128)";

            default:
                ORT_THROW_HR(E_NOTIMPL);
                return "";
            }
        }
        else if (desc.edgeType == MLOperatorEdgeType::SequenceTensor)
        {
            switch (desc.tensorDataType)
            {
            case MLOperatorTensorDataType::Float:
                return "seq(tensor(float))";

            case MLOperatorTensorDataType::UInt8:
                return "seq(tensor(uint8))";

            case MLOperatorTensorDataType::Int8:
                return "seq(tensor(int8))";

            case MLOperatorTensorDataType::UInt16:
                return "seq(tensor(uint16))";

            case MLOperatorTensorDataType::Int16:
                return "seq(tensor(int16))";

            case MLOperatorTensorDataType::Int32:
                return "seq(tensor(int32))";

            case MLOperatorTensorDataType::Int64:
                return "seq(tensor(int64))";

            case MLOperatorTensorDataType::String:
                return "seq(tensor(string))";

            case MLOperatorTensorDataType::Bool:
                return "seq(tensor(bool))";

            case MLOperatorTensorDataType::Float16:
                return "seq(tensor(float16))";

            case MLOperatorTensorDataType::Double:
                return "seq(tensor(double))";

            case MLOperatorTensorDataType::UInt32:
                return "seq(tensor(uint32))";

            case MLOperatorTensorDataType::UInt64:
                return "seq(tensor(uint64))";

            case MLOperatorTensorDataType::Complex64:
                return "seq(tensor(complext64))";

            case MLOperatorTensorDataType::Complex128:
                return "seq(tensor(complext128))";

            default:
                ORT_THROW_HR(E_NOTIMPL);
                return "";
            }
        }
        else
        {
            ORT_THROW_HR(E_NOTIMPL);
        }
#pragma warning(pop)
    }

    OpKernelInfoWrapper::OpKernelInfoWrapper(
        const onnxruntime::OpKernelInfo* kerneInfo,
        IUnknown* abiExecutionObject,
        const EdgeShapes* inputShapeOverrides,
        const EdgeShapes* inferredOutputShapes,
        bool allowInputShapeQuery,
        bool allowOutputShapeQuery,
        bool isInternalOperator,
        const AttributeMap* defaultAttributes,
        gsl::span<const uint32_t> requiredConstantCpuInputs,
        MLOperatorTensorGetter& constantInputGetter,
        const onnxruntime::OpKernelContext* kernelContext
        )
    :   OpNodeInfoWrapper(kerneInfo, inputShapeOverrides, defaultAttributes, requiredConstantCpuInputs, constantInputGetter, kernelContext),
        m_inferredOutputShapes(inferredOutputShapes),
        m_allowInputShapeQuery(allowInputShapeQuery),
        m_allowOutputShapeQuery(allowOutputShapeQuery),
        m_internalOperator(isInternalOperator),
        m_impl(kerneInfo),
        m_abiExecutionObject(abiExecutionObject)
    {
        const void* executionHandle = kerneInfo->GetExecutionProvider()->GetExecutionHandle();
        if (executionHandle)
        {
            // We assume the execution object inherits IUnknown as its first base
            ComPtr<IUnknown> providerExecutionObject = const_cast<IUnknown*>(static_cast<const IUnknown*>(executionHandle));
            providerExecutionObject.As(&m_winmlProvider);
        }

        assert(allowInputShapeQuery || !allowOutputShapeQuery);

        // The input may be exposed using non-overridden sizes.    Exposing output shapes requires
        // those shapes be provided here.
        assert(!allowOutputShapeQuery || (inferredOutputShapes != nullptr));
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetAttributeElementCount(
        _In_z_ const char* name,
        MLOperatorAttributeType type,
        uint32_t* elementCount
        ) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            *elementCount = 0;

            if (IsPrimitiveAttributeType(type))
            {
                *elementCount = m_impl->GetPrimitiveAttrElementCount(ToProto(type), std::string(name));
            }
            else
            {
                // ONNX runtime does not implement OpNodeProtoHelper<Impl_t>::GetPrimitiveAttrElementCount for tensors.
                // So we need to test presence a different way.

                const onnx::AttributeProto* attributeProto = m_impl->TryGetAttribute(std::string(name));
                *elementCount = attributeProto ? 1 : 0;
            }

            // Look for a value in the kernel's registered defaults if one was not found
            if (*elementCount == 0 && m_defaultAttributes)
            {
                auto defaultAttr = m_defaultAttributes->find(name);
                if (defaultAttr != m_defaultAttributes->end())
                {
                    *elementCount = static_cast<uint32_t>(defaultAttr->second.ElementCount());
                }
            }

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    template <MLOperatorAttributeType T>
    HRESULT OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetAttributeArrayHelper(
        _In_z_ const char* name,
        uint32_t elementCount,
        uint32_t elementByteSize,
        void* values
        ) const
    {
        using elementType_t = typename MLAttributeTypeTraits<T>::Type;
        static_assert(MLAttributeTypeTraits<T>::IsArray, "This function only works with array types.");
        ML_CHECK_BOOL(sizeof(elementType_t) == elementByteSize);

        THROW_IF_NOT_OK(m_impl->GetAttrs(name, gsl::span<elementType_t>(static_cast<typename MLAttributeTypeTraits<T>::Type*>(values), elementCount)));
        return S_OK;
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetAttribute(
        _In_z_ const char* name,
        MLOperatorAttributeType type,
        uint32_t elementCount,
        size_t elementByteSize,
        /*out*/void* attributeValue
        ) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            // Look for a value in the kernel's registered defaults if one does not exist otherwise
            if (m_impl->GetPrimitiveAttrElementCount(ToProto(type), name) == 0)
            {
                if (!m_defaultAttributes)
                {
                    ORT_THROW_HR(E_FAIL);
                }

                auto defaultAttr = m_defaultAttributes->find(name);
                if (defaultAttr == m_defaultAttributes->end())
                {
                    ORT_THROW_HR(E_FAIL);
                }

                defaultAttr->second.GetAttribute(type, elementCount, elementByteSize, /*out*/attributeValue);
            }
            else
            {
                switch (type)
                {
                case MLOperatorAttributeType::Float:
                    ML_CHECK_BOOL(elementCount == 1);
                    return GetAttributeHelper<MLOperatorAttributeType::Float>(name, static_cast<uint32_t>(elementByteSize), /*out*/attributeValue);

                case MLOperatorAttributeType::Int:
                    ML_CHECK_BOOL(elementCount == 1);
                    return GetAttributeHelper<MLOperatorAttributeType::Int>(name, static_cast<uint32_t>(elementByteSize), /*out*/attributeValue);

                case MLOperatorAttributeType::FloatArray:
                    return GetAttributeArrayHelper<MLOperatorAttributeType::FloatArray>(name, elementCount, static_cast<uint32_t>(elementByteSize), /*out*/attributeValue);

                case MLOperatorAttributeType::IntArray:
                    return GetAttributeArrayHelper<MLOperatorAttributeType::IntArray>(name, elementCount, static_cast<uint32_t>(elementByteSize), /*out*/attributeValue);

                default:
                    ML_CHECK_BOOL(false);
                    break;
              }
            }

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    const std::string* OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetStringAttribute(
        _In_z_ const char* name,
        uint32_t elementIndex
        ) const
    {
        // Get the proto attribute
        const onnx::AttributeProto* attr = m_impl->TryGetAttribute(std::string(name));

        // Look for a value in the kernel's registered defaults if one was not found
        if (!attr)
        {
            if (!m_defaultAttributes)
            {
                ORT_THROW_HR(E_FAIL);
            }

            auto defaultAttr = m_defaultAttributes->find(name);
            if (defaultAttr == m_defaultAttributes->end())
            {
                ORT_THROW_HR(E_FAIL);
            }

            return defaultAttr->second.GetStringAttribute(name, elementIndex);
        }
        else
        {
            // Get the string vector from the attribute
            if (attr->has_s())
            {
                return &attr->s();
            }
            else
            {
                //    Check the size of the vector
                ML_CHECK_BOOL(attr->strings_size() > 0);
                ML_CHECK_BOOL(elementIndex < static_cast<uint32_t>(attr->strings_size()));

                return &attr->strings(elementIndex);
            }
        }
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetStringAttributeElementLength(
        _In_z_ const char* name,
        uint32_t elementIndex,
        uint32_t* attributeElementByteLength
        ) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            *attributeElementByteLength = 0;
            const std::string* protoString = GetStringAttribute(name, elementIndex);

            // Check for overflow and casting safety
            ML_CHECK_BOOL(protoString->size() < protoString->size() + 1);
            ML_CHECK_BOOL(protoString->size() + 1 <= std::numeric_limits<uint32_t>::max());

            // Set the length including null termination
            *attributeElementByteLength = static_cast<uint32_t>(protoString->size() + 1);
            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetStringAttributeElement(
        _In_z_ const char* name,
        uint32_t elementIndex,
        uint32_t attributeElementByteLength,
        char* attributeElement
        ) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            const std::string* protoString = GetStringAttribute(name, elementIndex);

            size_t stringLength = protoString->size();
            ML_CHECK_BOOL(stringLength < attributeElementByteLength);
            memcpy(attributeElement, protoString->c_str(), stringLength + 1);

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    template <MLOperatorAttributeType T>
    HRESULT OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetAttributeHelper(
        _In_z_ const char* name,
        uint32_t elementByteSize,
        void* value
        ) const
    {
        using elementType_t = typename MLAttributeTypeTraits<T>::Type;
        static_assert(!MLAttributeTypeTraits<T>::IsArray, "This function only works for simple non-array types.");
        ML_CHECK_BOOL(sizeof(elementType_t) == elementByteSize);
        THROW_IF_NOT_OK(m_impl->template GetAttr<elementType_t>(name, static_cast<elementType_t*>(value)));
        return S_OK;
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetTensorAttribute(
        _In_z_ const char* name,
        _Outptr_ IMLOperatorTensor** tensor
        ) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            *tensor = nullptr;

          // Read the tensor if present, and wrap it in a IMLOperatorTensor.
          const onnx::AttributeProto* attributeProto = m_impl->TryGetAttribute(std::string(name));
          if (attributeProto)
          {
            if (attributeProto->has_t())
            {
              const onnx::TensorProto* tensorProto = &attributeProto->t();

              // An empty path is used as external weights are not currently supported in this case
              Microsoft::WRL::ComPtr<IMLOperatorTensor> tensorWrapper = wil::MakeOrThrow<OnnxTensorWrapper>(const_cast<onnx::TensorProto*>(tensorProto), onnxruntime::Path());
              *tensor = tensorWrapper.Detach();
              return S_OK;
            }
          }

            return E_INVALIDARG;  // The argument has no valid matching attribute.
        }
        ORT_CATCH_RETURN
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetInputEdgeDescription(uint32_t inputIndex, MLOperatorEdgeDescription* edgeDesc) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            memset(edgeDesc, 0, sizeof(*edgeDesc));
            const onnx::TypeProto* type = m_impl->GetInputType(inputIndex);
            ML_CHECK_BOOL(type != nullptr);
            *edgeDesc = ToMLEdgeDesc(type);

            assert(edgeDesc->edgeType != MLOperatorEdgeType::Undefined);
            assert(edgeDesc->tensorDataType != MLOperatorTensorDataType::Undefined);

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetOutputEdgeDescription(uint32_t outputIndex, MLOperatorEdgeDescription* edgeDesc) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            memset(edgeDesc, 0, sizeof(*edgeDesc));
            const onnx::TypeProto* type = m_impl->GetOutputType(outputIndex);
            ML_CHECK_BOOL(type != nullptr);
            *edgeDesc = ToMLEdgeDesc(type);

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetInputTensorShape(uint32_t inputIndex, uint32_t dimensionCount, uint32_t* dimensions) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            memset(dimensions, 0, dimensionCount * sizeof(dimensions[0]));
            if (inputIndex >= GetInputCount())
            {
                return E_INVALIDARG;
            }

            // Input shapes are determined either from the override or from the underlying proto
            if (m_inputShapesOverride)
            {
                if (m_inputShapesOverride->GetShape(inputIndex).size() != dimensionCount)
                {
                    return E_INVALIDARG;
                }

                for (uint32_t i = 0; i < dimensionCount; ++i)
                {
                    dimensions[i] = m_inputShapesOverride->GetShape(inputIndex)[i];
                }
            }
            else
            {
                const auto* inputType = m_impl->GetInputType(inputIndex);
                ML_CHECK_BOOL(inputType->has_tensor_type());
                for (uint32_t i = 0; i < dimensionCount; ++i)
                {
                    // Shape inference is only done when all dimensions of all inputs have known values,
                    // so the input tensors will always have shapes at this point.
                    assert(inputType->tensor_type().shape().dim(i).has_dim_value());
                    dimensions[i] = static_cast<uint32_t>(inputType->tensor_type().shape().dim(i).dim_value());
                }
            }

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetSequenceInputTensorShape(uint32_t inputIndex, uint32_t sequenceIndex, uint32_t dimensionCount, uint32_t* dimensions) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            memset(dimensions, 0, dimensionCount * sizeof(dimensions[0]));
            if (inputIndex >= GetInputCount())
            {
                return E_INVALIDARG;
            }

            // Input shapes are determined either from the override or from the underlying proto
            if (m_kernelContext)
            {
                assert(m_kernelContext->InputType(gsl::narrow_cast<int>(inputIndex))->IsTensorSequenceType());
                auto inputTensorSeq = m_kernelContext->Input<onnxruntime::TensorSeq>(gsl::narrow_cast<int>(inputIndex));
                ML_CHECK_BOOL(inputTensorSeq != nullptr);
                const auto& elemTensor = inputTensorSeq->Get(sequenceIndex);
                const auto& shape = elemTensor.Shape();
                for (uint32_t i = 0; i < dimensionCount; ++i)
                {
                    dimensions[i] = static_cast<uint32_t>(shape[i]);
                }
            }
            else if (m_inputShapesOverride)
            {
                if (m_inputShapesOverride->GetShape(inputIndex).size() != dimensionCount)
                {
                    return E_INVALIDARG;
                }

                for (uint32_t i = 0; i < dimensionCount; ++i)
                {
                    dimensions[i] = m_inputShapesOverride->GetShape(inputIndex)[i];
                }
            }
            else
            {
                const auto* inputType = m_impl->GetInputType(inputIndex);
                assert(inputType->has_sequence_type());
                ML_CHECK_BOOL(inputType->has_sequence_type());

                const auto& elemType = inputType->sequence_type().elem_type();

                for (uint32_t i = 0; i < dimensionCount; ++i)
                {
                    // Shape inference is only done when all dimensions of all inputs have known values,
                    // so the input tensors will always have shapes at this point.
                    assert(elemType.tensor_type().shape().dim(i).has_dim_value());
                    dimensions[i] = static_cast<uint32_t>(elemType.tensor_type().shape().dim(i).dim_value());
                }
            }

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    bool STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::IsInputValid(uint32_t inputIndex) const noexcept
    {
        if (IsClosed())
        {
            return false;
        }

        return (GetInputCount() > inputIndex) && !!m_impl->GetInputType(inputIndex);
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    bool STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::IsOutputValid(uint32_t outputIndex) const noexcept
    {
        if (IsClosed())
        {
            return false;
        }

        return (GetOutputCount() > outputIndex) && !!m_impl->GetOutputType(outputIndex);
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetInputTensorDimensionCount(uint32_t inputIndex, uint32_t* dimensionCount) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            *dimensionCount = 0;

            if (inputIndex >= GetInputCount())
            {
                return E_INVALIDARG;
            }

            // Input shapes are determined either from the override or from the underlying proto
            if (m_inputShapesOverride)
            {
                *dimensionCount = gsl::narrow_cast<uint32_t>(m_inputShapesOverride->GetShape(inputIndex).size());
            }
            else
            {
                const auto* inputType = m_impl->GetInputType(inputIndex);
                ML_CHECK_BOOL(inputType->has_tensor_type());

                // Shape inference is only done when all dimensions of all inputs have known values,
                // so the input tensors will always have shapes at this point.
                assert(inputType->tensor_type().has_shape());

                *dimensionCount = inputType->tensor_type().shape().dim_size();
            }

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetSequenceInputInfo(
        uint32_t inputIndex,
        uint32_t* inputCount,
        MLOperatorTensorDataType* dataType) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            ML_CHECK_BOOL(inputIndex < GetInputCount());
            ML_CHECK_BOOL(m_kernelContext != nullptr);

            // Input shapes are determined either from the input tensor, override or from the underlying proto
            assert(m_kernelContext->InputType(gsl::narrow_cast<int>(inputIndex))->IsTensorSequenceType());
            ML_CHECK_BOOL(m_kernelContext->InputType(gsl::narrow_cast<int>(inputIndex))->IsTensorSequenceType());
            auto inputTensorSeq = m_kernelContext->Input<onnxruntime::TensorSeq>(gsl::narrow_cast<int>(inputIndex));
            ML_CHECK_BOOL(inputTensorSeq != nullptr);
            *inputCount = static_cast<uint32_t>(inputTensorSeq->Size());
            *dataType = ToMLTensorDataType(inputTensorSeq->DataType());
            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetSequenceInputTensorDimensionCount(uint32_t inputIndex, uint32_t sequenceIndex, uint32_t* dimensionCount) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            *dimensionCount = 0;

            if (inputIndex >= GetInputCount())
            {
                return E_INVALIDARG;
            }

            // Input shapes are determined either from the input tensor, override or from the underlying proto
            if (m_kernelContext)
            {
                assert(m_kernelContext->InputType(gsl::narrow_cast<int>(inputIndex))->IsTensorSequenceType());
                auto inputTensorSeq = m_kernelContext->Input<onnxruntime::TensorSeq>(gsl::narrow_cast<int>(inputIndex));
                ML_CHECK_BOOL(inputTensorSeq != nullptr);
                const auto& elemTensor = inputTensorSeq->Get(sequenceIndex);
                *dimensionCount = static_cast<uint32_t>(elemTensor.Shape().NumDimensions());
            }
            else if (m_inputShapesOverride)
            {
                *dimensionCount = gsl::narrow_cast<uint32_t>(m_inputShapesOverride->GetShape(inputIndex).size());
            }
            else
            {
                const auto* inputType = m_impl->GetInputType(inputIndex);
                assert(inputType->has_sequence_type());
                ML_CHECK_BOOL(inputType->has_sequence_type());

                const auto& elemType = inputType->sequence_type().elem_type();

                // Shape inference is only done when all dimensions of all inputs have known values,
                // so the input tensors will always have shapes at this point.
                assert(elemType.tensor_type().has_shape());

                *dimensionCount = elemType.tensor_type().shape().dim_size();
            }

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetConstantInputTensor(uint32_t inputIndex, IMLOperatorTensor** tensor) const noexcept
    {
        ORT_TRY
        {
            bool inputRequiredAsConstant = std::find(
                                             m_requiredConstantCpuInputs.begin(),
                                             m_requiredConstantCpuInputs.end(),
                                             inputIndex) != m_requiredConstantCpuInputs.end();

            ORT_THROW_HR_IF(E_INVALIDARG, !inputRequiredAsConstant);

            auto constantInput = m_constantInputGetter(inputIndex);
            ORT_THROW_HR_IF(E_INVALIDARG, !std::holds_alternative<ComPtr<IMLOperatorTensor>>(constantInput));

            auto tensorWrapper = std::get<ComPtr<IMLOperatorTensor>>(constantInput);
            if (tensorWrapper == nullptr)
            {
                // This shouldn't happen since kernel creation is deferred and repeated when required constant inputs are not present.
                return E_UNEXPECTED;
            }

            *tensor = tensorWrapper.Detach();

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::TryGetConstantInputTensor(uint32_t inputIndex, IMLOperatorTensor** tensor) const noexcept
    {
        ORT_TRY
        {
            *tensor = nullptr;

            auto constantInput = m_constantInputGetter(inputIndex);
            if (!std::holds_alternative<ComPtr<IMLOperatorTensor>>(constantInput))
            {
                assert(std::find(
                    m_requiredConstantCpuInputs.begin(),
                    m_requiredConstantCpuInputs.end(),
                    inputIndex) == m_requiredConstantCpuInputs.end());

                return S_OK;
            }

            auto tensorWrapper = std::get<ComPtr<IMLOperatorTensor>>(constantInput);
            if (tensorWrapper == nullptr)
            {
                assert(std::find(
                    m_requiredConstantCpuInputs.begin(),
                    m_requiredConstantCpuInputs.end(),
                    inputIndex) == m_requiredConstantCpuInputs.end());

                return S_OK;
            }

            *tensor = tensorWrapper.Detach();

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT STDMETHODCALLTYPE OpKernelInfoWrapper::GetOutputTensorShape(uint32_t outputIndex, uint32_t dimensionCount, uint32_t* dimensions) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            memset(dimensions, 0, dimensionCount * sizeof(dimensions[0]));

            if (!HasOutputShapeDescription())
            {
                return E_FAIL;
            }

            if (outputIndex >= GetOutputCount())
            {
                return E_INVALIDARG;
            }

            if (m_inferredOutputShapes->GetShape(outputIndex).size() != dimensionCount)
            {
                return E_INVALIDARG;
            }

            for (uint32_t i = 0; i < dimensionCount; ++i)
            {
                dimensions[i] = m_inferredOutputShapes->GetShape(outputIndex)[i];
            }

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT STDMETHODCALLTYPE OpKernelInfoWrapper::GetOutputTensorDimensionCount(uint32_t outputIndex, uint32_t* dimensionCount) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            *dimensionCount = 0;

            if (!HasOutputShapeDescription())
            {
                return E_FAIL;
            }

            if (outputIndex >= GetOutputCount())
            {
                return E_INVALIDARG;
            }

            *dimensionCount = gsl::narrow_cast<uint32_t>(m_inferredOutputShapes->GetShape(outputIndex).size());

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    bool STDMETHODCALLTYPE OpKernelInfoWrapper::HasTensorShapeDescription() const noexcept
    {
        return m_allowInputShapeQuery;
    }

    HRESULT STDMETHODCALLTYPE OpKernelInfoWrapper::GetTensorShapeDescription(IMLOperatorTensorShapeDescription** shapeInfo) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            *shapeInfo = nullptr;

            if (!HasTensorShapeDescription())
            {
                *shapeInfo = nullptr;
                return E_FAIL;
                //return MLStatus::REQUIREMENT_NOT_REGISTERED;
            }

            ComPtr<IMLOperatorTensorShapeDescription> ret = const_cast<OpKernelInfoWrapper*>(this);
            *shapeInfo = ret.Detach();
            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    void STDMETHODCALLTYPE OpKernelInfoWrapper::GetExecutionInterface(IUnknown** executionInterface) const noexcept
    {
        m_abiExecutionObject.CopyTo(executionInterface);
    }

    uint32_t STDMETHODCALLTYPE OpKernelInfoWrapper::GetUtf8NameBufferSizeInBytes() const noexcept
    {
        // Include null terminator.
        return static_cast<uint32_t>(m_impl->node().Name().size() + 1);
    }

    HRESULT STDMETHODCALLTYPE OpKernelInfoWrapper::GetUtf8Name(uint32_t bufferSizeInBytes, char* outputName) const noexcept
    {
        if (bufferSizeInBytes == 0)
        {
            return E_INVALIDARG;
        }

        // Copy as many characters as possible, leaving room for the null terminator.
        const auto& nodeName = m_impl->node().Name();
        size_t charsCopied = nodeName.copy(outputName, bufferSizeInBytes - 1);

        // Write the null terminator.
        assert(charsCopied >= 0 && charsCopied < bufferSizeInBytes);
        outputName[charsCopied] = '\0';

        return S_OK;
    }

    uint32_t STDMETHODCALLTYPE OpKernelInfoWrapper::GetWideNameBufferSizeInBytes() const noexcept
    {
        const auto& name = m_impl->node().Name();
        if (name.empty())
        {
            // Include null terminator.
            return sizeof(wchar_t);
        }

        int requiredSizeInChars = MultiByteToWideChar(CP_UTF8, 0, name.data(), static_cast<int>(name.size()), nullptr, 0);
        assert(requiredSizeInChars > 0);

        // Include null terminator.
        return static_cast<uint32_t>((requiredSizeInChars + 1) * sizeof(wchar_t));
    }

    HRESULT STDMETHODCALLTYPE OpKernelInfoWrapper::GetWideName(uint32_t bufferSizeInBytes, wchar_t* outputName) const noexcept
    {
        // Buffer needs to be large enough to at least hold a null terminator.
        if (bufferSizeInBytes < sizeof(wchar_t))
        {
            return E_INVALIDARG;
        }

        const auto& nodeName = m_impl->node().Name();
        if (nodeName.empty())
        {
            outputName[0] = L'\0';
            return S_OK;
        }

        uint32_t bufferSizeInChars = bufferSizeInBytes / sizeof(wchar_t);
        int charsCopiedIfSucceeded = MultiByteToWideChar(CP_UTF8, 0, nodeName.data(), static_cast<int>(nodeName.size()), outputName, bufferSizeInChars);

        if (charsCopiedIfSucceeded > 0)
        {
            // The return value is only > 0 if ALL characters copied successfully.
            // Write null terminator at the end of copied chars, which may not be at the end of the buffer.
            outputName[charsCopiedIfSucceeded] = L'\0';
            return S_OK;
        }

        // An error must have occurred in MultiByteToWideChar.
        assert(charsCopiedIfSucceeded <= 0);
        auto lastError = GetLastError();

        if (lastError == ERROR_INSUFFICIENT_BUFFER)
        {
            // The buffer was too small, but MultiByteToWideChar will have copied as many chars as possible.
            // Truncate and overwrite last char with null terminator. Don't treat this as an error.
            outputName[bufferSizeInChars - 1] = L'\0';
            return S_OK;
        }

        assert(lastError == ERROR_INVALID_PARAMETER || lastError == ERROR_NO_UNICODE_TRANSLATION);
        return E_INVALIDARG;
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    uint32_t STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetInputCount() const noexcept
    {
        if (IsClosed())
        {
            return 0;
        }

        return m_impl->GetInputCount();
    }

    template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
    uint32_t STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetOutputCount() const noexcept
    {
        if (IsClosed())
        {
            return 0;
        }

        return m_impl->GetOutputCount();
    }

    bool STDMETHODCALLTYPE OpKernelInfoWrapper::HasOutputShapeDescription() const noexcept
    {
        return m_allowOutputShapeQuery;
    }

    DmlGraphOpKernelInfoWrapper::DmlGraphOpKernelInfoWrapper(
        const onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>* protoHelper,
        const void* executionHandle,
        bool isInternalOperator,
        const EdgeShapes* inputShapesOverrides,
        const EdgeShapes* inferredOutputShapes,
        const AttributeMap* defaultAttributes,
        DmlGraphNodeCreateInfo* graphNodeCreateInfo,
        gsl::span<const uint32_t> requiredConstantCpuInputs,
        MLOperatorTensorGetter& constantInputGetter
        )
    :   OpNodeInfoWrapper(protoHelper, inputShapesOverrides, defaultAttributes, requiredConstantCpuInputs, constantInputGetter, nullptr),
        m_inferredOutputShapes(inferredOutputShapes),
        m_internalOperator(isInternalOperator),
        m_graphNodeCreateInfo(graphNodeCreateInfo)
    {
        // We assume the execution object inherits IUnknown as its first base
        m_abiExecutionObject = const_cast<IUnknown*>(static_cast<const IUnknown*>(executionHandle));
        m_abiExecutionObject.As(&m_winmlProvider);
    }

    HRESULT STDMETHODCALLTYPE DmlGraphOpKernelInfoWrapper::GetOutputTensorShape(uint32_t outputIndex, uint32_t dimensionCount, uint32_t* dimensions) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            memset(dimensions, 0, dimensionCount * sizeof(dimensions[0]));

            if (!HasOutputShapeDescription())
            {
                return E_FAIL;
            }

            if (outputIndex >= GetOutputCount())
            {
                return E_INVALIDARG;
            }

            if (m_inferredOutputShapes->GetShape(outputIndex).size() != dimensionCount)
            {
                return E_INVALIDARG;
            }

            for (uint32_t i = 0; i < dimensionCount; ++i)
            {
                dimensions[i] = m_inferredOutputShapes->GetShape(outputIndex)[i];
            }

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT STDMETHODCALLTYPE DmlGraphOpKernelInfoWrapper::GetOutputTensorDimensionCount(uint32_t outputIndex, uint32_t* dimensionCount) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            *dimensionCount = 0;

            if (!HasOutputShapeDescription())
            {
                return E_FAIL;
            }

            if (outputIndex >= GetOutputCount())
            {
                return E_INVALIDARG;
            }

            *dimensionCount = gsl::narrow_cast<uint32_t>(m_inferredOutputShapes->GetShape(outputIndex).size());

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    bool STDMETHODCALLTYPE DmlGraphOpKernelInfoWrapper::HasTensorShapeDescription() const noexcept
    {
        return true;
    }

    HRESULT STDMETHODCALLTYPE DmlGraphOpKernelInfoWrapper::GetTensorShapeDescription(IMLOperatorTensorShapeDescription** shapeInfo) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            *shapeInfo = nullptr;

            if (!HasTensorShapeDescription())
            {
                *shapeInfo = nullptr;
                return E_FAIL;
                //return MLStatus::REQUIREMENT_NOT_REGISTERED;
            }

            ComPtr<IMLOperatorTensorShapeDescription> ret = const_cast<DmlGraphOpKernelInfoWrapper*>(this);
            *shapeInfo = ret.Detach();
            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    void STDMETHODCALLTYPE DmlGraphOpKernelInfoWrapper::GetExecutionInterface(IUnknown** executionInterface) const noexcept
    {
        m_abiExecutionObject.CopyTo(executionInterface);
    }

    bool STDMETHODCALLTYPE DmlGraphOpKernelInfoWrapper::HasOutputShapeDescription() const noexcept
    {
        // DML kernels are only used in graph in graph partitions when shapes are static
        return true;
    }

    bool STDMETHODCALLTYPE DmlGraphOpKernelInfoWrapper::IsDmlGraphNode() const noexcept
    {
        return (m_graphNodeCreateInfo != nullptr);
    }

    HRESULT STDMETHODCALLTYPE DmlGraphOpKernelInfoWrapper::SetDmlOperator(
        _In_ const MLOperatorGraphDesc* operatorGraphDesc
        ) const noexcept
    {
        ORT_TRY
        {
            assert(operatorGraphDesc != nullptr);
            // Either nodesAsOpDesc or nodesIDMLOperator can be present.
            assert(operatorGraphDesc->nodeCount == 0 || (!operatorGraphDesc->nodesAsOpDesc ^ !operatorGraphDesc->nodesAsIDMLOperator));

            if (operatorGraphDesc->nodesAsOpDesc)
            {
                m_graphNodeCreateInfo->nodesAsOperatorDesc = std::vector<std::unique_ptr<AbstractOperatorDesc>>();
                for (uint32_t nodeIndex = 0; nodeIndex < operatorGraphDesc->nodeCount; nodeIndex++)
                {
                    auto* node = operatorGraphDesc->nodesAsOpDesc[nodeIndex];
                    assert(node != nullptr);
                    AbstractOperatorDesc abstractDesc = SchemaHelpers::ConvertOperatorDesc(*node);
                    m_graphNodeCreateInfo->nodesAsOperatorDesc.push_back(std::make_unique<AbstractOperatorDesc>(std::move(abstractDesc)));
                }
            }
            else
            {
                m_graphNodeCreateInfo->nodesAsIDMLOperator = std::vector<Microsoft::WRL::ComPtr<IDMLOperator>>();
                for (uint32_t nodeIndex = 0; nodeIndex < operatorGraphDesc->nodeCount; nodeIndex++)
                {
                    auto* node = operatorGraphDesc->nodesAsIDMLOperator[nodeIndex];
                    assert(node != nullptr);
                    m_graphNodeCreateInfo->nodesAsIDMLOperator.push_back(node);
                }
            }

            // There can be operators (or kernels) which don't require any input.
            assert(operatorGraphDesc->inputEdgeCount == 0 || operatorGraphDesc->inputEdges != nullptr);
            m_graphNodeCreateInfo->inputEdges.insert(
                m_graphNodeCreateInfo->inputEdges.begin(),
                operatorGraphDesc->inputEdges,
                operatorGraphDesc->inputEdges + operatorGraphDesc->inputEdgeCount);

            // Operators (or kernels), which use single DML API, don't have any intermediate edge.
            assert(operatorGraphDesc->intermediateEdgeCount == 0 || operatorGraphDesc->intermediateEdges != nullptr);
            m_graphNodeCreateInfo->intermediateEdges.insert(
                m_graphNodeCreateInfo->intermediateEdges.begin(),
                operatorGraphDesc->intermediateEdges,
                operatorGraphDesc->intermediateEdges + operatorGraphDesc->intermediateEdgeCount);

            assert(operatorGraphDesc->outputEdgeCount == 0 || operatorGraphDesc->outputEdges != nullptr);
            m_graphNodeCreateInfo->outputEdges.insert(
                m_graphNodeCreateInfo->outputEdges.begin(),
                operatorGraphDesc->outputEdges,
                operatorGraphDesc->outputEdges + operatorGraphDesc->outputEdgeCount);

            m_graphNodeCreateInfo->nodeCount = operatorGraphDesc->nodeCount;
            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    OnnxTensorWrapper::OnnxTensorWrapper(onnx::TensorProto* impl, const onnxruntime::Path& modelPath) : m_impl(impl)
    {
        // The tensor may be stored as raw data or in typed fields.
        if (impl->has_raw_data())
        {
            m_dataPtr = reinterpret_cast<std::byte*>(impl->mutable_raw_data()->data());
            m_tensorByteSize = impl->raw_data().size();
        }
        else
        {
            std::tie(m_unpackedTensor, m_tensorByteSize) = UnpackTensor(*impl, modelPath);
            m_dataPtr = m_unpackedTensor.get();
        }
    }

    uint32_t STDMETHODCALLTYPE OnnxTensorWrapper::GetDimensionCount() const noexcept
    {
        if (IsClosed())
        {
            return 0;
        }

        return gsl::narrow_cast<uint32_t>(m_impl->dims().size());
    }

    HRESULT STDMETHODCALLTYPE OnnxTensorWrapper::GetShape(
        uint32_t dimensionCount,
        uint32_t* dimensions) const noexcept
    {
        ORT_TRY
        {
          VerifyNotClosed();

          std::fill(dimensions, dimensions + dimensionCount, 0u);

          uint32_t count = static_cast<uint32_t>(m_impl->dims().size());
          ML_CHECK_BOOL(dimensionCount == count);

          for (uint32_t i = 0; i < dimensionCount; ++i)
          {
            dimensions[i] = static_cast<uint32_t>(m_impl->dims()[i]);
          }

          return S_OK;
        }
        ORT_CATCH_RETURN
    }

    MLOperatorTensorDataType STDMETHODCALLTYPE OnnxTensorWrapper::GetTensorDataType() const noexcept
    {
        ORT_TRY
        {
          VerifyNotClosed();
          return ToMLTensorDataType(static_cast<onnx::TensorProto_DataType>(m_impl->data_type()));
        }
        ORT_CATCH_GENERIC
        {
          return MLOperatorTensorDataType::Undefined;
        }
    }

    bool STDMETHODCALLTYPE OnnxTensorWrapper::IsCpuData() const noexcept
    {
        return true;
    }

    bool STDMETHODCALLTYPE OnnxTensorWrapper::IsDataInterface() const noexcept
    {
        return false;
    }

    void* STDMETHODCALLTYPE OnnxTensorWrapper::GetData() noexcept
    {
        if (IsClosed())
        {
            return nullptr;
        }

        return m_dataPtr;
    }

    void STDMETHODCALLTYPE OnnxTensorWrapper::GetDataInterface(IUnknown** dataInterface) noexcept
    {
        *dataInterface = nullptr;
    }

    TensorWrapper::TensorWrapper(onnxruntime::Tensor* impl, bool isDataInterface, IWinmlExecutionProvider* provider, bool isInternalOperator)
    :   m_impl(impl),
        m_winmlExecutionProvider(provider),
        m_internalOperator(isInternalOperator),
        m_isDataInterface(isDataInterface)
    {
        if (impl)
        {
            if (isDataInterface)
            {
                // We assume that all data handles derive from IUnknown as their first base.
                m_dataInterface = static_cast<IUnknown*>(m_impl->MutableDataRaw());

                if (m_dataInterface)
                {
                    if (m_winmlExecutionProvider)
                    {
                        // The resource may require conversion to the layout expected according to the kernel options.
                        // This will return either the original object or a shadow copy which uses a different layout.
                        // This pattern assumes that Lotus is not re-using tensor allocations, so each output is
                        // a fresh allocation which will not trigger a conversion in the provider.
                        m_winmlExecutionProvider->GetShadowCopyIfRequired(m_internalOperator, m_dataInterface.Get(), m_dataInterfaceOrShadowCopy.GetAddressOf());

                        // Get the actual object to be returned from the ABI, which varies for internal and external
                        // kernels (i.e. ID3D12Resource, versus something that tracks the layout).
                        TranslateAllocationDataToAbi(
                            m_winmlExecutionProvider.Get(),
                            m_internalOperator,
                            m_impl->Location(),
                            m_dataInterfaceOrShadowCopy ? m_dataInterfaceOrShadowCopy.Get() : m_dataInterface.Get(),
                            m_abiDataInterface.GetAddressOf());
                    }
                    else
                    {
                        m_abiDataInterface = m_dataInterface;
                    }
                }
            }
            else
            {
                m_tensorData = m_impl->MutableDataRaw();
            }
        }
    }

    uint32_t STDMETHODCALLTYPE TensorWrapper::GetDimensionCount() const noexcept
    {
        if (IsClosed())
        {
            return 0;
        }

        return gsl::narrow_cast<uint32_t>(m_impl->Shape().NumDimensions());
    }

    HRESULT STDMETHODCALLTYPE TensorWrapper::GetShape(
        uint32_t dimensionCount,
        uint32_t* dimensions) const noexcept
    {
        ORT_TRY
        {
          VerifyNotClosed();

          std::fill(dimensions, dimensions + dimensionCount, 0u);

          uint32_t count = static_cast<uint32_t>(m_impl->Shape().NumDimensions());
          ML_CHECK_BOOL(dimensionCount == count);

          for (size_t i = 0; i < dimensionCount; ++i)
          {
            dimensions[i] = static_cast<uint32_t>(m_impl->Shape()[i]);
          }

          return S_OK;
        }
        ORT_CATCH_RETURN
    }

    MLOperatorTensorDataType STDMETHODCALLTYPE TensorWrapper::GetTensorDataType() const noexcept
    {
        ORT_TRY
        {
          VerifyNotClosed();
          return ToMLTensorDataType(m_impl->DataType());
        }
        ORT_CATCH_GENERIC
        {
          return MLOperatorTensorDataType::Undefined;
        }
    }

    bool STDMETHODCALLTYPE TensorWrapper::IsCpuData() const noexcept
    {
        if (IsClosed())
        {
            return true;
        }

        // tells caller whether this tensor is in CPU memory
        return !strcmp(m_impl->Location().name, onnxruntime::CPU) || m_impl->Location().mem_type == ::OrtMemType::OrtMemTypeCPUOutput || m_impl->Location().mem_type == ::OrtMemType::OrtMemTypeCPUInput;
    }

    bool STDMETHODCALLTYPE TensorWrapper::IsDataInterface() const noexcept
    {
        if (IsClosed())
        {
            return false;
        }

        return m_isDataInterface;
    }

    void* STDMETHODCALLTYPE TensorWrapper::GetData() noexcept
    {
        if (IsClosed())
        {
            return nullptr;
        }

        return m_isDataInterface ? nullptr : m_tensorData;
    }

    void STDMETHODCALLTYPE TensorWrapper::GetDataInterface(IUnknown** dataInterface) noexcept
    {
        if (!m_isDataInterface)
        {
            VerifyNotClosed();
            *dataInterface = nullptr;
        }
        else
        {
            m_abiDataInterface.CopyTo(dataInterface);
        }
    }

    void OpKernelContextWrapper::TransitionResourcesForOperatorIfRequired(bool isBeforeOp)
    {
        if (m_winmlProvider->TransitionsRequiredForOperator(m_internalOperator))
        {
            uint32_t totalInputTensorCount = 0;
            for (auto inputTensor : m_inputTensors)
            {
                totalInputTensorCount += static_cast<uint32_t>(inputTensor.size());
            }
            std::vector<IUnknown*> resourcesToTransition;
            resourcesToTransition.reserve(totalInputTensorCount + m_outputTensors.size() + m_temporaryAllocations.size());

            for (uint32_t i = 0; i < m_inputTensors.size(); ++i)
            {
                for (uint32_t j = 0; j < m_inputTensors[i].size(); ++j)
                {
                    ComPtr<IMLOperatorTensor> tensor;
                    if (m_inputTensors[i].size() == 1)
                    {
                        ORT_THROW_IF_FAILED(GetInputTensor(i, tensor.GetAddressOf()));
                    }
                    else
                    {
                        ORT_THROW_IF_FAILED(GetSequenceInputTensor(i, j, tensor.GetAddressOf()));
                    }

                    if (tensor)
                    {
                        ComPtr<IUnknown> resource;
                        tensor->GetDataInterface(resource.GetAddressOf());
                        if (resource)
                        {
                            resourcesToTransition.push_back(resource.Get());
                        }
                    }
                }
            }

            for (uint32_t i = 0; i < m_outputTensors.size(); ++i)
            {
                ComPtr<IMLOperatorTensor> tensor;
                ORT_THROW_IF_FAILED(GetOutputTensor(i, tensor.GetAddressOf()));

                ComPtr<IUnknown> resource;
                tensor->GetDataInterface(resource.GetAddressOf());
                if (resource)
                {
                    resourcesToTransition.push_back(resource.Get());
                }
            }

            for (auto& tempAlloc : m_temporaryAbiAllocations)
            {
                resourcesToTransition.push_back(tempAlloc.Get());
            }

            m_winmlProvider->TransitionResourcesForOperator(
                isBeforeOp,
                gsl::narrow_cast<uint32_t>(resourcesToTransition.size()),
                resourcesToTransition.data());
        }
    }

    OpKernelContextWrapper::OpKernelContextWrapper(
        onnxruntime::OpKernelContext* context,
        const onnxruntime::IExecutionProvider* provider,
        bool isInternalOperator,
        const EdgeShapes* outputShapes
        )
    :   m_impl(context), m_outputShapes(outputShapes), m_provider(provider), m_internalOperator(isInternalOperator)
    {
        // Pre-size tensor arrays.    Member methods return pointers to these which
        // are stored in these arrays, which would become stale if the vectors reallocate
        // their internal storage.
        m_inputTensors.resize(context->InputCount(), std::vector<ComPtr<TensorWrapper>>(1));
        m_outputTensors.resize(context->OutputCount(), std::vector<ComPtr<TensorWrapper>>(1));

        const void* executionHandle = m_provider->GetExecutionHandle();
        if (executionHandle)
        {
            // We assume the execution object inherits IUnknown as its first base
            m_providerExecutionObject = const_cast<IUnknown*>(static_cast<const IUnknown*>(executionHandle));
            m_providerExecutionObject.As(&m_winmlProvider);

            // Query the actual object to return through the ABI, based on options registered
            // with the kernel
            m_abiExecutionObject = m_providerExecutionObject;
            if (m_winmlProvider)
            {
                m_winmlProvider->GetABIExecutionInterfaceAndInvalidateState(isInternalOperator, m_abiExecutionObject.ReleaseAndGetAddressOf());
            }

            TransitionResourcesForOperatorIfRequired(true);
        }
    }

    OpKernelContextWrapper::~OpKernelContextWrapper()
    {
        ClearTempAllocations();
    }

    void OpKernelContextWrapper::ClearTempAllocations()
    {
        if (m_winmlProvider)
        {
            m_temporaryAllocations.clear();
            m_temporaryAbiAllocations.clear();
        }
    }

    void OpKernelContextWrapper::Close()
    {
        if (m_winmlProvider && m_winmlProvider->TransitionsRequiredForOperator(m_internalOperator))
        {
            TransitionResourcesForOperatorIfRequired(false);
        }

        for (auto& tensors : m_inputTensors)
        {
            for (auto& tensor : tensors)
            {
                if (tensor)
                {
                    tensor->Close();
                }
            }
        }

        for (auto& tensors : m_outputTensors)
        {
            for (auto& tensor : tensors)
            {
                if (tensor)
                {
                    tensor->Close();
                }
            }
        }

        ClearTempAllocations();

        Closable::Close();
    }

    bool STDMETHODCALLTYPE OpKernelContextWrapper::IsSequenceInputTensor(uint32_t inputIndex) const noexcept
    {
        assert(inputIndex < gsl::narrow_cast<uint32_t>(m_impl->InputCount()));
        return m_impl->InputType(inputIndex)->IsTensorSequenceType();
    }

    HRESULT STDMETHODCALLTYPE OpKernelContextWrapper::GetInputTensor(uint32_t inputIndex, IMLOperatorTensor** tensor) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();
            *tensor = nullptr;

            ML_CHECK_BOOL(inputIndex < m_inputTensors.size());

            auto opKernelContextWrapper = const_cast<OpKernelContextWrapper*>(this);
            if (m_inputTensors[inputIndex][0] == nullptr)
            {
                auto inputTensor = m_impl->Input<onnxruntime::Tensor>(gsl::narrow_cast<int>(inputIndex));
                if (inputTensor != nullptr)
                {
                    ComPtr<TensorWrapper> tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
                        const_cast<onnxruntime::Tensor*>(inputTensor),
                        IsAllocationInterface(inputTensor->Location()),
                        m_winmlProvider.Get(),
                        m_internalOperator);

                    opKernelContextWrapper->m_inputTensors[inputIndex][0] = tensorWrapper;
                }
            }

            if (opKernelContextWrapper->m_inputTensors[inputIndex][0] != nullptr)
            {
                opKernelContextWrapper->m_inputTensors[inputIndex][0].CopyTo(tensor);
            }
            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT STDMETHODCALLTYPE OpKernelContextWrapper::GetSequenceInputTensor(uint32_t inputIndex, uint32_t sequenceIndex, IMLOperatorTensor** tensor) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();
            *tensor = nullptr;

            auto opKernelContextWrapper = const_cast<OpKernelContextWrapper*>(this);

            ML_CHECK_BOOL(inputIndex < m_inputTensors.size());
            if (sequenceIndex >= m_inputTensors[inputIndex].size())
            {
                opKernelContextWrapper->m_inputTensors[inputIndex].resize(static_cast<size_t>(sequenceIndex)+1);
            }

            if (m_inputTensors[inputIndex][sequenceIndex] == nullptr)
            {
                auto inputTensorSeq = m_impl->Input<onnxruntime::TensorSeq>(gsl::narrow_cast<int>(inputIndex));
                ML_CHECK_BOOL(inputTensorSeq != nullptr);

                auto elemTensor = const_cast<onnxruntime::Tensor*>(&inputTensorSeq->Get(sequenceIndex));
                if (elemTensor != nullptr)
                {
                    ComPtr<TensorWrapper> tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
                        elemTensor,
                        IsAllocationInterface(elemTensor->Location()),
                        m_winmlProvider.Get(),
                        m_internalOperator);

                    opKernelContextWrapper->m_inputTensors[inputIndex][sequenceIndex] = tensorWrapper;
                }
            }

            if (opKernelContextWrapper->m_inputTensors[inputIndex][sequenceIndex] != nullptr)
            {
                opKernelContextWrapper->m_inputTensors[inputIndex][sequenceIndex].CopyTo(tensor);
            }
            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT STDMETHODCALLTYPE OpKernelContextWrapper::PrepareSequenceOutput(
        uint32_t outputIndex,
        MLOperatorTensorDataType dataType) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            auto opKernelContextWrapper = const_cast<OpKernelContextWrapper*>(this);

            ML_CHECK_BOOL(outputIndex < m_outputTensors.size());
            auto outputTensorSeq = m_impl->Output<onnxruntime::TensorSeq>(gsl::narrow_cast<int>(outputIndex));
            ML_CHECK_BOOL(outputTensorSeq != nullptr);

            auto mlDataType = ToMLDataType(MLOperatorEdgeType::Primitive, dataType);
            outputTensorSeq->SetType(mlDataType);

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT STDMETHODCALLTYPE OpKernelContextWrapper::GetSequenceOutputTensor(
        uint32_t outputIndex,
        uint32_t sequenceIndex,
        MLOperatorTensorDataType dataType,
        uint32_t dimensions,
        const uint32_t* dimensionSizes,
        bool gpuOutput,
        IMLOperatorTensor** tensor) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();
            *tensor = nullptr;

            auto opKernelContextWrapper = const_cast<OpKernelContextWrapper*>(this);

            ML_CHECK_BOOL(outputIndex < m_outputTensors.size());
            if (sequenceIndex >= m_outputTensors[outputIndex].size())
            {
                opKernelContextWrapper->m_outputTensors[outputIndex].resize(sequenceIndex+1);
            }

            // Verify that the provided shape matches the shape determined using the kernel's shape inference function.
            if (m_outputTensors[outputIndex][sequenceIndex] == nullptr)
            {
                auto outputTensorSeq = m_impl->Output<onnxruntime::TensorSeq>(gsl::narrow_cast<int>(outputIndex));
                ML_CHECK_BOOL(outputTensorSeq != nullptr);

                auto mlDataType = ToMLDataType(MLOperatorEdgeType::Primitive, dataType);

                if (outputTensorSeq->Size() == 0)
                {
                    outputTensorSeq->SetType(mlDataType);
                }

                onnxruntime::AllocatorPtr alloc;
                if (gpuOutput)
                {
                    auto status = m_impl->GetTempSpaceAllocator(&alloc);
                    ORT_THROW_HR_IF(E_INVALIDARG, !status.IsOK());
                }
                else
                {
                    auto status = m_impl->GetTempSpaceCPUAllocator(&alloc);
                    ORT_THROW_HR_IF(E_INVALIDARG, !status.IsOK());
                }

                std::vector<int64_t> shapeDims(dimensions);
                for (uint32_t i = 0; i < dimensions; ++i)
                {
                    shapeDims[i] = dimensionSizes[i];
                }

                auto target_tensor = onnxruntime::Tensor(mlDataType, onnxruntime::TensorShape(shapeDims), alloc);
                outputTensorSeq->Add(std::move(target_tensor));

                auto elemTensor = const_cast<onnxruntime::Tensor*>(&outputTensorSeq->Get(sequenceIndex));
                if (elemTensor != nullptr)
                {
                    ComPtr<TensorWrapper> tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
                        elemTensor,
                        IsAllocationInterface(elemTensor->Location()),
                        m_winmlProvider.Get(),
                        m_internalOperator);

                    opKernelContextWrapper->m_outputTensors[outputIndex][sequenceIndex] = tensorWrapper;
                }
            }

            if (opKernelContextWrapper->m_outputTensors[outputIndex][sequenceIndex] != nullptr)
            {
                opKernelContextWrapper->m_outputTensors[outputIndex][sequenceIndex].CopyTo(tensor);
            }

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT STDMETHODCALLTYPE OpKernelContextWrapper::GetSequenceInputInfo(uint32_t inputIndex, uint32_t* inputCount, MLOperatorTensorDataType* dataType) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            ML_CHECK_BOOL(inputIndex < m_inputTensors.size());

            assert(m_impl->InputType(gsl::narrow_cast<int>(inputIndex))->IsTensorSequenceType());
            ML_CHECK_BOOL(m_impl->InputType(gsl::narrow_cast<int>(inputIndex))->IsTensorSequenceType());
            auto inputTensorSeq = m_impl->Input<onnxruntime::TensorSeq>(gsl::narrow_cast<int>(inputIndex));
            ML_CHECK_BOOL(inputTensorSeq != nullptr);
            *inputCount = static_cast<uint32_t>(inputTensorSeq->Size());
            *dataType = ToMLTensorDataType(inputTensorSeq->DataType());
            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT STDMETHODCALLTYPE OpKernelContextWrapper::GetOutputTensor(uint32_t outputIndex, IMLOperatorTensor** tensor) noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            *tensor = nullptr;

            ML_CHECK_BOOL(outputIndex < m_outputTensors.size());

            // GetOutputTensor must be called unless a kernel provides shape inferencing,
            // in which case m_outputShapes will be valid here.
            if (!m_outputShapes)
            {
                return E_FAIL;
                //return MLStatus::SHAPE_INFERENCE_NOT_REGISTERED;
            }

            uint32_t dimensionCount = gsl::narrow_cast<uint32_t>(m_outputShapes->GetShape(outputIndex).size());
            return GetOutputTensor(outputIndex, dimensionCount, m_outputShapes->GetShape(outputIndex).data(), tensor);
        }
        ORT_CATCH_RETURN
    }

    HRESULT STDMETHODCALLTYPE OpKernelContextWrapper::GetOutputTensor(uint32_t outputIndex, uint32_t dimensions, const uint32_t* dimensionSizes, IMLOperatorTensor** tensor) noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();
            *tensor = nullptr;

            ML_CHECK_BOOL(outputIndex < m_outputTensors.size());

            // Verify that the provided shape matches the shape determined using the kernel's shape inference function.
            if (m_outputTensors[outputIndex][0] == nullptr)
            {
                if (m_outputShapes)
                {
                    if ((m_outputShapes->GetShape(outputIndex).size() != dimensions ||
                        memcmp(dimensionSizes, m_outputShapes->GetShape(outputIndex).data(), dimensions * sizeof(*dimensionSizes))))
                    {
                        return E_INVALIDARG;
                    }
                }
                std::vector<int64_t> convertedSizes(dimensions);
                for (size_t i = 0; i < dimensions; ++i)
                {
                    convertedSizes[i] = dimensionSizes[i];
                }

                onnxruntime::TensorShape shape(convertedSizes.data(), dimensions);
                auto outputTensor = m_impl->Output(outputIndex, shape);
                if (outputTensor)
                {
                    ComPtr<TensorWrapper> tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
                        const_cast<onnxruntime::Tensor*>(outputTensor),
                        IsAllocationInterface(outputTensor->Location()),
                        m_winmlProvider.Get(),
                        m_internalOperator);

                    const_cast<OpKernelContextWrapper*>(this)->m_outputTensors[outputIndex][0] = tensorWrapper;
                }
            }

            m_outputTensors[outputIndex][0].CopyTo(tensor);

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT STDMETHODCALLTYPE OpKernelContextWrapper::AllocateTemporaryData(size_t size, IUnknown** abiAllocation) const noexcept
    {
        ORT_TRY
        {
            uint64_t allocId;
            return AllocateTemporaryData(size, abiAllocation, &allocId);
        }
        ORT_CATCH_RETURN
    }

    HRESULT STDMETHODCALLTYPE OpKernelContextWrapper::AllocateTemporaryData(size_t size, IUnknown** abiAllocation, uint64_t* allocId) const
    {
        ORT_TRY
        {
            VerifyNotClosed();

            *abiAllocation = nullptr;
            onnxruntime::AllocatorPtr alloc;
            THROW_IF_NOT_OK(m_impl->GetTempSpaceAllocator(&alloc));

            if (!IsAllocationInterface(alloc->Info()))
            {
                return E_FAIL;
            }

            ComPtr<IUnknown> allocation;
            allocation.Attach(static_cast<IUnknown*>(alloc->Alloc(size)));

            *allocId = m_winmlProvider->TryGetPooledAllocationId(allocation.Get(), 0);

            TranslateAllocationDataToAbi(m_winmlProvider.Get(), m_internalOperator, alloc->Info(), allocation.Get(), abiAllocation);

            if (m_winmlProvider->TransitionsRequiredForOperator(m_internalOperator))
            {
                m_winmlProvider->TransitionResourcesForOperator(true, 1, abiAllocation);
            }

            // Ensure the allocation is freed and transitioned when the context destructs
            m_temporaryAllocations.push_back(allocation);
            m_temporaryAbiAllocations.push_back(*abiAllocation);

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    void STDMETHODCALLTYPE OpKernelContextWrapper::GetExecutionInterface(IUnknown** executionInterface) const noexcept
    {
        m_abiExecutionObject.CopyTo(executionInterface);
    }

    std::vector<IMLOperatorTensor*> OpKernelContextWrapper::GetInputTensors()
    {
        std::vector<IMLOperatorTensor*> ret;
        ret.reserve(m_inputTensors.size());

        for (int i = 0; i < m_impl->InputCount(); ++i)
        {
            ComPtr<IMLOperatorTensor> tensor;
            ORT_THROW_IF_FAILED(GetInputTensor(i, tensor.GetAddressOf()));
            ret.push_back(m_inputTensors[i][0].Get());
        }

        return ret;
    }

    std::vector<IMLOperatorTensor*> OpKernelContextWrapper::GetOutputTensors(const EdgeShapes& outputShapes)
    {
        std::vector<IMLOperatorTensor*> ret;
        ret.reserve(m_outputTensors.size());

        ORT_THROW_HR_IF(E_INVALIDARG, static_cast<size_t>(m_impl->OutputCount()) != outputShapes.EdgeCount());

        for (int i = 0; i < m_impl->OutputCount(); ++i)
        {
            ComPtr<IMLOperatorTensor> tensor;
            ORT_THROW_IF_FAILED(GetOutputTensor(
                i,
                static_cast<uint32_t>(outputShapes.GetShape(i).size()),
                outputShapes.GetShape(i).data(),
                tensor.GetAddressOf()));

            ret.push_back(m_outputTensors[i][0].Get());
        }

        return ret;
    }

    AbiOpKernel::AbiOpKernel(
        IMLOperatorKernelFactory* operatorFactory,
        const onnxruntime::OpKernelInfo& kerneInfo,
        bool requiresInputShapesAtCreation,
        bool requiresOutputShapesAtCreation,
        bool isInternalOperator,
        gsl::span<const uint32_t> requiredConstantCpuInputs,
        IMLOperatorShapeInferrer* shapeInferrer,
        const AttributeMap* defaultAttributes)
    :   OpKernel(kerneInfo),
        m_requiresInputShapesAtCreation(requiresInputShapesAtCreation),
        m_requiresOutputShapesAtCreation(requiresOutputShapesAtCreation),
        m_shapeInferrer(shapeInferrer),
        m_internalOperator(isInternalOperator),
        m_defaultAttributes(defaultAttributes)
    {
        assert(requiresInputShapesAtCreation || !requiresOutputShapesAtCreation);

        m_requiredConstantCpuInputs.assign(requiredConstantCpuInputs.begin(), requiredConstantCpuInputs.end());

        const void* executionHandle = kerneInfo.GetExecutionProvider()->GetExecutionHandle();
        if (executionHandle)
        {
            // We assume the execution object inherits IUnknown as its first base
            ComPtr<IUnknown> providerExecutionObject = const_cast<IUnknown*>(static_cast<const IUnknown*>(executionHandle));
            m_abiExecutionObject = providerExecutionObject;

            // Get the WinML-specific execution provider interface from the execution object.
            providerExecutionObject.As(&m_winmlProvider);

            if (m_winmlProvider)
            {
                // Get the particular object to return to a isInternalOperator based on the registration of that kernel.
                m_winmlProvider->GetABIExecutionInterfaceAndInvalidateState(isInternalOperator, m_abiExecutionObject.ReleaseAndGetAddressOf());
            }
        }

        bool requiredConstantCpuInputsAvailable = true;
        for (uint32_t index : requiredConstantCpuInputs)
        {
            const onnxruntime::Tensor* tensor = nullptr;
            if (!kerneInfo.TryGetConstantInput(index, &tensor) || !tensor)
            {
                requiredConstantCpuInputsAvailable = false;
                break;
            }
        }

        // If input sizes are either available or not required at creation, no need to delay kernel creation.
        if (requiredConstantCpuInputsAvailable && (!m_requiresInputShapesAtCreation || InputTensorShapesDefined()))
        {
            auto winmlProviderCapture = m_winmlProvider;
            auto internalOpCapture = m_internalOperator;

            MLOperatorTensorGetter constantInputGetter = [kerneInfo, winmlProviderCapture, internalOpCapture](uint32_t index)
            {
                Microsoft::WRL::ComPtr<IMLOperatorTensor> tensorWrapper = nullptr;
                const onnxruntime::Tensor* tensor = nullptr;
                if (kerneInfo.TryGetConstantInput(index, &tensor))
                {
                    tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
                        const_cast<onnxruntime::Tensor*>(tensor),
                        IsAllocationInterface(tensor->Location()),
                        winmlProviderCapture.Get(),
                        internalOpCapture);
                }

                return tensorWrapper;
            };

            // If the output size is not dynamic, infer it using the kernel.  Then if the output size was predicted
            // by schema, verify consistency.  The result of inference is stored in m_inferredOutputShapes.
            if (m_requiresOutputShapesAtCreation)
            {
                // Use the same list of required inputs for the shape inferrer and the kernel.
                InferAndVerifyOutputSizes(m_requiredConstantCpuInputs, constantInputGetter, nullptr, m_inferredOutputShapes);
            }

            // Create the kernel while allowing input shape and output shape queries according to options
            ComPtr<OpKernelInfoWrapper> kernelInfoWrapper = wil::MakeOrThrow<OpKernelInfoWrapper>(
                &kerneInfo,
                m_abiExecutionObject.Get(),
                nullptr,
                m_requiresOutputShapesAtCreation ? &m_inferredOutputShapes : nullptr,
                m_requiresInputShapesAtCreation,
                m_requiresOutputShapesAtCreation,
                isInternalOperator,
                m_defaultAttributes,
                m_requiredConstantCpuInputs,
                constantInputGetter,
                nullptr /*const onnxruntime::OpKernelContext* m_kernelContext*/);

            ORT_THROW_IF_FAILED(operatorFactory->CreateKernel(kernelInfoWrapper.Get(), m_kernel.GetAddressOf()));
            kernelInfoWrapper->Close();

            // Ensure that scheduled work, if any, is completed before freeing the kernel if the execution
            // provider requires this.
            if (m_winmlProvider)
            {
                m_winmlProvider->QueueReference(m_kernel.Get());
            }
        }
        else
        {
            m_operatorFactory = operatorFactory;
        }
    }

    onnxruntime::Status AbiOpKernel::Compute(onnxruntime::OpKernelContext* context) const
    {
        auto winmlProviderCapture = m_winmlProvider;
        auto internalOpCapture = m_internalOperator;

        MLOperatorTensorGetter constantInputGetter = [context, winmlProviderCapture, internalOpCapture](uint32_t index)
        {
            auto inputType = context->InputType(gsl::narrow_cast<int>(index));

            if (inputType != nullptr)
            {
                if (inputType->IsTensorType())
                {
                    Microsoft::WRL::ComPtr<IMLOperatorTensor> tensorWrapper = nullptr;

                    const auto* tensor = context->Input<onnxruntime::Tensor>(gsl::narrow_cast<int>(index));
                    if (tensor != nullptr)
                    {
                        tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
                            const_cast<onnxruntime::Tensor*>(tensor),
                            IsAllocationInterface(tensor->Location()),
                            winmlProviderCapture.Get(),
                            internalOpCapture);
                    }

                    return tensorWrapper;
                }
                else if (inputType->IsTensorSequenceType())
                {
                    std::vector<Microsoft::WRL::ComPtr<IMLOperatorTensor>> tensorWrappers;

                    const auto* tensorSequence = context->Input<onnxruntime::TensorSeq>(gsl::narrow_cast<int>(index));
                    if (tensorSequence != nullptr)
                    {
                        tensorWrappers.reserve(tensorSequence->Size());

                        for (uint32_t sequenceIndex = 0; sequenceIndex < tensorSequence->Size(); ++sequenceIndex)
                        {
                            auto& tensor = tensorSequence->Get(sequenceIndex);
                            auto tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
                                const_cast<onnxruntime::Tensor*>(&tensor),
                                IsAllocationInterface(tensor.Location()),
                                winmlProviderCapture.Get(),
                                internalOpCapture);
                        }
                    }
                }
                else
                {
                    assert(false);
                    ORT_THROW_HR(E_INVALIDARG);
                }
            }

            return Microsoft::WRL::ComPtr<IMLOperatorTensor>();
        };

        auto inferShapesAndCreateKernel = [&, context](const EdgeShapes& inputShapes, EdgeShapes& outputShapes) -> ComPtr<IMLOperatorKernel> {
            // If the output size is not dynamic, infer it using the kernel. The result of inference is stored in m_inferredOutputShapes.
            if (m_requiresOutputShapesAtCreation)
            {
                // Use the same list of required inputs for the shape inferrer and the kernel.
                InferAndVerifyOutputSizes(m_requiredConstantCpuInputs, constantInputGetter, &inputShapes, outputShapes);
            }

            // Create the kernel while allowing input shape and output shape queries according to options
            ComPtr<OpKernelInfoWrapper> kernelInfoWrapper = wil::MakeOrThrow<OpKernelInfoWrapper>(
                &Info(),
                m_abiExecutionObject.Get(),
                &inputShapes,
                m_requiresInputShapesAtCreation ? &outputShapes : nullptr,
                m_requiresInputShapesAtCreation,
                m_requiresOutputShapesAtCreation,
                m_internalOperator,
                m_defaultAttributes,
                m_requiredConstantCpuInputs,
                constantInputGetter,
                context /*const onnxruntime::OpKernelContext* m_kernelContext*/);

            ComPtr<IMLOperatorKernel> ret;
            ORT_THROW_IF_FAILED(m_operatorFactory->CreateKernel(kernelInfoWrapper.Get(), ret.GetAddressOf()));
            kernelInfoWrapper->Close();

            return ret;
        };

        // The kernel creation may have been delayed because input shapes were required but not inferred by schema.
        if (RequiresLazyInitialization())
        {
            std::lock_guard<std::mutex> lock(m_mutex);

            if (RequiresLazyInitialization())
            {
                m_inputShapesOfKernelInference = GetInputShapes(context);

                m_constantInputTensorContentsOfKernel.resize(context->InputCount());
                for (uint32_t index : m_requiredConstantCpuInputs)
                {
                    if (index >= m_constantInputTensorContentsOfKernel.size())
                    {
                        continue;
                    }

                    auto constantInput = constantInputGetter(index);

                    std::visit([this, context, index](auto&& arg) {
                        FillConstantInputs(arg, context, index);
                    }, constantInput);
                }

                m_kernel = inferShapesAndCreateKernel(m_inputShapesOfKernelInference, m_inferredOutputShapes);
                SetLazyInitialized();
            }
        }
        else if (m_inputShapesOfKernelInference.EdgeCount() > 0)
        {
            EdgeShapes local_input_shapes = GetInputShapes(context);

            bool requiredCpuInputsChanged = false;
            for (uint32_t index : m_requiredConstantCpuInputs)
            {
                if (index >= m_constantInputTensorContentsOfKernel.size())
                {
                    continue;
                }

                auto constantInput = constantInputGetter(index);
                requiredCpuInputsChanged = std::visit([this, index](auto&& arg){
                    return RequiredCpuInputChanged(arg, index);
                }, constantInput);

                if (requiredCpuInputsChanged)
                {
                    break;
                }
            }

            // In the edge case that the input size is changing across invocations and the kernel requires
            // its input size at construction, use a local instance of the kernel.
            if (local_input_shapes != m_inputShapesOfKernelInference || requiredCpuInputsChanged)
            {
                EdgeShapes localInferredOutputShapes;
                ComPtr<IMLOperatorKernel> localKernel = inferShapesAndCreateKernel(local_input_shapes, localInferredOutputShapes);

                ComPtr<OpKernelContextWrapper> kernelContextWrapper = wil::MakeOrThrow<OpKernelContextWrapper>(
                    context,
                    Info().GetExecutionProvider(),
                    m_internalOperator,
                    m_requiresOutputShapesAtCreation ? &localInferredOutputShapes : nullptr);

                ORT_THROW_IF_FAILED(localKernel->Compute(kernelContextWrapper.Get()));
                kernelContextWrapper->Close();

                // Ensure that scheduled work, if any, is completed before freeing the kernel if the execution
                // provider requires this.
                if (m_winmlProvider)
                {
                    m_winmlProvider->QueueReference(localKernel.Get());
                }
                return onnxruntime::Status();
            }
        }

        ComPtr<OpKernelContextWrapper> kernelContextWrapper = wil::MakeOrThrow<OpKernelContextWrapper>(
            context,
            Info().GetExecutionProvider(),
            m_internalOperator,
            m_requiresOutputShapesAtCreation ? &m_inferredOutputShapes : nullptr);

        ORT_THROW_IF_FAILED(m_kernel->Compute(kernelContextWrapper.Get()));
        kernelContextWrapper->Close();

        // Ensure that scheduled work, if any, is completed before freeing the kernel if the execution
        // provider requires this.
        if (m_winmlProvider)
        {
            m_winmlProvider->QueueReference(m_kernel.Get());
        }

        return onnxruntime::Status();
    }

    bool AbiOpKernel::RequiredCpuInputChanged(const ComPtr<IMLOperatorTensor>& constantTensor, uint32_t index) const
    {
        assert(std::holds_alternative<TensorContent>(m_constantInputTensorContentsOfKernel[index]));

        auto lastValue = std::get<TensorContent>(m_constantInputTensorContentsOfKernel[index]);
        MLOperatorTensor currentValue(constantTensor.Get());

        if (lastValue.isValid != (currentValue.GetInterface() != nullptr))
        {
            return false;
        }

        if (lastValue.isValid)
        {
            if (lastValue.shape != currentValue.GetShape() ||
                lastValue.type != currentValue.GetTensorDataType() ||
                currentValue.GetUnalignedTensorByteSize() != lastValue.data.size() ||
                (memcmp(lastValue.data.data(), currentValue.GetByteData(), lastValue.data.size()) != 0))
            {
                return true;
            }
        }

        return false;
    }

    bool AbiOpKernel::RequiredCpuInputChanged(const std::vector<ComPtr<IMLOperatorTensor>>& constantTensorSequence, uint32_t index) const
    {
        assert(std::holds_alternative<std::vector<TensorContent>>(m_constantInputTensorContentsOfKernel[index]));
        auto lastValues = std::get<std::vector<TensorContent>>(m_constantInputTensorContentsOfKernel[index]);

        for (uint32_t sequenceIndex = 0; sequenceIndex < constantTensorSequence.size(); ++sequenceIndex)
        {
            const auto& lastValue = lastValues[sequenceIndex];
            MLOperatorTensor currentValue(constantTensorSequence[sequenceIndex].Get());

            if (lastValue.isValid != (currentValue.GetInterface() != nullptr))
            {
                return false;
            }

            if (lastValue.isValid)
            {
                if (lastValue.shape != currentValue.GetShape() ||
                    lastValue.type != currentValue.GetTensorDataType() ||
                    currentValue.GetUnalignedTensorByteSize() != lastValue.data.size() ||
                    (memcmp(lastValue.data.data(), currentValue.GetByteData(), lastValue.data.size()) != 0))
                {
                    return true;
                }
            }
        }

        return false;
    }

    void AbiOpKernel::FillConstantInputs(const ComPtr<IMLOperatorTensor>& constantTensor, onnxruntime::OpKernelContext* context, uint32_t index) const
    {
        // Skip optional constant tensors.
        if (constantTensor != nullptr)
        {
            MLOperatorTensor tensor = MLOperatorTensor(constantTensor.Get());

            if (index >= static_cast<uint32_t>(context->InputCount()))
            {
                return;
            }

            TensorContent tensorContent{};
            tensorContent.isValid = (tensor.GetInterface() != nullptr);

            if (tensor.GetInterface() != nullptr)
            {
                tensorContent.shape = tensor.GetShape();
                tensorContent.type = tensor.GetTensorDataType();
                tensorContent.data.resize(tensor.GetUnalignedTensorByteSize());
            }

            tensorContent.data.assign(
                reinterpret_cast<const std::byte*>(tensor.GetByteData()),
                reinterpret_cast<const std::byte*>(tensor.GetByteData()) + tensor.GetUnalignedTensorByteSize());

            m_constantInputTensorContentsOfKernel[index] = std::move(tensorContent);
        }
    }

    void AbiOpKernel::FillConstantInputs(const std::vector<ComPtr<IMLOperatorTensor>>& constantTensorSequence, onnxruntime::OpKernelContext* context, uint32_t index) const
    {
        std::vector<TensorContent> tensorContent(constantTensorSequence.size());

        for (uint32_t i = 0; i < constantTensorSequence.size(); ++i)
        {
            const ComPtr<IMLOperatorTensor>& constantTensor = constantTensorSequence[i];

            // Skip optional constant tensors.
            if (constantTensor == nullptr)
            {
                continue;
            }

            MLOperatorTensor tensor = MLOperatorTensor(constantTensor.Get());

            if (index >= static_cast<uint32_t>(context->InputCount()))
            {
                continue;
            }
            tensorContent[i].isValid = (tensor.GetInterface() != nullptr);

            if (tensor.GetInterface() != nullptr)
            {
                tensorContent[i].shape = tensor.GetShape();
                tensorContent[i].type = tensor.GetTensorDataType();
                tensorContent[i].data.resize(tensor.GetUnalignedTensorByteSize());
            }
            tensorContent[i].data.assign(
                reinterpret_cast<const std::byte*>(tensor.GetByteData()),
                reinterpret_cast<const std::byte*>(tensor.GetByteData()) + tensor.GetUnalignedTensorByteSize());
        }

        m_constantInputTensorContentsOfKernel[index] = std::move(tensorContent);
    }

    bool AbiOpKernel::InputTensorShapesDefined() const
    {
        onnxruntime::ProtoHelperNodeContext protoContext(Node());
        onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext> info(&protoContext);
        return InputTensorShapesDefinedOnNode(info);
    }

    EdgeShapes AbiOpKernel::GetInputShapes(onnxruntime::OpKernelContext* context) const
    {
        EdgeShapes ret(context->InputCount());

        for (size_t i = 0; i < ret.EdgeCount(); ++i)
        {
            // The input type is null if unused
            auto inputType = context->InputType(static_cast<int>(i));
            if (inputType != nullptr && inputType->IsTensorType())
            {
                if (context->InputType(gsl::narrow_cast<int>(i))->IsTensorSequenceType())
                {
                    auto inputTensorSeq = context->Input<onnxruntime::TensorSeq>(gsl::narrow_cast<int>(i));
                    for (uint32_t sequenceIndex = 0; sequenceIndex < inputTensorSeq->Size(); ++sequenceIndex)
                    {
                        const auto& tensor = inputTensorSeq->Get(sequenceIndex);
                        ret.GetMutableShape(i).resize(tensor.Shape().GetDims().size());
                        for (size_t j = 0; j < ret.GetMutableShape(i).size(); ++j)
                        {
                            ret.GetMutableShape(i)[j] = gsl::narrow_cast<uint32_t>(tensor.Shape().GetDims()[j]);
                        }
                    }
                }
                else if (context->InputType(gsl::narrow_cast<int>(i))->IsTensorType())
                {
                    const onnxruntime::Tensor* tensor = context->Input<onnxruntime::Tensor>(gsl::narrow_cast<int>(i));
                    if (tensor)
                    {
                        ret.GetMutableShape(i).resize(tensor->Shape().GetDims().size());
                        for (size_t j = 0; j < ret.GetMutableShape(i).size(); ++j)
                        {
                            ret.GetMutableShape(i)[j] = gsl::narrow_cast<uint32_t>(tensor->Shape().GetDims()[j]);
                        }
                    }
                }
                else
                {
                    ORT_THROW_HR(E_INVALIDARG);
                }
            }
        }

        return ret;
    }

    void AbiOpKernel::InferAndVerifyOutputSizes(
        gsl::span<const uint32_t> requiredConstantCpuInputs,
        MLOperatorTensorGetter& constantInputGetter,
        const EdgeShapes* inputShapes,
        EdgeShapes& outputShapes) const
    {
        // call the non member function (below)
        Windows::AI::MachineLearning::Adapter::InferAndVerifyOutputSizes(
            Node(),
            m_defaultAttributes,
            m_shapeInferrer.Get(),
            requiredConstantCpuInputs,
            constantInputGetter,
            inputShapes,
            outputShapes
        );
    }

    void InferAndVerifyOutputSizes(
        const onnxruntime::Node& node,
        const AttributeMap* defaultAttributes,
        IMLOperatorShapeInferrer* shapeInferrer,
        gsl::span<const uint32_t> requiredConstantCpuInputs,
        MLOperatorTensorGetter& constantInputGetter,
        const EdgeShapes* inputShapes,
        EdgeShapes& outputShapes)
    {
        onnxruntime::ProtoHelperNodeContext protoContext(node);
        onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext> info(&protoContext);

        ComPtr<MLKernelInferenceContext> inferenceContext = wil::MakeOrThrow<MLKernelInferenceContext>(&info, inputShapes, outputShapes, defaultAttributes, requiredConstantCpuInputs, constantInputGetter);

        outputShapes.Reset(info.GetOutputCount());

        ORT_THROW_IF_FAILED(shapeInferrer->InferOutputShapes(inferenceContext.Get()));
        inferenceContext->Close();

        for (size_t outputIndex = 0; outputIndex < outputShapes.EdgeCount(); ++outputIndex)
        {
            const onnx::TypeProto* outputProto = info.GetOutputType(outputIndex);

            // Skip this output if it is not valid.
            if (outputProto == nullptr)
            {
                continue;
            }

            if (outputProto->value_case() != onnx::TypeProto::kTensorType)
            {
                assert(outputShapes.GetShape(outputIndex).empty());
                ML_CHECK_BOOL(outputShapes.GetShape(outputIndex).empty());
                continue;
            }

            const auto& tensorType = outputProto->tensor_type();

            if (tensorType.has_shape())
            {
                const auto& shape = tensorType.shape();
                assert(static_cast<size_t>(shape.dim_size()) == outputShapes.GetShape(outputIndex).size());
                ML_CHECK_BOOL(static_cast<size_t>(shape.dim_size()) == outputShapes.GetShape(outputIndex).size());

                for (uint32_t output_dim = 0; output_dim < outputShapes.GetShape(outputIndex).size(); ++output_dim)
                {
                    if (shape.dim(output_dim).has_dim_value())
                    {
                        int64_t expected_size = shape.dim(output_dim).dim_value();
                        int64_t actual_size = outputShapes.GetShape(outputIndex)[output_dim];
                        assert(expected_size == actual_size);
                        ML_CHECK_BOOL(expected_size == actual_size);
                    }
                }
            }
        }
    }

    ComPtr<MLSchemaInferenceContext> MLSchemaInferenceContext::Create(onnxruntime::OpNodeProtoHelper<onnx::InferenceContext>* info,
        onnx::InferenceContext* ctx,
        gsl::span<const uint32_t> requiredConstantCpuInputs)
    {
        MLOperatorTensorGetter mlOperatorTensorGetter = MLOperatorTensorGetter(
            [ctx](uint32_t index)
            {
                // An empty path is used as external weights are not currently supported in this case
                Microsoft::WRL::ComPtr<IMLOperatorTensor> tensorWrapper = wil::MakeOrThrow<OnnxTensorWrapper>(
                    const_cast<onnx::TensorProto*>(ctx->getInputData(index)), onnxruntime::Path());
                return tensorWrapper;
            }
        );

        return wil::MakeOrThrow<MLSchemaInferenceContext>(info, ctx, requiredConstantCpuInputs, mlOperatorTensorGetter);
    }

    MLSchemaInferenceContext::MLSchemaInferenceContext(
        onnxruntime::OpNodeProtoHelper<onnx::InferenceContext>* info,
        onnx::InferenceContext* ctx,
        gsl::span<const uint32_t> requiredConstantCpuInputs,
        MLOperatorTensorGetter& mLOperatorTensorGetter
        )
    :   OpNodeInfoWrapper(info, nullptr, nullptr, requiredConstantCpuInputs, mLOperatorTensorGetter),
        m_context(ctx)
    {
    }

    HRESULT STDMETHODCALLTYPE MLSchemaInferenceContext::SetOutputTensorShape(
        uint32_t outputIndex,
        uint32_t dimensionCount,
        const uint32_t* dimensions) noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            MLOperatorEdgeDescription edgeDesc;
            ORT_THROW_IF_FAILED(GetOutputEdgeDescription(outputIndex, &edgeDesc));

            // In the process of calling mutable_tensor_type, the type may switch from undefined to tensor.
            // This is done here in case the dimension count is zero (scalar)
            m_context->getOutputType(outputIndex)->mutable_tensor_type();

            for (uint32_t i = 0; i < dimensionCount; ++i)
            {
                auto dim = m_context->getOutputType(outputIndex)->mutable_tensor_type()->mutable_shape()->add_dim();
                dim->set_dim_value(dimensions[i]);
            }

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT STDMETHODCALLTYPE MLSchemaInferenceContext::SetOutputEdgeDescription(
        uint32_t outputIndex,
        const MLOperatorEdgeDescription* edgeDesc) const noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            std::string typeStr = ToTypeString(*edgeDesc);
            m_context->getOutputType(outputIndex)->CopyFrom(onnx::Utils::DataTypeUtils::ToTypeProto(&typeStr));
            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    HRESULT STDMETHODCALLTYPE MLKernelInferenceContext::SetOutputTensorShape(
        uint32_t outputIndex,
        uint32_t dimensionCount,
        const uint32_t* dimensions) noexcept
    {
        ORT_TRY
        {
            VerifyNotClosed();

            if (outputIndex >= m_inferredOutputShapes.EdgeCount())
            {
                return E_INVALIDARG;
            }

            m_inferredOutputShapes.GetMutableShape(outputIndex).assign(dimensions, dimensions + dimensionCount);

            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    ComPtr<MLSupportQueryContext> MLSupportQueryContext::Create(onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>* info,
        const AttributeMap* defaultAttributes)
    {
        MLOperatorTensorGetter mLOperatorTensorGetter = MLOperatorTensorGetter();
        return wil::MakeOrThrow<MLSupportQueryContext>(info, defaultAttributes, mLOperatorTensorGetter);
    }

    MLSupportQueryContext::MLSupportQueryContext(
        onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>* info,
        const AttributeMap* defaultAttributes,
        MLOperatorTensorGetter& mLOperatorTensorGetter
        )
    :   OpNodeInfoWrapper(info, nullptr, defaultAttributes, gsl::span<const uint32_t>(), mLOperatorTensorGetter)
    {
    }

    bool TryGetStaticShapeIfTensor(
        const onnx::TypeProto* inputProto,
        std::vector<uint32_t>& shapeDims)
    {
        // Skip this input if it is not valid.
        if (inputProto == nullptr)
        {
            return true;
        }

        if (inputProto->value_case() != onnx::TypeProto::kTensorType)
        {
            return true;
        }

        const auto& tensorType = inputProto->tensor_type();

        if (!tensorType.has_shape())
        {
            return false;
        }

        const auto& shape = tensorType.shape();
        shapeDims.resize(shape.dim_size());

        for (uint32_t dimIndex = 0; dimIndex < static_cast<uint32_t>(shape.dim_size()); ++dimIndex)
        {
            if (!shape.dim(dimIndex).has_dim_value())
            {
                return false;
            }

            shapeDims[dimIndex] = gsl::narrow<uint32_t>(shape.dim(dimIndex).dim_value());
        }

        return true;
    }

    bool TryGetStaticInputShapes(const onnxruntime::Node& node, EdgeShapes& inputShapes)
    {
        onnxruntime::ProtoHelperNodeContext protoContext(node);
        onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext> info(&protoContext);

        inputShapes.Reset(info.GetInputCount());

        for (size_t inputIndex = 0; inputIndex < inputShapes.EdgeCount(); ++inputIndex)
        {
            const onnx::TypeProto* inputProto = info.GetInputType(inputIndex);
            if (!TryGetStaticShapeIfTensor(inputProto, inputShapes.GetMutableShape(inputIndex)))
            {
                return false;
            }
        }

        return true;
    }

    bool TryGetStaticOutputShapes(const onnxruntime::Node& node, EdgeShapes& outputShapes)
    {
        onnxruntime::ProtoHelperNodeContext protoContext(node);
        onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext> info(&protoContext);

        outputShapes.Reset(info.GetOutputCount());

        for (size_t outputIndex = 0; outputIndex < outputShapes.EdgeCount(); ++outputIndex)
        {
            const onnx::TypeProto* outputProto = info.GetOutputType(outputIndex);
            if (!TryGetStaticShapeIfTensor(outputProto, outputShapes.GetMutableShape(outputIndex)))
            {
                return false;
            }
        }

        return true;
    }

    bool ContainsEmptyDimensions(const EdgeShapes& shapes, gsl::span<const uint32_t> ignoredShapeIndices)
    {
        for (size_t i = 0; i < shapes.EdgeCount(); i++)
        {
            const std::vector<uint32_t>& shape = shapes.GetShape(i);

            if (std::find(shape.begin(), shape.end(), 0u) != shape.end() &&
                std::find(ignoredShapeIndices.begin(), ignoredShapeIndices.end(), i) == ignoredShapeIndices.end())
            {
                return true;
            }
        }

        return false;
    }

    std::tuple<std::unique_ptr<std::byte[]>, size_t> UnpackTensor(
        const onnx::TensorProto& initializer,
        const onnxruntime::Path& modelPath)
    {
        std::unique_ptr<std::byte[]> unpackedTensor;
        size_t tensorByteSize = 0;

#define CASE_PROTO(X, Y, Z)                                                                        \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X: {                           \
    size_t elementCount = initializer.##Z();                                                       \
    tensorByteSize = elementCount * sizeof(Y);                                                     \
    unpackedTensor.reset(new std::byte[tensorByteSize]);                                           \
    ORT_THROW_HR_IF(E_FAIL, !onnxruntime::utils::UnpackTensor(                                     \
                             initializer,                                                          \
                             modelPath,                                                            \
                             reinterpret_cast<Y*>(unpackedTensor.get()), elementCount)             \
                             .IsOK());                                                             \
    break;                                                                                         \
  }
        switch (initializer.data_type())
        {
        CASE_PROTO(FLOAT, float, float_data_size);
        CASE_PROTO(DOUBLE, double, double_data_size);
        CASE_PROTO(BOOL, bool, int32_data_size);
        CASE_PROTO(INT8, int8_t, int32_data_size);
        CASE_PROTO(INT16, int16_t, int32_data_size);
        CASE_PROTO(INT32, int32_t, int32_data_size);
        CASE_PROTO(INT64, int64_t, int64_data_size);
        CASE_PROTO(UINT8, uint8_t, int32_data_size);
        CASE_PROTO(UINT16, uint16_t, int32_data_size);
        CASE_PROTO(UINT32, uint32_t, uint64_data_size);
        CASE_PROTO(UINT64, uint64_t, uint64_data_size);
        CASE_PROTO(FLOAT16, onnxruntime::MLFloat16, int32_data_size);
        default: ORT_THROW_HR(E_INVALIDARG);
        }

        return std::make_tuple(std::move(unpackedTensor), tensorByteSize);
    }
}  // namespace winrt::Windows::AI::MachineLearning::implementation
