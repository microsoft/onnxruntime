// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/dml/DmlExecutionProvider/inc/MLOperatorAuthor.h"
#include "MLOperatorAuthorPrivate.h"

#ifdef ORT_NO_EXCEPTIONS
#define ML_CHECK_BOOL(x) ORT_THROW_HR_IF(E_INVALIDARG, !(x))
#else
#define ML_CHECK_BOOL(x) THROW_HR_IF(E_INVALIDARG, !(x))
#endif

namespace onnxruntime
{
    struct MLFloat16;
}

using MLFloat16 = onnxruntime::MLFloat16;

//
// Traits for numeric attribute types
//
template <typename T>
struct MLTypeTraits
{
};

template <>
struct MLTypeTraits<float>
{
    static const MLOperatorAttributeType AttributeType = MLOperatorAttributeType::Float;
    static const MLOperatorAttributeType AttributeVectorType = MLOperatorAttributeType::FloatArray;
    static const MLOperatorTensorDataType TensorType = MLOperatorTensorDataType::Float;
};

template <>
struct MLTypeTraits<int32_t>
{
    static const MLOperatorTensorDataType TensorType = MLOperatorTensorDataType::Int32;
};

template <>
struct MLTypeTraits<uint8_t>
{
    static const MLOperatorTensorDataType TensorType = MLOperatorTensorDataType::UInt8;
};

template <>
struct MLTypeTraits<int8_t>
{
    static const MLOperatorTensorDataType TensorType = MLOperatorTensorDataType::Int8;
};

template <>
struct MLTypeTraits<uint16_t>
{
    static const MLOperatorTensorDataType TensorType = MLOperatorTensorDataType::UInt16;
};

template <>
struct MLTypeTraits<int16_t>
{
    static const MLOperatorTensorDataType TensorType = MLOperatorTensorDataType::Int16;
};

template <>
struct MLTypeTraits<int64_t>
{
    static const MLOperatorTensorDataType TensorType = MLOperatorTensorDataType::Int64;
    static const MLOperatorAttributeType AttributeType = MLOperatorAttributeType::Int;
    static const MLOperatorAttributeType AttributeVectorType = MLOperatorAttributeType::IntArray;
};

template <>
struct MLTypeTraits<bool>
{
    static const MLOperatorTensorDataType TensorType = MLOperatorTensorDataType::Bool;
};

template <>
struct MLTypeTraits<double>
{
    static const MLOperatorTensorDataType TensorType = MLOperatorTensorDataType::Double;
};

template <>
struct MLTypeTraits<uint32_t>
{
    static const MLOperatorTensorDataType TensorType = MLOperatorTensorDataType::UInt32;
};

template <>
struct MLTypeTraits<uint64_t>
{
    static const MLOperatorTensorDataType TensorType = MLOperatorTensorDataType::UInt64;
};

template <>
struct MLTypeTraits<onnxruntime::MLFloat16>
{
  static const MLOperatorTensorDataType TensorType = MLOperatorTensorDataType::Float16;
};

inline uint32_t ComputeElementCountFromDimensions(gsl::span<const uint32_t> dimensions)
{
    return std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<uint32_t>());
}

#pragma warning(push)
#pragma warning(disable:4702)
inline size_t GetByteSizeFromMlDataType(MLOperatorTensorDataType tensorDataType)
{
    switch (tensorDataType)
    {
    case MLOperatorTensorDataType::Float: return 4;
    case MLOperatorTensorDataType::UInt8: return 1;
    case MLOperatorTensorDataType::Int8: return 1;
    case MLOperatorTensorDataType::UInt16: return 2;
    case MLOperatorTensorDataType::Int16: return 2;
    case MLOperatorTensorDataType::Int32: return 4;
    case MLOperatorTensorDataType::Int64: return 8;
    case MLOperatorTensorDataType::String: ORT_THROW_HR(E_INVALIDARG);
    case MLOperatorTensorDataType::Bool: return 1;
    case MLOperatorTensorDataType::Float16: return 2;
    case MLOperatorTensorDataType::Double: return 8;
    case MLOperatorTensorDataType::UInt32: return 4;
    case MLOperatorTensorDataType::UInt64: return 8;
    case MLOperatorTensorDataType::Complex64: return 8;
    case MLOperatorTensorDataType::Complex128: return 16;
    case MLOperatorTensorDataType::Undefined:
    default:
        ORT_THROW_HR(E_INVALIDARG);
        return 0;
    };
}
#pragma warning(pop)

using MLConstStringParam = const char*;
class MLOperatorKernelContext;

//
// Wrappers for ABI objects consumed by kernels.
// These wrappers provide typesafe methods which use STL types and convert
// return values to exceptions.
//

class MLOperatorTensorShapeDescription
{
 public:
    MLOperatorTensorShapeDescription(IMLOperatorTensorShapeDescription* impl) : m_impl(impl) {}

    uint32_t GetInputTensorDimensionCount(uint32_t inputIndex) const
    {
        uint32_t ret;
        ORT_THROW_IF_FAILED(m_impl->GetInputTensorDimensionCount(inputIndex, &ret));
        return ret;
    }

    std::vector<uint32_t> GetInputTensorShape(uint32_t inputIndex) const
    {
        std::vector<uint32_t> ret;
        uint32_t dimensionCount = GetInputTensorDimensionCount(inputIndex);
        ret.resize(dimensionCount);

        ORT_THROW_IF_FAILED(m_impl->GetInputTensorShape(inputIndex, dimensionCount, ret.data()));
        return ret;
    }

    uint32_t GetSequenceInputCount(uint32_t inputIndex) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorTensorShapeDescriptionPrivate> private_impl;
        m_impl.As(&private_impl);
        uint32_t inputCount = 0;
        MLOperatorTensorDataType dataType;
        ORT_THROW_IF_FAILED(private_impl->GetSequenceInputInfo(inputIndex, &inputCount, &dataType));
        return inputCount;
    }

    MLOperatorTensorDataType GetSequenceInputDataType(uint32_t inputIndex) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorTensorShapeDescriptionPrivate> private_impl;
        m_impl.As(&private_impl);
        uint32_t inputCount = 0;
        MLOperatorTensorDataType dataType;
        ORT_THROW_IF_FAILED(private_impl->GetSequenceInputInfo(inputIndex, &inputCount, &dataType));
        return dataType;
    }

    uint32_t GetSequenceInputTensorDimensionCount(uint32_t inputIndex, uint32_t sequenceIndex) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorTensorShapeDescriptionPrivate> private_impl;
        m_impl.As(&private_impl);

        uint32_t ret;
        ORT_THROW_IF_FAILED(private_impl->GetSequenceInputTensorDimensionCount(inputIndex, sequenceIndex, &ret));
        return ret;
    }

    std::vector<uint32_t> GetSequenceInputTensorShape(uint32_t inputIndex, uint32_t sequenceIndex) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorTensorShapeDescriptionPrivate> private_impl;
        m_impl.As(&private_impl);

        std::vector<uint32_t> ret;
        uint32_t dimensionCount = GetSequenceInputTensorDimensionCount(inputIndex, sequenceIndex);
        ret.resize(dimensionCount);

        ORT_THROW_IF_FAILED(private_impl->GetSequenceInputTensorShape(inputIndex, sequenceIndex, dimensionCount, ret.data()));
        return ret;
    }

    bool HasOutputShapeDescription() const noexcept
    {
        return m_impl->HasOutputShapeDescription();
    }

    uint32_t GetOutputTensorDimensionCount(uint32_t outputIndex) const
    {
        uint32_t ret;
        ORT_THROW_IF_FAILED(m_impl->GetOutputTensorDimensionCount(outputIndex, &ret));
        return ret;
    }

    std::vector<uint32_t> GetOutputTensorShape(uint32_t outputIndex) const
    {
        std::vector<uint32_t> ret;
        uint32_t dimensionCount = GetOutputTensorDimensionCount(outputIndex);
        ret.resize(dimensionCount);

        ORT_THROW_IF_FAILED(m_impl->GetOutputTensorShape(outputIndex, dimensionCount, ret.data()));
        return ret;
    }

    Microsoft::WRL::ComPtr<IMLOperatorTensorShapeDescription> GetInterface() const noexcept { return m_impl; }

 protected:
    Microsoft::WRL::ComPtr<IMLOperatorTensorShapeDescription> m_impl;
};

class MLOperatorAttributes
{
 public:
    MLOperatorAttributes(IMLOperatorAttributes* impl) : m_impl(impl)
    {
    }

    // For cases of interop where the caller needs to pass the unwrapped class across a boundary.
    Microsoft::WRL::ComPtr<IMLOperatorAttributes> GetInterface() const noexcept
    {
        return m_impl;
    }

    uint32_t GetAttributeElementCount(
        _In_z_ MLConstStringParam name,
        MLOperatorAttributeType type) const
    {
        uint32_t elementCount;
        ORT_THROW_IF_FAILED(m_impl->GetAttributeElementCount(name, type, &elementCount));
        return elementCount;
    }

    bool HasAttribute(_In_z_ MLConstStringParam name, MLOperatorAttributeType type) const noexcept
    {
        return GetAttributeElementCount(name, type) > 0;
    }

    //
    // Templatized methods to query numeric attributes using MLTypeTraits
    //
    template <typename T>
    T GetAttribute(_In_z_ MLConstStringParam name) const
    {
        T value;

        ORT_THROW_IF_FAILED(m_impl->GetAttribute(
                name,
                MLTypeTraits<T>::AttributeType,
                1,
                sizeof(T),
                &value));

        return value;
    }

    template <typename T>
    std::vector<T> GetAttributeVector(_In_z_ MLConstStringParam name) const
    {
        uint32_t count = GetAttributeElementCount(name, MLTypeTraits<T>::AttributeVectorType);
        std::vector<T> values(count);

        ORT_THROW_IF_FAILED(m_impl->GetAttribute(
                name,
                MLTypeTraits<T>::AttributeVectorType,
                count,
                sizeof(T),
                values.data()));

        return values;
    }

    std::string GetAttribute(_In_z_ MLConstStringParam name) const
    {
        return GetAttributeElement(name, 0);
    }

    std::vector<std::string> GetAttributeVector(_In_z_ MLConstStringParam name) const
    {
        uint32_t count = GetAttributeElementCount(name, MLOperatorAttributeType::StringArray);
        std::vector<std::string> values;
        values.resize(count);

        for (uint32_t i = 0; i < count; ++i)
        {
            values[i] = GetAttributeElement(name, i);
        }

        return values;
    }

    std::string GetAttributeElement(_In_z_ MLConstStringParam name, uint32_t elementIndex) const
    {
        uint32_t length = 0;
        ORT_THROW_IF_FAILED(m_impl->GetStringAttributeElementLength(name, elementIndex, &length));

        // Construct a string by copying a character array.    The copy can be removed with C++17
        // using the non-const std::basic_string::data method.
        std::vector<char> temp(length);
        ORT_THROW_IF_FAILED(m_impl->GetStringAttributeElement(name, elementIndex, length, temp.data()));
        std::string value(temp.data());
        return value;
    }

    std::vector<int32_t> GetOptionalAttributeVectorInt32(MLConstStringParam attributeName) const
    {
        std::vector<int32_t> vector32Bit;
        if (HasAttribute(attributeName, MLOperatorAttributeType::IntArray))
        {
            auto vector64Bit = GetAttributeVector<int64_t>(attributeName);
            vector32Bit.resize(vector64Bit.size());
            std::transform(vector64Bit.begin(), vector64Bit.end(), /*out*/vector32Bit.begin(), [](auto i)
                                    {return gsl::narrow_cast<int32_t>(std::clamp<int64_t>(i, INT32_MIN, INT32_MAX)); });
        }
        return vector32Bit;
    }

    std::vector<std::string> GetOptionalStringAttributeVector(MLConstStringParam attributeName) const
    {
        return HasAttribute(attributeName, MLOperatorAttributeType::StringArray)
            ?  GetAttributeVector(attributeName)
            :  std::vector<std::string>{}; // Empty vector if attribute absent.
    }

    // Not implemented
    template <typename T> T GetOptionalAttribute(MLConstStringParam attributeName, T defaultValue) const;

    template <>
    int32_t GetOptionalAttribute<int32_t>(MLConstStringParam attributeName, int32_t defaultValue) const
    {
        return HasAttribute(attributeName, MLOperatorAttributeType::Int)
            ?  gsl::narrow_cast<int32_t>(GetAttribute<int64_t>(attributeName))
            :  defaultValue;
    }

    template <>
    uint32_t GetOptionalAttribute<uint32_t>(MLConstStringParam attributeName, uint32_t defaultValue) const
    {
        return HasAttribute(attributeName, MLOperatorAttributeType::Int)
            ?  gsl::narrow_cast<uint32_t>(GetAttribute<int64_t>(attributeName))
            :  defaultValue;
    }

    template <>
    int64_t GetOptionalAttribute<int64_t>(MLConstStringParam attributeName, int64_t defaultValue) const
    {
        return HasAttribute(attributeName, MLOperatorAttributeType::Int)
            ?  GetAttribute<int64_t>(attributeName)
            :  defaultValue;
    }

    template <>
    float GetOptionalAttribute<float>(MLConstStringParam attributeName, float defaultValue) const
    {
        return HasAttribute(attributeName, MLOperatorAttributeType::Float)
            ? GetAttribute<float>(attributeName)
            : defaultValue;
    }

    template <>
    std::vector<float> GetOptionalAttribute<std::vector<float>>(MLConstStringParam attributeName, std::vector<float> defaultValue) const
    {
        return HasAttribute(attributeName, MLOperatorAttributeType::FloatArray)
            ?  GetAttributeVector<float>(attributeName)
            :  defaultValue;
    }

    template <>
    bool GetOptionalAttribute<bool>(MLConstStringParam attributeName, bool defaultValue) const
    {
        return HasAttribute(attributeName, MLOperatorAttributeType::Int)
            ?  gsl::narrow_cast<bool>(GetAttribute<int64_t>(attributeName))
            :  defaultValue;
    }

    template <>
    std::string GetOptionalAttribute<std::string>(MLConstStringParam attributeName, std::string defaultValue) const
    {
        return HasAttribute(attributeName, MLOperatorAttributeType::String)
            ?  GetAttribute(attributeName)
            :  defaultValue;
    }

 private:
    Microsoft::WRL::ComPtr<IMLOperatorAttributes> m_impl;
};

class MLOperatorTensor
{
public:
    MLOperatorTensor(IMLOperatorTensor* impl) : m_impl(impl) {}

    // For cases of interop where the caller needs to pass the unwrapped class across a boundary.
    Microsoft::WRL::ComPtr<IMLOperatorTensor> GetInterface() const noexcept
    {
        return m_impl;
    }

    // Need default constructor for usage in STL containers.
    MLOperatorTensor() = default;
    MLOperatorTensor(const MLOperatorTensor&) = default;
    MLOperatorTensor(MLOperatorTensor&&) = default;
    MLOperatorTensor& operator=(const MLOperatorTensor&) = default;

    uint32_t GetDimensionCount() const
    {
        return m_impl->GetDimensionCount();
    }

    const std::vector<uint32_t>& GetShape() const
    {
        if (m_dimensionsCache.empty())
        {
            uint32_t dimensionCount = GetDimensionCount();
            const_cast<MLOperatorTensor*>(this)->m_dimensionsCache.resize(dimensionCount);
            ORT_THROW_IF_FAILED(m_impl->GetShape(dimensionCount, const_cast<MLOperatorTensor*>(this)->m_dimensionsCache.data()));
        }

        return m_dimensionsCache;
    }

    uint32_t GetTotalElementCount() const
    {
        return ComputeElementCountFromDimensions(GetShape());
    }

    size_t GetUnalignedTensorByteSize() const
    {
        return GetTotalElementCount() * GetByteSizeFromMlDataType(GetTensorDataType());
    }

    MLOperatorTensorDataType GetTensorDataType() const noexcept
    {
        return m_impl->GetTensorDataType();
    }

    bool IsCpuData() const noexcept
    {
        return m_impl->IsCpuData();
    }

    bool IsDataInterface() const noexcept
    {
        return m_impl->IsDataInterface();
    }

    // Return data as an explicitly typed array, verifying the requested type
    // is the actual data type in the tensor.
    template <typename T>
    T* GetData()
    {
        ML_CHECK_BOOL(GetTensorDataType() == MLTypeTraits<T>::TensorType);
        ML_CHECK_BOOL(!IsDataInterface());

        return static_cast<T*>(m_impl->GetData());
    }

    template <typename T>
    const T* GetData() const
    {
        ML_CHECK_BOOL(GetTensorDataType() == MLTypeTraits<T>::TensorType);
        ML_CHECK_BOOL(!IsDataInterface());

        return static_cast<const T*>(m_impl->GetData());
    }

    // Return as raw bytes, regardless of underlying type, which is useful when
    // needing to agnostically copy memory.
    const void* GetByteData() const
    {
        ML_CHECK_BOOL(!IsDataInterface());

        return m_impl->GetData();
    }

    void* GetByteData()
    {
        ML_CHECK_BOOL(!IsDataInterface());

        return m_impl->GetData();
    }

    Microsoft::WRL::ComPtr<IUnknown> GetDataInterface()
    {
        ML_CHECK_BOOL(IsDataInterface());
        Microsoft::WRL::ComPtr<IUnknown> ret;
        m_impl->GetDataInterface(&ret);
        return ret;
    }

 private:
    Microsoft::WRL::ComPtr<IMLOperatorTensor> m_impl;
    std::vector<uint32_t> m_dimensionsCache;
};

class MLOperatorKernelCreationContext : public MLOperatorAttributes
{
public:
    MLOperatorKernelCreationContext(IMLOperatorKernelCreationContext* impl) : MLOperatorAttributes(impl), m_impl(impl)
    {
        m_impl.As(&m_implPrivate);
        m_impl.As(&m_nodeWrapperImpl);
    }

    // For cases of interop where the caller needs to pass the unwrapped class across a boundary.
    Microsoft::WRL::ComPtr<IMLOperatorKernelCreationContext> GetInterface() const noexcept
    {
        return m_impl;
    }

    IMLOperatorKernelCreationContextNodeWrapperPrivate* GetNodeWrapperInterface() const noexcept
    {
        return m_nodeWrapperImpl.Get();
    }

    Microsoft::WRL::ComPtr<IUnknown> GetExecutionInterface() const noexcept
    {
        Microsoft::WRL::ComPtr<IUnknown> ret;
        m_impl->GetExecutionInterface(&ret);
        return ret;
    }

    uint32_t GetInputCount() const noexcept
    {
        return m_impl->GetInputCount();
    }

    uint32_t GetOutputCount() const noexcept
    {
        return m_impl->GetOutputCount();
    }

    bool IsInputValid(uint32_t index) const {
        return m_impl->IsInputValid(index);
    }

    bool IsOutputValid(uint32_t index) const {
        return m_impl->IsOutputValid(index);
    }

    MLOperatorEdgeDescription GetInputEdgeDescription(uint32_t inputIndex) const
    {
        MLOperatorEdgeDescription ret;
        ORT_THROW_IF_FAILED(m_impl->GetInputEdgeDescription(inputIndex, &ret));

        return ret;
    }

    MLOperatorEdgeDescription GetOutputEdgeDescription(uint32_t outputIndex) const
    {
        MLOperatorEdgeDescription ret = {};
        ORT_THROW_IF_FAILED(m_impl->GetOutputEdgeDescription(outputIndex, &ret));

        return ret;
    }

    bool HasTensorShapeDescription() const noexcept
    {
        return m_impl->HasTensorShapeDescription();
    }

    MLOperatorTensorShapeDescription GetTensorShapeDescription() const
    {
        Microsoft::WRL::ComPtr<IMLOperatorTensorShapeDescription> ret;
        ORT_THROW_IF_FAILED(m_impl->GetTensorShapeDescription(&ret));
        return MLOperatorTensorShapeDescription(ret.Get());
    }

    MLOperatorTensor GetConstantInputTensor(uint32_t inputIndex) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorTensor> tensor;
        ORT_THROW_IF_FAILED(m_implPrivate->GetConstantInputTensor(inputIndex, &tensor));
        return MLOperatorTensor(tensor.Get());
    }

    uint32_t GetInputTensorDimensionCount(uint32_t inputIndex) const
    {
        auto shapeDesc = GetTensorShapeDescription();
        return shapeDesc.GetInputTensorDimensionCount(inputIndex);
    }

    std::vector<uint32_t> GetInputTensorShape(uint32_t inputIndex) const
    {
        auto shapeDesc = GetTensorShapeDescription();
        return shapeDesc.GetInputTensorShape(inputIndex);
    }

    uint32_t GetSequenceInputCount(uint32_t inputIndex) const
    {
        auto shapeDesc = GetTensorShapeDescription();
        return shapeDesc.GetSequenceInputCount(inputIndex);
    }

    MLOperatorTensorDataType GetSequenceInputDataType(uint32_t inputIndex) const
    {
        auto shapeDesc = GetTensorShapeDescription();
        return shapeDesc.GetSequenceInputDataType(inputIndex);
    }

    uint32_t GetSequenceInputTensorDimensionCount(uint32_t inputIndex, uint32_t sequenceIndex) const
    {
        auto shapeDesc = GetTensorShapeDescription();
        return shapeDesc.GetSequenceInputTensorDimensionCount(inputIndex, sequenceIndex);
    }

    std::vector<uint32_t> GetSequenceInputTensorShape(uint32_t inputIndex, uint32_t sequenceIndex) const
    {
        auto shapeDesc = GetTensorShapeDescription();
        return shapeDesc.GetSequenceInputTensorShape(inputIndex, sequenceIndex);
    }

 private:
    Microsoft::WRL::ComPtr<IMLOperatorKernelCreationContext> m_impl;
    Microsoft::WRL::ComPtr<IMLOperatorKernelCreationContextPrivate> m_implPrivate;
    Microsoft::WRL::ComPtr<IMLOperatorKernelCreationContextNodeWrapperPrivate> m_nodeWrapperImpl;
};

class MLShapeInferenceContext : public MLOperatorAttributes
{
public:
    MLShapeInferenceContext(IMLOperatorShapeInferenceContext* impl) : MLOperatorAttributes(impl)
    {
        ORT_THROW_IF_FAILED(impl->QueryInterface(m_impl.GetAddressOf()));
    }

    // For cases of interop where the caller needs to pass the unwrapped class across a boundary.
    Microsoft::WRL::ComPtr<IMLOperatorShapeInferenceContextPrivate> GetInterface() const noexcept
    {
        return m_impl;
    }

    uint32_t GetInputCount() const noexcept
    {
        return m_impl->GetInputCount();
    }

    uint32_t GetOutputCount() const noexcept
    {
        return m_impl->GetOutputCount();
    }

    // Returns true if an input to the operator is valid.
    // This returns false for optional omitted inputs and invalid indices.
    bool IsInputValid(uint32_t inputIndex) const noexcept
    {
        return m_impl->IsInputValid(inputIndex);
    }

    // Returns true if an output to the operator is valid.
    // This returns false for optional omitted inputs and invalid indices.
    bool IsOutputValid(uint32_t inputIndex) const noexcept
    {
        return m_impl->IsOutputValid(inputIndex);
    }

    MLOperatorEdgeDescription GetInputEdgeDescription(uint32_t inputIndex) const
    {
        MLOperatorEdgeDescription ret;
        ORT_THROW_IF_FAILED(m_impl->GetInputEdgeDescription(inputIndex, &ret));

        return ret;
    }

    uint32_t GetInputTensorDimensionCount(uint32_t inputIndex) const
    {
        uint32_t ret;
        ORT_THROW_IF_FAILED(m_impl->GetInputTensorDimensionCount(inputIndex, &ret));
        return ret;
    }

    std::vector<uint32_t> GetInputTensorShape(uint32_t inputIndex) const
    {
        std::vector<uint32_t> ret;
        uint32_t dimensionCount = GetInputTensorDimensionCount(inputIndex);
        ret.resize(dimensionCount);

        ORT_THROW_IF_FAILED(m_impl->GetInputTensorShape(inputIndex, dimensionCount, ret.data()));
        return ret;
    }

    uint32_t GetSequenceInputCount(uint32_t inputIndex) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorShapeInferenceContextPrivate> private_impl;
        m_impl.As(&private_impl);
        uint32_t inputCount = 0;
        MLOperatorTensorDataType dataType;
        ORT_THROW_IF_FAILED(private_impl->GetSequenceInputInfo(inputIndex, &inputCount, &dataType));
        return inputCount;
    }

    MLOperatorTensorDataType GetSequenceInputDataType(uint32_t inputIndex) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorShapeInferenceContextPrivate> private_impl;
        m_impl.As(&private_impl);
        uint32_t inputCount = 0;
        MLOperatorTensorDataType dataType;
        ORT_THROW_IF_FAILED(private_impl->GetSequenceInputInfo(inputIndex, &inputCount, &dataType));
        return dataType;
    }

    uint32_t GetSequenceInputTensorDimensionCount(uint32_t inputIndex, uint32_t sequenceIndex) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorShapeInferenceContextPrivate> private_impl;
        m_impl.As(&private_impl);

        uint32_t ret;
        ORT_THROW_IF_FAILED(private_impl->GetSequenceInputTensorDimensionCount(inputIndex, sequenceIndex, &ret));
        return ret;
    }

    std::vector<uint32_t> GetSequenceInputTensorShape(uint32_t inputIndex, uint32_t sequenceIndex) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorShapeInferenceContextPrivate> private_impl;
        m_impl.As(&private_impl);

        std::vector<uint32_t> ret;
        uint32_t dimensionCount = GetSequenceInputTensorDimensionCount(inputIndex, sequenceIndex);
        ret.resize(dimensionCount);

        ORT_THROW_IF_FAILED(private_impl->GetSequenceInputTensorShape(inputIndex, sequenceIndex, dimensionCount, ret.data()));
        return ret;
    }


    void SetOutputTensorShape(uint32_t outputIndex, const std::vector<uint32_t>& outputDimensions)
    {
        ORT_THROW_IF_FAILED(m_impl->SetOutputTensorShape(outputIndex, static_cast<uint32_t>(outputDimensions.size()), outputDimensions.data()));
    }

    MLOperatorTensor GetConstantInputTensor(uint32_t inputIndex) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorTensor> tensor;
        ORT_THROW_IF_FAILED(m_impl->GetConstantInputTensor(inputIndex, &tensor));
        return MLOperatorTensor(tensor.Get());
    }

 private:
    Microsoft::WRL::ComPtr<IMLOperatorShapeInferenceContextPrivate> m_impl;
};

class MLOperatorTypeInferenceContext : public MLOperatorAttributes
{
public:
    MLOperatorTypeInferenceContext(IMLOperatorTypeInferenceContext* impl) : MLOperatorAttributes(impl), m_impl(impl) {}

    // For cases of interop where the caller needs to pass the unwrapped class across a boundary.
    Microsoft::WRL::ComPtr<IMLOperatorTypeInferenceContext> GetInterface() const noexcept
    {
        return m_impl;
    }

    uint32_t GetInputCount() const noexcept
    {
        return m_impl->GetInputCount();
    }

    uint32_t GetOutputCount() const noexcept
    {
        return m_impl->GetOutputCount();
    }

    MLOperatorEdgeDescription GetInputEdgeDescription(uint32_t inputIndex) const
    {
        MLOperatorEdgeDescription desc;
        ORT_THROW_IF_FAILED(m_impl->GetInputEdgeDescription(inputIndex, &desc));

        return desc;
    }

    void SetOutputEdgeDescription(uint32_t outputIndex, const MLOperatorEdgeDescription* edgeDesc) const
    {
        ORT_THROW_IF_FAILED(m_impl->SetOutputEdgeDescription(outputIndex, edgeDesc));
    }

 private:
    Microsoft::WRL::ComPtr<IMLOperatorTypeInferenceContext> m_impl;
};

class MLOperatorKernelContext
{
public:
    MLOperatorKernelContext(IMLOperatorKernelContext* impl) : m_impl(impl) {}

    // Retrieve the underlying ABI compatible interface from the wrapper, for cases of interop
    // between components or different DLLs where the caller needs to pass the unwrapped class
    // across a boundary. e.g. Operator implementations may use the helper classes so that
    // they can use exceptions without checking every return value, but then they need to pass
    // results onward to a different component which expects the lower level currency.
    Microsoft::WRL::ComPtr<IMLOperatorKernelContext> GetInterface() const noexcept
    {
        return m_impl;
    }

    bool IsSequenceInputTensor(uint32_t inputIndex) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorKernelContextPrivate> operatorKernelContext;
        m_impl.As(&operatorKernelContext);
        return operatorKernelContext->IsSequenceInputTensor(inputIndex);
    }

    uint32_t GetSequenceInputCount(uint32_t inputIndex) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorKernelContextPrivate> operatorKernelContext;
        m_impl.As(&operatorKernelContext);
        uint32_t inputCount = 0;
        MLOperatorTensorDataType dataType;
        ORT_THROW_IF_FAILED(operatorKernelContext->GetSequenceInputInfo(inputIndex, &inputCount, &dataType));
        return inputCount;
    }

    MLOperatorTensorDataType GetSequenceInputDataType(uint32_t inputIndex) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorKernelContextPrivate> operatorKernelContext;
        m_impl.As(&operatorKernelContext);
        uint32_t inputCount = 0;
        MLOperatorTensorDataType dataType;
        ORT_THROW_IF_FAILED(operatorKernelContext->GetSequenceInputInfo(inputIndex, &inputCount, &dataType));
        return dataType;
    }

    MLOperatorTensor GetSequenceInputTensor(uint32_t inputIndex, uint32_t sequenceIndex) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorKernelContextPrivate> operatorKernelContext;
        m_impl.As(&operatorKernelContext);

        Microsoft::WRL::ComPtr<IMLOperatorTensor> tensor;
        ORT_THROW_HR_IF(E_INVALIDARG, !operatorKernelContext->IsSequenceInputTensor(inputIndex));
        ORT_THROW_IF_FAILED(operatorKernelContext->GetSequenceInputTensor(inputIndex, sequenceIndex, &tensor));
        return tensor.Get();
    }

    void PrepareSequenceOutput(uint32_t outputIndex, MLOperatorTensorDataType dataType) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorKernelContextPrivate> operatorKernelContext;
        m_impl.As(&operatorKernelContext);
        ORT_THROW_IF_FAILED(operatorKernelContext->PrepareSequenceOutput(outputIndex, dataType));
    }

    MLOperatorTensor GetSequenceOutputTensor(
        uint32_t outputIndex,
        uint32_t sequenceIndex,
        MLOperatorTensorDataType dataType,
        uint32_t dimensions,
        const uint32_t* dimensionSizes,
        bool gpuOutput) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorKernelContextPrivate> operatorKernelContext;
        m_impl.As(&operatorKernelContext);

        Microsoft::WRL::ComPtr<IMLOperatorTensor> tensor;
        ORT_THROW_IF_FAILED(operatorKernelContext->GetSequenceOutputTensor(
            outputIndex,
            sequenceIndex,
            dataType,
            dimensions,
            dimensionSizes,
            gpuOutput,
            &tensor));
        return tensor.Get();
    }

    MLOperatorTensor GetInputTensor(uint32_t inputIndex) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorTensor> tensor;
        ORT_THROW_IF_FAILED(m_impl->GetInputTensor(inputIndex, &tensor));
        return tensor.Get();
    }

    MLOperatorTensor GetOutputTensor(uint32_t outputIndex) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorTensor> tensor;
        ORT_THROW_IF_FAILED(m_impl->GetOutputTensor(outputIndex, &tensor));
        return tensor.Get();
    }

    MLOperatorTensor GetOutputTensor(uint32_t outputIndex, const std::vector<uint32_t> dimensionSizes) const
    {
        Microsoft::WRL::ComPtr<IMLOperatorTensor> tensor;
        ORT_THROW_IF_FAILED(m_impl->GetOutputTensor(outputIndex, static_cast<uint32_t>(dimensionSizes.size()), dimensionSizes.data(), &tensor));
        return tensor.Get();
    }

    Microsoft::WRL::ComPtr<IUnknown> AllocateTemporaryData(size_t size) const
    {
        Microsoft::WRL::ComPtr<IUnknown> ret;
        ORT_THROW_IF_FAILED(m_impl->AllocateTemporaryData(size, &ret));
        return ret;
    }

    Microsoft::WRL::ComPtr<IUnknown> GetExecutionInterface() const noexcept
    {
        Microsoft::WRL::ComPtr<IUnknown> ret;
        m_impl->GetExecutionInterface(&ret);
        return ret;
    }

 private:
    Microsoft::WRL::ComPtr<IMLOperatorKernelContext> m_impl;
};

// Helper class for operator implementations, templatized by the
// implementation type. This class converts ABI types to wrappers,
// supports STL types, and converts exceptions to return values.
template <class T>
class MLOperatorKernel : public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IMLOperatorKernel>,
    public T
{
public:
    static HRESULT STDMETHODCALLTYPE CreateInstance(IMLOperatorKernelCreationContext& info, IMLOperatorKernel** opKernel) noexcept
    {
        ORT_TRY
        {
            Microsoft::WRL::ComPtr<MLOperatorKernel> kernel = wil::MakeOrThrow<MLOperatorKernel>(MLOperatorKernelCreationContext(&info));

            *opKernel = kernel.Detach();
            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    MLOperatorKernel(const MLOperatorKernelCreationContext& info) : T(info)
    {
    }

    virtual ~MLOperatorKernel()
    {
    }

    HRESULT STDMETHODCALLTYPE Compute(IMLOperatorKernelContext* context) noexcept override
    {
        ORT_TRY
        {
            T::Compute(MLOperatorKernelContext(context));
            return S_OK;
        }
        ORT_CATCH_RETURN
    }

    using T::Compute;
};

using MLOperatorTypeInferenceFunction = void (CALLBACK*)(IMLOperatorTypeInferenceContext*);
using MLOperatorShapeInferenceFunction = void (CALLBACK*)(IMLOperatorShapeInferenceContext*);
using MLOperatorKernelCreateFn = void(CALLBACK*)(IMLOperatorKernelCreationContext*, IMLOperatorKernel**);
using MLOperatorSupportQueryFunction = void (CALLBACK*)(IMLOperatorSupportQueryContextPrivate*, bool*);

class MLOperatorShapeInferrer : public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IMLOperatorShapeInferrer>
{
public:
    MLOperatorShapeInferrer(MLOperatorShapeInferenceFunction shapeInferenceFn) :
        m_shapeInferenceFn(shapeInferenceFn)
    {}

    HRESULT STDMETHODCALLTYPE InferOutputShapes(IMLOperatorShapeInferenceContext* context) noexcept override
    {
        ORT_TRY
        {
            m_shapeInferenceFn(context);
            return S_OK;
        }
        ORT_CATCH_RETURN
    }

private:
    MLOperatorShapeInferenceFunction m_shapeInferenceFn = nullptr;
};

class MLOperatorSupportQuery : public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IMLOperatorSupportQueryPrivate>
{
public:
    MLOperatorSupportQuery(MLOperatorSupportQueryFunction queryFn) :
        m_queryFn(queryFn)
    {}

    HRESULT STDMETHODCALLTYPE QuerySupport(
        IMLOperatorSupportQueryContextPrivate* context,
        BOOL* isSupported) noexcept override
    {
        ORT_TRY
        {
            bool fIsSupported = false;
            m_queryFn(context, &fIsSupported);
            *isSupported = fIsSupported ? TRUE : FALSE;
            return S_OK;
        }
        ORT_CATCH_RETURN
    }

private:
    MLOperatorSupportQueryFunction m_queryFn = nullptr;
};

class MLOperatorTypeInferrer : public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IMLOperatorTypeInferrer>
{
public:
    MLOperatorTypeInferrer(MLOperatorTypeInferenceFunction typeInferenceFn) :
        m_typeInferenceFn(typeInferenceFn)
    {}

    HRESULT STDMETHODCALLTYPE InferOutputTypes(IMLOperatorTypeInferenceContext* context) noexcept override
    {
        ORT_TRY
        {
            m_typeInferenceFn(context);
            return S_OK;
        }
        ORT_CATCH_RETURN
    }

private:
    MLOperatorTypeInferenceFunction m_typeInferenceFn = nullptr;
};

class MLOperatorKernelFactory : public Microsoft::WRL::RuntimeClass<
    Microsoft::WRL::RuntimeClassFlags<Microsoft::WRL::ClassicCom>, IMLOperatorKernelFactory>
{
public:
    MLOperatorKernelFactory(MLOperatorKernelCreateFn createFn) :
        m_createFn(createFn)
    {}

    HRESULT STDMETHODCALLTYPE CreateKernel(
        IMLOperatorKernelCreationContext* context,
        _COM_Outptr_ IMLOperatorKernel** kernel) noexcept override
    {
        ORT_TRY
        {
            m_createFn(context, kernel);
            return S_OK;
        }
        ORT_CATCH_RETURN
    }

private:
    MLOperatorKernelCreateFn m_createFn = nullptr;
};
