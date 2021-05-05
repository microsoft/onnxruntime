// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

#include "core/framework/customregistry.h"
#include "core/framework/execution_frame.h"
#include "core/session/onnxruntime_c_api.h"
#include "core/providers/dml/DmlExecutionProvider/inc/MLOperatorAuthor.h"

#include "MLOperatorAuthorImpl.h"
#include "core/providers/dml/OperatorAuthorHelper/MLOperatorAuthorPrivate.h"

using namespace Microsoft::WRL;

namespace Windows::AI::MachineLearning::Adapter
{

size_t AttributeValue::ElementCount() const {
  switch (type) {
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
      THROW_HR(E_FAIL);
  }
}

void AttributeValue::GetAttribute(
    MLOperatorAttributeType type,
    uint32_t elementCount,
    size_t elementByteSize,
    void* value) const {
  switch (type) {
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
      THROW_HR(E_INVALIDARG);
  }
}

const std::string* AttributeValue::GetStringAttribute(
    _In_z_ const char* name,
    uint32_t elementIndex) const {
  ML_CHECK_BOOL((type == MLOperatorAttributeType::String && elementIndex == 0 && strings.size() == 1) ||
                (type == MLOperatorAttributeType::StringArray && elementIndex < strings.size()));

  return &strings.data()[elementIndex];
}

bool IsAllocationInterface(const ::OrtMemoryInfo& info) {
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
    IUnknown** abiAllocation) {
  if (winmlProvider) {
    winmlProvider->GetABIDataInterface(isInternalOperator, allocation, abiAllocation);
  } else {
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

onnx::AttributeProto_AttributeType ToProto(MLOperatorAttributeType type) {
  switch (type) {
    case MLOperatorAttributeType::Float:
      return MLAttributeTypeTraits<MLOperatorAttributeType::Float>::ProtoType;
    case MLOperatorAttributeType::Int:
      return MLAttributeTypeTraits<MLOperatorAttributeType::Int>::ProtoType;
    case MLOperatorAttributeType::FloatArray:
      return MLAttributeTypeTraits<MLOperatorAttributeType::FloatArray>::ProtoType;
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

bool IsPrimitiveAttributeType(MLOperatorAttributeType type) {
  switch (type) {
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

template <>
struct MLTypeTraits<onnxruntime::MLFloat16> {
  static const MLOperatorTensorDataType TensorType = MLOperatorTensorDataType::Float16;
};

#define ML_TENSOR_TYPE_CASE(x)          \
  if (onnxruntime::utils::IsPrimitiveDataType<x>(type)) { \
    return MLTypeTraits<x>::TensorType; \
  }

::MLOperatorTensorDataType ToMLTensorDataType(onnxruntime::MLDataType type) {
  if (onnxruntime::utils::IsDataTypeString(type)) {
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

  THROW_HR(E_NOTIMPL);
}

#undef ML_TENSOR_TYPE_CASE
#define ML_TENSOR_TYPE_CASE(x)                            \
  if (type == MLTypeTraits<x>::TensorType) {              \
    return onnxruntime::DataTypeImpl::GetTensorType<x>(); \
  }

onnxruntime::MLDataType ToTensorDataType(::MLOperatorTensorDataType type) {
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

  THROW_HR(E_NOTIMPL);
}

::MLOperatorTensorDataType ToMLTensorDataType(onnx::TensorProto_DataType type) {
  switch (type) {
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
      THROW_HR(E_NOTIMPL);
  }
}

::MLOperatorEdgeDescription ToMLEdgeDesc(const onnx::TypeProto* type) {
  // Initialized to undefined class and data type
  MLOperatorEdgeDescription ret = {};

  ML_CHECK_BOOL(type->value_case() == onnx::TypeProto::kTensorType ||
                type->value_case() == onnx::TypeProto::VALUE_NOT_SET);

  if (type->value_case() == onnx::TypeProto::kTensorType) {
    ret.edgeType = MLOperatorEdgeType::Tensor;
    const onnx::TypeProto_Tensor tensorType = type->tensor_type();
    if (tensorType.has_elem_type()) {
      ret.tensorDataType = ToMLTensorDataType(onnx::TensorProto_DataType(tensorType.elem_type()));
    }
  }

  return ret;
}

std::string ToTypeString(MLOperatorEdgeDescription desc) {
  if (desc.edgeType != MLOperatorEdgeType::Tensor) {
    THROW_HR(E_NOTIMPL);
  }

  switch (desc.tensorDataType) {
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
      THROW_HR(E_NOTIMPL);
  }
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
    MLOperatorTensorGetter& constantInputGetter) : m_impl(kerneInfo),
                                                   m_abiExecutionObject(abiExecutionObject),
                                                   m_inferredOutputShapes(inferredOutputShapes),
                                                   m_allowInputShapeQuery(allowInputShapeQuery),
                                                   m_allowOutputShapeQuery(allowOutputShapeQuery),
                                                   m_internalOperator(isInternalOperator),
                                                   OpNodeInfoWrapper(kerneInfo, inputShapeOverrides, defaultAttributes, requiredConstantCpuInputs, constantInputGetter) {
  const void* executionHandle = kerneInfo->GetExecutionProvider()->GetExecutionHandle();
  if (executionHandle) {
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
    uint32_t* elementCount) const noexcept try {
  VerifyNotClosed();

  *elementCount = 0;

  if (IsPrimitiveAttributeType(type)) {
    *elementCount = m_impl->GetPrimitiveAttrElementCount(ToProto(type), std::string(name));
  } else {
    // ONNX runtime does not implement OpNodeProtoHelper<Impl_t>::GetPrimitiveAttrElementCount for tensors.
    // So we need to test presence a different way.

    const onnx::AttributeProto* attributeProto = m_impl->TryGetAttribute(std::string(name));
    *elementCount = attributeProto ? 1 : 0;
  }

  // Look for a value in the kernel's registered defaults if one was not found
  if (*elementCount == 0 && m_defaultAttributes) {
    auto defaultAttr = m_defaultAttributes->find(name);
    if (defaultAttr != m_defaultAttributes->end()) {
      *elementCount = static_cast<uint32_t>(defaultAttr->second.ElementCount());
    }
  }

  return S_OK;
}
CATCH_RETURN();

template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
template <MLOperatorAttributeType T>
HRESULT OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetAttributeArrayHelper(
    _In_z_ const char* name,
    uint32_t elementCount,
    uint32_t elementByteSize,
    void* values) const {
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
    void* value) const noexcept try {
  VerifyNotClosed();

  // Look for a value in the kernel's registered defaults if one does not exist otherwise
  if (m_impl->GetPrimitiveAttrElementCount(ToProto(type), name) == 0) {
    if (!m_defaultAttributes) {
      THROW_HR(E_FAIL);
    }

    auto defaultAttr = m_defaultAttributes->find(name);
    if (defaultAttr == m_defaultAttributes->end()) {
      THROW_HR(E_FAIL);
    }

    defaultAttr->second.GetAttribute(type, elementCount, elementByteSize, value);
  } else {
    switch (type) {
      case MLOperatorAttributeType::Float:
        ML_CHECK_BOOL(elementCount == 1);
        return GetAttributeHelper<MLOperatorAttributeType::Float>(name, static_cast<uint32_t>(elementByteSize), value);

      case MLOperatorAttributeType::Int:
        ML_CHECK_BOOL(elementCount == 1);
        return GetAttributeHelper<MLOperatorAttributeType::Int>(name, static_cast<uint32_t>(elementByteSize), value);

      case MLOperatorAttributeType::FloatArray:
        return GetAttributeArrayHelper<MLOperatorAttributeType::FloatArray>(name, elementCount, static_cast<uint32_t>(elementByteSize), value);

      case MLOperatorAttributeType::IntArray:
        return GetAttributeArrayHelper<MLOperatorAttributeType::IntArray>(name, elementCount, static_cast<uint32_t>(elementByteSize), value);

      default:
        ML_CHECK_BOOL(false);
        break;
    }
  }

  return S_OK;
}
CATCH_RETURN();

template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
const std::string* OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetStringAttribute(
    _In_z_ const char* name,
    uint32_t elementIndex) const {
  // Get the proto attribute
  const onnx::AttributeProto* attr = m_impl->TryGetAttribute(std::string(name));

  // Look for a value in the kernel's registered defaults if one was not found
  if (!attr) {
    if (!m_defaultAttributes) {
      THROW_HR(E_FAIL);
    }

    auto defaultAttr = m_defaultAttributes->find(name);
    if (defaultAttr == m_defaultAttributes->end()) {
      THROW_HR(E_FAIL);
    }

    return defaultAttr->second.GetStringAttribute(name, elementIndex);
  } else {
    // Get the string vector from the attribute
    if (attr->has_s()) {
      return &attr->s();
    } else {
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
    uint32_t* attributeElementByteLength) const noexcept try {
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
CATCH_RETURN();

template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetStringAttributeElement(
    _In_z_ const char* name,
    uint32_t elementIndex,
    uint32_t attributeElementByteLength,
    char* attributeElement) const noexcept try {
  VerifyNotClosed();

  const std::string* protoString = GetStringAttribute(name, elementIndex);

  size_t stringLength = protoString->size();
  ML_CHECK_BOOL(stringLength < attributeElementByteLength);
  memcpy(attributeElement, protoString->c_str(), stringLength + 1);

  return S_OK;
}
CATCH_RETURN();

template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
template <MLOperatorAttributeType T>
HRESULT OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetAttributeHelper(
    _In_z_ const char* name,
    uint32_t elementByteSize,
    void* value) const {
  using elementType_t = typename MLAttributeTypeTraits<T>::Type;
  static_assert(!typename MLAttributeTypeTraits<T>::IsArray, "This function only works for simple non-array types.");
  ML_CHECK_BOOL(sizeof(elementType_t) == elementByteSize);
  THROW_IF_NOT_OK(m_impl->template GetAttr<elementType_t>(name, static_cast<elementType_t*>(value)));
  return S_OK;
}

template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetTensorAttribute(
    _In_z_ const char* name,
    _Outptr_ IMLOperatorTensor** tensor) const noexcept try {
  VerifyNotClosed();

  *tensor = nullptr;

  // Read the tensor if present, and wrap it in a IMLOperatorTensor.
  const onnx::AttributeProto* attributeProto = m_impl->TryGetAttribute(std::string(name));
  if (attributeProto) {
    if (attributeProto->has_t()) {
      const onnx::TensorProto* tensorProto = &attributeProto->t();
      Microsoft::WRL::ComPtr<IMLOperatorTensor> tensorWrapper = wil::MakeOrThrow<OnnxTensorWrapper>(const_cast<onnx::TensorProto*>(tensorProto));
      *tensor = tensorWrapper.Detach();
      return S_OK;
    }
  }

  return E_INVALIDARG;  // The argument has no valid matching attribute.
}
CATCH_RETURN();

template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetInputEdgeDescription(uint32_t inputIndex, MLOperatorEdgeDescription* edgeDesc) const noexcept try {
  VerifyNotClosed();

  memset(edgeDesc, 0, sizeof(*edgeDesc));
  const onnx::TypeProto* type = m_impl->GetInputType(inputIndex);
  ML_CHECK_BOOL(type != nullptr);
  *edgeDesc = ToMLEdgeDesc(type);

  assert(edgeDesc->edgeType != MLOperatorEdgeType::Undefined);
  assert((edgeDesc->edgeType != MLOperatorEdgeType::Tensor /*&& edgeDesc->edgeType != MLOperatorEdgeType::TensorSequence*/) ||
         edgeDesc->tensorDataType != MLOperatorTensorDataType::Undefined);

  return S_OK;
}
CATCH_RETURN();

template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetOutputEdgeDescription(uint32_t outputIndex, MLOperatorEdgeDescription* edgeDesc) const noexcept try {
  VerifyNotClosed();

  memset(edgeDesc, 0, sizeof(*edgeDesc));
  const onnx::TypeProto* type = m_impl->GetOutputType(outputIndex);
  ML_CHECK_BOOL(type != nullptr);
  *edgeDesc = ToMLEdgeDesc(type);

  return S_OK;
}
CATCH_RETURN();

template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetInputTensorShape(uint32_t inputIndex, uint32_t dimensionCount, uint32_t* dimensions) const noexcept try {
  VerifyNotClosed();

  memset(dimensions, 0, dimensionCount * sizeof(dimensions[0]));
  if (inputIndex >= GetInputCount()) {
    return E_INVALIDARG;
  }

  // Input shapes are determined either from the override or from the underlying proto
  if (m_inputShapesOverride) {
    if (m_inputShapesOverride->GetShape(inputIndex).size() != dimensionCount) {
      return E_INVALIDARG;
    }

    for (uint32_t i = 0; i < dimensionCount; ++i) {
      dimensions[i] = m_inputShapesOverride->GetShape(inputIndex)[i];
    }
  } else {
    const auto* inputType = m_impl->GetInputType(inputIndex);
    ML_CHECK_BOOL(inputType->has_tensor_type());
    for (uint32_t i = 0; i < dimensionCount; ++i) {
      // Shape inference is only done when all dimensions of all inputs have known values,
      // so the input tensors will always have shapes at this point.
      assert(inputType->tensor_type().shape().dim(i).has_dim_value());
      dimensions[i] = static_cast<uint32_t>(inputType->tensor_type().shape().dim(i).dim_value());
    }
  }

  return S_OK;
}
CATCH_RETURN();

template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
bool STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::IsInputValid(uint32_t inputIndex) const noexcept {
  if (IsClosed()) {
    return false;
  }

  return (GetInputCount() > inputIndex) && !!m_impl->GetInputType(inputIndex);
}

template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
bool STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::IsOutputValid(uint32_t outputIndex) const noexcept {
  if (IsClosed()) {
    return false;
  }

  return (GetOutputCount() > outputIndex) && !!m_impl->GetOutputType(outputIndex);
}

template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetInputTensorDimensionCount(uint32_t inputIndex, uint32_t* dimensionCount) const noexcept try {
  VerifyNotClosed();

  *dimensionCount = 0;

  if (inputIndex >= GetInputCount()) {
    return E_INVALIDARG;
  }

  // Input shapes are determined either from the override or from the underlying proto
  if (m_inputShapesOverride) {
    *dimensionCount = gsl::narrow_cast<uint32_t>(m_inputShapesOverride->GetShape(inputIndex).size());
  } else {
    const auto* inputType = m_impl->GetInputType(inputIndex);
    ML_CHECK_BOOL(inputType->has_tensor_type());

    // Shape inference is only done when all dimensions of all inputs have known values,
    // so the input tensors will always have shapes at this point.
    assert(inputType->tensor_type().has_shape());

    *dimensionCount = inputType->tensor_type().shape().dim_size();
  }

  return S_OK;
}
CATCH_RETURN();

template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
HRESULT STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetConstantInputTensor(uint32_t inputIndex, IMLOperatorTensor** tensor) const noexcept try {
  bool inputRequiredAsConstant = std::find(
                                     m_requiredConstantCpuInputs.begin(),
                                     m_requiredConstantCpuInputs.end(),
                                     inputIndex) != m_requiredConstantCpuInputs.end();

  THROW_HR_IF(E_INVALIDARG, !inputRequiredAsConstant);

  ComPtr<IMLOperatorTensor> tensorWrapper = m_constantInputGetter(inputIndex);

  if (tensorWrapper == nullptr) {
    // This shouldn't happen since kernel creation is deferred and repeated when required constant inputs are not present.
    return E_UNEXPECTED;
  }

  *tensor = tensorWrapper.Detach();

  return S_OK;
}
CATCH_RETURN();

HRESULT STDMETHODCALLTYPE OpKernelInfoWrapper::GetOutputTensorShape(uint32_t outputIndex, uint32_t dimensionCount, uint32_t* dimensions) const noexcept try {
  VerifyNotClosed();

  memset(dimensions, 0, dimensionCount * sizeof(dimensions[0]));

  if (!HasOutputShapeDescription()) {
    return E_FAIL;
  }

  if (outputIndex >= GetOutputCount()) {
    return E_INVALIDARG;
  }

  if (m_inferredOutputShapes->GetShape(outputIndex).size() != dimensionCount) {
    return E_INVALIDARG;
  }

  for (uint32_t i = 0; i < dimensionCount; ++i) {
    dimensions[i] = m_inferredOutputShapes->GetShape(outputIndex)[i];
  }

  return S_OK;
}
CATCH_RETURN();

HRESULT STDMETHODCALLTYPE OpKernelInfoWrapper::GetOutputTensorDimensionCount(uint32_t outputIndex, uint32_t* dimensionCount) const noexcept try {
  VerifyNotClosed();

  *dimensionCount = 0;

  if (!HasOutputShapeDescription()) {
    return E_FAIL;
  }

  if (outputIndex >= GetOutputCount()) {
    return E_INVALIDARG;
  }

  *dimensionCount = gsl::narrow_cast<uint32_t>(m_inferredOutputShapes->GetShape(outputIndex).size());

  return S_OK;
}
CATCH_RETURN();

bool STDMETHODCALLTYPE OpKernelInfoWrapper::HasTensorShapeDescription() const noexcept {
  return m_allowInputShapeQuery;
}

HRESULT STDMETHODCALLTYPE OpKernelInfoWrapper::GetTensorShapeDescription(IMLOperatorTensorShapeDescription** shapeInfo) const noexcept try {
  VerifyNotClosed();

  *shapeInfo = nullptr;

  if (!HasTensorShapeDescription()) {
    *shapeInfo = nullptr;
    return E_FAIL;
    //return MLStatus::REQUIREMENT_NOT_REGISTERED;
  }

  ComPtr<IMLOperatorTensorShapeDescription> ret = const_cast<OpKernelInfoWrapper*>(this);
  *shapeInfo = ret.Detach();
  return S_OK;
}
CATCH_RETURN();

void STDMETHODCALLTYPE OpKernelInfoWrapper::GetExecutionInterface(IUnknown** executionInterface) const noexcept {
  m_abiExecutionObject.CopyTo(executionInterface);
}

template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
uint32_t STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetInputCount() const noexcept {
  if (IsClosed()) {
    return 0;
  }

  return m_impl->GetInputCount();
}

template <class NodeInfoImpl_t, class Base1_t, class Base2_t>
uint32_t STDMETHODCALLTYPE OpNodeInfoWrapper<NodeInfoImpl_t, Base1_t, Base2_t>::GetOutputCount() const noexcept {
  if (IsClosed()) {
    return 0;
  }

  return m_impl->GetOutputCount();
}

bool STDMETHODCALLTYPE OpKernelInfoWrapper::HasOutputShapeDescription() const noexcept {
  return m_allowOutputShapeQuery;
}

DmlGraphOpKernelInfoWrapper::DmlGraphOpKernelInfoWrapper(
    const onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>* protoHelper,
    const void* executionHandle,
    bool isInternalOperator,
    const EdgeShapes* inferredOutputShapes,
    const AttributeMap* defaultAttributes,
    DmlGraphNodeCreateInfo* graphNodeCreateInfo,
    gsl::span<const uint32_t> requiredConstantCpuInputs,
    MLOperatorTensorGetter& constantInputGetter) : m_internalOperator(isInternalOperator),
                                                   m_inferredOutputShapes(inferredOutputShapes),
                                                   m_graphNodeCreateInfo(graphNodeCreateInfo),
                                                   OpNodeInfoWrapper(protoHelper, nullptr, defaultAttributes, requiredConstantCpuInputs, constantInputGetter) {
  // We assume the execution object inherits IUnknown as its first base
  m_abiExecutionObject = const_cast<IUnknown*>(static_cast<const IUnknown*>(executionHandle));
  m_abiExecutionObject.As(&m_winmlProvider);
}

HRESULT STDMETHODCALLTYPE DmlGraphOpKernelInfoWrapper::GetOutputTensorShape(uint32_t outputIndex, uint32_t dimensionCount, uint32_t* dimensions) const noexcept try {
  VerifyNotClosed();

  memset(dimensions, 0, dimensionCount * sizeof(dimensions[0]));

  if (!HasOutputShapeDescription()) {
    return E_FAIL;
  }

  if (outputIndex >= GetOutputCount()) {
    return E_INVALIDARG;
  }

  if (m_inferredOutputShapes->GetShape(outputIndex).size() != dimensionCount) {
    return E_INVALIDARG;
  }

  for (uint32_t i = 0; i < dimensionCount; ++i) {
    dimensions[i] = m_inferredOutputShapes->GetShape(outputIndex)[i];
  }

  return S_OK;
}
CATCH_RETURN();

HRESULT STDMETHODCALLTYPE DmlGraphOpKernelInfoWrapper::GetOutputTensorDimensionCount(uint32_t outputIndex, uint32_t* dimensionCount) const noexcept try {
  VerifyNotClosed();

  *dimensionCount = 0;

  if (!HasOutputShapeDescription()) {
    return E_FAIL;
  }

  if (outputIndex >= GetOutputCount()) {
    return E_INVALIDARG;
  }

  *dimensionCount = gsl::narrow_cast<uint32_t>(m_inferredOutputShapes->GetShape(outputIndex).size());

  return S_OK;
}
CATCH_RETURN();

bool STDMETHODCALLTYPE DmlGraphOpKernelInfoWrapper::HasTensorShapeDescription() const noexcept {
  return true;
}

HRESULT STDMETHODCALLTYPE DmlGraphOpKernelInfoWrapper::GetTensorShapeDescription(IMLOperatorTensorShapeDescription** shapeInfo) const noexcept try {
  VerifyNotClosed();

  *shapeInfo = nullptr;

  if (!HasTensorShapeDescription()) {
    *shapeInfo = nullptr;
    return E_FAIL;
    //return MLStatus::REQUIREMENT_NOT_REGISTERED;
  }

  ComPtr<IMLOperatorTensorShapeDescription> ret = const_cast<DmlGraphOpKernelInfoWrapper*>(this);
  *shapeInfo = ret.Detach();
  return S_OK;
}
CATCH_RETURN();

void STDMETHODCALLTYPE DmlGraphOpKernelInfoWrapper::GetExecutionInterface(IUnknown** executionInterface) const noexcept {
  m_abiExecutionObject.CopyTo(executionInterface);
}

bool STDMETHODCALLTYPE DmlGraphOpKernelInfoWrapper::HasOutputShapeDescription() const noexcept {
  // DML kernels are only used in graph in graph partitions when shapes are static
  return true;
}

bool STDMETHODCALLTYPE DmlGraphOpKernelInfoWrapper::IsDmlGraphNode() const noexcept {
  return (m_graphNodeCreateInfo != nullptr);
}

void DmlGraphOpKernelInfoWrapper::SetDmlProperties(_In_ const MLOperatorKernelDmlProperties* dmlProperties) const {
  // Populate the mappings between DML in/outs and kernel in/outs.  By default they are the same.
  if (dmlProperties && dmlProperties->kernelInputIndices) {
    m_graphNodeCreateInfo->kernelInputIndices.insert(
        m_graphNodeCreateInfo->kernelInputIndices.begin(),
        dmlProperties->kernelInputIndices,
        dmlProperties->kernelInputIndices + dmlProperties->dmlInputCount);
  } else {
    m_graphNodeCreateInfo->kernelInputIndices.resize(dmlProperties ? dmlProperties->dmlInputCount : GetInputCount());
    std::iota(m_graphNodeCreateInfo->kernelInputIndices.begin(), m_graphNodeCreateInfo->kernelInputIndices.end(), 0);
  }

  if (dmlProperties && dmlProperties->kernelOutputIndices) {
    m_graphNodeCreateInfo->kernelOutputIndices.insert(
        m_graphNodeCreateInfo->kernelOutputIndices.begin(),
        dmlProperties->kernelOutputIndices,
        dmlProperties->kernelOutputIndices + dmlProperties->dmlOutputCount);
  } else {
    m_graphNodeCreateInfo->kernelOutputIndices.resize(dmlProperties ? dmlProperties->dmlOutputCount : GetOutputCount());
    std::iota(m_graphNodeCreateInfo->kernelOutputIndices.begin(), m_graphNodeCreateInfo->kernelOutputIndices.end(), 0);
  }

  m_graphNodeCreateInfo->allowHalfPrecisionComputation = dmlProperties ? dmlProperties->allowHalfPrecisionComputation : true;
}

HRESULT STDMETHODCALLTYPE DmlGraphOpKernelInfoWrapper::SetDmlOperator(
    IDMLOperator* op,
    _In_ const DML_OPERATOR_DESC* desc,
    _In_opt_ const MLOperatorKernelDmlProperties* dmlProperties) const noexcept try {
  ML_CHECK_BOOL(op != nullptr);
  ML_CHECK_BOOL(dmlProperties != nullptr);

  m_graphNodeCreateInfo->initialized = true;

  SetDmlProperties(dmlProperties);

  m_graphNodeCreateInfo->op = op;
  m_graphNodeCreateInfo->desc = std::make_unique<AbstractOperatorDesc>(SchemaHelpers::ConvertOperatorDesc(*desc));

  return S_OK;
}
CATCH_RETURN();

OnnxTensorWrapper::OnnxTensorWrapper(onnx::TensorProto* impl) : m_impl(impl) {
  // The tensor may be stored as raw data or in typed fields.
  if (impl->has_raw_data()) {
    m_dataPtr = reinterpret_cast<std::byte*>(impl->mutable_raw_data()->data());
    m_tensorByteSize = impl->raw_data().size();
  } else {
    std::tie(m_unpackedTensor, m_tensorByteSize) = UnpackTensor(*impl);
    m_dataPtr = m_unpackedTensor.get();
  }
}

uint32_t STDMETHODCALLTYPE OnnxTensorWrapper::GetDimensionCount() const noexcept {
  if (IsClosed()) {
    return 0;
  }

  return gsl::narrow_cast<uint32_t>(m_impl->dims().size());
}

HRESULT STDMETHODCALLTYPE OnnxTensorWrapper::GetShape(
    uint32_t dimensionCount,
    uint32_t* dimensions) const noexcept try {
  VerifyNotClosed();

  std::fill(dimensions, dimensions + dimensionCount, 0u);

  uint32_t count = static_cast<uint32_t>(m_impl->dims().size());
  ML_CHECK_BOOL(dimensionCount == count);

  for (uint32_t i = 0; i < dimensionCount; ++i) {
    dimensions[i] = static_cast<uint32_t>(m_impl->dims()[i]);
  }

  return S_OK;
}
CATCH_RETURN();

MLOperatorTensorDataType STDMETHODCALLTYPE OnnxTensorWrapper::GetTensorDataType() const noexcept {
  try {
    VerifyNotClosed();
    return ToMLTensorDataType(static_cast<onnx::TensorProto_DataType>(m_impl->data_type()));
  } catch (...) {
    return MLOperatorTensorDataType::Undefined;
  }
}

bool STDMETHODCALLTYPE OnnxTensorWrapper::IsCpuData() const noexcept {
  return true;
}

bool STDMETHODCALLTYPE OnnxTensorWrapper::IsDataInterface() const noexcept {
  return false;
}

void* STDMETHODCALLTYPE OnnxTensorWrapper::GetData() noexcept {
  if (IsClosed()) {
    return nullptr;
  }

  return m_dataPtr;
}

void STDMETHODCALLTYPE OnnxTensorWrapper::GetDataInterface(IUnknown** dataInterface) noexcept {
  *dataInterface = nullptr;
}

TensorWrapper::TensorWrapper(onnxruntime::Tensor* impl, bool isDataInterface, IWinmlExecutionProvider* provider, bool isInternalOperator) : m_impl(impl), m_isDataInterface(isDataInterface), m_winmlExecutionProvider(provider), m_internalOperator(isInternalOperator) {
  if (impl) {
    if (isDataInterface) {
      // We assume that all data handles derive from IUnknown as their first base.
      m_dataInterface = static_cast<IUnknown*>(m_impl->MutableDataRaw());

      if (m_dataInterface) {
        if (m_winmlExecutionProvider) {
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
        } else {
          m_abiDataInterface = m_dataInterface;
        }
      }
    } else {
      m_tensorData = m_impl->MutableDataRaw();
    }
  }
}

uint32_t STDMETHODCALLTYPE TensorWrapper::GetDimensionCount() const noexcept {
  if (IsClosed()) {
    return 0;
  }

  return gsl::narrow_cast<uint32_t>(m_impl->Shape().NumDimensions());
}

HRESULT STDMETHODCALLTYPE TensorWrapper::GetShape(
    uint32_t dimensionCount,
    uint32_t* dimensions) const noexcept try {
  VerifyNotClosed();

  std::fill(dimensions, dimensions + dimensionCount, 0u);

  uint32_t count = static_cast<uint32_t>(m_impl->Shape().NumDimensions());
  ML_CHECK_BOOL(dimensionCount == count);

  for (size_t i = 0; i < dimensionCount; ++i) {
    dimensions[i] = static_cast<uint32_t>(m_impl->Shape()[i]);
  }

  return S_OK;
}
CATCH_RETURN();

MLOperatorTensorDataType STDMETHODCALLTYPE TensorWrapper::GetTensorDataType() const noexcept {
  try {
    VerifyNotClosed();
    return ToMLTensorDataType(m_impl->DataType());
  } catch (...) {
    return MLOperatorTensorDataType::Undefined;
  }
}

bool STDMETHODCALLTYPE TensorWrapper::IsCpuData() const noexcept {
  if (IsClosed()) {
    return true;
  }

  // tells caller whether this tensor is in CPU memory
  return !strcmp(m_impl->Location().name, onnxruntime::CPU) || m_impl->Location().mem_type == ::OrtMemType::OrtMemTypeCPUOutput || m_impl->Location().mem_type == ::OrtMemType::OrtMemTypeCPUInput;
}

bool STDMETHODCALLTYPE TensorWrapper::IsDataInterface() const noexcept {
  if (IsClosed()) {
    return false;
  }

  return m_isDataInterface;
}

void* STDMETHODCALLTYPE TensorWrapper::GetData() noexcept {
  if (IsClosed()) {
    return nullptr;
  }

  return m_isDataInterface ? nullptr : m_tensorData;
}

void STDMETHODCALLTYPE TensorWrapper::GetDataInterface(IUnknown** dataInterface) noexcept {
  if (!m_isDataInterface) {
    VerifyNotClosed();
    *dataInterface = nullptr;
  } else {
    m_abiDataInterface.CopyTo(dataInterface);
  }
}

void OpKernelContextWrapper::TransitionResourcesForOperatorIfRequired(bool isBeforeOp) {
  if (m_winmlProvider->TransitionsRequiredForOperator(m_internalOperator)) {
    std::vector<IUnknown*> resourcesToTransition;
    resourcesToTransition.reserve(m_inputTensors.size() + m_outputTensors.size() + m_temporaryAllocations.size());

    for (uint32_t i = 0; i < m_inputTensors.size(); ++i) {
      ComPtr<IMLOperatorTensor> tensor;
      THROW_IF_FAILED(GetInputTensor(i, tensor.GetAddressOf()));

      ComPtr<IUnknown> resource;
      tensor->GetDataInterface(resource.GetAddressOf());
      if (resource) {
        resourcesToTransition.push_back(resource.Get());
      }
    }

    for (uint32_t i = 0; i < m_outputTensors.size(); ++i) {
      ComPtr<IMLOperatorTensor> tensor;
      THROW_IF_FAILED(GetOutputTensor(i, tensor.GetAddressOf()));

      ComPtr<IUnknown> resource;
      tensor->GetDataInterface(resource.GetAddressOf());
      if (resource) {
        resourcesToTransition.push_back(resource.Get());
      }
    }

    for (auto& tempAlloc : m_temporaryAbiAllocations) {
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
    const EdgeShapes* outputShapes) : m_impl(context), m_provider(provider), m_internalOperator(isInternalOperator), m_outputShapes(outputShapes) {
  // Pre-size tensor arrays.    Member methods return pointers to these which
  // are stored in these arrays, which would become stale if the vectors reallocate
  // their internal storage.
  m_inputTensors.resize(context->InputCount());
  m_outputTensors.resize(context->OutputCount());

  const void* executionHandle = m_provider->GetExecutionHandle();
  if (executionHandle) {
    // We assume the execution object inherits IUnknown as its first base
    m_providerExecutionObject = const_cast<IUnknown*>(static_cast<const IUnknown*>(executionHandle));
    m_providerExecutionObject.As(&m_winmlProvider);

    // Query the actual object to return through the ABI, based on options registered
    // with the kernel
    m_abiExecutionObject = m_providerExecutionObject;
    if (m_winmlProvider) {
      m_winmlProvider->GetABIExecutionInterface(isInternalOperator, m_abiExecutionObject.ReleaseAndGetAddressOf());
    }

    TransitionResourcesForOperatorIfRequired(true);
  }
}

OpKernelContextWrapper::~OpKernelContextWrapper() {
  ClearTempAllocations();
}

void OpKernelContextWrapper::ClearTempAllocations() {
  if (m_winmlProvider) {
    m_temporaryAllocations.clear();
    m_temporaryAbiAllocations.clear();
  }
}

void OpKernelContextWrapper::Close() {
  if (m_winmlProvider && m_winmlProvider->TransitionsRequiredForOperator(m_internalOperator)) {
    TransitionResourcesForOperatorIfRequired(false);
  }

  for (auto& tensor : m_inputTensors) {
    if (tensor) {
      tensor->Close();
    }
  }

  for (auto& tensor : m_outputTensors) {
    if (tensor) {
      tensor->Close();
    }
  }

  ClearTempAllocations();

  __super::Close();
}

HRESULT STDMETHODCALLTYPE OpKernelContextWrapper::GetInputTensor(uint32_t inputIndex, IMLOperatorTensor** tensor) const noexcept try {
  VerifyNotClosed();
  *tensor = nullptr;

  ML_CHECK_BOOL(inputIndex < m_inputTensors.size());

  if (m_inputTensors[inputIndex]->GetInterface() == nullptr) {
    auto inputTensor = m_impl->Input<onnxruntime::Tensor>(inputIndex);

    ComPtr<TensorWrapper> tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
        const_cast<onnxruntime::Tensor*>(inputTensor),
        IsAllocationInterface(inputTensor->Location()),
        m_winmlProvider.Get(),
        m_internalOperator);

    const_cast<OpKernelContextWrapper*>(this)->m_inputTensors[inputIndex] = tensorWrapper;
  }

  const_cast<OpKernelContextWrapper*>(this)->m_inputTensors[inputIndex].CopyTo(tensor);

  return S_OK;
}
CATCH_RETURN();

HRESULT STDMETHODCALLTYPE OpKernelContextWrapper::GetOutputTensor(uint32_t outputIndex, IMLOperatorTensor** tensor) noexcept try {
  VerifyNotClosed();

  *tensor = nullptr;

  ML_CHECK_BOOL(outputIndex < m_outputTensors.size());

  // GetOutputTensor must be called unless a kernel provides shape inferencing,
  // in which case m_outputShapes will be valid here.
  if (!m_outputShapes) {
    return E_FAIL;
    //return MLStatus::SHAPE_INFERENCE_NOT_REGISTERED;
  }

  uint32_t dimensionCount = gsl::narrow_cast<uint32_t>(m_outputShapes->GetShape(outputIndex).size());
  return GetOutputTensor(outputIndex, dimensionCount, m_outputShapes->GetShape(outputIndex).data(), tensor);
}
CATCH_RETURN();

HRESULT STDMETHODCALLTYPE OpKernelContextWrapper::GetOutputTensor(uint32_t outputIndex, uint32_t dimensions, const uint32_t* dimensionSizes, IMLOperatorTensor** tensor) noexcept try {
  VerifyNotClosed();
  *tensor = nullptr;

  ML_CHECK_BOOL(outputIndex < m_outputTensors.size());

  // Verify that the provided shape matches the shape determined using the kernel's shape inference function.
  if (m_outputTensors[outputIndex]->GetInterface() == nullptr) {
    if (m_outputShapes) {
      if ((m_outputShapes->GetShape(outputIndex).size() != dimensions ||
           memcmp(dimensionSizes, m_outputShapes->GetShape(outputIndex).data(), dimensions * sizeof(*dimensionSizes)))) {
        return E_INVALIDARG;
      }
    }
    std::vector<int64_t> convertedSizes(dimensions);
    for (size_t i = 0; i < dimensions; ++i) {
      convertedSizes[i] = dimensionSizes[i];
    }

    onnxruntime::TensorShape shape(convertedSizes.data(), dimensions);
    auto outputTensor = m_impl->Output(outputIndex, shape);
    if (outputTensor) {
      ComPtr<TensorWrapper> tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
          const_cast<onnxruntime::Tensor*>(outputTensor),
          IsAllocationInterface(outputTensor->Location()),
          m_winmlProvider.Get(),
          m_internalOperator);

      const_cast<OpKernelContextWrapper*>(this)->m_outputTensors[outputIndex] = tensorWrapper;
    }
  }

  m_outputTensors[outputIndex].CopyTo(tensor);

  return S_OK;
}
CATCH_RETURN();

HRESULT STDMETHODCALLTYPE OpKernelContextWrapper::AllocateTemporaryData(size_t size, IUnknown** abiAllocation) const try {
  uint64_t allocId;
  return AllocateTemporaryData(size, abiAllocation, &allocId);
}
CATCH_RETURN();

HRESULT STDMETHODCALLTYPE OpKernelContextWrapper::AllocateTemporaryData(size_t size, IUnknown** abiAllocation, uint64_t* allocId) const try {
  VerifyNotClosed();

  *abiAllocation = nullptr;
  onnxruntime::AllocatorPtr alloc;
  THROW_IF_NOT_OK(m_impl->GetTempSpaceAllocator(&alloc));

  if (!IsAllocationInterface(alloc->Info())) {
    return E_FAIL;
  }

  ComPtr<IUnknown> allocation;
  allocation.Attach(static_cast<IUnknown*>(alloc->Alloc(size)));

  *allocId = m_winmlProvider->TryGetPooledAllocationId(allocation.Get(), 0);

  TranslateAllocationDataToAbi(m_winmlProvider.Get(), m_internalOperator, alloc->Info(), allocation.Get(), abiAllocation);

  if (m_winmlProvider->TransitionsRequiredForOperator(m_internalOperator)) {
    m_winmlProvider->TransitionResourcesForOperator(true, 1, abiAllocation);
  }

  // Ensure the allocation is freed and transitioned when the context destructs
  m_temporaryAllocations.push_back(allocation);
  m_temporaryAbiAllocations.push_back(*abiAllocation);

  return S_OK;
}
CATCH_RETURN();

void STDMETHODCALLTYPE OpKernelContextWrapper::GetExecutionInterface(IUnknown** executionInterface) const noexcept {
  m_abiExecutionObject.CopyTo(executionInterface);
}

std::vector<IMLOperatorTensor*> OpKernelContextWrapper::GetInputTensors() {
  std::vector<IMLOperatorTensor*> ret;
  ret.reserve(m_inputTensors.size());

  for (int i = 0; i < m_impl->InputCount(); ++i) {
    ComPtr<IMLOperatorTensor> tensor;
    THROW_IF_FAILED(GetInputTensor(i, tensor.GetAddressOf()));
    ret.push_back(m_inputTensors[i].Get());
  }

  return ret;
}

std::vector<IMLOperatorTensor*> OpKernelContextWrapper::GetOutputTensors(const EdgeShapes& outputShapes) {
  std::vector<IMLOperatorTensor*> ret;
  ret.reserve(m_outputTensors.size());

  THROW_HR_IF(E_INVALIDARG, m_impl->OutputCount() != outputShapes.EdgeCount());

  for (int i = 0; i < m_impl->OutputCount(); ++i) {
    ComPtr<IMLOperatorTensor> tensor;
    THROW_IF_FAILED(GetOutputTensor(
        i,
        static_cast<uint32_t>(outputShapes.GetShape(i).size()),
        outputShapes.GetShape(i).data(),
        tensor.GetAddressOf()));

    ret.push_back(m_outputTensors[i].Get());
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
    const AttributeMap* defaultAttributes) : OpKernel(kerneInfo),
                                             m_requiresInputShapesAtCreation(requiresInputShapesAtCreation),
                                             m_requiresOutputShapesAtCreation(requiresOutputShapesAtCreation),
                                             m_internalOperator(isInternalOperator),
                                             m_shapeInferrer(shapeInferrer),
                                             m_defaultAttributes(defaultAttributes) {
  assert(requiresInputShapesAtCreation || !requiresOutputShapesAtCreation);

  m_requiredConstantCpuInputs.assign(requiredConstantCpuInputs.begin(), requiredConstantCpuInputs.end());

  const void* executionHandle = kerneInfo.GetExecutionProvider()->GetExecutionHandle();
  if (executionHandle) {
    // We assume the execution object inherits IUnknown as its first base
    ComPtr<IUnknown> providerExecutionObject = const_cast<IUnknown*>(static_cast<const IUnknown*>(executionHandle));
    m_abiExecutionObject = providerExecutionObject;

    // Get the WinML-specific execution provider interface from the execution object.
    providerExecutionObject.As(&m_winmlProvider);

    if (m_winmlProvider) {
      // Get the particular object to return to a isInternalOperator based on the registration of that kernel.
      m_winmlProvider->GetABIExecutionInterface(isInternalOperator, m_abiExecutionObject.ReleaseAndGetAddressOf());
    }
  }

  bool requiredConstantCpuInputsAvailable = true;
  for (uint32_t index : requiredConstantCpuInputs) {
    const onnxruntime::Tensor* tensor = nullptr;
    if (!kerneInfo.TryGetConstantInput(index, &tensor) || !tensor) {
      requiredConstantCpuInputsAvailable = false;
      break;
    }
  }

  // If input sizes are either available or not required at creation, no need to delay kernel creation.
  if (requiredConstantCpuInputsAvailable && (!m_requiresInputShapesAtCreation || InputTensorShapesDefined())) {
    auto winmlProviderCapture = m_winmlProvider;
    auto internalOpCapture = m_internalOperator;

    MLOperatorTensorGetter constantInputGetter = [kerneInfo, winmlProviderCapture, internalOpCapture](uint32_t index) {
      Microsoft::WRL::ComPtr<IMLOperatorTensor> tensorWrapper = nullptr;
      const onnxruntime::Tensor* tensor = nullptr;
      if (kerneInfo.TryGetConstantInput(index, &tensor)) {
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
    if (m_requiresOutputShapesAtCreation) {
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
        constantInputGetter);

    THROW_IF_FAILED(operatorFactory->CreateKernel(kernelInfoWrapper.Get(), m_kernel.GetAddressOf()));
    kernelInfoWrapper->Close();

    // Ensure that scheduled work, if any, is completed before freeing the kernel if the execution
    // provider requires this.
    if (m_winmlProvider) {
      m_winmlProvider->QueueReference(m_kernel.Get());
    }
  } else {
    m_operatorFactory = operatorFactory;
  }
}

onnxruntime::Status AbiOpKernel::Compute(onnxruntime::OpKernelContext* context) const {
  auto winmlProviderCapture = m_winmlProvider;
  auto internalOpCapture = m_internalOperator;

  MLOperatorTensorGetter constantInputGetter = [context, winmlProviderCapture, internalOpCapture](uint32_t index) {
    Microsoft::WRL::ComPtr<IMLOperatorTensor> tensorWrapper = nullptr;
    const onnxruntime::Tensor* tensor = context->Input<onnxruntime::Tensor>(static_cast<int>(index));
    if (tensor != nullptr)
    {
        tensorWrapper = wil::MakeOrThrow<TensorWrapper>(
            const_cast<onnxruntime::Tensor*>(tensor),
            tensor ? IsAllocationInterface(tensor->Location()) : false,
            winmlProviderCapture.Get(),
            internalOpCapture);
    }

    return tensorWrapper;
  };

  auto inferShapesAndCreateKernel = [&](const EdgeShapes& inputShapes, EdgeShapes& outputShapes) -> ComPtr<IMLOperatorKernel> {
    // If the output size is not dynamic, infer it using the kernel. The result of inference is stored in m_inferredOutputShapes.
    if (m_requiresOutputShapesAtCreation) {
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
        constantInputGetter);

    ComPtr<IMLOperatorKernel> ret;
    THROW_IF_FAILED(m_operatorFactory->CreateKernel(kernelInfoWrapper.Get(), ret.GetAddressOf()));
    kernelInfoWrapper->Close();

    return ret;
  };

  // The kernel creation may have been delayed because input shapes were required but not inferred by schema.
  if (RequiresLazyInitialization()) {
    std::lock_guard<std::mutex> lock(m_mutex);

    if (RequiresLazyInitialization()) {
      m_inputShapesOfKernelInference = GetInputShapes(context);

      m_constantInputTensorContentsOfKernel.resize(context->InputCount());
      for (uint32_t index : m_requiredConstantCpuInputs) {
        const onnxruntime::Tensor* weakTensor = context->Input<onnxruntime::Tensor>(static_cast<int>(index));

        // Skip optional constant tensors.
        if (weakTensor != nullptr)
        {
          MLOperatorTensor tensor = MLOperatorTensor(constantInputGetter(index).Get());

          if (index >= static_cast<uint32_t>(context->InputCount())) {
            continue;
          }
          m_constantInputTensorContentsOfKernel[index].isValid = (tensor.GetInterface() != nullptr);

          if (tensor.GetInterface() != nullptr) {
            m_constantInputTensorContentsOfKernel[index].shape = tensor.GetShape();
            m_constantInputTensorContentsOfKernel[index].type = tensor.GetTensorDataType();
            m_constantInputTensorContentsOfKernel[index].data.resize(tensor.GetUnalignedTensorByteSize());
          }
          m_constantInputTensorContentsOfKernel[index].data.assign(
              reinterpret_cast<const std::byte*>(tensor.GetByteData()),
              reinterpret_cast<const std::byte*>(tensor.GetByteData()) + tensor.GetUnalignedTensorByteSize());
        }
      }

      m_kernel = inferShapesAndCreateKernel(m_inputShapesOfKernelInference, m_inferredOutputShapes);
      SetLazyInitialized();
    }
  } else if (m_inputShapesOfKernelInference.EdgeCount() > 0) {
    EdgeShapes local_input_shapes = GetInputShapes(context);

    bool requiredCpuInputsChanged = false;
    for (uint32_t index : m_requiredConstantCpuInputs) {
      if (index >= m_constantInputTensorContentsOfKernel.size()) {
        continue;
      }

      const TensorContent& lastValue = m_constantInputTensorContentsOfKernel[index];
      MLOperatorTensor currentValue(constantInputGetter(index).Get());

      if (lastValue.isValid != (currentValue.GetInterface() != nullptr)) {
        break;
      }

      if (lastValue.isValid) {
        if (lastValue.shape != currentValue.GetShape() ||
            lastValue.type != currentValue.GetTensorDataType() ||
            currentValue.GetUnalignedTensorByteSize() != lastValue.data.size() ||
            (memcmp(lastValue.data.data(), currentValue.GetByteData(), lastValue.data.size()) != 0)) {
          requiredCpuInputsChanged = true;
          break;
        }
      }
    }

    // In the edge case that the input size is changing across invocations and the kernel requires
    // its input size at construction, use a local instance of the kernel.
    if (local_input_shapes != m_inputShapesOfKernelInference || requiredCpuInputsChanged) {
      EdgeShapes localInferredOutputShapes;
      ComPtr<IMLOperatorKernel> localKernel = inferShapesAndCreateKernel(local_input_shapes, localInferredOutputShapes);

      ComPtr<OpKernelContextWrapper> kernelContextWrapper = wil::MakeOrThrow<OpKernelContextWrapper>(
          context,
          Info().GetExecutionProvider(),
          m_internalOperator,
          m_requiresOutputShapesAtCreation ? &localInferredOutputShapes : nullptr);

      THROW_IF_FAILED(localKernel->Compute(kernelContextWrapper.Get()));
      kernelContextWrapper->Close();

      // Ensure that scheduled work, if any, is completed before freeing the kernel if the execution
      // provider requires this.
      if (m_winmlProvider) {
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

  THROW_IF_FAILED(m_kernel->Compute(kernelContextWrapper.Get()));
  kernelContextWrapper->Close();

  // Ensure that scheduled work, if any, is completed before freeing the kernel if the execution
  // provider requires this.
  if (m_winmlProvider) {
    m_winmlProvider->QueueReference(m_kernel.Get());
  }

  return onnxruntime::Status();
}

bool AbiOpKernel::InputTensorShapesDefined() const {
  onnxruntime::ProtoHelperNodeContext protoContext(Node());
  onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext> info(&protoContext);

  return InputTensorShapesDefinedOnNode(info);
}

EdgeShapes AbiOpKernel::GetInputShapes(onnxruntime::OpKernelContext* context) const {
  EdgeShapes ret(context->InputCount());

  for (size_t i = 0; i < ret.EdgeCount(); ++i) {
    // The input type is null if unused
    auto inputType = context->InputType(static_cast<int>(i));
    if (inputType != nullptr && inputType->IsTensorType()) {
      const onnxruntime::Tensor* tensor = context->Input<onnxruntime::Tensor>(static_cast<int>(i));
      if (tensor) {
        ret.GetMutableShape(i).resize(tensor->Shape().GetDims().size());
        for (size_t j = 0; j < ret.GetMutableShape(i).size(); ++j) {
          ret.GetMutableShape(i)[j] = gsl::narrow_cast<uint32_t>(tensor->Shape().GetDims()[j]);
        }
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
    EdgeShapes& outputShapes) {
  onnxruntime::ProtoHelperNodeContext protoContext(node);
  onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext> info(&protoContext);

  ComPtr<MLKernelInferenceContext> inferenceContext = wil::MakeOrThrow<MLKernelInferenceContext>(&info, inputShapes, outputShapes, defaultAttributes, requiredConstantCpuInputs, constantInputGetter);

  outputShapes.Reset(info.GetOutputCount());

  THROW_IF_FAILED(shapeInferrer->InferOutputShapes(inferenceContext.Get()));
  inferenceContext->Close();

  for (size_t outputIndex = 0; outputIndex < outputShapes.EdgeCount(); ++outputIndex) {
    const onnx::TypeProto* outputProto = info.GetOutputType(outputIndex);

    // Skip this output if it is not valid.
    if (outputProto == nullptr) {
      continue;
    }

    if (outputProto->value_case() != onnx::TypeProto::kTensorType) {
      ML_CHECK_BOOL(outputShapes.GetShape(outputIndex).empty());
      continue;
    }

    const auto& tensorType = outputProto->tensor_type();

    if (tensorType.has_shape()) {
      const auto& shape = tensorType.shape();
      ML_CHECK_BOOL(shape.dim_size() == outputShapes.GetShape(outputIndex).size());

      for (uint32_t output_dim = 0; output_dim < outputShapes.GetShape(outputIndex).size(); ++output_dim) {
        if (shape.dim(output_dim).has_dim_value()) {
          int64_t expected_size = shape.dim(output_dim).dim_value();
          int64_t actual_size = outputShapes.GetShape(outputIndex)[output_dim];
          ML_CHECK_BOOL(expected_size == actual_size);
        }
      }
    }
  }
}

MLSchemaInferenceContext::MLSchemaInferenceContext(
    onnxruntime::OpNodeProtoHelper<onnx::InferenceContext>* info,
    onnx::InferenceContext* ctx,
    gsl::span<const uint32_t> requiredConstantCpuInputs) : OpNodeInfoWrapper(info,
                                                                             nullptr,
                                                                             nullptr,
                                                                             requiredConstantCpuInputs,
                                                                             MLOperatorTensorGetter([ctx](uint32_t index) {
                                                                               Microsoft::WRL::ComPtr<IMLOperatorTensor> tensorWrapper = wil::MakeOrThrow<OnnxTensorWrapper>(
                                                                                   const_cast<onnx::TensorProto*>(ctx->getInputData(index)));
                                                                               return tensorWrapper;
                                                                             })),
                                                           m_context(ctx) {
}

HRESULT STDMETHODCALLTYPE MLSchemaInferenceContext::SetOutputTensorShape(
    uint32_t outputIndex,
    uint32_t dimensionCount,
    const uint32_t* dimensions) noexcept try {
  VerifyNotClosed();

  MLOperatorEdgeDescription edgeDesc;
  THROW_IF_FAILED(GetOutputEdgeDescription(outputIndex, &edgeDesc));
  ML_CHECK_BOOL(edgeDesc.edgeType == MLOperatorEdgeType::Undefined || edgeDesc.edgeType == MLOperatorEdgeType::Tensor);

  // In the process of calling mutable_tensor_type, the type may switch from undefined to tensor.
  // This is done here in case the dimension count is zero (scalar)
  m_context->getOutputType(outputIndex)->mutable_tensor_type();

  for (uint32_t i = 0; i < dimensionCount; ++i) {
    auto dim = m_context->getOutputType(outputIndex)->mutable_tensor_type()->mutable_shape()->add_dim();
    dim->set_dim_value(dimensions[i]);
  }

  return S_OK;
}
CATCH_RETURN();

HRESULT STDMETHODCALLTYPE MLSchemaInferenceContext::SetOutputEdgeDescription(
    uint32_t outputIndex,
    const MLOperatorEdgeDescription* edgeDesc) const noexcept try {
  VerifyNotClosed();

  std::string typeStr = ToTypeString(*edgeDesc);
  m_context->getOutputType(outputIndex)->CopyFrom(onnx::Utils::DataTypeUtils::ToTypeProto(&typeStr));
  return S_OK;
}
CATCH_RETURN();

HRESULT STDMETHODCALLTYPE MLKernelInferenceContext::SetOutputTensorShape(
    uint32_t outputIndex,
    uint32_t dimensionCount,
    const uint32_t* dimensions) noexcept try {
  VerifyNotClosed();

  if (outputIndex >= m_inferredOutputShapes.EdgeCount()) {
    return E_INVALIDARG;
  }

  m_inferredOutputShapes.GetMutableShape(outputIndex).assign(dimensions, dimensions + dimensionCount);

  return S_OK;
}
CATCH_RETURN();

MLSupportQueryContext::MLSupportQueryContext(
        onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext>* info,
        const AttributeMap* defaultAttributes) : 
    OpNodeInfoWrapper(info, nullptr, defaultAttributes, gsl::span<const uint32_t>(), MLOperatorTensorGetter())
{
}

bool TryGetStaticShapeIfTensor(
    const onnx::TypeProto* inputProto,
    std::vector<uint32_t>& shapeDims) {
  // Skip this input if it is not valid.
  if (inputProto == nullptr) {
    return true;
  }

  if (inputProto->value_case() != onnx::TypeProto::kTensorType) {
    return true;
  }

  const auto& tensorType = inputProto->tensor_type();

  if (!tensorType.has_shape()) {
    return false;
  }

  const auto& shape = tensorType.shape();
  shapeDims.resize(shape.dim_size());

  for (uint32_t dimIndex = 0; dimIndex < static_cast<uint32_t>(shape.dim_size()); ++dimIndex) {
    if (!shape.dim(dimIndex).has_dim_value()) {
      return false;
    }

    shapeDims[dimIndex] = gsl::narrow<uint32_t>(shape.dim(dimIndex).dim_value());
  }

  return true;
}

bool TryGetStaticInputShapes(const onnxruntime::Node& node, EdgeShapes& inputShapes) {
  onnxruntime::ProtoHelperNodeContext protoContext(node);
  onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext> info(&protoContext);

  inputShapes.Reset(info.GetInputCount());

  for (size_t inputIndex = 0; inputIndex < inputShapes.EdgeCount(); ++inputIndex) {
    const onnx::TypeProto* inputProto = info.GetInputType(inputIndex);
    if (!TryGetStaticShapeIfTensor(inputProto, inputShapes.GetMutableShape(inputIndex))) {
      return false;
    }
  }

  return true;
}

bool TryGetStaticOutputShapes(const onnxruntime::Node& node, EdgeShapes& outputShapes) {
  onnxruntime::ProtoHelperNodeContext protoContext(node);
  onnxruntime::OpNodeProtoHelper<onnxruntime::ProtoHelperNodeContext> info(&protoContext);

  outputShapes.Reset(info.GetOutputCount());

  for (size_t outputIndex = 0; outputIndex < outputShapes.EdgeCount(); ++outputIndex) {
    const onnx::TypeProto* outputProto = info.GetOutputType(outputIndex);
    if (!TryGetStaticShapeIfTensor(outputProto, outputShapes.GetMutableShape(outputIndex))) {
      return false;
    }
  }

  return true;
}

bool ContainsEmptyDimensions(const EdgeShapes& shapes, gsl::span<const uint32_t> ignoredShapeIndices) {
  for (size_t i = 0; i < shapes.EdgeCount(); i++) {
    const std::vector<uint32_t>& shape = shapes.GetShape(i);

    if (std::find(shape.begin(), shape.end(), 0) != shape.end() && 
        std::find(ignoredShapeIndices.begin(), ignoredShapeIndices.end(), i) == ignoredShapeIndices.end()) {
          return true;
    }
  }

  return false;
}

std::tuple<std::unique_ptr<std::byte[]>, size_t> UnpackTensor(const onnx::TensorProto& initializer) {
  std::unique_ptr<std::byte[]> unpackedTensor;
  size_t tensorByteSize = 0;

#define CASE_PROTO(X, Y, Z)                                                                        \
  case ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_##X: {                           \
    size_t elementCount = initializer.##Z();                                                       \
    tensorByteSize = elementCount * sizeof(Y);                                                     \
    unpackedTensor.reset(new std::byte[tensorByteSize]);                                           \
    THROW_HR_IF(E_FAIL, !onnxruntime::utils::UnpackTensor(                                         \
                             initializer,                                                          \
                             initializer.has_raw_data() ? initializer.raw_data().data() : nullptr, \
                             initializer.has_raw_data() ? initializer.raw_data().size() : 0,       \
                             reinterpret_cast<Y*>(unpackedTensor.get()), elementCount)             \
                             .IsOK());                                                             \
    break;                                                                                         \
  }
  switch (initializer.data_type()) {
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
    CASE_PROTO(UINT64, uint64_t, int64_data_size);
    CASE_PROTO(FLOAT16, onnxruntime::MLFloat16, int32_data_size);
    default:
      THROW_HR(E_INVALIDARG);
  }

  return std::make_tuple(std::move(unpackedTensor), tensorByteSize);
}
}  // namespace winrt::Windows::AI::MachineLearning::implementation