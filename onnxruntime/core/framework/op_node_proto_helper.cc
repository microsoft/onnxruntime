// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/onnx_protobuf.h"
#include "core/graph/op.h"
#include "gsl/gsl"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

template <class T>
bool HasTyped(const AttributeProto*);

template <>
inline bool HasTyped<float>(const AttributeProto* attr) {
  return utils::HasFloat(*attr);
}
template <>
inline bool HasTyped<int64_t>(const AttributeProto* attr) {
  return utils::HasInt(*attr);
}
template <>
inline bool HasTyped<std::string>(const AttributeProto* attr) {
  return utils::HasString(*attr);
}

template <>
inline bool HasTyped<TensorProto>(const AttributeProto* attr) {
  return utils::HasTensor(*attr);
}
template <>
inline bool HasTyped<GraphProto>(const AttributeProto* attr) {
  return utils::HasGraph(*attr);
}

#define ORT_DEFINE_GET_ATTR(IMPL_T, T, type)                                                       \
  template <>                                                                                      \
  template <>                                                                                      \
  Status OpNodeProtoHelper<IMPL_T>::GetAttr<T>(                                                    \
      const std::string& name, T* value) const {                                                   \
    const AttributeProto* attr = TryGetAttribute(name);                                            \
    if (!attr) {                                                                                   \
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "No attribute with name:'", name, "'is defined."); \
    }                                                                                              \
    if (!HasTyped<T>(attr)) {                                                                      \
      return Status(ONNXRUNTIME, FAIL, "Attibute name and type don't match");                      \
    } else {                                                                                       \
      *value = static_cast<T>(attr->type());                                                       \
      return Status::OK();                                                                         \
    }                                                                                              \
  }

#define ORT_DEFINE_GET_ATTRS(IMPL_T, T, list)                                      \
  template <>                                                                      \
  template <>                                                                      \
  Status OpNodeProtoHelper<IMPL_T>::GetAttrs<T>(                                   \
      const std::string& name, std::vector<T>& values) const {                     \
    const AttributeProto* attr = TryGetAttribute(name);                            \
    if (!attr) {                                                                   \
      return Status(ONNXRUNTIME, FAIL, "No attribute with this name is defined."); \
    }                                                                              \
    values.reserve(attr->list##_size());                                           \
    for (int i = 0; i < attr->list##_size(); ++i) {                                \
      values.push_back(static_cast<T>(attr->list(i)));                             \
    }                                                                              \
    return Status::OK();                                                           \
  }                                                                                \
  template <>                                                                      \
  template <>                                                                      \
  Status OpNodeProtoHelper<IMPL_T>::GetAttrs<T>(                                   \
      const std::string& name, gsl::span<T> values) const {                        \
    const AttributeProto* attr = TryGetAttribute(name);                            \
    if (!attr) {                                                                   \
      return Status(ONNXRUNTIME, FAIL, "No attribute with this name is defined."); \
    }                                                                              \
    ORT_ENFORCE(values.size() == static_cast<size_t>(attr->list##_size()));        \
    for (int i = 0; i < attr->list##_size(); ++i) {                                \
      values[i] = static_cast<T>(attr->list(i));                                   \
    }                                                                              \
    return Status::OK();                                                           \
  }

#if !defined(ORT_MINIMAL_BUILD)
#define ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(type, list)   \
  ORT_DEFINE_GET_ATTR(ProtoHelperNodeContext, type, list) \
  ORT_DEFINE_GET_ATTR(InferenceContext, type, list)

#define ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(type, list)   \
  ORT_DEFINE_GET_ATTRS(ProtoHelperNodeContext, type, list) \
  ORT_DEFINE_GET_ATTRS(InferenceContext, type, list)
#else
#define ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(type, list) \
  ORT_DEFINE_GET_ATTR(ProtoHelperNodeContext, type, list)

#define ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(type, list) \
  ORT_DEFINE_GET_ATTRS(ProtoHelperNodeContext, type, list)
#endif

ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(float, f)
ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(int64_t, i)
ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(std::string, s)
ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(TensorProto, t)
ORT_DEFINE_GET_ATTR_SPECIALIZATIONS(GraphProto, g)
ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(float, floats)
ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(int64_t, ints)
ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(std::string, strings)
ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(TensorProto, tensors)
ORT_DEFINE_GET_ATTRS_SPECIALIZATIONS(GraphProto, graphs)

size_t ProtoHelperNodeContext::getNumInputs() const {
  return node_.InputDefs().size();
}

size_t ProtoHelperNodeContext::getNumOutputs() const {
  return node_.OutputDefs().size();
}

const AttributeProto* ProtoHelperNodeContext::getAttribute(const std::string& name) const {
  const onnxruntime::NodeAttributes& attributes = node_.GetAttributes();
  auto it = attributes.find(name);
  if (it != attributes.end()) {
    return &it->second;
  }
  return nullptr;
}

const TypeProto* ProtoHelperNodeContext::getInputType(size_t index) const {
  return node_.InputDefs()[index]->TypeAsProto();
}

const TypeProto* ProtoHelperNodeContext::getOutputType(size_t index) const {
  return node_.OutputDefs()[index]->TypeAsProto();
}

template <class Impl_t>
uint32_t OpNodeProtoHelper<Impl_t>::GetPrimitiveAttrElementCount(AttributeProto_AttributeType type,
                                                                 const std::string& name) const noexcept {
  const AttributeProto* attr = impl_->getAttribute(name);
  if (attr) {
    switch (type) {
      case AttributeProto_AttributeType_FLOAT:
      case AttributeProto_AttributeType_INT:
      case AttributeProto_AttributeType_STRING:
        return 1;

      case AttributeProto_AttributeType_FLOATS:
        return attr->floats_size();
      case AttributeProto_AttributeType_INTS:
        return attr->ints_size();
      case AttributeProto_AttributeType_STRINGS:
        return attr->strings_size();

        // The following are unsupported through this method
      case AttributeProto_AttributeType_UNDEFINED:
      case AttributeProto_AttributeType_TENSOR:
      case AttributeProto_AttributeType_GRAPH:
      case AttributeProto_AttributeType_SPARSE_TENSOR:
      case AttributeProto_AttributeType_TENSORS:
      case AttributeProto_AttributeType_GRAPHS:
      case AttributeProto_AttributeType_SPARSE_TENSORS:
      default:
        return 0;
    }
  }

  return 0;
}

template <class Impl_t>
bool OpNodeProtoHelper<Impl_t>::HasPrimitiveAttribute(AttributeProto_AttributeType type,
                                                      const std::string& name) const noexcept {
  return GetPrimitiveAttrElementCount(type, name) > 0;
}

template class OpNodeProtoHelper<ProtoHelperNodeContext>;
#if !defined(ORT_MINIMAL_BUILD)
template class OpNodeProtoHelper<InferenceContext>;
#endif

}  // namespace onnxruntime
