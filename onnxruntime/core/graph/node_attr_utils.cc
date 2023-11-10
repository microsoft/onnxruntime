// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/node_attr_utils.h"

#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"

using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::AttributeProto_AttributeType;
using ONNX_NAMESPACE::TensorProto;
using ONNX_NAMESPACE::SparseTensorProto;
using ONNX_NAMESPACE::TypeProto;
using ONNX_NAMESPACE::NodeProto;
using ONNX_NAMESPACE::GraphProto;


namespace onnxruntime::utils {

static void SetNameAndType(std::string attr_name, AttributeProto_AttributeType attr_type, AttributeProto& a) {
  a.set_name(std::move(attr_name));
  a.set_type(attr_type);
}

#define MAKE_BASIC_ATTR_IMPL(type, enumType, field)                 \
  AttributeProto MakeAttribute(std::string attr_name, type value) { \
    AttributeProto a;                                               \
    a.set_##field(std::move(value));                                \
    SetNameAndType(std::move(attr_name), enumType, a);              \
    return a;                                                       \
  }

#define MAKE_ATTR_IMPL(type, enumType, field)                       \
  AttributeProto MakeAttribute(std::string attr_name, type value) { \
    AttributeProto a;                                               \
    *(a.mutable_##field()) = std::move(value);                      \
    SetNameAndType(std::move(attr_name), enumType, a);              \
    return a;                                                       \
  }

#define MAKE_LIST_ATTR_IMPL(type, enumType, field)                                    \
  AttributeProto MakeAttribute(std::string attr_name, gsl::span<const type> values) { \
    AttributeProto a;                                                                 \
    auto* mutable_field = a.mutable_##field();                                        \
    for (const auto& val : values) {                                                  \
      *(mutable_field->Add()) = val;                                                  \
    }                                                                                 \
    SetNameAndType(std::move(attr_name), enumType, a);                                \
    return a;                                                                         \
  }

MAKE_BASIC_ATTR_IMPL(int64_t, AttributeProto_AttributeType::AttributeProto_AttributeType_INT, i)
MAKE_LIST_ATTR_IMPL(int64_t, AttributeProto_AttributeType::AttributeProto_AttributeType_INTS, ints)

MAKE_BASIC_ATTR_IMPL(float, AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT, f)
MAKE_LIST_ATTR_IMPL(float, AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS, floats)

MAKE_ATTR_IMPL(std::string, AttributeProto_AttributeType::AttributeProto_AttributeType_STRING, s)
MAKE_LIST_ATTR_IMPL(std::string, AttributeProto_AttributeType::AttributeProto_AttributeType_STRINGS, strings)

MAKE_ATTR_IMPL(TensorProto, AttributeProto_AttributeType::AttributeProto_AttributeType_TENSOR, t)
MAKE_LIST_ATTR_IMPL(TensorProto, AttributeProto_AttributeType::AttributeProto_AttributeType_TENSORS, tensors)

#if !defined(DISABLE_SPARSE_TENSORS)
MAKE_ATTR_IMPL(SparseTensorProto, AttributeProto_AttributeType::AttributeProto_AttributeType_SPARSE_TENSOR,
               sparse_tensor)
MAKE_LIST_ATTR_IMPL(SparseTensorProto, AttributeProto_AttributeType::AttributeProto_AttributeType_SPARSE_TENSORS,
                    sparse_tensors)
#endif

MAKE_ATTR_IMPL(TypeProto, AttributeProto_AttributeType::AttributeProto_AttributeType_TYPE_PROTO, tp)
MAKE_LIST_ATTR_IMPL(TypeProto, AttributeProto_AttributeType::AttributeProto_AttributeType_TYPE_PROTOS, type_protos)

MAKE_ATTR_IMPL(GraphProto, AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPH, g)
MAKE_LIST_ATTR_IMPL(GraphProto, AttributeProto_AttributeType::AttributeProto_AttributeType_GRAPHS, graphs)

#undef MAKE_BASIC_ATTR_IMPL
#undef MAKE_ATTR_IMPL
#undef MAKE_LIST_ATTR_IMPL

std::pair<NodeAttributes::iterator, bool> SetNodeAttribute(AttributeProto attribute,
                                                           NodeAttributes& node_attributes) {
  ORT_ENFORCE(utils::HasName(attribute), "AttributeProto must have a name.");
  std::string name = attribute.name();
  return node_attributes.insert_or_assign(std::move(name), std::move(attribute));
}

}  // namespace onnxruntime::utils
