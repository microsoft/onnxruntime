// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include <gsl/gsl>

#include "onnx/onnx_pb.h"

#include "core/graph/basic_types.h"

namespace onnxruntime::utils {

/**
 * Creates an AttributeProto with the specified name and value(s).
 */
ONNX_NAMESPACE::AttributeProto MakeAttribute(std::string attr_name, int64_t value);
ONNX_NAMESPACE::AttributeProto MakeAttribute(std::string attr_name, gsl::span<const int64_t> values);

#define DECLARE_MAKE_ATTRIBUTE_FNS(type)                                           \
  ONNX_NAMESPACE::AttributeProto MakeAttribute(std::string attr_name, type value); \
  ONNX_NAMESPACE::AttributeProto MakeAttribute(std::string attr_name, gsl::span<const type> values)

DECLARE_MAKE_ATTRIBUTE_FNS(float);
DECLARE_MAKE_ATTRIBUTE_FNS(std::string);
DECLARE_MAKE_ATTRIBUTE_FNS(ONNX_NAMESPACE::TensorProto);
#if !defined(DISABLE_SPARSE_TENSORS)
DECLARE_MAKE_ATTRIBUTE_FNS(ONNX_NAMESPACE::SparseTensorProto);
#endif
DECLARE_MAKE_ATTRIBUTE_FNS(ONNX_NAMESPACE::TypeProto);
DECLARE_MAKE_ATTRIBUTE_FNS(ONNX_NAMESPACE::GraphProto);

#undef DECLARE_MAKE_ATTRIBUTE_FNS

// preferred overload for string literals
inline ONNX_NAMESPACE::AttributeProto MakeAttribute(std::string attr_name, const char* value) {
  return MakeAttribute(std::move(attr_name), std::string{value});
}

/**
 * Sets an attribute in `node_attributes` with key `attribute.name()` and value `attribute`.
 * @return True if a new attribute was added, false if an existing attribute was overwritten.
 */
bool SetNodeAttribute(ONNX_NAMESPACE::AttributeProto attribute, NodeAttributes& node_attributes);

}  // namespace onnxruntime::utils
