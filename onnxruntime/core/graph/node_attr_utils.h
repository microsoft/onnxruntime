// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "core/common/gsl.h"

#include "onnx/onnx_pb.h"

#include "core/graph/basic_types.h"

namespace onnxruntime::utils {

// keep these signatures in sync with DECLARE_MAKE_ATTRIBUTE_FNS below
/** Creates an AttributeProto with the specified name and value. */
ONNX_NAMESPACE::AttributeProto MakeAttribute(std::string attr_name, int64_t value);
/** Creates an AttributeProto with the specified name and values. */
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

// The below overload is made so the compiler does not attempt to resolve
// string literals with the gsl::span overload
inline ONNX_NAMESPACE::AttributeProto MakeAttribute(std::string attr_name, const char* value) {
  return MakeAttribute(std::move(attr_name), std::string{value});
}

/**
 * Sets an attribute in `node_attributes` with key `attribute.name()` and value `attribute`.
 * If an attribute with the same name exists, it will be overwritten.
 * @return Pair of (iterator to attribute, whether attribute was added (true) or updated (false)).
 */
std::pair<NodeAttributes::iterator, bool> SetNodeAttribute(ONNX_NAMESPACE::AttributeProto attribute,
                                                           NodeAttributes& node_attributes);

}  // namespace onnxruntime::utils
