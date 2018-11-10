// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <unordered_map>
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "onnx/defs/schema.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#include "core/common/status.h"
#include "core/graph/constants.h"

namespace onnxruntime {
using AttrType = ONNX_NAMESPACE::AttributeProto_AttributeType;
using NodeAttributes = std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto>;

// This string array should exactly match the AttrType defined above.
/*
AttributeProto_AttributeType_UNDEFINED = 0,
AttributeProto_AttributeType_FLOAT = 1,
AttributeProto_AttributeType_INT = 2,
AttributeProto_AttributeType_STRING = 3,
AttributeProto_AttributeType_TENSOR = 4,
AttributeProto_AttributeType_GRAPH = 5,
AttributeProto_AttributeType_FLOATS = 6,
AttributeProto_AttributeType_INTS = 7,
AttributeProto_AttributeType_STRINGS = 8,
AttributeProto_AttributeType_TENSORS = 9,
AttributeProto_AttributeType_GRAPHS = 10
*/
static constexpr const char* kAttrTypeStrings[] =
    {
        "UNDEFINED",
        "FLOAT",
        "INT",
        "STRING",
        "TENSOR",
        "GRAPH",
        "FLOATS",
        "INTS",
        "STRINGS",
        "TENSORS",
        "GRAPHS"};

class TypeUtils {
 public:
  // Get attribute type given attribute proto data.
  static ::onnxruntime::common::Status GetType(const ONNX_NAMESPACE::AttributeProto& attr, AttrType& type);
  static bool IsValidAttribute(const ONNX_NAMESPACE::AttributeProto& attribute);
};
}  // namespace onnxruntime
