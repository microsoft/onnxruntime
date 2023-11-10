// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstring>
#include "core/graph/constants.h"
#include "core/graph/op.h"

using ONNX_NAMESPACE::AttributeProto;


using namespace ::onnxruntime::common;
namespace onnxruntime {

bool TypeUtils::IsValidAttribute(const AttributeProto& attr) {
  using namespace ONNX_NAMESPACE;
  if (attr.name().empty()) {
    return false;
  }

  if (attr.type() == AttributeProto_AttributeType_UNDEFINED) {
    const int num_fields =
        attr.has_f() +
        attr.has_i() +
        attr.has_s() +
        attr.has_t() +
        attr.has_g() +
        (attr.floats_size() > 0) +
        (attr.ints_size() > 0) +
        (attr.strings_size() > 0) +
        (attr.tensors_size() > 0) +
        (attr.graphs_size() > 0);

    if (num_fields != 1) {
      return false;
    }
  }
  return true;
}

Status TypeUtils::GetType(const AttributeProto& attr, AttrType& type) {
  if (!TypeUtils::IsValidAttribute(attr)) {
    return Status(ONNXRUNTIME, onnxruntime::common::StatusCode::FAIL, "Invalid AttributeProto.");
  }

  type = attr.type();
  if (AttrType::AttributeProto_AttributeType_UNDEFINED == type) {
    if (attr.has_f()) {
      type = AttrType::AttributeProto_AttributeType_FLOAT;
    } else if (attr.has_i()) {
      type = AttrType::AttributeProto_AttributeType_INT;
    } else if (attr.has_s()) {
      type = AttrType::AttributeProto_AttributeType_STRING;
    } else if (attr.has_t()) {
      type = AttrType::AttributeProto_AttributeType_TENSOR;
    } else if (attr.has_g()) {
      type = AttrType::AttributeProto_AttributeType_GRAPH;
    } else if (attr.has_sparse_tensor()) {
      type = AttrType::AttributeProto_AttributeType_SPARSE_TENSOR;
    } else if (attr.floats_size()) {
      type = AttrType::AttributeProto_AttributeType_FLOATS;
    } else if (attr.ints_size()) {
      type = AttrType::AttributeProto_AttributeType_INTS;
    } else if (attr.strings_size()) {
      type = AttrType::AttributeProto_AttributeType_STRINGS;
    } else if (attr.tensors_size()) {
      type = AttrType::AttributeProto_AttributeType_TENSORS;
    } else if (attr.graphs_size()) {
      type = AttrType::AttributeProto_AttributeType_GRAPHS;
    } else if (attr.sparse_tensors_size()) {
      type = AttrType::AttributeProto_AttributeType_SPARSE_TENSORS;
    } else {
      return Status(ONNXRUNTIME, onnxruntime::common::StatusCode::FAIL, "Invalid AttributeProto.");
    }
  }
  return Status::OK();
}
}  // namespace onnxruntime
