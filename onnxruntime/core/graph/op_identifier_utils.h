// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/op_identifier.h"

#include "core/common/status.h"
#include "core/graph/graph.h"

#if !defined(ORT_MINIMAL_BUILD)

#include "onnx/defs/schema.h"  // for ONNX_NAMESPACE::OpSchema

#endif  // !defined(ORT_MINIMAL_BUILD)

namespace flatbuffers {
class FlatBufferBuilder;

template <typename T>
struct Offset;

struct String;
}  // namespace flatbuffers

namespace onnxruntime {

namespace fbs::utils {

#if !defined(ORT_MINIMAL_BUILD)

Status SaveOpIdentifierOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                 const onnxruntime::OpIdentifier& op_id,
                                 flatbuffers::Offset<flatbuffers::String>& fbs_op_id_str);

#endif  // !defined(ORT_MINIMAL_BUILD)

Status LoadOpIdentifierOrtFormat(const flatbuffers::String& fbs_op_id_str,
                                 onnxruntime::OpIdentifier& op_id);

}  // namespace fbs::utils

namespace utils {

inline onnxruntime::OpIdentifier MakeOpId(const Node& node) {
  return onnxruntime::OpIdentifier{node.Domain(), node.OpType(), node.SinceVersion()};
}

#if !defined(ORT_MINIMAL_BUILD)

inline onnxruntime::OpIdentifier MakeOpId(const ONNX_NAMESPACE::OpSchema& op_schema) {
  return onnxruntime::OpIdentifier{op_schema.domain(), op_schema.Name(), op_schema.SinceVersion()};
}

#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace utils

}  // namespace onnxruntime
