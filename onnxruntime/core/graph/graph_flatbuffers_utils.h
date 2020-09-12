// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <core/graph/graph.h>

namespace onnxruntime {
namespace experimental {
namespace utils {

onnxruntime::common::Status SaveValueInfoOrtFormat(
    flatbuffers::FlatBufferBuilder& builder, const ONNX_NAMESPACE::ValueInfoProto& value_info_proto,
    flatbuffers::Offset<fbs::ValueInfo>& fbs_value_info) ORT_MUST_USE_RESULT;

onnxruntime::common::Status SaveInitializerOrtFormat(
    flatbuffers::FlatBufferBuilder& builder, const ONNX_NAMESPACE::TensorProto& initializer,
    flatbuffers::Offset<fbs::Tensor>& fbs_tensor) ORT_MUST_USE_RESULT;

// Convert a given AttributeProto into fbs::Attribute
// Note, we current do not support graphs, and sparse_tensor(s)
//       If the attribute type is a graph, we need to use the supplied graph,
//       instead of the GraphProto in attr_proto
onnxruntime::common::Status SaveAttributeOrtFormat(
    flatbuffers::FlatBufferBuilder& builder, const ONNX_NAMESPACE::AttributeProto& attr_proto,
    flatbuffers::Offset<fbs::Attribute>& fbs_attr, const onnxruntime::Graph* graph) ORT_MUST_USE_RESULT;

#if defined(ENABLE_ORT_FORMAT_LOAD)
inline void LoadStringFromOrtFormat(std::string& dst, const flatbuffers::String* fbs_string) {
  if (fbs_string)
    dst = fbs_string->c_str();
}

onnxruntime::common::Status LoadInitializerOrtFormat(
    const fbs::Tensor& fbs_tensor, ONNX_NAMESPACE::TensorProto& initializer) ORT_MUST_USE_RESULT;

onnxruntime::common::Status LoadValueInfoOrtFormat(
    const fbs::ValueInfo& fbs_value_info, ONNX_NAMESPACE::ValueInfoProto& value_info_proto) ORT_MUST_USE_RESULT;

// Load a give fbs::Attribute into AttributeProto
// Note, If the attribute type is a graph, we will leave an empty graph in attr_proto,
//       and set the deserialized Graph to the param graph
onnxruntime::common::Status LoadAttributeOrtFormat(const fbs::Attribute& fbs_attr,
                                                   ONNX_NAMESPACE::AttributeProto& attr_proto,
                                                   std::unique_ptr<onnxruntime::Graph>& sub_graph,
                                                   Graph& graph, Node& node,
                                                   const logging::Logger& logger) ORT_MUST_USE_RESULT;

#endif

}  // namespace utils
}  // namespace experimental
}  // namespace onnxruntime
