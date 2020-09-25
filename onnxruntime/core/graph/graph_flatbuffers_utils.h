// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace ONNX_NAMESPACE {
class TensorProto;
class AttributeProto;
}  // namespace ONNX_NAMESPACE

namespace flatbuffers {
class FlatBufferBuilder;
template <typename T>
struct Offset;
}  // namespace flatbuffers

namespace onnxruntime {

class Graph;
class Node;

namespace logging {
class Logger;
}

namespace experimental {

namespace fbs {
struct Attribute;
struct Tensor;
}  // namespace fbs

namespace utils {

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

onnxruntime::common::Status LoadInitializerOrtFormat(
    const fbs::Tensor& fbs_tensor, ONNX_NAMESPACE::TensorProto& initializer) ORT_MUST_USE_RESULT;

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
