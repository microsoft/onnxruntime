// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace ONNX_NAMESPACE {
class TensorProto;
class SparseTensorProto;
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
class Path;

namespace logging {
class Logger;
}

namespace experimental {

namespace fbs {
struct Attribute;
struct Tensor;
}  // namespace fbs

namespace utils {

// TODO, add ORT_MUST_USE_RESULT when it is moved to a different header
onnxruntime::common::Status SaveInitializerOrtFormat(
    flatbuffers::FlatBufferBuilder& builder, const ONNX_NAMESPACE::TensorProto& initializer,
    const Path& model_path, flatbuffers::Offset<fbs::Tensor>& fbs_tensor);

onnxruntime::common::Status SaveSparseInitializerOrtFormat(
    flatbuffers::FlatBufferBuilder& builder, const ONNX_NAMESPACE::SparseTensorProto& initializer,
    const Path& model_path, flatbuffers::Offset<fbs::SparseTensor>& fbs_sparse_tensor);

// Convert a given AttributeProto into fbs::Attribute
// Note, we current do not support graphs, and sparse_tensor(s)
//       If the attribute type is a graph, we need to use the supplied Graph instance,
//       instead of the GraphProto in attr_proto
onnxruntime::common::Status SaveAttributeOrtFormat(
    flatbuffers::FlatBufferBuilder& builder, const ONNX_NAMESPACE::AttributeProto& attr_proto,
    flatbuffers::Offset<fbs::Attribute>& fbs_attr, const Path& model_path,
    const onnxruntime::Graph* subgraph);

#if defined(ENABLE_ORT_FORMAT_LOAD)

onnxruntime::common::Status LoadInitializerOrtFormat(
    const fbs::Tensor& fbs_tensor, ONNX_NAMESPACE::TensorProto& initializer);

onnxruntime::common::Status LoadSparseInitializerOrtFormat(const fbs::SparseTensor& fbs_sparse_tensor,
                                                           ONNX_NAMESPACE::SparseTensorProto& initializer);

// Load a give fbs::Attribute into AttributeProto
// Note, If the attribute type is a graph, we will leave an empty graph in attr_proto,
//       and set the deserialized Graph to the param graph
onnxruntime::common::Status LoadAttributeOrtFormat(const fbs::Attribute& fbs_attr,
                                                   ONNX_NAMESPACE::AttributeProto& attr_proto,
                                                   std::unique_ptr<onnxruntime::Graph>& sub_graph,
                                                   Graph& graph, Node& node,
                                                   const logging::Logger& logger);

#endif

}  // namespace utils
}  // namespace experimental
}  // namespace onnxruntime
