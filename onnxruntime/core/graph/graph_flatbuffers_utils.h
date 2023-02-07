// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/common/status.h"
#include "core/graph/ort_format_load_options.h"

namespace ONNX_NAMESPACE {
class AttributeProto;
class TensorProto;

#if !defined(DISABLE_SPARSE_TENSORS)
class SparseTensorProto;
#endif  // !defined(DISABLE_SPARSE_TENSORS)
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

namespace fbs {
struct Attribute;
struct Tensor;

#if !defined(DISABLE_SPARSE_TENSORS)
struct SparseTensor;
#endif  // !defined(DISABLE_SPARSE_TENSORS)

namespace utils {

Status SaveInitializerOrtFormat(
    flatbuffers::FlatBufferBuilder& builder, const ONNX_NAMESPACE::TensorProto& initializer,
    const Path& model_path, flatbuffers::Offset<fbs::Tensor>& fbs_tensor);

#if !defined(DISABLE_SPARSE_TENSORS)
Status SaveSparseInitializerOrtFormat(
    flatbuffers::FlatBufferBuilder& builder, const ONNX_NAMESPACE::SparseTensorProto& initializer,
    const Path& model_path, flatbuffers::Offset<fbs::SparseTensor>& fbs_sparse_tensor);
#endif  // !defined(DISABLE_SPARSE_TENSORS)

// Convert a given AttributeProto into fbs::Attribute
// Note, we current do not support graphs, and sparse_tensor(s)
//       If the attribute type is a graph, we need to use the supplied Graph instance,
//       instead of the GraphProto in attr_proto
Status SaveAttributeOrtFormat(
    flatbuffers::FlatBufferBuilder& builder, const ONNX_NAMESPACE::AttributeProto& attr_proto,
    flatbuffers::Offset<fbs::Attribute>& fbs_attr, const Path& model_path,
    const onnxruntime::Graph* subgraph);

/// <summary>
/// Load an initializer from an ORT format flatbuffer.
/// </summary>
/// <param name="fbs_tensor">Flatbuffer Tensor</param>
/// <param name="initializer">TensorProto to load data into</param>
/// <param name="load_options">ORT format load options</param>
/// <returns>Status</returns>
Status LoadInitializerOrtFormat(const fbs::Tensor& fbs_tensor,
                                ONNX_NAMESPACE::TensorProto& initializer,
                                const OrtFormatLoadOptions& load_options);

#if !defined(DISABLE_SPARSE_TENSORS)
Status LoadSparseInitializerOrtFormat(const fbs::SparseTensor& fbs_sparse_tensor,
                                      ONNX_NAMESPACE::SparseTensorProto& initializer,
                                      const OrtFormatLoadOptions& load_options);
#endif  // !defined(DISABLE_SPARSE_TENSORS)

// Load a give fbs::Attribute into AttributeProto
// Note, If the attribute type is a graph, we will leave an empty graph in attr_proto,
//       and set the deserialized Graph to the param graph
Status LoadAttributeOrtFormat(const fbs::Attribute& fbs_attr,
                              ONNX_NAMESPACE::AttributeProto& attr_proto,
                              std::unique_ptr<onnxruntime::Graph>& sub_graph,
                              onnxruntime::Graph& graph, onnxruntime::Node& node,
                              const OrtFormatLoadOptions& load_options,
                              const logging::Logger& logger);

}  // namespace utils
}  // namespace fbs
}  // namespace onnxruntime
