// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/common/status.h>
#include <core/graph/basic_types.h>
#include <core/session/onnxruntime_c_api.h>

namespace flatbuffers {
class FlatBufferBuilder;

template <typename T>
struct Offset;

struct String;

template <typename T>
class Vector;
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
struct OperatorSetId;
struct Tensor;
struct ValueInfo;
}  // namespace fbs

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

void LoadStringFromOrtFormat(std::string& dst, const flatbuffers::String* fbs_string);

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

onnxruntime::common::Status LoadOpsetImportOrtFormat(
    const flatbuffers::Vector<flatbuffers::Offset<fbs::OperatorSetId>>* fbs_op_set_ids,
    std::unordered_map<std::string, int>& domain_to_version) ORT_MUST_USE_RESULT;

#endif

// check if filename ends in .ort
template <typename T>
bool IsOrtFormatModel(const std::basic_string<T>& filename) {
  auto len = filename.size();
  return len > 4 &&
         filename[len - 4] == '.' &&
         std::tolower(filename[len - 3]) == 'o' &&
         std::tolower(filename[len - 2]) == 'r' &&
         std::tolower(filename[len - 1]) == 't';
}

// check if bytes has the flatbuffer ORT identifier
bool IsOrtFormatModelBytes(const void* bytes, int num_bytes);

}  // namespace utils
}  // namespace experimental
}  // namespace onnxruntime
