// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>

#include "core/common/common.h"
#include "core/common/path_string.h"
#include "core/common/status.h"

namespace ONNX_NAMESPACE {
class ValueInfoProto;
}

namespace flatbuffers {
class FlatBufferBuilder;

template <typename T>
struct Offset;

struct String;

template <typename T>
class Vector;
}  // namespace flatbuffers

namespace onnxruntime {

namespace fbs {
struct OperatorSetId;
struct ValueInfo;

namespace utils {

constexpr auto kInvalidOrtFormatModelMessage = "Invalid ORT format model.";

// Will only create string in flatbuffers when has_string is true
flatbuffers::Offset<flatbuffers::String> SaveStringToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                                               bool has_string, const std::string& src);

onnxruntime::common::Status SaveValueInfoOrtFormat(
    flatbuffers::FlatBufferBuilder& builder, const ONNX_NAMESPACE::ValueInfoProto& value_info_proto,
    flatbuffers::Offset<fbs::ValueInfo>& fbs_value_info);

void LoadStringFromOrtFormat(std::string& dst, const flatbuffers::String* fbs_string);

// This macro is to be used on a protobuf message (protobuf_msg), which will not create an empty string field (str_field)
// if fbs_string is null
#define LOAD_STR_FROM_ORT_FORMAT(protobuf_msg, str_field, fbs_string) \
  {                                                                   \
    if (fbs_string)                                                   \
      protobuf_msg.set_##str_field(fbs_string->c_str());              \
  }

onnxruntime::common::Status LoadValueInfoOrtFormat(
    const fbs::ValueInfo& fbs_value_info, ONNX_NAMESPACE::ValueInfoProto& value_info_proto);

onnxruntime::common::Status LoadOpsetImportOrtFormat(
    const flatbuffers::Vector<flatbuffers::Offset<fbs::OperatorSetId>>* fbs_op_set_ids,
    std::unordered_map<std::string, int>& domain_to_version);

// check if filename ends in .ort
bool IsOrtFormatModel(const PathString& filename);

// check if bytes has the flatbuffer ORT identifier
bool IsOrtFormatModelBytes(const void* bytes, int num_bytes);

}  // namespace utils
}  // namespace fbs
}  // namespace onnxruntime

#define ORT_FORMAT_RETURN_IF_NULL(expr, expr_description)            \
  ORT_RETURN_IF((expr) == nullptr, (expr_description), " is null. ", \
                onnxruntime::fbs::utils::kInvalidOrtFormatModelMessage)
