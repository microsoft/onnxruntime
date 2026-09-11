// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/schema_abi_digest.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>

#include "core/common/sha256.h"
#include "core/graph/onnx_protobuf.h"
#include "onnx/defs/schema.h"

namespace onnxruntime {
namespace {

class CanonicalWriter {
 public:
  void AddUint64(uint64_t value) {
    for (size_t i = 0; i < sizeof(value); ++i) {
      bytes_.push_back(static_cast<char>((value >> (i * 8)) & 0xff));
    }
  }

  void AddBool(bool value) { AddUint64(value ? 1 : 0); }

  void AddString(std::string_view value) {
    AddUint64(value.size());
    bytes_.append(value.data(), value.size());
  }

  const std::string& Bytes() const { return bytes_; }

 private:
  std::string bytes_;
};

std::string NormalizeTypeString(std::string_view value) {
  std::string normalized;
  normalized.reserve(value.size());
  for (char c : value) {
    const bool is_ascii_whitespace = c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
    if (!is_ascii_whitespace) {
      normalized.push_back(static_cast<char>(c));
    }
  }
  return normalized;
}

Status SerializeDeterministically(const google::protobuf::MessageLite& message,
                                  std::string& serialized) {
  serialized.clear();
  {
    google::protobuf::io::StringOutputStream string_stream(&serialized);
    google::protobuf::io::CodedOutputStream coded_stream(&string_stream);
    coded_stream.SetSerializationDeterministic(true);
    if (!message.SerializeToCodedStream(&coded_stream)) {
      return Status(common::ONNXRUNTIME, common::FAIL,
                    "Failed to serialize an operator schema attribute default value.");
    }
  }
  return Status::OK();
}

void AddFormalParameters(CanonicalWriter& writer,
                         const std::vector<ONNX_NAMESPACE::OpSchema::FormalParameter>& parameters) {
  writer.AddUint64(parameters.size());
  for (const auto& parameter : parameters) {
    writer.AddString(parameter.GetName());
    writer.AddString(NormalizeTypeString(parameter.GetTypeStr()));
    writer.AddUint64(static_cast<uint64_t>(parameter.GetOption()));
    writer.AddBool(parameter.GetIsHomogeneous());
    writer.AddUint64(static_cast<uint64_t>(parameter.GetMinArity()));
    writer.AddUint64(static_cast<uint64_t>(parameter.GetDifferentiationCategory()));
  }
}

}  // namespace

Status ComputeSchemaAbiDigest(const ONNX_NAMESPACE::OpSchema& schema,
                              SchemaAbiDigest& digest) {
  CanonicalWriter writer;
  writer.AddString("ort.schema_abi.v1");
  writer.AddString(schema.domain());
  writer.AddString(schema.Name());
  writer.AddUint64(static_cast<uint64_t>(schema.since_version()));
  writer.AddBool(schema.deprecated());
  writer.AddUint64(static_cast<uint64_t>(schema.min_input()));
  writer.AddUint64(static_cast<uint64_t>(schema.max_input()));
  writer.AddUint64(static_cast<uint64_t>(schema.min_output()));
  writer.AddUint64(static_cast<uint64_t>(schema.max_output()));

  AddFormalParameters(writer, schema.inputs());
  AddFormalParameters(writer, schema.outputs());

  std::vector<const ONNX_NAMESPACE::OpSchema::TypeConstraintParam*> type_constraints;
  type_constraints.reserve(schema.typeConstraintParams().size());
  for (const auto& type_constraint : schema.typeConstraintParams()) {
    type_constraints.push_back(&type_constraint);
  }
  std::sort(type_constraints.begin(), type_constraints.end(), [](const auto* lhs, const auto* rhs) {
    return lhs->type_param_str < rhs->type_param_str;
  });

  writer.AddUint64(type_constraints.size());
  for (const auto* type_constraint : type_constraints) {
    writer.AddString(type_constraint->type_param_str);
    std::vector<std::string> allowed_types;
    allowed_types.reserve(type_constraint->allowed_type_strs.size());
    for (const auto& allowed_type : type_constraint->allowed_type_strs) {
      allowed_types.push_back(NormalizeTypeString(allowed_type));
    }
    std::sort(allowed_types.begin(), allowed_types.end());
    writer.AddUint64(allowed_types.size());
    for (const auto& allowed_type : allowed_types) {
      writer.AddString(allowed_type);
    }
  }

  std::vector<const ONNX_NAMESPACE::OpSchema::Attribute*> attributes;
  attributes.reserve(schema.attributes().size());
  for (const auto& attribute_entry : schema.attributes()) {
    attributes.push_back(&attribute_entry.second);
  }
  std::sort(attributes.begin(), attributes.end(), [](const auto* lhs, const auto* rhs) {
    return lhs->name < rhs->name;
  });

  writer.AddUint64(attributes.size());
  for (const auto* attribute : attributes) {
    writer.AddString(attribute->name);
    writer.AddUint64(static_cast<uint64_t>(attribute->type));
    writer.AddBool(attribute->required);

    const bool has_default = attribute->default_value.ByteSizeLong() != 0;
    writer.AddBool(has_default);
    if (has_default) {
      auto default_value = attribute->default_value;
      default_value.clear_doc_string();
      std::string serialized_default;
      auto status = SerializeDeterministically(default_value, serialized_default);
      if (!status.IsOK()) {
        return status;
      }
      writer.AddString(serialized_default);
    }
  }

  Sha256 sha256;
  sha256.Update(writer.Bytes().data(), writer.Bytes().size());
  sha256.Final(digest.data());
  return Status::OK();
}

}  // namespace onnxruntime
