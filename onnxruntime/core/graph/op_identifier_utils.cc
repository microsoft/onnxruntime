// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/op_identifier_utils.h"

#include "core/flatbuffers/flatbuffers_utils.h"
#include "core/flatbuffers/schema/ort.fbs.h"

namespace onnxruntime::fbs::utils {

#if !defined(ORT_MINIMAL_BUILD)

Status SaveOpIdentifierOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                 const onnxruntime::OpIdentifier& op_id,
                                 flatbuffers::Offset<flatbuffers::String>& fbs_op_id_str) {
  const auto op_id_str = op_id.ToString();
  fbs_op_id_str = builder.CreateSharedString(op_id_str);
  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

Status LoadOpIdentifierOrtFormat(const flatbuffers::String& fbs_op_id_str,
                                 onnxruntime::OpIdentifier& op_id) {
  ORT_RETURN_IF_ERROR(onnxruntime::OpIdentifier::LoadFromString(fbs_op_id_str.string_view(), op_id));
  return Status::OK();
}

}  // namespace onnxruntime::fbs::utils
