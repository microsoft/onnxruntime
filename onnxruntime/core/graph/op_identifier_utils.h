// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/op_identifier.h"

#include "core/common/status.h"

namespace flatbuffers {
class FlatBufferBuilder;

template <typename T>
struct Offset;
}  // namespace flatbuffers

namespace onnxruntime {

using common::Status;

namespace fbs {

struct OpIdentifier;

namespace utils {

#if !defined(ORT_MINIMAL_BUILD)

Status SaveOpIdentifierOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                                 const onnxruntime::OpIdentifier& op_id,
                                 flatbuffers::Offset<fbs::OpIdentifier>& fbs_op_id);

#endif  // !defined(ORT_MINIMAL_BUILD)

Status LoadOpIdentifierOrtFormat(const fbs::OpIdentifier& fbs_op_id,
                                 onnxruntime::OpIdentifier& op_id);

}  // namespace utils
}  // namespace fbs
}  // namespace onnxruntime
