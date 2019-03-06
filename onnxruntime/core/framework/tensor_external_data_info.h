// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include "core/common/status.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
class ExternalDataInfo {
 private:
  std::basic_string<ORTCHAR_T> rel_path_;
  //-1 means doesn't exist
  ptrdiff_t offset_;
  //-1 means doesn't exist
  ptrdiff_t length_;
  std::string checksum_;

 public:
  const std::basic_string<ORTCHAR_T>& GetRelPath() const { return rel_path_; }

  ptrdiff_t GetOffset() const { return offset_; }
  ptrdiff_t GetLength() const { return length_; }

  const std::string& GetChecksum() const { return checksum_; }

  // If the value of 'offset' or 'length' field is larger the max value of ssize_t, this function will treat it as a
  // wrong value and return FAIL.
  static common::Status Create(const ::google::protobuf::RepeatedPtrField<::ONNX_NAMESPACE::StringStringEntryProto>& input,
                               std::unique_ptr<ExternalDataInfo>& out);
};
}  // namespace onnxruntime