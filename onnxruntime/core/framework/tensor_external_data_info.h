// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <filesystem>
#include <string>
#include <tuple>

#include "core/common/status.h"
#include "core/common/path_string.h"
#include "core/framework/prepacked_weights_container.h"
#include "core/graph/onnx_protobuf.h"

namespace onnxruntime {

class ExternalDataInfo {
 public:
#ifdef _WIN32
  using OFFSET_TYPE = int64_t;
#else
  using OFFSET_TYPE = off_t;
#endif

  const PathString& GetRelPath() const { return rel_path_; }

  OFFSET_TYPE GetOffset() const { return offset_; }
  size_t GetLength() const { return length_; }

  const std::string& GetChecksum() const { return checksum_; }

  // If the value of 'offset' or 'length' field is larger the max value of ssize_t, this function will treat it as a
  // wrong value and return FAIL.
  static common::Status Create(
      const ::google::protobuf::RepeatedPtrField<::ONNX_NAMESPACE::StringStringEntryProto>& input,
      std::unique_ptr<ExternalDataInfo>& out);

  static void SetExternalLocationToProto(const std::filesystem::path& external_file_path,
                                         int64_t offset,
                                         size_t tensor_bytes_size,
                                         ::ONNX_NAMESPACE::TensorProto& proto);

  static void AddPrepackedEntriesToProto(const PrepackedForSerialization::BlobsInderect& prepacked_for_write,
                                         ::ONNX_NAMESPACE::TensorProto& proto);

  using PrepackedInfo = std::tuple<OFFSET_TYPE, size_t, std::string>;
  using PrepackedInfos = std::unordered_map<std::string, std::vector<PrepackedInfo>>;

  bool HasPrepackedInfo() const noexcept { return !prepacked_infos_.empty(); }

  PrepackedInfos&& TakePrepackedInfos() { return std::move(prepacked_infos_); }

 private:
  PathString rel_path_;
  OFFSET_TYPE offset_ = 0;

  // 0 means the whole file
  size_t length_ = 0;
  std::string checksum_;

  // Pre-packed blobs found associated with this TensorProto if present
  // format key, offset, length, checksum
  PrepackedInfos prepacked_infos_;
};
}  // namespace onnxruntime
