// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cmath>
#include <filesystem>
#include <ostream>
#include <string>
#include <tuple>

#include <core/common/inlined_containers_fwd.h>
#include "core/common/path_string.h"
#include "core/common/safeint.h"
#include "core/common/status.h"
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

  static common::Status Create(
      const ::google::protobuf::RepeatedPtrField<::ONNX_NAMESPACE::StringStringEntryProto>& input,
      std::unique_ptr<ExternalDataInfo>& out);

  static void SetExternalLocationToProto(const std::filesystem::path& external_file_path,
                                         int64_t offset,
                                         size_t tensor_bytes_size,
                                         ::ONNX_NAMESPACE::TensorProto& proto);

  // Pads the output with zeros according to the specified alignment_factor
  // It updates external_offset for alignment.
  // need to do padding before write actual tensor data as we do offset alignment at the begin of
  // large tensors (offset need to be page aligned) like below:
  // \242\2557\256\023.\031&0000000000000000\332)k+\253\246\342\246(&\006!\347\232\374\236\325\026\032+\36XXXX
  // |<---smaller tensor---->|<---padding--->|<------------------large tensor----------------------------->|
  static std::ostream& AlignAndPad(std::ostream& stream, int64_t alignment_factor, int64_t& external_offset) {
    // Align to the next page or alloc granularity boundary
    SafeInt<int64_t> safe_external_offset = external_offset;
    int64_t new_external_offset = ((safe_external_offset + alignment_factor - 1) / alignment_factor) *
                                  alignment_factor;

    // padding tensor with zeros for alignment
    for (int64_t index = external_offset; index != new_external_offset; ++index) {
      stream << '\0';
    }
    external_offset = new_external_offset;
    return stream;
  }

  static std::ostream& WritePrepackedToFileAndAddToProto(
      const PrepackedWeightsForGraph& prepacked_for_graph,
      const InlinedHashSet<std::string>& blob_keys,
      bool align, int64_t align_threshold, int64_t on_disk_alignment_factor,
      std::ostream& os,
      int64_t& external_offset,
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
