// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cmath>
#include <filesystem>
#include <ostream>
#include <string>
#include <tuple>
#include <vector>

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

  // Optional slice spec keys in TensorProto::external_data. Adding a slice spec to an
  // initializer is purely additive: models that do not set any of these keys behave
  // exactly as before (no slicing is performed).
  //
  //   "source_shape" - comma-separated dims of the full (un-sliced) tensor stored in
  //                    the external file. Presence of this key enables slicing.
  //                    No default. (e.g. "2048,8192")
  //   "slice_starts" - comma-separated per-dim start indices (in elements).
  //                    Same rank as source_shape.
  //                    DEFAULT (key absent): all zeros — slice begins at the origin.
  //   "slice_sizes"  - comma-separated per-dim slice sizes (in elements).
  //                    Same rank as source_shape. Must equal TensorProto.dims.
  //                    DEFAULT (key absent): source_shape[d] - slice_starts[d] for
  //                    each d — slice extends to the end of each source dim.
  //
  // When source_shape is present, `offset` is the byte offset of the START of the
  // source tensor within the file, and `length` (if present) must equal the byte
  // size of the SOURCE tensor (not the slice).
  static constexpr const char* kSourceShapeKey = "source_shape";
  static constexpr const char* kSliceStartsKey = "slice_starts";
  static constexpr const char* kSliceSizesKey = "slice_sizes";

  ExternalDataInfo();

#if !defined(ORT_MINIMAL_BUILD)
  ExternalDataInfo(const PathString& rel_path, OFFSET_TYPE offset, size_t length);
#endif

  const PathString& GetRelPath() const { return rel_path_; }

  OFFSET_TYPE GetOffset() const { return offset_; }
  size_t GetLength() const { return length_; }

  const std::string& GetChecksum() const { return checksum_; }

  // True if this entry declares a multi-dimensional slice view of a larger
  // tensor stored in the external file.
  bool HasSliceSpec() const noexcept { return !source_shape_.empty(); }

  const std::vector<int64_t>& GetSourceShape() const noexcept { return source_shape_; }
  const std::vector<int64_t>& GetSliceStarts() const noexcept { return slice_starts_; }
  const std::vector<int64_t>& GetSliceSizes() const noexcept { return slice_sizes_; }

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
      bool align, int64_t align_threshold, int64_t on_disk_alignment,
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

  // Multi-dim slice spec (optional). Empty source_shape_ means no slicing.
  std::vector<int64_t> source_shape_;
  std::vector<int64_t> slice_starts_;
  std::vector<int64_t> slice_sizes_;

  // Pre-packed blobs found associated with this TensorProto if present
  // format key, offset, length, checksum
  PrepackedInfos prepacked_infos_;
};
}  // namespace onnxruntime
