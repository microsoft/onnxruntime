// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tensor_external_data_info.h"
#include "core/common/common.h"
#include "core/common/narrow.h"
#include "core/common/parse_string.h"
#include "core/common/safeint.h"
#include "core/common/string_utils.h"
#include "core/platform/path_lib.h"

#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif
using ::google::protobuf::RepeatedPtrField;
using ::ONNX_NAMESPACE::StringStringEntryProto;

namespace onnxruntime {
ExternalDataInfo::ExternalDataInfo() = default;

#if !defined(ORT_MINIMAL_BUILD)
ExternalDataInfo::ExternalDataInfo(const PathString& rel_path, OFFSET_TYPE offset, size_t length)
    : rel_path_(rel_path), offset_(offset), length_(length) {}
#endif

Status ExternalDataInfo::Create(const RepeatedPtrField<StringStringEntryProto>& input,
                                std::unique_ptr<ExternalDataInfo>& external_data_info_result) {
  auto external_data_info = std::make_unique<ExternalDataInfo>();
  PrepackedInfos prepacked_infos;

  // Helper to parse a comma-separated list of int64 values, e.g. "2048,4096,512".
  // Trims surrounding whitespace; rejects empty fields.
  auto parse_int64_list = [](const std::string& key,
                             const std::string& value,
                             std::vector<int64_t>& out) -> Status {
    out.clear();
    auto fields = utils::SplitString(value, ",", false);
    out.reserve(fields.size());
    for (const auto& f : fields) {
      // utils::SplitString returns string_views with no trim. Reject empty/whitespace fields
      // to fail loudly on typos like "1,,2" or "1, ,2".
      std::string_view sv(f);
      while (!sv.empty() && (sv.front() == ' ' || sv.front() == '\t')) sv.remove_prefix(1);
      while (!sv.empty() && (sv.back() == ' ' || sv.back() == '\t')) sv.remove_suffix(1);
      if (sv.empty()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "external_data ", key,
                               " contains empty value: \"", value, "\"");
      }
      int64_t v;
      ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(std::string(sv), v));
      out.push_back(v);
    }
    if (out.empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "external_data ", key, " must not be empty");
    }
    return Status::OK();
  };

  const int input_size = input.size();

  for (int i = 0; i != input_size; ++i) {
    StringStringEntryProto stringmap = input[i];
    if (!stringmap.has_key())
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model format error! Need a key for the external data info");
    if (!stringmap.has_value())
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model format error! Need a value for the external data info");

    if (stringmap.key() == "location" && !stringmap.value().empty()) {
      external_data_info->rel_path_ = ToWideString(stringmap.value());
    } else if (stringmap.key() == "offset" && !stringmap.value().empty()) {
      ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(stringmap.value(), external_data_info->offset_));
    } else if (stringmap.key() == "length" && !stringmap.value().empty()) {
      ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(stringmap.value(), external_data_info->length_));
    } else if (stringmap.key() == "checksum" && !stringmap.value().empty()) {
      external_data_info->checksum_ = stringmap.value();
    } else if (stringmap.key() == kSourceShapeKey && !stringmap.value().empty()) {
      ORT_RETURN_IF_ERROR(parse_int64_list(kSourceShapeKey, stringmap.value(),
                                           external_data_info->source_shape_));
    } else if (stringmap.key() == kSliceStartsKey && !stringmap.value().empty()) {
      ORT_RETURN_IF_ERROR(parse_int64_list(kSliceStartsKey, stringmap.value(),
                                           external_data_info->slice_starts_));
    } else if (stringmap.key() == kSliceSizesKey && !stringmap.value().empty()) {
      ORT_RETURN_IF_ERROR(parse_int64_list(kSliceSizesKey, stringmap.value(),
                                           external_data_info->slice_sizes_));
    } else if (stringmap.key().find("prepacked", 0) == 0) {
      // Starts with 'prepacked', each has its own key.
      // Each prepacked entry may have multiple blobs with the same key
      // we output them with the same key
      // format = key|offset;length;checksum[|offset;length;checksum]
      // We are ignoring invalid entries (should not be any), and rely
      // on in memory pre-packs regenerated in this case.
      // users can over-write this file with the correct pre-packed info.
      const std::string& prepacked = stringmap.value();
      if (!prepacked.empty()) {
        auto split_fields = utils::SplitString(prepacked, "|", false);
        if (split_fields.size() > 1) {
          const std::string key{split_fields[0]};
          auto& blob_infos = prepacked_infos[key];
          for (size_t f = 1; f < split_fields.size(); ++f) {
            const auto& blob = split_fields[f];
            auto blob_fields = utils::SplitString(blob, ";", false);
            if (blob_fields.size() == 3) {
              OFFSET_TYPE offset;
              size_t len;
              ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(blob_fields[0], offset));
              ORT_RETURN_IF_ERROR(ParseStringWithClassicLocale(blob_fields[1], len));
              blob_infos.push_back(std::make_tuple(offset, len, std::string(blob_fields[2])));
            }
          }
          if (blob_infos.empty()) {
            prepacked_infos.erase(key);
          }
        }
      }
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model format error!");
    }
  }

  if (external_data_info->rel_path_.empty()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "model format error! Missing 'location'");
  }

  // Slice spec validation. `source_shape` gates the feature: when present, the entry
  // describes a slice of a larger tensor on disk. The other two keys are optional with
  // sensible defaults so that authors only need to specify what differs from the default:
  //   - slice_starts defaults to all-zeros (slice begins at the source origin)
  //   - slice_sizes  defaults to source_shape[d] - slice_starts[d]
  //                  (slice extends to the end of each source dim)
  // Specifying neither => the slice covers the entire source tensor (a no-op slice that
  // is equivalent to omitting the slice spec, but accepted for self-describing models).
  if (!external_data_info->source_shape_.empty()) {
    const auto rank = external_data_info->source_shape_.size();

    if (external_data_info->slice_starts_.empty()) {
      external_data_info->slice_starts_.assign(rank, 0);
    } else if (external_data_info->slice_starts_.size() != rank) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "external_data '", kSliceStartsKey, "' rank ",
                             external_data_info->slice_starts_.size(),
                             " does not match '", kSourceShapeKey, "' rank ", rank);
    }

    if (external_data_info->slice_sizes_.empty()) {
      // Default: from slice_starts to the end of each dim.
      external_data_info->slice_sizes_.resize(rank);
      for (size_t d = 0; d < rank; ++d) {
        external_data_info->slice_sizes_[d] =
            external_data_info->source_shape_[d] - external_data_info->slice_starts_[d];
      }
    } else if (external_data_info->slice_sizes_.size() != rank) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "external_data '", kSliceSizesKey, "' rank ",
                             external_data_info->slice_sizes_.size(),
                             " does not match '", kSourceShapeKey, "' rank ", rank);
    }

    for (size_t d = 0; d < rank; ++d) {
      const int64_t src = external_data_info->source_shape_[d];
      const int64_t start = external_data_info->slice_starts_[d];
      const int64_t size = external_data_info->slice_sizes_[d];
      if (src <= 0) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "external_data '", kSourceShapeKey, "' dim ", d,
                               " must be > 0, got ", src);
      }
      if (start < 0 || size < 0 || start > src || size > src || start + size > src) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "external_data slice out of bounds at dim ", d,
                               ": source_shape=", src,
                               ", slice_start=", start, ", slice_size=", size);
      }
    }
  } else {
    // No source_shape but slice_starts or slice_sizes present is a typo guard.
    // (Note: the pre-existing parser also rejected unknown keys, so this is not a regression.)
    if (!external_data_info->slice_starts_.empty() || !external_data_info->slice_sizes_.empty()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "external_data has '", kSliceStartsKey, "' or '", kSliceSizesKey,
                             "' without '", kSourceShapeKey, "'");
    }
  }

  if (!prepacked_infos.empty()) {
    external_data_info->prepacked_infos_ = std::move(prepacked_infos);
  }

  external_data_info_result = std::move(external_data_info);
  return Status::OK();
}
void ExternalDataInfo::SetExternalLocationToProto(const std::filesystem::path& external_file_path,
                                                  int64_t external_offset, size_t tensor_bytes_size,
                                                  ::ONNX_NAMESPACE::TensorProto& proto) {
  proto.set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation::TensorProto_DataLocation_EXTERNAL);

  auto* location = proto.add_external_data();
  location->set_key("location");
  location->set_value(ToUTF8String(external_file_path.native()));

  auto* offset = proto.add_external_data();
  offset->set_key("offset");
  offset->set_value(std::to_string(external_offset));

  auto* length = proto.add_external_data();
  length->set_key("length");
  length->set_value(std::to_string(tensor_bytes_size));
}

std::ostream& ExternalDataInfo::WritePrepackedToFileAndAddToProto(
    const PrepackedWeightsForGraph& prepacked_for_graph,
    const InlinedHashSet<std::string>& blob_keys, bool align,
    int64_t align_threshold, int64_t on_disk_alignment,
    std::ostream& os, int64_t& external_offset, ::ONNX_NAMESPACE::TensorProto& proto) {
  size_t key_count = 0;
  for (const auto& key : blob_keys) {
    size_t prepack_count = 0;
    const auto* prepacked_weights = prepacked_for_graph.GetPrepackedWeights(key);
    ORT_ENFORCE(prepacked_weights != nullptr, "Prepacked weights not found for key ", key);
    std::stringstream prepacked_entry;
    prepacked_entry << key << "|";
    for (size_t i = 0, size = prepacked_weights->buffers_.size(); i < size; ++i) {
      const auto size_in_bytes = prepacked_weights->buffer_sizes_[i];
      if (align && static_cast<int64_t>(size_in_bytes) > align_threshold) {
        // return early on error
        if (!AlignAndPad(os, on_disk_alignment, external_offset)) {
          return os;
        }
      }
      if (prepack_count++ > 0) {
        prepacked_entry << "|";
      }
      // Checksum is currently not validated
      prepacked_entry << external_offset << ";" << size_in_bytes << ";0";
      if (!os.write(reinterpret_cast<const char*>(prepacked_weights->buffers_[i].get()), size_in_bytes)) {
        return os;
      }
      external_offset = SafeInt<int64_t>(external_offset) + size_in_bytes;
    }
    auto* prepacked = proto.add_external_data();
    std::string prepacked_key("prepacked_");
    prepacked_key.append(std::to_string(key_count++));
    prepacked->set_key(std::move(prepacked_key));
    prepacked->set_value(prepacked_entry.str());
  }
  return os;
}
}  // namespace onnxruntime