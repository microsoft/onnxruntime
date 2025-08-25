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
Status ExternalDataInfo::Create(const RepeatedPtrField<StringStringEntryProto>& input,
                                std::unique_ptr<ExternalDataInfo>& external_data_info_result) {
  auto external_data_info = std::make_unique<ExternalDataInfo>();
  PrepackedInfos prepacked_infos;

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
    int64_t align_threshold, int64_t on_disk_alignment_factor,
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
        if (!AlignAndPad(os, on_disk_alignment_factor, external_offset)) {
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