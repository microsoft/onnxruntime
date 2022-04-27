// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/framework/checkpointing.h"

#include <algorithm>
#include <fstream>
#include <utility>
#include <vector>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/path.h"
#include "core/framework/data_transfer_utils.h"
#include "core/framework/endian_utils.h"
#include "core/framework/ort_value.h"
#include "core/framework/tensor_external_data_info.h"
#include "core/framework/tensorprotoutils.h"
#include "core/platform/env.h"
#include "core/platform/path_lib.h"
#include "orttraining/core/framework/protobuf_message_sequence.h"
#include "core/util/protobuf_parsing_utils.h"

namespace onnxruntime {
namespace training {

namespace {

constexpr const PathChar* k_tensors_file_name = ORT_TSTR("tensors.pbseq");
constexpr const PathChar* k_tensors_data_file_name = ORT_TSTR("tensors.bin");
constexpr const PathChar* k_properties_file_name = ORT_TSTR("properties.pbseq");

PathString GetCheckpointTensorsFilePath(const PathString& checkpoint_directory) {
  return ConcatPathComponent<PathChar>(checkpoint_directory, k_tensors_file_name);
}

PathString GetCheckpointTensorsDataFilePath(const PathString& checkpoint_directory) {
  return ConcatPathComponent<PathChar>(checkpoint_directory, k_tensors_data_file_name);
}

PathString GetCheckpointPropertiesFilePath(const PathString& checkpoint_directory) {
  return ConcatPathComponent<PathChar>(checkpoint_directory, k_properties_file_name);
}

std::vector<std::string> GetOrderedOrtValueNames(const NameMLValMap& name_to_value) {
  std::vector<std::string> ordered_names{};
  ordered_names.reserve(name_to_value.size());
  std::transform(
      name_to_value.begin(), name_to_value.end(), std::back_inserter(ordered_names),
      [](const NameMLValMap::value_type& v) { return v.first; });
  std::sort(ordered_names.begin(), ordered_names.end());
  return ordered_names;
}

}  // namespace

Status SaveRuntimeTensors(
    const PathString& tensors_path,
    const PathString& tensors_data_path,
    const DataTransferManager& data_transfer_manager,
    const NameMLValMap& ort_values,
    bool force_save_as_external_data) {
  // just write data file basename to TensorProto - this will get overwritten
  //   with the actual path when loading the checkpoint
  const PathString tensors_data_relative_path = GetLastComponent(tensors_data_path);
  const std::vector<std::string> ordered_tensor_names = GetOrderedOrtValueNames(ort_values);
  bool save_as_external_data = force_save_as_external_data;

  // Check whether there is potential protobuf size limit was hit.
  if (!save_as_external_data) {
    unsigned long total_bytes = 0;
    constexpr unsigned long PROTOBUF_UPPER_LIMIT = 2 * 1000 * 1000 * 1000;
    for (const auto& tensor_name : ordered_tensor_names) {
      const OrtValue& ort_value = ort_values.at(tensor_name);
      ORT_RETURN_IF_NOT(ort_value.IsTensor(), "ort_value.IsTensor() was false");
      const Tensor& src_tensor = ort_value.Get<Tensor>();
      total_bytes += src_tensor.SizeInBytes();

      if (total_bytes >= PROTOBUF_UPPER_LIMIT) {
        save_as_external_data = true;
        break;
      }
    }
  }

  // Convert Tensor to TensorProto.
  std::vector<char> tensor_data_buffer{};
  static const OrtMemoryInfo cpu_alloc_info{onnxruntime::CPU, OrtDeviceAllocator};
  std::vector<ONNX_NAMESPACE::TensorProto> saved_tensor_protos{};
  saved_tensor_protos.reserve(ordered_tensor_names.size());

  for (const auto& tensor_name : ordered_tensor_names) {
    const OrtValue& ort_value = ort_values.at(tensor_name);
    ORT_RETURN_IF_NOT(ort_value.IsTensor(), "ort_value.IsTensor() was false");
    const Tensor& src_tensor = ort_value.Get<Tensor>();
    tensor_data_buffer.resize(src_tensor.SizeInBytes());
    auto& tensor_location = src_tensor.Location();

    if (tensor_location.device.Type() == OrtDevice::CPU ||
        tensor_location.mem_type == OrtMemTypeCPUInput ||
        tensor_location.mem_type == OrtMemTypeCPUOutput ||
        tensor_location.device.Type() == OrtDevice::GPU) {
      gsl::span<char> dst_span = gsl::make_span(tensor_data_buffer);
      ORT_RETURN_IF_NOT(src_tensor.SizeInBytes() == static_cast<size_t>(dst_span.size_bytes()), "src size != dst size");
      Tensor dst_tensor{src_tensor.DataType(), src_tensor.Shape(), dst_span.data(), cpu_alloc_info};
      ORT_RETURN_IF_ERROR(data_transfer_manager.CopyTensor(src_tensor, dst_tensor));

      ONNX_NAMESPACE::TensorProto tensor_proto;
      if (save_as_external_data) {
        tensor_proto = utils::TensorToTensorProto(src_tensor, tensor_name, true, tensors_data_relative_path);
      } else {
        tensor_proto = utils::TensorToTensorProto(src_tensor, tensor_name);
      }
      saved_tensor_protos.emplace_back(tensor_proto);
    } else {
      ORT_THROW("Unsupported device type for saving tensors");
    }
  }

  // Save TensorProto to file.
  ORT_RETURN_IF_ERROR(SaveTensorProtosToFile(tensors_path, saved_tensor_protos));
  return Status::OK();
}

namespace {
Status SaveProperties(
    const PathString& properties_path,
    const std::unordered_map<std::string, std::string>& properties) {
  std::vector<ONNX_NAMESPACE::StringStringEntryProto> property_protos{};
  property_protos.reserve(properties.size());
  std::transform(
      properties.begin(), properties.end(), std::back_inserter(property_protos),
      [](const std::pair<std::string, std::string>& property) {
        ONNX_NAMESPACE::StringStringEntryProto property_proto{};
        property_proto.set_key(property.first);
        property_proto.set_value(property.second);
        return property_proto;
      });

  // ensure consistent ordering
  std::sort(
      property_protos.begin(), property_protos.end(),
      [](const ONNX_NAMESPACE::StringStringEntryProto& a,
         const ONNX_NAMESPACE::StringStringEntryProto& b) {
        return a.key() < b.key();
      });

  ORT_RETURN_IF_ERROR(WithOpenFile(
      properties_path, false,
      [&property_protos](int fd) {
        google::protobuf::io::FileOutputStream output{fd};
        ORT_RETURN_IF_ERROR(WriteProtoMessageSequence(property_protos, output));
        return Status::OK();
      }));

  return Status::OK();
}

}  // namespace

Status SaveModelCheckpoint(
    const PathString& checkpoint_path,
    const DataTransferManager& data_transfer_manager,
    const NameMLValMap& runtime_tensors,
    const std::unordered_map<std::string, std::string>& properties) {
  LOGS_DEFAULT(INFO) << "Saving model checkpoint files to " << ToUTF8String(checkpoint_path);

  LOGS_DEFAULT_IF(Env::Default().FolderExists(checkpoint_path), WARNING)
      << "Checkpoint directory exists - data may be overwritten.";

  ORT_RETURN_IF_ERROR(Env::Default().CreateFolder(checkpoint_path));

  // write tensors files
  ORT_RETURN_IF_ERROR(SaveRuntimeTensors(
      GetCheckpointTensorsFilePath(checkpoint_path),
      GetCheckpointTensorsDataFilePath(checkpoint_path),
      data_transfer_manager, runtime_tensors));

  // write properties file
  ORT_RETURN_IF_ERROR(SaveProperties(
      GetCheckpointPropertiesFilePath(checkpoint_path), properties));

  LOGS_DEFAULT(INFO) << "Model checkpoint saved successfully.";

  return Status::OK();
}

Status SaveTensorProtosToFile(const PathString& proto_file_path,
                              const std::vector<ONNX_NAMESPACE::TensorProto>& tensor_protos_to_save) {
  ORT_RETURN_IF_ERROR(WithOpenFile(
      proto_file_path, false,
      [&tensor_protos_to_save](int fd) {
        google::protobuf::io::FileOutputStream output{fd};
        ORT_RETURN_IF_ERROR(WriteProtoMessageSequence(tensor_protos_to_save, output));
        return Status::OK();
      }));

  return Status::OK();
}

namespace {
Status UpdateTensorsExternalDataLocations(
    const PathString& external_data_path,
    std::vector<ONNX_NAMESPACE::TensorProto>& tensor_protos) {
  for (auto& tensor_proto : tensor_protos) {
    if (tensor_proto.data_location() != ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL) {
      continue;
    }

    auto& external_data = *tensor_proto.mutable_external_data();
    auto location_it = std::find_if(
        external_data.begin(), external_data.end(),
        [](ONNX_NAMESPACE::StringStringEntryProto& kvp) { return kvp.key() == "location"; });
    ORT_RETURN_IF_NOT(location_it != external_data.end(), "location_it == external_data.end()");

    // TODO is the encoding correct? https://github.com/onnx/onnx/issues/2392
    location_it->set_value(ToUTF8String(external_data_path));
  }

  return Status::OK();
}
}  // namespace

Status LoadModelCheckpoint(
    const PathString& checkpoint_path,
    const PathString& model_path,
    std::vector<ONNX_NAMESPACE::TensorProto>& tensor_protos,
    std::unordered_map<std::string, std::string>& properties) {
  LOGS_DEFAULT(INFO) << "Loading model checkpoint files from " << ToUTF8String(checkpoint_path);

  // read tensors file
  std::vector<ONNX_NAMESPACE::TensorProto> loaded_tensor_protos{};
  ORT_RETURN_IF_ERROR(WithOpenFile(
      GetCheckpointTensorsFilePath(checkpoint_path), true,
      [&loaded_tensor_protos](int fd) {
        google::protobuf::io::FileInputStream input{fd};
        ORT_RETURN_IF_ERROR(ReadProtoMessageSequence(loaded_tensor_protos, input));
        return Status::OK();
      }));

  // set external data locations
  {
    PathString model_directory_path{}, model_directory_canonical_path{};
    ORT_RETURN_IF_ERROR(GetDirNameFromFilePath(
        model_path, model_directory_path));
    ORT_RETURN_IF_ERROR(Env::Default().GetCanonicalPath(
        model_directory_path, model_directory_canonical_path));

    PathString checkpoint_canonical_path{};
    ORT_RETURN_IF_ERROR(Env::Default().GetCanonicalPath(
        checkpoint_path, checkpoint_canonical_path));

    Path relative_tensors_data_path_obj{};
    ORT_RETURN_IF_ERROR(RelativePath(
        Path::Parse(model_directory_canonical_path),
        Path::Parse(GetCheckpointTensorsDataFilePath(checkpoint_canonical_path)),
        relative_tensors_data_path_obj));
    ORT_RETURN_IF_ERROR(UpdateTensorsExternalDataLocations(
        relative_tensors_data_path_obj.ToPathString(), loaded_tensor_protos));
  }

  // read properties file
  std::vector<ONNX_NAMESPACE::StringStringEntryProto> loaded_property_protos{};
  ORT_RETURN_IF_ERROR(WithOpenFile(
      GetCheckpointPropertiesFilePath(checkpoint_path), true,
      [&loaded_property_protos](int fd) {
        google::protobuf::io::FileInputStream input{fd};
        ORT_RETURN_IF_ERROR(ReadProtoMessageSequence(loaded_property_protos, input));
        return Status::OK();
      }));

  std::unordered_map<std::string, std::string> loaded_properties{};
  std::transform(
      loaded_property_protos.begin(), loaded_property_protos.end(),
      std::inserter(loaded_properties, loaded_properties.end()),
      [](const ONNX_NAMESPACE::StringStringEntryProto& property_proto) {
        return std::make_pair(property_proto.key(), property_proto.value());
      });

  tensor_protos = std::move(loaded_tensor_protos);
  properties = std::move(loaded_properties);

  LOGS_DEFAULT(INFO) << "Model checkpoint loaded successfully.";

  return Status::OK();
}

}  // namespace training
}  // namespace onnxruntime
