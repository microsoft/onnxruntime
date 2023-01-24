// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/inlined_containers.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/path.h"
#include "core/framework/framework_common.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/platform/env.h"
#include "core/platform/path_lib.h"
#include "core/util/protobuf_parsing_utils.h"

#include "orttraining/core/framework/checkpoint_common.h"
#include "orttraining/core/framework/protobuf_message_sequence.h"
#include "orttraining/training_api/checkpoint.h"
#include "orttraining/training_api/utils.h"

namespace onnxruntime {
namespace training {
namespace api {

namespace {

const PathString k_tensor_proto_file_name = ORT_TSTR("tensors.pbseq");
const PathString k_tensor_proto_properties_file_name = ORT_TSTR("properties.pbseq");
const PathString k_trainable_param_root_prefix = ORT_TSTR("paramtrain");
const PathString k_non_trainable_param_root_prefix = ORT_TSTR("paramfrozen");
const PathString k_optimizer_root_prefix = ORT_TSTR("optim");
const PathString k_property_root_prefix = ORT_TSTR("custom");
const PathString k_name_separator = ORT_TSTR("_");

const char builtin_lr_property_name[] = "builtin.initial_learning_rate";
const char builtin_step_property_name[] = "builtin.step";

/**
 * @brief Create TensorProtos From OrtValue objects
 *
 * @param name_to_ort_value name to OrtValue mapping.
 * @param data_transfer_manager data transfer manager to copy the tensor in OrtValue.
 * @param saved_tensor_protos saved results.
 * @return Status
 */
Status CreateTensorProtosFromOrtValues(
    const NameMLValMap& name_to_ort_value,
    const DataTransferManager& data_transfer_manager,
    std::vector<ONNX_NAMESPACE::TensorProto>& saved_tensor_protos) {
  // Order the tensors by name.
  InlinedVector<std::string> ordered_tensor_names{};
  ordered_tensor_names.reserve(name_to_ort_value.size());
  std::transform(name_to_ort_value.begin(), name_to_ort_value.end(), std::back_inserter(ordered_tensor_names),
                 [](const NameMLValMap::value_type& v) { return v.first; });
  std::sort(ordered_tensor_names.begin(), ordered_tensor_names.end());

  saved_tensor_protos.reserve(ordered_tensor_names.size());

  uint64_t total_bytes = 0;
  constexpr uint64_t PROTOBUF_UPPER_LIMIT = 2 * 1000 * 1000 * 1000;
  for (const auto& tensor_name : ordered_tensor_names) {
    const OrtValue& ort_value = name_to_ort_value.at(tensor_name);
    ORT_RETURN_IF_NOT(ort_value.IsTensor(), "ort_value.IsTensor() was false");
    const Tensor& src_tensor = ort_value.Get<Tensor>();

    // Currently large model size not considered, so exception thrown here
    // when protobuf upper limit hit.
    total_bytes += static_cast<uint64_t>(src_tensor.SizeInBytes());
    if (total_bytes >= PROTOBUF_UPPER_LIMIT) {
      ORT_THROW("checkpoint file size hit upper limit.");
    }

    saved_tensor_protos.emplace_back(utils::CopyTensorToTensorProto(
        src_tensor, tensor_name, data_transfer_manager));
  }

  return Status::OK();
}

PathString GetTensorProtoFilePath(const PathString& checkpoint_directory, const PathString& filename_prefix) {
  std::basic_ostringstream<PathChar> oss;
  oss << filename_prefix << k_name_separator << k_tensor_proto_file_name;
  return ConcatPathComponent<PathChar>(checkpoint_directory, oss.str());
}

PathString GetTensorProtoPropertiesFilePath(
    const PathString& checkpoint_directory, const PathString& filename_prefix) {
  std::basic_ostringstream<PathChar> oss;
  oss << filename_prefix << k_name_separator << k_tensor_proto_properties_file_name;
  return ConcatPathComponent<PathChar>(checkpoint_directory, oss.str());
}

PathString StringConcat(
    const PathString& s_a, const PathString& s_b,
    const PathString& del = k_name_separator) {
  std::basic_ostringstream<PathChar> oss;
  oss << s_a << del << s_b;
  return oss.str();
}

void StringSplit(const PathString& s, std::vector<PathString>& results,
                 const PathString& del = k_name_separator) {
  ORT_ENFORCE(!s.empty(), "String to split is empty");
  size_t start = 0;
  size_t end = s.find(del);
  while (end != std::string::npos) {
    results.push_back(s.substr(start, end - start));
    start = end + del.size();
    end = s.find(del, start);
  }
  results.push_back(s.substr(start, end - start));
}

bool StringStartsWith(PathString const& s, PathString const& p) {
  return s.rfind(p, 0) == 0;
}

bool StringEndsWith(PathString const& s, PathString const& p) {
  if (p.size() > s.size()) return false;
  return std::equal(p.rbegin(), p.rend(), s.rbegin());
}

void WriteTensorProtoToFile(const PathString& file_path,
                            const std::vector<ONNX_NAMESPACE::TensorProto>& tensor_protos,
                            std::string caller_context) {
  auto file_write_status = WithOpenFile(
      file_path, false,
      [&tensor_protos](int fd) {
        google::protobuf::io::FileOutputStream output{fd};
        ORT_RETURN_IF_ERROR(WriteProtoMessageSequence(tensor_protos, output));
        return Status::OK();
      });

  ORT_ENFORCE(file_write_status.IsOK(), caller_context, " write file failed: ", ToUTF8String(file_path));
}

void LoadTensorProtoFromFile(const PathString& file_path,
                             std::vector<ONNX_NAMESPACE::TensorProto>& tensor_protos,
                             std::string caller_context) {
  auto file_read_status = WithOpenFile(
      file_path, true,
      [&tensor_protos](int fd) {
        google::protobuf::io::FileInputStream input{fd};
        ORT_RETURN_IF_ERROR(ReadProtoMessageSequence(tensor_protos, input));
        return Status::OK();
      });

  ORT_ENFORCE(file_read_status.IsOK(), caller_context, " load file failed: ", ToUTF8String(file_path));
}

template <typename Func>
void FilterFilesFromDirectory(const PathString& folder_path, Func func) {
  LoopDir(folder_path, [&func](const PathChar* filename, OrtFileType file_type) -> bool {
    if (filename[0] == '.' || file_type == OrtFileType::TYPE_DIR) {
      return true;
    }

    return func(filename);
  });
}

Status OrtSaveInternal(
    const std::vector<ONNX_NAMESPACE::TensorProto>& trainable_tensor_protos,
    const std::vector<ONNX_NAMESPACE::TensorProto>& non_trainable_tensor_protos,
    const PathString& checkpoint_path) {
  // Make sure name unique across trainable and non-trainable lists.
  std::set<std::string> trainable_unique_names;
  std::set<std::string> non_trainable_unique_names;
  InlinedVector<std::string> inter_sec;
  auto check_unique = [](const std::vector<ONNX_NAMESPACE::TensorProto>& tensor_protos,
                         std::set<std::string>& unique_names) {
    for (auto& tensor_proto : tensor_protos) {
      ORT_ENFORCE(unique_names.find(tensor_proto.name()) == unique_names.end(),
                  "Duplicated tensor proto named ", tensor_proto.name());
      unique_names.emplace(tensor_proto.name());
    }
  };
  check_unique(trainable_tensor_protos, trainable_unique_names);
  check_unique(non_trainable_tensor_protos, non_trainable_unique_names);
  std::set_intersection(trainable_unique_names.begin(), trainable_unique_names.end(),
                        non_trainable_unique_names.begin(), non_trainable_unique_names.end(),
                        std::back_inserter(inter_sec));
  ORT_RETURN_IF_NOT(inter_sec.empty(), "Tensor name exists in both trainable param list and non-trainable param list.");

  // Keep following saving logic aligned with OrtSaveModuleStatesInternal.
  LOGS_DEFAULT(INFO)
      << "Saving model checkpoint files to " << ToUTF8String(checkpoint_path);
  LOGS_DEFAULT_IF(Env::Default().FolderExists(checkpoint_path), WARNING)
      << "Checkpoint directory exists - data may be overwritten.";
  ORT_RETURN_IF_ERROR(Env::Default().CreateFolder(checkpoint_path));

  // Save TensorProto to file.
  if (trainable_tensor_protos.size() > 0) {
    WriteTensorProtoToFile(
        GetTensorProtoFilePath(checkpoint_path, k_trainable_param_root_prefix),
        trainable_tensor_protos, "[trainable_param]");
  }

  if (non_trainable_tensor_protos.size() > 0) {
    WriteTensorProtoToFile(
        GetTensorProtoFilePath(checkpoint_path, k_non_trainable_param_root_prefix),
        non_trainable_tensor_protos, "[non_trainable_param]");
  }

  return Status::OK();
}

Status OrtSaveModuleStatesInternal(ModuleCheckpointState& module_state,
                                   const PathString& parameter_folder_path) {
  // Write weight tensors files.
  const auto& param_states = module_state.named_parameters;
  if (!param_states.empty()) {
    ORT_ENFORCE(module_state.train_session_data_transfer_mgr,
                "module checkpoint state has null train_session_data_transfer_mgr.");

    InlinedHashMap<PathString, std::unordered_map<std::string, OrtValue>>
        parameter_ort_values;
    for (auto it = param_states.begin(); it != param_states.end(); ++it) {
      if (it->second->RequiresGrad()) {
        parameter_ort_values[k_trainable_param_root_prefix].insert({it->first, it->second->Data()});
      } else {
        parameter_ort_values[k_non_trainable_param_root_prefix].insert({it->first, it->second->Data()});
      }
    }

    // Parameters saving.
    for (auto& pair : parameter_ort_values) {
      std::vector<ONNX_NAMESPACE::TensorProto> param_tensor_protos;
      ORT_RETURN_IF_ERROR(CreateTensorProtosFromOrtValues(
          pair.second,
          *module_state.train_session_data_transfer_mgr,
          param_tensor_protos));

      // Save TensorProto to file.
      WriteTensorProtoToFile(
          GetTensorProtoFilePath(parameter_folder_path, pair.first),
          param_tensor_protos, "[param]");
    }
  }

  return Status::OK();
}

Status OrtSaveOptimizerStatesInternal(OptimizerCheckpointState& optimizer_state,
                                      const PathString& checkpoint_path) {
  if (optimizer_state.group_named_optimizer_states.empty()) {
    return Status::OK();
  }

  ORT_ENFORCE(optimizer_state.optimizer_session_data_transfer_mgr,
              "optimizer checkpoint state has null optimizer_session_data_transfer_mgr.");

  // Write optimizer state tensors files.
  for (auto& group_named_optimizer_state : optimizer_state.group_named_optimizer_states) {
    const PathString group_name = ToPathString(group_named_optimizer_state.first);
    const std::shared_ptr<GroupOptimizerState>& group_optimizer_state_ptr = group_named_optimizer_state.second;
    const PathString& cur_group_filename_prefix =
        StringConcat(k_optimizer_root_prefix, group_name);

    // Re-organize optimizer_state_ort_values mapping
    // Firstly indexed by momentum names; Secondly indexed by parameter names.
    InlinedHashMap<std::string, std::unordered_map<std::string, OrtValue>> optimizer_state_ort_values;
    for (const auto& [param_name, param_optimizer_state] : group_optimizer_state_ptr->param_named_optimizer_states) {
      for (const auto& [momentum_name, m_state_val] : param_optimizer_state.momentum_named_states) {
        if (optimizer_state_ort_values.find(momentum_name) == optimizer_state_ort_values.end()) {
          std::unordered_map<std::string, OrtValue> param_name_to_ortvalue{{param_name, m_state_val}};
          optimizer_state_ort_values.insert({momentum_name, param_name_to_ortvalue});
        } else {
          optimizer_state_ort_values[momentum_name].insert({param_name, m_state_val});
        }
      }
    }

    // Save each optimizer state (of all parameters) into single file.
    // For example: save "momentum_1" of all parameters into one file.
    for (auto& pair : optimizer_state_ort_values) {
      const PathString momentum_name = ToPathString(pair.first);
      const std::unordered_map<std::string, OrtValue>& param_name_to_ortvalue = pair.second;
      const PathString& cur_state_filename_prefix =
          StringConcat(cur_group_filename_prefix, momentum_name);

      std::vector<ONNX_NAMESPACE::TensorProto> saved_tensor_protos;
      ORT_RETURN_IF_ERROR(CreateTensorProtosFromOrtValues(
          param_name_to_ortvalue,
          *optimizer_state.optimizer_session_data_transfer_mgr,
          saved_tensor_protos));

      // Save TensorProto to file.
      WriteTensorProtoToFile(
          GetTensorProtoFilePath(checkpoint_path, cur_state_filename_prefix),
          saved_tensor_protos, "[optimizer_state]");
    }

    // Storing group-wise properties.
    PropertyBag properties;
    properties.AddProperty(builtin_lr_property_name, group_optimizer_state_ptr->initial_lr);
    properties.AddProperty(builtin_step_property_name, group_optimizer_state_ptr->step);
    std::vector<ONNX_NAMESPACE::TensorProto> group_wise_properties_tensor_protos;
    properties.ToTensorProtos(group_wise_properties_tensor_protos);

    WriteTensorProtoToFile(
        GetTensorProtoPropertiesFilePath(checkpoint_path, cur_group_filename_prefix),
        group_wise_properties_tensor_protos, "[param_group_properties]");
  }

  return Status::OK();
}

Status OrtSaveInternal(
    CheckpointState& state, const PathString& checkpoint_path) {
  LOGS_DEFAULT(INFO) << "Saving model checkpoint files to " << ToUTF8String(checkpoint_path);
  LOGS_DEFAULT_IF(Env::Default().FolderExists(checkpoint_path), WARNING)
      << "Checkpoint directory exists - data may be overwritten.";
  ORT_RETURN_IF_ERROR(Env::Default().CreateFolder(checkpoint_path));

  // Write weight tensors files.
  ORT_RETURN_IF_ERROR(OrtSaveModuleStatesInternal(state.module_checkpoint_state, checkpoint_path));

  // Write optimizer state tensors files.
  ORT_RETURN_IF_ERROR(OrtSaveOptimizerStatesInternal(state.optimizer_checkpoint_state, checkpoint_path));

  // Write properties file
  const PropertyBag& property_bag = state.property_bag;
  if (property_bag.Size() > 0) {
    std::vector<ONNX_NAMESPACE::TensorProto> properties_tensor_protos;
    property_bag.ToTensorProtos(properties_tensor_protos);

    WriteTensorProtoToFile(
        GetTensorProtoPropertiesFilePath(checkpoint_path, k_property_root_prefix),
        properties_tensor_protos, "[custom_properties]");
  }

  LOGS_DEFAULT(INFO) << "Checkpoint saved successfully.";
  return Status::OK();
}

Status OrtLoadModuleStatesInternal(
    const PathString& parameter_folder_path, ModuleCheckpointState& module_state) {
  // Find parameter files.
  InlinedVector<std::pair<PathString, bool>> param_filenames;
  FilterFilesFromDirectory(
      parameter_folder_path,
      [&param_filenames](const PathChar* filename) -> bool {
        PathString filename_str = filename;
        if (StringStartsWith(filename_str, k_trainable_param_root_prefix)) {
          param_filenames.push_back(std::make_pair(filename_str, true));
        } else if (StringStartsWith(filename_str, k_non_trainable_param_root_prefix)) {
          param_filenames.push_back(std::make_pair(filename_str, false));
        }
        return true;
      });

  if (param_filenames.empty()) {
    return Status::OK();
  }

  // Parameter parsing.
  auto& named_parameters = module_state.named_parameters;
  auto load_model_proto_into_module =
      [&named_parameters](const PathString module_state_file_path, bool is_trainable) -> Status {
    std::vector<ONNX_NAMESPACE::TensorProto> param_tensor_protos{};

    LoadTensorProtoFromFile(module_state_file_path, param_tensor_protos, "[params]");

    std::unordered_map<std::string, OrtValue> name_to_ort_values;
    ORT_RETURN_IF_ERROR(CreateOrtValuesFromTensorProtos(param_tensor_protos, name_to_ort_values));
    for (auto it = name_to_ort_values.begin(); it != name_to_ort_values.end(); ++it) {
      auto param = std::make_shared<Parameter>(it->first, it->second, is_trainable);
      named_parameters.insert({it->first, param});
    }
    return Status::OK();
  };

  for (auto& pair : param_filenames) {
    auto param_file_path = ConcatPathComponent<PathChar>(parameter_folder_path, pair.first);
    ORT_RETURN_IF_ERROR(load_model_proto_into_module(param_file_path, pair.second));
  }

  return Status::OK();
}

Status OrtLoadOptimizerStatesInternal(const PathString& optimizer_folder_path,
                                      OptimizerCheckpointState& optimizer_state) {
  // Optimizer states parsing.
  std::vector<PathString> optim_state_filenames;
  std::vector<PathString> optim_property_filenames;
  FilterFilesFromDirectory(
      optimizer_folder_path,
      [&optim_state_filenames, &optim_property_filenames](const PathChar* filename) -> bool {
        PathString filename_str = filename;
        if (StringStartsWith(filename_str, k_optimizer_root_prefix)) {
          if (StringEndsWith(filename_str, k_tensor_proto_file_name)) {
            optim_state_filenames.push_back(filename_str);
          } else if (StringEndsWith(filename_str, k_tensor_proto_properties_file_name)) {
            optim_property_filenames.push_back(filename_str);
          } else {
            ORT_THROW("Unexpected file extension.");
          }
        }
        return true;
      });

  auto& grouped_optimizer_states = optimizer_state.group_named_optimizer_states;
  // For each optimizer state files, parse the data and feed into grouped_optimizer_states.
  for (auto& filename : optim_state_filenames) {
    std::vector<PathString> results;
    StringSplit(filename, results);
    ORT_ENFORCE(results.size() >= 3U, "Incorrect optimizer state filename.");
    const std::string& group_name = ToUTF8String(results[1]);
    const std::string& momentum_name = ToUTF8String(results[2]);

    const PathString cur_group_filename_prefix =
        StringConcat(k_optimizer_root_prefix, results[1]);
    PathString cur_momentum_state_filename_prefix =
        StringConcat(cur_group_filename_prefix, results[2]);
    ORT_ENFORCE(filename.compare(StringConcat(cur_momentum_state_filename_prefix, k_tensor_proto_file_name)) == 0);

    if (grouped_optimizer_states.find(group_name) == grouped_optimizer_states.end()) {
      grouped_optimizer_states.insert({group_name, std::make_shared<GroupOptimizerState>()});
    }

    auto& group_optimizer_state = grouped_optimizer_states[group_name];
    std::unordered_map<std::string, ParameterOptimizerState>&
        param_optimizer_states = group_optimizer_state->param_named_optimizer_states;

    const PathString& tensor_file_path = GetTensorProtoFilePath(optimizer_folder_path,
                                                                cur_momentum_state_filename_prefix);
    std::vector<ONNX_NAMESPACE::TensorProto> param_optimizer_state_tensor_protos{};
    LoadTensorProtoFromFile(tensor_file_path, param_optimizer_state_tensor_protos, "[optimizer_state]");

    std::unordered_map<std::string, OrtValue> name_to_ort_values;
    ORT_RETURN_IF_ERROR(CreateOrtValuesFromTensorProtos(param_optimizer_state_tensor_protos, name_to_ort_values));
    for (auto& pair : name_to_ort_values) {
      auto& param_name = pair.first;
      if (param_optimizer_states.find(param_name) == param_optimizer_states.end()) {
        ParameterOptimizerState param_state;
        param_optimizer_states.insert({param_name, param_state});
      }
      param_optimizer_states[param_name].momentum_named_states.insert({momentum_name, std::move(pair.second)});
    }
  }

  // For each optimizer properties files, parse the data and feed into grouped_optimizer_states.
  for (auto& filename : optim_property_filenames) {
    std::vector<PathString> results;
    StringSplit(filename, results);
    ORT_ENFORCE(results.size() >= 2U, "Incorrect optimizer property filename.");
    const std::string& group_name = ToUTF8String(results[1]);

    if (grouped_optimizer_states.find(group_name) == grouped_optimizer_states.end()) {
      grouped_optimizer_states.insert({group_name, std::make_shared<GroupOptimizerState>()});
    }

    auto& group_optimizer_state = grouped_optimizer_states[group_name];

    // Parse group-wise properties.
    const PathString cur_group_filename_prefix = StringConcat(k_optimizer_root_prefix, results[1]);
    const PathString& tensor_file_path = GetTensorProtoPropertiesFilePath(optimizer_folder_path,
                                                                          cur_group_filename_prefix);
    std::vector<ONNX_NAMESPACE::TensorProto> group_wise_property_protos{};
    LoadTensorProtoFromFile(tensor_file_path, group_wise_property_protos, "[optimizer_groupwise_property]");

    PropertyBag properties;
    for (auto& property_proto : group_wise_property_protos) {
      properties.AddProperty(property_proto);
    }

    group_optimizer_state->initial_lr = properties.GetProperty<float>(builtin_lr_property_name);
    group_optimizer_state->step = properties.GetProperty<int64_t>(builtin_step_property_name);
    grouped_optimizer_states.insert({group_name, group_optimizer_state});
  }

  return Status::OK();
}

Status OrtLoadCustomPropertyInternal(const PathString& property_folder_path,
                                     PropertyBag& property_bag) {
  // Find custom property files.
  std::vector<PathString> custom_property_filenames;
  FilterFilesFromDirectory(
      property_folder_path,
      [&custom_property_filenames](const PathChar* filename) -> bool {
        PathString filename_str = filename;
        if (StringStartsWith(filename_str, k_property_root_prefix)) {
          custom_property_filenames.push_back(filename_str);
        }
        return true;
      });

  if (custom_property_filenames.empty()) {
    return Status::OK();
  }

  for (auto& property_file_path : custom_property_filenames) {
    std::vector<ONNX_NAMESPACE::TensorProto> property_protos{};
    auto property_file_full_path = ConcatPathComponent<PathChar>(property_folder_path, property_file_path);
    LoadTensorProtoFromFile(property_file_full_path, property_protos, "[custom_property]");

    for (auto& property_proto : property_protos) {
      property_bag.AddProperty(property_proto);
    }
  }

  return Status::OK();
}

Status OrtLoadInternal(const PathString& checkpoint_path,
                       ONNX_NAMESPACE::ModelProto& model_proto) {
  // Find tensor proto files.
  InlinedHashMap<std::string, ONNX_NAMESPACE::TensorProto> param_tensor_protos;
  InlinedVector<PathString> tensor_proto_filenames;

  FilterFilesFromDirectory(
      checkpoint_path,
      [&tensor_proto_filenames](const PathChar* filename) -> bool {
        PathString filename_str = filename;
        if (StringEndsWith(filename_str, k_tensor_proto_file_name)) {
          tensor_proto_filenames.push_back(filename_str);
        }
        return true;
      });

  // Load tensor protos to the tensorProto Vector
  for (const auto& tensor_file_path : tensor_proto_filenames) {
    std::vector<ONNX_NAMESPACE::TensorProto> tensor_protos{};
    const auto tensor_file_full_path = ConcatPathComponent<PathChar>(checkpoint_path, tensor_file_path);
    LoadTensorProtoFromFile(tensor_file_full_path, tensor_protos, "[params]");

    for (auto& tensor_proto : tensor_protos) {
      auto tensor_proto_name = tensor_proto.name();
      param_tensor_protos.emplace(std::make_pair(tensor_proto_name, std::move(tensor_proto)));
    }
  }

  // Load imported initializers into the Model
  for (auto& init : *(model_proto.mutable_graph()->mutable_initializer())) {
    ORT_ENFORCE(init.has_name(), "An initializer should have a name.");
    auto it = param_tensor_protos.find(init.name());
    ORT_ENFORCE(it != param_tensor_protos.end(),
                "The initializer name was not found in the checkpoint file loaded.");
    init = it->second;
  }

  return Status::OK();
}

Status OrtLoadInternal(const PathString& checkpoint_path, CheckpointState& state) {
  ORT_ENFORCE(Env::Default().FolderExists(checkpoint_path), "Checkpoint folder not exit");
  ORT_RETURN_IF_ERROR(OrtLoadModuleStatesInternal(checkpoint_path, state.module_checkpoint_state));
  ORT_RETURN_IF_ERROR(OrtLoadOptimizerStatesInternal(checkpoint_path, state.optimizer_checkpoint_state));
  ORT_RETURN_IF_ERROR(OrtLoadCustomPropertyInternal(checkpoint_path, state.property_bag));
  return Status::OK();
}

}  // namespace

Status SaveCheckpoint(const std::vector<ONNX_NAMESPACE::TensorProto>& trainable_tensor_protos,
                      const std::vector<ONNX_NAMESPACE::TensorProto>& non_trainable_tensor_protos,
                      const PathString& checkpoint_path) {
  return OrtSaveInternal(trainable_tensor_protos, non_trainable_tensor_protos, checkpoint_path);
}

Status SaveCheckpoint(CheckpointState& states, const PathString& checkpoint_path) {
  return OrtSaveInternal(states, checkpoint_path);
}

Status LoadCheckpoint(const PathString& checkpoint_path, CheckpointState& checkpoint_states) {
  return OrtLoadInternal(checkpoint_path, checkpoint_states);
}

Status LoadCheckpointToModel(const PathString& checkpoint_path,
                             ONNX_NAMESPACE::ModelProto& model_proto) {
  return OrtLoadInternal(checkpoint_path, model_proto);
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
