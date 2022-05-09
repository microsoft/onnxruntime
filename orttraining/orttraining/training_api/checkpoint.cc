// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/platform/path_lib.h"
#include "core/platform/env.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/util/protobuf_parsing_utils.h"
#include "onnx/defs/tensor_proto_util.h"
#include "orttraining/core/framework/protobuf_message_sequence.h"
#include "orttraining/training_api/checkpoint.h"
#include "core/common/path.h"
#include "core/platform/env.h"
#include "core/platform/path_lib.h"
#include "core/framework/framework_common.h"
#include "orttraining/core/framework/checkpoint_common.h"

namespace onnxruntime {
namespace training {
namespace api {

namespace {

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
  std::vector<std::string> ordered_tensor_names{};
  ordered_tensor_names.reserve(name_to_ort_value.size());
  std::transform(name_to_ort_value.begin(), name_to_ort_value.end(), std::back_inserter(ordered_tensor_names),
                 [](const NameMLValMap::value_type& v) { return v.first; });
  std::sort(ordered_tensor_names.begin(), ordered_tensor_names.end());

  // Copy the tensor data and create TensorProto storing the data.
  std::vector<char> tensor_data_buffer{};
  static const OrtMemoryInfo cpu_alloc_info{onnxruntime::CPU, OrtDeviceAllocator};

  saved_tensor_protos.reserve(ordered_tensor_names.size());

  unsigned long total_bytes = 0;
  constexpr unsigned long PROTOBUF_UPPER_LIMIT = 2 * 1000 * 1000 * 1000;
  for (const auto& tensor_name : ordered_tensor_names) {
    const OrtValue& ort_value = name_to_ort_value.at(tensor_name);
    ORT_RETURN_IF_NOT(ort_value.IsTensor(), "ort_value.IsTensor() was false");
    const Tensor& src_tensor = ort_value.Get<Tensor>();
    tensor_data_buffer.resize(src_tensor.SizeInBytes());

    // Currently large model size not considered, so exception thrown here
    // when protobuf upper limit hit.
    total_bytes += src_tensor.SizeInBytes();
    if (total_bytes >= PROTOBUF_UPPER_LIMIT) {
      ORT_THROW("checkpoint file size hit upper limit.");
    }

    auto& tensor_location = src_tensor.Location();
    if (tensor_location.device.Type() == OrtDevice::CPU ||
        tensor_location.mem_type == OrtMemTypeCPUInput ||
        tensor_location.mem_type == OrtMemTypeCPUOutput ||
        tensor_location.device.Type() == OrtDevice::GPU) {
      gsl::span<char> dst_span = gsl::make_span(tensor_data_buffer);
      ORT_RETURN_IF_NOT(src_tensor.SizeInBytes() == static_cast<size_t>(dst_span.size_bytes()), "src size != dst size");
      Tensor dst_tensor{src_tensor.DataType(), src_tensor.Shape(), dst_span.data(), cpu_alloc_info};
      ORT_RETURN_IF_ERROR(data_transfer_manager.CopyTensor(src_tensor, dst_tensor));

      // Convert Tensor to TensorProto.
      ONNX_NAMESPACE::TensorProto tensor_proto;
      tensor_proto = utils::TensorToTensorProto(dst_tensor, tensor_name);
      saved_tensor_protos.emplace_back(tensor_proto);
    } else {
      ORT_THROW("Unsupported device type for saving tensors");
    }
  }

  return Status::OK();
}

constexpr const char* k_tensor_proto_file_name = "tensors.pbseq";
constexpr const char* k_tensor_proto_properties_file_name = "properties.pbseq";
constexpr const char* k_trainable_param_root_prefix = "paramtrain";
constexpr const char* k_non_trainable_param_root_prefix = "paramfrozen";
constexpr const char* k_optimizer_root_prefix = "optim";
constexpr const char* k_property_root_prefix = "custom";
constexpr const char* k_name_seperator = "_";

PathString GetTensorProtoFilePath(const PathString& checkpoint_directory, const std::string& filename_prefix) {
  return ConcatPathComponent<PathChar>(checkpoint_directory, ORT_TSTR(filename_prefix + k_name_seperator) + k_tensor_proto_file_name);
}

PathString GetTensorProtoPropertiesFilePath(const PathString& checkpoint_directory, const std::string& filename_prefix) {
  return ConcatPathComponent<PathChar>(checkpoint_directory, ORT_TSTR(filename_prefix + k_name_seperator) + k_tensor_proto_properties_file_name);
}

std::string StringConcat(const std::string& s_a, const std::string& s_b, const std::string& del = k_name_seperator) {
  return s_a + del + s_b;
}

void StringSplit(const std::string& s, std::vector<std::string>& results, const std::string& del = k_name_seperator) {
  ORT_ENFORCE(!s.empty(), "String to split is empty");
  int start = 0;
  int end = s.find(del);
  while (end != -1) {
    results.push_back(s.substr(start, end - start));
    start = end + del.size();
    end = s.find(del, start);
  }
  results.push_back(s.substr(start, end - start));
}

bool StringStartsWith(std::string const& s, std::string const& p) {
  return s.rfind(p, 0) == 0;
}

bool StringEndsWith(std::string const& s, std::string const& p) {
  if (p.size() > s.size()) return false;
  return std::equal(p.rbegin(), p.rend(), s.rbegin());
}

Status OrtSaveInternal(
    const std::string& model_uri,
    const std::vector<std::string>& trainable_param_names,
    const PathString& checkpoint_path) {
  auto logger_ptr = std::make_unique<logging::Logger>(logging::LoggingManager::DefaultLogger());
  std::shared_ptr<Model> p_model;
  ORT_RETURN_IF_ERROR(Model::Load(model_uri, p_model, nullptr, *logger_ptr));
  Graph& graph = p_model->MainGraph();
  std::vector<ONNX_NAMESPACE::TensorProto> trainable_weight_values;
  std::vector<ONNX_NAMESPACE::TensorProto> non_trainable_weight_values;
  const auto& initializer_tensors = graph.GetAllInitializedTensors();
  for (const std::pair<std::string, const ONNX_NAMESPACE::TensorProto*>& pair : initializer_tensors) {
    if (std::find(trainable_param_names.begin(), trainable_param_names.end(), pair.first) != trainable_param_names.end()) {
      trainable_weight_values.emplace_back(static_cast<ONNX_NAMESPACE::TensorProto>(*pair.second));
    } else {
      non_trainable_weight_values.emplace_back(static_cast<ONNX_NAMESPACE::TensorProto>(*pair.second));
    }
  }
  ORT_RETURN_IF_NOT(trainable_weight_values.size() == trainable_param_names.size(),
                    "Mismatch count of trainable weights and given weight names.");

  // Keep following saving logic aligned with OrtSaveModuleStatesInternal.
  LOGS_DEFAULT(INFO) << "Saving model checkpoint files to " << ToUTF8String(checkpoint_path);
  LOGS_DEFAULT_IF(Env::Default().FolderExists(checkpoint_path), WARNING)
      << "Checkpoint directory exists - data may be overwritten.";
  ORT_RETURN_IF_ERROR(Env::Default().CreateFolder(checkpoint_path));

  // Save TensorProto to file.
  if (trainable_weight_values.size() > 0) {
    ORT_RETURN_IF_ERROR(WithOpenFile(
        GetTensorProtoFilePath(checkpoint_path, k_trainable_param_root_prefix), false,
        [&trainable_weight_values](int fd) {
          google::protobuf::io::FileOutputStream output{fd};
          ORT_RETURN_IF_ERROR(WriteProtoMessageSequence(trainable_weight_values, output));
          return Status::OK();
        }));
  }

  if (non_trainable_weight_values.size() > 0) {
    ORT_RETURN_IF_ERROR(WithOpenFile(
        GetTensorProtoFilePath(checkpoint_path, k_non_trainable_param_root_prefix), false,
        [&non_trainable_weight_values](int fd) {
          google::protobuf::io::FileOutputStream output{fd};
          ORT_RETURN_IF_ERROR(WriteProtoMessageSequence(non_trainable_weight_values, output));
          return Status::OK();
        }));
  }

  return Status::OK();
}

Status OrtSaveModuleStatesInternal(ModuleCheckpointStates& module_states,
                                   const PathString& parameter_folder_path) {
  // Write weight tensors files.
  const auto& param_states = module_states.named_parameters;
  if (!param_states.empty()) {
    ORT_ENFORCE(module_states.train_session_data_transfer_mgr,
                "module checkpoint state has null train_session_data_transfer_mgr.");

    std::unordered_map<std::string, std::unordered_map<std::string, OrtValue>> parameter_ort_values;
    parameter_ort_values[k_trainable_param_root_prefix] = {};
    parameter_ort_values[k_non_trainable_param_root_prefix] = {};
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
          *module_states.train_session_data_transfer_mgr,
          param_tensor_protos));

      // Save TensorProto to file.
      ORT_RETURN_IF_ERROR(WithOpenFile(
          GetTensorProtoFilePath(parameter_folder_path, pair.first), false,
          [&param_tensor_protos](int fd) {
            google::protobuf::io::FileOutputStream output{fd};
            ORT_RETURN_IF_ERROR(WriteProtoMessageSequence(param_tensor_protos, output));
            return Status::OK();
          }));
    }
  }

  return Status::OK();
}

Status OrtSaveOptimizerStatesInternal(OptimizerCheckpointStates& optimizer_states,
                                      const PathString& checkpoint_path) {
  if (optimizer_states.group_named_optimizer_states.empty()) {
    return Status::OK();
  }

  ORT_ENFORCE(optimizer_states.optimizer_session_data_transfer_mgr,
              "optimizer checkpoint state has null optimizer_session_data_transfer_mgr.");

  // Write optimizer state tensors files.
  for (auto& group_named_optimizer_state : optimizer_states.group_named_optimizer_states) {
    const std::string& group_name = group_named_optimizer_state.first;
    const std::shared_ptr<GroupOptimizerState>& group_optimizer_state_ptr = group_named_optimizer_state.second;
    const std::string& cur_group_filename_prefix = StringConcat(k_optimizer_root_prefix, group_name);

    /**
     * Re-organize optimizer_state_ort_values mapping
     * Firstly indexed by moment state names; Secondly indexed by parameter names.
     */
    std::unordered_map<std::string, std::unordered_map<std::string, OrtValue>> optimizer_state_ort_values;
    for (const std::pair<std::string, ParameterOptimizerState>&
             param_named_optimizer_state : group_optimizer_state_ptr->param_named_optimizer_states_) {
      const std::string& param_name = param_named_optimizer_state.first;
      const auto& param_optimizer_state = param_named_optimizer_state.second;

      for (const std::pair<std::string, std::shared_ptr<OrtValue>>&
               m_state : param_optimizer_state.states_) {
        const std::string& m_state_name = m_state.first;
        const std::shared_ptr<OrtValue>& m_state_val = m_state.second;

        if (optimizer_state_ort_values.find(m_state_name) == optimizer_state_ort_values.end()) {
          std::unordered_map<std::string, OrtValue> param_name_to_ortvalue{{param_name, *(m_state_val)}};
          optimizer_state_ort_values.insert({m_state_name, param_name_to_ortvalue});
        } else {
          optimizer_state_ort_values[m_state_name].insert({param_name, *(m_state_val)});
        }
      }
    }

    /**
     * Save each optimizer state (of all parameters) into single file.
     * For example: save "momentum_1" of all parameters into one file.
     */
    for (auto& pair : optimizer_state_ort_values) {
      const auto& state_name = pair.first;
      const std::unordered_map<std::string, OrtValue>& param_name_to_ortvalue = pair.second;
      const std::string& cur_state_filename_prefix = StringConcat(cur_group_filename_prefix, state_name);

      std::vector<ONNX_NAMESPACE::TensorProto> saved_tensor_protos;
      ORT_RETURN_IF_ERROR(CreateTensorProtosFromOrtValues(
          param_name_to_ortvalue,
          *optimizer_states.optimizer_session_data_transfer_mgr,
          saved_tensor_protos));

      // Save TensorProto to file.
      ORT_RETURN_IF_ERROR(WithOpenFile(
          GetTensorProtoFilePath(checkpoint_path, cur_state_filename_prefix), false,
          [&saved_tensor_protos](int fd) {
            google::protobuf::io::FileOutputStream output{fd};
            ORT_RETURN_IF_ERROR(WriteProtoMessageSequence(saved_tensor_protos, output));
            return Status::OK();
          }));
    }

    // Storing group-wise properties.
    std::vector<std::unique_ptr<CheckpointProperty>> group_wise_properties;
    group_wise_properties.emplace_back(
        std::make_unique<TypedCheckpointProperty<float>>("learning_rate_", group_optimizer_state_ptr->learning_rate_));
    group_wise_properties.emplace_back(
        std::make_unique<TypedCheckpointProperty<int64_t>>("step_", group_optimizer_state_ptr->step_));

    std::vector<ONNX_NAMESPACE::TensorProto> group_wise_properties_tensor_protos;
    for (auto it = group_wise_properties.begin(); it != group_wise_properties.end(); ++it) {
      group_wise_properties_tensor_protos.emplace_back((*it)->ToTensorProto());
    }

    ORT_RETURN_IF_ERROR(WithOpenFile(
        GetTensorProtoPropertiesFilePath(checkpoint_path, cur_group_filename_prefix), false,
        [&group_wise_properties_tensor_protos](int fd) {
          google::protobuf::io::FileOutputStream output{fd};
          ORT_RETURN_IF_ERROR(WriteProtoMessageSequence(group_wise_properties_tensor_protos, output));
          return Status::OK();
        }));
  }

  return Status::OK();
}

Status OrtSaveInternal(
    CheckpointStates& states, const PathString& checkpoint_path) {
  LOGS_DEFAULT(INFO) << "Saving model checkpoint files to " << ToUTF8String(checkpoint_path);
  LOGS_DEFAULT_IF(Env::Default().FolderExists(checkpoint_path), WARNING)
      << "Checkpoint directory exists - data may be overwritten.";
  ORT_RETURN_IF_ERROR(Env::Default().CreateFolder(checkpoint_path));

  // Write weight tensors files.
  ORT_RETURN_IF_ERROR(OrtSaveModuleStatesInternal(states.module_checkpoint_states, checkpoint_path));

  // Write optimizer state tensors files.
  ORT_RETURN_IF_ERROR(OrtSaveOptimizerStatesInternal(states.optimizer_checkpoint_states, checkpoint_path));

  // Write properties file
  const PropertyBag& custom_properties = states.custom_properties;
  if (custom_properties.Size() > 0) {
    std::vector<ONNX_NAMESPACE::TensorProto> properties_tensor_protos;
    custom_properties.ToTensorProtos(properties_tensor_protos);

    ORT_RETURN_IF_ERROR(WithOpenFile(
        GetTensorProtoPropertiesFilePath(checkpoint_path, k_property_root_prefix), false,
        [&properties_tensor_protos](int fd) {
          google::protobuf::io::FileOutputStream output{fd};
          ORT_RETURN_IF_ERROR(WriteProtoMessageSequence(properties_tensor_protos, output));
          return Status::OK();
        }));
  }

  LOGS_DEFAULT(INFO) << "Model checkpoint saved successfully.";
  return Status::OK();
}

Status OrtLoadModuleStatesInternal(
    const PathString& parameter_folder_path, ModuleCheckpointStates& module_states) {
  // Parameter parsing.
  auto& named_parameters = module_states.named_parameters;
  std::vector<std::string> param_root_prefixes{k_trainable_param_root_prefix, k_non_trainable_param_root_prefix};

  for (auto& root_prefix : param_root_prefixes) {
    const PathString module_state_file_path = GetTensorProtoFilePath(parameter_folder_path, root_prefix);
    bool is_trainable = root_prefix == k_trainable_param_root_prefix;
    std::vector<ONNX_NAMESPACE::TensorProto> param_tensor_protos{};
    auto file_read_status = WithOpenFile(
        module_state_file_path, true,
        [&param_tensor_protos](int fd) {
          google::protobuf::io::FileInputStream input{fd};
          ORT_RETURN_IF_ERROR(ReadProtoMessageSequence(param_tensor_protos, input));
          return Status::OK();
        });

    if (!file_read_status.IsOK()) {
      LOGS_DEFAULT(WARNING) << ToUTF8String(module_state_file_path) << " trainable module state file not found or read failure, skip it.";
      return Status::OK();
    }

    std::vector<const ONNX_NAMESPACE::TensorProto*> param_tensor_proto_ptrs{};
    for (ONNX_NAMESPACE::TensorProto& param_tensor_proto : param_tensor_protos) {
      param_tensor_proto_ptrs.emplace_back(&param_tensor_proto);
    }

    std::unordered_map<std::string, OrtValue> name_to_ort_values;
    ORT_RETURN_IF_ERROR(CreateOrtValuesFromTensorProtos(param_tensor_proto_ptrs, name_to_ort_values));
    for (auto it = name_to_ort_values.begin(); it != name_to_ort_values.end(); ++it) {
      auto param = std::make_shared<Parameter>(it->first, it->second);
      ORT_RETURN_IF_ERROR(param->SetRequiresGrad(is_trainable));
      named_parameters.insert({it->first, param});
    }
  }

  return Status::OK();
}

Status OrtLoadOptimizerStatesInternal(
    const PathString& optimizer_folder_path, OptimizerCheckpointStates& optimizer_states) {
  if (!Env::Default().FolderExists(optimizer_folder_path)) {
    return Status::OK();
  }

  // Optimizer states parsing.
  std::vector<std::string> optim_state_filenames;
  std::vector<std::string> optim_property_filenames;
  LoopDir(optimizer_folder_path,
          [&optim_state_filenames, &optim_property_filenames](
              const PathChar* filename, OrtFileType file_type) -> bool {
            std::string filename_str = filename;
            if (filename_str[0] == '.' ||
                file_type == OrtFileType::TYPE_DIR ||
                !StringStartsWith(filename_str, k_optimizer_root_prefix)) {
              return true;
            }

            if (StringEndsWith(filename_str, k_tensor_proto_file_name)) {
              optim_state_filenames.push_back(filename_str);
            } else if (StringEndsWith(filename_str, k_tensor_proto_properties_file_name)) {
              optim_property_filenames.push_back(filename_str);
            } else {
              ORT_THROW("Unexpected file extension.");
            }
            return true;
          });

  auto& grouped_optimizer_states = optimizer_states.group_named_optimizer_states;
  // For each optimizer state files, parse the data and feed into grouped_optimizer_states.
  for (auto& filename : optim_state_filenames) {
    std::vector<std::string> results;
    StringSplit(filename, results);
    const std::string& group_name = results[1];
    const std::string& state_name = results[2];
    const std::string& cur_group_filename_prefix = StringConcat(k_optimizer_root_prefix, group_name);
    std::string cur_state_filename_prefix = StringConcat(cur_group_filename_prefix, state_name);

    ORT_ENFORCE(filename.compare(StringConcat(cur_state_filename_prefix, k_tensor_proto_file_name)) == 0);

    if (grouped_optimizer_states.find(group_name) == grouped_optimizer_states.end()) {
      grouped_optimizer_states.insert({group_name, std::make_shared<GroupOptimizerState>()});
    }

    auto& optimizer_state_in_this_group = grouped_optimizer_states[group_name];
    std::unordered_map<std::string, ParameterOptimizerState>&
        param_optimizer_state = optimizer_state_in_this_group->param_named_optimizer_states_;

    const PathString& tensor_file_path = GetTensorProtoFilePath(optimizer_folder_path, cur_state_filename_prefix);
    std::vector<ONNX_NAMESPACE::TensorProto> param_optimizer_state_tensor_protos{};
    auto file_read_status = WithOpenFile(
        tensor_file_path, true,
        [&param_optimizer_state_tensor_protos](int fd) {
          google::protobuf::io::FileInputStream input{fd};
          ORT_RETURN_IF_ERROR(ReadProtoMessageSequence(param_optimizer_state_tensor_protos, input));
          return Status::OK();
        });

    if (!file_read_status.IsOK()) {
      LOGS_DEFAULT(WARNING) << ToUTF8String(tensor_file_path) << " optimizer state file not found or read failure, skip it.";
      return Status::OK();
    }

    std::vector<const ONNX_NAMESPACE::TensorProto*> param_optimizer_state_tensor_proto_ptrs{};
    for (ONNX_NAMESPACE::TensorProto& param_optimizer_state_tensor_proto : param_optimizer_state_tensor_protos) {
      param_optimizer_state_tensor_proto_ptrs.emplace_back(&param_optimizer_state_tensor_proto);
    }

    std::unordered_map<std::string, OrtValue> name_to_ort_values;
    ORT_RETURN_IF_ERROR(CreateOrtValuesFromTensorProtos(param_optimizer_state_tensor_proto_ptrs, name_to_ort_values));

    for (auto& pair : name_to_ort_values) {
      auto& param_name = pair.first;
      if (param_optimizer_state.find(param_name) == param_optimizer_state.end()) {
        ParameterOptimizerState param_state;
        param_optimizer_state.insert({param_name, param_state});
      }
      param_optimizer_state[param_name].states_.insert({state_name, std::make_shared<OrtValue>(pair.second)});
    }
  }

  for (auto& filename : optim_property_filenames) {
    std::vector<std::string> results;
    StringSplit(filename, results);
    const std::string& group_name = results[1];

    if (grouped_optimizer_states.find(group_name) == grouped_optimizer_states.end()) {
      grouped_optimizer_states.insert({group_name, std::make_shared<GroupOptimizerState>()});
    }

    auto& optimizer_state_in_this_group = grouped_optimizer_states[group_name];

    // Parse group-wise properties.
    const std::string& cur_group_filename_prefix = StringConcat(k_optimizer_root_prefix, group_name);
    const PathString& tensor_file_path = GetTensorProtoPropertiesFilePath(optimizer_folder_path, cur_group_filename_prefix);
    std::vector<ONNX_NAMESPACE::TensorProto> group_wise_property_protos{};
    auto file_read_status = WithOpenFile(
        tensor_file_path, true,
        [&group_wise_property_protos](int fd) {
          google::protobuf::io::FileInputStream input{fd};
          ORT_RETURN_IF_ERROR(ReadProtoMessageSequence(group_wise_property_protos, input));
          return Status::OK();
        });

    if (!file_read_status.IsOK()) {
      LOGS_DEFAULT(WARNING) << ToUTF8String(tensor_file_path) << " optimizer state group-wise property file not found or read failure, skip it.";
      return Status::OK();
    }

    for (auto& property_proto : group_wise_property_protos) {
      if (property_proto.name().compare("learning_rate_") == 0) {
        optimizer_state_in_this_group->learning_rate_ = TypedCheckpointProperty<float>(property_proto).GetData();
      } else if (property_proto.name().compare("step_") == 0) {
        optimizer_state_in_this_group->step_ = TypedCheckpointProperty<int64_t>(property_proto).GetData();
      } else {
        continue;
      }
    }

    grouped_optimizer_states.insert({group_name, optimizer_state_in_this_group});
  }

  return Status::OK();
}

Status OrtLoadInternal(const PathString& checkpoint_path, CheckpointStates& states) {
  ORT_RETURN_IF_ERROR(OrtLoadModuleStatesInternal(checkpoint_path, states.module_checkpoint_states));

  ORT_RETURN_IF_ERROR(OrtLoadOptimizerStatesInternal(checkpoint_path, states.optimizer_checkpoint_states));

  // Parse other checkpoint properties.
  const PathString property_file_path = GetTensorProtoPropertiesFilePath(checkpoint_path, k_property_root_prefix);
  std::vector<ONNX_NAMESPACE::TensorProto> property_protos{};
  auto file_read_status = WithOpenFile(
      property_file_path, true,
      [&property_protos](int fd) {
        google::protobuf::io::FileInputStream input{fd};
        ORT_RETURN_IF_ERROR(ReadProtoMessageSequence(property_protos, input));
        return Status::OK();
      });

  if (!file_read_status.IsOK()) {
    LOGS_DEFAULT(WARNING) << ToUTF8String(property_file_path) << " custom property file not found or read failure, skip it.";
    return Status::OK();
  }

  PropertyBag& custom_properties = states.custom_properties;
  for (auto& property_proto : property_protos) {
    custom_properties.AddProperty(property_proto);
  }

  return Status::OK();
}

}  // namespace

Status SaveCheckpoint(const std::string& model_uri,
                      const std::vector<std::string>& trainable_param_names,
                      const PathString& checkpoint_path) {
  return OrtSaveInternal(model_uri, trainable_param_names, checkpoint_path);
}

Status SaveCheckpoint(CheckpointStates& states, const PathString& checkpoint_path) {
  return OrtSaveInternal(states, checkpoint_path);
}

Status LoadCheckpoint(const PathString& checkpoint_path, CheckpointStates& checkpoint_states) {
  return OrtLoadInternal(checkpoint_path, checkpoint_states);
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
