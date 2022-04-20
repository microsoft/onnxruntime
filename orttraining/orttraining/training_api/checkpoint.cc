// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ENABLE_TRAINING) && defined(ENABLE_TRAINING_ON_DEVICE)

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/platform/path_lib.h"
#include "core/platform/env.h"
#include "orttraining/core/framework/checkpointing.h"
#include "orttraining/training_api/checkpoint.h"
#include <type_traits>
#include "core/util/protobuf_parsing_utils.h"
#include "orttraining/core/framework/protobuf_message_sequence.h"
#include "onnx/defs/tensor_proto_util.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace training {
namespace api_test {

PathString CreateFolderIfNotExists(const PathString& path, const std::string& folder_name) {
  PathString new_folder_path = path + GetPathSep<PathChar>() + ORT_TSTR(folder_name);
  LOGS_DEFAULT_IF(Env::Default().FolderExists(new_folder_path), WARNING)
      << ToUTF8String(new_folder_path) << " directory exists - data may be overwritten.";

  ORT_ENFORCE(Env::Default().CreateFolder(new_folder_path).IsOK());

  return new_folder_path;
}

Status CreateOrtValuesFromTensorProtos(
    const std::vector<ONNX_NAMESPACE::TensorProto>& tensor_protos,
    NameMLValMap& name_to_ort_value) {
  static const OrtMemoryInfo cpu_alloc_info{onnxruntime::CPU, OrtDeviceAllocator};
  std::vector<std::vector<char>> tensor_buffers{};

  for (const auto& tensor_proto : tensor_protos) {
    const auto* tensor_type = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto.data_type());
    const size_t element_size = tensor_type->GetElementType()->Size();
    const TensorShape shape{
        tensor_proto.dims().data(), static_cast<size_t>(tensor_proto.dims().size())};

    std::vector<char> tensor_buffer{};
    tensor_buffer.resize(element_size * shape.Size());

    const MemBuffer mem_buffer{tensor_buffer.data(), tensor_buffer.size(), cpu_alloc_info};

    OrtValue ort_value;

    ORT_RETURN_IF_ERROR(onnxruntime::utils::TensorProtoToMLValue(
        Env::Default(), nullptr, tensor_proto, mem_buffer,
        ort_value));

    name_to_ort_value.emplace(tensor_proto.name(), ort_value);
    tensor_buffers.emplace_back(std::move(tensor_buffer));
  }

  return Status::OK();
}

const std::vector<std::string> ParseStringData(
    const ONNX_NAMESPACE::TensorProto* tensor_proto) {
  ORT_ENFORCE(!tensor_proto->has_data_type() ||
                  tensor_proto->data_type() == ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED,
              "Invalid string data type.");
  ORT_ENFORCE(tensor_proto->data_type() != ONNX_NAMESPACE::TensorProto_DataType_STRING,
              "ParseStringData type mismatch for tensor ");

  ORT_ENFORCE(!(tensor_proto->has_data_location() &&
                tensor_proto->data_location() == ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL),
              "Cannot parse string data from external tensors.");

  ORT_ENFORCE(!tensor_proto->has_raw_data(),
              "stringcontent is required to be stored in repeated"
              "bytes string_data field. raw_data type cannot be string.");

  std::vector<std::string> res;
  const auto& data = tensor_proto->string_data();
  int expected_size = 1;
  for (int i = 0; i < tensor_proto->dims_size(); ++i) {
    expected_size *= tensor_proto->dims(i);
  }

  ORT_ENFORCE(tensor_proto->dims_size() != 0 && data.size() != expected_size, "Data size mismatch.");
  res.insert(res.end(), data.begin(), data.end());
  return res;
}

Status CheckpointUtils::Ort_Save_Internal(
    const std::vector<ONNX_NAMESPACE::TensorProto>& tensor_protos,
    const PathString& checkpoint_path) {
  std::unordered_map<std::string, OrtValue> name_to_ort_value;
  ORT_RETURN_IF_ERROR(CreateOrtValuesFromTensorProtos(tensor_protos, name_to_ort_value));

  CheckpointStates states;
  auto& named_parameters = states.module_checkpoint_states.named_parameters;
  for (auto it = name_to_ort_value.begin(); it != name_to_ort_value.end(); ++it) {
    named_parameters.insert({it->first, std::make_shared<Parameter>(it->first, it->second)});
  }

  ORT_RETURN_IF_ERROR(Ort_Save_Internal(states, checkpoint_path));
  return Status::OK();
}

Status CheckpointUtils::Ort_Save_Internal(
    CheckpointStates& states, const PathString& checkpoint_path) {
  LOGS_DEFAULT(INFO) << "Saving model checkpoint files to " << ToUTF8String(checkpoint_path);
  LOGS_DEFAULT_IF(Env::Default().FolderExists(checkpoint_path), WARNING)
      << "Checkpoint directory exists - data may be overwritten.";
  ORT_RETURN_IF_ERROR(Env::Default().CreateFolder(checkpoint_path));

  {
    // Write weight tensors files.
    const PathString parameter_folder_path = CreateFolderIfNotExists(checkpoint_path, "parameters");
    const auto& param_states = states.module_checkpoint_states.named_parameters;
    std::unordered_map<std::string, OrtValue> model_parameter_ort_values;
    for (auto it = param_states.begin(); it != param_states.end(); ++it) {
      model_parameter_ort_values.insert({it->first, it->second->data()});
    }

    ORT_RETURN_IF_ERROR(SaveRuntimeTensors(
        GetCheckpointTensorsFilePath(parameter_folder_path),
        GetCheckpointTensorsDataFilePath(parameter_folder_path),
        *states.module_checkpoint_states.train_session_data_transfer_mgr_, model_parameter_ort_values));
  }

  {
    // Write optimizer state tensors files.
    const PathString optimizer_folder_path = CreateFolderIfNotExists(checkpoint_path, "optimizers");

    // Currently we only have one single group, but it would be simple to extend
    // supporting multiple groups in the future.
    for (auto& group_named_optimizer_state : states.optimizer_checkpoint_states.group_named_optimizer_states) {
      const std::string& group_folder_name = group_named_optimizer_state.first;
      const std::shared_ptr<GroupOptimizerState>& group_optimizer_state_ptr = group_named_optimizer_state.second;

      const PathString cur_group_folder_path = CreateFolderIfNotExists(optimizer_folder_path, group_folder_name);

      // Write optimizer states for parameters in current group.
      // Under "group_<index>" folder, there will be multiple subfolders:
      // Each folder represent a optimizer state (for example, momentum_1, momentus_2 for Adam optimizers)

      // Re-organize optimizer_state_ort_values mapping
      // > Firstly indexed by moment state names;
      // > Secondly indexed by parameter names.
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
      for (auto& pair : optimizer_state_ort_values) {
        const auto& state_name = pair.first;
        const std::unordered_map<std::string, OrtValue>& param_name_to_ortvalue = pair.second;
        const PathString opt_state_folder_path = CreateFolderIfNotExists(optimizer_folder_path, state_name);
        ORT_RETURN_IF_ERROR(SaveRuntimeTensors(
            GetCheckpointTensorsFilePath(opt_state_folder_path),
            GetCheckpointTensorsDataFilePath(opt_state_folder_path),
            *states.optimizer_checkpoint_states.optimizer_session_data_transfer_mgr_,
            param_name_to_ortvalue));
      }

      //,
      // TypedCheckpointProperty("step_", group_optimizer_state_ptr->step_)

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

      ORT_RETURN_IF_ERROR(SaveTensorProtosToFile(
          GetCheckpointPropertiesFilePath(cur_group_folder_path),
          group_wise_properties_tensor_protos));
    }
  }

  // Write properties file
  // Save properties into a checkpoint property file (with postfix .prop).
  const std::unordered_map<std::string, std::shared_ptr<CheckpointProperty>>& named_properties = states.named_properties;
  std::vector<ONNX_NAMESPACE::TensorProto> properties_tensor_protos;
  for (auto it = named_properties.begin(); it != named_properties.end(); ++it) {
    properties_tensor_protos.emplace_back(it->second->ToTensorProto());
  }
  ORT_RETURN_IF_ERROR(SaveTensorProtosToFile(GetCheckpointPropertiesFilePath(checkpoint_path),
                                             properties_tensor_protos));

  LOGS_DEFAULT(INFO) << "Model checkpoint saved successfully.";
  return Status::OK();
}

Status CheckpointUtils::Ort_Load_Internal(const PathString& checkpoint_path, CheckpointStates& states) {
  // Parameter parsing.
  {
    auto& named_parameters = states.module_checkpoint_states.named_parameters;
    PathString param_folder_path = checkpoint_path + GetPathSep<PathChar>() + ORT_TSTR("parameters");
    std::vector<ONNX_NAMESPACE::TensorProto> param_tensor_protos{};
    ORT_RETURN_IF_ERROR(WithOpenFile(
        GetCheckpointTensorsFilePath(param_folder_path), true,
        [&param_tensor_protos](int fd) {
          google::protobuf::io::FileInputStream input{fd};
          ORT_RETURN_IF_ERROR(ReadProtoMessageSequence(param_tensor_protos, input));
          return Status::OK();
        }));

    std::unordered_map<std::string, OrtValue> name_to_ort_values;
    ORT_RETURN_IF_ERROR(CreateOrtValuesFromTensorProtos(param_tensor_protos, name_to_ort_values));
    for (auto it = name_to_ort_values.begin(); it != name_to_ort_values.end(); ++it) {
      named_parameters.insert({it->first, std::make_shared<Parameter>(it->first, it->second)});
    }
  }

  // Optimizer states parsing.
  {
    auto& grouped_optimizer_states = states.optimizer_checkpoint_states.group_named_optimizer_states;
    const PathString optimizer_folder_path = checkpoint_path + GetPathSep<PathChar>() + ORT_TSTR("optimizers");

    std::unordered_map<std::string, PathString> group_folder_paths;
    LoopDir(optimizer_folder_path,
            [&group_folder_paths, &optimizer_folder_path](const PathChar* filename, OrtFileType file_type) -> bool {
              PathString filename_str = filename;
              if (filename_str[0] == '.' ||
                  file_type != OrtFileType::TYPE_DIR) {
                return true;
              }
              group_folder_paths.insert({filename_str, ConcatPathComponent<PathChar>(optimizer_folder_path, filename_str)});
              return true;
            });

    // Go though every group.
    for (auto& group : group_folder_paths) {
      const auto& group_name = group.first;
      const auto& group_folder_path = group.second;
      auto optimizer_state_in_this_group = std::make_shared<GroupOptimizerState>();
      std::unordered_map<std::string, ParameterOptimizerState>&
          param_optimizer_state = optimizer_state_in_this_group->param_named_optimizer_states_;

      std::unordered_map<std::string, PathString> param_optimizer_state_folder_paths_in_this_group;
      LoopDir(group.second,
              [&param_optimizer_state_folder_paths_in_this_group, &group_folder_path](
                  const PathChar* filename, OrtFileType file_type) -> bool {
                PathString filename_str = filename;
                if (filename_str[0] == '.' ||
                    file_type != OrtFileType::TYPE_DIR) {
                  return true;
                }
                param_optimizer_state_folder_paths_in_this_group.insert(
                    {filename_str, ConcatPathComponent<PathChar>(group_folder_path, filename_str)});
                return true;
              });

      // Process momentum_1 for all parameters in the first iteration; then momentum_2 in the second iteration.
      for (auto& state_name_to_folder : param_optimizer_state_folder_paths_in_this_group) {
        auto& state_name = state_name_to_folder.first;
        std::vector<ONNX_NAMESPACE::TensorProto> param_optimizer_state_tensor_protos{};
        ORT_RETURN_IF_ERROR(WithOpenFile(
            GetCheckpointTensorsFilePath(state_name_to_folder.second), true,
            [&param_optimizer_state_tensor_protos](int fd) {
              google::protobuf::io::FileInputStream input{fd};
              ORT_RETURN_IF_ERROR(ReadProtoMessageSequence(param_optimizer_state_tensor_protos, input));
              return Status::OK();
            }));

        std::unordered_map<std::string, OrtValue> name_to_ort_values;
        ORT_RETURN_IF_ERROR(CreateOrtValuesFromTensorProtos(param_optimizer_state_tensor_protos, name_to_ort_values));

        for (auto& pair : name_to_ort_values) {
          auto& param_name = pair.first;
          if (param_optimizer_state.find(param_name) == param_optimizer_state.end()) {
            ParameterOptimizerState param_state;
            param_optimizer_state.insert({param_name, param_state});
          }
          param_optimizer_state[param_name].states_.insert({state_name, std::make_shared<OrtValue>(pair.second)});
        }
      }

      // Parse group-wise properties.
      std::vector<ONNX_NAMESPACE::TensorProto> group_wise_property_protos{};
      ORT_RETURN_IF_ERROR(WithOpenFile(
          GetCheckpointPropertiesFilePath(group.second), true,
          [&group_wise_property_protos](int fd) {
            google::protobuf::io::FileInputStream input{fd};
            ORT_RETURN_IF_ERROR(ReadProtoMessageSequence(group_wise_property_protos, input));
            return Status::OK();
          }));

      for (auto& property_proto : group_wise_property_protos) {
        if (property_proto.name().compare("learning_rate_") == 0) {
          optimizer_state_in_this_group->learning_rate_ = ONNX_NAMESPACE::ParseData<float>(&property_proto).at(0);
        } else if (property_proto.name().compare("step_") == 0) {
          optimizer_state_in_this_group->step_ = ONNX_NAMESPACE::ParseData<int64_t>(&property_proto).at(0);
        } else {
          continue;
        }
      }

      grouped_optimizer_states.insert({group_name, optimizer_state_in_this_group});
    }
  }

  // Parse other checkpoint properties.
  {
    std::unordered_map<std::string, std::shared_ptr<CheckpointProperty>>&
        named_properties = states.named_properties;

    std::vector<ONNX_NAMESPACE::TensorProto> property_protos{};
    ORT_RETURN_IF_ERROR(WithOpenFile(
        GetCheckpointPropertiesFilePath(checkpoint_path), true,
        [&property_protos](int fd) {
          google::protobuf::io::FileInputStream input{fd};
          ORT_RETURN_IF_ERROR(ReadProtoMessageSequence(property_protos, input));
          return Status::OK();
        }));

    for (auto& property_proto : property_protos) {
      const std::string& tensor_name = property_proto.name();
      auto data_type = property_proto.data_type();
      switch (data_type) {
        case ONNX_NAMESPACE::TensorProto::FLOAT: {
          const std::vector<float>& flt_parsed = ONNX_NAMESPACE::ParseData<float>(&property_proto);
          ORT_ENFORCE(flt_parsed.size() == static_cast<size_t>(1), "only support scalar float properties.");
          named_properties.insert({tensor_name,
                                   std::make_shared<TypedCheckpointProperty<float>>(
                                       tensor_name,
                                       flt_parsed.at(0))});
          break;
        }
        case ONNX_NAMESPACE::TensorProto::STRING: {
          const std::vector<std::string>& str_parsed = ParseStringData(&property_proto);
          ORT_ENFORCE(str_parsed.size() == static_cast<size_t>(1), "only support scalar string properties.");
          named_properties.insert({tensor_name,
                                   std::make_shared<TypedCheckpointProperty<std::string>>(
                                       tensor_name,
                                       str_parsed.at(0))});
          break;
        }
        case ONNX_NAMESPACE::TensorProto::INT64: {
          const std::vector<int64_t>& int_parsed = ONNX_NAMESPACE::ParseData<int64_t>(&property_proto);
          ORT_ENFORCE(int_parsed.size() == static_cast<size_t>(1), "only support scalar int64_t properties.");
          named_properties.insert({tensor_name,
                                   std::make_shared<TypedCheckpointProperty<int64_t>>(
                                       tensor_name,
                                       int_parsed.at(0))});
          break;
        }
        default:
          ORT_THROW("Unsupported input data type of ", data_type);
      }
    }
  }

  return Status::OK();
}

}  // namespace api_test
}  // namespace training
}  // namespace onnxruntime

#endif
