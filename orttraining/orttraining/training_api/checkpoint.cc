// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(ENABLE_TRAINING) && defined(ENABLE_TRAINING_ON_DEVICE)

#include "orttraining/core/framework/checkpointing.h"
#include "orttraining/training_api/checkpoint.h"
#include <type_traits>

namespace onnxruntime {
namespace training {
namespace api {

TypedCheckpointProperty::TypedCheckpointProperty(const ONNX_NAMESPACE::TensorProto& t) {
  auto& tensor_name = t.name();
  prop_name_ = tensor_name;

  auto data_type = t.data_type();
  switch (data_type) {
    case ONNX_NAMESPACE::TensorProto::FLOAT:
      prop_value_ = ONNX_NAMESPACE::ParseData<float>(t).at(0);
      break;
    case ONNX_NAMESPACE::TensorProto::STRING:
      prop_value_ = ONNX_NAMESPACE::ParseData<std::string>(t).at(0);
      break;
    case ONNX_NAMESPACE::TensorProto::INT64:
      prop_value_ = ONNX_NAMESPACE::ParseData<int64_t>(t).at(0);
      break;
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }
}

PathString CreateFolderIfNotExists(const PathString& path, const std::string& folder_name) {
  PathString new_folder_path = path + GetPathSep<PathChar>() + ORT_TSTR(folder_name);
  LOGS_DEFAULT_IF(Env::Default().FolderExists(new_folder_path), WARNING)
      << ToUTF8String(new_folder_path) << " directory exists - data may be overwritten.";

  ORT_RETURN_IF_ERROR(Env::Default().CreateFolder(new_folder_path));

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

    ORT_RETURN_IF_ERROR(utils::TensorProtoToMLValue(
        Env::Default(), nullptr, tensor_proto, mem_buffer,
        ort_value));

    name_to_ort_value.emplace(tensor_proto.name(), ort_value);
    tensor_buffers.emplace_back(std::move(tensor_buffer));
  }

  return Status::OK();
}

Status CheckpointUtils::Ort_Save_Internal(
    const std::vector<ONNX_NAMESPACE::TensorProto>& tensor_protos,
    const PathString& checkpoint_path) {
  std::unordered_map<std::string, OrtValue> name_to_ort_value;
  ORT_RETURN_IF_ERROR(CreateOrtValuesFromTensorProtos(tensor_protos, name_to_ort_value));

  CheckpointStates states;
  const auto& named_parameters = states.named_parameters;
  std::transform(name_to_ort_value.begin(), name_to_ort_value.end(),
                 std::inserter(named_parameters, named_parameters.begin()),
                 [](auto pair) { return {{pair.first, std::make_shared<api_test::Parameter>(pair.first, pair.second)}}; });

  ORT_RETURN_IF_ERROR(Ort_Save(states, checkpoint_path));

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
    const auto& param_states = states.named_parameters;
    std::unordered_map<std::string, OrtValue> model_parameter_ort_values;
    std::transform(param_states.begin(), param_states.end(),
                   std::inserter(model_parameter_ort_values, model_parameter_ort_values.begin()),
                   [](auto pair) { return {{pair.first, pair.second->data()}}; });
    ORT_RETURN_IF_ERROR(SaveRuntimeTensors(
        GetCheckpointTensorsFilePath(parameter_folder_path),
        GetCheckpointTensorsDataFilePath(parameter_folder_path),
        *train_session_data_transfer_mgr_, model_parameter_ort_values));
  }

  {
    // Write optimizer state tensors files.
    const PathString optimizer_folder_path = CreateFolderIfNotExists(checkpoint_path, "optimizers");

    // Currently we only have one single group, but it would be simple to extend
    // supporting multiple groups in the future.
    for (auto it : states.grouped_optimizer_states) {
      const std::string group_folder_name = it->first;
      const PathString cur_group_folder_path = CreateFolderIfNotExists(optimizer_folder_path, group_folder_name);

      // Write optimizer states for parameters in current group.
      // Under "group_<index>" folder, there will be multiple subfolders:
      // Each folder represent a optimizer state (for example, momentum_1, momentus_2 for Adam optimizers)

      std::unordered_map<std::string, std::unordered_map<std::string, OrtValue>> optimizer_state_ort_values;
      const auto& param_opt_states = it->second->param_optimizer_states_;
      for (auto& param_opt_state_it : param_opt_states) {
        const std::unordered_map<std::string, std::shared_ptr<OrtValue>>&
            param_opt_state = param_opt_state_it->second.states_;
        const std::string& state_name = param_opt_state->first;
        optimizer_state_ort_values[state_name].insert({{param_opt_state_it->first, *(param_opt_state->second)}});
      }
      for (auto it : optimizer_state_ort_values) {
        auto& state_name = it->first;
        const PathString opt_state_folder_path = CreateFolderIfNotExists(optimizer_folder_path, state_name);
        ORT_RETURN_IF_ERROR(SaveRuntimeTensors(
            GetCheckpointTensorsFilePath(opt_state_folder_path),
            GetCheckpointTensorsDataFilePath(opt_state_folder_path),
            *optimizer_session_data_transfer_mgr_, it->second));
      }

      // Storing group-wise properties.
      std::vector<CheckpointProperty> group_wise_properties{
          TypedCheckpointProperty("learning_rate_", it->second->learning_rate_),
          TypedCheckpointProperty("step_", it->second->step_)};
      std::vector<ONNX_NAMESPACE::TensorProto> group_wise_properties_tensor_protos;
      std::transform(group_wise_properties.begin(), group_wise_properties.end(),
                     std::inserter(group_wise_properties_tensor_protos,
                                   group_wise_properties_tensor_protos.begin()),
                     [](auto val) { return val.ToTensorProto(); });
      ORT_RETURN_IF_ERROR(SaveTensorProtosToFile(
          GetCheckpointPropertiesFilePath(cur_group_folder_path),
          group_wise_properties_tensor_protos));
    }
  }

  // Write properties file
  // Save properties into a checkpoint property file (with postfix .prop).
  const std::unordered_map<std::string, std::shared_ptr<CheckpointProperty>>& named_properties = states.named_properties;
  std::vector<ONNX_NAMESPACE::TensorProto> properties_tensor_protos;
  std::transform(named_properties.begin(), named_properties.end(),
                 std::inserter(properties_tensor_protos,
                               properties_tensor_protos.begin()),
                 [](auto pair) { return pair.second->ToTensorProto(); });
  ORT_RETURN_IF_ERROR(SaveTensorProtosToFile(
      GetCheckpointPropertiesFilePath(checkpoint_path),
      properties_tensor_protos));

  LOGS_DEFAULT(INFO) << "Model checkpoint saved successfully.";

  return Status::OK();
}

Status CheckpointUtils::Ort_Load_Internal(const PathString& checkpoint_path, CheckpointStates& states) {
  // Parameter parsing.
  {
    auto& named_parameters = states.named_parameters;
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

    std::transform(name_to_ort_values.begin(), name_to_ort_values.end(),
                   std::inserter(named_parameters, named_parameters.begin()),
                   [](auto pair) {
                     return {{pair.first, std::make_shared<api_test::Parameter>(pair.first, pair.second)}};
                   });
  }

  // Optimizer states parsing.
  {
    auto& grouped_optimizer_states = states.grouped_optimizer_states;
    const PathString optimizer_folder_path = checkpoint_path + GetPathSep<PathChar>() + ORT_TSTR("optimizers");

    std::unordered_map<std::string, PathString> group_folder_paths;
    LoopDir(optimizer_folder_path,
            [&group_folder_paths](const PathChar* filename, OrtFileType file_type) -> bool {
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
      auto optimizer_state_in_this_group = std::make_shared<OptimizerState>();
      std::unordered_map<std::string, ParameterOptimizerState>&
          param_optimizer_state = optimizer_state_in_this_group->param_optimizer_states_;

      std::unordered_map<std::string, PathString> param_optimizer_state_folder_paths_in_this_group;
      LoopDir(group.second,
              [&param_optimizer_state_folder_paths_in_this_group](
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
      for (auto& param_optimizer_state : param_optimizer_state_folder_paths_in_this_group) {
        auto& state_name = param_optimizer_state.first;
        std::vector<ONNX_NAMESPACE::TensorProto> param_optimizer_state_tensor_protos{};
        ORT_RETURN_IF_ERROR(WithOpenFile(
            GetCheckpointTensorsFilePath(param_optimizer_state.second), true,
            [&param_optimizer_state_tensor_protos](int fd) {
              google::protobuf::io::FileInputStream input{fd};
              ORT_RETURN_IF_ERROR(ReadProtoMessageSequence(param_optimizer_state_tensor_protos, input));
              return Status::OK();
            }));

        std::unordered_map<std::string, OrtValue> name_to_ort_values;
        ORT_RETURN_IF_ERROR(CreateOrtValuesFromTensorProtos(param_optimizer_state_tensor_protos, name_to_ort_values));

        for (auto& pair : name_to_ort_values) {
          auto& param_name = pair.first;
          std::unordered_map<std::string, std::shared_ptr<OrtValue>>
              states = {state_name, std::make_shared<OrtValue>(pair.second)};
          if (param_optimizer_state.find() == param_optimizer_state.end()) {
            param_optimizer_state.insert({param_name, states});
          } else {
            param_optimizer_state[param_name].insert(states);
          }
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
        auto property = TypedCheckpointProperty(property_proto);
        if (property.GetName().compare("learning_rate_") == 0) {
          optimizer_state_in_this_group->learning_rate_ = property.GetData<float>();
        } else if (property.GetName().compare("step_") == 0) {
          optimizer_state_in_this_group->step_ = property.GetData<int64_t>();
        } else {
          continue;
        }
      }

      grouped_optimizer_states.insert({group.first, optimizer_state_in_this_group});
    }
  }

  // Parse other checkpoint properties.
  {
    auto std::unordered_map<std::string, std::shared_ptr<CheckpointProperty>>&
        named_properties = states.grouped_optimizer_states;

    std::vector<ONNX_NAMESPACE::TensorProto> property_protos{};
    ORT_RETURN_IF_ERROR(WithOpenFile(
        GetCheckpointPropertiesFilePath(group.second), true,
        [&property_protos](int fd) {
          google::protobuf::io::FileInputStream input{fd};
          ORT_RETURN_IF_ERROR(ReadProtoMessageSequence(property_protos, input));
          return Status::OK();
        }));

    for (auto& property_proto : property_protos) {
      auto property_ptr = std::make_shared<TypedCheckpointProperty>(property_proto);
      named_properties.insert({property_ptr->GetName(), property_ptr});
    }
  }

  return Status::OK();
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime

#endif
