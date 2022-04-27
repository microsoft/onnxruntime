// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/providers/cpu/cpu_execution_provider.h"

namespace onnxruntime {
namespace training {
namespace api_test {

namespace {

constexpr const char* k_tensor_proto_file_name = ORT_TSTR("tensors.pbseq");
constexpr const char* k_external_data_file_name = ORT_TSTR("tensors.bin");
constexpr const char* k_tensor_proto_properties_file_name = ORT_TSTR("properties.pbseq");

constexpr const char* k_param_root_prefix = "param";
constexpr const char* k_optimizer_root_prefix = "optim";
constexpr const char* k_property_root_prefix = "prop";
constexpr const char* k_name_seperator = "_";

PathString GetTensorProtoFilePath(const PathString& checkpoint_directory, const std::string& filename_prefix) {
  return ConcatPathComponent<PathChar>(checkpoint_directory, ORT_TSTR(filename_prefix + k_name_seperator) + k_tensor_proto_file_name);
}

PathString GetExternalDataFilePath(const PathString& checkpoint_directory, const std::string& filename_prefix) {
  return ConcatPathComponent<PathChar>(checkpoint_directory, ORT_TSTR(filename_prefix + k_name_seperator) + k_external_data_file_name);
}

PathString GetTensorProtoPropertiesFilePath(const PathString& checkpoint_directory, const std::string& filename_prefix) {
  return ConcatPathComponent<PathChar>(checkpoint_directory, ORT_TSTR(filename_prefix + k_name_seperator) + k_tensor_proto_properties_file_name);
}

std::string StringConcat(const std::string& s_a, const std::string& s_b, const std::string& del = k_name_seperator) {
  return s_a + del + s_b;
}

void StringSplit(const std::string& s, std::vector<std::string>& results, const std::string& del = k_name_seperator) {
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
}  // namespace

Status CreateOrtValuesFromTensorProtos(
    const std::vector<const ONNX_NAMESPACE::TensorProto*>& tensor_protos,
    NameMLValMap& name_to_ort_value) {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  static AllocatorPtr cpu_allocator = cpu_provider.GetAllocator(0, OrtMemTypeDefault);

  for (const auto tensor_proto : tensor_protos) {
    TensorShape tensor_shape{utils::GetTensorShapeFromTensorProto(*tensor_proto)};
    const DataTypeImpl* tensor_dtype = DataTypeImpl::TensorTypeFromONNXEnum(tensor_proto->data_type())->GetElementType();
    auto p_tensor = std::make_unique<Tensor>(tensor_dtype, tensor_shape, cpu_allocator);
    ORT_RETURN_IF_ERROR(utils::TensorProtoToTensor(Env::Default(), nullptr, *tensor_proto, *p_tensor));

    OrtValue ort_value;
    ort_value.Init(p_tensor.release(), DataTypeImpl::GetType<Tensor>(), DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
    name_to_ort_value.emplace(tensor_proto->name(), ort_value);
  }

  return Status::OK();
}

Status CheckpointUtils::OrtSaveInternal(
    const std::string& model_uri,
    const std::vector<std::string>& trainable_param_names,
    const PathString& checkpoint_path) {
  auto logger_ptr = std::make_unique<logging::Logger>(logging::LoggingManager::DefaultLogger());
  std::shared_ptr<Model> p_model;
  ORT_RETURN_IF_ERROR(Model::Load(model_uri, p_model, nullptr, *logger_ptr));
  Graph& graph = p_model->MainGraph();

  std::vector<const ONNX_NAMESPACE::TensorProto*> trainable_weight_values;
  trainable_weight_values.reserve(trainable_param_names.size());
  for (size_t i = 0; i < trainable_param_names.size(); ++i) {
    const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
    ORT_ENFORCE(graph.GetInitializedTensor(trainable_param_names[i], tensor_proto),
                "Failed to find weight values: ", trainable_param_names[i]);
    trainable_weight_values.emplace_back(tensor_proto);
  }

  std::unordered_map<std::string, OrtValue> name_to_ort_value;
  ORT_RETURN_IF_ERROR(CreateOrtValuesFromTensorProtos(trainable_weight_values, name_to_ort_value));

  auto cpu_data_transfer = std::make_unique<CPUDataTransfer>();
  DataTransferManager data_transfer_mgr;
  auto st = data_transfer_mgr.RegisterDataTransfer(std::move(cpu_data_transfer));
  if (!st.IsOK()) {
    return st;
  }

  CheckpointStates states;
  states.module_checkpoint_states.train_session_data_transfer_mgr_ = &data_transfer_mgr;
  auto& named_parameters = states.module_checkpoint_states.named_parameters;
  for (auto it = name_to_ort_value.begin(); it != name_to_ort_value.end(); ++it) {
    auto param = std::make_shared<Parameter>(it->first, it->second);
    bool is_trainable =
        std::find(trainable_param_names.begin(), trainable_param_names.end(), param->Name()) != trainable_param_names.end();
    ORT_RETURN_IF_ERROR(param->SetRequiresGrad(is_trainable));
    named_parameters.insert({it->first, param});
  }

  ORT_RETURN_IF_ERROR(OrtSaveInternal(states, checkpoint_path));
  return Status::OK();
}

Status CheckpointUtils::OrtSaveInternal(
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
  const std::unordered_map<std::string, std::shared_ptr<CheckpointProperty>>& named_properties = states.named_properties;
  if (!named_properties.empty()) {
    std::vector<ONNX_NAMESPACE::TensorProto> properties_tensor_protos;
    for (auto it = named_properties.begin(); it != named_properties.end(); ++it) {
      properties_tensor_protos.emplace_back(it->second->ToTensorProto());
    }
    ORT_RETURN_IF_ERROR(SaveTensorProtosToFile(
        GetTensorProtoPropertiesFilePath(checkpoint_path, k_property_root_prefix),
        properties_tensor_protos));
  }

  LOGS_DEFAULT(INFO) << "Model checkpoint saved successfully.";
  return Status::OK();
}

Status CheckpointUtils::OrtSaveModuleStatesInternal(ModuleCheckpointStates& module_states,
                                                    const PathString& parameter_folder_path) {
  // Write weight tensors files.
  const auto& param_states = module_states.named_parameters;
  if (!param_states.empty()) {
    std::unordered_map<std::string, OrtValue> model_parameter_ort_values;
    for (auto it = param_states.begin(); it != param_states.end(); ++it) {
      model_parameter_ort_values.insert({it->first, it->second->Data()});
    }

    ORT_ENFORCE(module_states.train_session_data_transfer_mgr_,
                "module checkpoint state has null train_session_data_transfer_mgr.");
    ORT_RETURN_IF_ERROR(SaveRuntimeTensors(
        GetTensorProtoFilePath(parameter_folder_path, k_param_root_prefix),
        GetExternalDataFilePath(parameter_folder_path, k_param_root_prefix),
        *module_states.train_session_data_transfer_mgr_,
        model_parameter_ort_values));
  }

  return Status::OK();
}

Status CheckpointUtils::OrtSaveOptimizerStatesInternal(OptimizerCheckpointStates& optimizer_states,
                                                       const PathString& checkpoint_path) {
  if (optimizer_states.group_named_optimizer_states.empty()) {
    return Status::OK();
  }

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

      ORT_ENFORCE(optimizer_states.optimizer_session_data_transfer_mgr_,
                  "optimizer checkpoint state has null optimizer_session_data_transfer_mgr.");
      ORT_RETURN_IF_ERROR(SaveRuntimeTensors(
          GetTensorProtoFilePath(checkpoint_path, cur_state_filename_prefix),
          GetExternalDataFilePath(checkpoint_path, cur_state_filename_prefix),
          *optimizer_states.optimizer_session_data_transfer_mgr_,
          param_name_to_ortvalue));
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

    ORT_RETURN_IF_ERROR(SaveTensorProtosToFile(
        GetTensorProtoPropertiesFilePath(checkpoint_path, cur_group_filename_prefix),
        group_wise_properties_tensor_protos));
  }

  return Status::OK();
}

Status CheckpointUtils::OrtLoadInternal(const PathString& checkpoint_path, CheckpointStates& states) {
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
    LOGS_DEFAULT(WARNING) << ToUTF8String(property_file_path) << " optimizer state file read failure, skip it.";
    return Status::OK();
  }

  std::unordered_map<std::string, std::shared_ptr<CheckpointProperty>>& named_properties = states.named_properties;
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

  return Status::OK();
}

Status CheckpointUtils::OrtLoadModuleStatesInternal(
    const PathString& parameter_folder_path, ModuleCheckpointStates& module_states) {
  // Parameter parsing.
  const PathString module_state_file_path = GetTensorProtoFilePath(parameter_folder_path, k_param_root_prefix);
  auto& named_parameters = module_states.named_parameters;
  std::vector<ONNX_NAMESPACE::TensorProto> param_tensor_protos{};
  auto file_read_status = WithOpenFile(
      module_state_file_path, true,
      [&param_tensor_protos](int fd) {
        google::protobuf::io::FileInputStream input{fd};
        ORT_RETURN_IF_ERROR(ReadProtoMessageSequence(param_tensor_protos, input));
        return Status::OK();
      });

  if (!file_read_status.IsOK()) {
    LOGS_DEFAULT(WARNING) << ToUTF8String(module_state_file_path) << " module state file read failure, skip it.";
    return Status::OK();
  }

  std::vector<const ONNX_NAMESPACE::TensorProto*> param_tensor_proto_ptrs{};
  for (ONNX_NAMESPACE::TensorProto& param_tensor_proto : param_tensor_protos) {
    param_tensor_proto_ptrs.emplace_back(&param_tensor_proto);
  }

  std::unordered_map<std::string, OrtValue> name_to_ort_values;
  ORT_RETURN_IF_ERROR(CreateOrtValuesFromTensorProtos(param_tensor_proto_ptrs, name_to_ort_values));
  for (auto it = name_to_ort_values.begin(); it != name_to_ort_values.end(); ++it) {
    named_parameters.insert({it->first, std::make_shared<Parameter>(it->first, it->second)});
  }

  return Status::OK();
}

Status CheckpointUtils::OrtLoadOptimizerStatesInternal(
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

            if (StringEndsWith(filename_str, k_tensor_proto_file_name) ||
                StringEndsWith(filename_str, k_external_data_file_name)) {
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
      LOGS_DEFAULT(WARNING) << ToUTF8String(tensor_file_path) << " optimizer state file read failure, skip it.";
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
      LOGS_DEFAULT(WARNING) << ToUTF8String(tensor_file_path) << " optimizer state group-wise property file read failure, skip it.";
      return Status::OK();
    }

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

  return Status::OK();
}

}  // namespace api_test
}  // namespace training
}  // namespace onnxruntime
