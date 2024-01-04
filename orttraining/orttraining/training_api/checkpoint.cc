// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_api/checkpoint.h"

#include "core/flatbuffers/checkpoint_version.h"
#include "core/flatbuffers/schema/ort_training_checkpoint.fbs.h"
#include "core/framework/framework_common.h"
#include "core/graph/graph_flatbuffers_utils.h"

namespace onnxruntime::training::api {

namespace {

/**
 * @brief Sort keys of a hash map.
 * @param hash_map Hash map to sort.
 * @param keys Sorted keys.
 */
template <typename T>
InlinedVector<std::string> SortedKeys(const T& hash_map) {
  InlinedVector<std::string> keys;
  keys.reserve(hash_map.size());
  for ([[maybe_unused]] const auto& [key, value] : hash_map) {
    keys.push_back(key);
  }
  std::sort(keys.begin(), keys.end());

  return keys;
}

/**
 * @brief Create flatbuffer tensor from OrtValue object
 *
 * @param tensor_name Name of the tensor
 * @param ort_value OrtValue object to save to the flatbuffer tensor.
 * @param copy_tensor Function to copy the tensor to a cpu buffer.
 * @param builder Builder to create flatbuffer tensors.
 * @param flatbuffer_tensor Flatbuffer tensor to be populated.
 * @return Status of the operation.
 */
Status FlatbufferTensorFromOrtValue(
    const std::string& tensor_name, const OrtValue& ort_value,
    const std::function<Status(const onnxruntime::Tensor& src_tensor, onnxruntime::Tensor& dst_tensor)> copy_tensor,
    flatbuffers::FlatBufferBuilder& builder,
    flatbuffers::Offset<fbs::Tensor>& fbs_tensor) {
  // Check if the OrtValue is a tensor.
  ORT_RETURN_IF_NOT(ort_value.IsTensor(), "Only tensor OrtValues can be saved to a checkpoint.");
  const onnxruntime::Tensor& src_tensor = ort_value.Get<onnxruntime::Tensor>();

  // Check if the tensor is on CPU. If not, we need to copy the tensor to CPU before saving it.
  if (const auto& tensor_location = src_tensor.Location();
      tensor_location.device.Type() != OrtDevice::CPU) {
    InlinedVector<uint8_t> tensor_data_buffer{};
    tensor_data_buffer.resize(src_tensor.SizeInBytes());
    const OrtMemoryInfo cpu_alloc_info{onnxruntime::CPU, OrtDeviceAllocator};
    onnxruntime::Tensor dst_tensor{src_tensor.DataType(), src_tensor.Shape(), tensor_data_buffer.data(), cpu_alloc_info};
    ORT_RETURN_IF_ERROR(copy_tensor(src_tensor, dst_tensor));
    ORT_RETURN_IF_ERROR(fbs::utils::SaveOrtTensorOrtFormat(tensor_name, dst_tensor, builder, fbs_tensor));
  } else {
    ORT_RETURN_IF_ERROR(fbs::utils::SaveOrtTensorOrtFormat(tensor_name, src_tensor, builder, fbs_tensor));
  }

  return Status::OK();
}

/**
 * @brief Create OrtValue object from flatbuffer tensor
 *
 * @param fbs_tensor Flatbuffer tensor.
 * @param tensor_name Name of the tensor.
 * @param ort_value OrtValue object to be populated.
 * @return Status of the operation.
 */
Status OrtValueFromFlatbufferTensor(const fbs::Tensor& fbs_tensor,
                                    std::string& tensor_name, OrtValue& ort_value) {
  // The assumption is that the flatbuffer buffer will be destructed once the checkpoint has been loaded.
  // And so, we must allocate a buffer where the tensor data can be copied using the cpu allocator.
  // This buffer is owned by the OrtValue.
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  AllocatorPtr cpu_allocator = cpu_provider.CreatePreferredAllocators()[0];

  std::unique_ptr<Tensor> ort_tensor = std::make_unique<Tensor>();
  ORT_RETURN_IF_ERROR(fbs::utils::LoadOrtTensorOrtFormat(fbs_tensor, cpu_allocator, tensor_name, *ort_tensor));

  ort_value.Init(ort_tensor.release(), DataTypeImpl::GetType<onnxruntime::Tensor>(),
                 DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc());

  return Status::OK();
}

/**
 * @brief Create flatbuffer tensors from OrtValue objects
 *
 * @param name_to_ort_value Name to OrtValue map.
 * @param data_transfer_manager Data transfer manager to copy the OrtValue tensor to a cpu buffer.
 * @param builder Builder to create flatbuffer tensors.
 * @param flatbuffer_tensors Flatbuffer tensors to be populated.
 * @return Status of the operation.
 */
Status FlatbufferTensorsFromOrtValues(
    const InlinedHashMap<std::string, OrtValue>& name_to_ort_value,
    const DataTransferManager* data_transfer_manager,
    flatbuffers::FlatBufferBuilder& builder,
    std::vector<flatbuffers::Offset<fbs::Tensor>>& flatbuffer_tensors) {
  for (const auto& name : SortedKeys(name_to_ort_value)) {
    const OrtValue& ort_value = name_to_ort_value.at(name);
    flatbuffers::Offset<fbs::Tensor> fbs_tensor;
    ORT_RETURN_IF_ERROR(FlatbufferTensorFromOrtValue(
        name, ort_value, [&data_transfer_manager](const auto& src_tensor, auto& dst_tensor) {
          ORT_RETURN_IF_NOT(data_transfer_manager,
                            "Cannot save OrtValue to a checkpoint. Expected: A valid data transfer manager. ",
                            "Actual: nullptr.");
          return data_transfer_manager->CopyTensor(src_tensor, dst_tensor);
        },
        builder, fbs_tensor));
    flatbuffer_tensors.push_back(fbs_tensor);
  }

  return Status::OK();
}

/**
 * @brief Create OrtValue objects from flatbuffer tensors.
 *
 * @param flatbuffer_tensors Flatbuffer tensors.
 * @param name_to_ort_value Name to OrtValue map to be populated.
 * @return Status of the operation.
 */
Status OrtValuesFromFlatbufferTensors(
    const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::Tensor>>& flatbuffer_tensors,
    InlinedHashMap<std::string, OrtValue>& name_to_ort_value) {
  for (const auto* fbs_tensor : flatbuffer_tensors) {
    ORT_RETURN_IF_NOT(fbs_tensor, "Encountered a nullptr flatbuffer tensor. Checkpoint file is invalid.");

    std::string tensor_name;
    OrtValue ort_value;
    ORT_RETURN_IF_ERROR(OrtValueFromFlatbufferTensor(*fbs_tensor, tensor_name, ort_value));
    name_to_ort_value.emplace(std::move(tensor_name), std::move(ort_value));
  }

  return Status::OK();
}

namespace save {

/**
 * @brief Save from a checkpoint flatbuffer to file.
 * @param checkpoint_path Path to save the checkpoint file.
 * @param builder Flatbuffer builder containing the checkpoint buffer.
 * @return Status of the operation.
 *
 */
Status ToFile(const PathString& checkpoint_path, flatbuffers::FlatBufferBuilder& builder) {
  std::ofstream file(checkpoint_path, std::ios::binary);
  const uint8_t* buf = builder.GetBufferPointer();
  int size = builder.GetSize();
  file.write(reinterpret_cast<const char*>(buf), size);
  ORT_RETURN_IF_NOT(file, "Failed to save checkpoint to file: ", ToUTF8String(checkpoint_path));

  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD)
/**
 * @brief Save from ONNX initializers to a checkpoint file.
 *
 * @param trainable_tensor_protos trainable parameters in TensorProto format.
 * @param non_trainable_tensor_protos non-trainable parameters in TensorProto format.
 * @param checkpoint_path file where checkpoint is saved.
 * @return Status
 */
Status FromTensorProtos(
    gsl::span<const ONNX_NAMESPACE::TensorProto> trainable_tensor_protos,
    gsl::span<const ONNX_NAMESPACE::TensorProto> non_trainable_tensor_protos,
    const PathString& checkpoint_path) {
  const auto check_unique = [](gsl::span<const ONNX_NAMESPACE::TensorProto> tensor_protos,
                               InlinedHashSet<std::string>& unique_names) {
    for (const auto& tensor_proto : tensor_protos) {
      ORT_RETURN_IF_NOT(unique_names.find(tensor_proto.name()) == unique_names.end(),
                        "Duplicated tensor proto named ", tensor_proto.name());
      unique_names.insert(tensor_proto.name());
    }

    return Status::OK();
  };

  // Make sure name unique across trainable and non-trainable lists.
  InlinedHashSet<std::string> unique_names;
  unique_names.reserve(trainable_tensor_protos.size() + non_trainable_tensor_protos.size());
  ORT_RETURN_IF_ERROR(check_unique(trainable_tensor_protos, unique_names));
  ORT_RETURN_IF_ERROR(check_unique(non_trainable_tensor_protos, unique_names));

  constexpr size_t m_bytes = 1024 * 1024;
  size_t fbs_buffer_size = 0U;
  for (const auto& tensor_proto : trainable_tensor_protos) {
    fbs_buffer_size += tensor_proto.ByteSizeLong();
  }
  for (const auto& tensor_proto : non_trainable_tensor_protos) {
    fbs_buffer_size += tensor_proto.ByteSizeLong();
  }

  // Align buffer size to 1MB.
  fbs_buffer_size = std::max(fbs_buffer_size, m_bytes);
  fbs_buffer_size = ((fbs_buffer_size + m_bytes - 1) / m_bytes) * m_bytes;
  flatbuffers::FlatBufferBuilder builder(fbs_buffer_size);

  const auto tensor_protos_to_fbs_tensors = [&builder](const auto& tensor_protos, auto& fbs_tensors) {
    fbs_tensors.reserve(tensor_protos.size());
    for (const auto& tensor_proto : tensor_protos) {
      flatbuffers::Offset<fbs::Tensor> fbs_tensor;
      ORT_RETURN_IF_ERROR(
          fbs::utils::SaveInitializerOrtFormat(builder, tensor_proto, Path(), fbs_tensor));
      fbs_tensors.push_back(fbs_tensor);
    }

    return Status::OK();
  };

  std::vector<flatbuffers::Offset<fbs::Tensor>> trainable_tensors;
  ORT_RETURN_IF_ERROR(tensor_protos_to_fbs_tensors(trainable_tensor_protos, trainable_tensors));

  std::vector<flatbuffers::Offset<fbs::Tensor>> non_trainable_tensors;
  ORT_RETURN_IF_ERROR(tensor_protos_to_fbs_tensors(non_trainable_tensor_protos, non_trainable_tensors));

  const auto fbs_trainable_tensors = builder.CreateVector(trainable_tensors);
  const auto fbs_non_trainable_tensors = builder.CreateVector(non_trainable_tensors);

  fbs::ModuleStateBuilder module_state_builder(builder);
  module_state_builder.add_requires_grad_params(fbs_trainable_tensors);
  module_state_builder.add_frozen_params(fbs_non_trainable_tensors);
  flatbuffers::Offset<fbs::ModuleState> fbs_module_state = module_state_builder.Finish();

  fbs::CheckpointBuilder checkpoint_builder(builder);
  checkpoint_builder.add_version(kCheckpointVersion);
  checkpoint_builder.add_module_state(fbs_module_state);
  const auto checkpoint = checkpoint_builder.Finish();
  builder.Finish(checkpoint, fbs::CheckpointIdentifier());

  return save::ToFile(checkpoint_path, builder);
}
#endif

/**
 * @brief Save from the module state to a flatbuffer checkpoint module state.
 *
 * @param module_state module state containing the model's trainable and non-trainable parameters.
 * @param builder Flatbuffer builder.
 * @param fbs_module_state Flatbuffer module state to be populated.
 * @return Status of the operation.
 */
Status FromModuleState(const ModuleCheckpointState& module_state,
                       flatbuffers::FlatBufferBuilder& builder,
                       flatbuffers::Offset<fbs::ModuleState>& fbs_module_state) {
  if (module_state.named_parameters.empty()) {
    return Status::OK();
  }

  size_t num_requires_grad_params = std::count_if(
      module_state.named_parameters.begin(), module_state.named_parameters.end(),
      [](const auto& name_param_pair) { return name_param_pair.second->RequiresGrad(); });
  size_t num_frozen_params = module_state.named_parameters.size() - num_requires_grad_params;

  InlinedHashMap<std::string, OrtValue> requires_grad_params;
  requires_grad_params.reserve(num_requires_grad_params);
  InlinedHashMap<std::string, OrtValue> frozen_params;
  frozen_params.reserve(num_frozen_params);
  for (const auto& [name, value] : module_state.named_parameters) {
    if (value->RequiresGrad()) {
      requires_grad_params.insert({name, value->Data()});
    } else {
      frozen_params.insert({name, value->Data()});
    }
  }

  std::vector<flatbuffers::Offset<fbs::Tensor>> trainable_tensors;
  trainable_tensors.reserve(requires_grad_params.size());
  ORT_RETURN_IF_ERROR(FlatbufferTensorsFromOrtValues(
      requires_grad_params,
      module_state.train_session_data_transfer_mgr,
      builder, trainable_tensors));

  std::vector<flatbuffers::Offset<fbs::Tensor>> non_trainable_tensors;
  non_trainable_tensors.reserve(frozen_params.size());
  ORT_RETURN_IF_ERROR(FlatbufferTensorsFromOrtValues(
      frozen_params,
      module_state.train_session_data_transfer_mgr,
      builder, non_trainable_tensors));

  const auto fbs_trainable_tensors = builder.CreateVector(trainable_tensors);
  const auto fbs_non_trainable_tensors = builder.CreateVector(non_trainable_tensors);

  fbs::ModuleStateBuilder module_state_builder(builder);
  module_state_builder.add_requires_grad_params(fbs_trainable_tensors);
  module_state_builder.add_frozen_params(fbs_non_trainable_tensors);
  fbs_module_state = module_state_builder.Finish();

  return Status::OK();
}

/**
 * @brief Save from the optimizer state to a flatbuffer checkpoint optimizer state.
 *
 * @param optimizer_state optimizer state containing the optimizer's state (for example learning rate, step, first
 *                        and second order momentums ...).
 * @param builder Flatbuffer builder.
 * @param fbs_optimizer_groups Flatbuffer optimizer groups to be populated.
 * @return Status of the operation.
 */
Status FromOptimizerState(const OptimizerCheckpointState& optimizer_state,
                          flatbuffers::FlatBufferBuilder& builder,
                          std::vector<flatbuffers::Offset<fbs::OptimizerGroup>>& fbs_optimizer_groups) {
  if (optimizer_state.group_named_optimizer_states.empty()) {
    return Status::OK();
  }

  fbs_optimizer_groups.reserve(optimizer_state.group_named_optimizer_states.size());
  for (const auto& group_name : SortedKeys(optimizer_state.group_named_optimizer_states)) {
    const std::shared_ptr<GroupOptimizerState>& group_optimizer_state_ptr =
        optimizer_state.group_named_optimizer_states.at(group_name);

    std::vector<flatbuffers::Offset<fbs::ParameterOptimizerState>> optimizer_states;
    optimizer_states.reserve(group_optimizer_state_ptr->param_named_optimizer_states.size());

    for (const auto& param_name : SortedKeys(group_optimizer_state_ptr->param_named_optimizer_states)) {
      const auto& param_optimizer_state = group_optimizer_state_ptr->param_named_optimizer_states.at(param_name);
      std::vector<flatbuffers::Offset<fbs::Tensor>> momentums;
      momentums.reserve(param_optimizer_state.size());
      ORT_RETURN_IF_ERROR(FlatbufferTensorsFromOrtValues(
          param_optimizer_state,
          optimizer_state.optimizer_session_data_transfer_mgr,
          builder, momentums));

      const auto fbs_param_name = builder.CreateString(param_name);
      const auto fbs_momentums = builder.CreateVector(momentums);
      fbs::ParameterOptimizerStateBuilder optimizer_state_builder(builder);
      optimizer_state_builder.add_param_name(fbs_param_name);
      optimizer_state_builder.add_momentums(fbs_momentums);

      flatbuffers::Offset<fbs::ParameterOptimizerState> fbs_optimizer_state = optimizer_state_builder.Finish();
      optimizer_states.push_back(fbs_optimizer_state);
    }

    const auto fbs_group_name = builder.CreateString(group_name);
    const auto fbs_optimizer_states = builder.CreateVector(optimizer_states);

    fbs::OptimizerGroupBuilder optimizer_state_builder(builder);
    optimizer_state_builder.add_group_name(fbs_group_name);
    optimizer_state_builder.add_initial_learning_rate(group_optimizer_state_ptr->initial_lr);
    optimizer_state_builder.add_step(group_optimizer_state_ptr->step);
    optimizer_state_builder.add_optimizer_states(fbs_optimizer_states);
    auto fbs_optimizer_group = optimizer_state_builder.Finish();
    fbs_optimizer_groups.push_back(fbs_optimizer_group);
  }

  return Status::OK();
}

/**
 * @brief Save from user defined properties to a flatbuffer checkpoint property bag.
 *
 * @param property_bag user defined properties.
 * @param builder Flatbuffer builder.
 * @param fbs_property_bag Flatbuffer property bag to be populated.
 * @return Status of the operation.
 */
Status FromPropertyBag(const PropertyBag& property_bag, flatbuffers::FlatBufferBuilder& builder,
                       flatbuffers::Offset<fbs::PropertyBag>& fbs_property_bag) {
  if (property_bag.size() == 0U) {
    return Status::OK();
  }

  std::vector<flatbuffers::Offset<fbs::IntProperty>> ints;
  std::vector<flatbuffers::Offset<fbs::FloatProperty>> floats;
  std::vector<flatbuffers::Offset<fbs::StringProperty>> strings;
  for (const auto& name : SortedKeys(property_bag)) {
    const auto& value = property_bag.GetProperty<PropertyDataType>(name);
    const auto fbs_property_name = builder.CreateString(name);
    if (std::holds_alternative<int64_t>(value)) {
      fbs::IntPropertyBuilder int_property_builder(builder);
      int_property_builder.add_name(fbs_property_name);
      int_property_builder.add_value(std::get<int64_t>(value));
      flatbuffers::Offset<fbs::IntProperty> property = int_property_builder.Finish();
      ints.push_back(property);
    } else if (std::holds_alternative<float>(value)) {
      fbs::FloatPropertyBuilder float_property_builder(builder);
      float_property_builder.add_name(fbs_property_name);
      float_property_builder.add_value(std::get<float>(value));
      flatbuffers::Offset<fbs::FloatProperty> property = float_property_builder.Finish();
      floats.push_back(property);
    } else if (std::holds_alternative<std::string>(value)) {
      const auto fbs_property_value = builder.CreateString(std::get<std::string>(value));
      fbs::StringPropertyBuilder string_property_builder(builder);
      string_property_builder.add_name(fbs_property_name);
      string_property_builder.add_value(fbs_property_value);
      flatbuffers::Offset<fbs::StringProperty> property = string_property_builder.Finish();
      strings.push_back(property);
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown property type encountered in the property bag.");
    }
  }

  const auto fbs_ints = builder.CreateVector(ints);
  const auto fbs_floats = builder.CreateVector(floats);
  const auto fbs_strings = builder.CreateVector(strings);

  fbs::PropertyBagBuilder property_bag_builder(builder);
  property_bag_builder.add_ints(fbs_ints);
  property_bag_builder.add_floats(fbs_floats);
  property_bag_builder.add_strings(fbs_strings);
  fbs_property_bag = property_bag_builder.Finish();

  return Status::OK();
}

/**
 * @brief Save from a checkpoint state to a checkpoint file.
 *
 * @param state parameter/optimizer and other user defined training states.
 * @param checkpoint_path file where checkpoint is saved.
 * @param include_optimizer_state Whether to include optimizer state in the checkpoint.
 * @return Status of the operation.
 */
Status FromCheckpointState(
    const CheckpointState& state, const PathString& checkpoint_path, const bool include_optimizer_state) {
  flatbuffers::FlatBufferBuilder builder(1024);

  // Write weight tensors files.
  flatbuffers::Offset<fbs::ModuleState> module_state;
  ORT_RETURN_IF_ERROR(FromModuleState(state.module_checkpoint_state, builder, module_state));

  // Write optimizer state tensors files.
  std::vector<flatbuffers::Offset<fbs::OptimizerGroup>> optimizer_groups;
  if (include_optimizer_state) {
    ORT_RETURN_IF_ERROR(FromOptimizerState(state.optimizer_checkpoint_state, builder, optimizer_groups));
  }

  flatbuffers::Offset<fbs::PropertyBag> property_bag;
  ORT_RETURN_IF_ERROR(FromPropertyBag(state.property_bag, builder, property_bag));

  const auto fbs_optimizer_groups = builder.CreateVector(optimizer_groups);

  fbs::CheckpointBuilder checkpoint_builder(builder);
  checkpoint_builder.add_version(kCheckpointVersion);
  checkpoint_builder.add_module_state(module_state);
  checkpoint_builder.add_optimizer_groups(fbs_optimizer_groups);
  checkpoint_builder.add_property_bag(property_bag);
  const auto checkpoint = checkpoint_builder.Finish();
  builder.Finish(checkpoint, fbs::CheckpointIdentifier());

  return save::ToFile(checkpoint_path, builder);
}

}  // namespace save

namespace load {

/**
 * @brief Load checkpoint flatbuffer from file.
 * @param checkpoint_path Path to the checkpoint file.
 * @param checkpoint_bytes Contents of the checkpoint file in bytes.
 * @param checkpoint_span Checkpoint bytes represented as a span.
 * @return Status of the operation.
 *
 */
Status FromFile(const PathString& checkpoint_path, InlinedVector<uint8_t>& checkpoint_bytes) {
  size_t num_bytes = 0;
  ORT_RETURN_IF_ERROR(Env::Default().GetFileLength(checkpoint_path.c_str(), num_bytes));
  checkpoint_bytes.resize(num_bytes);

  std::ifstream bytes_stream(checkpoint_path, std::ifstream::in | std::ifstream::binary);
  bytes_stream.read(reinterpret_cast<char*>(checkpoint_bytes.data()), num_bytes);

  ORT_RETURN_IF_NOT(bytes_stream, "Loading checkpoint from ", ToUTF8String(checkpoint_path), " failed. Only ",
                    bytes_stream.gcount(), "/", num_bytes, " bytes could be read.");

  return Status::OK();
}

/**
 * @brief Load from a flatbuffer checkpoint module state to a module state.
 *
 * @param fbs_module_state Flatbuffer module state.
 * @param module_state Module state to be populated.
 * @return Status of the operation.
 */
Status ToModuleState(
    const onnxruntime::fbs::ModuleState& fbs_module_state, ModuleCheckpointState& module_state) {
  const auto* requires_grad_params = fbs_module_state.requires_grad_params();
  ORT_RETURN_IF_NOT(requires_grad_params, "Expected: Valid trainable tensors flatbuffer.",
                    " Actual: Encountered a nullptr. Checkpoint file is invalid");
  flatbuffers::uoffset_t trainable_params_size = requires_grad_params->size();
  InlinedHashMap<std::string, OrtValue> trainable_params;
  trainable_params.reserve(trainable_params_size);
  ORT_RETURN_IF_ERROR(OrtValuesFromFlatbufferTensors(*requires_grad_params, trainable_params));

  for (auto& [name, value] : trainable_params) {
    auto param = std::make_shared<Parameter>(name, value, true);
    module_state.named_parameters.insert({name, param});
  }

  const auto* frozen_params = fbs_module_state.frozen_params();
  ORT_RETURN_IF_NOT(frozen_params, "Expected: Valid non trainable tensors flatbuffer.",
                    " Actual: Encountered a nullptr. Checkpoint file is invalid");
  flatbuffers::uoffset_t non_trainable_params_size = frozen_params->size();
  InlinedHashMap<std::string, OrtValue> non_trainable_params;
  non_trainable_params.reserve(non_trainable_params_size);
  ORT_RETURN_IF_ERROR(OrtValuesFromFlatbufferTensors(*frozen_params, non_trainable_params));

  for (auto& [name, value] : non_trainable_params) {
    auto param = std::make_shared<Parameter>(name, value, false);
    module_state.named_parameters.insert({name, param});
  }

  return Status::OK();
}

/**
 * @brief Load from a flatbuffer checkpoint optimizer state to an optimizer state.
 *
 * @param optimizer_groups Flatbuffer optimizer groups.
 * @param optimizer_state Optimizer state to be populated.
 * @return Status of the operation.
 */
Status ToOptimizerState(
    const flatbuffers::Vector<flatbuffers::Offset<onnxruntime::fbs::OptimizerGroup>>& optimizer_groups,
    OptimizerCheckpointState& optimizer_state) {
  for (const auto* optimizer_group : optimizer_groups) {
    ORT_RETURN_IF_NOT(optimizer_group, "Expected: Valid optimizer groups flatbuffer.",
                      " Actual: Encountered a nullptr. Checkpoint file is invalid");

    std::string group_name = optimizer_group->group_name()->str();
    const int64_t step = optimizer_group->step();
    const float initial_learning_rate = optimizer_group->initial_learning_rate();

    const auto* parameter_optimizer_states = optimizer_group->optimizer_states();
    ORT_RETURN_IF_NOT(parameter_optimizer_states, "Expected: Valid parameter optimizer states flatbuffer.",
                      " Actual: Encountered a nullptr. Checkpoint file is invalid");

    [[maybe_unused]] auto [optimizer_state_it, inserted] =
        optimizer_state.group_named_optimizer_states.emplace(
            std::move(group_name), std::make_shared<GroupOptimizerState>());

    ORT_RETURN_IF_NOT(inserted, "Encountered a duplicate optimizer group name: ", group_name,
                      ". Checkpoint file is invalid.");

    optimizer_state_it->second->step = step;
    optimizer_state_it->second->initial_lr = initial_learning_rate;
    for (const auto* parameter_optimizer_state : *parameter_optimizer_states) {
      const std::string param_name = parameter_optimizer_state->param_name()->str();
      const auto* momentums = parameter_optimizer_state->momentums();
      ORT_RETURN_IF_NOT(momentums, "Expected: Valid optimizer momentum tensors flatbuffer.",
                        " Actual: Encountered a nullptr. Checkpoint file is invalid");
      ORT_RETURN_IF_ERROR(OrtValuesFromFlatbufferTensors(
          *momentums, optimizer_state_it->second->param_named_optimizer_states[param_name]));
    }
  }

  return Status::OK();
}

/**
 * @brief Load from a flatbuffer checkpoint property bag to a property bag.
 *
 * @param fbs_property_bag Flatbuffer property bag.
 * @param property_bag Property bag to be populated.
 * @return Status of the operation.
 */
Status ToPropertyBag(const onnxruntime::fbs::PropertyBag& fbs_property_bag,
                     PropertyBag& property_bag) {
  const auto* ints = fbs_property_bag.ints();
  if (nullptr != ints) {
    for (const auto* int_property : *ints) {
      ORT_RETURN_IF_NOT(int_property, "Checkpoint is invalid. Expected: Valid flatbuffer int property. ",
                        "Actual: nullptr.");
      ORT_RETURN_IF_NOT(int_property->name(), "Checkpoint is invalid. Expected: Valid flatbuffer int property name. ",
                        "Actual: nullptr.");
      const std::string name = int_property->name()->str();
      const auto value = int_property->value();
      property_bag.AddProperty(name, value);
    }
  }

  const auto* floats = fbs_property_bag.floats();
  if (nullptr != floats) {
    for (const auto* float_property : *floats) {
      ORT_RETURN_IF_NOT(float_property, "Checkpoint is invalid. Expected: Valid flatbuffer float property. ",
                        "Actual: nullptr.");
      ORT_RETURN_IF_NOT(float_property->name(), "Checkpoint is invalid. Expected: Valid flatbuffer float property name. ",
                        "Actual: nullptr.");
      const std::string name = float_property->name()->str();
      const auto value = float_property->value();
      property_bag.AddProperty(name, value);
    }
  }

  const auto* strings = fbs_property_bag.strings();
  if (nullptr != strings) {
    for (const auto* string_property : *strings) {
      ORT_RETURN_IF_NOT(string_property, "Checkpoint is invalid. Expected: Valid flatbuffer string property. ",
                        "Actual: nullptr.");
      ORT_RETURN_IF_NOT(string_property->name(), "Checkpoint is invalid. Expected: Valid flatbuffer string property name. ",
                        "Actual: nullptr.");
      ORT_RETURN_IF_NOT(string_property->value(), "Checkpoint is invalid. Expected: Valid flatbuffer string property value. ",
                        "Actual: nullptr.");
      const std::string name = string_property->name()->str();
      const std::string value = string_property->value()->str();
      property_bag.AddProperty(name, value);
    }
  }

  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD)
/**
 * @brief Load checkpoint from a checkpoint file to initializers in a model proto.
 *
 * @param checkpoint_path Path to the checkpoint file.
 * @param model_proto Model proto to be populated.
 * @return Status of the operation.
 */
Status ToModelProto(gsl::span<const uint8_t> checkpoint_bytes,
                    ONNX_NAMESPACE::ModelProto& model_proto) {
  flatbuffers::Verifier verifier(checkpoint_bytes.data(), checkpoint_bytes.size());
  ORT_RETURN_IF_NOT(fbs::VerifyCheckpointBuffer(verifier), "Checkpoint verification failed.");

  const auto* fbs_checkpoint = fbs::GetCheckpoint(checkpoint_bytes.data());
  ORT_RETURN_IF_NOT(fbs_checkpoint, "Checkpoint is invalid. Expected: Valid checkpoint flatbuffer. Actual: nullptr.");

  const auto checkpoint_version = fbs_checkpoint->version();
  ORT_RETURN_IF_NOT(IsCheckpointVersionSupported(checkpoint_version),
                    "Checkpoint is invalid. Checkpoint version ", checkpoint_version, " is not supported.");

  const auto* module_state = fbs_checkpoint->module_state();
  if (nullptr == module_state) {
    return Status::OK();
  }

  const auto* requires_grad_params = module_state->requires_grad_params();
  ORT_RETURN_IF_NOT(requires_grad_params,
                    "Checkpoint is invalid. Expected: Valid trainable params flatbuffer. Actual: nullptr.");

  const auto* frozen_params = module_state->frozen_params();
  ORT_RETURN_IF_NOT(frozen_params,
                    "Checkpoint is invalid. Expected: Valid non-trainable params flatbuffer. Actual: nullptr.");

  InlinedHashMap<std::string, ONNX_NAMESPACE::TensorProto> param_tensor_protos;
  param_tensor_protos.reserve(
      static_cast<size_t>(requires_grad_params->size()) + static_cast<size_t>(frozen_params->size()));

  const auto flatbuffer_tensors_to_tensor_protos = [&param_tensor_protos](const auto& flatbuffer_tensors) {
    OrtFormatLoadOptions load_options{false, false};
    for (const auto* fbs_tensor : flatbuffer_tensors) {
      ORT_RETURN_IF_NOT(fbs_tensor, "Checkpoint is invalid. Expected: Valid flatbuffer tensor. Actual: nullptr.");
      ONNX_NAMESPACE::TensorProto tensor_proto;
      ORT_RETURN_IF_ERROR(fbs::utils::LoadInitializerOrtFormat(*fbs_tensor, tensor_proto, load_options));
      param_tensor_protos.insert({fbs_tensor->name()->str(), tensor_proto});
    }

    return Status::OK();
  };

  ORT_RETURN_IF_ERROR(flatbuffer_tensors_to_tensor_protos(*requires_grad_params));
  ORT_RETURN_IF_ERROR(flatbuffer_tensors_to_tensor_protos(*frozen_params));

  // Copy loaded tensor protos to the initializers in the ModelProto
  for (auto& init : *(model_proto.mutable_graph()->mutable_initializer())) {
    ORT_RETURN_IF_NOT(init.has_name(), "ModelProto is invalid. Expected: All initializers must have names.");
    const auto it = param_tensor_protos.find(init.name());
    if (it == param_tensor_protos.cend()) {
      continue;
    }
    init.CopyFrom(it->second);
  }

  return Status::OK();
}
#endif

/**
 * @brief Load checkpoint from a checkpoint file to a checkpoint state.
 *
 * @param checkpoint_path Path to the checkpoint file.
 * @param state Checkpoint state to be populated.
 * @return Status of the operation.
 */
Status ToCheckpointState(gsl::span<const uint8_t> checkpoint_bytes, CheckpointState& state) {
  flatbuffers::Verifier verifier(checkpoint_bytes.data(), checkpoint_bytes.size());
  ORT_RETURN_IF_NOT(fbs::VerifyCheckpointBuffer(verifier), "Checkpoint verification failed.");

  const auto* fbs_checkpoint = fbs::GetCheckpoint(checkpoint_bytes.data());
  ORT_RETURN_IF_NOT(fbs_checkpoint, "Checkpoint is invalid. Expected: Valid checkpoint flatbuffer. Actual: nullptr.");

  const auto checkpoint_version = fbs_checkpoint->version();
  ORT_RETURN_IF_NOT(IsCheckpointVersionSupported(checkpoint_version),
                    "Checkpoint is invalid. Checkpoint version ", checkpoint_version, " is not supported.");

  const auto* fbs_module_state = fbs_checkpoint->module_state();
  if (nullptr != fbs_module_state) {
    ORT_RETURN_IF_ERROR(ToModuleState(*fbs_module_state, state.module_checkpoint_state));
  }

  const auto* fbs_optimizer_groups = fbs_checkpoint->optimizer_groups();
  if (nullptr != fbs_optimizer_groups) {
    ORT_RETURN_IF_ERROR(ToOptimizerState(*fbs_optimizer_groups, state.optimizer_checkpoint_state));
  }

  const auto* fbs_property_bag = fbs_checkpoint->property_bag();
  if (nullptr != fbs_property_bag) {
    ORT_RETURN_IF_ERROR(ToPropertyBag(*fbs_property_bag, state.property_bag));
  }

  return Status::OK();
}

}  // namespace load

}  // namespace

#if !defined(ORT_MINIMAL_BUILD)
Status SaveCheckpoint(gsl::span<const ONNX_NAMESPACE::TensorProto> trainable_tensor_protos,
                      gsl::span<const ONNX_NAMESPACE::TensorProto> non_trainable_tensor_protos,
                      const PathString& checkpoint_path) {
  ORT_RETURN_IF_NOT(FLATBUFFERS_LITTLEENDIAN, "ORT training checkpoint format only supports little-endian machines");
  return save::FromTensorProtos(trainable_tensor_protos, non_trainable_tensor_protos, checkpoint_path);
}
#endif

Status SaveCheckpoint(const CheckpointState& states, const PathString& checkpoint_path,
                      const bool include_optimizer_state) {
  ORT_RETURN_IF_NOT(FLATBUFFERS_LITTLEENDIAN, "ORT training checkpoint format only supports little-endian machines");
  return save::FromCheckpointState(states, checkpoint_path, include_optimizer_state);
}

Status LoadCheckpoint(const PathString& checkpoint_path, CheckpointState& checkpoint_states) {
  ORT_RETURN_IF_NOT(FLATBUFFERS_LITTLEENDIAN, "ORT training checkpoint format only supports little-endian machines");

  InlinedVector<uint8_t> checkpoint_bytes;
  ORT_RETURN_IF_ERROR(load::FromFile(checkpoint_path, checkpoint_bytes));
  return load::ToCheckpointState(checkpoint_bytes, checkpoint_states);
}

Status LoadCheckpointFromBuffer(gsl::span<const uint8_t> checkpoint_bytes, CheckpointState& checkpoint_state) {
  ORT_RETURN_IF_NOT(FLATBUFFERS_LITTLEENDIAN, "ORT training checkpoint format only supports little-endian machines");

  return load::ToCheckpointState(checkpoint_bytes, checkpoint_state);
}

#if !defined(ORT_MINIMAL_BUILD)
Status LoadCheckpointToModel(const PathString& checkpoint_path,
                             ONNX_NAMESPACE::ModelProto& model_proto) {
  ORT_RETURN_IF_NOT(FLATBUFFERS_LITTLEENDIAN, "ORT training checkpoint format only supports little-endian machines");

  InlinedVector<uint8_t> checkpoint_bytes;
  ORT_RETURN_IF_ERROR(load::FromFile(checkpoint_path, checkpoint_bytes));
  return load::ToModelProto(checkpoint_bytes, model_proto);
}
#endif

}  // namespace onnxruntime::training::api
