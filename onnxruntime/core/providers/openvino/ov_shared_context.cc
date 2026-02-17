// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "ov_shared_context.h"
#include "ov_interface.h"

#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "openvino/core/type/element_type.hpp"

namespace onnxruntime {
namespace openvino_ep {

SharedContext::SharedContext(const std::filesystem::path& bin_path)
    : bin_path_(bin_path),
      bin_manager_(bin_path_),
      weight_file_manager_(WeightFileManager::Get()) {
}

static bool InRange(size_t offset, size_t size, size_t total_size) {
  return (offset < total_size) && (size <= total_size) && (offset <= total_size - size);
}

// Weights file handling
SharedContext::WeightsFile::WeightsFile(const std::filesystem::path& filename) : file_(filename, std::ios::in | std::ios::binary), file_path_(filename) {
  try {
    file_.exceptions(std::ifstream::failbit | std::ifstream::badbit);
    weights_size_ = std::filesystem::file_size(filename);
  } catch (std::exception& e) {
    ORT_THROW("Error: Failed to open weight file at ", filename.string(), " ", e.what());
  }
}

void SharedContext::WeightsFile::LoadWeights(size_t file_offset, void* data, size_t size) {
  ORT_ENFORCE(InRange(file_offset, size, weights_size_), "Error: File offset is out of bounds.");
  file_.seekg(file_offset);
  file_.read(static_cast<char*>(data), size);
}

const void* SharedContext::WeightsFile::TryGetOrCreateDeviceMapping(std::optional<ov::RemoteContext>& remote_context) {
  std::string dev_name{};
  if (remote_context) {
    dev_name = remote_context->get_device_name();
  }

  auto [it, inserted] = imported_device_tensors_.emplace(dev_name, MappingContainer{});
  if (inserted) {
    if (dev_name == "NPU") {
      // try to import the memory mapped file to remote tensor
#if (OPENVINO_VERSION_MAJOR > 2025 || (OPENVINO_VERSION_MAJOR == 2025 && OPENVINO_VERSION_MINOR >= 3))
      ORT_ENFORCE(remote_context, "Error: Remote context is required for NPU device.");
      auto npu_context = remote_context->as<ov::intel_npu::level_zero::ZeroContext>();
      auto&& l0_tensor = npu_context.create_tensor(ov::element::Type_t::u8, {weights_size_}, ov::intel_npu::FileDescriptor(file_path_));
      it->second = MappingContainer{.ptr_ = l0_tensor.get(), .tensor_ = l0_tensor};
#endif
    } else if (dev_name.empty()) {
      // CPU/virtual device case, create a CPU tensor memory mapped from file
      const auto&& mmaped_tensor = ov::read_tensor_data(file_path_);

      // Suppress warning for tensor.data() returning const in 2026.0. Should be removable after 2026.0 is min supported ov version.
      OPENVINO_SUPPRESS_DEPRECATED_START
      it->second = MappingContainer{.ptr_ = mmaped_tensor.data(), .tensor_ = mmaped_tensor};
      OPENVINO_SUPPRESS_DEPRECATED_END
    }
  }

  return it->second.ptr_;
}

void SharedContext::LoadTensorFromFile(
    Metadata::Value& value,
    const std::filesystem::path& model_dir,
    std::optional<ov::RemoteContext>& remote_context,
    const ov::element::Type& element_type,
    const ov::Shape& dimensions) {
  const auto weights_location = model_dir / value.serialized.location;
  auto& weights_file = weight_files_[weights_location];
  if (!weights_file) {
    weights_file = weight_file_manager_->GetOrCreateWeightsFile(weights_location);
  }

  ov::Tensor tensor;
  const uint8_t* mmaped_weights = static_cast<const uint8_t*>(weights_file->TryGetOrCreateDeviceMapping(remote_context));
  if (mmaped_weights) {
    // We have memory mapped weights. Create a Tensor view into it for this value.
    ORT_ENFORCE(InRange(value.serialized.data_offset, value.serialized.size, weights_file->Size()), "File offset + size outside of external initializer file");
    const void* mmapped_offset = static_cast<const void*>(mmaped_weights + value.serialized.data_offset);
#if OPENVINO_VERSION_AT_LEAST(2026, 0)
    // In OV 2026.0 we can pass read-only tensors as inputs.
    tensor = ov::Tensor(element_type, dimensions, mmapped_offset);
#else
    tensor = ov::Tensor(element_type, dimensions, const_cast<void*>(mmapped_offset));
#endif
  } else {
    ORT_ENFORCE(remote_context, "Unexpected: Don't have remote context and memory mapped weights is null!");
    // Can't mmap the file to device tensor, create a host tensor and copy the data
    tensor = remote_context->create_host_tensor(element_type, dimensions);
    ORT_ENFORCE(tensor.get_byte_size() == value.serialized.size, "Remote tensor size mismatch");
    weights_file->LoadWeights(value.serialized.data_offset, tensor.data(), value.serialized.size);
  }

  ORT_ENFORCE(tensor.get_byte_size() == value.serialized.size, "Tensor size mismatch");
  value.tensor = std::make_shared<const ov::Tensor>(std::move(tensor));
}

void SharedContext::SetSharedWeightsOnInferRequest(ov::InferRequest& ir, const std::filesystem::path& model_dir) {
  auto&& compiled_model = ir.get_compiled_model();
  std::optional<ov::RemoteContext> opt_remote_ctx;
  try {
    opt_remote_ctx = compiled_model.get_context();
  } catch (ov::Exception&) {
    // CPU may not have a remote context.
  }

  std::unique_lock<std::shared_mutex> ul(mutex_);
  for (const auto& input : compiled_model.inputs()) {
    const std::string tensor_name = *input.get_names().begin();

    auto it = metadata_.find(tensor_name);
    if (it == metadata_.end()) continue;  // No shared weight for this tensor
    auto& value = it->second;

    if (!value.tensor) {
      LoadTensorFromFile(value, model_dir, opt_remote_ctx, input.get_element_type(), input.get_shape());
    }
    ir.set_tensor(tensor_name, *value.tensor);
  }
}

void SharedContext::Serialize(std::ostream& stream) {
  bin_manager_.Serialize(stream, shared_from_this());
}

void SharedContext::Deserialize(std::istream& stream) {
  bin_manager_.Deserialize(stream, shared_from_this());
}

void SharedContext::Serialize() {
  bin_manager_.Serialize(shared_from_this());
}

void SharedContext::Deserialize() {
  bin_manager_.Deserialize(shared_from_this());
}

}  // namespace openvino_ep
}  // namespace onnxruntime
