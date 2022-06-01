// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/snpe/snpe_lib.h"
#include <iostream>
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/IUserBufferFactory.hpp"
#include "SNPE/SNPEBuilder.hpp"

namespace onnxruntime {
namespace contrib {
namespace snpe {

size_t CalcSizeFromDims(const zdl::DlSystem::Dimension* dims, size_t rank, size_t elementSize) {
  if (rank == 0) return 0;
  size_t size = elementSize;
  while (rank--) {
    size *= *dims;
    dims++;
  }
  return size;
}

bool SnpeLib::SetupUserBufferAttribute(const std::string& name) {
  auto buffer_attributes = snpe_->getInputOutputBufferAttributes(name.c_str());
  if (!buffer_attributes) {
    LOGS_DEFAULT(ERROR) << "Error obtaining attributes for input/output tensor" << name;
    return false;
  }
  zdl::DlSystem::UserBufferEncoding* user_buffer_encoding = nullptr;
  size_t buffer_element_size = 0;
  if (BufferType::TF8 == buffer_type_ || BufferType::TF16 == buffer_type_) {
    int bit_width = buffer_type_ == BufferType::TF16 ? 16 : 8;
    buffer_element_size = bit_width / 8;
    auto encoding = (*buffer_attributes)->getEncoding();
    if (!encoding) {
      LOGS_DEFAULT(ERROR) << "Failed to get buffer encoding for: " << name;
      return false;
    }
    user_buffer_encoding_.push_back(std::make_unique<zdl::DlSystem::UserBufferEncodingTfN>(*encoding));
    user_buffer_encoding = user_buffer_encoding_.back().get();
  } else if (BufferType::FLOAT == buffer_type_) {
    buffer_element_size = sizeof(float);
    user_buffer_encoding_.push_back(std::make_unique<zdl::DlSystem::UserBufferEncodingFloat>());
    user_buffer_encoding = user_buffer_encoding_.back().get();
  } else if (BufferType::UINT8 == buffer_type_) {
    buffer_element_size = 1;
    user_buffer_encoding_.push_back(std::make_unique<zdl::DlSystem::UserBufferEncodingUnsigned8Bit>());
    user_buffer_encoding = user_buffer_encoding_.back().get();
  }

  zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> tensor_shape = (*buffer_attributes)->getDims();
  auto shape_rank = (*tensor_shape).rank();
  std::vector<size_t> strides(shape_rank);
  strides[strides.size() - 1] = buffer_element_size;
  size_t stride = strides[strides.size() - 1];
  for (size_t j = shape_rank - 1; j > 0; j--) {
    stride *= (*tensor_shape)[j];
    strides[j - 1] = stride;
  }
  size_t bufSize = CalcSizeFromDims((*tensor_shape).getDimensions(), shape_rank, buffer_element_size);

  user_buffer_attr_table_.emplace(name, UserBufferAttribute(bufSize, strides, user_buffer_encoding));

  return true;
}

bool SnpeLib::SetupUserBufferAttributes(const std::vector<std::string>& tensor_names) {
  for (size_t i = 0; i < tensor_names.size(); ++i) {
    bool rt = SetupUserBufferAttribute(tensor_names.at(i));
    if (!rt) {
      LOGS_DEFAULT(ERROR) << "Failed to set user buffer attribute for: " << tensor_names.at(i);
      return false;
    }
  }

  return true;
}

bool SnpeLib::CheckInputsSize(const std::vector<std::string>& input_tensor_names,
                              const std::vector<int64_t>& input_sizes) {
  size_t elementSize = 1;
  for (size_t i = 0; i < input_tensor_names.size(); ++i) {
    auto input_shape = snpe_->getInputDimensions(input_tensor_names.at(i).c_str());
    if (!input_shape) {
      LOGS_DEFAULT(ERROR) << "Snpe cannot get input shape for input name: " << input_tensor_names.at(i);
      return false;
    }
    int64_t input_size = CalcSizeFromDims((*input_shape).getDimensions(), (*input_shape).rank(), elementSize);
    if (input_sizes.at(i) != input_size) {
      LOGS_DEFAULT(ERROR) << "Input size mismatch for : " << input_tensor_names.at(i)
                          << " size on dlc: " << input_size
                          << " size on onnx: " << input_sizes.at(i);
      return false;
    }
  }
  return true;
}

bool SnpeLib::SetupInputTensors(const std::vector<std::string>& input_tensor_names) {
  input_tensor_map_.clear();
  input_tensors_.clear();
  input_tensors_.resize(input_tensor_names.size());
  for (size_t i = 0; i < input_tensor_names.size(); ++i) {
    auto input_shape = snpe_->getInputDimensions(input_tensor_names.at(i).c_str());
    if (!input_shape) {
      LOGS_DEFAULT(ERROR) << "Snpe cannot get input shape for input name: " << input_tensor_names.at(i);
      input_tensor_map_.clear();
      input_tensors_.clear();
      return false;
    }
    input_tensors_[i] = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(*input_shape);
    zdl::DlSystem::ITensor* input_tensor = input_tensors_[i].get();
    if (!input_tensor) {
      LOGS_DEFAULT(ERROR) << "Snpe cannot create ITensor!";
      input_tensor_map_.clear();
      input_tensors_.clear();
      return false;
    }
    input_tensor_map_.add(input_tensor_names.at(i).c_str(), input_tensor);
  }
  return true;
}

bool SnpeLib::InitializeSnpe(zdl::DlContainer::IDlContainer* container,
                             const std::vector<std::string>& output_tensor_names,
                             const std::vector<std::string>& input_tensor_names,
                             const std::vector<int64_t>& input_sizes,
                             const SnpeRuntimeOptions& snpe_settings) {
  if (!snpe_settings.GetRuntimeTarget().IsAvailable()) {
    LOGS_DEFAULT(INFO) << "Provided runtime target is not available" << snpe_settings.GetRuntimeTarget().ToString();
    return false;
  }

  if (0 == output_tensor_names.size() || 0 == input_tensor_names.size()) {
    LOGS_DEFAULT(ERROR) << "input names or output names are empty!";
    return false;
  }

  zdl::SNPE::SNPEBuilder snpe_builder(container);
  zdl::DlSystem::StringList dl_output_tensor_names = zdl::DlSystem::StringList();
  for (auto it = output_tensor_names.begin(); it != output_tensor_names.end(); ++it) {
    dl_output_tensor_names.append((*it).c_str());
  }

  buffer_type_ = snpe_settings.GetBufferType();
  bool use_user_buffer = buffer_type_ == BufferType::ITENSOR ? false : true;
  snpe_builder.setOutputTensors(dl_output_tensor_names)
              .setRuntimeProcessor(snpe_settings.GetRuntimeTarget().Get())
              .setExecutionPriorityHint(snpe_settings.GetExecutionPriority())
              .setUseUserSuppliedBuffers(use_user_buffer);
#ifdef __ANDROID__
  // use sustained performance mode on android variants.
  LOGS_DEFAULT(INFO) << "setPerformanceProfile to SUSTAINED_HIGH_PERFORMANCE for Android environment!";
  snpe_builder.setPerformanceProfile(zdl::DlSystem::PerformanceProfile_t::SUSTAINED_HIGH_PERFORMANCE);
#endif

#ifdef _WIN32
  if (snpe_settings.GetRuntimeTarget().Get() == zdl::DlSystem::Runtime_t::DSP) {
    zdl::DlSystem::PlatformConfig platformConfig;
    if (!platformConfig.setPlatformOptionValue("unsignedPD", "ON")) {
      LOGS_DEFAULT(ERROR) << "unsignedPD is not available";
      return false;
    }
    LOGS_DEFAULT(INFO) << "set platform config to unsignedPD";
    snpe_builder.setPlatformConfig(platformConfig);
  }
#endif

  snpe_ = snpe_builder.build();
  if (nullptr == snpe_) {
    LOGS_DEFAULT(ERROR) << "Failed to create snpe instance!";
    return false;
  }

  // Make sure shape of snpe inputs are same with onnx model inputs
  bool status = CheckInputsSize(input_tensor_names, input_sizes);
  if (!status) {
    return false;
  }

  if (BufferType::UNKNOWN == buffer_type_) {
    LOGS_DEFAULT(ERROR) << "Buffer type unknown!";
    return false;
  } else if (BufferType::ITENSOR == buffer_type_) {
    return SetupInputTensors(input_tensor_names);
  } else {
    bool rt = SetupUserBufferAttributes(input_tensor_names);
    rt |= SetupUserBufferAttributes(output_tensor_names);
    if (!rt) {
      return false;
    }
  }

  return true;
}

bool SnpeLib::Initialize(const char* dlcPath,
                         const std::vector<std::string>& output_layer_names,
                         const std::vector<std::string>& input_layer_names,
                         const std::vector<int64_t>& input_sizes,
                         const SnpeRuntimeOptions& snpe_settings) {
  auto container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(dlcPath));
  if (!container) {
    LOGS_DEFAULT(ERROR) << "failed open " << dlcPath << " container file";
    return false;
  }

  bool rt = InitializeSnpe(container.get(), output_layer_names, input_layer_names, input_sizes, snpe_settings);
  if (!rt) {
    LOGS_DEFAULT(ERROR) << "failed to build snpe " << GetSnpeErrorString();
    return false;
  }

  return true;
}

bool SnpeLib::Initialize(const unsigned char* dlcData, size_t size,
                         const std::vector<std::string>& output_layer_names,
                         const std::vector<std::string>& input_layer_names,
                         const std::vector<int64_t>& input_sizes,
                         const SnpeRuntimeOptions& snpe_settings) {
  auto container = zdl::DlContainer::IDlContainer::open(dlcData, size);
  if (container == nullptr) {
    LOGS_DEFAULT(ERROR) << "failed open container buffer";
    return false;
  }

  bool rt = InitializeSnpe(container.get(), output_layer_names, input_layer_names, input_sizes, snpe_settings);
  if (!rt) {
    LOGS_DEFAULT(ERROR) << "failed to build snpe " << GetSnpeErrorString();
    return false;
  }

  return true;
}

bool SnpeLib::SnpeProcessMultipleOutput(const unsigned char* input,
                                        size_t input_size,
                                        size_t output_number,
                                        unsigned char* outputs[],
                                        size_t output_sizes[],
                                        const std::unordered_map<std::string, size_t>& output_names_index) {
  try {
    zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> input_shape = snpe_->getInputDimensions();
    if (!input_shape) {
      LOGS_DEFAULT(ERROR) << "Snpe cannot get input shape";
      return false;
    }

    if (input_tensors_.size() < 1) {
      LOGS_DEFAULT(ERROR) << "Failed to create Snpe ITensor at initialization time.";
      return false;
    }
    zdl::DlSystem::ITensor* input_tensor = input_tensors_[0].get();
    // ensure size of the input buffer matches input shape buffer size
    size_t input_data_size = input_tensor->getSize() * sizeof(float);
    if (input_data_size != input_size) {
      LOGS_DEFAULT(ERROR) << "Snpe input size incorrect: expected bytes "
                          << input_data_size << " given bytes " << input_size;
      return false;
    }
    memcpy(input_tensor->begin().dataPointer(), input, input_size);

    zdl::DlSystem::TensorMap output_tensor_map;
    bool result = snpe_->execute(input_tensor, output_tensor_map);
    if (!result) {
      LOGS_DEFAULT(ERROR) << "Snpe Error while executing the network: " << GetSnpeErrorString();
      return false;
    }
    if (output_tensor_map.size() == 0) {
      return false;
    }

    zdl::DlSystem::StringList tensor_names = output_tensor_map.getTensorNames();

    for (size_t i = 0; i < output_number; i++) {
      zdl::DlSystem::ITensor* tensor = output_tensor_map.getTensor(tensor_names.at(i));

      auto pos = output_names_index.find(tensor_names.at(i));
      size_t output_name_index = 0;
      if (pos == output_names_index.end()) {
        LOGS_DEFAULT(ERROR) << "Something wrong with output name: " << tensor_names.at(i);
        return false;
      }
      output_name_index = pos->second;

      // ensure size of the output buffer matches output shape buffer size
      size_t output_data_size = tensor->getSize() * sizeof(float);
      if (output_data_size > output_sizes[output_name_index]) {
        LOGS_DEFAULT(ERROR) << "Snpe output size incorrect: output_layer: " << tensor_names.at(i)
                            << " expected bytes " << output_data_size
                            << " given bytes " << output_sizes[output_name_index];
        return false;
      }
      memcpy(outputs[output_name_index], tensor->cbegin().dataPointer(), output_data_size);
    }

    return true;
  } catch (...) {
    LOGS_DEFAULT(ERROR) << "Snpe threw exception";
    return false;
  }
}

bool SnpeLib::SnpeProcess(const unsigned char* input, size_t input_size, unsigned char* output, size_t output_size,
                          const std::unordered_map<std::string, size_t>& output_names_index) {
  // Use SnpeProcessMultipleOutput with 1 output layer
  const int output_layer = 1;
  unsigned char* outputs_array[output_layer];
  size_t output_sizes_array[output_layer];
  outputs_array[0] = output;
  output_sizes_array[0] = output_size;
  return SnpeProcessMultipleOutput(input, input_size, output_layer, outputs_array,
                                   output_sizes_array, output_names_index);
}

bool SnpeLib::SnpeProcessMultiInputsMultiOutputs(const unsigned char** inputs,
                                                 const size_t* input_sizes,
                                                 size_t input_number,
                                                 unsigned char** outputs,
                                                 const size_t* output_sizes,
                                                 size_t output_number,
                                                 const std::unordered_map<std::string, size_t>& output_names_index) {
  try {
    if (input_number != input_tensors_.size()) {
      LOGS_DEFAULT(ERROR) << "Snpe number of inputs doesn't match: expected "
                          << input_number << " given " << input_tensors_.size();
      return false;
    }
    for (size_t i = 0; i < input_number; ++i) {
      zdl::DlSystem::ITensor* input_tensor = input_tensors_[i].get();
      // ensure size of the input buffer matches input shape buffer size
      size_t input_data_size = input_tensor->getSize() * sizeof(float);
      if (input_data_size != input_sizes[i]) {
        LOGS_DEFAULT(ERROR) << "Snpe input size incorrect: expected bytes " << input_data_size
                            << ", given bytes "  << input_sizes[i];
        return false;
      }
      memcpy(input_tensor->begin().dataPointer(), inputs[i], input_sizes[i]);
    }
    zdl::DlSystem::TensorMap output_tensor_map;
    bool result = snpe_->execute(input_tensor_map_, output_tensor_map);
    if (!result) {
      LOGS_DEFAULT(ERROR) << "Snpe Error while executing the network: " << GetSnpeErrorString();
      return false;
    }
    if (output_tensor_map.size() == 0) {
      return false;
    }

    zdl::DlSystem::StringList tensor_names = output_tensor_map.getTensorNames();

    for (size_t i = 0; i < output_number; i++) {
      zdl::DlSystem::ITensor* tensor = output_tensor_map.getTensor(tensor_names.at(i));

      auto pos = output_names_index.find(tensor_names.at(i));
      size_t output_name_index = 0;
      if (pos == output_names_index.end()) {
        LOGS_DEFAULT(ERROR) << "Something wrong with output name: " << tensor_names.at(i);
        return false;
      }
      output_name_index = pos->second;

      // ensure size of the output buffer matches output shape buffer size
      size_t output_data_size = tensor->getSize() * sizeof(float);
      if (output_data_size > output_sizes[output_name_index]) {
        LOGS_DEFAULT(ERROR) << "Snpe output size incorrect: output_layer" << tensor_names.at(i)
                            << " expected bytes " << output_data_size
                            << " given bytes " << output_sizes[output_name_index];
        return false;
      }
      memcpy(outputs[output_name_index], tensor->cbegin().dataPointer(), output_data_size);
    }

    return true;
  } catch (...) {
    LOGS_DEFAULT(ERROR) << "Snpe threw exception";
    return false;
  }
}

bool SnpeLib::SnpeProcessWithUserBuffer(const std::vector<std::string>& input_names,
                                        const unsigned char** inputs,
                                        size_t input_number,
                                        unsigned char** outputs,
                                        const std::unordered_map<std::string, size_t>& output_names_index) {
  zdl::DlSystem::IUserBufferFactory& user_buffer_factory = zdl::SNPE::SNPEFactory::getUserBufferFactory();
  snpe_user_input_buffers_.clear();
  snpe_user_input_buffers_.reserve(input_number);
  zdl::DlSystem::UserBufferMap input_buffer_map;
  for (size_t i = 0; i < input_number; ++i) {
    std::string input_name(input_names.at(i));
    auto buffer_attr_pos = user_buffer_attr_table_.at(input_name);
    snpe_user_input_buffers_.push_back(user_buffer_factory.createUserBuffer(const_cast<unsigned char*>(inputs[i]),
                                                                            buffer_attr_pos.buffer_size,
                                                                            buffer_attr_pos.strides,
                                                                            buffer_attr_pos.user_buffer_encoding));
    input_buffer_map.add(input_name.c_str(), snpe_user_input_buffers_.back().get());
  }

  zdl::DlSystem::UserBufferMap output_buffer_map;
  snpe_user_output_buffers_.clear();
  snpe_user_output_buffers_.reserve(output_names_index.size());
  for (auto it = output_names_index.begin(); it != output_names_index.end(); ++it) {
    auto buffer_attr_pos = user_buffer_attr_table_.at(it->first);
    snpe_user_output_buffers_.push_back(user_buffer_factory.createUserBuffer(outputs[it->second],
                                                                             buffer_attr_pos.buffer_size,
                                                                             buffer_attr_pos.strides,
                                                                             buffer_attr_pos.user_buffer_encoding));
    output_buffer_map.add(it->first.c_str(), snpe_user_output_buffers_.back().get());
  }

  // Execute the input buffer map on the model with SNPE
  bool result = snpe_->execute(input_buffer_map, output_buffer_map);
  if (!result) {
    LOGS_DEFAULT(ERROR) << "Snpe Error while executing the network: " << GetSnpeErrorString();
    return false;
  }

  return true;
}

int SnpeLib::RegisterUDOs(const std::string udo_dir, const std::vector<std::string>& udo_file_names) {
  int udos_registered = 0;

  for (const auto& udo_file : udo_file_names) {
    std::string full_path = udo_dir + "/" + udo_file;
    bool result = zdl::SNPE::SNPEFactory::addOpPackage(full_path);
    if (result) {
      ++udos_registered;
    } else {
      LOGS_DEFAULT(ERROR) << "Failed to register SNPE UDO library: " << full_path << " :" << GetSnpeErrorString();
    }
  }
  return udos_registered;
}

std::unique_ptr<SnpeLib> SnpeLibFactory(const unsigned char* dlc_data,
                                        size_t size,
                                        const std::unordered_map<std::string, std::string>& options,
                                        const std::vector<std::string>& output_layer_names,
                                        const std::vector<std::string>& input_layer_names,
                                        const std::vector<int64_t>& input_sizes,
                                        BufferType& buffer_type) {
  std::unique_ptr<SnpeLib> object(new SnpeLib());

  if (!object) {
    ORT_THROW("failed to make snpe library");
  }

  SnpeRuntimeOptions runtime_options(options);
  if (!object->Initialize(dlc_data, size, output_layer_names, input_layer_names, input_sizes, runtime_options)) {
    ORT_THROW("failed to initialize dlc from buffer");
  }

  buffer_type = runtime_options.GetBufferType();

  return object;
}

}  // namespace snpe
}  // namespace contrib
}  // namespace onnxruntime
