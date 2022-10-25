// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/snpe/snpe_lib.h"
#include <iostream>
#include <unordered_map>
#include <memory>
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "DlContainer/IDlContainer.hpp"
#include "DlSystem/ITensorFactory.hpp"
#include "DlSystem/IUserBufferFactory.hpp"
#include "SNPE/SNPEBuilder.hpp"

namespace onnxruntime {
namespace contrib {
namespace snpe {

size_t CalcSizeFromDims(const zdl::DlSystem::Dimension* dims, size_t rank, size_t elementSize) {
  if (rank == 0) return 0;
  SafeInt<size_t> size = elementSize;
  while (rank--) {
    size = size * (*dims);
    dims++;
  }
  return size;
}

Status SnpeLib::SetupUserBufferAttribute(const std::string& name) {
  auto buffer_attributes = snpe_->getInputOutputBufferAttributes(name.c_str());
  ORT_ENFORCE(buffer_attributes, "Error obtaining attributes for input/output tensor: ", name);
  zdl::DlSystem::UserBufferEncoding* user_buffer_encoding = nullptr;
  size_t buffer_element_size = 0;
  if (BufferType::TF8 == buffer_type_ || BufferType::TF16 == buffer_type_) {
    int bit_width = buffer_type_ == BufferType::TF16 ? 16 : 8;
    buffer_element_size = bit_width / 8;
    auto encoding = (*buffer_attributes)->getEncoding();
    ORT_ENFORCE(encoding, "Failed to get buffer encoding for: ", name);
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

  return Status::OK();
}

Status SnpeLib::SetupUserBufferAttributes(const std::vector<std::string>& tensor_names) {
  for (size_t i = 0; i < tensor_names.size(); ++i) {
    ORT_RETURN_IF_ERROR(SetupUserBufferAttribute(tensor_names[i]));
  }

  return Status::OK();
}

Status SnpeLib::CheckInputsSize(const std::vector<std::string>& input_tensor_names,
                              const std::vector<int64_t>& input_sizes) {
  size_t elementSize = 1;
  for (size_t i = 0; i < input_tensor_names.size(); ++i) {
    auto input_shape = snpe_->getInputDimensions(input_tensor_names[i].c_str());
    ORT_ENFORCE(input_shape, "Snpe cannot get input shape for input name: ", input_tensor_names[i]);
    int64_t input_size = CalcSizeFromDims((*input_shape).getDimensions(), (*input_shape).rank(), elementSize);
    ORT_RETURN_IF(input_sizes.at(i) != input_size,
                  "Input size mismatch for : ", input_tensor_names[i],
                  " size on dlc: ", input_size,
                  " size on onnx: ", input_sizes.at(i));
  }
  return Status::OK();
}

Status SnpeLib::SetupInputTensors(const std::vector<std::string>& input_tensor_names) {
  input_tensor_map_.clear();
  input_tensors_.clear();
  input_tensors_.resize(input_tensor_names.size());
  for (size_t i = 0; i < input_tensor_names.size(); ++i) {
    auto input_shape = snpe_->getInputDimensions(input_tensor_names[i].c_str());
    if (!input_shape) {
      input_tensor_map_.clear();
      input_tensors_.clear();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get input shape from Snpe for: ", input_tensor_names[i]);
    }
    input_tensors_[i] = zdl::SNPE::SNPEFactory::getTensorFactory().createTensor(*input_shape);
    zdl::DlSystem::ITensor* input_tensor = input_tensors_[i].get();
    if (!input_tensor) {
      input_tensor_map_.clear();
      input_tensors_.clear();
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Snpe cannot create ITensor for inputs!");
    }
    input_tensor_map_.add(input_tensor_names[i].c_str(), input_tensor);
  }
  return Status::OK();
}

Status SnpeLib::InitializeSnpe(zdl::DlContainer::IDlContainer* container,
                               const std::vector<std::string>& output_tensor_names,
                               const std::vector<std::string>& input_tensor_names,
                               const std::vector<int64_t>& input_sizes,
                               const SnpeRuntimeOptions& snpe_settings) {
  ORT_RETURN_IF_NOT(snpe_settings.GetRuntimeTarget().IsAvailable(),
                    "Provided runtime target is not available: ",
                    snpe_settings.GetRuntimeTarget().ToString());

  ORT_RETURN_IF(0 == output_tensor_names.size() || 0 == input_tensor_names.size(),
                "Input names or output names are empty!");

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
    ORT_RETURN_IF_NOT(platformConfig.setPlatformOptionValue("unsignedPD", "ON"),
                      "unsignedPD is not available! ", GetSnpeErrorString());
    snpe_builder.setPlatformConfig(platformConfig);
  }
#endif

  snpe_ = snpe_builder.build();
  ORT_ENFORCE(snpe_, "Failed to create snpe instance! ", GetSnpeErrorString());

  // Make sure shape of snpe inputs are same with onnx model inputs
  ORT_RETURN_IF_ERROR(CheckInputsSize(input_tensor_names, input_sizes));

  ORT_ENFORCE(BufferType::UNKNOWN != buffer_type_, "Buffer type unknown!");

  if (BufferType::ITENSOR == buffer_type_) {
    ORT_RETURN_IF_ERROR(SetupInputTensors(input_tensor_names));
  } else {
    ORT_RETURN_IF_ERROR(SetupUserBufferAttributes(input_tensor_names));
    ORT_RETURN_IF_ERROR(SetupUserBufferAttributes(output_tensor_names));
  }

  return Status::OK();
}

Status SnpeLib::Initialize(const char* dlcPath,
                           const std::vector<std::string>& output_layer_names,
                           const std::vector<std::string>& input_layer_names,
                           const std::vector<int64_t>& input_sizes,
                           const SnpeRuntimeOptions& snpe_settings) {
  auto container = zdl::DlContainer::IDlContainer::open(zdl::DlSystem::String(dlcPath));
  ORT_ENFORCE(container, "Failed open " , dlcPath, " container file");

  ORT_RETURN_IF_ERROR(InitializeSnpe(container.get(),
                                     output_layer_names,
                                     input_layer_names,
                                     input_sizes,
                                     snpe_settings));

  return Status::OK();
}

Status SnpeLib::Initialize(const unsigned char* dlcData, size_t size,
                           const std::vector<std::string>& output_layer_names,
                           const std::vector<std::string>& input_layer_names,
                           const std::vector<int64_t>& input_sizes,
                           const SnpeRuntimeOptions& snpe_settings) {
  auto container = zdl::DlContainer::IDlContainer::open(dlcData, size);
  ORT_ENFORCE(container, "failed open container buffer!");

  ORT_RETURN_IF_ERROR(InitializeSnpe(container.get(),
                                     output_layer_names,
                                     input_layer_names,
                                     input_sizes,
                                     snpe_settings));

  return Status::OK();
}

Status SnpeLib::SnpeProcessMultipleOutput(const unsigned char* input,
                                          size_t input_size,
                                          size_t output_number,
                                          unsigned char* outputs[],
                                          size_t output_sizes[],
                                          const std::unordered_map<std::string, size_t>& output_names_index) {
  try {
    zdl::DlSystem::Optional<zdl::DlSystem::TensorShape> input_shape = snpe_->getInputDimensions();
    ORT_ENFORCE(input_shape, "Snpe cannot get input shape");

    ORT_RETURN_IF(input_tensors_.size() < 1, "Failed to create Snpe ITensor at initialization time.");

    zdl::DlSystem::ITensor* input_tensor = input_tensors_[0].get();
    // ensure size of the input buffer matches input shape buffer size
    size_t input_data_size = input_tensor->getSize() * sizeof(float);
    ORT_RETURN_IF(input_data_size != input_size, "Snpe input size incorrect: expected bytes ",
                  input_data_size, " given bytes ", input_size);
    memcpy(input_tensor->begin().dataPointer(), input, input_size);

    zdl::DlSystem::TensorMap output_tensor_map;
    ORT_RETURN_IF_NOT(snpe_->execute(input_tensor, output_tensor_map),
                      "Snpe Error while executing the network: ",
                      GetSnpeErrorString());
    ORT_RETURN_IF(output_tensor_map.size() == 0, "Failed to get output tensor map, ", GetSnpeErrorString());

    zdl::DlSystem::StringList tensor_names = output_tensor_map.getTensorNames();

    for (size_t i = 0; i < output_number; i++) {
      zdl::DlSystem::ITensor* tensor = output_tensor_map.getTensor(tensor_names.at(i));

      auto pos = output_names_index.find(tensor_names.at(i));
      ORT_RETURN_IF(pos == output_names_index.end(), "Something wrong with output name: ", tensor_names.at(i));
      size_t output_name_index = pos->second;

      // ensure size of the output buffer matches output shape buffer size
      size_t output_data_size = tensor->getSize() * sizeof(float);
      ORT_RETURN_IF(output_data_size > output_sizes[output_name_index],
                    "Snpe output size incorrect: output_layer: ", tensor_names.at(i),
                    " expected bytes ", output_data_size,
                    " given bytes ", output_sizes[output_name_index]);
      memcpy(outputs[output_name_index], tensor->cbegin().dataPointer(), output_data_size);
    }

    return Status::OK();
  } catch (...) {
    LOGS_DEFAULT(ERROR) << "Snpe threw exception " << GetSnpeErrorString();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Snpe threw exception ", GetSnpeErrorString());
  }
}

Status SnpeLib::SnpeProcess(const unsigned char* input, size_t input_size, unsigned char* output, size_t output_size,
                          const std::unordered_map<std::string, size_t>& output_names_index) {
  // Use SnpeProcessMultipleOutput with 1 output layer
  const int output_layer = 1;
  unsigned char* outputs_array[output_layer];
  size_t output_sizes_array[output_layer];
  outputs_array[0] = output;
  output_sizes_array[0] = output_size;
  ORT_RETURN_IF_ERROR(SnpeProcessMultipleOutput(input, input_size, output_layer, outputs_array,
                                                output_sizes_array, output_names_index));
  return Status::OK();
}

Status SnpeLib::SnpeProcessMultiInputsMultiOutputs(const unsigned char** inputs,
                                                 const size_t* input_sizes,
                                                 size_t input_number,
                                                 unsigned char** outputs,
                                                 const size_t* output_sizes,
                                                 size_t output_number,
                                                 const std::unordered_map<std::string, size_t>& output_names_index) {
  try {
    ORT_RETURN_IF(input_number != input_tensors_.size(),
                  "Snpe number of inputs doesn't match: expected ", input_number,
                  " given ", input_tensors_.size());
    for (size_t i = 0; i < input_number; ++i) {
      zdl::DlSystem::ITensor* input_tensor = input_tensors_[i].get();
      // ensure size of the input buffer matches input shape buffer size
      size_t input_data_size = input_tensor->getSize() * sizeof(float);
      ORT_RETURN_IF(input_data_size != input_sizes[i],
                    "Snpe input size incorrect: expected bytes ", input_data_size,
                    ", given bytes ", input_sizes[i]);
      memcpy(input_tensor->begin().dataPointer(), inputs[i], input_sizes[i]);
    }
    zdl::DlSystem::TensorMap output_tensor_map;
    ORT_RETURN_IF_NOT(snpe_->execute(input_tensor_map_, output_tensor_map),
                      "Snpe Error while executing the network: ",
                      GetSnpeErrorString());
    ORT_RETURN_IF(output_tensor_map.size() == 0, "Failed to get output tensor map, ", GetSnpeErrorString());

    zdl::DlSystem::StringList tensor_names = output_tensor_map.getTensorNames();

    for (size_t i = 0; i < output_number; i++) {
      zdl::DlSystem::ITensor* tensor = output_tensor_map.getTensor(tensor_names.at(i));

      auto pos = output_names_index.find(tensor_names.at(i));
      ORT_RETURN_IF(pos == output_names_index.end(), "Something wrong with output name: ", tensor_names.at(i));
      size_t output_name_index = pos->second;

      // ensure size of the output buffer matches output shape buffer size
      size_t output_data_size = tensor->getSize() * sizeof(float);
      ORT_RETURN_IF(output_data_size > output_sizes[output_name_index],
                    "Snpe output size incorrect: output_layer: ", tensor_names.at(i),
                    " expected bytes ", output_data_size,
                    " given bytes ", output_sizes[output_name_index]);
      memcpy(outputs[output_name_index], tensor->cbegin().dataPointer(), output_data_size);
    }

    return Status::OK();
  } catch (...) {
    LOGS_DEFAULT(ERROR) << "Snpe threw exception " << GetSnpeErrorString();
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Snpe threw exception ", GetSnpeErrorString());
  }
}

Status SnpeLib::SnpeProcessWithUserBuffer(const std::vector<std::string>& input_names,
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
  ORT_RETURN_IF_NOT(snpe_->execute(input_buffer_map, output_buffer_map),
                    "Snpe Error while executing the network: ",
                    GetSnpeErrorString());

  return Status::OK();
}

std::unique_ptr<SnpeLib> SnpeLibFactory(const unsigned char* dlc_data,
                                        size_t size,
                                        const std::unordered_map<std::string, std::string>& options,
                                        const std::vector<std::string>& output_layer_names,
                                        const std::vector<std::string>& input_layer_names,
                                        const std::vector<int64_t>& input_sizes,
                                        BufferType& buffer_type) {
  auto object = std::make_unique<SnpeLib>();

  if (!object) {
    ORT_THROW("failed to make snpe library");
  }

  SnpeRuntimeOptions runtime_options(options);
  ORT_THROW_IF_ERROR(object->Initialize(dlc_data,
                                        size,
                                        output_layer_names,
                                        input_layer_names,
                                        input_sizes,
                                        runtime_options));

  buffer_type = runtime_options.GetBufferType();

  return object;
}

}  // namespace snpe
}  // namespace contrib
}  // namespace onnxruntime
